"""Memory-mapped token documents, deterministic mixtures, explicit consumed cursor."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

SAMPLER_VERSION = 2


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TokenStream:
    def __init__(self, manifest, pair_id, pairs, chunk_tokens, context_tokens, seed=42):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("format") != 1:
            raise ValueError("Unsupported token manifest")
        self.fingerprint = digest_file(self.path)
        self.pair_id, self.pairs = pair_id, pairs
        self.chunk_tokens, self.context_tokens = chunk_tokens, context_tokens
        if chunk_tokens < 1 or context_tokens < chunk_tokens:
            raise ValueError("Require context_tokens >= chunk_tokens >= 1")
        self.seed = seed
        self.sources = []
        weights = []
        if not self.manifest["sources"]:
            raise ValueError("Manifest must contain at least one source")
        for entry in self.manifest["sources"]:
            tokens = np.memmap(self.path.parent / entry["tokens"], dtype="<u4", mode="r")
            offsets = np.load(self.path.parent / entry["offsets"], mmap_mode="r")
            if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(np.diff(offsets) <= 0):
                raise ValueError("Invalid or empty document offsets")
            if not np.isfinite(entry["weight"]) or entry["weight"] <= 0:
                raise ValueError("Mixture weights must be positive")
            self.sources.append((tokens, offsets))
            weights.append(entry["weight"])
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.window = self.offset = self.chunks = 0
        self._current = None

    def _window(self):
        # Global window IDs are disjoint between pairs. Source and document selection
        # is deterministic weighted sampling with replacement, fixed across stages.
        global_id = self.window * self.pairs + self.pair_id
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, global_id]))
        source = int(rng.choice(len(self.sources), p=self.weights))
        tokens, offsets = self.sources[source]
        doc = int(rng.integers(len(offsets) - 1))
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        # Sample a contiguous window whose length is an exact number of chunks
        # whenever the document is long enough. A random start distributes the
        # omitted remainder across samples without padding or crossing documents.
        available = end - start
        if available >= self.chunk_tokens:
            length = min(available, self.context_tokens)
            length = (length // self.chunk_tokens) * self.chunk_tokens
        else:
            length = available
        start += int(rng.integers(available - length + 1))
        return tokens[start:start + length]

    def next(self):
        if self._current is None:
            self._current = self._window()
        end = min(len(self._current), self.offset + self.chunk_tokens)
        values = np.array(self._current[self.offset:end], dtype=np.int64)
        result = {"ids": torch.from_numpy(values)[None], "reset": self.offset == 0,
                  "stream": self.window * self.pairs + self.pair_id, "chunk": self.chunks,
                  "position": self.offset}
        self.offset = end
        self.chunks += 1
        if end == len(self._current):
            self.window += 1
            self.offset = 0
            self._current = None
        return result

    def state_dict(self):
        return {"window": self.window, "offset": self.offset, "chunks": self.chunks,
                "fingerprint": self.fingerprint, "pair_id": self.pair_id, "pairs": self.pairs,
                "seed": self.seed, "chunk_tokens": self.chunk_tokens, "context_tokens": self.context_tokens,
                "sampler_version": SAMPLER_VERSION}

    def load_state_dict(self, state):
        own = self.state_dict()
        for key in own.keys() - {"window", "offset", "chunks"}:
            if key not in state or own[key] != state[key]:
                raise ValueError(f"Reader resume mismatch: {key}")
        self.window, self.offset, self.chunks = state["window"], state["offset"], state["chunks"]
        self._current = None


class EpochTokenStream(TokenStream):
    """Visit every document token exactly once; shuffle disjoint context windows.

    No weighted replacement, truncation or padding. Windows stay within documents;
    ranks take disjoint entries of one deterministic shuffled window list.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parts = []
        for source, (_, offsets) in enumerate(self.sources):
            counts = (np.diff(offsets) + self.context_tokens - 1) // self.context_tokens
            docs = np.repeat(np.arange(len(counts)), counts)
            first = np.repeat(np.cumsum(counts) - counts, counts)
            starts = offsets[docs] + (np.arange(len(docs)) - first) * self.context_tokens
            ends = np.minimum(starts + self.context_tokens, offsets[docs + 1])
            parts.append(np.column_stack((np.full(len(docs), source), starts, ends)))
        windows = np.concatenate(parts).astype(np.int64, copy=False)
        np.random.default_rng(self.seed).shuffle(windows)
        self._all_windows = windows if getattr(self, "micro_batch_size", 1) > 1 else None
        costs = (windows[:, 2] - windows[:, 1] + self.chunk_tokens - 1) // self.chunk_tokens
        self.rank_chunks = [int(costs[r::self.pairs].sum()) for r in range(self.pairs)]
        self.total_tokens = sum(len(tokens) for tokens, _ in self.sources)
        self.windows = windows[self.pair_id::self.pairs].copy()
        self.consumed_tokens = 0

    def _window(self):
        if self.exhausted:
            raise StopIteration
        source, start, end = self.windows[self.window]
        return self.sources[source][0][start:end]

    @property
    def exhausted(self):
        return self.window >= len(self.windows)

    def next(self):
        if self.exhausted:
            return None
        batch = super().next()
        self.consumed_tokens += batch['ids'].numel()
        return batch

    def state_dict(self):
        return super().state_dict() | {"reader_mode": "epoch_v1", "consumed_tokens": self.consumed_tokens}

    def load_state_dict(self, state):
        if state.get("reader_mode") != "epoch_v1":
            raise ValueError("A sampled reader checkpoint cannot resume an exhaustive epoch")
        self.consumed_tokens = state["consumed_tokens"]
        super().load_state_dict(state)


class BatchedEpochTokenStream(EpochTokenStream):
    """Pack equal-length, disjoint windows into independent batch lanes.

    Every lane starts at position zero and ends at the same position, so the
    shared HF cache and TimeMix state can be reset between groups without
    padding, masking, or mixing context across documents.
    """

    def __init__(self, *args, micro_batch_size, **kwargs):
        if micro_batch_size < 2:
            raise ValueError("Batched reader requires micro_batch_size >= 2")
        self.micro_batch_size = micro_batch_size
        super().__init__(*args, **kwargs)
        self.groups = self._pack(self.windows)
        self.rank_chunks = []
        for rank in range(self.pairs):
            rank_windows = self._all_windows[rank::self.pairs]
            lengths, counts = np.unique(rank_windows[:, 2] - rank_windows[:, 1], return_counts=True)
            self.rank_chunks.append(int((((counts + micro_batch_size - 1) // micro_batch_size)
                                         * ((lengths + self.chunk_tokens - 1) // self.chunk_tokens)).sum()))
        del self.windows, self._all_windows
        self.group = self.offset = 0

    def _pack(self, windows):
        pending = {}
        groups = []
        for row in windows:
            length = int(row[2] - row[1])
            bucket = pending.setdefault(length, [])
            bucket.append(row)
            if len(bucket) == self.micro_batch_size:
                groups.append(np.asarray(bucket, dtype=np.int64))
                pending[length] = []
        groups.extend(np.asarray(bucket, dtype=np.int64) for bucket in pending.values() if bucket)
        return groups

    @property
    def exhausted(self):
        return self.group >= len(self.groups)

    def next(self):
        if self.exhausted:
            return None
        group = self.groups[self.group]
        end = min(int(group[0, 2] - group[0, 1]), self.offset + self.chunk_tokens)
        values = np.stack([np.asarray(self.sources[int(source)][0][int(start) + self.offset:int(start) + end],
                                      dtype=np.int64) for source, start, _ in group])
        result = {"ids": torch.from_numpy(values), "reset": self.offset == 0,
                  "position": self.offset, "group": self.group}
        self.consumed_tokens += values.size
        self.offset = end
        if end == group[0, 2] - group[0, 1]:
            self.group += 1
            self.offset = 0
        return result

    def state_dict(self):
        return {"reader_mode": "batched_epoch_v1", "fingerprint": self.fingerprint,
                "pair_id": self.pair_id, "pairs": self.pairs, "seed": self.seed,
                "chunk_tokens": self.chunk_tokens, "context_tokens": self.context_tokens,
                "sampler_version": SAMPLER_VERSION, "micro_batch_size": self.micro_batch_size,
                "group": self.group, "offset": self.offset, "consumed_tokens": self.consumed_tokens}

    def load_state_dict(self, state):
        own = self.state_dict()
        for key in own.keys() - {"group", "offset", "consumed_tokens"}:
            if own[key] != state.get(key):
                raise ValueError(f"Reader resume mismatch: {key}")
        self.group, self.offset = state["group"], state["offset"]
        self.consumed_tokens = state["consumed_tokens"]
