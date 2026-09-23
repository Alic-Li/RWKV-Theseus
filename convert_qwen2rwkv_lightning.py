#!/usr/bin/env python3
"""Export a LOCAL dense Qwen3.5-family hybrid checkpoint to Lightning-style PTH.

Qwen3.8-27B currently uses the qwen3_5 text architecture. Only storage layout and
names change here; GDN -> unclamped RWKV7 DPLR is a RUNTIME change. Full attention
is retained. This is NOT a canonical RWKV checkpoint for the unmodified backend.

Dependencies: torch, safetensors>=0.5, tqdm. No Transformers, CUDA or downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import zipfile

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

FORMAT = "rwkv_lightning_qwen_hybrid_v1"
DEFAULT_SRC = "./Qwen3.8-27B"
FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
CHUNK_BYTES = 32 << 20


@dataclass(frozen=True)
class Rule:
    source: str
    targets: tuple[str, ...]
    shape: tuple[int, ...]
    op: str = "copy"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def geometry(config: dict[str, Any]) -> dict[str, Any]:
    tc = config.get("text_config", config)
    if config.get("quantization_config") or tc.get("quantization_config"):
        raise ValueError("Use original BF16/FP16/FP32 weights, not FP8/INT4/quantized weights.")
    if tc.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        raise ValueError(f"Unsupported text architecture: {tc.get('model_type')!r}")
    names = ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
             "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
             "linear_value_head_dim", "linear_conv_kernel_dim", "num_attention_heads",
             "num_key_value_heads", "head_dim")
    g = {k: int(tc[k]) for k in names}
    if any(v <= 0 for v in g.values()):
        raise ValueError("All model dimensions must be positive.")
    types = tc.get("layer_types")
    if types is None:
        interval = int(tc["full_attention_interval"])
        if interval <= 0:
            raise ValueError("full_attention_interval must be positive.")
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                 for i in range(g["num_hidden_layers"])]
    if len(types) != g["num_hidden_layers"] or not set(types) <= {
        "linear_attention", "full_attention"
    }:
        raise ValueError("Invalid layer_types; only GDN + full attention is supported.")
    if tc.get("hidden_act", "silu") not in ("silu", "swish"):
        raise ValueError("This schema requires the source SiLU/SwiGLU activations.")
    if g["linear_num_value_heads"] % g["linear_num_key_heads"]:
        raise ValueError("GDN value-head count must be divisible by key-head count.")
    if g["num_attention_heads"] % g["num_key_value_heads"]:
        raise ValueError("Attention head count must be divisible by KV-head count.")
    # The exported geometry is explicit; the planned D128 kernel is square.
    if g["linear_key_head_dim"] != g["linear_value_head_dim"]:
        raise ValueError("This version targets square RWKV states: Dk must equal Dv.")
    if tc.get("attn_output_gate", True) is not True:
        raise ValueError("This version targets Qwen3.5-family gated full attention.")
    g["layer_types"] = list(types)
    g["key_dim"] = g["linear_num_key_heads"] * g["linear_key_head_dim"]
    g["value_dim"] = g["linear_num_value_heads"] * g["linear_value_head_dim"]
    g["attention_bias"] = bool(tc.get("attention_bias", False))
    g["tie_word_embeddings"] = bool(tc.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    return g


def checkpoint_index(src: Path) -> dict[str, str]:
    index = src / "model.safetensors.index.json"
    if index.is_file():
        weights = load_json(index)["weight_map"]
    else:
        paths = sorted(src.glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"No safetensors checkpoint in {src}")
        weights = {}
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name in weights:
                        raise ValueError(f"Duplicate input tensor: {name}")
                    weights[name] = path.name
    if not weights:
        raise ValueError("Empty checkpoint index.")
    for relative in set(weights.values()):
        # HF snapshot symlinks are legitimate, so do not reject resolved symlinks.
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe shard path in index: {relative}")
        if not (src / relative).is_file():
            raise FileNotFoundError(f"Missing shard: {src / relative}")
    return weights


def build_rules(keys: set[str], g: dict[str, Any]) -> tuple[list[Rule], str, bool]:
    prefixes = [p for p in ("model.language_model.", "language_model.", "model.")
                if p + "embed_tokens.weight" in keys]
    if len(prefixes) != 1:
        raise ValueError(f"Cannot uniquely identify the text backbone: {prefixes}")
    root = prefixes[0]
    c, f, vocab = g["hidden_size"], g["intermediate_size"], g["vocab_size"]
    kd, vd = g["key_dim"], g["value_dim"]
    hv, dv = g["linear_num_value_heads"], g["linear_value_head_dim"]
    qd = g["num_attention_heads"] * g["head_dim"]
    kvd = g["num_key_value_heads"] * g["head_dim"]
    rules: list[Rule] = []

    def add(source: str, target: str | tuple[str, ...], shape: tuple[int, ...], op: str = "copy"):
        if source not in keys:
            raise ValueError(f"Missing required source tensor: {source}")
        rules.append(Rule(source, (target,) if isinstance(target, str) else target, shape, op))

    add(root + "embed_tokens.weight", "emb.weight", (vocab, c))
    add(root + "norm.weight", "ln_out.weight", (c,))
    heads = [k for k in ("lm_head.weight", "model.lm_head.weight", "language_model.lm_head.weight") if k in keys]
    if len(heads) > 1:
        raise ValueError(f"Ambiguous LM head: {heads}")
    tied_alias = not heads
    if heads:
        add(heads[0], "head.weight", (vocab, c))
    elif not g["tie_word_embeddings"]:
        raise ValueError("Untied model is missing lm_head.weight.")

    for i, kind in enumerate(g["layer_types"]):
        old, new = f"{root}layers.{i}.", f"blocks.{i}."
        add(old + "input_layernorm.weight", new + "ln1.weight", (c,))
        add(old + "post_attention_layernorm.weight", new + "ln2.weight", (c,))
        for source, target, shape in (
            ("up_proj", "key", (f, c)),
            ("down_proj", "value", (c, f)),
            ("gate_proj", "gate", (f, c)),
        ):
            add(old + f"mlp.{source}.weight", new + f"ffn.{target}.weight", shape)
        a = new + "att."
        if kind == "linear_attention":
            p = old + "linear_attn."
            add(p + "in_proj_qkv.weight", tuple(a + s + ".weight" for s in
                ("receptance", "key", "value")), (2 * kd + vd, c), "split_qkv")
            for source, target, shape in (
                ("in_proj_z.weight", "gate.weight", (vd, c)),
                ("in_proj_a.weight", "decay.weight", (hv, c)),
                ("in_proj_b.weight", "beta.weight", (hv, c)),
                ("A_log", "A_log", (hv,)),
                ("dt_bias", "dt_bias", (hv,)),
                ("norm.weight", "ln_x.weight", (dv,)),
                ("out_proj.weight", "output.weight", (c, vd)),
            ):
                add(p + source, a + target, shape)
            add(p + "conv1d.weight", a + "conv1d.weight",
                (2 * kd + vd, 1, g["linear_conv_kernel_dim"]), "squeeze_conv")
            if p + "conv1d.bias" in keys:
                add(p + "conv1d.bias", a + "conv1d.bias", (2 * kd + vd,))
        else:
            p = old + "self_attn."
            add(p + "q_proj.weight", (a + "receptance.weight", a + "gate.weight"),
                (2 * qd, c), "split_q_gate")
            for source, target, shape in (
                ("k_proj.weight", "key.weight", (kvd, c)),
                ("v_proj.weight", "value.weight", (kvd, c)),
                ("o_proj.weight", "output.weight", (c, qd)),
                ("q_norm.weight", "q_norm.weight", (g["head_dim"],)),
                ("k_norm.weight", "k_norm.weight", (g["head_dim"],)),
            ):
                add(p + source, a + target, shape)
            if g["attention_bias"]:
                add(p + "q_proj.bias", (a + "receptance.bias", a + "gate.bias"),
                    (2 * qd,), "split_q_gate")
                for s, t, n in (("k_proj", "key", kvd), ("v_proj", "value", kvd),
                                ("o_proj", "output", c)):
                    add(p + s + ".bias", a + t + ".bias", (n,))
    consumed = {r.source for r in rules}
    unknown = sorted(k for k in keys if k.startswith(root) and k not in consumed)
    if unknown:
        raise ValueError("Unmapped TEXT tensors (not silently dropped):\n" + "\n".join(unknown[:30]))
    destinations = [n for r in rules for n in r.targets]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Destination tensor-name collision.")
    return rules, root, tied_alias


def transform(rule: Rule, x: torch.Tensor, g: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    if tuple(x.shape) != rule.shape:
        raise ValueError(f"Shape mismatch: {rule.source}: {tuple(x.shape)} != {rule.shape}")
    if rule.op == "split_qkv":
        ys = x.split((g["key_dim"], g["key_dim"], g["value_dim"]), dim=0)
    elif rule.op == "split_q_gate":
        # HF packs EACH head as [Q_head, gate_head], not [all Q, all gate].
        tail = tuple(x.shape[1:])
        packed = x.reshape(g["num_attention_heads"], 2, g["head_dim"], *tail)
        ys = tuple(packed[:, j].contiguous().reshape(-1, *tail) for j in (0, 1))
    elif rule.op == "squeeze_conv":
        ys = (x.squeeze(1),)
    else:
        ys = (x,)
    # Do NOT transpose linears, add one to norms, cast dtype, or duplicate GVA heads.
    return tuple(y.contiguous() for y in ys)


class ProgressWriter:
    """A normal torch.save file object, reporting real writes in bounded chunks."""
    def __init__(self, file, bar):
        self.file, self.bar = file, bar

    def write(self, data):
        view = memoryview(data).cast("B")
        for start in range(0, len(view), CHUNK_BYTES):
            piece = view[start:start + CHUNK_BYTES]
            written = self.file.write(piece)
            if written != len(piece):
                raise OSError("Short checkpoint write.")
            if self.bar.total is not None and self.bar.n + written > self.bar.total:
                self.bar.total = self.bar.n + written  # ZIP/pickle overhead is additional.
            self.bar.update(written)
        return len(view)

    def flush(self):
        return self.file.flush()

    def tell(self):
        return self.file.tell()


def verify_pth(path: Path, expected: dict[str, torch.Tensor], disable: bool) -> None:
    loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(loaded) != set(expected):
        raise ValueError("Saved checkpoint keys do not match the export.")
    total = sum(x.numel() * x.element_size() for x in expected.values())
    with tqdm(total=total, desc="Verify bytes", unit="B", unit_scale=True,
              unit_divisor=1024, disable=disable) as bar:
        for name, source in expected.items():
            result = loaded[name]
            if result.dtype != source.dtype or result.shape != source.shape or not result.is_contiguous():
                raise ValueError(f"Reloaded tensor metadata differs: {name}")
            # Bitwise comparison also preserves signed zero and NaN payloads.
            a = source.view(torch.uint8).reshape(-1)
            b = result.view(torch.uint8).reshape(-1)
            for start in range(0, a.numel(), CHUNK_BYTES):
                stop = min(start + CHUNK_BYTES, a.numel())
                if not torch.equal(a[start:stop], b[start:stop]):
                    raise ValueError(f"Reloaded tensor bytes differ: {name}, byte {start}")
                bar.update(stop - start)
    del loaded


def atomic_json(path: Path, value: Any) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def convert(src: Path, dst: Path, *, verify=False, dry_run=False, overwrite=False,
            no_progress=False) -> dict[str, Any]:
    src, dst = src.expanduser().resolve(), dst.expanduser().resolve()
    source_config_path = src / "config.json"
    config = load_json(source_config_path)
    g = geometry(config)
    index = checkpoint_index(src)
    rules, root, tied_alias = build_rules(set(index), g)
    out_config = dst.with_suffix(".config.json")
    out_manifest = dst.with_suffix(".manifest.json")
    if dst.suffix != ".pth":
        raise ValueError("Output filename must end with .pth")
    if not dry_run and not overwrite:
        for path in (dst, out_config, out_manifest):
            if path.exists():
                raise FileExistsError(f"Already exists: {path}; use --overwrite explicitly.")
    groups: dict[str, list[Rule]] = defaultdict(list)
    for rule in rules:
        groups[index[rule.source]].append(rule)
    input_bytes, dtype_counts = 0, Counter()
    with tqdm(total=len(rules), desc="Check headers", unit="tensor", disable=no_progress) as bar:
        for shard, group in sorted(groups.items()):
            with safe_open(str(src / shard), framework="pt", device="cpu") as f:
                for rule in group:
                    info = f.get_slice(rule.source)
                    shape, dtype = tuple(info.get_shape()), info.get_dtype()
                    if shape != rule.shape or dtype not in FLOAT_BYTES:
                        raise ValueError(f"Unsupported tensor: {rule.source}: {shape}, {dtype}; expected {rule.shape}")
                    input_bytes += math.prod(shape) * FLOAT_BYTES[dtype]
                    dtype_counts[dtype] += 1
                    bar.update(1)
    summary = {"source": str(src), "output": str(dst), "source_tensors": len(rules),
               "output_tensors": sum(len(r.targets) for r in rules) + int(tied_alias),
               "source_tensor_bytes": input_bytes, "dtypes": dict(dtype_counts),
               "layer_counts": dict(Counter(g["layer_types"])),
               "dropped_non_text_tensors": len(index) - len(rules)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return summary
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dst.parent).free < input_bytes + (256 << 20):
        raise OSError("Insufficient free disk space for a new PTH (old output is not removed first).")
    output, entries = {}, []
    with tqdm(total=len(rules), desc="Map weights", unit="tensor", disable=no_progress) as bar:
        for shard, group in sorted(groups.items()):
            bar.set_postfix_str(shard, refresh=False)
            with safe_open(str(src / shard), framework="pt", device="cpu") as f:
                for rule in group:
                    x = f.get_tensor(rule.source)
                    for name, value in zip(rule.targets, transform(rule, x, g), strict=True):
                        output[name] = value
                        entries.append({"source": rule.source, "shard": shard, "target": name,
                                        "transform": rule.op, "shape": list(value.shape),
                                        "dtype": str(value.dtype).removeprefix("torch."),
                                        "source_shape": list(rule.shape)})
                    bar.update(1)
    if tied_alias:
        output["head.weight"] = output["emb.weight"]
        entries.append({"source": root + "embed_tokens.weight", "target": "head.weight",
                        "transform": "tied_embedding_alias", "shape": list(output["head.weight"].shape),
                        "dtype": str(output["head.weight"].dtype).removeprefix("torch.")})
    # Contiguous GDN Q/K/V slices can share storage. PTH records their offsets;
    # the existing pth_tensor reader supports storage_offset. Save each storage once.
    storages = {x.untyped_storage().data_ptr(): x.untyped_storage().nbytes() for x in output.values()}
    storage_bytes = sum(storages.values())
    fd, temp = tempfile.mkstemp(prefix=dst.stem + ".", suffix=".partial", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as f, tqdm(total=storage_bytes, desc="Write PTH", unit="B",
                 unit_scale=True, unit_divisor=1024, disable=no_progress) as bar:
            torch.save(output, ProgressWriter(f, bar), pickle_protocol=2,
                       _use_new_zipfile_serialization=True)
            f.flush()
            os.fsync(f.fileno())
            bar.total = bar.n
            bar.refresh()
        # Check archive structure without rereading all tensor payloads.
        with zipfile.ZipFile(temp) as archive:
            if not any(n.endswith("/data.pkl") for n in archive.namelist()):
                raise ValueError("Missing data.pkl in the saved PTH.")
            if any(z.compress_type != zipfile.ZIP_STORED for z in archive.infolist()):
                raise ValueError("C++ PTH reader requires uncompressed storage.")
        if verify:
            verify_pth(Path(temp), output, no_progress)
        tc = config.get("text_config", config)
        metadata = {
            "format": FORMAT, "requires_hybrid_backend": True,
            "weights_file": dst.name, "weights_bytes": os.path.getsize(temp),
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(src), "source_tensor_prefix": root,
            "source_config_sha256": hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
            "source_config": config, "geometry": g, "summary": summary,
            "contract": {
                "checkpoint_linear_layout": "out_in_row_major",
                "dtype_policy": "preserve_source_dtype_and_bits",
                "plain_state_dict": True, "pickle_protocol": 2,
                "norm_weights": "raw_Qwen: 1+weight for ln1/ln2/ln_out/q_norm/k_norm",
                "gdn_output_norm": "ordinary_weight_RMSNorm_then_SiLU_gate",
                "norm_epsilon": float(tc.get("rms_norm_eps", 1e-6)),
                "gdn_qk_normalization": "x * rsqrt(sum(x*x) + 1e-6)",
                "gdn_state_layout": "B,Hv,K,V", "recommended_reference_state_dtype": "float32",
                "wkv_decay_input": "g = -exp(A_log) * softplus(decay_projection + dt_bias)",
                "wkv_decay_operation": "exp(g), NO clamp/sigmoid/exp(-exp(w))",
                "wkv_inputs": "r=q/sqrt(Dk), w=g, k=k, v=beta*v, a=k, b=-exp(g)*beta*k",
                "gdn_head_mapping": "hk = hv // (Hv / Hk); no weight-head duplication",
                "conv_layout": "packed_Q_K_V_channels,kernel; oldest_to_newest_tap_order",
                "attention": "source full GQA + Q/K norms + RoPE + sigmoid output gate; KV cache retained",
                "attention_q_gate_split": "source.view(Hq,2,Dh,C)[:,0/1]; reversible row split",
                "ffn": "value(silu(gate(x))*key(x)); NO ReLU-squared or time shift",
                "embedding": "direct embedding lookup; NO RWKV ln0 preprocessing",
                "tokenizer_source_dir": str(src),
            },
        }
        manifest = {"format": FORMAT, "tensors": entries,
                    "dropped_non_text_keys": sorted(set(index) - {r.source for r in rules})}
        atomic_json(out_config, metadata)
        atomic_json(out_manifest, manifest)
        # Commit the checkpoint only after writing and optional verification succeed.
        os.replace(temp, dst)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f"Saved: {dst}\nConfig: {out_config}\nManifest: {out_manifest}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_SRC), help="LOCAL HF checkpoint directory")
    parser.add_argument("--out", type=Path, help="Default: SRC/converted/Qwen3.8-27B-RWKV7-Hybrid.pth")
    parser.add_argument("--verify", action="store_true", help="Reload PTH using mmap and bitwise-check every exported tensor")
    parser.add_argument("--dry-run", action="store_true", help="Validate names/shapes/dtypes from headers; do not write anything")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing export files")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm (e.g. for log files)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for tensor repacking/verification (default: 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    dst = args.out or args.src / "converted" / "Qwen3.8-27B-RWKV7-Hybrid.pth"
    try:
        convert(args.src, dst, verify=args.verify, dry_run=args.dry_run,
                overwrite=args.overwrite, no_progress=args.no_progress)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
