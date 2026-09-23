#!/usr/bin/env python3
"""Convert the project's fixed conversation JSONL schema to token manifest v1."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


ROLES = {"system", "user", "assistant", "tool"}
TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_messages(value, line_number: int) -> list[dict]:
    """Validate and return one conversation in the documented fixed schema."""
    prefix = f"line {line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: top-level JSON value must be an object")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{prefix}: 'messages' must be a non-empty array")

    normalized = []
    previous = None
    for index, message in enumerate(messages):
        where = f"{prefix}, messages[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{where}: message must be an object")
        role = message.get("role")
        if role not in ROLES:
            raise ValueError(f"{where}: role must be one of {sorted(ROLES)}")
        allowed = {"role", "content", "reasoning_content"} if role == "assistant" else {"role", "content"}
        unknown = set(message) - allowed
        if unknown:
            raise ValueError(f"{where}: unsupported fields: {sorted(unknown)}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{where}: content must be a non-empty string")
        if role == "assistant" and "reasoning_content" in message:
            reasoning = message["reasoning_content"]
            if not isinstance(reasoning, str):
                raise ValueError(f"{where}: reasoning_content must be a string")

        if role == "system":
            if index != 0:
                raise ValueError(f"{where}: system is only allowed as the first message")
        elif previous in (None, "system") and role != "user":
            raise ValueError(f"{where}: the conversation must start with a user message")
        elif previous == "user" and role != "assistant":
            raise ValueError(f"{where}: user must be followed by assistant")
        elif previous == "assistant" and role not in {"user", "tool"}:
            raise ValueError(f"{where}: assistant must be followed by user or tool")
        elif previous == "tool" and role not in {"tool", "assistant"}:
            raise ValueError(f"{where}: tool must be followed by tool or assistant")

        normalized.append(message)
        previous = role

    roles = {message["role"] for message in normalized}
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{prefix}: conversation must contain user and assistant messages")
    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{prefix}: a complete training conversation must end with assistant")
    return normalized


def tokenizer_hashes(tokenizer_path: Path) -> dict[str, str]:
    if not tokenizer_path.is_dir():
        return {}
    result = {}
    for path in sorted(tokenizer_path.rglob("*")):
        if path.is_file() and path.name in TOKENIZER_FILES:
            result[path.relative_to(tokenizer_path).as_posix()] = sha256(path)
    return result


def scan_input(path: Path) -> tuple[str, int, tuple[int, int]]:
    """Hash and count records once so tqdm can display an exact percentage."""
    digest = hashlib.sha256()
    newlines = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            newlines += block.count(b"\n")
            last_byte = block[-1:]
    stat = path.stat()
    lines = newlines + int(stat.st_size > 0 and last_byte != b"\n")
    return digest.hexdigest(), lines, (stat.st_size, stat.st_mtime_ns)


def tokenize_line(line_number: int, line: str, tokenizer, vocab_size: int) -> np.ndarray:
    if not line.strip():
        raise ValueError(f"line {line_number}: blank lines are not allowed")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
    messages = validate_messages(value, line_number)
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    except Exception as error:
        raise ValueError(f"line {line_number}: chat template/tokenization failed: {error}") from error
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"line {line_number}: tokenizer returned an empty or invalid token list")
    if any(isinstance(token, bool) or not isinstance(token, (int, np.integer)) for token in token_ids):
        raise ValueError(f"line {line_number}: tokenizer returned a non-integer token ID")
    minimum, maximum = int(min(token_ids)), int(max(token_ids))
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(f"line {line_number}: token ID is outside tokenizer vocabulary [0, {vocab_size})")
    return np.asarray(token_ids, dtype="<u4")


def ordered_tokenize(source, tokenizer, vocab_size: int, workers: int):
    """Tokenize concurrently while yielding arrays in input order with bounded memory."""
    lines = enumerate(source, 1)
    if workers == 1:
        for line_number, line in lines:
            yield tokenize_line(line_number, line, tokenizer, vocab_size)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tokenize")
    pending = {}

    def submit_one():
        try:
            line_number, line = next(lines)
        except StopIteration:
            return False
        pending[line_number] = executor.submit(tokenize_line, line_number, line, tokenizer, vocab_size)
        return True

    try:
        for _ in range(workers * 2):
            if not submit_one():
                break
        next_line = 1
        while pending:
            future = pending.pop(next_line)
            yield future.result()
            submit_one()
            next_line += 1
    finally:
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def convert(input_path, tokenizer_path, output_path, *, weight=1.0, name=None,
            progress_interval=1, workers=None, tokenizer=None):
    input_path = Path(input_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_path = Path(output_path).resolve()
    building_path = output_path.with_name(output_path.name + ".building")

    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"local tokenizer directory does not exist: {tokenizer_path}")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if building_path.exists():
        raise FileExistsError(f"temporary output already exists: {building_path}")
    weight = float(weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be positive and finite")
    workers = int(workers if workers is not None else (os.cpu_count() or 1))
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if progress_interval < 0:
        raise ValueError("progress_interval must be >= 0")

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = int(len(tokenizer))
    if vocab_size <= 0 or vocab_size > np.iinfo(np.uint32).max:
        raise ValueError(f"tokenizer size cannot be represented as uint32: {vocab_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_path.mkdir()
    tokens_path = building_path / "corpus.bin"
    offsets_path = building_path / "corpus.offsets.npy"
    offsets = [0]
    documents = 0
    input_sha256, total_documents, input_stat = scan_input(input_path)

    try:
        from tqdm.auto import tqdm

        with input_path.open("r", encoding="utf-8") as source, tokens_path.open("wb") as tokens_file:
            progress = tqdm(total=total_documents, desc=f"tokenize ({workers} threads)", unit="doc",
                            miniters=max(1, progress_interval), disable=progress_interval == 0,
                            dynamic_ncols=True)
            try:
                for array in ordered_tokenize(source, tokenizer, vocab_size, workers):
                    array.tofile(tokens_file)
                    offsets.append(offsets[-1] + len(array))
                    documents += 1
                    progress.set_postfix(tokens=f"{offsets[-1]:,}", refresh=False)
                    progress.update(1)
            finally:
                progress.close()

        if documents == 0:
            raise ValueError("input JSONL contains no documents")
        stat = input_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != input_stat:
            raise RuntimeError("input JSONL changed during conversion")
        np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        manifest = {
            "format": 1,
            "vocab_size": vocab_size,
            "tokenizer": str(tokenizer_path),
            "tokenizer_files": tokenizer_hashes(tokenizer_path),
            "sources": [{
                "name": name or input_path.stem,
                "tokens": tokens_path.name,
                "offsets": offsets_path.name,
                "weight": weight,
                "documents": documents,
                "token_count": offsets[-1],
                "input_sha256": input_sha256,
                "tokens_sha256": sha256(tokens_path),
                "offsets_sha256": sha256(offsets_path),
            }],
        }
        (building_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(building_path, output_path)
    except BaseException:
        shutil.rmtree(building_path, ignore_errors=True)
        raise

    print(f"wrote {documents:,} documents / {offsets[-1]:,} tokens to {output_path}")
    return output_path / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert fixed-schema conversation JSONL to RWKV-Theseus manifest v1."
    )
    parser.add_argument("--input", required=True, help="UTF-8 JSONL input file")
    parser.add_argument("--tokenizer", required=True, help="local HF model/tokenizer directory")
    parser.add_argument("--output", required=True, help="new output directory (must not exist)")
    parser.add_argument("--name", help="source name stored as manifest metadata")
    parser.add_argument("--weight", type=float, default=1.0, help="source sampling weight (default: 1.0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="tokenization threads (default: all logical CPUs)")
    parser.add_argument("--progress-interval", type=int, default=1,
                        help="minimum documents per tqdm update; 0 disables tqdm")
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("--progress-interval must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main():
    args = parse_args()
    convert(args.input, args.tokenizer, args.output, weight=args.weight, name=args.name,
            progress_interval=args.progress_interval, workers=args.workers)


if __name__ == "__main__":
    main()
