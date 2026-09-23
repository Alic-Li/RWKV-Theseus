#!/usr/bin/env python3
"""Replace selected full-attention weights in a converted Lightning PTH.

Reads Theseus `migrated.safetensors` files from a directory, in natural path
order. Later files override earlier snapshots of the same tensor. The output
uses the same hybrid format, block names and out/in matrix layout as the
GDN converter. The backend detects each mixer from its tensor signature.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from theseus.timemix import TimeMix


BASE_FORMAT = "rwkv_lightning_qwen_hybrid_v1"
OUTPUT_FORMAT = BASE_FORMAT
LOW_RANK = {"w1", "w2", "a1", "a2", "g1", "g2"}


def export_tensor(name, value):
    """All matrices use [out,in]; broadcast vectors use [C]."""
    if name in LOW_RANK:
        return value.T.contiguous()
    if value.ndim == 3 and value.shape[:2] == (1, 1):
        return value.reshape(-1).contiguous()
    return value.contiguous()
LAYER_TYPE = "rwkv7_timemix"
DTYPES = {"BF16", "F16", "F32"}
CHUNK_BYTES = 32 << 20


def sidecar(path: Path, kind: str) -> Path:
    return path.with_suffix(f".{kind}.json")


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def parse_layers(value: str) -> list[int]:
    try:
        layers = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--layers must be comma-separated integer layer indices") from exc
    if not layers or any(i < 0 for i in layers) or len(set(layers)) != len(layers):
        raise ValueError("--layers needs distinct nonnegative layer indices")
    return layers


def expected_shapes(width: int, layer: int, depth: int, head_size: int) -> dict[str, tuple[int, ...]]:
    # Meta construction uses the actual Theseus module schema without allocating
    # its large dense matrices or initializing real weights.
    with torch.device("meta"):
        module = TimeMix(width, layer, depth, head_size=head_size, backend="reference")
    return {name: tuple(value.shape) for name, value in module.state_dict().items()}


def inspect_sources(folder: Path, layers: list[int], width: int, depth: int,
                    head_size: int | None = None):
    """Infer each layer's trained geometry from r_k; never reinterpret its heads."""
    files = ([folder] if folder.is_file() else
             sorted(folder.rglob("*.safetensors"), key=lambda p: natural_key(p.relative_to(folder))))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {folder}")
    chosen: dict[tuple[int, str], Path] = {}
    headers = {}
    superseded = 0
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                match = re.fullmatch(r"(\d+)\.(.+)", key)
                if match is None:
                    raise ValueError(f"Unexpected Theseus tensor key in {path}: {key}")
                layer, name = int(match[1]), match[2]
                if layer not in layers:
                    continue
                part = reader.get_slice(key)
                if part.get_dtype() not in DTYPES:
                    raise ValueError(f"Invalid tensor dtype {path}:{key}: {part.get_dtype()}")
                superseded += (layer, name) in chosen
                chosen[layer, name] = path
                headers[layer, name] = tuple(part.get_shape())
    head_sizes, shapes = {}, {}
    for layer in layers:
        rk = headers.get((layer, "r_k"))
        if rk is None:
            raise ValueError(f"Layer {layer} is incomplete; missing: r_k (needed to infer head size)")
        if len(rk) != 2 or rk[0] * rk[1] != width or rk[1] not in (64, 128):
            raise ValueError(f"Invalid layer {layer} r_k shape {rk}; shared WKV supports D64/D128")
        if head_size is not None and rk[1] != head_size:
            raise ValueError(f"Layer {layer} was trained with head_size={rk[1]}, "
                             f"but --head-size={head_size}; this option validates, it does not reshape heads")
        head_sizes[layer] = rk[1]
        shapes[layer] = expected_shapes(width, layer, depth, rk[1])
        names = {name for i, name in chosen if i == layer}
        missing, unknown = shapes[layer].keys() - names, names - shapes[layer].keys()
        if missing:
            raise ValueError(f"Layer {layer} is incomplete; missing: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"Unknown TimeMix parameters in layer {layer}: {', '.join(sorted(unknown))}")
        for name, shape in shapes[layer].items():
            if headers[layer, name] != shape:
                raise ValueError(f"Invalid tensor {chosen[layer, name]}:{layer}.{name}: "
                                 f"{headers[layer, name]} != {shape}")
    return files, chosen, superseded, shapes, head_sizes


class ProgressWriter:
    def __init__(self, file, bar):
        self.file, self.bar = file, bar

    def write(self, data):
        view = memoryview(data).cast("B")
        for start in range(0, len(view), CHUNK_BYTES):
            piece = view[start:start + CHUNK_BYTES]
            written = self.file.write(piece)
            if written != len(piece):
                raise OSError("Short checkpoint write")
            if self.bar.total is not None and self.bar.n + written > self.bar.total:
                self.bar.total = self.bar.n + written
            self.bar.update(written)
        return len(view)

    def flush(self):
        return self.file.flush()

    def tell(self):
        return self.file.tell()


def verify_pth(path: Path, expected: dict[str, torch.Tensor], quiet: bool):
    result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(result) != set(expected):
        raise ValueError("Saved checkpoint tensor keys differ")
    total = sum(t.numel() * t.element_size() for t in expected.values())
    with tqdm(total=total, desc="Verify bytes", unit="B", unit_scale=True,
              unit_divisor=1024, disable=quiet) as bar:
        for name, source in expected.items():
            target = result[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(f"Saved tensor metadata differs: {name}")
            a = source.contiguous().view(torch.uint8).reshape(-1)
            b = target.contiguous().view(torch.uint8).reshape(-1)
            for start in range(0, a.numel(), CHUNK_BYTES):
                stop = min(start + CHUNK_BYTES, a.numel())
                if not torch.equal(a[start:stop], b[start:stop]):
                    raise ValueError(f"Saved tensor bytes differ: {name}, byte {start}")
                bar.update(stop - start)


def atomic_json(path: Path, value):
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def replace(base: Path, weights_dir: Path, out: Path, layers: list[int], *, head_size=None,
            dtype="base", dry_run=False, verify=False, overwrite=False, quiet=False):
    base, weights_dir, out = (p.expanduser().resolve() for p in (base, weights_dir, out))
    if not base.is_file() or not weights_dir.exists():
        raise FileNotFoundError("--base must be a PTH file and --weights-dir must be a directory or safetensors file")
    if base == out or out.suffix != ".pth":
        raise ValueError("--out must be a new .pth path distinct from --base")
    config_path, manifest_path = sidecar(base, "config"), sidecar(base, "manifest")
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The base PTH needs its .config.json and .manifest.json sidecars")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("format") != BASE_FORMAT or manifest.get("format") != config["format"]:
        raise ValueError("Unsupported or inconsistent converted PTH metadata")
    if config.get("weights_file") != base.name:
        raise ValueError("Base config weights_file does not match --base")
    if not layers or len(set(layers)) != len(layers) or any(i < 0 for i in layers):
        raise ValueError("layers must be distinct nonnegative indices")
    geometry = config["geometry"]
    kinds = geometry["layer_types"]
    width, depth = int(geometry["hidden_size"]), int(geometry["num_hidden_layers"])
    if len(kinds) != depth or width <= 0 or head_size not in (None, 64, 128):
        raise ValueError("Invalid geometry or head_size")
    for layer in layers:
        if layer >= depth or kinds[layer] not in ("full_attention", "linear_attention", LAYER_TYPE):
            raise ValueError(f"Layer {layer} is not a replaceable mixer layer")
    files, chosen, superseded, shapes, head_sizes = inspect_sources(
        weights_dir, layers, width, depth, head_size)
    print(json.dumps({"base": str(base), "output": str(out), "layers": layers,
                      "head_sizes": head_sizes, "wkv_kernel": "shared_dplr_fp32_v1",
                      "safetensors_files": len(files), "selected_tensors": len(chosen),
                      "superseded_tensor_versions": superseded}, ensure_ascii=False), flush=True)
    model = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(model, dict) or "emb.weight" not in model:
        raise ValueError("Base PTH is not a converted model state_dict")
    base_dtype = model["emb.weight"].dtype
    if base_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported base dtype: {base_dtype}")
    old_keys = []
    for layer in layers:
        prefix = f"blocks.{layer}.att."
        found = [key for key in model if key.startswith(prefix)]
        if not found:
            raise ValueError(f"Base PTH has no attention parameters for layer {layer}")
        old_keys.extend(found)
    if dry_run:
        return
    outputs = (out, sidecar(out, "config"), sidecar(out, "manifest"))
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Output exists; choose a new --out or pass --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < base.stat().st_size + (512 << 20):
        raise OSError("Insufficient free disk space for a second PTH")

    selected_by_file = defaultdict(list)
    for (layer, name), path in chosen.items():
        selected_by_file[path].append((layer, name))
    replacement = {}
    entries = []
    for path in files:
        if path not in selected_by_file:
            continue
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for layer, name in sorted(selected_by_file[path]):
                source = f"{layer}.{name}"
                value = reader.get_tensor(source)
                if tuple(value.shape) != shapes[layer][name]:
                    raise ValueError(f"Tensor changed during conversion: {path}:{source}")
                target = f"blocks.{layer}.att.{name}"
                value = value.to(base_dtype) if dtype == "base" else value
                value = export_tensor(name, value)
                replacement[target] = value
                entries.append({"source": source, "file": str(path.relative_to(weights_dir if weights_dir.is_dir() else weights_dir.parent)),
                                "target": target, "shape": list(value.shape),
                                "dtype": str(value.dtype).removeprefix("torch."),
                                "transform": ("transpose_out_in" if name in LOW_RANK else "squeeze_vector"
                                              if len(shapes[layer][name]) == 3 else "copy"),
                                "dtype_policy": dtype})
    for key in old_keys:
        del model[key]
    model.update(replacement)

    config["format"] = OUTPUT_FORMAT
    config["weights_file"] = out.name
    config["exported_utc"] = datetime.now(timezone.utc).isoformat()
    config["requires_hybrid_backend"] = True
    config.pop("requires_hybrid_timemix_dispatch", None)
    config.setdefault("timemix_layers", {})
    for layer in layers:
        kinds[layer] = LAYER_TYPE
        config["timemix_layers"][str(layer)] = {"head_size": head_sizes[layer], "num_heads": width // head_sizes[layer],
                                                 "source": "RWKV-Theseus TimeMix"}
    config.setdefault("summary", {})["output_tensors"] = len(model)
    config["summary"]["layer_counts"] = dict(Counter(kinds))
    config.setdefault("contract", {})["checkpoint_linear_layout"] = "out_in_row_major"
    config["contract"]["timemix_weight_layout"] = "out_in_row_major_vectors_flat_v1"
    config["contract"]["timemix"] = (
        "RWKV-Theseus TimeMix; independent shift and FP32 WKV state per layer; "
        "GDN and TimeMix use the same shared WKV kernel with per-layer D64/D128")
    config["contract"].update({
        "wkv_kernel": "shared_dplr_fp32_v1", "wkv_state_layout": "B,H,K,V",
        "wkv_decay_input": "effective_multiplier",
        "wkv_decay_operation": "identity; decay is computed by the mixer adapter",
        "wkv_inputs": "r,decay,k,v,a,b: FP32 [B,T,H,D]; output: FP16",
        "gdn_wkv_inputs": "r=q/sqrt(D), decay=exp(g), k=k, v=beta*v, a=k, b=-exp(g)*beta*k",
        "timemix_wkv_inputs": "decay=exp(-0.6065306597126334*sigmoid(raw_w)); remaining inputs from TimeMix",
    })
    config["wkv_layers"] = {}
    for i, kind in enumerate(kinds):
        if kind == LAYER_TYPE:
            rk = model[f"blocks.{i}.att.r_k"]
            heads, size = rk.shape
        elif kind == "linear_attention":
            heads = int(geometry["linear_num_value_heads"])
            size = int(geometry["linear_key_head_dim"])
            if size not in (64, 128) or int(geometry["linear_value_head_dim"]) != size:
                raise ValueError("Shared WKV requires square D64/D128 GDN states")
        else:
            continue
        config["wkv_layers"][str(i)] = {"kind": kind, "head_size": size, "num_heads": heads}
    config["summary"]["output"] = str(out)
    manifest["format"] = OUTPUT_FORMAT
    manifest["tensors"] = [entry for entry in manifest["tensors"]
                           if not any(entry["target"].startswith(f"blocks.{layer}.att.") for layer in layers)] + entries
    manifest["replaced_attention_layers"] = sorted(
        {int(key) for key in config["timemix_layers"]})

    fd, temp = tempfile.mkstemp(prefix=out.stem + ".", suffix=".partial", dir=out.parent)
    try:
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                    for value in model.values()}
        storage_bytes = sum(storages.values())
        with os.fdopen(fd, "wb") as file, tqdm(total=storage_bytes, desc="Write PTH", unit="B",
                unit_scale=True, unit_divisor=1024, disable=quiet) as bar:
            torch.save(model, ProgressWriter(file, bar), pickle_protocol=2,
                       _use_new_zipfile_serialization=True)
            file.flush()
            os.fsync(file.fileno())
        with zipfile.ZipFile(temp) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError("Saved PTH lacks data.pkl")
            if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
                raise ValueError("Saved PTH contains compressed storages")
        if verify:
            verify_pth(Path(temp), model, quiet)
        config["weights_bytes"] = os.path.getsize(temp)
        atomic_json(sidecar(out, "config"), config)
        atomic_json(sidecar(out, "manifest"), manifest)
        os.replace(temp, out)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f"Saved: {out}\nConfig: {sidecar(out, 'config')}\nManifest: {sidecar(out, 'manifest')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Converted GDN hybrid .pth")
    parser.add_argument("--weights-dir", "--weights", required=True, type=Path,
                        help="Theseus .safetensors file or directory recursively containing snapshots")
    parser.add_argument("--layers", required=True, help="Decoder layer indices, e.g. 3 or 3,7,11")
    parser.add_argument("--out", required=True, type=Path, help="New .pth output path")
    parser.add_argument("--head-size", type=int, choices=(64, 128), default=None,
                        help="Optional assertion; by default infer each layer from r_k. Never reshapes heads.")
    parser.add_argument("--dtype", choices=("base", "source"), default="base",
                        help="Cast TimeMix to base PTH dtype (default) or preserve safetensors dtype")
    parser.add_argument("--dry-run", action="store_true", help="Check headers and metadata without writing PTH")
    parser.add_argument("--verify", action="store_true", help="Reload and byte-check every output tensor")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    try:
        replace(args.base, args.weights_dir, args.out, parse_layers(args.layers),
                head_size=args.head_size, dtype=args.dtype, dry_run=args.dry_run,
                verify=args.verify, overwrite=args.overwrite, quiet=args.no_progress)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
