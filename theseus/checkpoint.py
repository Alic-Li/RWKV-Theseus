"""Atomic checkpoints on a shared filesystem. Only load your own trusted .pt files."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import transformers
from safetensors.torch import save_file, load_file
from .hf_model import AttentionAdapter
from .data import digest_file


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def model_fingerprint(path):
    """Config/index hashes + shard sizes/mtimes, not an expensive full-weight hash."""
    path = Path(path)
    records = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
               for p in sorted(path.glob("*.safetensors"))}
    for p in [path / "config.json", path / "model.safetensors.index.json"]:
        if p.exists():
            records[p.name] = digest_file(p)
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def delta_state(model):
    result = {}
    for i, block in enumerate(model.model.layers):
        if isinstance(getattr(block, "self_attn", None), AttentionAdapter):
            for name, value in block.self_attn.core.state_dict().items():
                result[f"{i}.{name}"] = value.detach().cpu().contiguous()
    return result


def split_delta(tensors):
    result = {}
    for key, value in tensors.items():
        layer, name = key.split(".", 1)
        result.setdefault(int(layer), {})[name] = value
    return result


def resolve_checkpoint(path):
    path = Path(path)
    if path.name == "latest":
        path = path.parent / path.read_text().strip()
    elif not (path / "COMMITTED").exists() and (path / "latest").exists():
        path = path / (path / "latest").read_text().strip()
    if not (path / "COMMITTED").exists():
        raise ValueError(f"Not a committed checkpoint: {path}")
    return path


def read_checkpoint(path):
    path = resolve_checkpoint(path)
    meta = json.loads((path / "manifest.json").read_text())
    if meta["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    return path, meta, split_delta(load_file(str(path / "migrated.safetensors")))


def save_checkpoint(topo, cfg, stage, step, complete, order, model, runner, reader, optimizer, scheduler, base_id):
    root = Path(cfg["output"])
    name = f"stage_{stage + 1:02d}_step_{step:08d}" + ("_complete" if complete else "")
    final, temp = root / name, root / ("." + name + ".tmp")
    if topo.rank == topo.leader:
        root.mkdir(parents=True, exist_ok=True)
        if temp.exists():
            shutil.rmtree(temp)
        (temp / "ranks").mkdir(parents=True)
    topo.barrier()
    torch.save(cpu_tree({"runner": runner.state_dict(), "reader": reader.state_dict() if reader else None,
                         "rng": rng_state()}), temp / "ranks" / f"rank_{topo.rank:05d}.pt")
    if topo.rank == topo.leader:
        save_file(delta_state(model), str(temp / "migrated.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, temp / "optimizer.pt")
        meta = {"format": 1, "training_topology": "local_branch_v1", "stage": stage, "step": step, "complete": complete,
                "order": order, "config": cfg, "world": topo.world,
                "base_fingerprint": base_id, "train_fingerprint": digest_file(cfg["train_manifest"]),
                "validation_fingerprint": digest_file(cfg["validation_manifest"]),
                "torch": torch.__version__, "transformers": transformers.__version__}
        (temp / "manifest.json").write_text(json.dumps(meta, indent=2))
    topo.barrier()
    if topo.rank == topo.leader:
        (temp / "COMMITTED").write_text("1\n")
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite committed checkpoint {final}")
        os.replace(temp, final)
        latest_temp = root / ".latest.tmp"
        latest_temp.write_text(name + "\n")
        os.replace(latest_temp, root / "latest")
    topo.barrier()
    return final


def validate_resume(meta, cfg, topo, base_id, order):
    if meta["base_fingerprint"] != base_id:
        raise ValueError("Base weights changed")
    if meta["order"] != order:
        raise ValueError("Migration order changed; start a new run. Deep-to-shallow checkpoints cannot resume shallow-to-deep training.")
    for name in ("train", "validation"):
        if meta[name + "_fingerprint"] != digest_file(cfg[name + "_manifest"]):
            raise ValueError(f"{name} manifest changed")
    mutable = {"output", "log_interval", "checkpoint_interval", "validation_interval", "timeout_minutes",
               "lr", "warmup_steps", "constant_steps", "min_lr"}
    for key in cfg.keys() - mutable - {k for k in cfg if k.startswith("wandb_")}:
        if cfg[key] != meta["config"].get(key, "sampled" if key == "training_mode" else None):
            raise ValueError(f"Resume config changed: {key}")
    if not meta["complete"]:
        if meta.get("training_topology") != "local_branch_v1":
            raise ValueError("Old paired mid-stage checkpoints cannot resume local-branch training; use a completed stage")
        if meta["world"] != topo.world:
            raise ValueError("Mid-stream resume needs identical rank topology")
