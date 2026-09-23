import json
from pathlib import Path

DEFAULTS = {
    "model": "../Qwen3.8-27B", "train_manifest": "data/train/manifest.json",
    "validation_manifest": "data/val/manifest.json", "output": "runs/theseus_shallow_to_deep",
    "expected_attention_layers": 16, "head_size": 64, "backend": "cuda", "seed": 42,
    "chunk_tokens": 256, "context_tokens": 4096, "stage_steps": 1000, "training_mode": "epoch",
    "lr": 1e-4, "betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": 0.01,
    "clip_norm": 1.0, "warmup_steps": 300, "constant_steps": 3000, "min_lr": 3e-6, "grad_accum_steps": 1,
    "micro_batch_size": 1,
    "cosine_weight": 0.0, "loss_epsilon": 1e-6, "checkpoint_interval": 3000,
    "validation_interval": 100, "validation_chunks": 8, "log_interval": 10,
    "original_teacher_kl": False, "kl_token_block": 16, "debug_input": False,
    "wandb_mode": "disabled", "wandb_project": "RWKV-Theseus",
    "wandb_entity": None, "wandb_name": None,
    "distributed_backend": "nccl", "timeout_minutes": 60,
}


def load_config(path):
    supplied = json.loads(Path(path).read_text())
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    cfg = DEFAULTS | supplied
    for name in ("chunk_tokens", "context_tokens", "grad_accum_steps", "micro_batch_size", "validation_chunks",
                 "expected_attention_layers", "head_size", "kl_token_block", "log_interval"):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["context_tokens"] < cfg["chunk_tokens"]:
        raise ValueError("context_tokens must be >= chunk_tokens")
    if cfg["training_mode"] not in ("epoch", "sampled"):
        raise ValueError("training_mode must be epoch or sampled")
    if cfg["micro_batch_size"] > 1 and cfg["training_mode"] != "epoch":
        raise ValueError("micro_batch_size > 1 currently requires training_mode=epoch")
    if cfg["training_mode"] == "sampled":
        steps = cfg["stage_steps"] if isinstance(cfg["stage_steps"], list) else [cfg["stage_steps"]]
        if len(steps) not in (1, cfg["expected_attention_layers"]) or min(steps) < 1:
            raise ValueError("stage_steps must be positive or a per-stage list")
    if cfg["lr"] <= 0 or cfg["warmup_steps"] < 0 or cfg["clip_norm"] <= 0:
        raise ValueError("Invalid optimization configuration")
    for name in ("warmup_steps", "constant_steps"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 <= cfg["min_lr"] <= cfg["lr"]:
        raise ValueError("min_lr must be between zero and lr")
    if cfg["backend"] not in ("cuda", "reference") or cfg["distributed_backend"] not in ("nccl", "gloo"):
        raise ValueError("Unsupported backend")
    for name in ("checkpoint_interval", "validation_interval"):
        if cfg[name] < 0:
            raise ValueError(f"{name} must be >= 0")
    if cfg["backend"] == "cuda" and not 2 <= cfg["head_size"] <= 128:
        raise ValueError("CUDA supports head_size in 2..128")
    if cfg["backend"] == "cuda" and cfg["distributed_backend"] == "gloo":
        raise ValueError("Gloo is CPU smoke-test mode; choose backend=reference")
    if cfg["loss_epsilon"] <= 0 or cfg["cosine_weight"] < 0:
        raise ValueError("Invalid loss configuration")
    if cfg["wandb_mode"] not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode must be disabled, offline or online")
    if not isinstance(cfg["wandb_project"], str) or not cfg["wandb_project"].strip():
        raise ValueError("wandb_project must be nonempty")
    for name in ("model", "train_manifest", "validation_manifest", "output"):
        cfg[name] = str(Path(cfg[name]).resolve())
    return cfg


def steps_for(cfg, stage):
    steps = cfg["stage_steps"]
    return steps[stage if len(steps) > 1 else 0] if isinstance(steps, list) else steps
