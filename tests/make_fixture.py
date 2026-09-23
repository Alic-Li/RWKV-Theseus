"""Create a genuine tiny HF Qwen3.5 checkpoint for torchrun integration tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import torch
from test_hf_adapter import tiny_model
from test_data_loss import write_dataset
from theseus.config import DEFAULTS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--stages", type=int, default=2)
    p.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tiny_model(args.stages * 2).save_pretrained(root / "base")
    train = write_dataset(root / "train")
    val = write_dataset(root / "val")
    cfg = DEFAULTS | {"training_mode": "sampled", "model": str(root / "base"), "train_manifest": str(train), "validation_manifest": str(val),
        "output": str(root / "run"), "expected_attention_layers": args.stages, "head_size": 8,
        "backend": "cuda" if args.backend == "nccl" else "reference", "distributed_backend": args.backend, "chunk_tokens": 3, "context_tokens": 9,
        "stage_steps": 3, "warmup_steps": 1, "grad_accum_steps": 2, "validation_chunks": 2,
        "validation_interval": 0, "checkpoint_interval": 0, "log_interval": 1, "original_teacher_kl": True,
        "debug_input": True}
    (root / "config.json").write_text(json.dumps(cfg, indent=2))
    print(root / "config.json")
