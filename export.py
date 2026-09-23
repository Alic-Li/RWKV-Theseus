#!/usr/bin/env python3
"""Portable delta export, accompanied by this project's loader (not stock AutoModel)."""
import argparse
import json
from pathlib import Path
import shutil
from theseus.checkpoint import read_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    checkpoint, meta, delta = read_checkpoint(args.checkpoint)
    if not meta["complete"] or meta["stage"] + 1 != len(meta["order"]):
        raise ValueError("Final export requires the final committed stage")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(checkpoint / "migrated.safetensors", out / "migrated.safetensors")
    (out / "theseus.json").write_text(json.dumps({"format": 1, "base_model": meta["config"]["model"],
        "base_fingerprint": meta["base_fingerprint"], "layers": meta["order"],
        "head_size": meta["config"]["head_size"], "backend": meta["config"]["backend"],
        "transformers": "5.8.0", "state_layout": "B,H,value,key", "cross_layer_v_first": False}, indent=2))
    print(out)
