#!/usr/bin/env python3
"""Compose a completed migration into self-contained full model weights."""
import argparse
from theseus.checkpoint import read_checkpoint, model_fingerprint
from theseus.hf_model import load_model, install
from theseus.inference import save_export

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    checkpoint, meta, delta = read_checkpoint(args.checkpoint)
    if not meta["complete"] or set(delta) != set(meta["order"]):
        raise ValueError("Final export requires all layers to have completed training")
    cfg = meta["config"]
    if model_fingerprint(cfg["model"]) != meta["base_fingerprint"]:
        raise ValueError("Base weights changed")
    model = load_model(cfg["model"], "cpu")
    for layer in meta["order"]:
        install(model, layer, cfg, delta[layer])
    save_export(model, args.output, cfg, meta["order"], meta["base_fingerprint"], checkpoint=checkpoint)
    print(args.output)
