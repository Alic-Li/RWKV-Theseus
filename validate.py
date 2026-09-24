#!/usr/bin/env python3
"""Standalone local-branch validation with all-rank metric reduction."""
import argparse
import json
from theseus.config import load_config
from theseus.distributed import Topology
from theseus.hf_model import load_model, install, Runner
from theseus.checkpoint import read_checkpoint, model_fingerprint
from theseus.validation import validate, validate_parallel, validate_composition
from theseus.stage import prepare_student
from theseus.hf_model import Capture

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    topo = Topology(cfg["distributed_backend"], cfg["timeout_minutes"])
    topo.assert_same({"config": cfg, "checkpoint": args.checkpoint}, "validation configuration")
    _, meta, delta = read_checkpoint(args.checkpoint)
    if meta["base_fingerprint"] != model_fingerprint(cfg["model"]):
        raise ValueError("Validation base differs from training")
    model = load_model(cfg["model"], topo.device)
    if meta.get("training_topology") == "parallel_layers_v1":
        if meta["complete"]:
            for layer in meta["order"]:
                install(model, layer, cfg, delta[layer])
            results = validate_composition(topo, cfg, Runner(model))
        else:
            students = {layer: prepare_student(model, layer, cfg, topo, delta[layer],
                                               total_steps=1, detached=True) for layer in meta["order"]}
            captures = {layer: Capture(model.model.layers[layer].self_attn) for layer in meta["order"]}
            try:
                results = validate_parallel(topo, cfg, Runner(model), students, captures)
            finally:
                for capture in captures.values():
                    capture.close()
    else:
        for layer in meta["order"][:meta["stage"]]:
            install(model, layer, cfg, delta[layer])
        layer = meta["order"][meta["stage"]]
        install(model, layer, cfg, delta[layer], trainable=True, retain_teacher=True)
        model.requires_grad_(False)
        results = validate(topo, cfg, meta["stage"], meta["step"], Runner(model), layer)
    if topo.rank == topo.leader:
        print(json.dumps(results, indent=2))
    topo.close()
