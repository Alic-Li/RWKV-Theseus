#!/usr/bin/env python3
"""Single-GPU HF/load/cache/one-layer backward probe; does not start long training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from theseus.hf_model import load_model, Runner, Capture, install, migration_order
from theseus.losses import statistics, metrics, loss_sum

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    model = load_model(args.model, "cuda")
    order = migration_order(model)
    print(json.dumps({"loaded": True, "order": order, "allocated_gb": torch.cuda.memory_allocated() / 1e9}), flush=True)
    runner = Runner(model)
    capture = Capture(model.model.layers[order[0]].self_attn)
    ids = torch.randint(100, min(model.config.vocab_size, 1000), (1, args.tokens), device="cuda")
    full = runner.forward(ids).clone()
    target_x, target_y = capture.x.clone(), capture.y.clone()
    runner.reset()
    cut = args.tokens // 2
    chunked = torch.cat([runner.forward(ids[:, :cut]), runner.forward(ids[:, cut:])], 1)
    cache_metrics = metrics(statistics(chunked, full))
    print(json.dumps({"chunk_equivalence": cache_metrics}), flush=True)
    if cache_metrics["nmse"] > .005:
        raise RuntimeError("HF cached chunks deviate excessively")
    capture.close()
    if args.backward:
        runner.reset()
        adapter = install(model, order[0], {"head_size": 64, "backend": "cuda"}, trainable=True)
        adapter.__dict__["train_call"] = adapter.core
        adapter.teacher_input = target_x
        adapter.debug_input = True
        runner.forward(ids)
        stats = statistics(adapter.prediction, target_y)
        loss = loss_sum(stats, 0.) / stats[2]
        loss.backward()
        grads = [n for n, p in model.named_parameters() if p.grad is not None]
        assert grads and all(f"layers.{order[0]}.self_attn.core." in name for name in grads)
        assert adapter.core.output.weight.grad.abs().sum() > 0
        print(json.dumps({"backward": metrics(stats.detach()), "trainable_tensors": len(grads),
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}), flush=True)
