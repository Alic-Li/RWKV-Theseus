#!/usr/bin/env python3
"""Immutable local Qwen teacher and independent per-layer DDP TimeMix students."""
import argparse
from contextlib import nullcontext
import json
import random
import warnings
import os
from pathlib import Path

# HF's growing KV cache otherwise fragments the CUDA caching allocator across
# long windows. Respect an explicit allocator configuration from the caller.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from tqdm.auto import tqdm

# Known upstream deprecations and FLA's shape heuristic are noisy in this exact
# path. The Qwen call is [B,T,H,...]; T < H is valid for a short document tail.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated.*", category=FutureWarning,
                        module=r"torch\.jit\._script")
warnings.filterwarnings("ignore", message=r"tl\.make_block_ptr is deprecated.*", category=UserWarning,
                        module=r"triton\.language\.core")
warnings.filterwarnings("ignore", message=r"Input tensor shape suggests potential format mismatch:.*",
                        category=UserWarning, module=r"fla\.ops\.gated_delta_rule\.chunk")

import numpy as np
import torch
from theseus.config import load_config, steps_for
from theseus.distributed import Topology
from theseus.hf_model import load_model, migration_order, install, Runner, Capture, _state_to_device
from theseus.stage import prepare_student, lr_factor
from theseus.losses import statistics, metrics
from theseus.checkpoint import (read_checkpoint, save_checkpoint, validate_resume,
                                restore_rng, model_fingerprint, cpu_tree)
from theseus.validation import (reader_for, validate_parallel, validate_composition, summarize_layers)
from theseus.tracking import Tracker
from theseus.timemix import detach_state


def emit(topo, cfg, event, **fields):
    if topo.rank != topo.leader:
        return
    line = json.dumps({"event": event, **fields}, ensure_ascii=False)
    # tqdm.write(line)  # Keep event JSON in metrics.jsonl; console uses tqdm.
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metrics.jsonl").open("a") as f:
        f.write(line + "\n")
    topo.tracker.log(event, fields)


def run(cfg, resume=None, stop_after=None):
    tracker = Tracker()
    try:
        _run(cfg, resume, stop_after, tracker)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish()


def _run(cfg, resume, stop_after, tracker):
    topo = Topology(cfg["distributed_backend"], cfg["timeout_minutes"])
    topo.tracker = tracker
    topo.assert_same({"config": cfg, "resume": resume, "stop_after": stop_after,
                      "profile": os.environ.get("THESEUS_PROFILE", "0"),
                      "gdn_kernel": os.environ.get("THESEUS_GDN_KERNEL", "auto")}, "run configuration")
    torch.set_num_threads(1)
    random.seed(cfg["seed"] + topo.rank)
    np.random.seed(cfg["seed"] + topo.rank)
    torch.manual_seed(cfg["seed"])
    # Validate data bytes once before any model allocation, then synchronize.
    if topo.rank == 0:
        from theseus.data import digest_file
        for manifest in (cfg["train_manifest"], cfg["validation_manifest"]):
            data = json.loads(Path(manifest).read_text())
            for source in data["sources"]:
                for field in ("tokens", "offsets"):
                    if digest_file(Path(manifest).parent / source[field]) != source[field + "_sha256"]:
                        raise ValueError(f"Data content changed: {source[field]}")
    if not resume and (Path(cfg["output"]) / "latest").exists():
        raise ValueError("Output already has checkpoints; use --resume or a new output directory")
    topo.barrier()
    base_id = topo.broadcast_object(model_fingerprint(cfg["model"]) if topo.rank == topo.leader else None)
    model = load_model(cfg["model"], topo.device)
    for manifest in (cfg["train_manifest"], cfg["validation_manifest"]):
        data = json.loads(Path(manifest).read_text())
        if data["vocab_size"] > model.config.vocab_size:
            raise ValueError("Dataset vocabulary exceeds model embedding size")
        for name, expected in data.get("tokenizer_files", {}).items():
            from theseus.data import digest_file
            token_file = Path(cfg["model"]) / name
            if not token_file.exists() or digest_file(token_file) != expected:
                raise ValueError(f"Dataset/model tokenizer differs: {name}")
    order = migration_order(model, cfg["expected_attention_layers"])
    if cfg["cosine_weight"] != 0:
        raise ValueError("Parallel migration requires cosine_weight=0")
    meta = None
    delta = {}
    checkpoint = None
    if resume:
        checkpoint, meta, delta = read_checkpoint(resume)
        if meta.get("training_topology") != "parallel_layers_v1":
            raise ValueError("Sequential checkpoints cannot resume parallel migration; start a new run")
        validate_resume(meta, cfg, topo, base_id, order)
        if set(delta) != set(order):
            raise ValueError("Checkpoint must contain every migration layer")
    export_meta = Path(cfg["output"]) / "converted" / "theseus.json"
    if meta and meta["complete"] and export_meta.exists():
        from theseus.data import digest_file
        exported = json.loads(export_meta.read_text())
        if (exported.get("migration_fingerprint") != digest_file(checkpoint / "migrated.safetensors")
                or exported.get("base_fingerprint") != base_id):
            raise ValueError("Existing converted model differs from resumed checkpoint; choose a new output")
        topo.close()
        return
    saved = checkpoint
    tracker.start(cfg, topo, resume=bool(resume))
    runner = Runner(model)
    reader = reader_for(cfg, topo)
    max_steps = ((max(reader.rank_chunks) + cfg["grad_accum_steps"] - 1) // cfg["grad_accum_steps"]
                 if cfg["training_mode"] == "epoch" else steps_for(cfg, 0))
    step0 = meta["step"] if meta else 0
    students, captures = {}, {}
    for index, layer in enumerate(order):
        torch.manual_seed(cfg["seed"] + index)
        students[layer] = prepare_student(model, layer, cfg, topo, delta.get(layer),
                                          total_steps=max_steps, detached=True)
        captures[layer] = Capture(model.model.layers[layer].self_attn)
    if meta and not meta["complete"]:
        local = torch.load(checkpoint / "ranks" / f"rank_{topo.rank:05d}.pt", map_location="cpu", weights_only=False)
        runner.load_state_dict(local["runner"])
        reader.load_state_dict(local["reader"])
        opt = torch.load(checkpoint / "optimizer.pt", map_location=topo.device, weights_only=False)
        for layer, (adapter, _, optimizer, scheduler) in students.items():
            adapter.state = _state_to_device(local["students"][layer], topo.device)
            optimizer.load_state_dict(opt[layer]["optimizer"])
            scheduler.load_state_dict(opt[layer]["scheduler"])
            lrs = [cfg["lr"] * lr_factor(step0, cfg, max_steps)] * len(optimizer.param_groups)
            scheduler.base_lrs = [cfg["lr"]] * len(lrs)
            scheduler.last_epoch = step0
            scheduler._last_lr = lrs
            for group, lr in zip(optimizer.param_groups, lrs):
                group.update(lr=lr, initial_lr=cfg["lr"])
        restore_rng(local["rng"])
    delta.clear()
    local = opt = group = None
    topo.barrier()
    progress = tqdm(total=max_steps, initial=step0, desc=f"Parallel migration ({len(order)} layers)",
                    unit="step", dynamic_ncols=True, mininterval=0.5,
                    disable=topo.rank != topo.leader or os.environ.get("TQDM_DISABLE") == "1")
    for step in range(step0 + 1, max_steps + 1):
        accum = {layer: torch.zeros(3, device=topo.device) for layer in order}
        layer_metrics = {}
        for _, _, optimizer, _ in students.values():
            optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg["grad_accum_steps"]):
            batch = reader.next()
            if batch is not None:
                if batch["reset"]:
                    runner.reset()
                    for adapter, *_ in students.values():
                        adapter.state = None
                if runner.cache.cursor != batch["position"]:
                    raise RuntimeError("Cache position differs from consumed token stream")
                # Exactly one immutable Qwen pass supplies every layer boundary.
                runner.forward(batch["ids"].to(topo.device))
            last_micro = micro == cfg["grad_accum_steps"] - 1
            for layer in order:
                adapter, ddp, optimizer, scheduler = students[layer]
                capture = captures[layer]
                sync = nullcontext() if topo.world == 1 or last_micro else ddp.no_sync()
                with sync:
                    if batch is None:
                        # Exhausted ranks still participate in each layer's DDP collectives.
                        x = torch.zeros((1, 1, model.config.hidden_size), device=topo.device,
                                        dtype=torch.bfloat16)
                        with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                            prediction, state = ddp(x, None)
                        loss = prediction.float().sum() * 0
                    else:
                        x, target = capture.x, capture.y
                        with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                            prediction, state = ddp(x, adapter.state)
                        adapter.state = detach_state(state)
                        stats = statistics(prediction, target, cfg["loss_epsilon"])
                        accum[layer] += stats.detach()
                        # Normalize the accumulated token sum after DDP reduction.
                        loss = stats[0]
                    loss.backward()
                # No autograd-connected tensor survives this layer's backward.
                del prediction, state, loss, x
                if batch is not None:
                    del stats, target
                capture.x = capture.y = None
                if last_micro:
                    topo.student_sum(accum[layer])
                    count = accum[layer][2]
                    if count.item() <= 0:
                        raise RuntimeError("No training tokens in update")
                    for p in adapter.core.parameters():
                        if p.grad is not None:
                            p.grad.mul_(topo.world / count)
                    norm = torch.nn.utils.clip_grad_norm_(adapter.core.parameters(), cfg["clip_norm"],
                                                         error_if_nonfinite=True)
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    layer_metrics[f"layer_{layer:02d}"] = metrics(accum[layer]) | {"grad_norm": norm.item(), "lr": lr}
        summary = summarize_layers(layer_metrics)
        progress.set_postfix(sum_loss=f"{summary['sum_loss']:.5f}", mean_cos=f"{summary['mean_cos']:.5f}", refresh=False)
        progress.update(1)
        if step % cfg["log_interval"] == 0 or step == 1:
            emit(topo, cfg, "train", step=step, **summary,
                 tokens=next(iter(layer_metrics.values()))["tokens"], layers=layer_metrics)
        complete = step == max_steps
        if complete and cfg["training_mode"] == "epoch" and not reader.exhausted:
            raise RuntimeError("Training ended before the local token shard was exhausted")
        if complete or (cfg["validation_interval"] and step % cfg["validation_interval"] == 0):
            result = validate_parallel(topo, cfg, runner, students, captures)
            emit(topo, cfg, "validation", step=step, **result)
        stopping = stop_after is not None and step - step0 >= stop_after
        if complete or stopping or (cfg["checkpoint_interval"] and step % cfg["checkpoint_interval"] == 0):
            saved = save_checkpoint(topo, cfg, 0, step, complete, order, model, runner,
                                    reader, None, None, base_id, students=students)
            emit(topo, cfg, "checkpoint", path=str(saved), complete=complete)
        if stopping and not complete:
            progress.close()
            for capture in captures.values():
                capture.close()
            topo.close()
            return
    progress.close()
    for capture in captures.values():
        capture.close()
    # Release teacher, caches, DDP and Adam before reloading the pristine base.
    weights = {i: cpu_tree(item[0].core.state_dict()) for i, item in students.items()}
    students.clear()
    adapter = ddp = optimizer = scheduler = p = norm = local = opt = delta = group = None
    _ = None
    del runner, model
    import gc
    gc.collect()
    if topo.device.type == "cuda":
        torch.cuda.empty_cache()
    model = load_model(cfg["model"], topo.device)
    for layer in order:
        install(model, layer, cfg, weights[layer])
    del weights
    runner = Runner(model)
    result = validate_composition(topo, cfg, runner)
    emit(topo, cfg, "composition_validation", **result)
    if topo.rank == topo.leader:
        from theseus.inference import save_export
        save_export(model, Path(cfg["output"]) / "converted", cfg, order, base_id, checkpoint=saved)
    topo.barrier()
    emit(topo, cfg, "finished", layers=len(order), export=str(Path(cfg["output"]) / "converted"))
    topo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Committed checkpoint directory, output directory, or latest file")
    parser.add_argument("--stop-after", type=int, help="Save and stop after this many optimizer updates (smoke tests)")
    args = parser.parse_args()
    run(load_config(args.config), args.resume, args.stop_after)
