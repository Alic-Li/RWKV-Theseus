#!/usr/bin/env python3
"""One local teacher branch and DDP TimeMix per torchrun rank."""
import argparse
from contextlib import nullcontext
import json
import random
import time
import warnings
import os
from tqdm.auto import tqdm
from pathlib import Path

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
from theseus.hf_model import load_model, migration_order, install, Runner, Capture
from theseus.stage import prepare_student, promote, lr_factor
from theseus.losses import statistics, loss_sum, metrics
from theseus.checkpoint import (read_checkpoint, save_checkpoint, validate_resume,
                                restore_rng, model_fingerprint)
from theseus.validation import validate, reader_for
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
    meta = None
    delta = {}
    checkpoint = None
    start_stage = 0
    if resume:
        checkpoint, meta, delta = read_checkpoint(resume)
        validate_resume(meta, cfg, topo, base_id, order)
        start_stage = meta["stage"] + int(meta["complete"])
        for layer in order[:start_stage]:
            install(model, layer, cfg, delta[layer], trainable=False)
    tracker.start(cfg, topo, resume=bool(resume))
    runner = Runner(model)
    total_updates = 0
    for stage in range(start_stage, len(order)):
        layer = order[stage]
        continuing = meta is not None and not meta["complete"] and stage == meta["stage"]
        step0 = meta["step"] if continuing else 0
        # Same deterministic initialization regardless of whether earlier stages were resumed.
        torch.manual_seed(cfg["seed"] + stage)
        adapter = ddp = optimizer = scheduler = capture = None
        reader = reader_for(cfg, topo)
        max_steps = ((max(reader.rank_chunks) + cfg["grad_accum_steps"] - 1) // cfg["grad_accum_steps"]
                     if cfg["training_mode"] == "epoch" else steps_for(cfg, stage))
        adapter, ddp, optimizer, scheduler = prepare_student(
            model, layer, cfg, topo, delta[layer] if continuing else None, total_steps=max_steps)
        capture = Capture(adapter.teacher)
        runner.reset()
        if continuing:
            local = torch.load(checkpoint / "ranks" / f"rank_{topo.rank:05d}.pt", map_location="cpu", weights_only=False)
            runner.load_state_dict(local["runner"])
            if reader:
                reader.load_state_dict(local["reader"])
            opt = torch.load(checkpoint / "optimizer.pt", map_location=topo.device, weights_only=False)
            optimizer.load_state_dict(opt["optimizer"])
            scheduler.load_state_dict(opt["scheduler"])
            # Recompute the next update's LR with the current schedule. This
            # also lets old constant-LR checkpoints adopt the new schedule
            # without resetting Adam moments or restarting warmup.
            next_lrs = [cfg["lr"] * lr_factor(step0, cfg, max_steps)] * len(optimizer.param_groups)
            scheduler.base_lrs = [cfg["lr"]] * len(optimizer.param_groups)
            scheduler.last_epoch = step0
            scheduler._last_lr = next_lrs
            for group, lr in zip(optimizer.param_groups, next_lrs):
                group["lr"] = lr
                group["initial_lr"] = cfg["lr"]
            restore_rng(local["rng"])
        topo.barrier()
        emit(topo, cfg, "stage_start", stage=stage + 1, layer=layer, step=step0)
        tracker.epoch_steps = max_steps
        if cfg["training_mode"] == "epoch":
            emit(topo, cfg, "epoch_plan", stage=stage + 1, total_tokens=reader.total_tokens,
                 steps=max_steps, rank_chunks=reader.rank_chunks)

        progress = tqdm(total=max_steps, initial=step0, desc=f"Stage {stage + 1}/{len(order)} layer {layer}",
                        unit="step", dynamic_ncols=True, mininterval=0.5,
                        disable=topo.rank != topo.leader or os.environ.get("TQDM_DISABLE") == "1")
        throughput_started = time.perf_counter()
        throughput_steps = throughput_tokens = 0
        for step in range(step0 + 1, max_steps + 1):
            log_step = step % cfg["log_interval"] == 0 or step == 1
            profile = log_step and os.environ.get("THESEUS_PROFILE", "0") == "1"
            teacher_seconds = 0.
            profile_batches = []
            accum_stats = torch.zeros(3, device=topo.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(cfg["grad_accum_steps"]):
                batch = reader.next()
                sync = nullcontext() if micro == cfg["grad_accum_steps"] - 1 else ddp.no_sync()
                if batch is None:
                    # Keep collective ordering identical on exhausted ranks. A
                    # zero-loss dummy touches the same DDP graph, but contributes
                    # no tokens and never advances reader or recurrent state.
                    with sync:
                        with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                            dummy = torch.zeros((1, 1, model.config.hidden_size), device=topo.device,
                                                dtype=torch.bfloat16)
                            prediction, _ = ddp(dummy, None)
                        (prediction.float().sum() * 0).backward()
                    continue
                ids = batch["ids"].to(topo.device)
                if batch["reset"]:
                    runner.reset()
                if runner.cache.cursor != batch["position"]:
                    raise RuntimeError("Cache position differs from consumed token stream")
                if profile:
                    topo.synchronize()
                    teacher_started = time.perf_counter()
                adapter.teacher_mode = True
                try:
                    runner.forward(ids, through_layer=layer)
                finally:
                    adapter.teacher_mode = False
                if profile:
                    topo.synchronize()
                    teacher_seconds += time.perf_counter() - teacher_started
                    profile_batches.append({"tokens": ids.shape[1], "position": batch["position"], "reset": batch["reset"]})
                x, target = capture.x, capture.y
                with sync:
                    with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                        prediction, state = ddp(x.detach(), adapter.state)
                    adapter.state = detach_state(state)
                    stats = statistics(prediction, target, cfg["loss_epsilon"])
                    loss_sum(stats, cfg["cosine_weight"]).backward()
                accum_stats += stats.detach()
            topo.student_sum(accum_stats)
            scale = topo.world / accum_stats[2]
            for p in adapter.core.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.core.parameters(), cfg["clip_norm"], error_if_nonfinite=True)
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            if profile:
                import torch.distributed as dist
                reports = [None] * topo.world
                dist.all_gather_object(reports, {"rank": topo.rank, "teacher_seconds": teacher_seconds,
                                                "batches": profile_batches})
                emit(topo, cfg, "rank_profile", stage=stage + 1, step=step, ranks=reports)
            if topo.rank == topo.leader:
                current_metrics = metrics(accum_stats)
                throughput_steps += 1
                throughput_tokens += current_metrics["tokens"]
                elapsed = time.perf_counter() - throughput_started
                progress.set_postfix(nmse=f"{current_metrics['nmse']:.5f}",
                                     rrms=f"{current_metrics['rrms']:.5f}",
                                     cosine=f"{current_metrics['cosine']:.5f}", refresh=False)
                progress.update(1)
                if log_step:
                    emit(topo, cfg, "train", stage=stage + 1, layer=layer, step=step,
                         lr=lr_used, grad_norm=grad_norm.item(),
                         seconds_per_step=elapsed / throughput_steps,
                         tokens_per_second=throughput_tokens / elapsed, **current_metrics)
                    throughput_started = time.perf_counter()
                    throughput_steps = throughput_tokens = 0
            complete = step == max_steps
            if complete and cfg["training_mode"] == "epoch" and not reader.exhausted:
                raise RuntimeError("Stage ended before the local token shard was exhausted")
            if complete or (cfg["validation_interval"] and step % cfg["validation_interval"] == 0):
                result = validate(topo, cfg, stage, step, runner, layer)
                emit(topo, cfg, "validation", stage=stage + 1, step=step, **result)
            total_updates += 1
            stopping = stop_after is not None and total_updates >= stop_after
            if complete or stopping or (cfg["checkpoint_interval"] and step % cfg["checkpoint_interval"] == 0):
                saved = save_checkpoint(topo, cfg, stage, step, complete, order, model, runner,
                                        reader, optimizer, scheduler, base_id)
                emit(topo, cfg, "checkpoint", path=str(saved), complete=complete)
            if stopping:
                progress.close()
                if capture:
                    capture.close()
                topo.close()
                return
        progress.close()
        if capture:
            capture.close()
        if adapter:
            adapter.__dict__["train_call"] = None
        del ddp, optimizer, scheduler, adapter
        promote(topo, model, layer, cfg)
        runner.reset()
        meta = None
    emit(topo, cfg, "finished", stages=len(order))
    topo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Committed checkpoint directory, output directory, or latest file")
    parser.add_argument("--stop-after", type=int, help="Save and stop after this many optimizer updates (smoke tests)")
    args = parser.parse_args()
    run(load_config(args.config), args.resume, args.stop_after)
