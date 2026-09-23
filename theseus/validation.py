from contextlib import contextmanager
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from .data import TokenStream, EpochTokenStream
from .hf_model import AttentionAdapter, Capture, isolated_state
from .losses import statistics, metrics, kl_sum
from .timemix import detach_state


def reader_for(cfg, topo, validation=False):
    reader_class = EpochTokenStream if not validation and cfg.get("training_mode", "sampled") == "epoch" else TokenStream
    return reader_class(cfg["validation_manifest" if validation else "train_manifest"],
                       topo.rank, topo.world, cfg["chunk_tokens"], cfg["context_tokens"], cfg["seed"])


@contextmanager
def original_attention(model, model_path):
    """Temporarily reconstruct M0 from the immutable base, one attention at a time."""
    path = Path(model_path)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(path / "model.safetensors"), framework="pt") as f:
            weight_map = {key: "model.safetensors" for key in f.keys()}
    device = next(model.parameters()).device
    originals = {}
    try:
        for i, block in enumerate(model.model.layers):
            old = getattr(block, "self_attn", None)
            if not isinstance(old, AttentionAdapter):
                continue
            originals[i] = old
            # Keep DDP parameters on their original device/storage. Moving an
            # active core to CPU and back would invalidate reducer assumptions.
            module = Qwen3_5Attention(model.config, i).to(dtype=torch.bfloat16)
            prefix = next((p for p in (f"model.layers.{i}.self_attn.", f"model.language_model.layers.{i}.self_attn.")
                           if any(k.startswith(p) for k in weight_map)), None)
            if prefix is None:
                raise ValueError(f"Missing original attention {i}")
            state = {}
            for shard in sorted({v for k, v in weight_map.items() if k.startswith(prefix)}):
                with safe_open(str(path / shard), framework="pt") as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            state[key[len(prefix):]] = f.get_tensor(key)
            module.load_state_dict(state, strict=True)
            block.self_attn = module.to(device).requires_grad_(False).eval()
        yield
    finally:
        for i, module in originals.items():
            model.model.layers[i].self_attn = module


@torch.no_grad()
def validate(topo, cfg, stage, step, runner, layer):
    """Replay identical local tokens through isolated teacher/student caches.

    Only boundary targets are staged on CPU; no second model is allocated.
    """
    adapter = runner.model.model.layers[layer].self_attn
    total = torch.zeros((2, 3), device=topo.device, dtype=torch.float32)
    reader = reader_for(cfg, topo, validation=True)
    batches = [reader.next() for _ in range(cfg["validation_chunks"])]
    targets = []
    topo.barrier()
    with isolated_state(runner):
        capture = Capture(adapter.teacher)
        adapter.teacher_mode = True
        try:
            for batch in batches:
                if batch["reset"]:
                    runner.reset()
                runner.forward(batch["ids"].to(topo.device), through_layer=layer)
                targets.append((capture.x.cpu(), capture.y.cpu()))
        finally:
            adapter.teacher_mode = False
            capture.close()
    with isolated_state(runner):
        capture = Capture(adapter)
        forced_state = None
        try:
            for batch, (x, y) in zip(batches, targets):
                if batch["reset"]:
                    runner.reset()
                    forced_state = None
                x, y = x.to(topo.device), y.to(topo.device)
                runner.forward(batch["ids"].to(topo.device))
                if cfg["debug_input"]:
                    torch.testing.assert_close(capture.x, x, atol=.04, rtol=.02)
                total[1] += statistics(capture.y, y, cfg["loss_epsilon"])
                with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                    forced, forced_state = adapter.core(x, forced_state)
                forced_state = detach_state(forced_state)
                total[0] += statistics(forced, y, cfg["loss_epsilon"])
        finally:
            capture.close()
    topo.student_sum(total)
    results = {"teacher_forced": metrics(total[0]), "student_forward": metrics(total[1])}
    if cfg["original_teacher_kl"]:
        results.update(validate_original_kl(topo, cfg, stage, step, runner))
    topo.barrier()
    return results


@torch.no_grad()
def validate_original_kl(topo, cfg, stage, step, runner):
    # Store hidden states rather than vocabulary-sized logits. All ranks evaluate
    # their own data; original_attention restores M0 only for this isolated replay.
    reader = reader_for(cfg, topo, validation=True)
    batches = [reader.next() for _ in range(cfg["validation_chunks"])]
    teacher_hidden = []
    with isolated_state(runner):
        with original_attention(runner.model, cfg["model"]):
            runner.reset()
            for batch in batches:
                if batch["reset"]:
                    runner.reset()
                teacher_hidden.append(runner.forward(batch["ids"].to(topo.device)).cpu())
    total = torch.zeros(2, device=topo.device, dtype=torch.float32)
    with isolated_state(runner):
        for batch, teacher in zip(batches, teacher_hidden):
            if batch["reset"]:
                runner.reset()
            hidden = runner.forward(batch["ids"].to(topo.device))
            for start in range(0, hidden.shape[1], cfg["kl_token_block"]):
                end = start + cfg["kl_token_block"]
                with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                    logits = runner.model.lm_head(hidden[:, start:end])
                    target = runner.model.lm_head(teacher[:, start:end].to(topo.device))
                total[0] += kl_sum(logits, target)
                total[1] += logits.shape[0] * logits.shape[1]
    topo.student_sum(total)
    return {"original_teacher_kl": (total[0] / total[1]).item()}
