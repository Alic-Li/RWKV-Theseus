import inspect
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .hf_model import install


def lr_factor(step_index, cfg, total_steps):
    """LR for the next optimizer update (LambdaLR uses zero-based indices)."""
    step = step_index + 1
    warmup = cfg["warmup_steps"]
    plateau_end = warmup + cfg["constant_steps"]
    if step <= warmup:
        return step / warmup
    if step <= plateau_end:
        return 1.
    # Short smoke-test stages simply finish during warmup/plateau.
    progress = min(1., (step - plateau_end) / max(1, total_steps - plateau_end))
    floor = cfg["min_lr"] / cfg["lr"]
    return floor + (1. - floor) * .5 * (1. + math.cos(math.pi * progress))


def prepare_student(model, layer, cfg, topo, weights=None, *, total_steps, detached=False):
    model.requires_grad_(False)
    adapter = install(model, layer, cfg, weights=weights, trainable=True, retain_teacher=not detached, attach=not detached)
    adapter.debug_input = cfg["debug_input"]
    if topo.world == 1:
        ddp = adapter.core
    else:
        ddp_options = ({"forward_sync_buffers": False} if "forward_sync_buffers" in inspect.signature(DDP).parameters
                       else {"broadcast_buffers": False})
        ddp = DDP(adapter.core, process_group=topo.student_group,
                  device_ids=[topo.local_rank] if topo.device.type == "cuda" else None,
                  **ddp_options)
    adapter.__dict__["train_call"] = ddp
    decay, no_decay = [], []
    for name, p in adapter.core.named_parameters():
        (decay if name.endswith(".weight") and p.ndim == 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["adam_eps"],
                                  fused=topo.device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_factor(n, cfg, total_steps))
    return adapter, ddp, optimizer, scheduler


def promote(topo, model, layer, cfg):
    topo.barrier()
    if topo.rank == topo.leader:
        weights = {k: v.detach().cpu().to(torch.bfloat16) for k, v in model.model.layers[layer].self_attn.core.state_dict().items()}
    else:
        weights = None
    # One layer (~200 MB BF16), once per stage, in CPU tensor broadcasts.
    # Clear old DDP ownership first; don't mutate a DDP parameter set in place.
    weights = topo.broadcast_weights(weights)
    install(model, layer, cfg, weights=weights, trainable=False)
    topo.barrier()
