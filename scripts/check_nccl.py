#!/usr/bin/env python3
"""All-rank CUDA TimeMix/DDP preflight, without loading 27B.

Run: torchrun --standalone --nproc_per_node=2 scripts/check_nccl.py
CPU audit: torchrun --standalone --nproc_per_node=4 scripts/check_nccl.py --backend gloo
"""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from theseus.distributed import Topology
from theseus.losses import statistics, loss_sum
from theseus.timemix import TimeMix, detach_state


def fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def run(args):
    torch.set_num_threads(1)
    topo = Topology(args.backend, timeout_minutes=3)
    topo.assert_same(vars(args), "preflight arguments")
    width = 64
    core_backend = "cuda" if topo.device.type == "cuda" else "reference"
    for stage in range(2):
        torch.manual_seed(123 + stage)
        core = ddp = optimizer = None
        state = None
        with torch.device(topo.device):
            core = TimeMix(width, stage, 2, head_size=8, backend=core_backend)
        ddp_options = ({"forward_sync_buffers": False}
                       if "forward_sync_buffers" in inspect.signature(DDP).parameters
                       else {"broadcast_buffers": False})
        ddp = DDP(core, process_group=topo.student_group,
                  device_ids=[topo.local_rank] if topo.device.type == "cuda" else None,
                  **ddp_options)
        optimizer = torch.optim.AdamW(core.parameters(), lr=.001)
        topo.barrier()
        position = 0
        for step in range(args.steps):
            sums = torch.zeros(3, device=topo.device)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                # Rank-dependent sequence lengths test token-weighted reduction.
                length = 3 + (topo.rank + step + micro) % 5
                ids = torch.arange(position, position + length, device=topo.device)[None]
                x = torch.sin(ids.float()[..., None] + torch.arange(width, device=topo.device) / width).to(torch.bfloat16)
                target = (x.float() * .2 + .1).to(torch.bfloat16)
                with (ddp.no_sync() if micro == 0 else nullcontext()):
                    with torch.autocast(topo.device.type, dtype=torch.bfloat16):
                        prediction, state = ddp(x, state)
                    state = detach_state(state)
                    stats = statistics(prediction, target)
                    loss_sum(stats, .05).backward()
                sums += stats.detach()
                position += length
            topo.student_sum(sums)
            for p in core.parameters():
                if p.grad is not None:
                    p.grad.mul_(topo.world / sums[2].item())
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for p, values in optimizer.state.items():
                assert p.dtype == p.grad.dtype == values["exp_avg"].dtype == values["exp_avg_sq"].dtype == torch.float32
            topo.synchronize()
            reports = [None] * topo.world
            dist.all_gather_object(reports, fingerprint(core))
            if len({reports[r] for r in topo.students}) != 1:
                raise RuntimeError("Student parameters diverged after DDP update")
        snapshot = ({k: v.detach().cpu().to(torch.bfloat16) for k, v in core.state_dict().items()}
                    if topo.rank == topo.leader else None)
        weights = topo.broadcast_weights(snapshot)
        assert weights and all(t.dtype == torch.bfloat16 and t.isfinite().all() for t in weights.values())
        if topo.rank == topo.leader:
            print(json.dumps({"stage": stage + 1, "backend": args.backend, "replicas": topo.world,
                              "ddp_students": topo.students, "steps": args.steps, "status": "passed"}), flush=True)
        del ddp, core, optimizer
        topo.barrier()
    topo.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    run(args)
