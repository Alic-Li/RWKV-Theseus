"""All-rank local distillation DDP; Gloo control, NCCL gradients on CUDA."""
from datetime import timedelta
import hashlib
import json
import os
import socket
import torch
import torch.distributed as dist


class Topology:
    """Each rank owns a local teacher branch and a trainable TimeMix."""
    def __init__(self, backend="nccl", timeout_minutes=60):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        local_world = int(os.environ["LOCAL_WORLD_SIZE"])
        if self.world < 1 or local_world < 1 or self.world % local_world or self.local_rank != self.rank % local_world:
            raise ValueError("Use uniform torchrun rank allocation")
        self.backend = backend
        if backend == "nccl":
            if not torch.cuda.is_available() or torch.cuda.device_count() < local_world:
                raise RuntimeError("Need one visible GPU per local rank")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16-capable GPUs are required")
        elif backend == "gloo":
            self.device = torch.device("cpu")
        else:
            raise ValueError(backend)
        timeout = timedelta(minutes=timeout_minutes)
        dist.init_process_group("gloo", timeout=timeout)
        facts = (local_world, backend, socket.gethostname(),
                 str(torch.cuda.get_device_properties(self.device).uuid) if backend == "nccl" else None)
        reports = [None] * self.world
        dist.all_gather_object(reports, facts)
        if any(v[:2] != facts[:2] for v in reports):
            raise ValueError("Inconsistent rank configuration")
        devices = [v[2:] for v in reports if v[3] is not None]
        if len(devices) != len(set(devices)):
            raise ValueError("Multiple ranks share a GPU")
        self.students = list(range(self.world))
        self.leader = 0
        self.groups = []
        self.student_group = self._create_group(self.students, timeout)

    def _create_group(self, ranks, timeout):
        group = dist.new_group(ranks, backend=self.backend, timeout=timeout)
        self.groups.append(group)
        # Warm up each communicator before touching the next one. Surface NCCL
        # device/network failures before loading 27B weights, including size-1 DDP.
        if self.rank in ranks:
            value = torch.ones(1, device=self.device)
            dist.all_reduce(value, group=group)
            self.synchronize()
            if value.item() != len(ranks):
                raise RuntimeError("Process group warm-up failed")
        dist.barrier()  # Gloo, even for ranks outside the subgroup
        return group

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def barrier(self):
        self.synchronize()
        dist.barrier()

    def close(self):
        self.barrier()
        # PyTorch destroys all subgroups in a consistent order. Avoid GC teardown.
        dist.destroy_process_group()

    def assert_same(self, value, label):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        all_digests = [None] * self.world
        dist.all_gather_object(all_digests, digest)
        if len(set(all_digests)) != 1:
            raise ValueError(f"Ranks disagree on {label}: {all_digests}")

    def student_sum(self, tensor):
        # Same NCCL group and current-stream dependencies as DDP. The consumer
        # establishes completion; no device-wide drains around every reduction.
        dist.all_reduce(tensor, group=self.student_group)
        return tensor

    def broadcast_object(self, value, source=None):
        self.synchronize()
        box = [value]
        dist.broadcast_object_list(box, src=self.leader if source is None else source)
        return box[0]

    def broadcast_weights(self, state):
        """Stage snapshots use CPU tensor broadcasts, not a 200 MB pickle object."""
        self.barrier()
        spec = [(name, tuple(value.shape), str(value.dtype).removeprefix("torch."))
                for name, value in sorted(state.items())] if self.rank == self.leader else None
        spec = self.broadcast_object(spec)
        result = {}
        digest = hashlib.sha256()
        for name, shape, dtype in spec:
            value = (state[name].detach().cpu().contiguous() if self.rank == self.leader
                     else torch.empty(shape, dtype=getattr(torch, dtype)))
            # Raw bytes also support BF16 without relying on Gloo BF16 arithmetic.
            dist.broadcast(value.view(torch.uint8), src=self.leader)
            digest.update(name.encode())
            digest.update(memoryview(value.view(torch.uint8).numpy()))
            result[name] = value
        self.assert_same(digest.hexdigest(), "promoted weights")
        return result
