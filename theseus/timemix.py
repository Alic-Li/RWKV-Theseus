"""Standalone RWKV7 TimeMix, adapted from the pinned RWKV-Vibe source.

No cross-layer v_first. Original decoder depth controls initialization.
Source: https://github.com/RWKV-Vibe/RWKV-LM-V7/blob/
665472dab30952de9379a3a3a01eaa3453f1ad4e/src/model.py
Copied/adapted RWKV_Tmix_x070 formulas and generate_init_weight initialization.
Apache-2.0; license text is in LICENSE-RWKV7 at the project root.
Changes: independent blocks (no v_first), explicit shift/WKV state, native
PyTorch surrounding operations, and FP32 normalization/optimizer parameters.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .wkv7 import wkv7


class TimeMix(nn.Module):
    def __init__(self, width, layer_idx, depth, head_size=64, backend="cuda"):
        super().__init__()
        if width % head_size or head_size < 2:
            raise ValueError("width must be divisible by head_size >= 2")
        self.width, self.head_size = width, head_size
        self.heads, self.backend = width // head_size, backend
        c, n = width, head_size
        ratio = layer_idx / max(depth - 1, 1)
        reverse = 1 - layer_idx / depth
        d = torch.arange(c).reshape(1, 1, c) / c
        for name, power in [("r", .2), ("w", .9), ("k", .7), ("v", .7), ("a", .9), ("g", .2)]:
            setattr(self, "x_" + name, nn.Parameter(1 - d.pow(power * reverse)))
        linear = torch.arange(c) / (c - 1) - .5
        zigzag = (torch.arange(c) % n - (n - 1) / 2) / ((n - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        www = -6 + 6 * (torch.arange(c) / (c - 1)).pow(1 + ratio ** .3)
        rank = max(32, round(2.5 * math.sqrt(c) / 32) * 32)
        gate_rank = max(32, round(5 * math.sqrt(c) / 32) * 32)
        for name in ("w", "a"):
            setattr(self, name + "1", nn.Parameter(torch.zeros(c, rank)))
            p = nn.Parameter(torch.empty(rank, c))
            nn.init.orthogonal_(p, gain=.1 * max(1, math.sqrt(rank / c)))
            setattr(self, name + "2", p)
        self.w0 = nn.Parameter((www + .5 + zigzag * 2.5).reshape(1, 1, c))
        self.a0 = nn.Parameter((-.19 + zigzag * .3 + linear * .4).reshape(1, 1, c))
        self.g1 = nn.Parameter(torch.empty(c, gate_rank))
        self.g2 = nn.Parameter(torch.empty(gate_rank, c))
        nn.init.zeros_(self.g1)  # sigmoid(0)=0.5, so the gate is not identically zero
        nn.init.orthogonal_(self.g2, gain=.1 * max(1, math.sqrt(gate_rank / c)))
        self.k_k = nn.Parameter((.71 - linear * .1).reshape(1, 1, c))
        self.k_a = nn.Parameter(torch.full((1, 1, c), 1.02))
        self.r_k = nn.Parameter(torch.full((self.heads, n), -.04))
        for name, gain in [("receptance", 1.), ("key", .1), ("value", 1.), ("output", 0.)]:
            module = nn.Linear(c, c, bias=False)
            if gain:
                nn.init.orthogonal_(module.weight, gain=gain)
            else:
                nn.init.zeros_(module.weight)
            setattr(self, name, module)
        self.ln_x = nn.GroupNorm(self.heads, c, eps=64e-5)
        nn.init.constant_(self.ln_x.weight, ((1 + layer_idx) / depth) ** .7)

    def initial_state(self, x):
        return {"shift": torch.zeros(x.shape[0], self.width, device=x.device, dtype=x.dtype),
                "wkv": torch.zeros(x.shape[0], self.heads, self.head_size, self.head_size,
                                   device=x.device, dtype=torch.float32)}

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state(x)
        batch, length, _ = x.shape
        delta = torch.cat([state["shift"].unsqueeze(1), x[:, :-1]], 1) - x
        mixed = {name: (x + delta * getattr(self, "x_" + name)).to(x.dtype) for name in "rwkvag"}
        r = self.receptance(mixed["r"])
        w = self.w0 + torch.tanh(mixed["w"] @ self.w1) @ self.w2
        k, v = self.key(mixed["k"]), self.value(mixed["v"])
        a = torch.sigmoid(self.a0 + (mixed["a"] @ self.a1) @ self.a2)
        g = torch.sigmoid(mixed["g"] @ self.g1) @ self.g2
        shape = (batch, length, self.heads, self.head_size)
        kk = F.normalize((k.float() * self.k_k).reshape(shape), dim=-1, eps=1e-12)
        k = k * (1 + (a - 1) * self.k_a)
        args = [z.reshape(shape).to(r.dtype).contiguous() for z in (r, w, k, v, -kk, kk * a.reshape(shape))]
        y, end = wkv7(*args, state["wkv"], backend=self.backend)
        # Disable autocast for norms; cast back for the output projection.
        with torch.autocast(x.device.type, enabled=False):
            y = F.group_norm(y.float().reshape(batch * length, self.width), self.heads,
                             self.ln_x.weight.float(), self.ln_x.bias.float(), 64e-5).reshape(batch, length, self.width)
            extra = ((r.float().reshape(shape) * k.float().reshape(shape) * self.r_k).sum(-1, keepdim=True)
                     * v.float().reshape(shape)).reshape(batch, length, self.width)
            y = ((y + extra) * g.float()).to(x.dtype)
        return self.output(y).to(x.dtype), {"shift": x[:, -1], "wkv": end}


def detach_state(state):
    return None if state is None else {k: v.detach() for k, v in state.items()}
