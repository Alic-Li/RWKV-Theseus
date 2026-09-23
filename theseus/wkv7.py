"""RWKV7 recurrence. State layout: [batch, head, value, key].

The reference supports full autograd; the CUDA implementation saves FP32 state
at every timestep (simple and stable, but memory grows with chunk length).
Chunk boundaries are detached by the caller, never silently by this operator.
Recurrence source: RWKV-Vibe/RWKV-LM-V7, commit
665472dab30952de9379a3a3a01eaa3453f1ad4e, cuda/rwkv7_clampw.cu.
Apache-2.0, see LICENSE-RWKV7. This adaptation exposes s0/sT and uses direct
reverse-mode gradients with saved FP32 states instead of inverse reconstruction.
"""
from functools import lru_cache
from pathlib import Path
import torch


def reference(r, w, k, v, a, b, state):
    dtype = r.dtype
    r, w, k, v, a, b = [x.float() for x in (r, w, k, v, a, b)]
    decay = torch.exp(-0.6065306597126334 * torch.sigmoid(w))
    out = []
    state = state.float()
    for t in range(r.shape[1]):
        sa = (state * a[:, t].unsqueeze(-2)).sum(-1)
        state = (state * decay[:, t].unsqueeze(-2)
                 + sa.unsqueeze(-1) * b[:, t].unsqueeze(-2)
                 + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2))
        out.append((state * r[:, t].unsqueeze(-2)).sum(-1))
    return torch.stack(out, 1).to(dtype), state


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent.parent / "kernels"
    return load(name="theseus_wkv7_state_v1", sources=[str(root / "wkv7_state.cpp"), str(root / "wkv7_state.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)


class _WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, state):
        xs = [x.contiguous() for x in (r, w, k, v, a, b)]
        state = state.float().contiguous()
        y, end, history = extension().forward(*xs, state, True)
        ctx.save_for_backward(*xs, history)
        return y, end

    @staticmethod
    def backward(ctx, dy, dend):
        *xs, history = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(xs[0])
        if dend is None:
            dend = torch.zeros_like(history[:, :, 0])
        grads = extension().backward(*xs, history, dy.contiguous(), dend.float().contiguous())
        return tuple(g.to(x.dtype) for g, x in zip(grads[:6], xs)) + (grads[6],)


def wkv7(r, w, k, v, a, b, state, backend="cuda"):
    if backend == "reference":
        return reference(r, w, k, v, a, b, state)
    if backend != "cuda":
        raise ValueError(f"Unknown WKV backend: {backend}")
    if not r.is_cuda:
        raise ValueError("CUDA backend requested on CPU; use backend='reference' for tests")
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in (r, w, k, v, a, b, state)):
        xs = [x.contiguous() for x in (r, w, k, v, a, b)]
        y, end, _ = extension().forward(*xs, state.float().contiguous(), False)
        return y, end
    return _WKV.apply(r, w, k, v, a, b, state)
