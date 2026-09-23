import pytest
import torch
from theseus.timemix import TimeMix
from theseus.wkv7 import reference, wkv7


def test_streaming_output_and_state():
    model = TimeMix(32, 3, 8, head_size=8, backend="reference")
    torch.nn.init.normal_(model.output.weight, std=.03)
    x = torch.randn(2, 9, 32)
    full, end = model(x)
    a, state = model(x[:, :4])
    b, state = model(x[:, 4:], state)
    torch.testing.assert_close(torch.cat([a, b], 1), full)
    for key in state:
        torch.testing.assert_close(state[key], end[key])


def test_initialization_can_learn():
    model = TimeMix(32, 3, 8, head_size=8, backend="reference")
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    x, target = torch.randn(1, 5, 32), torch.randn(1, 5, 32)
    for step in range(3):
        opt.zero_grad()
        y, _ = model(x)
        (y - target).square().mean().backward()
        assert model.output.weight.grad.abs().sum() > 0
        if step:
            assert model.receptance.weight.grad.abs().sum() > 0
            assert model.g1.grad.abs().sum() > 0
        opt.step()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [8, 64])
def test_cuda_against_autograd_nonzero_state(dtype, n):
    shape = (2, 7, 2, n)
    xs = [(torch.randn(shape, device="cuda", dtype=dtype) * .1).requires_grad_() for _ in range(6)]
    s0 = (torch.randn(2, 2, n, n, device="cuda") * .1).requires_grad_()
    ref_x = [x.detach().clone().requires_grad_() for x in xs]
    ref_s = s0.detach().clone().requires_grad_()
    y, state = wkv7(*xs, s0)
    yr, sr = reference(*ref_x, ref_s)
    torch.testing.assert_close(y, yr, atol=.001 if dtype == torch.bfloat16 else 2e-6, rtol=.02 if dtype == torch.bfloat16 else 1e-4)
    torch.testing.assert_close(state, sr, atol=3e-6, rtol=3e-4)
    dy, ds = torch.randn_like(y), torch.randn_like(state)
    torch.autograd.backward((y, state), (dy, ds))
    torch.autograd.backward((yr, sr), (dy, ds))
    for a, b in zip([*xs, s0], [*ref_x, ref_s]):
        torch.testing.assert_close(a.grad, b.grad, atol=.004 if dtype == torch.bfloat16 else 3e-6, rtol=.025 if dtype == torch.bfloat16 else 5e-4)


def test_tbptt_detaches_only_at_chunk_boundary():
    model = TimeMix(16, 1, 4, head_size=4, backend="reference")
    torch.nn.init.normal_(model.output.weight, std=.05)
    x1 = torch.randn(1, 3, 16, requires_grad=True)
    x2 = torch.randn(1, 4, 16, requires_grad=True)
    from theseus.timemix import detach_state
    _, state = model(x1)
    y, _ = model(x2, detach_state(state))
    y.sum().backward()
    assert x1.grad is None
    assert x2.grad is not None and x2.grad.abs().sum() > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inference_chunking_and_current_stream():
    model = TimeMix(128, 3, 8, head_size=64, backend="cuda").cuda().to(torch.bfloat16)
    torch.nn.init.normal_(model.output.weight, std=.01)
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full, final = model(x)
        a, state = model(x[:, :5])
        b, state = model(x[:, 5:], state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(torch.cat([a, b], 1), full, atol=.006, rtol=.025)
    torch.testing.assert_close(state["wkv"], final["wkv"], atol=.0005, rtol=.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fp32_optimizer_states_and_learning():
    model = TimeMix(128, 3, 8, head_size=64, backend="cuda").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    x = torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, _ = model(x)
        loss = (y.float() - target.float()).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    for param, state in optimizer.state.items():
        assert param.dtype == param.grad.dtype == torch.float32
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
