import pytest
import torch
from theseus.stage import lr_factor

CFG = dict(lr=1e-4, min_lr=3e-6, warmup_steps=300, constant_steps=3000)


def test_optimizer_update_boundaries():
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.AdamW([p], lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_factor(n, CFG, 5300))
    rates = []
    for _ in range(5300):
        rates.append(opt.param_groups[0]['lr'])
        opt.step()
        scheduler.step()
    assert rates[0] == pytest.approx(1e-4 / 300)
    assert rates[299:3300] == pytest.approx([1e-4] * 3001)
    assert rates[3300] < 1e-4
    assert rates[4299] == pytest.approx((1e-4 + 3e-6) / 2)
    assert rates[-1] == pytest.approx(3e-6)
    assert all(a >= b for a, b in zip(rates[3299:], rates[3300:]))
    assert opt.param_groups[0]['lr'] == pytest.approx(3e-6)


def test_short_stage_and_disabled_phases():
    assert lr_factor(9, CFG, 10) == pytest.approx(10 / 300)
    assert lr_factor(499, CFG, 500) == 1.
    cfg = CFG | dict(warmup_steps=0, constant_steps=0)
    assert lr_factor(99, cfg, 100) == pytest.approx(.03)
