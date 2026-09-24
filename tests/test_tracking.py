import json
import sys
from types import SimpleNamespace
import pytest
from theseus.config import DEFAULTS
from theseus.tracking import Tracker


@pytest.mark.parametrize("training_mode", ["sampled", "epoch"])
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_leader_resume_and_metric_steps(tmp_path, monkeypatch, mode, training_mode):
    calls, logs, finished = [], [], []
    class Run:
        def define_metric(self, name, **kwargs):
            assert "*" not in name[:-1], "W&B supports suffix globs only"
        def log(self, values): logs.append(values)
        def finish(self, **kwargs): finished.append(kwargs)
    def init(**kwargs):
        calls.append(kwargs)
        return Run()
    ids = iter(["first", "second"])
    monkeypatch.setattr("theseus.tracking.uuid4", lambda: SimpleNamespace(hex=next(ids)))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, util=SimpleNamespace(generate_id=lambda: next(ids)), Settings=lambda **kwargs: kwargs))
    cfg = DEFAULTS | {"training_mode": training_mode, "output": str(tmp_path), "wandb_mode": mode,
                      "stage_steps": [2, 3], "cosine_weight": .05}
    tracker = Tracker()
    tracker.epoch_steps = 2
    tracker.start(cfg, SimpleNamespace(rank=0, leader=2))
    assert not calls
    tracker.start(cfg, SimpleNamespace(rank=2, leader=2))
    details = {"layer_03": dict(nmse=.4, cosine=.8, rrms=.4**.5, lr=.001, grad_norm=2)}
    tracker.log("train", dict(step=1, sum_loss=.4, mean_cos=.8, layers=details))
    tracker.log("validation", dict(step=1, sum_loss=.3, mean_cos=.9,
        layers={"layer_03": dict(nmse=.3, rrms=.3**.5, cosine=.9)}))
    assert logs[0]["progress/global_step"] == logs[1]["progress/global_step"] == 1
    assert logs[0]["sum_loss"] == .4
    assert logs[0]["mean_cos"] == .8
    assert logs[0]["layer_03/grad_norm"] == 2
    assert logs[1]["val/layer_03/cosine"] == .9
    assert set(k for k in logs[0] if "/" not in k) == {"sum_loss", "mean_cos"}
    tracker.log("checkpoint", {})
    assert len(logs) == 2
    tracker.finish()
    tracker.start(cfg, SimpleNamespace(rank=2, leader=2), resume=True)
    assert calls[-1]["id"] == ("first" if mode == "online" else "second")
    assert calls[-1]["resume"] == ("allow" if mode == "online" else None)
    tracker.finish(exit_code=1)
    assert finished == [{"exit_code": 0}, {"exit_code": 1}]


def test_disabled_does_not_import_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = Tracker()
    tracker.start(DEFAULTS, SimpleNamespace(rank=2, leader=2))
    tracker.log("train", {})
    tracker.finish()
