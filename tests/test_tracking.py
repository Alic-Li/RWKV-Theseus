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
        def define_metric(self, *args, **kwargs): pass
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
    tracker.log("train", dict(stage=2, step=1, nmse=.4, cosine=.8, lr=.001, grad_norm=2,
                              seconds_per_step=.25, tokens_per_second=2048))
    tracker.log("validation", dict(stage=2, step=1,
        student_forward=dict(nmse=.3, rrms=.3**.5, cosine=.9), original_teacher_kl=.01))
    assert logs[0]["progress/global_step"] == logs[1]["progress/global_step"] == 3
    assert logs[0]["train/loss"] == pytest.approx(.41)
    assert logs[0]["train/tokens_per_second"] == 2048
    assert logs[1]["val/original_teacher_kl"] == .01
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
