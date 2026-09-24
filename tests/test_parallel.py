"""Exercise the actual training loop's teacher isolation and update ordering."""
import json
from types import SimpleNamespace
import torch
import train
from test_resume import make_config
from theseus.hf_model import AttentionAdapter


def test_one_teacher_forward_and_immediate_layer_updates(tmp_path, monkeypatch):
    cfg, _ = make_config(tmp_path, 2)
    cfg.update(grad_accum_steps=1, original_teacher_kl=False)
    topo = SimpleNamespace(rank=0, leader=0, world=1, device=torch.device('cpu'),
        assert_same=lambda *args: None, barrier=lambda: None, close=lambda: None,
        broadcast_object=lambda value: value, student_sum=lambda value: value)
    monkeypatch.setattr(train, 'Topology', lambda *args: topo)
    events = []
    teachers = []
    load = train.load_model
    def load_model(*args):
        model = load(*args)
        teachers.append((model, {k: v.clone() for k, v in model.state_dict().items()}))
        return model
    monkeypatch.setattr(train, 'load_model', load_model)
    forward = train.Runner.forward
    def teacher_forward(runner, ids, *args, **kwargs):
        assert not any(isinstance(module, AttentionAdapter) for module in runner.model.modules())
        assert not any(p.requires_grad for p in runner.model.parameters())
        events.append('teacher')
        return forward(runner, ids, *args, **kwargs)
    monkeypatch.setattr(train.Runner, 'forward', teacher_forward)
    prepare = train.prepare_student
    def prepare_student(model, layer, *args, **kwargs):
        adapter, ddp, optimizer, scheduler = prepare(model, layer, *args, **kwargs)
        def pre(module, inputs):
            events.append(('forward', layer))
            assert not inputs[0].requires_grad and inputs[0].grad_fn is None
        adapter.core.register_forward_pre_hook(pre)
        next(adapter.core.parameters()).register_hook(lambda grad: events.append(('backward', layer)))
        # The scheduler has already wrapped step; retain its bookkeeping wrapper.
        step = optimizer.step
        def update(*args, **kwargs):
            events.append(('step', layer))
            return step(*args, **kwargs)
        update._wrapped_by_lr_sched = True
        optimizer.step = update
        return adapter, ddp, optimizer, scheduler
    monkeypatch.setattr(train, 'prepare_student', prepare_student)
    train.run(cfg, stop_after=1)
    assert events == ['teacher', ('forward', 1), ('backward', 1), ('step', 1),
                      ('forward', 3), ('backward', 3), ('step', 3)]
    model, before = teachers[0]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())
    logs = [json.loads(line) for line in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    metrics = next(item for item in logs if item['event'] == 'train')
    assert metrics['sum_loss'] == sum(item['nmse'] for item in metrics['layers'].values())
    assert metrics['mean_cos'] == sum(item['cosine'] for item in metrics['layers'].values()) / 2
    assert not (tmp_path/'full/converted').exists()
