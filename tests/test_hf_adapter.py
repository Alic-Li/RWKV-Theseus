import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from theseus.hf_model import Runner, Capture, install, isolated_state, load_model
from theseus.checkpoint import cpu_tree


def tiny_config(depth=4):
    return Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=depth,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        vocab_size=64, layer_types=["linear_attention", "full_attention"] * (depth // 2),
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                         "mrope_section": [1, 1, 0]})


def tiny_model(depth=4):
    cfg = tiny_config(depth)
    cfg._attn_implementation = "sdpa"
    return Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)


def test_hf_cached_chunk_and_all_replaced():
    model = tiny_model()
    runner = Runner(model)
    ids = torch.randint(0, 64, (1, 11))
    full = runner.forward(ids)
    runner.reset()
    parts = [runner.forward(ids[:, :4]), runner.forward(ids[:, 4:7]), runner.forward(ids[:, 7:])]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=.03, rtol=.03)
    for layer in [3, 1]:
        install(model, layer, {"head_size": 8, "backend": "reference"})
    runner.reset()
    full = runner.forward(ids)
    runner.reset()
    pieces = [runner.forward(ids[:, :4]), runner.forward(ids[:, 4:])]
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=.03, rtol=.03)
    assert runner.cache.cursor == 11


def test_truncated_teacher_matches_full_attention_boundary_across_chunks():
    full_model = tiny_model()
    short_model = tiny_model()
    short_model.load_state_dict(full_model.state_dict())
    full_runner, short_runner = Runner(full_model), Runner(short_model)
    full_capture = Capture(full_model.model.layers[1].self_attn)
    short_capture = Capture(short_model.model.layers[1].self_attn)
    try:
        for ids in (torch.randint(0, 64, (1, 6)), torch.randint(0, 64, (1, 3))):
            full_runner.forward(ids)
            short_runner.forward(ids, through_layer=1)
            torch.testing.assert_close(short_capture.x, full_capture.x, atol=.001, rtol=.001)
            torch.testing.assert_close(short_capture.y, full_capture.y, atol=.001, rtol=.001)
        assert full_runner.cache.cursor == short_runner.cache.cursor == 9
        assert short_model.config.num_hidden_layers == 4
    finally:
        full_capture.close()
        short_capture.close()


def test_adapter_only_current_layer_has_grad_and_runner_restore():
    teacher = tiny_model()
    student = tiny_model()
    student.load_state_dict(teacher.state_dict())
    tr, sr = Runner(teacher), Runner(student)
    adapter = install(student, 3, {"head_size": 8, "backend": "reference"}, trainable=True)
    adapter.__dict__["train_call"] = adapter.core
    capture = Capture(teacher.model.layers[3].self_attn)
    ids = torch.randint(0, 64, (1, 6))
    tr.forward(ids)
    adapter.teacher_input = capture.x
    adapter.debug_input = True
    sr.forward(ids)
    (adapter.prediction.float() - capture.y.float()).square().mean().backward()
    grads = [n for n, p in student.named_parameters() if p.grad is not None]
    assert grads and all(n.startswith("model.layers.3.self_attn.core.") for n in grads)
    adapter.teacher_input = adapter.prediction = None
    state = cpu_tree(sr.state_dict())
    more = torch.randint(0, 64, (1, 3))
    expected = sr.forward(more)
    sr.load_state_dict(state)
    actual = sr.forward(more)
    torch.testing.assert_close(actual, expected, atol=.001, rtol=.001)
    capture.close()


def test_isolated_validation_does_not_change_training():
    model = tiny_model()
    install(model, 3, {"head_size": 8, "backend": "reference"})
    runner = Runner(model)
    ids = torch.randint(0, 64, (1, 5))
    runner.forward(ids)
    state = cpu_tree(runner.state_dict())
    with isolated_state(runner):
        runner.forward(ids)
        runner.forward(ids)
    actual = runner.forward(ids)
    runner.load_state_dict(state)
    expected = runner.forward(ids)
    torch.testing.assert_close(actual, expected)


def test_native_loader(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    loaded = load_model(tmp_path, "cpu")
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b)


def test_local_branch_matches_separate_teacher_and_only_trains_timemix():
    base = tiny_model(depth=6)
    local = tiny_model(depth=6)
    local.load_state_dict(base.state_dict())
    cfg = {"head_size": 8, "backend": "reference"}
    from theseus.hf_model import migration_order
    assert migration_order(local, expected=3) == [1, 3, 5]
    prefix = install(base, 1, cfg)
    install(local, 1, cfg, weights=prefix.core.state_dict())
    adapter = install(local, 3, cfg, trainable=True, retain_teacher=True)
    a, b = Runner(base), Runner(local)
    ca, cb = Capture(base.model.layers[3].self_attn), Capture(adapter.teacher)
    suffix_calls = []
    hook = local.model.layers[4].register_forward_hook(lambda *args: suffix_calls.append(1))
    ffn_hook = local.model.layers[3].mlp.register_forward_hook(lambda *args: suffix_calls.append(2))
    try:
        for _ in range(3):
            ids = torch.randint(0, 64, (1, 3))
            a.forward(ids)
            adapter.teacher_mode = True
            b.forward(ids, through_layer=3)
            adapter.teacher_mode = False
            torch.testing.assert_close(ca.x, cb.x, atol=0, rtol=0)
            torch.testing.assert_close(ca.y, cb.y, atol=0, rtol=0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                pred, state = adapter.core(cb.x, adapter.state)
            from theseus.timemix import detach_state
            adapter.state = detach_state(state)
            (pred.float() - cb.y.float()).square().mean().backward()
        assert not suffix_calls
        assert b.cache.cursor == 9
        grads = [name for name, p in local.named_parameters() if p.grad is not None]
        assert grads and all(name.startswith("model.layers.3.self_attn.core.") for name in grads)
        assert adapter.state is not None
        assert local.model.layers[1].self_attn.state is not None
        assert all(not t.requires_grad for t in local.model.layers[1].self_attn.state.values())
    finally:
        ca.close()
        cb.close()
        hook.remove()
        ffn_hook.remove()


@__import__("pytest").mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gdn_inference_dispatch_matches_chunk_across_resets(monkeypatch):
    import transformers
    if transformers.__version__ != "5.17.0":
        __import__("pytest").skip("HF 5.17 kernel dispatch")
    from theseus import hf_model
    if not hf_model._has_fla:
        __import__("pytest").skip("FLA required for optimized dispatch")
    recurrent_lengths = []
    recurrent = hf_model._qwen.torch_recurrent_gated_delta_rule

    def record_recurrent(query, *args, **kwargs):
        recurrent_lengths.append(query.shape[1])
        return recurrent(query, *args, **kwargs)

    monkeypatch.setattr(hf_model._qwen, "torch_recurrent_gated_delta_rule", record_recurrent)
    base, optimized = tiny_model().cuda(), tiny_model().cuda()
    optimized.load_state_dict(base.state_dict())
    a, b = Runner(base), Runner(optimized)
    for reset, length in [(True, 512), (False, 257), (False, 511),
                          (False, 1), (True, 383), (False, 17), (True, 63)]:
        if reset:
            a.reset()
            b.reset()
        ids = torch.randint(0, 64, (1, length), device="cuda")
        monkeypatch.setenv("THESEUS_GDN_KERNEL", "chunk")
        expected = a.forward(ids)
        monkeypatch.setenv("THESEUS_GDN_KERNEL", "auto")
        actual = b.forward(ids)
        torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
        # State errors can appear only on the next chunk; also check states now.
        for old, new in zip(a.cache.layers, b.cache.layers):
            if hasattr(old, "recurrent_states"):
                for key, value in old.recurrent_states.items():
                    torch.testing.assert_close(new.recurrent_states[key], value, atol=.003, rtol=.03)
    assert {257, 383, 511, 512}.issubset(recurrent_lengths)
