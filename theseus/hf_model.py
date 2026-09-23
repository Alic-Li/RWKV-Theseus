"""Only the replacement boundary and cache cursor are customized; HF runs Qwen."""
from contextlib import contextmanager
from pathlib import Path
from functools import wraps
import inspect
import importlib.util
import os
import torch
from torch import nn
import transformers
from transformers import AutoConfig, Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
from .timemix import TimeMix, detach_state


def _cpu_safe_kernel(function):
    # HF 5.17 prefers installed FLA even on CPU. Keep its own undecorated
    # PyTorch implementation for CPU tests; CUDA retains HF's kernel dispatch.
    reference = inspect.unwrap(function)

    @wraps(function)
    def dispatch(*args, **kwargs):
        tensor = args[0] if args else next(v for v in kwargs.values() if torch.is_tensor(v))
        return (reference if tensor.device.type == "cpu" else function)(*args, **kwargs)
    return dispatch


if transformers.__version__ == "5.17.0":
    from transformers.models.qwen3_5 import modeling_qwen3_5 as _qwen
    for _name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                  "causal_conv1d_fn", "causal_conv1d_update"):
        setattr(_qwen, _name, _cpu_safe_kernel(getattr(_qwen, _name)))


    _chunk_gdn = _qwen.torch_chunk_gated_delta_rule
    _has_fla = importlib.util.find_spec("fla") is not None

    @wraps(_chunk_gdn)
    def _inference_gdn(query, key, value, **kwargs):
        # Frozen B=1 short chunks are faster with HF's fused recurrent dispatch.
        # Its token loop handles varying lengths without per-length chunk JIT.
        # Include 512-token training chunks and their 257..511-token tails:
        # otherwise a new tail length can stall one rank in chunk-kernel JIT.
        # Keep chunk/autograd kernels for longer/batched or differentiable work.
        if (_has_fla and os.environ.get("THESEUS_GDN_KERNEL", "auto") != "chunk"
                and query.is_cuda and query.dtype == torch.bfloat16
                and query.shape[0] == 1 and query.shape[1] <= 512
                and not torch.is_grad_enabled()):
            return _qwen.torch_recurrent_gated_delta_rule(query, key, value, **kwargs)
        return _chunk_gdn(query, key, value, **kwargs)

    _qwen.torch_chunk_gated_delta_rule = _inference_gdn


def _state_to_device(value, device):
    # HF 5.8 stores tensors directly; 5.17 stores per-state dictionaries.
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, torch.device):
        return device
    if isinstance(value, dict):
        return {k: _state_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_to_device(v, device) for v in value)
    return value



class HybridCache(DynamicCache):
    def __init__(self, config):
        super().__init__(config=config)
        self.cursor = 0

    def get_seq_length(self, layer_idx=0):
        return self.cursor

    def get_mask_sizes(self, query_length, layer_idx):
        return self.cursor + query_length, 0


class AttentionAdapter(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.state = None
        self.teacher_input = None
        self.prediction = None
        self.debug_input = False
        self.teacher = None
        self.teacher_mode = False
        # DDP must not be registered as a second child pointing at the same core.
        self.__dict__["train_call"] = None

    def forward(self, hidden_states, **kwargs):
        if self.teacher_mode:
            return self.teacher(hidden_states=hidden_states, **kwargs)
        if self.teacher_input is not None:
            if self.debug_input:
                torch.testing.assert_close(hidden_states, self.teacher_input, atol=.04, rtol=.02)
            with torch.enable_grad():
                y, state = self.train_call(self.teacher_input.detach(), self.state)
            self.prediction = y
        else:
            y, state = self.core(hidden_states, self.state)
        self.state = detach_state(state)
        return y.detach(), None


def load_model(path, device, dtype=torch.bfloat16):
    if transformers.__version__ not in {"5.8.0", "5.17.0"}:
        raise RuntimeError("This cache adapter supports transformers 5.8.0 or 5.17.0")
    path = str(Path(path).resolve())
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    text = config.get_text_config()
    if text.model_type != "qwen3_5_text":
        raise ValueError(f"Expected qwen3_5_text, got {text.model_type}")
    text._attn_implementation = "sdpa"
    # HF's native loader does the safetensor streaming and key conversion.
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        path, config=text, dtype=dtype, local_files_only=True, output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    missing = info.get("missing_keys", [])
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not k.startswith(("model.visual.", "mtp."))]
    if missing or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"Text checkpoint mismatch: {info}")
    model.requires_grad_(False).eval().to(device)
    return model


def migration_order(model, expected=16):
    order = [i for i, kind in enumerate(model.config.layer_types) if kind == "full_attention"]
    if len(order) != expected:
        raise ValueError(f"Expected {expected} full attention blocks, found {order}")
    return order


def install(model, layer, options, weights=None, trainable=False, retain_teacher=False):
    device = next(model.parameters()).device
    with torch.device("meta" if weights is not None else device):
        core = TimeMix(model.config.hidden_size, layer, model.config.num_hidden_layers,
                       head_size=options["head_size"], backend=options["backend"])
    if weights is not None:
        core.load_state_dict(weights, strict=True, assign=True)
    core.to(device=device, dtype=torch.float32 if trainable else torch.bfloat16)
    core.requires_grad_(trainable)
    adapter = AttentionAdapter(core)
    if retain_teacher:
        adapter.teacher = model.model.layers[layer].self_attn
        adapter.teacher.requires_grad_(False).eval()
    model.model.layers[layer].self_attn = adapter
    return adapter


class Capture:
    def __init__(self, attention):
        self.x = self.y = None
        self.pre = attention.register_forward_pre_hook(self._pre, with_kwargs=True)
        self.post = attention.register_forward_hook(self._post)

    def _pre(self, module, args, kwargs):
        self.x = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach()

    def _post(self, module, args, output):
        self.y = output[0].detach()

    def close(self):
        self.pre.remove()
        self.post.remove()
        self.x = self.y = None


class Runner:
    def __init__(self, model):
        self.model = model
        self.cache = HybridCache(model.config)

    def reset(self):
        self.cache = HybridCache(self.model.config)
        for block in self.model.model.layers:
            if isinstance(getattr(block, "self_attn", None), AttentionAdapter):
                block.self_attn.state = None

    @torch.no_grad()
    def forward(self, ids, through_layer=None):
        """HF forward, optionally stop exactly at the target attention output.

        A scoped hook unwinds HF forward before target residual/MLP/final norm.
        All preceding native cache updates are retained. No permanent patch to HF.
        The truncated return value is the attention output, not final hidden state.
        """
        positions = torch.arange(self.cache.cursor, self.cache.cursor + ids.shape[1], device=ids.device)[None]
        hook = None
        class BoundaryReached(Exception):
            pass
        boundary = []
        if through_layer is not None:
            if not 0 <= through_layer < self.model.config.num_hidden_layers:
                raise ValueError(f"Invalid decoder layer {through_layer}")
            def stop(module, args, output):
                boundary.append(output[0])
                raise BoundaryReached()
            hook = self.model.model.layers[through_layer].self_attn.register_forward_hook(stop)
        try:
            with torch.autocast(ids.device.type, dtype=torch.bfloat16):
                try:
                    output = self.model.model(input_ids=ids, position_ids=positions,
                                              past_key_values=self.cache, use_cache=True)
                    hidden = output.last_hidden_state
                except BoundaryReached:
                    hidden = boundary.pop()
        finally:
            if hook is not None:
                hook.remove()
        self.cache.cursor += ids.shape[1]
        return hidden

    def state_dict(self):
        return {"cursor": self.cache.cursor,
                "hf_layers": [dict(vars(layer)) for layer in self.cache.layers],
                "timemix": {i: block.self_attn.state for i, block in enumerate(self.model.model.layers)
                            if isinstance(getattr(block, "self_attn", None), AttentionAdapter)}}

    def load_state_dict(self, state):
        self.reset()
        self.cache.cursor = state["cursor"]
        device = next(self.model.parameters()).device
        for layer, attrs in zip(self.cache.layers, state["hf_layers"], strict=True):
            for k, v in attrs.items():
                setattr(layer, k, _state_to_device(v, device))
        for i, s in state["timemix"].items():
            self.model.model.layers[int(i)].self_attn.state = (
                None if s is None else {k: v.to(device) for k, v in s.items()})


@contextmanager
def isolated_state(runner):
    # Detach the live objects, not a shallow copy of cache tensors updated in place.
    cache = runner.cache
    states = {i: b.self_attn.state for i, b in enumerate(runner.model.model.layers)
              if isinstance(getattr(b, "self_attn", None), AttentionAdapter)}
    runner.reset()
    try:
        yield
    finally:
        runner.cache = cache
        for i, state in states.items():
            runner.model.model.layers[i].self_attn.state = state
