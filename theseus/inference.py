"""Self-contained converted checkpoints loaded with this project's TimeMix runtime."""
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from .checkpoint import model_fingerprint, split_delta
from .data import digest_file
from .hf_model import load_model, install, Runner


def save_export(model, path, cfg, order, base_id, checkpoint=None):
    """Write complete sharded weights only after final composition validation."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite export {path}")
    temp = path.with_name('.' + path.name + '.tmp')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp, safe_serialization=True, max_shard_size="5GB")
    for source in Path(cfg["model"]).iterdir():
        if source.is_file() and (source.name.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens', 'chat_template'))):
            shutil.copyfile(source, temp / source.name)
    (temp / "theseus.json").write_text(json.dumps({"format": 2, "layers": order,
        "head_size": cfg["head_size"], "backend": cfg["backend"], "base_fingerprint": base_id,
        "loader": "theseus.inference.load_export", "state_layout": "B,H,value,key",
        "migration_fingerprint": digest_file(Path(checkpoint) / "migrated.safetensors") if checkpoint else None}, indent=2))
    os.replace(temp, path)


def load_export(path, device="cuda", base_model=None):
    path = Path(path)
    meta = json.loads((path / "theseus.json").read_text())
    if meta["format"] == 2:
        config = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16)
            for layer in meta["layers"]:
                install(model, layer, meta)
        index = path / "model.safetensors.index.json"
        shards = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
                  else ["model.safetensors"])
        expected = set(model.state_dict())
        seen = set()
        for shard in shards:
            state = load_file(str(path / shard), device=str(device))
            if seen.intersection(state) or set(state) - expected:
                raise ValueError("Invalid converted checkpoint keys")
            model.load_state_dict(state, strict=False, assign=True)
            seen.update(state)
        if config.tie_word_embeddings and "lm_head.weight" not in seen:
            model.lm_head.weight = model.model.embed_tokens.weight
            seen.add("lm_head.weight")
        if expected != seen:
            raise ValueError(f"Missing converted weights: {expected - seen}")
        # Rotary frequencies are nonpersistent buffers: recreate on the target device.
        model.model.rotary_emb = type(model.model.rotary_emb)(config).to(device)
        model.requires_grad_(False).eval()
        return Runner(model)
    base = base_model or meta["base_model"]
    if model_fingerprint(base) != meta["base_fingerprint"]:
        raise ValueError("Export base fingerprint mismatch")
    model = load_model(base, device)
    delta = split_delta(load_file(str(path / "migrated.safetensors")))
    for layer in meta["layers"]:
        install(model, layer, meta, delta[layer])
    return Runner(model)
