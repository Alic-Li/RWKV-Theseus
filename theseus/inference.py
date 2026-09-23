"""Load exported deltas; Runner.forward returns hidden states and preserves state."""
import json
from pathlib import Path
from safetensors.torch import load_file
from .checkpoint import model_fingerprint, split_delta
from .hf_model import load_model, install, Runner


def load_export(path, device="cuda", base_model=None):
    path = Path(path)
    meta = json.loads((path / "theseus.json").read_text())
    base = base_model or meta["base_model"]
    if model_fingerprint(base) != meta["base_fingerprint"]:
        raise ValueError("Export base fingerprint mismatch")
    model = load_model(base, device)
    delta = split_delta(load_file(str(path / "migrated.safetensors")))
    for layer in meta["layers"]:
        install(model, layer, meta, delta[layer])
    return Runner(model)
