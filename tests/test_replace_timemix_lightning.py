import json

import pytest
import torch
from safetensors.torch import save_file

from replace_timemix_lightning import BASE_FORMAT, OUTPUT_FORMAT, export_tensor, replace, sidecar
from theseus.timemix import TimeMix


@pytest.mark.parametrize("head_size", [64, 128])
def test_replace_layer_preserves_others_and_uses_later_snapshot(tmp_path, head_size):
    base = tmp_path / "base.pth"
    original = {"emb.weight": torch.arange(256, dtype=torch.bfloat16).reshape(2, 128),
                "blocks.0.att.A_log": torch.arange(2, dtype=torch.bfloat16),
                "blocks.3.att.q_norm.weight": torch.ones(8, dtype=torch.bfloat16),
                "blocks.3.att.key.weight": torch.ones(8, 128, dtype=torch.bfloat16)}
    torch.save(original, base)
    config = {"format": BASE_FORMAT, "weights_file": base.name,
              "geometry": {"hidden_size": 128, "num_hidden_layers": 4,
                           "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                           "linear_num_value_heads": 2, "linear_key_head_dim": 128, "linear_value_head_dim": 128}}
    manifest = {"format": BASE_FORMAT, "tensors": [
        {"target": key, "source": key} for key in original]}
    sidecar(base, "config").write_text(json.dumps(config))
    sidecar(base, "manifest").write_text(json.dumps(manifest))
    folder = tmp_path / "weights"
    folder.mkdir()
    core = TimeMix(128, 3, 4, head_size=head_size, backend="reference")
    state = {"3." + key: value.detach().float().contiguous() for key, value in core.state_dict().items()}
    save_file(state, folder / "stage_1.safetensors")
    update = {"3.x_r": torch.full_like(state["3.x_r"], 0.375)}
    save_file(update, folder / "stage_2.safetensors")

    out = tmp_path / "replaced.pth"
    replace(base, folder, out, [3], verify=True, quiet=True)
    result = torch.load(out, map_location="cpu", weights_only=True)
    assert result["blocks.0.att.A_log"].equal(original["blocks.0.att.A_log"])
    assert "blocks.3.att.q_norm.weight" not in result
    assert "blocks.3.att.key.weight" in result
    assert result["blocks.3.att.x_r"].equal(update["3.x_r"].to(torch.bfloat16).reshape(-1))
    assert len([key for key in result if key.startswith("blocks.3.att.")]) == len(state)
    exported = json.loads(sidecar(out, "config").read_text())
    assert exported["format"] == OUTPUT_FORMAT
    assert exported["requires_hybrid_backend"] is True
    assert exported["contract"]["timemix_weight_layout"] == "out_in_row_major_vectors_flat_v1"
    for key, value in state.items():
        if key != "3.x_r":
            assert result["blocks.3.att." + key[2:]].equal(export_tensor(key[2:], value.to(torch.bfloat16)))
    assert exported["geometry"]["layer_types"][3] == "rwkv7_timemix"
    assert exported["wkv_layers"]["3"]["head_size"] == head_size
    assert exported["contract"]["wkv_kernel"] == "shared_dplr_fp32_v1"
    with pytest.raises(ValueError, match="was trained with head_size"):
        replace(base, folder, out, [3], head_size=128 if head_size == 64 else 64, dry_run=True)
    entries = json.loads(sidecar(out, "manifest").read_text())["tensors"]
    assert next(entry for entry in entries if entry["target"] == "blocks.3.att.x_r")["file"] == "stage_2.safetensors"
    # Incremental replacement accepts a single file and keeps the same contract.
    second = tmp_path / "second.pth"
    replace(out, folder / "stage_1.safetensors", second, [3], head_size=head_size,
            dtype="source", verify=True, quiet=True)
    result2 = torch.load(second, weights_only=True)
    assert result2["blocks.3.att.w1"].dtype == torch.float32
    assert result2["blocks.3.att.w1"].equal(state["3.w1"].T)
    assert result2["emb.weight"].equal(original["emb.weight"])
    for layers in ([], [-1], [3, 3]):
        with pytest.raises(ValueError, match="distinct nonnegative"):
            replace(base, folder, second, layers, head_size=head_size, dry_run=True)



def test_replace_rejects_incomplete_layer(tmp_path):
    base = tmp_path / "base.pth"
    torch.save({"emb.weight": torch.zeros(2, 128), "blocks.3.att.key.weight": torch.zeros(8, 128)}, base)
    sidecar(base, "config").write_text(json.dumps({"format": BASE_FORMAT, "weights_file": base.name,
        "geometry": {"hidden_size": 128, "num_hidden_layers": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                           "linear_num_value_heads": 2, "linear_key_head_dim": 128, "linear_value_head_dim": 128}}))
    sidecar(base, "manifest").write_text(json.dumps({"format": BASE_FORMAT, "tensors": []}))
    folder = tmp_path / "weights"
    folder.mkdir()
    save_file({"3.x_r": torch.zeros(1, 1, 128)}, folder / "partial.safetensors")
    with pytest.raises(ValueError, match="incomplete"):
        replace(base, folder, tmp_path / "out.pth", [3], head_size=64, dry_run=True)


def test_export_layout_roundtrip():
    core = TimeMix(64, 0, 2, head_size=64, backend="reference")
    for name, value in core.state_dict().items():
        converted = export_tensor(name, value)
        if name in {"w1", "w2", "a1", "a2", "g1", "g2"}:
            assert converted.equal(value.T)
        elif value.ndim == 3:
            assert converted.shape == (64,)
        else:
            assert converted.equal(value)
        assert converted.is_contiguous()


def test_replace_infers_mixed_head_sizes(tmp_path):
    base = tmp_path / 'base.pth'
    torch.save({'emb.weight': torch.zeros(2,128),
                'blocks.0.att.q_norm.weight': torch.ones(64),
                'blocks.1.att.q_norm.weight': torch.ones(64)}, base)
    sidecar(base,'config').write_text(json.dumps({
        'format': BASE_FORMAT, 'weights_file':base.name,
        'geometry':{'hidden_size':128,'num_hidden_layers':2,
                    'layer_types':['full_attention','full_attention']}}))
    sidecar(base,'manifest').write_text(json.dumps({'format':BASE_FORMAT,'tensors':[]}))
    tensors={}
    for layer, size in enumerate((64,128)):
        core=TimeMix(128,layer,2,head_size=size,backend='reference')
        tensors.update({f'{layer}.{k}':v.detach().contiguous() for k,v in core.state_dict().items()})
    source=tmp_path/'migrated.safetensors';save_file(tensors,source)
    output=tmp_path/'out.pth'
    replace(base,source,output,[0,1],verify=True,quiet=True)
    exported=torch.load(output,weights_only=True)
    assert exported['blocks.0.att.r_k'].shape==(2,64)
    assert exported['blocks.1.att.r_k'].shape==(1,128)
    config=json.loads(sidecar(output,'config').read_text())
    assert config['wkv_layers']['0']['head_size']==64
    assert config['wkv_layers']['1']['head_size']==128
