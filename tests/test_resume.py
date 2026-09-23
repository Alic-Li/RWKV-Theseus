"""End-to-end native torchrun tests on CPU. No distributed framework mocks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file
from test_hf_adapter import tiny_model
from test_data_loss import write_dataset
from theseus.config import DEFAULTS
from theseus.checkpoint import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def launch(config, ranks=2, extra=()):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}",
               str(ROOT / "train.py"), "--config", str(config), *extra]
    result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def make_config(tmp_path, stages):
    tiny_model(stages * 2).save_pretrained(tmp_path / "base")
    cfg = DEFAULTS | {"training_mode": "sampled", "model": str(tmp_path / "base"),
        "train_manifest": str(write_dataset(tmp_path / "train")),
        "validation_manifest": str(write_dataset(tmp_path / "val")),
        "output": str(tmp_path / "full"), "expected_attention_layers": stages,
        "head_size": 8, "backend": "reference", "distributed_backend": "gloo",
        "chunk_tokens": 3, "context_tokens": 9, "stage_steps": 3,
        "warmup_steps": 2, "grad_accum_steps": 2, "validation_chunks": 2,
        "validation_interval": 0, "checkpoint_interval": 0, "log_interval": 1,
        "original_teacher_kl": True, "debug_input": True}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return cfg, path


def test_midstream_and_stage_commit_resume(tmp_path):
    cfg, path = make_config(tmp_path, 2)
    cfg.update(warmup_steps=1, constant_steps=0)
    path.write_text(json.dumps(cfg))
    launch(path)
    cfg["output"] = str(tmp_path / "resumed")
    path.write_text(json.dumps(cfg))
    launch(path, extra=("--stop-after", "1"))
    launch(path, extra=("--resume", str(tmp_path / "resumed"), "--stop-after", "2"))
    _, mid, _ = read_checkpoint(tmp_path / "resumed")
    assert mid["stage"] == 0 and mid["complete"]
    # Resume exactly between stage commit and promotion.
    launch(path, extra=("--resume", str(tmp_path / "resumed")))
    af, _, _ = read_checkpoint(tmp_path / "full")
    bf, _, _ = read_checkpoint(tmp_path / "resumed")
    a, b = load_file(str(af / "migrated.safetensors")), load_file(str(bf / "migrated.safetensors"))
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    for rank in range(2):
        sa = torch.load(af / "ranks" / f"rank_{rank:05d}.pt", weights_only=False)
        sb = torch.load(bf / "ranks" / f"rank_{rank:05d}.pt", weights_only=False)
        assert sa["reader"] == sb["reader"]
        assert sa["runner"]["cursor"] == sb["runner"]["cursor"]
    # Standalone validation can read the final checkpoint.
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             str(ROOT / "validate.py"), "--config", str(path), "--checkpoint", str(bf)],
                            env=os.environ | {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "original_teacher_kl" in result.stdout


def test_four_rank_all_sixteen_stages_and_export(tmp_path):
    cfg, path = make_config(tmp_path, 16)
    cfg["stage_steps"] = 1
    cfg["validation_chunks"] = 1
    cfg["original_teacher_kl"] = False
    cfg["debug_input"] = False  # Exercise the optimized teacher-forced hot path.
    path.write_text(json.dumps(cfg))
    launch(path, ranks=4)
    checkpoint, meta, delta = read_checkpoint(tmp_path / "full")
    assert meta["complete"] and meta["stage"] == 15 and len(delta) == 16
    assert meta["order"] == list(range(1, 32, 2))
    out = tmp_path / "export"
    subprocess.run([sys.executable, str(ROOT / "export.py"), "--checkpoint", str(checkpoint), "--output", str(out)],
                   check=True, capture_output=True, timeout=30)
    from theseus.inference import load_export
    runner = load_export(out, device="cpu")
    for _ in range(2):
        hidden = runner.forward(torch.tensor([[1, 2, 3]]))
        assert torch.isfinite(hidden).all()
    assert runner.cache.cursor == 6


def test_single_rank_local_training(tmp_path):
    cfg, path = make_config(tmp_path, 1)
    cfg["stage_steps"] = 2
    cfg["validation_interval"] = 1
    path.write_text(json.dumps(cfg))
    result = launch(path, ranks=1)
    assert "Stage 1/1" in result.stderr
    for metric in ("nmse=", "rrms=", "cosine="):
        assert metric in result.stderr
    _, meta, delta = read_checkpoint(tmp_path / "full")
    assert meta["world"] == 1 and meta["training_topology"] == "local_branch_v1"
    assert meta["complete"] and len(delta) == 1


def test_epoch_uneven_ranks_exact_coverage_and_resume(tmp_path):
    cfg, path = make_config(tmp_path, 1)
    cfg.update(training_mode='epoch', context_tokens=100, stage_steps=1,
               original_teacher_kl=False, validation_chunks=1)
    path.write_text(json.dumps(cfg))
    # Three documents on four ranks: one rank is empty for the entire epoch.
    launch(path, ranks=4)
    complete, meta, _ = read_checkpoint(tmp_path / 'full')
    assert meta['complete'] and meta['step'] == 3  # stage_steps=1 is ignored
    events = [json.loads(line) for line in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert sum(e['tokens'] for e in events if e['event']=='train') == 37
    consumed = 0
    for rank in range(4):
        state = torch.load(complete/'ranks'/f'rank_{rank:05d}.pt', weights_only=False)
        consumed += state['reader']['consumed_tokens']
    assert consumed == 37
    cfg['output'] = str(tmp_path/'resumed')
    path.write_text(json.dumps(cfg))
    launch(path, ranks=4, extra=('--stop-after','1'))
    launch(path, ranks=4, extra=('--resume',str(tmp_path/'resumed')))
    resumed, _, _ = read_checkpoint(tmp_path/'resumed')
    a = load_file(str(complete/'migrated.safetensors'))
    b = load_file(str(resumed/'migrated.safetensors'))
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-7, rtol=1e-6)
    for rank in range(4):
        x = torch.load(complete/"ranks"/f"rank_{rank:05d}.pt", weights_only=False)
        y = torch.load(resumed/"ranks"/f"rank_{rank:05d}.pt", weights_only=False)
        assert x["reader"] == y["reader"]


def test_batched_epoch_single_rank_training_and_resume(tmp_path):
    cfg, path = make_config(tmp_path, 1)
    cfg.update(training_mode="epoch", micro_batch_size=2, context_tokens=7,
               original_teacher_kl=False, validation_interval=0, validation_chunks=1)
    path.write_text(json.dumps(cfg))
    launch(path, ranks=1, extra=("--stop-after", "1"))
    launch(path, ranks=1, extra=("--resume", str(tmp_path / "full")))
    checkpoint, meta, _ = read_checkpoint(tmp_path / "full")
    assert meta["complete"]
    state = torch.load(checkpoint / "ranks" / "rank_00000.pt", weights_only=False)
    assert state["reader"]["consumed_tokens"] == 37
    cfg["micro_batch_size"] = 4
    path.write_text(json.dumps(cfg))
    launch(path, ranks=1, extra=("--resume", str(tmp_path / "full")))
