import os
from pathlib import Path
import subprocess
import sys


def test_all_rank_weighted_gradient_matches_serial(tmp_path):
    worker = Path(__file__).with_name("weighted_ddp_worker.py")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", str(worker)],
                            env=os.environ | {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_ddp_rebuild_and_promotion_cpu():
    script = Path(__file__).resolve().parents[1] / "scripts/check_nccl.py"
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                             str(script), "--backend", "gloo", "--steps", "3"],
                            env=os.environ | {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('"status": "passed"') == 2


def test_reject_mismatched_run_configuration():
    script = Path(__file__).with_name("mismatched_config_worker.py")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script)],
                            env=os.environ | {"OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

