"""Explicitly skipped on a one-GPU host; never disguise Gloo as a NCCL test."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="NCCL audit requires >=2 physical GPUs")
def test_nccl_all_rank_ddp_and_stage_transition():
    root = Path(__file__).resolve().parents[1]
    ranks = 4 if torch.cuda.device_count() >= 4 else 2
    env = os.environ | {"OMP_NUM_THREADS": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}",
                             str(root / "scripts/check_nccl.py"), "--steps", "3"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
