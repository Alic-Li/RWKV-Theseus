import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytest

@pytest.fixture(autouse=True)
def small_threads():
    torch.set_num_threads(1)
    torch.manual_seed(123)
