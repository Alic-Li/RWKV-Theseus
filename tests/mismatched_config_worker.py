import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theseus.distributed import Topology

if __name__ == "__main__":
    topo = Topology("gloo")
    try:
        topo.assert_same({"grad_accum_steps": topo.rank + 1}, "test config")
    except ValueError as exc:
        assert "Ranks disagree" in str(exc)
    else:
        raise AssertionError("Inconsistent collective schedule was accepted")
    topo.close()
