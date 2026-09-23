import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from theseus.distributed import Topology
from theseus.losses import statistics, loss_sum
from theseus.timemix import TimeMix

if __name__ == "__main__":
    torch.set_num_threads(1)
    topo = Topology("gloo")
    torch.manual_seed(111)
    model = TimeMix(16, 1, 4, 4, "reference")
    torch.nn.init.normal_(model.output.weight, std=.01)
    serial = TimeMix(16, 1, 4, 4, "reference")
    serial.load_state_dict(model.state_dict())
    ddp = DDP(model, process_group=topo.student_group, broadcast_buffers=False)
    torch.manual_seed(222)
    data = [(torch.randn(1, length, 16), torch.randn(1, length, 16)) for length in range(3, 3 + topo.world)]
    x, target = data[topo.rank]
    y, _ = ddp(x)
    stats = statistics(y, target)
    loss_sum(stats, .05).backward()
    count = topo.student_sum(stats[2].detach().clone())
    for p in model.parameters():
        p.grad.mul_(topo.world / count)
    losses = []
    for x, target in data:
        y, _ = serial(x)
        losses.append(loss_sum(statistics(y, target), .05))
    (sum(losses) / sum(range(3, 3 + topo.world))).backward()
    for p, q in zip(model.parameters(), serial.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-4)
    topo.barrier()
    dist.destroy_process_group()
