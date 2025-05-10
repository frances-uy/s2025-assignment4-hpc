import torch
import torch.distributed as dist

def train_step_naive_ddp(model, optimizer, x, y, rank, measure_comm=False):
    model.train()
    optimizer.zero_grad()
    loss = model(x).mean()
    loss.backward()

    comm_start = comm_end = None
    if measure_comm:
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        comm_start.record()

    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= dist.get_world_size()

    if measure_comm:
        comm_end.record()
        torch.cuda.synchronize()
        return comm_start.elapsed_time(comm_end) / 1000  # seconds

    optimizer.step()
    return 0.0
