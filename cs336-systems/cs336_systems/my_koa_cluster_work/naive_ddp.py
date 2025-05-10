import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.functional import mse_loss
from functools import partial

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.net(x)

def average_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

def train_ddp(rank, world_size, x, y):
    setup(rank, world_size)

    model = ToyModel().cuda(rank)
    ref_state = torch.load("init_weights.pt", map_location=f"cuda:{rank}")
    model.load_state_dict(ref_state)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x_shard = x[rank::world_size].cuda(rank)
    y_shard = y[rank::world_size].cuda(rank)

    model.train()
    optimizer.zero_grad()
    output = model(x_shard)
    loss = mse_loss(output, y_shard)
    loss.backward()

    average_gradients(model)
    optimizer.step()

    result = model.state_dict() if rank == 0 else None
    cleanup()
    return result

def wrapper(rank, world_size, x, y, return_dict):
    weights = train_ddp(rank, world_size, x, y)
    if weights is not None:
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        return_dict["result"] = cpu_weights

def train_single(x, y):
    model = ToyModel().cuda()
    ref_state = torch.load("init_weights.pt", map_location="cuda")
    model.load_state_dict(ref_state)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    world_size = torch.cuda.device_count()

    grads = []
    for rank in range(world_size):
        x_shard = x[rank::world_size].cuda()
        y_shard = y[rank::world_size].cuda()
        optimizer.zero_grad()
        output = model(x_shard)
        loss = mse_loss(output, y_shard)
        loss.backward()
        grad_snapshot = {
            name: param.grad.clone().detach()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        grads.append(grad_snapshot)

    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = sum(g[name] for g in grads) / world_size

    optimizer.step()
    return model.state_dict()

def main():
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(42)  # Set seed
    world_size = torch.cuda.device_count()

    total_samples = 16 * world_size
    x = torch.randn(total_samples, 8)
    y = torch.randn(total_samples, 4)

    # ✅ Always use the same weights across runs
    init_model = ToyModel()
    init_model.load_state_dict(init_model.state_dict())  # get fresh init
    torch.save(init_model.state_dict(), "init_weights.pt")  # Save for both paths

    single_result = train_single(x, y)

    manager = mp.Manager()
    return_dict = manager.dict()

    wrapped_fn = partial(wrapper, world_size=world_size, x=x, y=y, return_dict=return_dict)
    mp.spawn(wrapped_fn, nprocs=world_size)

    ddp_result = return_dict["result"]

    max_diff = 0.0
    for k in single_result:
        diff = (single_result[k].cpu() - ddp_result[k]).abs().max().item()
        print(f"Param: {k}, max diff: {diff:.6f}")
        max_diff = max(max_diff, diff)

    print(f"\n DDP match check: max param difference = {max_diff:.6e}")
    assert max_diff < 1e-5, "DDP and single-process weights differ too much!"

if __name__ == "__main__":
    main()
