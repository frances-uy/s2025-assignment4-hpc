import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import time
import os

def run(rank, world_size, model_fn, x, y):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = model_fn().to(rank)
    ddp_model = model
    optimizer = optim.Adam(ddp_model.parameters())

    x = x.to(rank)
    y = y.to(rank)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16)

    # Forward + backward
    ddp_model.train()
    step_time = []
    comm_time = []

    for _ in range(3):  # Warm-up + 2 measure steps
        torch.cuda.synchronize()
        start_step = time.time()

        for xb, yb in dataloader:
            xb, yb = xb.to(rank), yb.to(rank)
            optimizer.zero_grad()
            logits = ddp_model(xb)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()

            # --- Flatten and all-reduce all gradients ---
            grads = [p.grad.data for p in ddp_model.parameters() if p.requires_grad]
            flat_grads = _flatten_dense_tensors(grads)

            torch.cuda.synchronize()
            start_comm = time.time()
            dist.all_reduce(flat_grads)
            flat_grads /= world_size
            torch.cuda.synchronize()
            end_comm = time.time()

            comm_time.append(end_comm - start_comm)

            unflattened = _unflatten_dense_tensors(flat_grads, grads)
            for p, g in zip(ddp_model.parameters(), unflattened):
                p.grad.data.copy_(g)

            optimizer.step()

        torch.cuda.synchronize()
        end_step = time.time()
        step_time.append(end_step - start_step)

    if rank == 0:
        avg_step = sum(step_time[1:]) / 2
        avg_comm = sum(comm_time[1:]) / 2
        print(f"[FLAT DDP] Avg step time: {avg_step:.4f}s | Comm time: {avg_comm:.4f}s")

    dist.destroy_process_group()

def launch_flat_ddp(world_size, model_fn, x, y):
    mp.spawn(run, args=(world_size, model_fn, x, y), nprocs=world_size)
