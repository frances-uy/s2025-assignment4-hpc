import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from time import time
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(self.embedding(x))

def average_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

def run(rank, world_size, model_size, trace_dir):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    sizes = {
        "small": (768, 10000),
        "medium": (1024, 10000),
        "large": (1280, 10000),
        "xl": (2048, 10000),
    }
    d_model, vocab_size = sizes[model_size]
    model = SimpleTransformer(d_model, vocab_size).to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 16
    seq_len = 128
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    trace_path = os.path.join(trace_dir, f"{model_size}_rank{rank}")
    os.makedirs(trace_path, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=True
    ) as prof:
        model.train()
        for _ in range(5):  # 5 steps total
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))
            loss.backward()
            average_gradients(model)
            optimizer.step()
            prof.step()

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="trace_naive")
    parser.add_argument("--model_size", type=str, default="xl")
    args = parser.parse_args()

    world_size = 2
    mp.spawn(run, args=(world_size, args.model_size, args.trace_dir), nprocs=world_size)

if __name__ == "__main__":
    main()
