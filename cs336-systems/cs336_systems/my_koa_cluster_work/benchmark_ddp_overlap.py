import os
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from adapters import get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward
from transformer import Transformer  # Ensure this file exists

MODEL_CONFIGS = {
    "small":  {"d_model": 128, "d_ff": 512,  "num_layers": 2,  "num_heads": 4, "vocab_size": 1000, "context_length": 64},
    "medium": {"d_model": 256, "d_ff": 1024, "num_layers": 4,  "num_heads": 4, "vocab_size": 1000, "context_length": 64},
    "large":  {"d_model": 512, "d_ff": 2048, "num_layers": 8,  "num_heads": 8, "vocab_size": 1000, "context_length": 64},
    "xl":     {"d_model": 1024,"d_ff": 4096, "num_layers": 16, "num_heads": 8, "vocab_size": 1000, "context_length": 64},
}

def ddp_worker(rank, world_size, model_size, trace_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    config = MODEL_CONFIGS[model_size]
    model = Transformer(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        vocab_size=config["vocab_size"],
        context_length=config["context_length"]
    ).to(device)

    ddp_model = get_ddp_individual_parameters(model)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    batch_size = 8
    x = torch.randint(0, config["vocab_size"], (batch_size, config["context_length"])).to(device)
    y = torch.randint(0, config["vocab_size"], (batch_size, config["context_length"])).to(device)

    trace_path = os.path.join(trace_dir, f"{model_size}_rank{rank}")
    os.makedirs(trace_path, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(5):  # Profile 5 iterations
            optimizer.zero_grad()
            logits = ddp_model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))
            loss.backward()
            ddp_individual_parameters_on_after_backward(ddp_model, optimizer)
            optimizer.step()
            prof.step()

    dist.destroy_process_group()

def benchmark(model_size="xl", world_size=2, trace_dir="trace_overlap"):
    mp.spawn(ddp_worker, args=(world_size, model_size, trace_dir), nprocs=world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="trace_overlap")
    parser.add_argument("--model_size", type=str, default="xl")
    args = parser.parse_args()

    benchmark(model_size=args.model_size, trace_dir=args.trace_dir)
