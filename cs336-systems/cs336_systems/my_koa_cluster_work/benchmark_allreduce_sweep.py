import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import csv
from itertools import product
from multiprocessing import Manager

def allreduce_worker(rank, world_size, backend, device_str, tensor_size_mb, return_dict):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        device = torch.device(device_str)
        size = (tensor_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        tensor = torch.ones(size, dtype=torch.float32, device=device)

        # Warm-up
        for _ in range(3):
            dist.all_reduce(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / 10
        if rank == 0:
            return_dict["result"] = avg_time

        dist.destroy_process_group()
    except Exception as e:
        if rank == 0:
            return_dict["result"] = f"ERROR: {str(e)}"

def run_one_config(world_size, backend, device, tensor_size_mb):
    manager = Manager()
    return_dict = manager.dict()
    mp.spawn(
        allreduce_worker,
        args=(world_size, backend, device, tensor_size_mb, return_dict),
        nprocs=world_size,
        join=True
    )
    return return_dict.get("result", "ERROR: No result")

def main():
    print("Starting benchmark sweep...")
    configs = list(product(
        ["gloo", "nccl"],            # backends
        ["cpu", "cuda"],             # devices
        [2, 4, 6],                   # world sizes
        [0.5, 1, 10, 50, 100, 500, 1024]  # tensor sizes in MB
    ))

    output_file = "allreduce_results.csv"
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Backend", "Device", "World Size", "Tensor Size (MB)", "Avg Time (s)"])

        for backend, device, world_size, size in configs:
            # Skip invalid combos
            if device == "cuda" and not torch.cuda.is_available():
                continue
            if backend == "nccl" and device != "cuda":
                continue

            print(f"▶Running: {backend.upper()} | {device.upper()} | Procs: {world_size} | Size: {size}MB")

            try:
                avg_time = run_one_config(world_size, backend, device, int(size))
                writer.writerow([backend, device, world_size, size, avg_time])
                print(f"Result: {avg_time}")
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([backend, device, world_size, size, "FAIL"])

    print(f"Done! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
