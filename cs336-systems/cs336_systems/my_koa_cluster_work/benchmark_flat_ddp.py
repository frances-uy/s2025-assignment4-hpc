import torch
from transformer import Transformer
from ddp_flat_trainer import launch_flat_ddp

def generate_fake_data(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y

MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}

def make_model(size_key):
    config = MODEL_SIZES[size_key]
    return lambda: Transformer(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        vocab_size=10_000,
        context_length=128,
    )

if __name__ == "__main__":
    device = "cuda"
    x, y = generate_fake_data(batch_size=32, seq_len=128, vocab_size=10_000, device=device)

    print("=== Benchmarking FLAT DDP (SMALL model) ===")
    launch_flat_ddp(world_size=2, model_fn=make_model("small"), x=x, y=y)
