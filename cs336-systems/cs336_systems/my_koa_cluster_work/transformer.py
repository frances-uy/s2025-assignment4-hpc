import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
        return self.scale * x / (norm + self.eps)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is (B, T, num_heads, head_dim)

        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # (B, num_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, num_heads, vocab_size, context_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.token_embedding(x) + self.position_embedding[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
