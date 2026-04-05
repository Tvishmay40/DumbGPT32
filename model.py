# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import config

class MultiHeadAttention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.c_attn = nn.Linear(config.EMBED_SIZE, 3 * config.EMBED_SIZE, bias=False)
        self.c_proj = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)).view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(config.EMBED_SIZE, dim=2)
        k = k.view(B, T, config.NUM_HEADS, C // config.NUM_HEADS).transpose(1, 2)
        q = q.view(B, T, config.NUM_HEADS, C // config.NUM_HEADS).transpose(1, 2)
        v = v.view(B, T, config.NUM_HEADS, C // config.NUM_HEADS).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.EMBED_SIZE)
        self.attn = MultiHeadAttention(vocab_size)
        self.ln_2 = nn.LayerNorm(config.EMBED_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(config.EMBED_SIZE, 4 * config.EMBED_SIZE),
            nn.GELU(),
            nn.Linear(4 * config.EMBED_SIZE, config.EMBED_SIZE),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MicroLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.EMBED_SIZE)
        self.position_embedding = nn.Embedding(config.BLOCK_SIZE, config.EMBED_SIZE)
        self.blocks = nn.Sequential(*[Block(vocab_size) for _ in range(config.NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.lm_head = nn.Linear(config.EMBED_SIZE, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=config.DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx