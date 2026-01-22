from fsspec.transaction import Transaction
from torch._numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelArgs
from typing import Optional




class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.weight = self.embedding.weight
    
    def forward(self, x):
        out = self.embedding(x)
        return out


def precompute_rop_embeddings(head_dim: int, seq_len: int, base: int):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()/head_dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb

def rotate_halfs(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotatoryPositionalEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.seq_len = config.max_seq_len
        self.rope_base = config.rope_base
        self.head_dim = config.head_dim
        self.n_head = config.n_head
        emb = precompute_rop_embeddings(self.head_dim, self.seq_len, self.rope_base)
        self.register_buffer("cos", torch.cos(emb))
        self.register_buffer("sin", torch.sin(emb))
    
    def forward(self, x):
        B, N, T, H = x.shape
        cos_values = self.cos[:T, :] # [T, HEAD_DIM]
        sin_values = self.sin[:T, :] # [T, HEAD_DIM]
        cos_values = cos_values.view(1, 1, T, self.head_dim)
        sin_values = sin_values.view(1, 1, T, self.head_dim)
        rotate_x = rotate_halfs(x)
        x = (x*cos_values + rotate_x*sin_values)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.alpha = nn.Parameter(torch.ones((1, self.dim)))
        self.beta = nn.Parameter(torch.zeros((1, self.dim)))
        self.eps = config.norm_eps
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        main = (x-mean)/torch.sqrt(var+self.eps)
        out = main*self.alpha + self.beta
        return out

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_head
        self.n_heads = config.n_head
        self.seq_len = config.max_seq_len
        self.rope_positions = RotatoryPositionalEmbeddings(config)
        self.kv_groups = self.n_heads//self.n_kv_heads
        self.dp = nn.Dropout(config.dropout)

        self.wq = nn.Linear(in_features=self.dim, out_features=self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(in_features=self.dim, out_features=self.n_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(in_features=self.dim, out_features=self.n_kv_heads * self.head_dim, bias=True)

        self.wo = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.register_buffer('tril', torch.tril(torch.ones(self.seq_len, self.seq_len)).bool().unsqueeze(0).unsqueeze(1))
    

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x) # [B, T, n_head * head_dim]
        k = self.wk(x) # [B, T, n_kv_head * head_dim]
        v = self.wv(x) # [B, T, n_kv_head * head_dim]

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_head, T, head_dim]
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B, n_kv_head, T, head_dim]
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.kv_groups, dim=1) # [B, n_kv_head*kv_group, T, head_dim]

        q = self.rope_positions(q)
        k = self.rope_positions(k).repeat_interleave(self.kv_groups, dim=1)
        qk = q @ k.transpose(-2, -1) # [B, n_head, T, T]
        qk = qk * (self.head_dim**(-0.5))
        qk = qk.masked_fill(~self.tril[:, :, :T, :T], value=float('-inf'))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
                mask = mask[:, :, :, :T].to(qk.device)
            qk = qk.masked_fill(~mask, value=float('-inf'))
        qk = F.softmax(qk, dim=-1)
        attns = qk @ v
        attns = attns.transpose(1, 2).reshape(B, T, C)
        attns = self.dp(attns)
        attns = self.wo(attns)
        return attns


class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.ffn_dim_multiplier = config.ffn_dim_multiplier
        self.hidden_dim = self.ffn_dim_multiplier * self.dim
        self.multiple_of = config.multiple_of
        self.hidden_dim = self.multiple_of * ((self.hidden_dim + self.multiple_of-1)//self.multiple_of)
        self.w1 = nn.Linear(self.dim, self.hidden_dim)
        self.w2 = nn.Linear(self.dim, self.hidden_dim)
        self.w3 = nn.Linear(self.hidden_dim, self.dim)

        self.silu = nn.SiLU()

    def forward(self, x):
        out = self.w3(self.silu(self.w1(x)) * self.w2(x))
        return out


class ResidualNetworks(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dp = nn.Dropout(config.dropout)
    
    def forward(self, x, prev_layer_output):
        out = x + self.dp(prev_layer_output)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_block = GroupedQueryAttention(config)
        self.res_block = ResidualNetworks(config)
        self.ffn_block = FeedForwardNetwork(config)
        self.norm1 = LayerNormalization(config)
        self.norm2 = LayerNormalization(config)
    

    def forward(self, x, mask=None):
        x =  self.res_block(x, self.attn_block(self.norm1(x), mask))
        x = self.res_block(x, self.ffn_block(self.norm2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(self.n_layer)])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x



class LLamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.embeddings = InputEmbeddings(config)
        self.decoder = Decoder(config)
        self.norm1 = LayerNormalization(config)
        self.proj = nn.Linear(self.dim, self.vocab_size)

        self.proj.weight = self.embeddings.weight
    
    def forward(self, x, mask=None):
        # B, T
        x = self.embeddings(x)
        x = self.decoder(x, mask)
        x = self.norm1(x)
        x = self.proj(x)
        return x
    




if __name__ == '__main__':
    config = ModelArgs()
    x = torch.randint(0, config.vocab_size, (10, config.max_seq_len))
    print(x.shape)
    model = LLamaModel(config)
    out = model(x)
    import pdb; pdb.set_trace()