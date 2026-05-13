"""Tiny Graph Attention Network in raw PyTorch.

Avoids the PyTorch Geometric dependency (which often needs CUDA-matched
wheels and compiles native extensions on Windows).  Two GAT layers with
masked dot-product attention are sufficient for the smoke-test scope laid
out in the plan.

Inputs are dense and padded: a fixed-size node feature tensor
``(B, N, F_in)`` together with an adjacency tensor ``(B, N, N)`` and a node
mask ``(B, N)``.  Self-loops are added implicitly via the identity diagonal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Multi-head graph attention layer with additive scoring."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = out_dim // heads
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Two attention vectors per head (GATv2-style: tanh applied later)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,           # (B, N, in_dim)
        adj: torch.Tensor,         # (B, N, N) with 1 marking edges (incoming for dst)
        node_mask: torch.Tensor,   # (B, N) bool/int
    ) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.linear(x).view(B, N, self.heads, self.head_dim)        # (B,N,H,Dh)

        # Per-head additive attention scores e_ij = a_src·h_i + a_dst·h_j
        e_src = (h * self.attn_src).sum(-1)  # (B,N,H)
        e_dst = (h * self.attn_dst).sum(-1)  # (B,N,H)
        # e[b, i, j, h] = e_src[b, i, h] + e_dst[b, j, h]
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)                      # (B,N,N,H)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Adjacency mask + self-loops + padded-node mask
        eye = torch.eye(N, device=x.device, dtype=adj.dtype).unsqueeze(0)
        a = ((adj + eye) > 0).float().unsqueeze(-1)                       # (B,N,N,1)
        pad_j = node_mask.float().unsqueeze(1).unsqueeze(-1)              # (B,1,N,1)
        valid = a * pad_j
        e = e.masked_fill(valid == 0, float("-inf"))

        attn = F.softmax(e, dim=2)                                        # (B,N,N,H)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        # out[b, i, h, d] = sum_j attn[b,i,j,h] * h[b,j,h,d]
        out = torch.einsum("bijh,bjhd->bihd", attn, h)
        out = out.reshape(B, N, self.heads * self.head_dim)
        return out


class GATEncoder(nn.Module):
    """Two-layer GAT producing a single graph embedding via masked mean pool."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads=heads)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = F.elu(self.gat1(x, adj, node_mask))
        h = F.elu(self.gat2(h, adj, node_mask))
        # masked mean pool over valid nodes
        m = node_mask.float().unsqueeze(-1)                               # (B,N,1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return pooled
