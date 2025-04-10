from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention"]

class Attention(nn.Module):
    """Attention with optional cross-attention and support for cu_seqlens"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        # We will compute Q, K, V differently for cross-attention
        # so let's just keep a single linear projection for Q, K, V each.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn

    def q_proj(self, x):
        return F.linear(x, self.qkv.weight[:self.dim], self.qkv.bias[:self.dim] if self.qkv.bias is not None else None)

    def k_proj(self, x):
        return F.linear(x, self.qkv.weight[self.dim:2*self.dim], self.qkv.bias[self.dim:2*self.dim] if self.qkv.bias is not None else None)

    def v_proj(self, x):
        return F.linear(x, self.qkv.weight[2*self.dim:], self.qkv.bias[2*self.dim:] if self.qkv.bias is not None else None)
    
    def self_attention(self, x, x_attn_mask=None, rpb=None):
        # Reshape Q, K, V
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D), (B, H, N, D), (B, H, N, D)

        if self.use_flash_attn:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=x_attn_mask, dropout_p=self.attn_drop.p
            )

            # Reshape back
            attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim)
            x = attn_output.transpose(1, 2).reshape(B, N, C)
            _attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale 
            if rpb is not None:
                attn = attn + rpb
            if x_attn_mask is not None:
                attn = attn + (~x_attn_mask).to(attn.dtype).masked_fill(~x_attn_mask, -1e9)
            attn = attn.softmax(dim=-1)
            _attn = attn
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, _attn
    
    def cross_attention(self, q, kv):
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        B,Nq,C = q.shape
        B,Nv,C = v.shape
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nv, self.num_heads, self.head_dim).transpose(1, 2)


        if self.use_flash_attn: # crashes on huge batch sizes
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p
            )
        else: # no flash attn :-(
            qk = torch.einsum("b h n d, b h m d -> b h n m", q, k)
            qk = qk * self.scale
            qk = qk.softmax(dim=-1)
            qk = self.attn_drop(qk)
            x = torch.einsum("b h n m, b h m d -> b h n d", qk, v)


        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        _attn = None # no intermediates srry
        return x, _attn


    def forward(self, q, kv=None):
        """
        Args:
            q (torch.Tensor): Queries of shape (B, N, C) or (sum(seqlens), C) if using cu_seqlens.
            qkv_attn_mask (torch.Tensor, optional): Attention mask.
            kv (torch.Tensor, optional): If provided, should be for K, V. Otherwise, K, V come from x (self-attention).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors after attention.
        """
        if kv is None:  # Regular self-attention
            x, _attn = self.self_attention(q)
        else:  # Cross attention
            x, _attn = self.cross_attention(q, kv)

        return x, _attn