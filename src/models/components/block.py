import torch
import torch.nn as nn
from timm.models.layers import DropPath

from .attention import Attention
from .mlp import Mlp


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flash_attn=True,
        use_kv=False,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.norm1_kv = norm_layer(dim) if use_kv else None
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
        )

        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        q,
        kv=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) where B is the batch size, N is the sequence length, and C is the feature dimension.
            y (torch.Tensor, optional): Optional second input tensor of shape (B, M, C) where M is the sequence length for the second input.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): Output tensor after processing x.
                - y (torch.Tensor): Output tensor after processing y (if y is provided, otherwise None).
                - attn (torch.Tensor): Attention weights.
        """

        if self.norm1_kv is not None:
            assert kv is not None, "kv must be provided if use_kv is True"

        # apply norm1 to both q and kv together!
        q_normed = self.norm1(q)
        kv_normed = self.norm1_kv(kv) if self.norm1_kv is not None else None

        _q, attn = self.attn(
            q=q_normed,
            kv=kv_normed,
        )
        q = q + self.drop_path(_q)

        ffn_q = self.mlp(self.norm2(q))

        q = q + self.drop_path(ffn_q)

        return q, attn
