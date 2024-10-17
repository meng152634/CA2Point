# -*- coding: utf-8 -*-
import torch


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class MultiHeadLinearAttention(torch.nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 num_heads=8,
                 eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads
        self.eps = eps

        self.query = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.key = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.value = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)

        self.feature_map = elu_feature_map

        self.merge = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)

    def forward(self, x):
        """
        Multi-Head Linear Attention
        Args:
            x: [B, HW, C]
        Returns:

        """
        bs = x.size(0)

        query = self.query(x).view(bs, -1, self.num_heads, self.head_dim)  # [2*B, HW, num_head, head_dim]
        key = self.key(x).view(bs, -1, self.num_heads, self.head_dim)  # [2*B, HW, num_head, head_dim]
        value = self.value(x).view(bs, -1, self.num_heads, self.head_dim)  # [2*B, HW, num_head, head_dim]

        query = self.feature_map(query)  # [2*B, HW, num_head, head_dim]
        key = self.feature_map(key)      # [2*B, HW, num_head, head_dim]

        v_length = value.size(1)
        value = value / v_length
        kv = torch.einsum("nlhd,nlhv->nhdv", key, value)  # [2*B, num_head, head_dim, head_dim]
        z = 1 / (torch.einsum("nlhd,nhd->nlh", query, key.sum(dim=1)) + self.eps)  # [2*B, HW, num_head]
        queried_value = torch.einsum("nlhd,nhdv,nlh->nlhv", query, kv, z) * v_length  # [2*B, HW, num_head, head_dim]

        queried_value = queried_value.contiguous()
        message = queried_value.view(bs, -1, self.hidden_dim)  # [2*B, HW, C]
        message = self.merge(message)

        return message


class LinearTransformerBlock(torch.nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 num_heads=8):
        super().__init__()

        self.linear_attn = MultiHeadLinearAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 2, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim, bias=False)
        )

        self.attn_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        # Linear Attention
        message = self.linear_attn(x)  # [2*B, HW, C]
        message = self.attn_norm(message)

        # Feed-Forward Network
        message = self.mlp(torch.cat([x, message], dim=2))  # [2*B, HW, C]
        message = self.ffn_norm(message)

        return residual + message


class LinearTransformerEncoder(torch.nn.Module):
    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 attn_hidden_dim=128,
                 mlp_dim=256,
                 mlp_ratio=2.0):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            block = LinearTransformerBlock(hidden_dim=attn_hidden_dim, num_heads=num_heads)
            self.blocks.append(block)

        self._init_weight()

    def _init_weight(self):
        for p in self.blocks.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
