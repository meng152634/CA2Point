# -*- coding: utf-8 -*-
import math
import torch


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 num_heads=8,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads

        self.query = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.proj = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn_dropout = torch.nn.Dropout(p=0.)
        self.proj_dropout = torch.nn.Dropout(p=0.)
        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute([0, 2, 1, 3])

    def forward(self, x):
        # B, N, hidden_dim = x.shape
        query = self.transpose_for_scores(self.query(x))
        key = self.transpose_for_scores(self.key(x))
        value = self.transpose_for_scores(self.value(x))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        # [B, num_heads, N, hidden_dim // num_heads] --> [B, N, num_heads, hidden_dim // num_heads]
        context = context.permute([0, 2, 1, 3]).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_dim,)
        context = context.view(*new_context_shape)
        attn_out = self.proj(context)
        attn_out = self.proj_dropout(attn_out)
        return attn_out


class Mlp(torch.nn.Module):
    def __init__(self,
                 in_features=128,
                 mid_features=2048,
                 out_features=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=mid_features)
        self.fc2 = torch.nn.Linear(in_features=mid_features, out_features=out_features)

        self.act_fn = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.normal_(self.fc1.bias, std=1e-6)
        torch.nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 num_heads=8,
                 mlp_dim=2048,
                 mlp_ratio=4.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attn_norm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.ffn_norm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-6)

        self.attn = MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.mlp_dim = self.hidden_dim * mlp_ratio
        self.ffn = Mlp(in_features=hidden_dim, mid_features=mlp_dim, out_features=hidden_dim)

    def forward(self, x):
        h = x
        x = self.attn(self.attn_norm(x))
        x = x + h

        h = x
        x = self.ffn(self.ffn_norm(x))
        x = x + h
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 num_layers=12,
                 num_heads=8,
                 attn_hidden_dim=128,
                 mlp_dim=2048,
                 mlp_ratio=4.0):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(hidden_dim=attn_hidden_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                                     mlp_ratio=mlp_ratio)
            self.blocks.append(block)

        self.encoder_norm = torch.nn.LayerNorm(attn_hidden_dim, eps=1e-6)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        encoded = self.encoder_norm(x)
        return encoded


class LGCA(torch.nn.Module):
    def __init__(self,
                 input_size=None,
                 in_channels=128,
                 is_adapool=False,
                 num_layers=12,
                 num_heads=8,
                 attn_hidden_dim=128,
                 mlp_dim=2048):
        """

        Args:
            in_channels:
            is_avgpool:
        """
        super().__init__()
        self.is_adapool = is_adapool
        self.input_size = input_size
        if self.is_adapool:
            self.adapool = torch.nn.AdaptiveAvgPool2d(output_size=(32, 32))
            self.input_size = (32, 32)

        self.embedding = torch.nn.Conv2d(in_channels=in_channels, out_channels=attn_hidden_dim,
                                         kernel_size=1, stride=1, padding=0)

        n_tokens = self.input_size[0] * self.input_size[1]
        self.position_embeddings = torch.nn.Parameter(torch.zeros(1, n_tokens, attn_hidden_dim))
        self.dropout = torch.nn.Dropout(0.1)

        # Transformer Block
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, num_heads=num_heads,
                                                      attn_hidden_dim=attn_hidden_dim, mlp_dim=mlp_dim,
                                                      mlp_ratio=4.0)

        self.weight = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.active = torch.nn.Softplus()
        # self.active = torch.nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out', nonlinearity='conv2d')
        torch.nn.init.kaiming_normal_(self.weight.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, debug=False):
        B, C, H, W = x.shape
        l_context = x

        if self.is_adapool:
            x = self.adapool(x)

        x = self.embedding(x)
        x = x.flatten(2)  # [B, C, H, W] --> [B, C, HW]
        x = x.transpose(2, 1)  # [B, C, HW] --> [B, HW, C]

        # position embedding
        x = x + self.position_embeddings
        embedding = self.dropout(x)

        encoded = self.transformer_encoder(embedding)

        x = encoded.permute([0, 2, 1])  # [B, HW, C] --> [B, C, HW]
        x = x.contiguous().view(B, C, self.input_size[0], self.input_size[1])
        g_context = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear')

        # generate global weight
        weight = self.weight(l_context)
        weight = self.active(weight)

        out = l_context + g_context * weight
        if debug:
            return out, weight
        else:
            return out
