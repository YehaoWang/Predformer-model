import torch
import torch.nn as nn


class GatedAttentionUnit(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 r_forward: int = 4,
                 attn_bias: bool = False,
                 ):
        super(GatedAttentionUnit, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, attn_bias)
        self.sglu = SwiGLU(d_model, dropout, r_forward, attn_bias)
        self.drop_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_cGlu = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_cGLU = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        x_ = x
        if return_attn:
            x, attn_map = self.attn(x, return_attn)
        else:
            x = self.attn(x)
        x = self.norm_attn(self.drop_attn(x) + x_)  # Add & Drop & Norm
        x = self.norm_cGLU(self.drop_cGlu(self.sglu(x)) + x)
        return x if not return_attn else (x, attn_map)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob > 0.0 and self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            x = x.div(keep_prob) * random_tensor
        return x

class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 r_forward: int = 4,
                 bias: bool = False,
                 ):
        super(SwiGLU, self).__init__()
        d_hidden = d_model * r_forward

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.gate_silu = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self.silu = nn.SiLU()
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.init_norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self, x):
        B, L, D = x.shape
        x = self.init_norm(x)
        return self.drop2(self.down_proj(self.drop1(self.gate_proj(x) * self.silu(self.gate_silu(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 attn_bias: bool = False,
                 ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.factor = self.d_heads ** -0.5
        self.eps = 1e-6

        # Norms & Drops:
        self.init_norm = nn.LayerNorm(normalized_shape=d_model, eps=self.eps)
        self.attn_drop = nn.Dropout(dropout)

        # Projections:
        self.q_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=attn_bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        x = self.init_norm(x)
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_heads).transpose(1, 2)  # [b,h,l,d]
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_heads).transpose(1, 2)  # [b,h,l,d]
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_heads).transpose(1, 2)  # [b,h,l,d]
        attn_map = (torch.einsum('bhnd,bhgd->bhng', q, k) * self.factor).softmax(dim=-1)
        x = torch.einsum('bhng,bhgd->bhnd', self.attn_drop(attn_map), v)
        x = self.o_proj(x.transpose(1, 2).reshape(B, L, D))
        return x if not return_attn else (x, attn_map)