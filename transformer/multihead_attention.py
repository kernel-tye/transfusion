# -*- coding: utf-8 -*-
# date: 2018-11-30 16:35

from .functional import clones, attention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)




# --------- MGSA: MHA + (Geodesic Self-Attention) + Gate 并联（Encoder专用） ---------


def _to_bias(mask, B, H, Tq, Tk, device, dtype):
    """
    接受 [B,T,T] / [B,1,T,T] 的bool mask，转加性bias [B,H,T,T]，True=可见, False=-inf
    """
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if mask.dim() == 4:
        # 可能是 [B,1,T,T]
        mask = mask[:, 0]
    # 现在是 [B,T,T]
    bias = torch.zeros(B, 1, Tq, Tk, device=device, dtype=dtype)
    bias = bias.masked_fill(~mask.unsqueeze(1), float("-inf"))  # [B,1,T,T]
    if H is not None:
        bias = bias.expand(B, H, Tq, Tk)
    return bias

class _GeodesicSelfAttention(nn.Module):
    """
    简化的“测地”注意力：每个头一组对角马氏尺度，score = -0.5 * (q-k)ᵀ M (q-k) / sqrt(d)
    仅用于 Encoder 自注意力（q==k==v场景）
    """
    def __init__(self, d_model, h, dropout=0.1, scale=0.5):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.dk = d_model // h
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.log_diag = nn.Parameter(torch.zeros(h, self.dk))  # 每头一个对角尺度
        self.drop = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, x, mask=None):  # x: [B,T,D]
        B, T, D = x.shape
        H, Dh = self.h, self.dk
        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)

        diag = F.softplus(self.log_diag) + 1e-6         # [H,Dh] 正定
        qM   = q * diag.view(1, H, 1, Dh)               # [B,H,T,Dh]
        q2M  = (q * qM).sum(-1)                         # [B,H,T]
        k2M  = (k * (k * diag.view(1, H, 1, Dh))).sum(-1)  # [B,H,T]
        cross = torch.einsum('bhtd,bhTd->bhtT', qM, k)  # [B,H,T,T]
        d = q2M.unsqueeze(-1) + k2M.unsqueeze(-2) - 2.0 * cross  # [B,H,T,T]

        scores = -0.5 * d / math.sqrt(Dh)
        bias = _to_bias(mask, B, H, T, T, x.device, x.dtype)
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(torch.isneginf(scores).all(dim=-1, keepdim=True), 0.0)
        p = torch.softmax(scores * self.scale, dim=-1)
        p = self.drop(p)
        out = torch.einsum('bhtT,bhTd->bhtd', p, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)

class MGSAParallelEncoder(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, gsa_scale=0.25):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.dk = d_model // h

        self.mha = MultiHeadAttention(h, d_model, dropout=dropout)
        self.gsa = _GeodesicSelfAttention(d_model, h, dropout=dropout, scale=gsa_scale)
        self.gsa_mix = nn.Parameter(torch.tensor(-0.5))  # sigmoid(-0.5) ≈ 0.38，前期就有少量 GSA

        # NEW: 分支层归一化，防止某一支幅值偏大
        self.ln_mha = nn.LayerNorm(d_model)
        self.ln_gsa = nn.LayerNorm(d_model)

        # NEW: GSA 渐进阀门（初始≈0，sigmoid/tanh都可）
        self.gsa_mix = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5; 如想初始更小用 -2.0
        with torch.no_grad():
            self.gsa_mix.copy_(torch.tensor(-2.0))      # 初始≈0.12，更温和

        # Gate：先让 MHA 主导
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        # 原来更保守的偏置 2.5 会把 alpha 顶很高；改小一点
        with torch.no_grad():
            self.gate[-1].bias.fill_(0.5)  # 初期 alpha ≈ sigmoid(0.5) ≈ 0.62
        # 可选：给 alpha 一个上限系数，避免 saturate=1.0
        self.alpha_cap = 0.9

        self._last_alpha_mean = None
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 自注意力场景；否则回退到标准 MHA
        if (query is not key) or (key is not value):
            return self.mha(query, key, value, mask)

        x = query
        # 把 [B,1,T,T] 兼容成 [B,T,T] 给原 MHA
        mask_for_mha = None
        if mask is not None:
            mask_for_mha = mask[:, 0] if (mask.dim() == 4 and mask.size(1) == 1) else mask

        mha_out = self.mha(x, x, x, mask_for_mha)  # [B,T,D]
        gsa_out = self.gsa(x, mask)                # [B,T,D]

        # NEW: 对齐标度
        mha_out = self.ln_mha(mha_out)
        gsa_out = self.ln_gsa(gsa_out)

        # NEW: 渐进混合系数（0~1）
        mix = torch.sigmoid(self.gsa_mix)  # 初始≈0.12，后续可自己学大
        gsa_out = mix * gsa_out

        alpha = torch.sigmoid(self.gate(x))  # [B,T,1]
        alpha = self.alpha_cap * alpha  # 把可达上限压到 ~0.9
        self._last_alpha_mean = alpha.mean().detach()

        z = alpha * mha_out + (1.0 - alpha) * gsa_out
        return self.drop(z)