#!/usr/bin/env python3
"""Symmetric late cross-allele attention block + distance-logistic readout (cross-encoder ASB head).

Sits on top of the shared DNABERT-2 bi-encoder: given per-position hidden states for the two alleles
(H_ref, H_alt, produced by the SAME backbone), one ALiBi-free cross-attention block lets each allele
attend to the other, then the existing distance-logistic readout (ell = a*||z1-z2|| + b) is applied
to the interaction-aware pooled reps.

SYMMETRY (the load-bearing property): the block uses SHARED q/k/v/o weights in both directions, so
swapping (ref, alt) exactly exchanges the two outputs and leaves ell invariant -- matching the ASB
label's direction-agnosticism, exactly as the bi-encoder ||z1-z2|| does. Enforced by a unit test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAlleleInteraction(nn.Module):
    def __init__(self, dim=768, d_x=256, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_x % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_x // n_heads
        self.q = nn.Linear(dim, d_x)
        self.k = nn.Linear(dim, d_x)
        self.v = nn.Linear(dim, d_x)
        self.o = nn.Linear(d_x, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _attend(self, Hq, Hk, key_mask):
        # standard scaled-dot-product multihead cross-attention, NO ALiBi / no positional bias
        B, Sq, _ = Hq.shape
        Sk = Hk.shape[1]
        q = self.q(Hq).view(B, Sq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(Hk).view(B, Sk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(Hk).view(B, Sk, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B,H,Sq,Sk)
        if key_mask is not None:                                              # mask padded keys
            scores = scores.masked_fill(~key_mask.bool()[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, Sq, -1)         # (B,Sq,d_x)
        return self.o(ctx)

    def forward(self, H_ref, H_alt, mask_ref, mask_alt):
        # shared weights both directions -> symmetric under (ref,alt) swap
        H_ref2 = self.ln(H_ref + self.drop(self._attend(H_ref, H_alt, mask_alt)))
        H_alt2 = self.ln(H_alt + self.drop(self._attend(H_alt, H_ref, mask_ref)))
        return H_ref2, H_alt2


def _masked_mean(H, mask):
    m = mask.float().unsqueeze(-1)
    return (H * m).sum(1) / m.sum(1).clamp_min(1.0)


def pool(H, mask, mode, var_tok_idx=None, width=2):
    """center_mean-analog uses masked mean; variant_focus pools a +/-width window around var_tok_idx."""
    if mode == "variant_focus":
        if var_tok_idx is None:
            raise ValueError("variant_focus readout needs var_tok_idx")
        B, S, _ = H.shape
        idx = torch.arange(S, device=H.device)[None, :]
        centre = var_tok_idx if torch.is_tensor(var_tok_idx) else torch.full((B,), var_tok_idx, device=H.device)
        sel = (idx >= (centre[:, None] - width)) & (idx <= (centre[:, None] + width)) & mask.bool()
        return _masked_mean(H, sel)
    return _masked_mean(H, mask)          # "mean" / center-analog


class CrossEncoderReadout(nn.Module):
    """Interaction block + symmetric distance-logistic readout -> ASB logit ell."""
    def __init__(self, dim=768, proj_dim=128, d_x=256, n_heads=4, dropout=0.1, readout="mean"):
        super().__init__()
        self.inter = CrossAlleleInteraction(dim, d_x, n_heads, dropout)
        self.proj = nn.Linear(dim, proj_dim)
        self.dist_a = nn.Parameter(torch.tensor(0.5413))     # softplus(a_raw) -> a>0, matches model.py init
        self.dist_b = nn.Parameter(torch.tensor(0.0))
        self.readout = readout

    def forward(self, H_ref, H_alt, mask_ref, mask_alt, var_tok_idx=None):
        H_ref2, H_alt2 = self.inter(H_ref, H_alt, mask_ref, mask_alt)
        z1 = self.proj(pool(H_ref2, mask_ref, self.readout, var_tok_idx))
        z2 = self.proj(pool(H_alt2, mask_alt, self.readout, var_tok_idx))
        s = torch.norm(z1 - z2, dim=-1)
        a = F.softplus(self.dist_a)
        return a * s + self.dist_b                            # ell (ASB logit)
