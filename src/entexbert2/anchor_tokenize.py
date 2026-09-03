#!/usr/bin/env python3
"""Anchored tokenization for cross-allele attention.

Build an ALIGNED, equal-length token grid for a ref/alt window pair that differ by a single base at
`var_pos`. Scheme (isolate-the-variant-base): tokenize the LEFT flank natively (BPE), the variant
base as its OWN single token, and the RIGHT flank natively, then concatenate. Because ref and alt are
identical except at `var_pos`, the flank token ids are identical and the sequences differ in EXACTLY
one token (the variant-base token). Guarantees, for any tokenizer with single-nucleotide tokens:
  - len(ids_ref) == len(ids_alt)                      (equal length -> positions align)
  - ids_ref and ids_alt differ at EXACTLY index var_tok_idx
  - var_pos is EXPLICIT (not assumed center) -> off-center / jitter windows are supported

OOD note: flanks are native BPE (in-distribution); only the two splice points (left|base, base|right)
and the lone base token are mildly OOD. This is the price of a guaranteed aligned grid; it is far less
OOD than fixed-k-mer tokenization.
"""
from typing import Dict, List


def anchor_tokenize(tokenizer, ref_seq: str, alt_seq: str, var_pos: int,
                    max_len: int = None, add_special: bool = True) -> Dict[str, List[int]]:
    if len(ref_seq) != len(alt_seq):
        raise ValueError(f"ref/alt length mismatch: {len(ref_seq)} vs {len(alt_seq)}")
    if not (0 <= var_pos < len(ref_seq)):
        raise ValueError(f"var_pos {var_pos} out of range [0,{len(ref_seq)})")
    if ref_seq[:var_pos] != alt_seq[:var_pos] or ref_seq[var_pos + 1:] != alt_seq[var_pos + 1:]:
        raise ValueError("ref and alt differ OUTSIDE var_pos (expected a single-base difference)")

    def enc(s: str) -> List[int]:
        if not s:
            return []
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    left_ids = enc(ref_seq[:var_pos])          # identical for ref & alt (same left flank)
    right_ids = enc(ref_seq[var_pos + 1:])     # identical for ref & alt (same right flank)
    ref_base_id = enc(ref_seq[var_pos])
    alt_base_id = enc(alt_seq[var_pos])
    if len(ref_base_id) != 1 or len(alt_base_id) != 1:
        raise ValueError(f"variant base did not map to a single token "
                         f"(ref {ref_seq[var_pos]!r}->{ref_base_id}, alt {alt_seq[var_pos]!r}->{alt_base_id}); "
                         f"tokenizer lacks single-nucleotide tokens")

    # symmetric flank trim so the variant token stays in-window when max_len is tight
    if max_len is not None:
        n_special = (2 if add_special else 0)
        budget = max_len - n_special - 1               # reserve 1 slot for the variant token
        if budget < 0:
            raise ValueError("max_len too small to hold even the variant token + specials")
        half = budget // 2
        keep_left = min(len(left_ids), half)
        keep_right = min(len(right_ids), budget - keep_left)
        keep_left = min(len(left_ids), budget - keep_right)   # reclaim if right flank was short
        left_ids = left_ids[len(left_ids) - keep_left:]       # keep flank NEAREST the variant
        right_ids = right_ids[:keep_right]

    cls = [tokenizer.cls_token_id] if (add_special and tokenizer.cls_token_id is not None) else []
    sep = [tokenizer.sep_token_id] if (add_special and tokenizer.sep_token_id is not None) else []
    var_tok_idx = len(cls) + len(left_ids)
    ids_ref = cls + left_ids + ref_base_id + right_ids + sep
    ids_alt = cls + left_ids + alt_base_id + right_ids + sep
    attn = [1] * len(ids_ref)

    if max_len is not None and len(ids_ref) < max_len:      # right-pad
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        n = max_len - len(ids_ref)
        ids_ref = ids_ref + [pad_id] * n
        ids_alt = ids_alt + [pad_id] * n
        attn = attn + [0] * n

    return {"input_ids_ref": ids_ref, "input_ids_alt": ids_alt,
            "attention_mask": attn, "var_tok_idx": var_tok_idx}
