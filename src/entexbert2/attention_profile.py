#!/usr/bin/env python
"""
Attention-profile extraction for entexBERT-2 -- the head-INDEPENDENT interpretability track.

Attention lives in the backbone, so this runs on ANY checkpoint's backbone OR the raw pretrained
DNABERT-2 (--raw_model), with no ASB/binding head required. That makes the attention row the
cleanest Stage-1-vs-no-Stage-1-vs-raw control: does binding SFT sharpen attention onto motif
positions relative to the untrained backbone?

We request the ALiBi-free attention the modified bert_layers exposes: with output_attentions=True
the model returns `reported_probs` = softmax(content_scores + padding_bias) with the ALiBi
positional bias REMOVED (the model's own forward is unchanged; this is a reporting-only tensor),
shape (B, H, S, S). The model's internal computation still uses content+ALiBi+padding.

Aggregation to a per-bp track (so it overlays on the same axis as ISM importance):
  received attention per token = mean over heads and over query positions of att[:, :, :, key]
     (how much every position, on average, attends TO this token) -- the standard "attention
     received" salience. --agg cls uses only the [CLS] query row instead.
  BPE -> bp: each token owns a (start,end) bp span (tokenizer offset mapping); its scalar
     attention is spread uniformly across the bp it covers, so specials ([CLS]/[SEP], span (0,0))
     contribute nothing to the bp track. Output track length = L (window length in bp).

Output .npz mirrors the ISM density arrays: importance (N, L) = the per-bp attention track, plus
onehot (N, L, 4) so the same plotter (plot_ism_aggregate_density.py) can overlay motif density.

Runs on the cluster (GPU + checkpoint) OR locally on CPU (output_attentions forces the non-flash
PyTorch path, so no Triton/GPU is strictly required -- just slower).
"""
import argparse, json, os, numpy as np, pandas as pd

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}


def load_backbone_and_tokenizer(checkpoint_dir=None, raw_model=None, device="cpu"):
    """Return (backbone, tokenizer). Either a trained checkpoint's backbone (weights loaded via
    model_io) or the raw pretrained model directly (--raw_model, no head)."""
    import torch, transformers
    if raw_model:
        tok = transformers.AutoTokenizer.from_pretrained(raw_model, trust_remote_code=True)
        cfg = transformers.AutoConfig.from_pretrained(raw_model, trust_remote_code=True)
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tok.pad_token_id            # config may not carry it
        backbone = transformers.AutoModel.from_pretrained(
            raw_model, trust_remote_code=True, config=cfg)
        backbone.to(device).eval()
        return backbone, tok
    # trained checkpoint: reuse model_io so pooling/weights match training exactly
    from entexbert2.model_io import load_model_and_tokenizer
    model, tok, _rc = load_model_and_tokenizer(checkpoint_dir, device=device)
    return model.backbone, tok


def attention_track(backbone, tok, seqs, layer=-1, agg="received", device="cpu"):
    """Per-window per-bp attention track. Returns (N, L) array aligned to bp."""
    import torch
    N, L = len(seqs), len(seqs[0])
    track = np.zeros((N, L), dtype=np.float32)
    for w, s in enumerate(seqs):
        enc = tok(s, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = backbone(**enc, output_attentions=True, return_dict=True)
        att = out.attentions[layer][0]                     # (H, S, S) for this single window
        if agg == "cls":
            recv = att[:, 0, :].mean(0)                    # CLS query row, mean over heads -> (S,)
        else:
            recv = att.mean(0).mean(0)                     # mean over heads & queries -> (S,) received
        recv = recv.detach().cpu().numpy()
        for tokval, (st, en) in zip(recv.tolist(), offs):
            if en > st:                                    # skip specials (span (0,0))
                track[w, st:en] += tokval / (en - st)      # spread token attn across its bp span
    return track


def onehot_of(seqs):
    N, L = len(seqs), len(seqs[0])
    oh = np.zeros((N, L, 4), dtype=np.float32)
    for w, s in enumerate(seqs):
        for i, ch in enumerate(s):
            if ch in B2I:
                oh[w, i, B2I[ch]] = 1.0
    return oh


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_dir", help="any trained checkpoint (uses its backbone)")
    src.add_argument("--raw_model", help="raw pretrained DNABERT-2 dir/hub id (no head)")
    ap.add_argument("--windows_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--rank_col", default="binding_label_raw",
                    help="rank windows desc by this col, take top n_windows (skip if absent)")
    ap.add_argument("--feature_col", default="feature_type",
                    help="keep rows whose feature_col contains --feature_keep (skip if absent)")
    ap.add_argument("--feature_keep", default="peak")
    ap.add_argument("--n_windows", type=int, default=30)
    ap.add_argument("--layer", type=int, default=-1, help="transformer layer to read (default last)")
    ap.add_argument("--agg", default="received", choices=["received", "cls"],
                    help="'received' = mean attention TO each token over heads&queries; "
                         "'cls' = the [CLS] query row")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.windows_csv)
    if a.feature_col in df.columns:
        df = df[df[a.feature_col].astype(str).str.contains(a.feature_keep, case=False, na=False)]
    if a.rank_col in df.columns:
        df = df.sort_values(a.rank_col, ascending=False)
    df = df.head(a.n_windows).reset_index(drop=True)
    seqs = df[a.seq_col].astype(str).str.upper().str.strip().tolist()
    if not seqs:
        raise SystemExit("no windows selected -- check --seq_col/--feature_col/--rank_col")
    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise SystemExit(f"windows must be equal length; {len(bad)} differ (L0={L})")

    src_desc = a.raw_model if a.raw_model else a.checkpoint_dir
    print(f"[ATT] source={'raw ' if a.raw_model else ''}{src_desc} | {len(seqs)} windows x L={L} "
          f"| layer={a.layer} agg={a.agg}")
    backbone, tok = load_backbone_and_tokenizer(a.checkpoint_dir, a.raw_model, device=a.device)
    track = attention_track(backbone, tok, seqs, layer=a.layer, agg=a.agg, device=a.device)

    # save under the ISM array names so plot_ism_aggregate_density.py works unchanged.
    np.savez_compressed(a.out, seqs=np.array(seqs), importance=track, onehot=onehot_of(seqs),
                        mode="attention", layer=a.layer, agg=a.agg)
    print(f"[ATT] saved -> {a.out} | track range [{track.min():.4f},{track.max():.4f}] "
          f"mean per-window peak {track.max(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()