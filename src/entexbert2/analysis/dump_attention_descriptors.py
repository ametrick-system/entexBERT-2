#!/usr/bin/env python
"""
Dump final-layer CENTER->all attention descriptors for AS/non-AS reference windows, in the SAME
npz schema probe_stage1_embeddings.py reads -- so the attention PCA + coverage-shortcut check reuse
the exact same honest pipeline as the hidden-embedding probe.

Descriptor (#1, decision-relevant): for each ref window, take the FINAL-layer, ALiBi-removed
attention FROM the center (variant) token TO all positions (mean over heads), then spread each
token's scalar across its bp span (tokenizer offset mapping) -> a fixed-length (L=257) per-locus
"where does the decision token look" vector. The model pools the center tokens to decide, so this
is the attention most tied to the ASB call.

ALiBi-removed attention: with output_attentions=True the modified bert_layers returns
reported_probs = softmax(content + padding_bias) (positional ALiBi bias removed), shape (B,H,S,S).
output_attentions forces the non-flash path (slower) -> class-balanced --max_rows subsample.

Backbone: a trained checkpoint's backbone (--checkpoint_dir, head-agnostic) OR raw DNABERT-2
(--raw_model) as the no-fine-tuning control.

  python dump_attention_descriptors.py \
    --checkpoint_dir $ASB/experiments/ctcf/stage1_trunk/runs/reg \
    --windows_csv    $WORK/ctcf_asb_ism_windows_entex.csv \
    --out            $WORK/attn_center_ctcf_entex.npz --device cuda --max_rows 6000
"""
import argparse, numpy as np, pandas as pd, torch


def load_backbone(checkpoint_dir, raw_model, device):
    import transformers
    if raw_model:
        tok = transformers.AutoTokenizer.from_pretrained(raw_model, trust_remote_code=True)
        cfg = transformers.AutoConfig.from_pretrained(raw_model, trust_remote_code=True)
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tok.pad_token_id
        bb = transformers.AutoModel.from_pretrained(raw_model, config=cfg, trust_remote_code=True)
        return bb.to(device).eval(), tok
    from entexbert2.model_io import load_model_and_tokenizer
    model, tok, _ = load_model_and_tokenizer(checkpoint_dir, device=device)
    return model.backbone.to(device).eval(), tok


def center_attention_track(backbone, tok, seqs, layer=-1, center_bp=None, device="cpu"):
    """(N, L) center-token->all attention, ALiBi-removed, spread to bp. L = window length."""
    N = len(seqs); L = len(seqs[0])
    cbp = (L // 2) if center_bp is None else int(center_bp)
    track = np.zeros((N, L), dtype=np.float32)
    for w, s in enumerate(seqs):
        assert len(s) == L, f"window {w} length {len(s)} != {L} (windows must be equal length)"
        enc = tok(s, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = backbone(**enc, output_attentions=True, return_dict=True)
        att = out.attentions[layer][0]                       # (H, S, S), ALiBi-removed
        # center token = the non-special token whose bp span covers cbp
        ctok = None
        for ti, (st, en) in enumerate(offs):
            if en > st and st <= cbp < en:
                ctok = ti; break
        if ctok is None:
            ctok = len(offs) // 2
        recv = att[:, ctok, :].mean(0).detach().cpu().numpy()  # (S,) attn FROM center token
        for ti, (st, en) in enumerate(offs):
            if en > st:
                track[w, st:en] += recv[ti] / (en - st)        # spread across bp span
        if (w + 1) % 500 == 0:
            print(f"[attn] {w+1}/{N}", flush=True)
    return track


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_dir", help="trained checkpoint (uses its backbone; head-agnostic)")
    src.add_argument("--raw_model", help="raw pretrained DNABERT-2 dir/hub id (no-fine-tuning control)")
    ap.add_argument("--windows_csv", required=True, help="output of build_asb_ism_windows.py")
    ap.add_argument("--seq_col", default="sequence1")
    ap.add_argument("--layer", type=int, default=-1, help="transformer layer (default last)")
    ap.add_argument("--center_bp", type=int, default=None, help="0-based center bp (default L//2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_rows", type=int, default=6000, help="0=all; else class-balanced subsample (attn is slow)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.windows_csv)
    for c in (a.seq_col, "as_label"):
        assert c in df.columns, f"missing column {c}"
    labels = (df["as_label"].astype(str).str.upper() == "AS").astype(int).to_numpy()
    if a.max_rows and a.max_rows > 0 and len(df) > a.max_rows:
        rng = np.random.default_rng(a.seed)
        idx = np.arange(len(df)); pos = idx[labels == 1]; neg = idx[labels == 0]
        k = min(a.max_rows // 2, len(pos))
        keep = np.r_[rng.choice(pos, k, replace=False),
                     rng.choice(neg, min(a.max_rows - k, len(neg)), replace=False)]
        keep.sort(); df = df.iloc[keep].reset_index(drop=True); labels = labels[keep]
    seqs = df[a.seq_col].astype(str).tolist()
    print(f"[in] {len(seqs)} ref windows | AS={int(labels.sum())} nonAS={int((labels==0).sum())} "
          f"| layer={a.layer}")

    backbone, tok = load_backbone(a.checkpoint_dir, a.raw_model, a.device)
    track = center_attention_track(backbone, tok, seqs, layer=a.layer, center_bp=a.center_bp,
                                   device=a.device)
    print(f"[dump] attention descriptor {track.shape}")

    def col(name, default): return df[name].to_numpy() if name in df.columns else default
    np.savez_compressed(
        a.out,
        embeddings=track.astype(np.float32),          # (N, L) -> probe treats this as the representation
        labels=labels.astype(np.int8),
        mu=np.zeros(len(df), dtype=np.float32),        # no head here; probe skips constant mu
        total_reads=col("total_reads", np.zeros(len(df))).astype(np.float32),
        chrom=col("chr", np.array(["?"] * len(df))).astype(str),
        anchor=col("anchor", np.zeros(len(df))).astype(np.int64),
        sequence=df[a.seq_col].astype(str).to_numpy())
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
