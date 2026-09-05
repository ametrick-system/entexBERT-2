#!/usr/bin/env python
"""
Dump Stage-1 binding-trunk pooled REFERENCE-window embeddings for AS/non-AS hetSNV loci.

Reads a windows CSV from build_asb_ism_windows.py (sequence1 = reference window, as_label,
total_reads, chr, anchor). Feeds sequence1 as a SINGLE sequence to the regression trunk via
model_io.run_inference, which returns the pooled final-layer embedding (the trunk's own
center_mean pooling) and the binding score mu. Saves an npz (embeddings + labels + coverage +
coords + sequence + mu) for probe_stage1_embeddings.py.

Trunk must be a Stage-1 regression checkpoint (task=regression). GPU recommended.

  python dump_stage1_ref_embeddings.py \
    --checkpoint_dir $ASB/experiments/ctcf/stage1_trunk/runs/reg \
    --windows_csv    $WORK/ctcf_asb_ism_windows_entex.csv \
    --out            $WORK/stage1_ref_emb_ctcf_entex.npz --device cuda
"""
import argparse, numpy as np, pandas as pd
from entexbert2.model_io import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="Stage-1 binding trunk run dir (task=regression)")
    ap.add_argument("--windows_csv", required=True, help="output of build_asb_ism_windows.py")
    ap.add_argument("--seq_col", default="sequence1", help="reference-window column")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0, help="0=all; else class-balanced subsample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.windows_csv)
    for c in (a.seq_col, "as_label"):
        assert c in df.columns, f"missing column {c} in {a.windows_csv}"
    labels = (df["as_label"].astype(str).str.upper() == "AS").astype(int).to_numpy()

    if a.max_rows and a.max_rows > 0 and len(df) > a.max_rows:
        rng = np.random.default_rng(a.seed)
        idx = np.arange(len(df)); pos = idx[labels == 1]; neg = idx[labels == 0]
        k = min(a.max_rows // 2, len(pos))
        keep = np.r_[rng.choice(pos, k, replace=False),
                     rng.choice(neg, min(a.max_rows - k, len(neg)), replace=False)]
        keep.sort(); df = df.iloc[keep].reset_index(drop=True); labels = labels[keep]

    texts = df[a.seq_col].astype(str).tolist()   # list[str] -> single-seq (ref_single) mode
    print(f"[in] {len(texts)} ref windows | AS={int(labels.sum())} nonAS={int((labels == 0).sum())}")

    # regression trunk, single-seq: run_inference returns (logits=mu, emb=pool1, run_config)
    logits, emb, rc = run_inference(a.checkpoint_dir, texts,
                                    batch_size=a.batch_size, device=a.device, dump_pools=False)
    assert rc.get("task") == "regression", f"expected a regression trunk, got task={rc.get('task')}"
    emb = np.asarray(emb, dtype=np.float32)                                  # (N, H) pooled embedding
    mu = np.asarray(logits, dtype=np.float32).reshape(len(texts), -1)[:, 0]  # binding score
    assert emb.shape[0] == len(labels), f"row mismatch {emb.shape} vs {len(labels)}"
    print(f"[dump] embeddings {emb.shape} | task={rc.get('task')} pooling={rc.get('pooling_mode')}")

    def col(name, default): return df[name].to_numpy() if name in df.columns else default
    np.savez_compressed(
        a.out,
        embeddings=emb, labels=labels.astype(np.int8), mu=mu,
        total_reads=col("total_reads", np.zeros(len(df))).astype(np.float32),
        chrom=col("chr", np.array(["?"] * len(df))).astype(str),
        anchor=col("anchor", np.zeros(len(df))).astype(np.int64),
        sequence=df[a.seq_col].astype(str).to_numpy())
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
