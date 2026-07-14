#!/usr/bin/env python
"""
eval_stratified_by_depth.py — post-hoc evaluation of a trained regression checkpoint,
reporting dev/test Spearman WITHIN read-depth bins (not just pooled).

WHY: when the depth-supervised (heteroscedastic / weighted_mse) head down-weights low-depth
loci, pooled Spearman can rise for two different reasons: (a) the model genuinely learned
sequence->effect signal, or (b) it merely stopped chasing noise. Stratifying by depth
separates them: strong Spearman IN THE HIGH-DEPTH BIN = real signal on trustworthy labels.

Reuses the model's own inference path (entexbert2.model_io) so predictions match training
exactly. Reads the 'depth' column from the split CSV (present when the build used
depth_col: total_reads). No retraining, CPU is fine.

Usage (McCleary, eb2 env, from repo root):
    python eval_stratified_by_depth.py \
        --checkpoint experiments/<exp>/runs/ref_alt_reg \
        --data_dir   experiments/<exp>/inputs/<DATASET>/fold0 \
        --split dev --bins 3
"""
import argparse, csv, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr

def read_split(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    paired = "sequence1" in rows[0] and "sequence2" in rows[0]
    if "depth" not in rows[0]:
        raise ValueError(f"{path} has no 'depth' column — rebuild inputs with depth_col: total_reads")
    y     = np.array([float(r["label"]) for r in rows], dtype=float)
    depth = np.array([float(r["depth"]) for r in rows], dtype=float)
    return rows, y, depth, paired

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="run dir with run_config.json + weights")
    ap.add_argument("--data_dir", required=True, help="dir holding dev.csv/test.csv (has 'depth')")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--bins", type=int, default=3, help="number of depth quantile bins")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from entexbert2.model_io import load_model_and_tokenizer, logits_and_embeddings

    model, tokenizer, run_config = load_model_and_tokenizer(args.checkpoint, device=args.device)
    rows, y, depth, paired = read_split(os.path.join(args.data_dir, f"{args.split}.csv"))
    maxlen = run_config.get("model_max_length", 512)

    def tok(seqs):
        enc = tokenizer(seqs, return_tensors="pt", padding="longest",
                        max_length=maxlen, truncation=True)
        return enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)

    preds = np.empty(len(rows), dtype=float)
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i+args.batch_size]
        if paired:
            ids,  m  = tok([r["sequence1"] for r in chunk])
            ida, ma  = tok([r["sequence2"] for r in chunk])
            logits, _ = logits_and_embeddings(model, ids, m, input_ids_alt=ida, attention_mask_alt=ma)
        else:
            ids, m = tok([r["sequence"] for r in chunk])
            logits, _ = logits_and_embeddings(model, ids, m)
        preds[i:i+len(chunk)] = logits.squeeze(-1).cpu().numpy()

    def line(tag, mask):
        n = int(mask.sum())
        if n < 3 or np.std(preds[mask]) == 0 or np.std(y[mask]) == 0:
            print(f"  {tag:28s} n={n:5d}  Spearman=  n/a  Pearson=  n/a"); return
        sr = spearmanr(preds[mask], y[mask]).correlation
        pr = pearsonr(preds[mask], y[mask])[0]
        lo, hi = depth[mask].min(), depth[mask].max()
        print(f"  {tag:28s} n={n:5d}  depth[{lo:6.0f},{hi:6.0f}]  Spearman={sr:+.3f}  Pearson={pr:+.3f}")

    print(f"\n=== {args.split}: Spearman/Pearson, pooled and stratified by depth ({args.bins} bins) ===")
    line("POOLED (all loci)", np.ones(len(y), dtype=bool))
    edges = np.quantile(depth, np.linspace(0, 1, args.bins + 1))
    for b in range(args.bins):
        lo, hi = edges[b], edges[b+1]
        mask = (depth >= lo) & (depth <= hi) if b == args.bins-1 else (depth >= lo) & (depth < hi)
        line(f"depth bin {b+1}/{args.bins}", mask)
    print("\nREAD: strong Spearman in the TOP depth bin = real signal on reliable loci.")
    print("      All bins weak but pooled ok = improvement was noise-suppression, not learning.")

if __name__ == "__main__":
    main()
