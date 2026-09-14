#!/usr/bin/env python
"""
dump_trunk_pred.py -- dump Stage-1 binding TRUNK predictions (mu) vs actual label on the held-out
TEST split, per cell, for the pred-vs-actual grid (fig_binding_pred_grid.py). Reuses the SAME scoring
path as plot_binding_pred_vs_actual (model_io.run_inference, regression -> mu). GPU. Deploy as a
module: src/entexbert2/analysis/dump_trunk_pred.py.

  python -m entexbert2.analysis.dump_trunk_pred \
      --checkpoint_dir <cell>/stage1_trunk/runs/reg \
      --test_csv <cell>/stage1_data/test.csv --out <cell>/eval/trunk_pred.csv
"""
import argparse
import numpy as np, pandas as pd
from entexbert2.model_io import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the Stage-1 regression trunk")
    ap.add_argument("--test_csv", required=True, help="stage1_data/test.csv (sequence + label)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    df = pd.read_csv(a.test_csv)
    seqcol = "sequence" if "sequence" in df.columns else "sequence1"
    assert seqcol in df.columns and "label" in df.columns, df.columns.tolist()
    logits, _emb, _rc = run_inference(a.checkpoint_dir, df[seqcol].astype(str).tolist(),
                                      a.batch_size, a.device, None)
    arr = np.asarray(logits)
    mu = arr[:, 0] if arr.ndim > 1 else arr.reshape(-1)
    out = pd.DataFrame({"label": df["label"].astype(float).to_numpy(), "mu": mu})
    if arr.ndim > 1 and arr.shape[1] > 1:          # a 2-output trunk also predicts sigma
        out["sigma"] = arr[:, 1]
    out.to_csv(a.out, index=False)
    y, m = out["label"].to_numpy(), out["mu"].to_numpy()
    r2 = 1.0 - np.sum((y - m) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    print(f"[trunk_pred] {len(out)} loci -> {a.out}  r2={r2:.3f}  mu[{mu.min():.2f},{mu.max():.2f}]")


if __name__ == "__main__":
    main()
