#!/usr/bin/env python3
"""
plot_binding_pred_vs_actual.py — predicted vs actual for a Stage-1 binding regressor (any TF; pass --tf_label CTCF/EP300 for the title).

Runs single-sequence inference on the held-out TEST split and plots predicted mu
against the true log1p fold-change label. Colors points by predicted sigma when the
model has a test-time sigma head. Run from repo root (conda activate eb2):

  python plot_binding_pred_vs_actual.py \
      --checkpoint_dir <.../runs/reg> \
      --test_csv <.../inputs/ENC-002__TF-ChIP-seq_CTCF/fold0/test.csv> \
      --out binding_pred_vs_actual.png
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_points", type=int, default=8000, help="subsample for legible scatter")
    ap.add_argument("--tf_label", default="binding", help="TF name for the title (e.g. CTCF, EP300)")
    ap.add_argument("--out", default="binding_pred_vs_actual.png")
    a = ap.parse_args()

    from entexbert2.model_io import run_inference
    df = pd.read_csv(a.test_csv)
    assert "sequence" in df.columns and "label" in df.columns, df.columns.tolist()
    texts = df["sequence"].tolist()                      # single-sequence inference
    logits, _emb, rc = run_inference(a.checkpoint_dir, texts, a.batch_size, a.device, None)
    arr = np.asarray(logits, dtype=float).reshape(len(df), -1)
    mu = arr[:, 0]
    sigma = np.exp(0.5 * arr[:, 1]) if arr.shape[1] >= 2 else None
    y = df["label"].to_numpy(dtype=float)

    r2 = r2_score(y, mu); pr = pearsonr(y, mu)[0]; sr = spearmanr(y, mu).correlation
    print(f"n={len(y)}  R2={r2:.4f}  Pearson={pr:.4f}  Spearman={sr:.4f}"
          + (f"  sigma[min={sigma.min():.3f} med={np.median(sigma):.3f} max={sigma.max():.3f}]"
             if sigma is not None else "  (no sigma head)"))

    # subsample for a legible scatter (keep the fit stats on the full set, reported above)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(y), min(a.max_points, len(y)), replace=False)
    yp, mp = y[idx], mu[idx]
    sp = sigma[idx] if sigma is not None else None

    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=300)
    lo = float(min(yp.min(), mp.min())); hi = float(max(yp.max(), mp.max()))
    pad = 0.04 * (hi - lo)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="#999999", lw=1.0,
            ls="--", zorder=1, label="y = x")
    if sp is not None:
        sc = ax.scatter(yp, mp, c=sp, s=7, alpha=0.45, linewidths=0,
                        cmap="viridis", zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("predicted $\\sigma$ (uncertainty)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    else:
        ax.scatter(yp, mp, s=7, alpha=0.4, linewidths=0, color="#1f3a5f", zorder=2)

    ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("actual binding  (log1p fold-change)", fontsize=8)
    ax.set_ylabel("predicted  $\\mu$", fontsize=8)
    ax.set_title(f"Held-out {a.tf_label} binding: predicted vs actual", fontsize=8, loc="left")
    # headline fit stats as one annotation block (values off the axis)
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nPearson = {pr:.3f}\n$n$ = {len(y):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    ax.tick_params(labelsize=6)
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()
