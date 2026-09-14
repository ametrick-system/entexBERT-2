#!/usr/bin/env python
"""
fig_binding_pred_grid.py -- per-assay GRID of the Stage-1 binding TRUNK: predicted mu vs actual label,
as a log-density hexbin, one panel per assay (same small-multiples layout as fig_ism_confusion_grid).
Reads each cell's trunk_pred.csv (dump_trunk_pred.py: columns label, mu[, sigma]). Login node, no model.

  python fig_binding_pred_grid.py --exp_root experiments \
      --assays CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC --donor ENC-002 \
      --title "ENC-002" --out fig_binding_pred_grid.png
Per-cell dump defaults to <cell>/eval/trunk_pred.csv; override the directory with --pred_dir
(files named <assay_lower>_trunk_pred.csv there). Each panel annotates n, r2, Spearman rho.
"""
import argparse, os, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--assays", required=True, help="comma-separated, e.g. CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC")
    ap.add_argument("--donor", default="ENC-002")
    ap.add_argument("--pred_dir", default=None,
                    help="dir of <assay>_trunk_pred.csv; else <cell>/eval/trunk_pred.csv")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--gridsize", type=int, default=45)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    ncols = a.ncols
    nrows = math.ceil(len(assays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(assays):]:
        ax.axis("off")

    for i, (ax, assay) in enumerate(zip(axes_flat, assays)):
        cell = os.path.join(a.exp_root, f"{assay.lower()}_{a.donor}")
        pv = (os.path.join(a.pred_dir, f"{assay.lower()}_trunk_pred.csv")
              if a.pred_dir else os.path.join(cell, "eval", "trunk_pred.csv"))
        if not os.path.exists(pv):
            ax.text(0.5, 0.5, f"{assay}\n(no trunk_pred)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5"); ax.set_title(assay, fontsize=10)
            continue
        d = pd.read_csv(pv)
        y = d["label"].to_numpy(float); mu = d["mu"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(mu)
        y, mu = y[ok], mu[ok]
        r2 = 1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12) if len(y) > 2 else np.nan
        sr = spearmanr(y, mu).correlation if len(y) > 2 else np.nan
        ax.hexbin(y, mu, gridsize=a.gridsize, cmap="magma_r", norm=LogNorm(), mincnt=1, linewidths=0)
        lo = float(min(y.min(), mu.min())); hi = float(max(y.max(), mu.max()))
        ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=0.8, zorder=3)
        ax.set_title(assay, fontsize=10)
        ax.text(0.04, 0.96, f"n={len(y)}\n$r^2$={r2:.2f}\n$\\rho$={sr:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        ax.set_xlabel("actual binding (log1p FC)", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel(r"predicted $\mu$", fontsize=8)
        ax.tick_params(labelsize=7)
        print(f"[{assay}] n={len(y)} r2={r2:.3f} spearman={sr:.3f}")

    if a.title:
        fig.suptitle(f"Stage-1 trunk binding: predicted vs actual \u2014 {a.title}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if a.title else 1.0])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[wrote] {a.out}  ({len(assays)} assays)")


if __name__ == "__main__":
    main()
