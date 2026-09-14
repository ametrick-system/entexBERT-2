#!/usr/bin/env python
"""
fig_contrast_violin_grid.py -- per-assay GRID of VIOLIN plots showing the distribution of a contrast
quantity split by AS vs non-AS. Re-casts the Fig F contrast composite (which used ECDFs / a logit
histogram) as violins, one panel per assay.

Reads each cell's contrast npz (dump_contrast_embeddings.py -> contrast_<arm>_entex.npz), which carries
per-locus: labels, ell (logit P(ASB)), trunk_norm (||h1-h2||), head_norm (distance s=||P(h1)-P(h2)||).
Choose which quantity to violin with --metric:
    head_norm  (default) = distance s   -- the head's learned contrast, the discriminative axis
    trunk_norm           = ||h1-h2||    -- the frozen trunk's raw contrast (usually NOT discriminative)
    ell                  = logit P(ASB) -- the decision axis
Each panel annotates the rank-AUROC of that metric (AS vs non-AS separation) and n per class. Login node.

  python fig_contrast_violin_grid.py --exp_root experiments --donor ENC-002 --arm ref \
      --metric head_norm --assays CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC,H3K4me3,H3K27ac,H3K4me1,H3K9me3,H3K36me3,H3K27me3 \
      --title "Head contrast distance s by class (reference), ENC-002" --out fig_contrast_violin_head_ref.png
"""
import argparse, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AS_COLOR = "#c44e52"     # crimson
NONAS_COLOR = "#4c72b0"  # blue
METRIC_XLABEL = {"head_norm": "distance  s = ||P(h1)-P(h2)||",
                 "trunk_norm": "trunk contrast  ||h1-h2||",
                 "ell": "ell = logit P(ASB)",
                 "gain": "distance gained  z(head) - z(trunk)"}


def rank_auroc(y, x):
    """AUROC of score x for binary label y (1=AS), tie-aware via average ranks."""
    y = np.asarray(y); x = np.asarray(x, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_metric(exp_root, assay, donor, arm, npz_name, metric):
    p = os.path.join(exp_root, f"{assay.lower()}_{donor}", "eval", npz_name.replace("<arm>", arm))
    if not os.path.exists(p):
        return None, None, p
    d = np.load(p)
    y = d["labels"] if "labels" in d else d["as_label"]
    y = np.asarray([1 if str(v) in ("1", "AS", "True", "1.0") else int(float(v)) if str(v).replace('.','',1).isdigit() else 0
                    for v in y]) if y.dtype.kind in "US" else (np.asarray(y) > 0.5).astype(int)
    def norm_of(name):
        if name in d:
            return np.asarray(d[name], dtype=float)
        ck = "trunk_contrast" if name == "trunk_norm" else "head_contrast"
        return np.linalg.norm(d[ck], axis=1) if ck in d else None

    def zscore(v):
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v - v.mean()

    if metric == "gain":
        # distance the HEAD gains over the trunk, per locus. Raw head_norm - trunk_norm is
        # scale-confounded (different spaces), so standardize each across loci first: a locus the
        # head ranks more-separated than the trunk -> positive; less -> negative. Centered at 0.
        hn, tn = norm_of("head_norm"), norm_of("trunk_norm")
        if hn is None or tn is None:
            return None, None, p
        val = zscore(hn) - zscore(tn)
    elif metric in d:
        val = np.asarray(d[metric], dtype=float)
    elif metric in ("trunk_norm", "head_norm"):
        val = norm_of(metric)
        if val is None:
            return None, None, p
    else:
        return None, None, p
    return val, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--assays", required=True)
    ap.add_argument("--donor", default="ENC-002")
    ap.add_argument("--arm", default="ref", choices=["ref", "personal"])
    ap.add_argument("--metric", default="head_norm", choices=["head_norm", "trunk_norm", "ell", "gain"])
    ap.add_argument("--npz_name", default="contrast_<arm>_entex.npz",
                    help="per-cell npz under <cell>/eval/ ; '<arm>' is substituted")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--clip_pct", type=float, default=99.5, help="clip upper tail to this percentile for display")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    ncols = a.ncols
    nrows = math.ceil(len(assays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(assays):]:
        ax.axis("off")

    print(f"{'assay':18} {'n_AS':>7} {'n_nonAS':>8} {'AUROC':>7}  metric={a.metric}")
    for i, (ax, assay) in enumerate(zip(axes_flat, assays)):
        val, y, p = load_metric(a.exp_root, assay, a.donor, a.arm, a.npz_name, a.metric)
        if val is None:
            ax.text(0.5, 0.5, f"{assay}\n(no contrast npz)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
            ax.set_title(assay, fontsize=10); ax.set_xticks([]); continue
        au = rank_auroc(y, val)
        # clip display tail (violins squish on long tails); keep both classes on the same clip
        if a.metric in ("ell", "gain"):   # centered quantities: clip both tails
            lo, hi = np.percentile(val, [0.5, 99.5]); vshow = np.clip(val, lo, hi)
        else:
            hi = np.percentile(val, a.clip_pct); lo = float(val.min())
            vshow = np.clip(val, lo, hi)
        data = [vshow[y == 0], vshow[y == 1]]
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        print(f"{assay:18} {n1:>7} {n0:>8} {au:>7.3f}  ({os.path.basename(p)})")
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=False,
                              widths=0.85)
        for b, c in zip(parts["bodies"], [NONAS_COLOR, AS_COLOR]):
            b.set_facecolor(c); b.set_alpha(0.6); b.set_edgecolor("none")
        if "cmedians" in parts:
            parts["cmedians"].set_color("0.15"); parts["cmedians"].set_linewidth(1.2)
        if a.metric in ("ell", "gain"):
            ax.axhline(0.0, ls="--", color="k", lw=1.0, alpha=0.7)  # ell: p=0.5 ; gain: no head-over-trunk gain
        ax.set_title(f"{assay}  (AUROC {au:.2f})", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"non-AS\n(n={n0})", f"AS\n(n={n1})"] if i // ncols == nrows - 1
                           else ["non-AS", "AS"], fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel(METRIC_XLABEL[a.metric], fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    if a.title:
        fig.suptitle(a.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if a.title else 1.0])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[wrote] {a.out}  ({len(assays)} assays, arm={a.arm}, metric={a.metric})")


if __name__ == "__main__":
    main()
