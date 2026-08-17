#!/usr/bin/env python
"""
Visualize whether the classification contrast HEAD separates AS from non-AS loci more than the
raw TRUNK, using the dump from dump_contrast_embeddings.py.

Design note (why THESE panels): the head is a symmetric DISTANCE classifier,
    s = ||P(h1) - P(h2)||,   ell = a*s + b  (a>0),   p = sigmoid(ell).
So the AS signal is RADIAL (distance-from-origin) and lives in the TAIL of the norm: h1,h2 differ
at one base out of ~256, so most pairs collapse near 0 and only a minority of AS loci are pushed
out. A 2-PC scatter centered at the origin cannot show radial separation, and a linear-x histogram
buries the informative tail under the zero spike. We therefore lead with:
  Row 1  ECDF of the contrast norm by class (trunk | head) -- tail-legible; the gap between the AS
         and non-AS curves IS the separation. Titles report norm-as-classifier AUROC.
         (For the head, norm AUROC == the model's classification AUROC on this set, since
          ell = a*s + b is monotone in s -- a built-in consistency check.)
  Row 2  left : ell = logit P(ASB) distribution by class -- the actual decision axis.
         right: head-contrast PCA (2D), SUPPORTING only (radial/tail signal is largely invisible here).

numpy/sklearn/matplotlib only (no torch).

  python plot_contrast_pca.py --npz $WORK/pca_contrast_test.npz --out pca_contrast_test.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

AS_COLOR, NONAS_COLOR = "#d1495b", "#3d6cb3"   # AS = red, non-AS = blue


def _auroc(y, score):
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")


def _softplus(x):
    # log(1+e^x), numerically stable; no scipy dependency
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def resolve_ab(d, a_cli=None, a_raw_cli=None, b_cli=None):
    """
    Resolve the trained logistic-link scalars a>0 and b for ell = a*s + b, so the decision
    boundary p=0.5 (ell=0) maps to a distance radius  delta* = -b/a  on the head-norm axis.

    Priority: explicit CLI (--a or --a_raw, and --b) > keys stashed in the npz by the dump script
    (a/b, or a_raw/b, or dist_a/dist_b). Returns (a, b) or (None, None) if unavailable.
    """
    if b_cli is not None and (a_cli is not None or a_raw_cli is not None):
        a = float(a_cli) if a_cli is not None else float(_softplus(np.array(a_raw_cli)))
        return a, float(b_cli)
    keys = set(d.files) if hasattr(d, "files") else set(d.keys())
    def _get(k):
        return float(np.asarray(d[k]).reshape(-1)[0]) if k in keys else None
    b = _get("b") if _get("b") is not None else _get("dist_b")
    if "a" in keys:
        a = _get("a")
    elif "a_raw" in keys or "dist_a" in keys:
        raw = _get("a_raw") if "a_raw" in keys else _get("dist_a")
        a = float(_softplus(np.array(raw)))
    else:
        a = None
    if a is not None and b is not None and a > 0:
        return a, b
    return None, None


def ecdf(ax, val, y, title, xlabel, logx="auto", vline=None, vlabel=None):
    """ECDF of `val` by class. Separation = vertical gap between the two curves.
    logx='auto' uses a log x-axis only when the values span >~1.5 decades (the head norm has a
    zero-pileup + long tail; the trunk norm sits in a narrow band where log-x just crowds ticks)."""
    for lab, c, name in [(0, NONAS_COLOR, "non-AS"), (1, AS_COLOR, "AS")]:
        v = np.sort(val[y == lab])
        f = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, f, color=c, lw=2, label=f"{name} (n={int((y==lab).sum())})")
    pos = val[val > 0]
    if logx == "auto":
        span = (np.log10(np.percentile(pos, 99.5)) - np.log10(np.percentile(pos, 1))) if len(pos) else 0
        logx = span > 1.5
    if logx:
        ax.set_xscale("log")
        lo = max(np.percentile(pos, 0.5), 1e-6) if len(pos) else 1e-6
        ax.set_xlim(lo, np.percentile(val, 99.9))
    else:
        lo, hi = np.percentile(val, [0.2, 99.5])
        ax.set_xlim(lo, hi)
    au = _auroc(y, val)
    if vline is not None and vline > 0:
        ax.axvline(vline, color="k", lw=1.4, ls="--", alpha=0.8,
                   label=(vlabel or f"decision radius = {vline:.3g}"))
    ax.set_xlabel(xlabel); ax.set_ylabel("cumulative fraction")
    ax.set_title(f"{title}\nnorm AUROC = {au:.3f}  (median AS {np.median(val[y==1]):.3f} / non-AS {np.median(val[y==0]):.3f})")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, which="both")


def ell_hist(ax, ell, y):
    lo, hi = np.percentile(ell, [0.5, 99.5])
    bins = np.linspace(lo, hi, 45)
    ax.hist(ell[y == 0], bins=bins, color=NONAS_COLOR, alpha=0.55, density=True, label="non-AS")
    ax.hist(ell[y == 1], bins=bins, color=AS_COLOR, alpha=0.55, density=True, label="AS")
    ax.axvline(0.0, color="k", lw=1.4, ls="--", alpha=0.8, label="p=0.5 (ell=0)")  # SAME boundary as delta*=-b/a
    au = _auroc(y, ell)
    ax.set_xlabel("ell = logit P(ASB)   (dashed p=0.5 == distance radius -b/a)"); ax.set_ylabel("density")
    ax.set_title(f"decision axis: model logit by class\nell AUROC = {au:.3f}")
    ax.legend(loc="best", frameon=False, fontsize=8)


def pca_scatter(ax, X, y):
    p = PCA(n_components=2, random_state=0)
    Z = p.fit_transform(X); evr = p.explained_variance_ratio_
    for lab, c, name in [(0, NONAS_COLOR, "non-AS"), (1, AS_COLOR, "AS")]:
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=6, c=c, alpha=0.4, linewidths=0, rasterized=True,
                   label=f"{name} (n={int(m.sum())})")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("head contrast P(h1)-P(h2), PCA\n(SUPPORTING: radial/tail signal mostly hidden here)")
    ax.legend(loc="best", frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="pca_contrast.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--linx", action="store_true", help="linear x for the ECDFs (default log-x)")
    ap.add_argument("--a", type=float, default=None, help="trained slope a (>0); else read from npz")
    ap.add_argument("--a_raw", type=float, default=None, help="trained a_raw (a=softplus(a_raw)); else read from npz")
    ap.add_argument("--b", type=float, default=None, help="trained bias b; else read from npz")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = d["labels"].astype(int)
    head = d["head_contrast"]
    tnorm = d["trunk_norm"] if "trunk_norm" in d else np.linalg.norm(d["trunk_contrast"], axis=1)
    hnorm = d["head_norm"] if "head_norm" in d else np.linalg.norm(head, axis=1)
    ell = d["ell"] if "ell" in d else hnorm  # ell should always be present
    print(f"[load] {len(y)} loci | pos={int(y.sum())} neg={int((y==0).sum())}")

    # decision radius delta* = -b/a : p>0.5 <=> ell>0 <=> distance s > delta*. Draw it on the
    # head-norm axis (where s lives). Only meaningful when b<0 (rare-positive => negative bias).
    a, b = resolve_ab(d, a_cli=args.a, a_raw_cli=args.a_raw, b_cli=args.b)
    radius = None
    if a is not None and b is not None:
        radius = -b / a
        beyond = hnorm > radius
        f_as = float((beyond & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
        f_non = float((beyond & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
        print(f"[boundary] a={a:.4g} b={b:.4g} -> decision radius delta* = -b/a = {radius:.4g}")
        print(f"[boundary] fraction beyond radius (called AS): AS {f_as:.3f} | non-AS {f_non:.3f}"
              f"  (=> precision-limiting non-AS spillover {f_non:.3f})")
        if radius <= 0:
            print("[boundary] radius <= 0 (b>=0): every locus is called AS at p=0.5; line omitted.")
            radius = None
    else:
        print("[boundary] a/b not provided and not found in npz -> skipping decision-radius line "
              "(pass --a/--a_raw and --b, or have the dump script stash them).")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    lx = False if args.linx else "auto"
    ecdf(axes[0, 0], tnorm, y, "TRUNK contrast norm  ||h1-h2||", "contrast norm  ||.||", logx=lx)
    ecdf(axes[0, 1], hnorm, y, "HEAD contrast norm  = distance s", "distance  s = ||P(h1)-P(h2)||", logx=lx,
         vline=radius, vlabel=(f"decision radius -b/a = {radius:.3g}" if radius else None))
    ell_hist(axes[1, 0], ell, y)
    pca_scatter(axes[1, 1], head, y)
    fig.suptitle("Does the contrast head pull AS apart more than the trunk?  (radial separation, tail-legible)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[write] {args.out}")
    print(f"[summary] trunk-norm AUROC={_auroc(y,tnorm):.4f} | head-norm AUROC={_auroc(y,hnorm):.4f} | "
          f"ell AUROC={_auroc(y,ell):.4f}")


if __name__ == "__main__":
    main()
