#!/usr/bin/env python
"""plot_contrast_composite_cell.py -- Fig-F contrast composite for ONE (individual, assay), EN-TEx only.
Three panels (reproduces the reference figure):
  col 0  TRUNK contrast norm ||h1-h2||             -- ECDF by class (separation = gap between curves)
  col 1  HEAD  contrast norm = distance s = ||P(h1)-P(h2)||  -- ECDF, log-x, decision radius -b/a drawn
  col 2  ell = logit P(ASB) by class               -- histogram, dashed p=0.5 (ell=0)

Reads the per-cell contrast npz from dump_contrast_embeddings.py:
  <exp_root>/<assay_lower>_<donor>/eval/contrast_<arm>_entex.npz
(carries labels, ell, trunk_norm, head_norm [+ dist_a/dist_b for the decision radius]).
Helper panels are the verbatim ecdf/ell_hist/resolve_ab/render_row from plot_contrast_composite.py.
numpy/sklearn/matplotlib only (no torch). Login node.

  python plot_contrast_composite_cell.py --exp_root experiments --donor ENC-002 --assay CTCF \
      --arm personal --out fig_contrast_composite_ctcf_ENC-002.png
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def render_row(axes, d, set_label, linx):
    y = d["labels"].astype(int)
    tnorm = d["trunk_norm"] if "trunk_norm" in d else np.linalg.norm(d["trunk_contrast"], axis=1)
    hnorm = d["head_norm"] if "head_norm" in d else np.linalg.norm(d["head_contrast"], axis=1)
    ell = d["ell"] if "ell" in d else hnorm
    a, b = resolve_ab(d)
    radius = (-b / a) if (a is not None and b is not None and -b / a > 0) else None
    lx = False if linx else "auto"
    ecdf(axes[0], tnorm, y, f"[{set_label}] TRUNK contrast norm  ||h1-h2||", "contrast norm  ||.||", logx=lx)
    ecdf(axes[1], hnorm, y, f"[{set_label}] HEAD contrast norm  = distance s",
         "distance  s = ||P(h1)-P(h2)||", logx=lx,
         vline=radius, vlabel=(f"decision radius -b/a = {radius:.3g}" if radius else None))
    ell_hist(axes[2], ell, y)
    return dict(set=set_label, n=len(y), pos=int(y.sum()),
                trunk_auroc=float(_auroc(y, tnorm)), head_auroc=float(_auroc(y, hnorm)),
                ell_auroc=float(_auroc(y, ell)), radius=radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--arm", default="personal", choices=["ref", "personal"])
    ap.add_argument("--npz_name", default="contrast_<arm>_entex.npz",
                    help="per-cell npz under <cell>/eval/ ; '<arm>' is substituted")
    ap.add_argument("--tf_label", default="", help="panel/suptitle label (default '<assay> <donor>')")
    ap.add_argument("--linx", action="store_true", help="linear x for the ECDFs (default log-x auto)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cell = os.path.join(args.exp_root, f"{args.assay.lower()}_{args.donor}")
    npz = os.path.join(cell, "eval", args.npz_name.replace("<arm>", args.arm))
    if not os.path.exists(npz):
        raise SystemExit(f"contrast npz not found: {npz}\n"
                         f"  (run dump_contrast_embeddings for {args.assay}/{args.donor} {args.arm} first)")
    d = np.load(npz, allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    s = render_row(axes, d, "EN-TEx", args.linx)
    label = args.tf_label or f"{args.assay} {args.donor}"
    fig.suptitle(f"{label} ({args.arm}) -- contrast head vs trunk: does the head pull AS apart?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = args.out or f"fig_contrast_composite_{args.assay.lower()}_{args.donor}_{args.arm}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[{args.assay}/{args.donor} {args.arm}] n={s['n']} pos={s['pos']} | "
          f"trunk-norm AUROC={s['trunk_auroc']:.4f} | head-norm AUROC={s['head_auroc']:.4f} | "
          f"ell AUROC={s['ell_auroc']:.4f} | decision radius -b/a={s['radius']}")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
