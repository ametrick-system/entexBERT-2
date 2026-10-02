#!/usr/bin/env python
"""fig_matched_vs_crossdonor.py -- per-assay GRID: for each training donor, the WITHIN-INDIVIDUAL AUROC
(matched, read from eval_results.json) vs the AVERAGE CROSS-DONOR AUROC (that head scored on the OTHER
donors' Stage-2 test.csv, mean over them, from score_crossdonor -> crossdonor_<arm>.csv). Both are the
SAME metric (test.csv roc_auc on ell), so the drop from matched to cross-donor is a clean generalization
gap. One panel per assay; x-axis = training donors; two bars each. Login node, no GPU.

  python fig_matched_vs_crossdonor.py --exp_root experiments --arm ref \\
      --assays CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC,H3K4me3,H3K27ac,H3K4me1,H3K9me3,H3K36me3,H3K27me3 \\
      --donors ENC-001,ENC-002,ENC-003,ENC-004 --title "Within- vs cross-individual (reference)" \\
      --out fig_matched_vs_crossdonor_ref.png
"""
import argparse, os, json, math, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATCHED = "#5b9bd5"    # within-individual (eval_results.json)
CROSS = "#e8988a"      # cross-donor mean (score_crossdonor)
KEY_CANDIDATES = ["eval_auroc", "eval_roc_auc", "eval_auc", "auroc", "roc_auc", "auc"]


def find_auroc(d, forced=None):
    if forced:
        return float(d[forced]) if forced in d else None
    for k in KEY_CANDIDATES:
        if k in d:
            return float(d[k])
    cands = [k for k in d if "auroc" in k.lower()] or [k for k in d if "auc" in k.lower()]
    if cands:
        cands.sort(key=lambda k: ("auroc" not in k.lower(), k))
        return float(d[cands[0]])
    return None


def matched_auroc(exp_root, assay, donor, arm, run_subdir, metric_key):
    """within-individual AUROC from eval_results.json (nested results/<clf_...>/ dir)."""
    base = os.path.join(exp_root, f"{assay.lower()}_{donor}", f"stage2_{arm}", "runs", run_subdir)
    hits = (glob.glob(os.path.join(base, "results", "*", "eval_results.json"))
            or glob.glob(os.path.join(base, "**", "eval_results.json"), recursive=True))
    if not hits:
        return None
    with open(sorted(hits)[0]) as fh:
        return find_auroc(json.load(fh), metric_key)


def cross_mean(exp_root, assay, donor, arm):
    """mean cross-donor AUROC for this head from crossdonor_<arm>.csv (donor_kind == cross-donor)."""
    p = os.path.join(exp_root, f"{assay.lower()}_{donor}", "eval", f"crossdonor_{arm}.csv")
    if not os.path.exists(p):
        return None, 0
    df = pd.read_csv(p)
    cd = df[(df["donor_kind"] == "cross-donor") & df["auroc"].notna()]
    if len(cd) == 0:
        return None, 0
    return float(cd["auroc"].mean()), len(cd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--assays", required=True)
    ap.add_argument("--donors", default="ENC-001,ENC-002,ENC-003,ENC-004")
    ap.add_argument("--arm", choices=["ref", "personal"], default="ref")
    ap.add_argument("--run_subdir", default="clf_s20_seed20")
    ap.add_argument("--metric_key", default=None)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    donors = [s for s in a.donors.split(",") if s]
    ncols = a.ncols
    nrows = math.ceil(len(assays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(assays):]:
        ax.axis("off")

    x = np.arange(len(donors)); w = 0.38
    print(f"{'assay':16} {'donor':10} {'matched':>8} {'cross_mean':>11} {'n_cross':>7}")
    for i, (ax, assay) in enumerate(zip(axes_flat, assays)):
        mvals, cvals = [], []
        for d in donors:
            m = matched_auroc(a.exp_root, assay, d, a.arm, a.run_subdir, a.metric_key)
            c, ncd = cross_mean(a.exp_root, assay, d, a.arm)
            mvals.append(m if m is not None else np.nan)
            cvals.append(c if c is not None else np.nan)
            print(f"{assay:16} {d:10} {('%.3f'%m) if m is not None else '  --':>8} "
                  f"{('%.3f'%c) if c is not None else '  --':>11} {ncd:>7}")
        ax.bar(x - w/2, np.nan_to_num(mvals), w, color=MATCHED, label="within (matched)")
        ax.bar(x + w/2, np.nan_to_num(cvals), w, color=CROSS, label="cross-donor (mean)")
        ax.axhline(0.5, ls=":", color="0.45", lw=1.0, zorder=0)
        ax.set_title(assay, fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("ENC-", "") for d in donors],
                           fontsize=8, rotation=0)
        ax.set_yticks(np.arange(0, 1.001, 0.25))
        if i % ncols == 0:
            ax.set_ylabel("AUROC", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    h = [plt.Rectangle((0, 0), 1, 1, color=MATCHED), plt.Rectangle((0, 0), 1, 1, color=CROSS)]
    fig.legend(h, ["within-individual (eval_results.json)", "cross-donor mean (other individuals)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96 if a.title else 1.0])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[wrote] {a.out}  ({len(assays)} assays, arm={a.arm})")


if __name__ == "__main__":
    main()
