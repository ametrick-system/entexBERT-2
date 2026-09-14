#!/usr/bin/env python
"""
fig_auroc_ref_personal_grid.py -- per-assay GRID of bar charts: ref vs personal ASB-head AUROC,
read from each head's eval_results.json (HF Trainer held-out Stage-2 test metric). One panel per
assay, two bars (personal / reference), y-axis 0.5-1.0. Matches the collaborator's small-multiples
layout. Login node, no model, no GPU.

NOTE on the metric: eval_results.json is leak-free w.r.t. the HEAD's training and a matched
ref-vs-personal comparison (ref & personal share the per-assay fold+salt), but it is NOT
coverage-matched, so it is generally HIGHER than the honest score_asb leak_free+covmatch number.
Report the score_asb value in the paper; this figure reproduces the eval_results.json numbers.

Per cell it reads:
  <exp_root>/<assay_lower>_<donor>/stage2_<arm>/runs/<run_subdir>/eval_results.json   (arm=ref|personal)
The AUROC key is auto-detected (eval_auroc / eval_roc_auc / eval_auc / ...); override with --metric_key.

  python fig_auroc_ref_personal_grid.py --exp_root experiments --donor ENC-002 \
      --assays ATAC,CTCF,H3K4me1,H3K4me3,H3K9me3,H3K27ac,H3K27me3,H3K36me3,EP300,POLR2A,POLR2AphosphoS5 \
      --title "AUROC (Personal vs. Reference), ENC-002" --out fig_auroc_ref_personal_grid.png
"""
import argparse, os, json, math, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERSONAL = "#e8988a"   # salmon
REFERENCE = "#5b9bd5"  # blue
KEY_CANDIDATES = ["eval_auroc", "eval_roc_auc", "eval_auc", "auroc", "roc_auc", "auc",
                  "eval_AUROC", "eval_auROC"]


def find_auroc(d, forced=None):
    """Return (value, key) for the AUROC in an eval_results.json dict, or (None, None)."""
    if forced:
        return (float(d[forced]), forced) if forced in d else (None, None)
    for k in KEY_CANDIDATES:
        if k in d:
            return float(d[k]), k
    # fuzzy: any key mentioning auroc/auc (prefer 'auroc')
    cands = [k for k in d if "auroc" in k.lower()] or [k for k in d if "auc" in k.lower()]
    if cands:
        cands.sort(key=lambda k: ("auroc" not in k.lower(), k))
        return float(d[cands[0]]), cands[0]
    return None, None


def load_auroc(exp_root, assay, donor, arm, run_subdir, metric_key):
    # eval_results.json lives under stage2_<arm>/runs/<run_subdir>/results/<clf_...>/ (inner dir varies
    # by assay+arm), so glob for it rather than hardcode the leaf name.
    base = os.path.join(exp_root, f"{assay.lower()}_{donor}", f"stage2_{arm}", "runs", run_subdir)
    hits = (glob.glob(os.path.join(base, "results", "*", "eval_results.json"))
            or glob.glob(os.path.join(base, "**", "eval_results.json"), recursive=True))
    if not hits:
        return None, None, base
    with open(sorted(hits)[0]) as fh:
        d = json.load(fh)
    v, k = find_auroc(d, metric_key)
    return v, k, hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--assays", required=True, help="comma-separated (panel order)")
    ap.add_argument("--donor", default="ENC-002")
    ap.add_argument("--run_subdir", default="clf_s20_seed20", help="head run dir under stage2_<arm>/runs/")
    ap.add_argument("--metric_key", default=None, help="force the eval_results.json AUROC key")
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    ystep = 0.1 if (1.0 - a.ymin) <= 0.6 else 0.2   # denser ticks when zoomed in
    ncols = a.ncols
    nrows = math.ceil(len(assays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(assays):]:
        ax.axis("off")

    print(f"{'assay':18} {'personal':>9} {'reference':>10}  key")
    for i, (ax, assay) in enumerate(zip(axes_flat, assays)):
        pv, pk, pp = load_auroc(a.exp_root, assay, a.donor, "personal", a.run_subdir, a.metric_key)
        rv, rk, rp = load_auroc(a.exp_root, assay, a.donor, "ref", a.run_subdir, a.metric_key)
        if rv is None:  # older layouts named the dir stage2_reference
            rv, rk, rp = load_auroc(a.exp_root, assay, a.donor, "reference", a.run_subdir, a.metric_key)
        key = pk or rk or "?"
        print(f"{assay:18} {('%.3f'%pv) if pv is not None else '  --':>9} "
              f"{('%.3f'%rv) if rv is not None else '  --':>10}  {key}")
        if pv is None and rv is None:
            ax.text(0.5, 0.5, f"{assay}\n(no eval_results)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
            ax.set_title(assay, fontsize=10); ax.set_ylim(a.ymin, 1.0)
            ax.set_xticks([]); continue
        heights = [pv if pv is not None else 0.0, rv if rv is not None else 0.0]
        bars = ax.bar([0, 1], heights, width=0.7, color=[PERSONAL, REFERENCE],
                      edgecolor="none")
        for x, h, v in zip([0, 1], heights, [pv, rv]):
            if v is not None:
                ax.text(x, min(h + 0.012, 0.995), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0.5, ls=":", color="0.45", lw=1.0, zorder=0)   # chance
        ax.set_title(assay, fontsize=10)
        ax.set_ylim(a.ymin, 1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["personal", "reference"] if i // ncols == nrows - 1 else ["", ""],
                           fontsize=8)
        ax.set_yticks(np.arange(a.ymin, 1.001, ystep))
        if i % ncols == 0:
            ax.set_ylabel("AUROC", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if a.title else 1.0])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[wrote] {a.out}  ({len(assays)} assays)")


if __name__ == "__main__":
    main()
