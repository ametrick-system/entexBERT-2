#!/usr/bin/env python
"""
Aggregate saliency x target-motif density over confusion classes (TP / FP / FN / TN).

Groups loci by the model's prediction correctness and, per class, overlays the MEAN ISM saliency
(smoothed) on the target-motif density (fraction of the class's windows with a target-motif hit
covering each position). Answers: does the model's saliency sit on the motif when it is RIGHT (TP)
vs when it MISSES (FN) / FALSE-ALARMS (FP)?

Inputs are ISM npz files (ism_saliency.py). pos_npz = positives (label 1: peaks, or AS loci);
neg_npz = negatives (label 0: background, or non-AS). Prediction score = --score_key (base_score);
a threshold (auto or fixed) splits each label into correct/incorrect. With only pos_npz you get the
TP vs FN half (both are true positives, split by predicted score).

  python plot_aggregate_saliency_motif.py --pos_npz ism_trunk_peak.npz --neg_npz ism_trunk_background.npz \
    --jaspar jaspar2024_core_vert_pfms.json --motif_query CTCF --out fig_agg.png
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_agg")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

B = {"A": 0, "C": 1, "G": 2, "T": 3}


def onehot_batch(seqs):
    N = len(seqs); L = len(str(seqs[0]))
    X = np.zeros((N, 4, L), dtype=np.float32)
    for n, s in enumerate(seqs):
        for i, ch in enumerate(str(s).upper()):
            if ch in B:
                X[n, B[ch], i] = 1.0
    return X


def target_pwms(jaspar_json, query, pseudo=0.1):
    J = json.load(open(jaspar_json)); out = {}
    for mid, v in J.items():
        if query.upper() == "ALL" or query.upper() in v.get("name", mid).upper():
            p = v["pfm"]; M = np.array([p["A"], p["C"], p["G"], p["T"]], dtype=np.float64) + pseudo
            out[mid] = M / M.sum(0, keepdims=True)
    return out


def coverage(seqs, motifs, thr, count=False):
    from memelite import fimo
    X = onehot_batch(seqs); N, _, L = X.shape
    hits = fimo(motifs, X, threshold=thr)
    cov = np.zeros((N, L), dtype=np.float32)
    for h in hits:
        for _, r in h.iterrows():
            w, s, e = int(r["sequence_name"]), int(r.start), int(r.end)
            if count:
                cov[w, s:e] += 1.0    # motif-hit COUNT (all-motif track, mirrors the per-locus figure)
            else:
                cov[w, s:e] = 1.0     # indicator: window covered by the target motif
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_npz", required=True)
    ap.add_argument("--neg_npz", default=None)
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--motif_query", default="CTCF", help="substring of TF name for the density track")
    ap.add_argument("--score_key", default="base_score", help="per-window prediction score in the npz")
    ap.add_argument("--sal_key", default="importance")
    ap.add_argument("--threshold", default="auto", help="'auto' or a float on the prediction score")
    ap.add_argument("--by_label", action="store_true",
                    help="skip confusion split; show pos vs neg panels (use when no per-locus prediction score exists)")
    ap.add_argument("--pos_label", default="positives")
    ap.add_argument("--neg_label", default="negatives")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=float, default=3.0, help="Gaussian sigma (bp) for the saliency line")
    ap.add_argument("--motif_smooth", type=float, default=0.0, help="Gaussian sigma (bp) for the gray motif-density track; 0 = stepped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--score_csv", default=None,
                    help="perVariant CSV with a per-locus prediction; joined to the npz by chr+anchor to drive "
                         "the confusion split (requires a coord-carrying npz). Overrides --score_key.")
    ap.add_argument("--score_col", default="abs_delta", help="prediction column in --score_csv")
    ap.add_argument("--donor_kind", default="matched", help="filter --score_csv to this donor_kind (or 'all')")
    a = ap.parse_args()

    def load(fn, lab):
        z = np.load(fn, allow_pickle=True)
        n = len(z["seqs"])
        chrom = z["chr"].astype(str) if "chr" in z else np.array(["?"] * n)
        anchor = z["anchor"].astype(np.int64) if "anchor" in z else np.full(n, -1, np.int64)
        return dict(sal=z[a.sal_key].astype(float), seqs=z["seqs"],
                    pred=z[a.score_key].astype(float), label=np.full(n, lab), chrom=chrom, anchor=anchor)
    P = load(a.pos_npz, 1)
    data = [P] + ([load(a.neg_npz, 0)] if a.neg_npz else [])
    sal = np.concatenate([d["sal"] for d in data]); seqs = np.concatenate([d["seqs"] for d in data])
    pred = np.concatenate([d["pred"] for d in data]); label = np.concatenate([d["label"] for d in data])
    chrom = np.concatenate([d["chrom"] for d in data]); anchor = np.concatenate([d["anchor"] for d in data])

    # optional: join an external per-locus prediction (perVariant) by chr+anchor -> real confusion split.
    # abs_delta is sequence-only, so per (chr,ref_start) it is tissue-invariant within one donor_kind.
    if a.score_csv:
        if (anchor < 0).all():
            raise SystemExit("--score_csv given but the npz has no 'anchor' coordinate; re-run ism_saliency with "
                             "the coord-carrying save (chr/anchor) so windows can be joined to the scores.")
        sc = pd.read_csv(a.score_csv)
        if a.donor_kind != "all" and "donor_kind" in sc.columns:
            sc = sc[sc["donor_kind"] == a.donor_kind]
        key = sc.groupby(["chr", "ref_start"])[a.score_col].median()
        pred = np.array([key.get((c, int(an)), np.nan) for c, an in zip(chrom, anchor)], dtype=float)
        keep = ~np.isnan(pred)
        print(f"[score_csv] joined {int(keep.sum())}/{len(pred)} windows to {a.score_col} "
              f"(donor_kind={a.donor_kind}); dropping {int((~keep).sum())} unmatched")
        sal, seqs, label, pred = sal[keep], seqs[keep], label[keep], pred[keep]
    L = sal.shape[1]; c = L // 2; pos = np.arange(L) - c

    label_mode = a.by_label or float(np.std(pred)) < 1e-6   # no usable per-locus prediction -> label stratification
    thr_v = float("nan")
    motifs = target_pwms(a.jaspar, a.motif_query)
    cov = coverage(seqs, motifs, a.fimo_threshold, count=(a.motif_query.upper() == "ALL"))  # count if all-motif track

    if label_mode:
        classes = {a.pos_label: label == 1}
        if a.neg_npz:
            classes[a.neg_label] = label == 0
    else:
        if a.threshold == "auto":
            if a.neg_npz:                               # Youden J on the ROC
                from sklearn.metrics import roc_curve
                fpr, tpr, thr = roc_curve(label, pred); thr_v = float(thr[np.argmax(tpr - fpr)])
            else:
                thr_v = float(np.median(pred[label == 1]))
        else:
            thr_v = float(a.threshold)
        hi = pred >= thr_v
        classes = {"TP": (label == 1) & hi, "FN": (label == 1) & ~hi}
        if a.neg_npz:
            classes.update({"FP": (label == 0) & hi, "TN": (label == 0) & ~hi})

    # per-class mean saliency (smoothed) + motif density
    prof = {}
    for k, m in classes.items():
        if m.sum() == 0:
            prof[k] = (np.zeros(L), np.zeros(L), 0); continue
        ms = sal[m].mean(0); ms = gaussian_filter1d(ms, a.smooth) if a.smooth > 0 else ms
        dens = cov[m].mean(0); dens = gaussian_filter1d(dens, a.motif_smooth) if a.motif_smooth > 0 else dens
        prof[k] = (ms, dens, int(m.sum()))
    smax = max((p[0].max() for p in prof.values() if p[2]), default=1)
    smin = min((p[0].min() for p in prof.values() if p[2]), default=0)
    dmax = max((p[1].max() for p in prof.values() if p[2]), default=1) or 1

    if label_mode:
        order = [a.pos_label] + ([a.neg_label] if a.neg_npz else []); nr = 1
    else:
        order = ["TP", "FN", "FP", "TN"] if a.neg_npz else ["TP", "FN"]
        nr = 2 if a.neg_npz else 1
    fig, axes = plt.subplots(nr, 2, figsize=(10, 3.0 * nr), squeeze=False)
    axes = axes.ravel()
    for ax, k in zip(axes, order):
        ms, dens, n = prof[k]
        ax.fill_between(pos, dens, step=(None if a.motif_smooth > 0 else "mid"), color="0.82", lw=0, zorder=1)
        ax.set_ylim(0, dmax * 1.15); ax.set_xlim(pos[0], pos[-1])
        ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=0)
        _ylab = "mean motif count" if a.motif_query.upper() == "ALL" else f"{a.motif_query} density"
        ax.set_ylabel(_ylab, color="0.45", fontsize=9); ax.tick_params(axis="y", colors="0.45")
        axr = ax.twinx()
        axr.plot(pos, ms, color="#2c6fbb", lw=2.0, zorder=3)
        axr.set_ylim(smin - 0.02 * abs(smin), smax * 1.1); axr.set_ylabel("mean ISM saliency", color="#2c6fbb", fontsize=9)
        axr.tick_params(axis="y", colors="#2c6fbb")
        ax.set_title(f"{k}  (n={n})", fontsize=11, fontweight="bold")
        for s in ("top",): ax.spines[s].set_visible(False); axr.spines[s].set_visible(False)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(order) - 2):len(order)]:
        ax.set_xlabel("position relative to summit/variant (bp)")
    if a.title:
        fig.suptitle(a.title, fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    cnts = {k: prof[k][2] for k in order}
    split = "label-mode (no prediction)" if label_mode else f"thr={thr_v:.3f} on {a.score_key}"
    print(f"[wrote] {a.out} | {split} | classes={cnts} | motif='{a.motif_query}' ({len(motifs)} PWMs)")


if __name__ == "__main__":
    main()
