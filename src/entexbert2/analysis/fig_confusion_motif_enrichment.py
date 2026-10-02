#!/usr/bin/env python
"""fig_confusion_motif_enrichment.py -- per-position enrichment of ONE motif, split by the head's
confusion outcome (TP/FP/FN/TN), aggregated over loci and centered on the variant (like the attached
panel). The sharp TP peak at 0 = loci the head correctly called AS carry the motif at the variant.

For a cell (assay, donor, arm) it loads the head ISM npzs (ism_head_<arm>_{AS,nonAS}.npz -> window seq
+ truth as_label + chr + anchor), joins the per-locus prediction ell from the eval perVariant
(score_<arm>_hetsnv_perVariant.csv.gz; predicted-AS = ell > --threshold), assigns each locus to
TP/FP/FN/TN, FIMO-scans the chosen --motif over each locus's window, and sums the per-position motif
coverage within each class -> four 'Matches vs Position' curves. Raw counts by default (a class with
more loci sits higher); --per_locus divides by class size for a per-locus rate.

  python fig_confusion_motif_enrichment.py --exp_root experiments --assay CTCF --donor ENC-001 \
     --arm personal --jaspar jaspar2024_core_vert_pfms.json --motif CTCF --window 128 \
     --title "CTCF enrichment (individual 1, CTCF assay)" --out fig_ctcf_conf_enrich.png
"""
import argparse, os, json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

B = {"A": 0, "C": 1, "G": 2, "T": 3}
# TP blue, FP red, FN green, TN orange (screenshot omits a legend; a deliverable needs one)
CLS_STYLE = {"TP": "#3b6fd4", "FP": "#d1495b", "FN": "#2e8b57", "TN": "#e08a1e"}
CLASSES = ("TP", "FP", "FN", "TN")


def select_pwm(jaspar, motif, pseudo=0.1):
    J = json.load(open(jaspar)); tok = motif.strip().upper()
    for mid, v in J.items():
        name = v.get("name", mid)
        if tok in name.upper() or tok == mid.upper():
            p = v["pfm"]; M = np.array([p["A"], p["C"], p["G"], p["T"]], float) + pseudo
            return mid, name, (M / M.sum(0, keepdims=True))
    sys.exit(f"--motif {motif!r} matched no JASPAR entry")


def scan_one(seq, mid, pwm, threshold):
    """FIMO one motif over one sequence -> list of (start, end) hit spans (same pattern as the other figs)."""
    from memelite import fimo
    X = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in B:
            X[0, B[ch], i] = 1.0
    spans = []
    for df in fimo({mid: pwm}, X, threshold=threshold):
        for _, r in df.iterrows():
            spans.append((int(r["start"]), int(r["end"])))
    return spans


def load_arm(exp_root, assay, donor, arm):
    cell = os.path.join(exp_root, f"{assay.lower()}_{donor}")
    seqs, asl, chrom, anc = [], [], [], []
    for cls in ("AS", "nonAS"):
        p = os.path.join(cell, "ism", f"ism_head_{arm}_{cls}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True); n = len(z["seqs"])
        seqs += [str(s) for s in z["seqs"]]
        asl += list(z["as_label"].astype(str)) if "as_label" in z else [cls] * n
        chrom += list(z["chr"].astype(str)) if "chr" in z else ["?"] * n
        anc += list(z["anchor"].astype(int)) if "anchor" in z else [-1] * n
    pv = os.path.join(cell, "eval", f"score_{arm}_hetsnv_perVariant.csv.gz")
    return seqs, np.array(asl), np.array(chrom), np.array(anc), pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True); ap.add_argument("--assay", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--arm", default="personal", choices=["ref", "personal"])
    ap.add_argument("--jaspar", required=True); ap.add_argument("--motif", required=True)
    ap.add_argument("--score_col", default="delta", help="perVariant column = the head logit ell")
    ap.add_argument("--threshold", type=float, default=0.0, help="predicted-AS = ell > threshold (ell=0 default)")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--per_locus", action="store_true", help="divide each class curve by its #loci (rate, not raw count)")
    ap.add_argument("--donor_kind", default="all")
    ap.add_argument("--title", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seqs, asl, chrom, anc, pv = load_arm(a.exp_root, a.assay, a.donor, a.arm)
    if not seqs:
        sys.exit(f"no head ISM npz for {a.assay}/{a.donor} {a.arm}")
    if not os.path.exists(pv):
        sys.exit(f"no perVariant at {pv} (run plot_ism/eval first)")
    sc = pd.read_csv(pv)
    if a.donor_kind != "all" and "donor_kind" in sc.columns:
        sc = sc[sc["donor_kind"] == a.donor_kind]
    key = sc.groupby(["chr", "ref_start"])[a.score_col].mean()
    keyd = {(str(c), int(p)): v for (c, p), v in key.items()}
    pred = np.array([keyd.get((c, int(an)), np.nan) for c, an in zip(chrom, anc)])
    hit = np.isfinite(pred); is_as = (asl == "AS"); pas = pred > a.threshold
    cls_of = np.full(len(seqs), "", dtype=object)
    cls_of[hit & is_as & pas] = "TP"; cls_of[hit & is_as & ~pas] = "FN"
    cls_of[hit & ~is_as & pas] = "FP"; cls_of[hit & ~is_as & ~pas] = "TN"
    counts = {c: int((cls_of == c).sum()) for c in CLASSES}
    print(f"[join] {int(hit.sum())}/{len(seqs)} loci matched perVariant (hit-rate {hit.mean():.2f}) | classes {counts}")
    if hit.sum() == 0:
        sys.exit("no loci joined the perVariant (npz vs perVariant locus-set mismatch -- use the personal arm "
                 "or rebuild the ref ISM with the donor-filter fix)")

    mid, name, pwm = select_pwm(a.jaspar, a.motif)
    W = a.window; grid = np.arange(-W, W + 1)
    acc = {c: np.zeros(len(grid)) for c in CLASSES}
    for i, seq in enumerate(seqs):
        c = cls_of[i]
        if c == "":
            continue
        L = len(seq); center = L // 2
        cov = np.zeros(L, dtype=bool)                          # per-locus coverage: cap each locus at 1/position
        for (s, e) in scan_one(seq, mid, pwm, a.fimo_threshold):
            cov[max(0, s):min(e, L)] = True                    # (overlapping hits / both strands don't double-count)
        for p in np.nonzero(cov)[0]:
            off = p - center
            if -W <= off <= W:
                acc[c][off + W] += 1.0
    if a.per_locus:
        for c in CLASSES:
            if counts[c]:
                acc[c] /= counts[c]

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for c in CLASSES:
        y = gaussian_filter1d(acc[c], a.smooth) if a.smooth else acc[c]
        ax.plot(grid, y, color=CLS_STYLE[c], lw=1.5, label=f"{c} (n={counts[c]})")
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("Position")
    ax.set_ylabel("Fraction of loci with motif" if a.per_locus else "Loci with motif")
    ax.set_title(a.title or f"{name} enrichment ({a.donor}, {a.assay} assay)", fontsize=10)
    ax.set_xlim(-W, W); ax.legend(frameon=False, fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {a.out}  (motif {name}, arm {a.arm})")


if __name__ == "__main__":
    main()
