#!/usr/bin/env python
"""fig_confusion_enrichment_grid.py -- Panel-A GRID: per-position motif enrichment split by the head's
confusion outcome (TP/FP/FN/TN). One ROW per assay, one COLUMN per top-enriched motif -- the aggregate
small-multiples companion to the single-locus panel (fig_ism_motif_example / fig_confusion_motif_enrichment).

Per (assay, motif) cell: over that assay's loci (classified TP/FP/FN/TN by joining the head ISM npz truth
`as_label` to the eval perVariant `ell`), sum the per-position boolean motif coverage within each class
-> four 'Matches vs Position' curves, centered on the variant. Top motifs per assay are auto-selected by
TP central enrichment (raw count), or pass an explicit per-assay list with --motifs_per_assay.

Uses the personal arm by default (donor-clean); the reference arm needs the donor-filter ISM rebuild.

  python fig_confusion_enrichment_grid.py --exp_root experiments --donor ENC-002 --arm personal \
     --jaspar jaspar2024_core_vert_pfms.json --assays ATAC,CTCF,H3K27ac,H3K4me3 --n_motifs 4 \
     --candidates ALL --window 128 --max_loci 400 --out fig_panelA_grid.png
"""
import argparse, os, json, sys
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
B = {"A": 0, "C": 1, "G": 2, "T": 3}
CLS_STYLE = {"TP": "#3b6fd4", "FP": "#d1495b", "FN": "#2e8b57", "TN": "#e08a1e"}
CLASSES = ("TP", "FP", "FN", "TN")


def load_pwms(jaspar, candidates, pseudo=0.1):
    J = json.load(open(jaspar))
    allm = (candidates.strip().upper() == "ALL")
    toks = [] if allm else [t.strip().upper() for t in candidates.split(",") if t.strip()]
    out = {}
    for mid, v in J.items():
        name = v.get("name", mid)
        if allm or any(t in name.upper() or t == mid.upper() for t in toks):
            p = v["pfm"]; M = np.array([p["A"], p["C"], p["G"], p["T"]], float) + pseudo
            out[mid] = (M / M.sum(0, keepdims=True), name)
    if not out:
        sys.exit(f"--candidates {candidates!r} matched no JASPAR entries")
    return out


def scan_all(seq, pwms, threshold):
    """One FIMO pass over ALL pwms for one sequence -> {motif_name: [(start,end), ...]}."""
    from memelite import fimo
    X = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in B:
            X[0, B[ch], i] = 1.0
    items = list(pwms.items())
    res = fimo({mid: pwm for mid, (pwm, _) in items}, X, threshold=threshold)
    out = {}
    for (mid, (pwm, name)), df in zip(items, res):
        if len(df):
            out.setdefault(name, []).extend((int(r["start"]), int(r["end"])) for _, r in df.iterrows())
    return out


def scan_ranked(seq, pwms, threshold):
    """One FIMO pass -> {motif_name: best -log10 p} (same ranking fig_ism_motif_example uses for lanes)."""
    from memelite import fimo
    X = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in B:
            X[0, B[ch], i] = 1.0
    items = list(pwms.items())
    res = fimo({mid: pwm for mid, (pwm, _) in items}, X, threshold=threshold)
    best = {}
    for (mid, (pwm, name)), df in zip(items, res):
        if not len(df):
            continue
        pcol = next((c for c in ("p-value", "p_value", "pvalue", "pval") if c in df.columns), None)
        if pcol is not None:
            val = float(-np.log10(df[pcol].astype(float).clip(lower=1e-300)).max())
        elif "score" in df.columns:
            val = float(df["score"].astype(float).max())
        else:
            val = float(len(df))
        best[name] = max(best.get(name, -1e9), val)
    return best


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


def classify(asl, chrom, anc, pv, score_col, thr, donor_kind):
    sc = pd.read_csv(pv)
    if donor_kind != "all" and "donor_kind" in sc.columns:
        sc = sc[sc["donor_kind"] == donor_kind]
    key = sc.groupby(["chr", "ref_start"])[score_col].mean()
    keyd = {(str(c), int(p)): v for (c, p), v in key.items()}
    pred = np.array([keyd.get((c, int(a)), np.nan) for c, a in zip(chrom, anc)])
    hit = np.isfinite(pred); is_as = (asl == "AS"); pas = pred > thr
    cls = np.full(len(asl), "", dtype=object)
    cls[hit & is_as & pas] = "TP"; cls[hit & is_as & ~pas] = "FN"
    cls[hit & ~is_as & pas] = "FP"; cls[hit & ~is_as & ~pas] = "TN"
    return cls


def aggregate_assay(exp_root, assay, donor, arm, pwms, a, example=None):
    """Return acc[name][cls] -> per-position matches (grid), class counts, and (if --example_locus)
    the ranked top motifs AT that example locus (by -log10 p, matching Panel B)."""
    seqs, asl, chrom, anc, pv = load_arm(exp_root, assay, donor, arm)
    if not seqs or not os.path.exists(pv):
        return None, None, None
    ex_top = None
    if example is not None:
        exc, exa = example
        idxs = [i for i in range(len(seqs)) if str(chrom[i]) == exc and abs(int(anc[i]) - exa) <= 2]
        if idxs:
            best = scan_ranked(seqs[idxs[0]], pwms, a.fimo_threshold)
            ex_top = sorted(best, key=lambda n: -best[n])
        else:
            print(f"  [warn] {assay}: example locus {exc}:{exa} not in npz (anchors near: "
                  f"{sorted(int(x) for x, c in zip(anc, chrom) if str(c) == exc)[:6]})")
    cls_of = classify(asl, chrom, anc, pv, a.score_col, a.threshold, a.donor_kind)
    W = a.window; grid_len = 2 * W + 1
    acc = defaultdict(lambda: {c: np.zeros(grid_len) for c in CLASSES})
    counts = {c: 0 for c in CLASSES}
    per_class_seen = {c: 0 for c in CLASSES}
    order = np.argsort(-np.array([1 if c else 0 for c in cls_of]))  # classified first
    for i in order:
        c = cls_of[i]
        if c == "":
            continue
        if a.max_loci and per_class_seen[c] >= a.max_loci:
            continue
        per_class_seen[c] += 1; counts[c] += 1
        seq = seqs[i]; L = len(seq); center = L // 2
        for name, spans in scan_all(seq, pwms, a.fimo_threshold).items():
            cov = np.zeros(L, dtype=bool)
            for s, e in spans:
                cov[max(0, s):min(e, L)] = True
            for p in np.nonzero(cov)[0]:
                off = p - center
                if -W <= off <= W:
                    acc[name][c][off + W] += 1.0
    return acc, counts, ex_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True); ap.add_argument("--donor", required=True)
    ap.add_argument("--arm", default="personal", choices=["ref", "personal"])
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--assays", default="ATAC,CTCF", help="comma-separated (one row each)")
    ap.add_argument("--n_motifs", type=int, default=4, help="columns per assay")
    ap.add_argument("--candidates", default="ALL", help="JASPAR name substrings/ids to consider, or ALL")
    ap.add_argument("--motifs_per_assay", default=None,
                    help="explicit override, e.g. 'ATAC:ZNF740,SP3;CTCF:CTCF,KLF16' (skips auto-select)")
    ap.add_argument("--example_locus", default=None,
                    help="take each row's column motifs from that assay's Panel-B locus (top --n_motifs by "
                         "-log10 p, matching Panel B). e.g. 'ATAC:chr9:133133861;CTCF:chr17:20320976' "
                         "(chr:anchor; anchor = figure START + 128). Use with --candidates ALL.")
    ap.add_argument("--score_col", default="delta"); ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=128); ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--central_bp", type=int, default=10, help="+/-bp for the TP-enrichment ranking")
    ap.add_argument("--max_loci", type=int, default=400, help="cap loci scanned per class (0 = all)")
    ap.add_argument("--per_locus", action="store_true")
    ap.add_argument("--donor_kind", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    pwms = load_pwms(a.jaspar, a.candidates)
    override = {}
    if a.motifs_per_assay:
        for part in a.motifs_per_assay.split(";"):
            k, v = part.split(":"); override[k.strip().upper()] = [m.strip() for m in v.split(",") if m.strip()]
    examples = {}
    if a.example_locus:
        for part in a.example_locus.split(";"):
            k, rest = part.split(":", 1); ec, ea = rest.split(":")
            examples[k.strip().upper()] = (ec.strip(), int(ea))

    W = a.window; grid = np.arange(-W, W + 1); cb = a.central_bp
    per_assay = {}   # assay -> (acc, counts, chosen[list of names])
    for assay in assays:
        acc, counts, ex_top = aggregate_assay(a.exp_root, assay, a.donor, a.arm, pwms, a,
                                              example=examples.get(assay.upper()))
        if acc is None:
            print(f"[{assay}] no npz/perVariant -- row will be blank"); per_assay[assay] = (None, None, []); continue
        if ex_top:                                             # columns = this assay's Panel-B example motifs
            chosen = ex_top[:a.n_motifs]
        elif assay.upper() in override:
            names = [n for n in acc if any(t.upper() in n.upper() for t in override[assay.upper()])]
            chosen = sorted(names, key=lambda n: -acc[n]["TP"][W - cb:W + cb + 1].mean())[:a.n_motifs]
        else:
            chosen = sorted(acc, key=lambda n: -acc[n]["TP"][W - cb:W + cb + 1].mean())[:a.n_motifs]
        per_assay[assay] = (acc, counts, chosen)
        print(f"[{assay}] classes {counts} | columns: {chosen}")

    nrows, ncols = len(assays), a.n_motifs
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.9 * nrows), squeeze=False)
    for r, assay in enumerate(assays):
        acc, counts, chosen = per_assay[assay]
        axes[r][0].set_ylabel(f"{assay}\nMatches", fontsize=8)
        for cix in range(ncols):
            ax = axes[r][cix]
            if acc is None or cix >= len(chosen):
                ax.axis("off"); continue
            name = chosen[cix]
            for c in CLASSES:
                y = acc[name][c].copy()
                if a.per_locus and counts[c]:
                    y = y / counts[c]
                if a.smooth:
                    y = gaussian_filter1d(y, a.smooth)
                ax.plot(grid, y, color=CLS_STYLE[c], lw=1.1)
            ax.axvline(0, color="0.75", lw=0.7, zorder=0)
            ax.set_title(name, fontsize=8.5)
            ax.set_xlim(-W, W); ax.tick_params(labelsize=6.5)
            if r == nrows - 1:
                ax.set_xlabel("Position", fontsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    handles = [plt.Line2D([0], [0], color=CLS_STYLE[c], lw=1.6) for c in CLASSES]
    fig.legend(handles, list(CLASSES), loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {a.out}  ({nrows} assays x {ncols} motifs, arm={a.arm})")


if __name__ == "__main__":
    main()
