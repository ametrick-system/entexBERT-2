#!/usr/bin/env python
"""
Wave-1 motif-enrichment figures (no GPU). Produces, for CTCF and EP300:
  E1  peak-vs-background motif enrichment in Stage-1 BINDING windows (EN-TEx-derived; the built
      Stage-1 CSVs already carry the sequence + feature_type/label, so NO FASTA extraction needed).
  E2  AS-vs-non-AS motif enrichment around SNVs, for BOTH label sources:
        - ADASTRA  (evalset CSV, 'label' column)
        - EN-TEx   (hetSNVs.tsv, 'imbalance_significance', filtered by assay)
      E2 windows are +/-flank hg38 REFERENCE sequence centered on each SNV (pyfaidx extraction).

Cofactor panels: CTCF -> {CTCF,SP1,KLF4,EGR1};  EP300 -> {JUN,FOS,CEBPB}  (EP300 has no self-motif;
these are the enhancer recruiters). Groups are balanced (equal N, seed 0) so densities compare.
Local FIMO via memelite; needs NUMBA_CACHE_DIR set (done at import).
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASES = "ACGT"; IDX = {b: i for i, b in enumerate(BASES)}
CTCF_TFS  = ["CTCF", "SP1", "KLF4", "EGR1"]
EP300_TFS = ["JUN", "FOS", "CEBPB", "HNF4A"]   # AP-1(JUN/FOS)+CEBPB recruiters + HNF4A lineage TF (endoderm/digestive enhancers)


def load_motifs(path, tfs):
    db = json.load(open(path)); want = {t.upper() for t in tfs}; out = {}
    for k, v in db.items():
        nm = v.get("name", k)
        if nm.upper() in want:
            out[nm] = np.array([v["pfm"][b] for b in BASES], dtype=np.float64)
    return out


def onehot_stack(seqs):
    L = len(seqs[0]); N = len(seqs)
    oh = np.zeros((N, 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            k = IDX.get(b)
            if k is not None:
                oh[i, k, j] = 1.0
    return oh


def density(seqs, motifs, fimo, threshold):
    L = len(seqs[0]); N = len(seqs)
    res = fimo(motifs, onehot_stack(seqs), threshold=threshold)
    cover = {nm: np.zeros(L) for nm in motifs}
    for r in res:
        if len(r) == 0:
            continue
        nm = str(r["motif_name"].iloc[0])
        for _, row in r.iterrows():
            s, e = int(row["start"]), int(min(row["end"], L))
            cover[nm][s:e] += 1
    return {nm: cover[nm] / N for nm in motifs}


def smooth(x, w):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


def balance(a, b, seed=0):
    rng = np.random.default_rng(seed); n = min(len(a), len(b))
    ai = rng.choice(len(a), n, replace=False); bi = rng.choice(len(b), n, replace=False)
    return [a[i] for i in ai], [b[i] for i in bi]


def revcomp(s):
    c = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
    return "".join(c.get(b, "N") for b in reversed(s))


def extract_windows(fa, chrom, pos, flank, one_based=True):
    """+/-flank hg38 reference window centered on pos. one_based=True: pos is 1-based (ADASTRA
    'pos'); one_based=False: pos is 0-based (hetSNV 'ref_start', matches score_asb.build_windows).
    Returns uppercased ACGT str or None."""
    L = 2 * flank + 1
    p0 = (pos - 1) if one_based else pos          # -> 0-based coordinate of the SNV base
    try:
        seq = fa[chrom][p0 - flank: p0 + flank + 1].seq.upper()
    except Exception:
        return None
    return seq if len(seq) == L else None


def two_group_panel(groupA, groupB, labelA, labelB, tfs, motifs_path, fimo, flank,
                    threshold, smooth_w, title, out):
    motifs = load_motifs(motifs_path, tfs)
    order = [nm for nm in tfs if nm in motifs]     # keep requested order, only found
    A, B = balance(groupA, groupB)
    L = len(A[0])
    print(f"[{out}] {labelA}={len(A)} {labelB}={len(B)} L={L} motifs={order}")
    dA = density(A, {k: motifs[k] for k in order}, fimo, threshold)
    dB = density(B, {k: motifs[k] for k in order}, fimo, threshold)
    x = np.arange(L) - (L // 2)
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(5.2, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, nm in zip(axes, order):
        ax.plot(x, smooth(dA[nm], smooth_w), color="#1f4e9c", lw=1.8, label=labelA)
        ax.plot(x, smooth(dB[nm], smooth_w), color="#9db8e0", lw=1.4, label=labelB)
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_title(nm, fontsize=11); ax.set_ylabel("motif density"); ax.margins(x=0)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("position relative to center (bp)")
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    c = L // 2
    for nm in order:
        print(f"   {nm:6s} center {labelA}={smooth(dA[nm],smooth_w)[c]:.4f} "
              f"{labelB}={smooth(dB[nm],smooth_w)[c]:.4f}")


def run_e2_adastra(fa, csv, tfs, motifs_path, fimo, a, tf_label, src_label):
    df = pd.read_csv(csv)
    df = df[df["chr"].astype(str).str.startswith("chr")]
    AS, NON = [], []
    for _, r in df.iterrows():
        w = extract_windows(fa, str(r["chr"]), int(r["pos"]), a.flank)
        if w is None:
            continue
        (AS if int(r["label"]) == 1 else NON).append(w)
    two_group_panel(AS, NON, "AS", "non-AS", tfs, motifs_path, fimo, a.flank,
                    a.fimo_threshold, a.smooth,
                    f"{tf_label} AS-SNV motif enrichment ({src_label})",
                    os.path.join(a.outdir, f"fig_E2_{tf_label.lower()}_{src_label.lower()}.png"))


_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}


def run_e2_entex(fa, hetsnv, assay, tfs, motifs_path, fimo, a, tf_label):
    """EN-TEx hetSNVs.tsv. Schema mirrors score_asb.load_hetsnv EXACTLY: position is 'ref_start'
    (0-based), there is NO total_cover column (read depth = allele base counts cA/cC/cG/cT), and the
    AS label is 'imbalance_significance'. Windows use the 0-based path (one_based=False)."""
    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "assay", "cA", "cC", "cG", "cT", "imbalance_significance"]
    df = pd.read_csv(hetsnv, sep="\t", usecols=lambda c: c in usecols)
    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    df = df.reset_index(drop=True)
    df["chr"] = df["chr"].astype(str)

    # depth = hap1_count + hap2_count from the base-count columns (as load_hetsnv does)
    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if (col in df.columns and pd.notna(row[col])) else 0.0
    df["total_reads"] = df.apply(
        lambda r: base_count(r, "hap1_allele") + base_count(r, "hap2_allele"), axis=1)
    if a.min_total_reads:
        n0 = len(df)
        df = df[df["total_reads"] >= a.min_total_reads].reset_index(drop=True)
        print(f"[E2 {tf_label} EN-TEx] total_reads>={a.min_total_reads}: {len(df)}/{n0} rows kept")

    # dedup to unique loci (reported per donor+tissue): AS if significant in ANY row at that locus
    g = df.groupby(["chr", "ref_start"])["imbalance_significance"].max().reset_index()
    AS, NON = [], []
    for _, r in g.iterrows():
        w = extract_windows(fa, str(r["chr"]), int(r["ref_start"]), a.flank, one_based=False)
        if w is None:
            continue
        (AS if int(r["imbalance_significance"]) == 1 else NON).append(w)
    two_group_panel(AS, NON, "AS", "non-AS", tfs, motifs_path, fimo, a.flank,
                    a.fimo_threshold, a.smooth,
                    f"{tf_label} AS-SNV motif enrichment (EN-TEx)",
                    os.path.join(a.outdir, f"fig_E2_{tf_label.lower()}_entex.png"))


def run_e1_binding(stage1_dir, tfs, motifs_path, fimo, a, tf_label):
    """Peak vs background from the built Stage-1 windows (sequence already extracted)."""
    frames = []
    for fn in ["train.csv", "dev.csv", "test.csv"]:
        p = os.path.join(stage1_dir, fn)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        m = os.path.join(stage1_dir, fn.replace(".csv", ".meta.csv"))
        if os.path.exists(m):
            frames[-1] = pd.read_csv(m)     # prefer meta (has feature_type)
    df = pd.concat(frames, ignore_index=True)
    seqcol = "sequence" if "sequence" in df.columns else df.columns[0]
    if "feature_type" in df.columns:
        peak = df[df["feature_type"].astype(str).str.contains("peak", case=False, na=False)][seqcol].tolist()
        bg   = df[df["feature_type"].astype(str).str.contains("bg|background", case=False, regex=True, na=False)][seqcol].tolist()
        split = "feature_type"
    else:
        lab = df["label"].astype(float)
        peak = df[lab > lab.median()][seqcol].tolist()
        bg   = df[lab <= lab.median()][seqcol].tolist()
        split = "label>median"
    peak = [str(s) for s in peak]; bg = [str(s) for s in bg]
    L = len(peak[0]); peak = [s for s in peak if len(s) == L]; bg = [s for s in bg if len(s) == L]
    print(f"[E1 {tf_label}] split={split} peak={len(peak)} bg={len(bg)} L={L}")
    two_group_panel(peak, bg, "peak", "background", tfs, motifs_path, fimo, L // 2,
                    a.fimo_threshold, a.smooth,
                    f"{tf_label} binding-window motif enrichment (peak vs bg)",
                    os.path.join(a.outdir, f"fig_E1_{tf_label.lower()}_binding.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--ctcf_adastra", required=True)
    ap.add_argument("--ep300_adastra", required=True)
    ap.add_argument("--hetsnv", required=True)
    ap.add_argument("--ctcf_stage1", required=True)
    ap.add_argument("--ep300_stage1", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--flank", type=int, default=75)
    ap.add_argument("--min_total_reads", type=int, default=20)
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--only", default="all", help="comma list: e2_ctcf_adastra,e2_ep300_adastra,e2_ctcf_entex,e2_ep300_entex,e1_ctcf,e1_ep300")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    from memelite import fimo
    import pyfaidx
    fa = pyfaidx.Fasta(a.ref_fasta, sequence_always_upper=True)
    todo = set(a.only.split(",")) if a.only != "all" else {
        "e2_ctcf_adastra","e2_ep300_adastra","e2_ctcf_entex","e2_ep300_entex","e1_ctcf","e1_ep300"}

    if "e2_ctcf_adastra" in todo:
        run_e2_adastra(fa, a.ctcf_adastra, CTCF_TFS, a.jaspar, fimo, a, "CTCF", "ADASTRA")
    if "e2_ep300_adastra" in todo:
        run_e2_adastra(fa, a.ep300_adastra, EP300_TFS, a.jaspar, fimo, a, "EP300", "ADASTRA")
    if "e2_ctcf_entex" in todo:
        run_e2_entex(fa, a.hetsnv, "CTCF", CTCF_TFS, a.jaspar, fimo, a, "CTCF")
    if "e2_ep300_entex" in todo:
        run_e2_entex(fa, a.hetsnv, "EP300", EP300_TFS, a.jaspar, fimo, a, "EP300")
    if "e1_ctcf" in todo:
        run_e1_binding(a.ctcf_stage1, CTCF_TFS, a.jaspar, fimo, a, "CTCF")
    if "e1_ep300" in todo:
        run_e1_binding(a.ep300_stage1, EP300_TFS, a.jaspar, fimo, a, "EP300")
    print("[done] figures ->", a.outdir)


if __name__ == "__main__":
    main()
