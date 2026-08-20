#!/usr/bin/env python
"""
Build AS/non-AS ASB-window CSVs for head-ISM, one per eval set (ADASTRA, EN-TEx), for a given TF.

Each output row carries the SAME twin windows the model scored (score_asb.build_windows):
  sequence1 = reference/hap1 window  (allele1 substituted at the center of a 257bp hg38 window)
  sequence2 = alt/hap2 window        (allele2 at the center; the twin partner)
  as_label  = "AS" or "nonAS"        (feature column ism_saliency filters on)

So head-ISM runs as:
  ism_saliency --windows_csv <this> --seq_col sequence1 --partner_col sequence2 \
               --feature_col as_label --feature_keep AS|nonAS --twin_baseline ref|alt

ADASTRA schema: chr, pos (1-based), ref, alt, label (1=AS).
EN-TEx schema (mirrors score_asb.load_hetsnv / plot_wave1_enrichment exactly): chr,
  ref_start (0-based), ref_allele, hap1_allele, hap2_allele, assay, cA/cC/cG/cT (depth), and
  imbalance_significance (AS label). Deduped to unique loci (AS if significant in ANY row).

Center substitution matches score_asb.build_windows: window = left_bp + 1 + right_bp, center base
replaced by allele1 (seq1) / allele2 (seq2). CPU-only (pyfaidx); no model needed.
"""
import argparse, os, numpy as np, pandas as pd
from pyfaidx import Fasta

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}


def make_pair(fa, chrom, pos, a1, a2, left_bp, right_bp, pos_is_1based):
    """Return (seq1, seq2) with a1/a2 substituted at the window center, or (None, None)."""
    if chrom not in fa:
        return None, None
    p0 = (int(pos) - 1) if pos_is_1based else int(pos)
    start, end = p0 - left_bp, p0 + right_bp + 1
    if start < 0 or end > len(fa[chrom]):
        return None, None
    seq = str(fa[chrom][start:end]).upper()
    win = left_bp + 1 + right_bp
    if len(seq) != win:
        return None, None
    c = left_bp
    return (seq[:c] + str(a1).upper() + seq[c + 1:],
            seq[:c] + str(a2).upper() + seq[c + 1:])


def build_adastra(fa, csv, left_bp, right_bp):
    df = pd.read_csv(csv)
    df = df[df["chr"].astype(str).str.startswith("chr")]
    rows = []
    for _, r in df.iterrows():
        s1, s2 = make_pair(fa, str(r["chr"]), int(r["pos"]), r["ref"], r["alt"],
                           left_bp, right_bp, pos_is_1based=True)
        if s1 is None:
            continue
        # ADASTRA per-locus read depth = total_cover (reads at the SNV; analogous to EN-TEx's summed
        # base counts). Ranking head-ISM by this gives the best-powered ADASTRA loci, not file order.
        # Fall back to total_reads, then 0.0, if a variant schema lacks total_cover.
        if "total_cover" in df.columns and pd.notna(r.get("total_cover")):
            depth = float(r["total_cover"])
        elif "total_reads" in df.columns and pd.notna(r.get("total_reads")):
            depth = float(r["total_reads"])
        else:
            depth = 0.0
        rows.append((s1, s2, "AS" if int(r["label"]) == 1 else "nonAS", depth))
    return pd.DataFrame(rows, columns=["sequence1", "sequence2", "as_label", "total_reads"])


def build_entex(fa, tsv, assay, min_total_reads, left_bp, right_bp):
    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "assay", "cA", "cC", "cG", "cT", "imbalance_significance"]
    df = pd.read_csv(tsv, sep="\t", usecols=lambda c: c in usecols)
    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    df = df.reset_index(drop=True)
    df["chr"] = df["chr"].astype(str)

    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if (col in df.columns and pd.notna(row[col])) else 0.0
    df["total_reads"] = df.apply(
        lambda r: base_count(r, "hap1_allele") + base_count(r, "hap2_allele"), axis=1)
    if min_total_reads:
        n0 = len(df)
        df = df[df["total_reads"] >= min_total_reads].reset_index(drop=True)
        print(f"[entex] total_reads>={min_total_reads}: {len(df)}/{n0} rows kept")

    # dedup to unique loci: AS if significant in ANY row; keep first alleles + max depth at locus
    g = df.groupby(["chr", "ref_start"]).agg(
        ref_allele=("ref_allele", "first"),
        hap1_allele=("hap1_allele", "first"),
        hap2_allele=("hap2_allele", "first"),
        total_reads=("total_reads", "max"),
        imbalance_significance=("imbalance_significance", "max")).reset_index()
    rows = []
    for _, r in g.iterrows():
        s1, s2 = make_pair(fa, str(r["chr"]), int(r["ref_start"]),
                           r["hap1_allele"], r["hap2_allele"], left_bp, right_bp,
                           pos_is_1based=False)
        if s1 is None:
            continue
        rows.append((s1, s2, "AS" if int(r["imbalance_significance"]) == 1 else "nonAS",
                     float(r["total_reads"])))
    return pd.DataFrame(rows, columns=["sequence1", "sequence2", "as_label", "total_reads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--tf_label", required=True, help="e.g. ctcf / ep300 (for output filenames)")
    ap.add_argument("--adastra_csv", default=None)
    ap.add_argument("--hetsnv_tsv", default=None)
    ap.add_argument("--assay", default=None, help="EN-TEx assay filter, e.g. CTCF / EP300")
    ap.add_argument("--min_total_reads", type=float, default=0)
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    fa = Fasta(a.ref_fasta, sequence_always_upper=True)
    tf = a.tf_label.lower()

    if a.adastra_csv:
        d = build_adastra(fa, a.adastra_csv, a.left_bp, a.right_bp)
        out = os.path.join(a.outdir, f"{tf}_asb_ism_windows_adastra.csv")
        d.to_csv(out, index=False)
        vc = d["as_label"].value_counts().to_dict()
        print(f"[adastra] wrote {out}: {len(d)} loci {vc}")

    if a.hetsnv_tsv:
        d = build_entex(fa, a.hetsnv_tsv, a.assay, a.min_total_reads, a.left_bp, a.right_bp)
        out = os.path.join(a.outdir, f"{tf}_asb_ism_windows_entex.csv")
        d.to_csv(out, index=False)
        vc = d["as_label"].value_counts().to_dict()
        print(f"[entex] wrote {out}: {len(d)} loci {vc}")


if __name__ == "__main__":
    main()