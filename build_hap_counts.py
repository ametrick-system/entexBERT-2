#!/usr/bin/env python3
"""
build_hap_counts.py — aggregate EN-TEx hetSNV read counts per (donor, locus) via tissue pooling
into a per-locus haplotype-count table for the Stage-2 ASB head.

Emitted columns (one row per donor-locus):
  chr, ref_start, ref_allele, hap1_allele, hap2_allele, donor,
  k (= summed hap1 reads), n (= summed total reads), imbalance_significance (any-sig),
  assay

The Stage-2 ASB head reads this table via the hap_counts row source: n supplies the per-locus
depth (privileged precision weight n_eff) and imbalance_significance the binary AS label;
chr/ref_start drive the 100kb-bin leakage split and the held-out-chromosome test

To run:
  python build_hap_counts.py \
      --hetsnv_tsv /path/to/hetsnv_tsv \
      --assay assay_name --out /path/to/out.csv
"""
import argparse
import numpy as np, pandas as pd

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hetsnv_tsv", required=True)
    ap.add_argument("--assay", default="CTCF", help="assay substring filter; 'ALL' keeps all")
    ap.add_argument("--min_total_reads", type=int, default=1,
                    help="drop aggregated loci with n < this (n=0 has no likelihood)")
    ap.add_argument("--out", default="hap_counts.csv")
    ap.add_argument("--per_tissue_out", default=None,
                    help="optional: also write a per-(donor,tissue,locus) companion table "
                         "(NOT a training input; for later tissue-resolved figures)")
    a = ap.parse_args()

    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "tissue", "assay", "cA", "cC", "cG", "cT",
               "imbalance_significance"]
               
    # Chunked read + per-chunk assay filter
    n_rows = 0
    keep = []
    for ch in pd.read_csv(a.hetsnv_tsv, sep="\t", usecols=lambda c: c in usecols, chunksize=1_000_000):
        n_rows += len(ch)
        if a.assay and a.assay.upper() != "ALL":
            ch = ch[ch["assay"].astype(str).str.contains(a.assay, case=False, na=False)]
        if len(ch):
            keep.append(ch)
    df = (pd.concat(keep, ignore_index=True) if keep
          else pd.DataFrame(columns=usecols)).reset_index(drop=True)

    # per-row hap counts from the per-base columns (the base each haplotype carries)
    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if col in row and pd.notna(row[col]) else 0.0
    df["hap1_count"] = df.apply(lambda r: base_count(r, "hap1_allele"), axis=1)
    df["hap2_count"] = df.apply(lambda r: base_count(r, "hap2_allele"), axis=1)
    df["imbalance_significance"] = df["imbalance_significance"].astype(int)

    # per-row reads in this tissue, counted only when this tissue itself called AS
    df["as_reads"] = (df["hap1_count"] + df["hap2_count"]) * (df["imbalance_significance"] == 1)

    # tissue pool to form aggregated (donor, locus) entries
    key = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele", "donor"]
    agg = (df.groupby(key, observed=True)
             .agg(k=("hap1_count", "sum"),
                  hap2_reads=("hap2_count", "sum"),
                  imbalance_significance=("imbalance_significance", "max"),
                  n_as_reads=("as_reads", "sum"),
                  n_tissues=("tissue", "nunique"))
             .reset_index())
    agg["k"] = agg["k"].round().astype(int)
    agg["n"] = (agg["k"] + agg["hap2_reads"].round().astype(int)).astype(int)
    agg = agg.drop(columns="hap2_reads")
    agg["assay"] = a.assay

    n0 = len(agg)
    agg = agg[agg["n"] >= a.min_total_reads].reset_index(drop=True)
    agg["total_reads"] = agg["n"]
    agg["ref_allele_ratio"] = np.where(agg["n"] > 0, agg["k"] / agg["n"], np.nan)
    agg["n_as_reads"] = agg["n_as_reads"].round().astype(int)
    # weight_depth = depth of the evidence for the label: AS-calling-tissue reads for positives, total pooled reads for negatives
    agg["weight_depth"] = np.where(agg["imbalance_significance"] == 1, agg["n_as_reads"], agg["n"])
    cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
            "donor", "k", "n", "total_reads", "ref_allele_ratio", "n_as_reads", "weight_depth",
            "imbalance_significance", "assay", "n_tissues"]
    agg[cols].to_csv(a.out, index=False)

    print(f"[in]  {n_rows} rows (assay={a.assay})")
    print(f"[agg] {n0} donor-loci -> {len(agg)} kept (n>={a.min_total_reads}; dropped {n0-len(agg)})")
    print(f"[pos] imbalance_significant loci: {int(agg.imbalance_significance.sum())} "
          f"({agg.imbalance_significance.mean()*100:.1f}%)")
    print(f"[k/n] median n={int(agg.n.median())}, median k/n={ (agg.k/agg.n).median():.3f}")
    print(f"[wrote] {a.out}")

    # Optional tissue-resolved table for figures
    # One row per (donor, tissue, locus) with per-tissue k/n/ratio/call, joinable to the pooled table by (chr, ref_start, donor)
    if a.per_tissue_out:
        pt = df.copy()
        pt["k"] = pt["hap1_count"].round().astype(int)
        pt["n"] = (pt["k"] + pt["hap2_count"].round().astype(int)).astype(int)
        pt = pt[pt["n"] >= a.min_total_reads].reset_index(drop=True)
        pt["total_reads"] = pt["n"]
        pt["ref_allele_ratio"] = np.where(pt["n"] > 0, pt["k"] / pt["n"], np.nan)
        pt["assay"] = a.assay
        pt_cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
                   "donor", "tissue", "assay", "k", "n", "total_reads",
                   "ref_allele_ratio", "imbalance_significance"]
        pt[pt_cols].to_csv(a.per_tissue_out, index=False)
        print(f"[per-tissue] wrote {a.per_tissue_out}: {len(pt)} donor-tissue-loci "
              f"({pt['tissue'].nunique()} tissues)")

if __name__ == "__main__":
    main()
