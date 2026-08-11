#!/usr/bin/env python3
"""
build_betabinom_counts.py — aggregate EN-TEx hetSNV read counts per (donor, locus)
into a beta-binomial training table.

The hetSNV TSV has one row per (donor, tissue, locus). For the beta-binomial count
model we want ONE observation per unique (donor, haplotype-sequence) locus, with reads
SUMMED across tissues (tissue-agnostic, matching ADASTRA pooling and the Track-1
consensus model). This script produces that table; the finetune row-source reads it
like any other SNV table.

Emitted columns (one row per donor-locus):
  chr, ref_start, ref_allele, hap1_allele, hap2_allele, donor,
  k (= summed hap1 reads), n (= summed total reads), imbalance_significance (any-sig),
  assay
The finetune betabinomial task reads k and n as its label pair; chr/ref_start drive the
100kb-bin leakage split and the held-out-chromosome test.

  python build_betabinom_counts.py \
      --hetsnv_tsv /home/asm242/palmer_scratch/entex_data/hetSNVs.tsv \
      --assay CTCF --out ctcf_betabinom_counts.csv
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
    ap.add_argument("--out", default="betabinom_counts.csv")
    a = ap.parse_args()

    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "tissue", "assay", "cA", "cC", "cG", "cT",
               "imbalance_significance"]
    df = pd.read_csv(a.hetsnv_tsv, sep="\t", usecols=lambda c: c in usecols)
    n_rows = len(df)
    if a.assay and a.assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(a.assay, case=False, na=False)]
    df = df.reset_index(drop=True)

    # per-ROW hap counts from the per-base columns (the base each haplotype carries)
    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if col in row and pd.notna(row[col]) else 0.0
    df["hap1_count"] = df.apply(lambda r: base_count(r, "hap1_allele"), axis=1)
    df["hap2_count"] = df.apply(lambda r: base_count(r, "hap2_allele"), axis=1)
    df["imbalance_significance"] = df["imbalance_significance"].astype(int)

    # AGGREGATE across tissues within (donor, locus): sum reads, OR the significance call.
    key = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele", "donor"]
    agg = (df.groupby(key, observed=True)
             .agg(k=("hap1_count", "sum"),
                  hap2_reads=("hap2_count", "sum"),
                  imbalance_significance=("imbalance_significance", "max"),
                  n_tissues=("tissue", "nunique"))
             .reset_index())
    agg["k"] = agg["k"].round().astype(int)
    agg["n"] = (agg["k"] + agg["hap2_reads"].round().astype(int)).astype(int)
    agg = agg.drop(columns="hap2_reads")
    agg["assay"] = a.assay

    n0 = len(agg)
    agg = agg[agg["n"] >= a.min_total_reads].reset_index(drop=True)
    cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
            "donor", "k", "n", "imbalance_significance", "assay", "n_tissues"]
    agg[cols].to_csv(a.out, index=False)

    print(f"[in]  {n_rows} rows (assay={a.assay})")
    print(f"[agg] {n0} donor-loci -> {len(agg)} kept (n>={a.min_total_reads}; dropped {n0-len(agg)})")
    print(f"[pos] imbalance_significant loci: {int(agg.imbalance_significance.sum())} "
          f"({agg.imbalance_significance.mean()*100:.1f}%)")
    print(f"[k/n] median n={int(agg.n.median())}, median k/n={ (agg.k/agg.n).median():.3f}")
    print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()
