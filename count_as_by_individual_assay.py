#!/usr/bin/env python3
"""
Count allele-specific (AS) positive vs negative rows per individual (donor) per assay
in the raw EN-TEx hetSNVs TSV.

Positive = imbalance_significance == 1 (the binary AS label your pipeline uses via
make_as_class_label_spec). Reads the (large) TSV in chunks.

Optionally also reports counts AFTER a --min_total_reads filter, replicating the pipeline's
read-depth filter (total_reads = c<hap1_allele> + c<hap2_allele> from the cA/cC/cG/cT columns),
since the post-filter prevalence -- not the raw prevalence -- is what your training set sees.

This tells you the class balance per (donor, assay), which decides the train/dev/test
balancing strategy and whether class weighting is needed.
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", required=True, help="Path to the raw hetSNVs TSV.")
    p.add_argument("--out_csv", default=None, help="Where to write the per-(donor,assay) table.")
    p.add_argument("--label_col", default="imbalance_significance")
    p.add_argument("--min_total_reads", type=int, default=None,
                   help="If set, ALSO report counts after this read-depth filter (matches the pipeline).")
    p.add_argument("--chunksize", type=int, default=200000)
    return p.parse_args()


def hap_read_counts(df, allele_col):
    """Vectorized: for each row, the cA/cC/cG/cT count matching the allele in allele_col."""
    out = np.zeros(len(df), dtype=float)
    alleles = df[allele_col].astype(str).str.upper().to_numpy()
    for b in BASES:
        col = f"c{b}"
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} in TSV for read-depth filtering.")
        m = alleles == b
        if m.any():
            out[m] = pd.to_numeric(df.loc[m, col], errors="coerce").to_numpy()
    return out


def accumulate(acc, sub, label_col):
    g = sub.groupby(["donor", "assay", label_col]).size()
    for (donor, assay, lv), n in g.items():
        acc[(str(donor), str(assay))][int(lv)] += int(n)


def to_table(acc):
    label_vals = sorted({lv for d in acc.values() for lv in d})
    rows = []
    for (donor, assay), counts in sorted(acc.items()):
        total = sum(counts.values())
        n_pos = counts.get(1, 0)
        n_neg = counts.get(0, 0)
        row = {"donor": donor, "assay": assay,
               "n_negative": n_neg, "n_positive": n_pos, "n_total": total,
               "frac_positive": (n_pos / total) if total else np.nan}
        for lv in label_vals:
            if lv not in (0, 1):
                row[f"n_label_{lv}"] = counts.get(lv, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    usecols = ["donor", "assay", args.label_col]
    if args.min_total_reads is not None:
        usecols += ["hap1_allele", "hap2_allele"] + [f"c{b}" for b in BASES]

    raw = defaultdict(lambda: defaultdict(int))
    filt = defaultdict(lambda: defaultdict(int)) if args.min_total_reads is not None else None

    n_chunks = 0
    for chunk in pd.read_csv(args.tsv, sep="\t", usecols=usecols, chunksize=args.chunksize):
        n_chunks += 1
        chunk = chunk.dropna(subset=[args.label_col])
        chunk[args.label_col] = pd.to_numeric(chunk[args.label_col], errors="coerce")
        chunk = chunk.dropna(subset=[args.label_col])
        accumulate(raw, chunk, args.label_col)
        if filt is not None:
            tot = hap_read_counts(chunk, "hap1_allele") + hap_read_counts(chunk, "hap2_allele")
            accumulate(filt, chunk[tot >= args.min_total_reads], args.label_col)
    print(f"Read {n_chunks} chunk(s).")

    raw_tbl = to_table(raw)
    other = [c for c in raw_tbl.columns if c.startswith("n_label_")]
    if other:
        print(f"WARNING: label values other than 0/1 present ({other}); positive is defined as ==1.")

    print("\n=== RAW counts per (donor, assay) ===")
    print(raw_tbl.to_string(index=False))
    print(f"\nOverall raw positive fraction: {raw_tbl['n_positive'].sum() / raw_tbl['n_total'].sum():.4f}")
    print(f"frac_positive range across (donor,assay): "
          f"{raw_tbl['frac_positive'].min():.4f} - {raw_tbl['frac_positive'].max():.4f}")

    out = raw_tbl
    if filt is not None:
        filt_tbl = to_table(filt).rename(columns={
            "n_negative": "n_negative_filt", "n_positive": "n_positive_filt",
            "n_total": "n_total_filt", "frac_positive": "frac_positive_filt"})
        out = raw_tbl.merge(filt_tbl[["donor", "assay", "n_negative_filt", "n_positive_filt",
                                      "n_total_filt", "frac_positive_filt"]],
                            on=["donor", "assay"], how="left")
        print(f"\n=== AFTER min_total_reads >= {args.min_total_reads} (matches pipeline) ===")
        print(filt_tbl.to_string(index=False))
        denom = out["n_total_filt"].sum()
        if denom:
            print(f"\nOverall filtered positive fraction: {out['n_positive_filt'].sum() / denom:.4f}")
            print(f"filtered frac_positive range: "
                  f"{out['frac_positive_filt'].min():.4f} - {out['frac_positive_filt'].max():.4f}")

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
