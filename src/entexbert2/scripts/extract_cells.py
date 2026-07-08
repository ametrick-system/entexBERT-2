#!/usr/bin/env python3
"""
extract_cells.py — diagnostic pass over the raw EN-TEx hetSNVs TSV, to be run BEFORE
building any training inputs. It answers the two questions the hybrid genomic-bin partition
needs answered up front:

  1. NaN p_betabinom fraction, per (donor, assay[, tissue]) cell.
     The ref_single baseline EXCLUDES untested sites (rows with NaN p_betabinom) via the
     drop_aux_nan / neg_log10_p_betabinom mechanism in utils.py. This reports exactly how many
     rows that policy drops, so the exclude-vs-keep(p_tested) decision is made with the number
     in hand (per the recorded FLAG A rationale).

  2. Per-chromosome POSITIVE counts, per cell and pooled.
     PartitionSpec.fold_assignment holds out whole chromosome(s) as the TEST set. To pick the
     held-out test chromosome(s) you want a small set of chromosomes that TOGETHER carry ~10%
     of positives (avoiding chr1/chr2, which are too large to hold out cleanly). This prints the
     per-chromosome positive counts and, with --suggest_test_frac, a candidate held-out set +
     a ready-to-paste fold_assignment dict.

Definitions (match the training pipeline exactly):
  * positive          = imbalance_significance == 1   (the binary AS label; make_as_class_label_spec)
  * tested            = p_betabinom is NOT NaN        (drop_aux_nan excludes the untested rows)
  * read-depth filter = total_reads (= c<hap1_allele> + c<hap2_allele> from cA/cC/cG/cT) >= min_total_reads
                        Post-filter counts are what the training set actually sees, so pass
                        --min_total_reads to match your run config.
  * chromosome key    = the 'chr' column; position anchor = 'ref_start' (== the 'SNV' anchor
                        that assign_split_column bins on), so these counts align 1:1 with the split.

Reads the (large) TSV in chunks; no genome file or model needed.

Usage:
  python extract_cells.py --tsv hetSNVs_default_AS.tsv \
      --min_total_reads 10 \
      --out_prefix diag/ctcf \
      --suggest_test_frac 0.10
Optionally restrict to specific cells:
  --donors ENC-002,ENC-003   --assays TF-ChIP-seq_CTCF
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]

# Read only what we need. These names match load_as_table's usecols in utils.py.
USECOLS = ["chr", "ref_start", "donor", "assay", "tissue",
           "hap1_allele", "hap2_allele", "cA", "cC", "cG", "cT",
           "p_betabinom", "imbalance_significance"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", required=True, help="Path to the raw hetSNVs TSV.")
    p.add_argument("--out_prefix", default=None,
                   help="If set, write <prefix>_cells.csv and <prefix>_per_chrom.csv.")
    p.add_argument("--label_col", default="imbalance_significance")
    p.add_argument("--p_col", default="p_betabinom")
    p.add_argument("--min_total_reads", type=int, default=None,
                   help="Apply this read-depth filter (matches the pipeline). "
                        "Counts are reported AFTER it when set.")
    p.add_argument("--group_tissue", action="store_true",
                   help="Split cells by tissue too (default: pool tissues within donor+assay).")
    p.add_argument("--donors", default=None, help="Comma-separated donors to keep (default: all).")
    p.add_argument("--assays", default=None, help="Comma-separated assays to keep (default: all).")
    p.add_argument("--suggest_test_frac", type=float, default=None,
                   help="If set, suggest a held-out TEST chromosome set carrying ~this fraction "
                        "of pooled positives (e.g. 0.10) and print a fold_assignment dict.")
    p.add_argument("--exclude_from_test", default="chr1,chr2,chrX,chrY,chrM,chrMT",
                   help="Chromosomes never suggested for the held-out test set (too big / special).")
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


def main():
    args = parse_args()
    keep_donors = set(s.strip() for s in args.donors.split(",")) if args.donors else None
    keep_assays = set(s.strip() for s in args.assays.split(",")) if args.assays else None
    cell_keys = ["donor", "assay"] + (["tissue"] if args.group_tissue else [])

    # Accumulators (all keyed by the cell tuple).
    # per-cell scalar counts:
    n_total = defaultdict(int)     # rows in cell (post read-depth filter)
    n_pos = defaultdict(int)       # imbalance_significance == 1
    n_nan_p = defaultdict(int)     # p_betabinom is NaN (untested -> excluded by baseline)
    n_pos_tested = defaultdict(int)  # positive AND tested (what the baseline actually trains on)
    # per (cell, chrom) positive/total counts:
    chrom_pos = defaultdict(lambda: defaultdict(int))    # cell -> chrom -> positives (tested)
    chrom_total = defaultdict(lambda: defaultdict(int))  # cell -> chrom -> total (tested)

    label_col, p_col = args.label_col, args.p_col
    n_chunks = 0
    for chunk in pd.read_csv(args.tsv, sep="\t", usecols=USECOLS, chunksize=args.chunksize):
        n_chunks += 1
        if keep_donors is not None:
            chunk = chunk[chunk["donor"].isin(keep_donors)]
        if keep_assays is not None:
            chunk = chunk[chunk["assay"].isin(keep_assays)]
        if chunk.empty:
            continue

        # Read-depth filter (post-filter is what training sees).
        if args.min_total_reads is not None:
            tot = hap_read_counts(chunk, "hap1_allele") + hap_read_counts(chunk, "hap2_allele")
            chunk = chunk[tot >= args.min_total_reads]
            if chunk.empty:
                continue

        lab = pd.to_numeric(chunk[label_col], errors="coerce")
        is_pos = (lab == 1).to_numpy()
        is_nan_p = chunk[p_col].isna().to_numpy()
        is_tested = ~is_nan_p
        chroms = chunk["chr"].astype(str).to_numpy()

        # group once per (cell) via pandas groupby on the cell keys
        gcols = chunk[cell_keys].astype(str)
        cell_series = gcols.agg("|".join, axis=1).to_numpy()

        for cell in np.unique(cell_series):
            m = cell_series == cell
            key = tuple(cell.split("|"))
            n_total[key] += int(m.sum())
            n_pos[key] += int(is_pos[m].sum())
            n_nan_p[key] += int(is_nan_p[m].sum())
            m_pt = m & is_pos & is_tested
            n_pos_tested[key] += int(m_pt.sum())
            # per-chrom counts on the TESTED set (what the exclude-policy baseline keeps)
            mt = m & is_tested
            cp, ct = chrom_pos[key], chrom_total[key]
            ch_t = chroms[mt]
            pos_t = is_pos[mt]
            # vectorized bincount over unique chroms in this slice
            uch, inv = np.unique(ch_t, return_inverse=True)
            tot_by = np.bincount(inv, minlength=len(uch))
            pos_by = np.bincount(inv, weights=pos_t.astype(float), minlength=len(uch))
            for c, tt, pp in zip(uch.tolist(), tot_by.tolist(), pos_by.tolist()):
                ct[c] += int(tt)
                cp[c] += int(pp)

    print(f"Read {n_chunks} chunk(s).")
    if not n_total:
        print("No rows matched the donor/assay filters.")
        return

    # ---- per-cell table ----
    rows = []
    for key in sorted(n_total):
        tot = n_total[key]
        rec = dict(zip(cell_keys, key))
        rec.update({
            "n_total": tot,
            "n_positive": n_pos[key],
            "frac_positive": (n_pos[key] / tot) if tot else np.nan,
            "n_nan_p": n_nan_p[key],
            "frac_nan_p": (n_nan_p[key] / tot) if tot else np.nan,
            "n_positive_tested": n_pos_tested[key],
        })
        rows.append(rec)
    cells = pd.DataFrame(rows)
    print("\n=== per-cell summary "
          f"({'post min_total_reads>=%d' % args.min_total_reads if args.min_total_reads is not None else 'no read-depth filter'}) ===")
    print(cells.to_string(index=False))
    denom = cells["n_total"].sum()
    print(f"\npooled positive fraction: {cells['n_positive'].sum()/denom:.4f}")
    print(f"pooled NaN-p fraction (rows the exclude policy DROPS): {cells['n_nan_p'].sum()/denom:.4f}")
    print(f"frac_nan_p range across cells: {cells['frac_nan_p'].min():.4f} - {cells['frac_nan_p'].max():.4f}")

    # ---- per-chromosome table (pooled over cells, on the TESTED set) ----
    pooled_pos = defaultdict(int)
    pooled_tot = defaultdict(int)
    for key in chrom_pos:
        for c, v in chrom_pos[key].items():
            pooled_pos[c] += v
        for c, v in chrom_total[key].items():
            pooled_tot[c] += v

    def chrom_sort_key(c):
        s = c[3:] if c.lower().startswith("chr") else c
        return (0, int(s)) if s.isdigit() else (1, s)

    ptot_pos = sum(pooled_pos.values())
    prows = []
    for c in sorted(pooled_pos, key=chrom_sort_key):
        prows.append({"chr": c, "n_total_tested": pooled_tot[c], "n_positive_tested": pooled_pos[c],
                      "frac_of_all_positives": pooled_pos[c] / ptot_pos if ptot_pos else np.nan})
    per_chrom = pd.DataFrame(prows)
    print("\n=== per-chromosome positive counts (pooled across cells, TESTED rows only) ===")
    print(per_chrom.to_string(index=False))
    print(f"\ntotal tested positives (pooled): {ptot_pos}")

    # ---- optional held-out test-chromosome suggestion ----
    if args.suggest_test_frac is not None and ptot_pos:
        exclude = set(s.strip() for s in args.exclude_from_test.split(",") if s.strip())
        target = args.suggest_test_frac * ptot_pos
        # greedily add mid-size chromosomes (by positive count, descending, excluding the giants)
        # until cumulative positives reach the target; keeps the held-out set small.
        cand = [(c, pooled_pos[c]) for c in pooled_pos if c not in exclude]
        cand.sort(key=lambda t: t[1], reverse=True)
        chosen, cum = [], 0
        for c, v in cand:
            if cum >= target:
                break
            chosen.append(c)
            cum += v
        fold_assignment = {c: 0 for c in sorted(chosen, key=chrom_sort_key)}
        print(f"\n=== suggested held-out TEST set (~{args.suggest_test_frac:.0%} of positives) ===")
        print(f"chromosomes: {sorted(chosen, key=chrom_sort_key)}")
        print(f"positives held out: {cum} / {ptot_pos} = {cum/ptot_pos:.3f}")
        print("fold_assignment (paste into PartitionSpec / run_experiment partition config):")
        print(f"  {fold_assignment}")
        print("NOTE: fold_assignment must be IDENTICAL across all cells or cross-individual "
              "leakage returns. Review chromosome sizes before committing.")

    if args.out_prefix:
        cells.to_csv(f"{args.out_prefix}_cells.csv", index=False)
        per_chrom.to_csv(f"{args.out_prefix}_per_chrom.csv", index=False)
        print(f"\nWrote {args.out_prefix}_cells.csv and {args.out_prefix}_per_chrom.csv")


if __name__ == "__main__":
    main()
