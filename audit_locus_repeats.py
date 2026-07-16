#!/usr/bin/env python3
"""
audit_locus_repeats.py — partition DNABERT test-set leakage into TWO mechanisms, using the
cCRE master table + the reference genome, per (donor, assay):

  (1) SAME-LOCUS tissue replication  — the same cCRE (one region_id) tested in multiple tissues
      becomes multiple byte-identical rows. A random split scatters them across train/test.
      FIXABLE by grouping the split on region_id.  Measured WITHOUT the genome:
      rows-per-region_id from the master table = tissue multiplicity.

  (2) GENOMIC REPEATS — two DIFFERENT cCREs (different region_id, different coordinates) that carry
      an IDENTICAL reference sequence (repeat families, segmental dups). A locus-grouped split does
      NOT remove these. INTRINSIC to the genome. Measured by extracting each locus's hg38 sequence,
      hashing it, and counting hashes shared across >1 distinct region_id.

Why this is the definitive test: the k-mer split files only carry sequence+label, so exact-sequence
overlap alone cannot tell "same cCRE across tissues" from "different cCREs, same sequence". The
region_id + coordinates here resolve it directly.

Master table columns (tab-sep, with header):
  chr start end region_id hap1_count hap2_count experiment_accession donor tissue assay
  hap1_allele_ratio p_betabinom imbalance_significance

Usage
-----
  python audit_locus_repeats.py \
      --master /home/asm242/palmer_scratch/as/ccre/.../cCREs_default_AS_mapped.tsv \
      --ref_fasta /home/asm242/reference_genome/hg38.fa \
      --out_dir leakage_audit --out_prefix entex_locus \
      [--donors ENC-001 ENC-002 ENC-003 ENC-004] [--assays TF-ChIP-seq_CTCF ...]

Requires pyfaidx (present in the eb2 env). Extraction is per UNIQUE locus, so it is cheap.
"""
import argparse, hashlib, os, sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--out_dir", default="leakage_audit")
    ap.add_argument("--out_prefix", default="entex_locus")
    ap.add_argument("--donors", nargs="+", default=None)
    ap.add_argument("--assays", nargs="+", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from pyfaidx import Fasta
    fa = Fasta(args.ref_fasta, sequence_always_upper=True, as_raw=True)

    usecols = ["chr", "start", "end", "region_id", "donor", "tissue", "assay", "imbalance_significance"]
    df = pd.read_csv(args.master, sep="\t", usecols=usecols)
    if args.donors:
        df = df[df["donor"].isin(args.donors)]
    if args.assays:
        df = df[df["assay"].isin(args.assays)]

    # cache reference-sequence hash per unique locus (chr,start,end) — extraction is the only genome cost
    loci = df[["chr", "start", "end"]].drop_duplicates()
    seqhash = {}
    missing = 0
    for chrom, s, e in loci.itertuples(index=False):
        try:
            seq = fa[str(chrom)][int(s):int(e)]
        except Exception:
            missing += 1; seq = None
        seqhash[(chrom, int(s), int(e))] = (hashlib.sha1(seq.encode()).hexdigest() if seq else None)
    if missing:
        print(f"[warn] {missing} loci could not be extracted (chrom not in fasta?)")

    df["seqhash"] = [seqhash[(c, int(s), int(e))] for c, s, e in
                     df[["chr", "start", "end"]].itertuples(index=False)]

    rows = []
    for (donor, assay), g in df.groupby(["donor", "assay"]):
        n_rows = len(g)                                   # (cCRE, tissue) rows = what the pipeline pools
        n_region = g["region_id"].nunique()               # distinct cCREs
        # (1) tissue replication: rows per region_id
        tissue_rep = n_rows / max(n_region, 1)
        # unique reference sequences among the DISTINCT loci
        gl = g.drop_duplicates("region_id")               # one row per cCRE
        valid = gl[gl["seqhash"].notna()]
        n_loci = len(valid)
        n_uniq_seq = valid["seqhash"].nunique()
        # (2) genomic repeats: distinct region_ids sharing a sequence hash
        #     count loci that collide with >=1 OTHER distinct-region locus on seqhash
        hash_region_counts = valid.groupby("seqhash")["region_id"].nunique()
        repeat_hashes = set(hash_region_counts[hash_region_counts > 1].index)
        loci_in_repeats = int(valid["seqhash"].isin(repeat_hashes).sum())
        genomic_repeat_locus_frac = loci_in_repeats / max(n_loci, 1)
        rows.append({
            "donor": donor, "assay": assay,
            "rows_cCRExTissue": n_rows, "distinct_cCREs": n_region,
            "tissue_replication": tissue_rep,
            "distinct_loci_extracted": n_loci, "unique_ref_seqs": n_uniq_seq,
            "genomic_repeat_locus_frac": genomic_repeat_locus_frac,
            "loci_in_repeats": loci_in_repeats,
        })

    out = pd.DataFrame(rows).sort_values(["assay", "donor"])
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_by_dataset.csv")
    out.to_csv(out_csv, index=False)

    print("=== PER-(donor,assay) MECHANISM SPLIT ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== HEADLINE ===")
    print(f"mean TISSUE replication (same cCRE, multiple tissues): {out['tissue_replication'].mean():.2f}x")
    print(f"  -> this is the SAME-LOCUS 'bug' magnitude; fixed by grouping the split on region_id")
    print(f"mean GENOMIC-REPEAT locus fraction (distinct cCREs sharing a sequence): "
          f"{out['genomic_repeat_locus_frac'].mean()*100:.2f}%")
    print(f"  -> this is the INTRINSIC-genome floor that survives region_id grouping")
    print(f"wrote {out_csv}")

    # plot: tissue replication (bar) vs genomic-repeat floor (bar), per assay (donor-averaged)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    a = out.groupby("assay").agg(tissue=("tissue_replication", "mean"),
                                 repeat=("genomic_repeat_locus_frac", "mean")).sort_values("tissue")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.4*len(a)+1.6))
    y = range(len(a))
    ax1.barh(list(y), a["tissue"], color="#c1442e"); ax1.set_yticks(list(y)); ax1.set_yticklabels(a.index, fontsize=6)
    ax1.axvline(1.0, ls="--", lw=0.8, color="grey"); ax1.set_xlabel("tissue replication (rows / cCRE)")
    ax1.set_title("(1) same-locus replication", loc="left", fontsize=8)
    ax2.barh(list(y), a["repeat"]*100, color="#5a5a5a"); ax2.set_yticks(list(y)); ax2.set_yticklabels([])
    ax2.set_xlabel("% of distinct cCREs sharing a sequence")
    ax2.set_title("(2) genomic repeats", loc="left", fontsize=8)
    for i, v in zip(y, a["repeat"]*100):
        ax2.text(v+0.05, i, f"{v:.1f}%", va="center", fontsize=6)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_mechanism.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
