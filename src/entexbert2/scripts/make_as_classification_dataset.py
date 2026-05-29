#!/usr/bin/env python3

"""
Build an entexBERT-2 AS classification dataset.

Default intended use:
    CTCF AS classification across all tissues for one EN-TEx donor.

Main task:
    hap1/hap2 sequence pair -> imbalance_significance

Important:
    Uses group-aware splitting by variant/window/alleles to prevent duplicate
    sequence windows from leaking across train/dev/test.
"""

import argparse
import os

from pyfaidx import Fasta

from entexbert2.utils import (
    BalanceSpec,
    LabelSpec,
    SNVWindowSpec,
    build_as_prediction_dataset,
    imbalance_significance_label,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build AS classification dataset for entexBERT-2."
    )

    parser.add_argument("--as_tsv", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--assay", default="TF-ChIP-seq_CTCF")
    parser.add_argument("--donor", default="ENC-001")
    parser.add_argument(
        "--tissue",
        default=None,
        help=(
            "Optional tissue filter. Omit or pass NONE for all tissues."
        ),
    )

    parser.add_argument("--min_total_reads", type=int, default=10)

    parser.add_argument("--left_bp", type=int, default=256)
    parser.add_argument("--right_bp", type=int, default=256)
    parser.add_argument("--chrom_sizes", default=None)

    parser.add_argument(
        "--input_mode",
        default="hap_pair",
        choices=[
            "ref_single",
            "hap1_single",
            "hap2_single",
            "hap_pair",
            "ref_hap1_pair",
            "ref_hap2_pair",
        ],
    )

    parser.add_argument(
        "--balance_strategy",
        default="per_tissue_binary",
        choices=["none", "global_binary", "per_tissue_binary"],
    )

    parser.add_argument("--split_ratio", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100000)

    return parser.parse_args()


def main():
    args = parse_args()

    tissue = args.tissue
    if tissue in {"NONE", "None", "none", "", "all", "ALL"}:
        tissue = None

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading reference FASTA...")
    ref = Fasta(args.ref_fasta)

    primary_label = LabelSpec(
        name="imbalance_significance",
        fn=imbalance_significance_label,
        task_type="classification",
    )

    window_spec = SNVWindowSpec(
        left_bp=args.left_bp,
        right_bp=args.right_bp,
        chrom_sizes_path=args.chrom_sizes,
    )

    balance_spec = BalanceSpec(
        strategy=args.balance_strategy,
        label_col="imbalance_significance",
        random_state=args.seed,
    )

    # This is the critical leakage-prevention key.
    # Rows with the same window and same hap alleles get assigned to one split.
    group_cols = [
        "chr",
        "bed_start",
        "bed_end",
        "ref_allele",
        "hap1_allele",
        "hap2_allele",
    ]

    print("\nBuilding AS classification dataset")
    print(f"Assay:       {args.assay}")
    print(f"Donor:       {args.donor}")
    print(f"Tissue:      {tissue if tissue is not None else 'ALL'}")
    print(f"Input mode:  {args.input_mode}")
    print(f"Balance:     {args.balance_strategy}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Group cols:  {group_cols}")

    df = build_as_prediction_dataset(
        input_tsv=args.as_tsv,
        output_dir=args.output_dir,
        ref_fasta=ref,
        assay=args.assay,
        donor=args.donor,
        tissue=tissue,
        primary_label=primary_label,
        window_spec=window_spec,
        input_mode=args.input_mode,
        min_total_reads=args.min_total_reads,
        balance_spec=balance_spec,
        aux_labels=None,
        split_ratio=tuple(args.split_ratio),
        seed=args.seed,
        chunksize=args.chunksize,
        skip_ambiguous=True,
        group_cols=group_cols,
    )

    print("\nDone.")
    print(f"Final rows before CSV column restriction: {len(df)}")
    print("\nFinal label counts:")
    print(df["imbalance_significance"].value_counts().sort_index())
    print("\nTissue counts:")
    print(df["tissue"].value_counts())

if __name__ == "__main__":
    main()