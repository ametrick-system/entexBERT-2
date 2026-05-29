#!/usr/bin/env python3

"""
Build an entexBERT-2 regression dataset from a BigWig target track.

Example task:
    sequence around SNV -> log1p(max fold-change-over-control signal around SNV)

This produces:
    output_dir/train.csv
    output_dir/dev.csv
    output_dir/test.csv

For single-sequence modes:
    sequence,label

For paired-sequence modes:
    sequence1,sequence2,label
"""

import argparse
import os

from pyfaidx import Fasta

from entexbert2.utils import (
    BalanceSpec,
    SNVWindowSpec,
    build_as_prediction_dataset,
    make_bigwig_label_spec,
    log1p_transform,
    identity_transform,
)

def parse_split_ratio(values):
    if len(values) != 3:
        raise argparse.ArgumentTypeError("split_ratio must have exactly 3 values.")
    vals = tuple(float(v) for v in values)
    if abs(sum(vals) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"split_ratio must sum to 1.0, got {vals}.")
    return vals

def get_transform_fn(name: str):
    name = name.lower()

    if name == "log1p":
        return log1p_transform
    if name == "identity":
        return identity_transform

    raise ValueError(f"Unsupported transform: {name}")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build train/dev/test CSVs for regression of a BigWig-derived "
            "signal target around EN-TEx AS SNVs."
        )
    )

    # Core files
    parser.add_argument(
        "--as_tsv",
        required=True,
        help="Path to EN-TEx hetSNVs_default_AS.tsv",
    )
    parser.add_argument(
        "--ref_fasta",
        required=True,
        help="Path to hg38 reference FASTA",
    )
    parser.add_argument(
        "--bigwig",
        required=True,
        help="Path to BigWig target file, e.g. fold-change-over-control bigWig",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where train/dev/test CSVs will be written",
    )

    # Experiment naming
    parser.add_argument(
        "--target_name",
        required=True,
        help=(
            "Short target name used in the label column before renaming to label. "
            "Examples: CTCF, ATAC, H3K27ac, DNase"
        ),
    )
    parser.add_argument(
        "--label_name",
        default=None,
        help=(
            "Optional explicit label name. If omitted, one is generated from "
            "target_name, transform, signal_mode, region, and radius."
        ),
    )

    # EN-TEx filters
    parser.add_argument(
        "--assay",
        required=True,
        help=(
            "Assay name in hetSNVs_default_AS.tsv used to choose SNV rows. "
            "Example: TF-ChIP-seq_CTCF"
        ),
    )
    parser.add_argument(
        "--donor",
        required=True,
        help="Donor ID, e.g. ENC-001",
    )
    parser.add_argument(
        "--tissue",
        required=True,
        help="Tissue name exactly as it appears in hetSNVs_default_AS.tsv",
    )
    parser.add_argument(
        "--min_total_reads",
        type=int,
        default=10,
        help="Minimum hap1+hap2 allele-supporting reads required",
    )

    # Sequence window
    parser.add_argument(
        "--left_bp",
        type=int,
        default=128,
        help="Number of bp to include to the left of the SNV",
    )
    parser.add_argument(
        "--right_bp",
        type=int,
        default=128,
        help="Number of bp to include to the right of the SNV",
    )
    parser.add_argument(
        "--chrom_sizes",
        default=None,
        help="Optional chrom.sizes file",
    )

    # BigWig target definition
    parser.add_argument(
        "--signal_mode",
        default="max",
        choices=[
            "mean",
            "max",
            "min",
            "sum",
            "std",
            "coverage",
            "mean_nonzero",
            "max_abs",
        ],
        help="How to summarize BigWig values over the queried region",
    )
    parser.add_argument(
        "--signal_region",
        default="snv_radius",
        choices=["window", "snv", "snv_radius"],
        help=(
            "Region to query in the BigWig: full sequence window, single SNV base, "
            "or ±radius around the SNV"
        ),
    )
    parser.add_argument(
        "--signal_radius_bp",
        type=int,
        default=20,
        help="Radius around SNV when --signal_region snv_radius",
    )
    parser.add_argument(
        "--target_transform",
        default="log1p",
        choices=["log1p", "identity"],
        help="Transform applied to raw BigWig summary value",
    )
    parser.add_argument(
        "--missing_value",
        type=float,
        default=0.0,
        help="Value used when BigWig has no finite signal in queried region",
    )

    # Sequence mode
    parser.add_argument(
        "--input_mode",
        default="ref_single",
        choices=[
            "ref_single",
            "hap1_single",
            "hap2_single",
            "hap_pair",
            "ref_hap1_pair",
            "ref_hap2_pair",
        ],
        help=(
            "Sequence input mode. For locus-level BigWig signal, ref_single is "
            "usually the cleanest baseline."
        ),
    )

    # Optional balancing
    parser.add_argument(
        "--balance_strategy",
        default="none",
        choices=["none", "global_binary", "per_tissue_binary"],
        help=(
            "Optional balancing strategy. For continuous BigWig regression, "
            "usually use none."
        ),
    )
    parser.add_argument(
        "--balance_label_col",
        default="imbalance_significance",
        help="Column used if balance_strategy is binary balancing",
    )

    # Split/reproducibility
    parser.add_argument(
        "--split_ratio",
        nargs=3,
        default=(0.8, 0.1, 0.1),
        type=float,
        metavar=("TRAIN", "DEV", "TEST"),
        help="Train/dev/test split fractions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size for reading AS TSV",
    )

    return parser.parse_args()


def build_label_name(args) -> str:
    if args.label_name is not None:
        return args.label_name

    parts = [
        args.target_name,
        args.target_transform,
        args.signal_mode,
        args.signal_region,
    ]

    if args.signal_region == "snv_radius":
        parts.append(f"pm{args.signal_radius_bp}")

    return "_".join(parts)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_ratio = tuple(args.split_ratio)
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

    label_name = build_label_name(args)
    transform_fn = get_transform_fn(args.target_transform)

    print("Loading reference FASTA...")
    ref = Fasta(args.ref_fasta)

    print("\nBuilding BigWig regression dataset")
    print(f"Target name:        {args.target_name}")
    print(f"Label name:         {label_name}")
    print(f"BigWig:             {args.bigwig}")
    print(f"Signal mode:        {args.signal_mode}")
    print(f"Signal region:      {args.signal_region}")
    print(f"Signal radius bp:   {args.signal_radius_bp}")
    print(f"Transform:          {args.target_transform}")
    print(f"Assay filter:       {args.assay}")
    print(f"Tissue filter:      {args.tissue}")
    print(f"Donor filter:       {args.donor}")
    print(f"Input mode:         {args.input_mode}")
    print(f"Output dir:         {args.output_dir}")

    primary_label = make_bigwig_label_spec(
        name=label_name,
        bigwig_path=args.bigwig,
        mode=args.signal_mode,
        region=args.signal_region,
        radius_bp=args.signal_radius_bp,
        transform_fn=transform_fn,
        missing_value=args.missing_value,
        use_values=True,
        exact=True,
    )

    window_spec = SNVWindowSpec(
        left_bp=args.left_bp,
        right_bp=args.right_bp,
        chrom_sizes_path=args.chrom_sizes,
    )

    balance_spec = BalanceSpec(
        strategy=args.balance_strategy,
        label_col=args.balance_label_col,
        random_state=args.seed,
    )

    df = build_as_prediction_dataset(
        input_tsv=args.as_tsv,
        output_dir=args.output_dir,
        ref_fasta=ref,
        assay=args.assay,
        donor=args.donor,
        tissue=args.tissue,
        primary_label=primary_label,
        window_spec=window_spec,
        input_mode=args.input_mode,
        min_total_reads=args.min_total_reads,
        balance_spec=balance_spec,
        aux_labels=None,
        split_ratio=split_ratio,
        seed=args.seed,
        chunksize=args.chunksize,
        skip_ambiguous=True,
    )

    print("\nDone.")
    print(f"Final dataframe examples: {len(df)}")
    print(f"CSV files written to: {args.output_dir}")
    print("\nTarget summary before train/dev/test split:")
    print(df[label_name].describe())

if __name__ == "__main__":
    main()