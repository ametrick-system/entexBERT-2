#!/usr/bin/env python3

"""
Build an entexBERT-2 LUPI dataset.

Main task:
    hap1/hap2 sequence pair -> imbalance_significance

Auxiliary task:
    hap1/hap2 sequence pair -> log1p(max BigWig signal near SNV)

Output CSV columns:
    sequence1,sequence2,label,<aux_name>
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
    make_bigwig_label_spec,
    log1p_transform,
    identity_transform,
)


def get_transform_fn(name: str):
    name = name.lower()

    if name == "log1p":
        return log1p_transform
    if name == "identity":
        return identity_transform

    raise ValueError(f"Unsupported transform: {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build hap_pair AS classification dataset with BigWig LUPI aux target."
    )

    parser.add_argument("--as_tsv", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--bigwig", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--assay", default="TF-ChIP-seq_CTCF")
    parser.add_argument("--donor", default="ENC-001")
    parser.add_argument("--tissue", required=True)

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
        default="global_binary",
        choices=["none", "global_binary", "per_tissue_binary"],
    )

    parser.add_argument(
        "--aux_name",
        default="ctcf_log1p_max_fc_snv_pm20",
        help="Name of the BigWig auxiliary label column.",
    )

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
    )

    parser.add_argument(
        "--signal_region",
        default="snv_radius",
        choices=["window", "snv", "snv_radius"],
    )

    parser.add_argument("--signal_radius_bp", type=int, default=20)

    parser.add_argument(
        "--target_transform",
        default="log1p",
        choices=["log1p", "identity"],
    )

    parser.add_argument("--missing_value", type=float, default=0.0)

    parser.add_argument("--split_ratio", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100000)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading reference FASTA...")
    ref = Fasta(args.ref_fasta)

    print("\nBuilding AS classification + BigWig LUPI dataset")
    print(f"Main label: imbalance_significance")
    print(f"Aux label:  {args.aux_name}")
    print(f"BigWig:     {args.bigwig}")
    print(f"Tissue:     {args.tissue}")
    print(f"Input mode: {args.input_mode}")

    main_label = LabelSpec(
        name="imbalance_significance",
        fn=imbalance_significance_label,
        task_type="classification",
    )

    aux_label = make_bigwig_label_spec(
        name=args.aux_name,
        bigwig_path=args.bigwig,
        mode=args.signal_mode,
        region=args.signal_region,
        radius_bp=args.signal_radius_bp,
        transform_fn=get_transform_fn(args.target_transform),
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
        label_col="imbalance_significance",
        random_state=args.seed,
    )

    df = build_as_prediction_dataset(
        input_tsv=args.as_tsv,
        output_dir=args.output_dir,
        ref_fasta=ref,
        assay=args.assay,
        donor=args.donor,
        tissue=args.tissue,
        primary_label=main_label,
        window_spec=window_spec,
        input_mode=args.input_mode,
        min_total_reads=args.min_total_reads,
        balance_spec=balance_spec,
        aux_labels=[aux_label],
        split_ratio=tuple(args.split_ratio),
        seed=args.seed,
        chunksize=args.chunksize,
        skip_ambiguous=True,
    )

    print("\nDone.")
    print(f"Rows: {len(df)}")

    print("\nMain label counts:")
    print(df["imbalance_significance"].value_counts().sort_index())

    print(f"\nAux label summary: {args.aux_name}")
    print(df[args.aux_name].describe())


if __name__ == "__main__":
    main()