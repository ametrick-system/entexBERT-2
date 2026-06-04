#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot attention_profiles_long.csv with an optional window around the SNV."
    )

    parser.add_argument("--attention_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_bp", type=int, default=100)
    parser.add_argument("--title_suffix", default="")
    parser.add_argument(
        "--hap_values",
        default="hap1,hap2,hap1_plus_hap2,hap1_minus_hap2",
        help="Comma-separated values to plot.",
    )

    return parser.parse_args()


def plot_one(df, value_name, value_col, output_dir, window_bp, title_suffix):
    sub = df.copy()

    if window_bp is not None:
        sub = sub[
            sub["position_relative_to_snv"].between(-window_bp, window_bp)
        ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for category, group in sub.groupby("confusion_category"):
        mean_profile = (
            group.groupby("position_relative_to_snv")[value_col]
            .mean()
            .sort_index()
        )

        ax.plot(
            mean_profile.index,
            mean_profile.values,
            linewidth=2,
            label=category,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    if "minus" in value_name:
        ax.axhline(0, linewidth=0.75)

    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel("Average attention")
    ax.set_title(f"{value_name} average attention, ±{window_bp} bp\n{title_suffix}")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"average_attention_{value_name}_pm{window_bp}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.attention_csv)

    required = {
        "example_id",
        "confusion_category",
        "hap",
        "position_relative_to_snv",
        "attention",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    hap_values = [x.strip() for x in args.hap_values.split(",") if x.strip()]

    # Plot hap1 / hap2 directly.
    for hap in ["hap1", "hap2", "ref"]:
        if hap in hap_values and hap in set(df["hap"]):
            sub = df[df["hap"] == hap].copy()
            plot_one(
                df=sub,
                value_name=hap,
                value_col="attention",
                output_dir=output_dir,
                window_bp=args.window_bp,
                title_suffix=args.title_suffix,
            )

    # Build hap-pair summaries if both hap1/hap2 exist.
    if {"hap1", "hap2"}.issubset(set(df["hap"])):
        pivot = df.pivot_table(
            index=["example_id", "confusion_category", "position_relative_to_snv"],
            columns="hap",
            values="attention",
        ).reset_index()

        pivot["hap1_plus_hap2"] = pivot["hap1"] + pivot["hap2"]
        pivot["hap1_minus_hap2"] = pivot["hap1"] - pivot["hap2"]

        for value_name in ["hap1_plus_hap2", "hap1_minus_hap2"]:
            if value_name in hap_values:
                plot_one(
                    df=pivot,
                    value_name=value_name,
                    value_col=value_name,
                    output_dir=output_dir,
                    window_bp=args.window_bp,
                    title_suffix=args.title_suffix,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()