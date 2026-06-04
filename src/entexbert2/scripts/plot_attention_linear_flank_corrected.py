#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot attention profiles with a linear flank baseline subtracted. "
            "Input should be attention_profiles_long.csv from plot_average_attention_profiles.py."
        )
    )

    parser.add_argument("--attention_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--value",
        default="hap1_plus_hap2",
        choices=["hap1", "hap2", "ref", "hap1_plus_hap2", "hap1_minus_hap2"],
        help="Which attention profile to plot/correct.",
    )

    parser.add_argument(
        "--window_bp",
        type=int,
        default=100,
        help="Plot positions from -window_bp to +window_bp.",
    )

    parser.add_argument(
        "--flank_inner_bp",
        type=int,
        default=50,
        help="Inner edge of flank region used for baseline fitting.",
    )

    parser.add_argument(
        "--flank_outer_bp",
        type=int,
        default=100,
        help="Outer edge of flank region used for baseline fitting.",
    )

    parser.add_argument(
        "--baseline_scope",
        default="per_category",
        choices=["per_category", "global"],
        help=(
            "per_category: fit a separate linear flank baseline for each TP/FP/TN/FN curve. "
            "global: fit one linear flank baseline across all categories and subtract it from all curves."
        ),
    )

    parser.add_argument(
        "--categories",
        default="TP,FP,TN,FN",
        help="Comma-separated confusion categories to include.",
    )

    parser.add_argument(
        "--title_suffix",
        default="",
        help="Extra text to include in plot titles.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def load_attention_table(path):
    df = pd.read_csv(path)

    required = {
        "example_id",
        "confusion_category",
        "hap",
        "position_relative_to_snv",
        "attention",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns from attention CSV: {missing}")

    return df


def build_value_table(df, value):
    """
    Returns a table with:
      example_id, confusion_category, position_relative_to_snv, value
    where value is the requested profile.
    """
    if value in {"hap1", "hap2", "ref"}:
        sub = df[df["hap"] == value].copy()
        if sub.empty:
            raise ValueError(f"No rows found for hap/value '{value}'")

        out = sub[
            [
                "example_id",
                "confusion_category",
                "position_relative_to_snv",
                "attention",
            ]
        ].copy()
        out = out.rename(columns={"attention": value})
        return out

    if value in {"hap1_plus_hap2", "hap1_minus_hap2"}:
        if not {"hap1", "hap2"}.issubset(set(df["hap"].unique())):
            raise ValueError(
                f"Cannot compute {value}: attention CSV does not contain both hap1 and hap2."
            )

        pivot = df.pivot_table(
            index=["example_id", "confusion_category", "position_relative_to_snv"],
            columns="hap",
            values="attention",
            aggfunc="mean",
        ).reset_index()

        if value == "hap1_plus_hap2":
            pivot[value] = pivot["hap1"] + pivot["hap2"]
        else:
            pivot[value] = pivot["hap1"] - pivot["hap2"]

        return pivot[
            [
                "example_id",
                "confusion_category",
                "position_relative_to_snv",
                value,
            ]
        ].copy()

    raise ValueError(f"Unsupported value: {value}")


def mean_profiles(value_df, value_col, categories, window_bp):
    sub = value_df[value_df["confusion_category"].isin(categories)].copy()
    sub = sub[
        sub["position_relative_to_snv"].between(-window_bp, window_bp)
    ].copy()

    profiles = (
        sub.groupby(["confusion_category", "position_relative_to_snv"])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "mean_attention"})
    )

    return profiles


def fit_line(x, y):
    """
    Fit y = slope*x + intercept.
    Returns slope, intercept.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise ValueError("Need at least two finite points to fit a line.")

    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def add_linear_baseline(
    profiles,
    flank_inner_bp,
    flank_outer_bp,
    baseline_scope,
):
    profiles = profiles.copy()

    is_flank = (
        profiles["position_relative_to_snv"].abs().between(
            flank_inner_bp,
            flank_outer_bp,
            inclusive="both",
        )
    )
    flank_df = profiles[is_flank].copy()

    if flank_df.empty:
        raise ValueError(
            "No flank positions found for baseline fitting. "
            f"Check flank_inner_bp={flank_inner_bp}, flank_outer_bp={flank_outer_bp}."
        )

    corrected_parts = []
    baseline_rows = []

    if baseline_scope == "global":
        slope, intercept = fit_line(
            flank_df["position_relative_to_snv"],
            flank_df["mean_attention"],
        )

        for category, group in profiles.groupby("confusion_category"):
            group = group.copy()
            group["baseline_slope"] = slope
            group["baseline_intercept"] = intercept
            group["linear_baseline"] = (
                slope * group["position_relative_to_snv"] + intercept
            )
            group["attention_minus_linear_flank"] = (
                group["mean_attention"] - group["linear_baseline"]
            )
            corrected_parts.append(group)

            baseline_rows.append({
                "confusion_category": category,
                "baseline_scope": "global",
                "slope": slope,
                "intercept": intercept,
                "n_flank_points": int(len(flank_df)),
            })

    elif baseline_scope == "per_category":
        for category, group in profiles.groupby("confusion_category"):
            group = group.copy()

            flank_group = group[
                group["position_relative_to_snv"].abs().between(
                    flank_inner_bp,
                    flank_outer_bp,
                    inclusive="both",
                )
            ].copy()

            if len(flank_group) < 2:
                print(
                    f"Warning: skipping {category}, not enough flank points "
                    f"for linear baseline."
                )
                continue

            slope, intercept = fit_line(
                flank_group["position_relative_to_snv"],
                flank_group["mean_attention"],
            )

            group["baseline_slope"] = slope
            group["baseline_intercept"] = intercept
            group["linear_baseline"] = (
                slope * group["position_relative_to_snv"] + intercept
            )
            group["attention_minus_linear_flank"] = (
                group["mean_attention"] - group["linear_baseline"]
            )
            corrected_parts.append(group)

            baseline_rows.append({
                "confusion_category": category,
                "baseline_scope": "per_category",
                "slope": slope,
                "intercept": intercept,
                "n_flank_points": int(len(flank_group)),
            })

    else:
        raise ValueError(f"Unsupported baseline_scope: {baseline_scope}")

    corrected = pd.concat(corrected_parts, ignore_index=True)
    baseline_df = pd.DataFrame(baseline_rows)

    return corrected, baseline_df


def plot_raw_and_baseline(
    corrected,
    output_path,
    value_name,
    window_bp,
    flank_inner_bp,
    flank_outer_bp,
    baseline_scope,
    title_suffix,
    dpi,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for category, group in corrected.groupby("confusion_category"):
        group = group.sort_values("position_relative_to_snv")

        ax.plot(
            group["position_relative_to_snv"],
            group["mean_attention"],
            linewidth=2,
            label=f"{category} raw",
        )

        ax.plot(
            group["position_relative_to_snv"],
            group["linear_baseline"],
            linewidth=1,
            linestyle="--",
            alpha=0.8,
            label=f"{category} flank fit",
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvspan(-flank_outer_bp, -flank_inner_bp, alpha=0.08)
    ax.axvspan(flank_inner_bp, flank_outer_bp, alpha=0.08)

    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel("Average attention")
    ax.set_title(
        f"{value_name} raw attention + linear flank baseline, ±{window_bp} bp\n"
        f"baseline={baseline_scope}, flank={flank_inner_bp}-{flank_outer_bp} bp | {title_suffix}"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_corrected(
    corrected,
    output_path,
    value_name,
    window_bp,
    flank_inner_bp,
    flank_outer_bp,
    baseline_scope,
    title_suffix,
    dpi,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for category, group in corrected.groupby("confusion_category"):
        group = group.sort_values("position_relative_to_snv")

        ax.plot(
            group["position_relative_to_snv"],
            group["attention_minus_linear_flank"],
            linewidth=2,
            label=category,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=0.75)
    ax.axvspan(-flank_outer_bp, -flank_inner_bp, alpha=0.08)
    ax.axvspan(flank_inner_bp, flank_outer_bp, alpha=0.08)

    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel("Attention minus linear flank baseline")
    ax.set_title(
        f"{value_name} linear-flank-corrected attention, ±{window_bp} bp\n"
        f"baseline={baseline_scope}, flank={flank_inner_bp}-{flank_outer_bp} bp | {title_suffix}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved {output_path}")


def summarize_windows(corrected, output_path):
    """
    Summarize corrected and raw attention around SNV and in flanks.
    """
    df = corrected.copy()

    def assign_region(pos):
        if -10 <= pos <= 10:
            return "snv_pm10"
        if -25 <= pos <= 25:
            return "snv_pm25"
        if -100 <= pos <= -50:
            return "left_flank_100_50"
        if 50 <= pos <= 100:
            return "right_flank_50_100"
        return "other"

    df["region"] = df["position_relative_to_snv"].apply(assign_region)
    df = df[df["region"] != "other"].copy()

    summary = (
        df.groupby(["confusion_category", "region"])
        .agg(
            raw_mean_attention=("mean_attention", "mean"),
            corrected_mean_attention=("attention_minus_linear_flank", "mean"),
        )
        .reset_index()
    )

    pivot_raw = summary.pivot(
        index="confusion_category",
        columns="region",
        values="raw_mean_attention",
    )
    pivot_corr = summary.pivot(
        index="confusion_category",
        columns="region",
        values="corrected_mean_attention",
    )

    pivot_raw.columns = [f"raw_{c}" for c in pivot_raw.columns]
    pivot_corr.columns = [f"corrected_{c}" for c in pivot_corr.columns]

    out = pd.concat([pivot_raw, pivot_corr], axis=1).reset_index()

    if "corrected_snv_pm10" in out.columns:
        if "corrected_left_flank_100_50" in out.columns:
            out["corrected_snv_pm10_minus_left_flank"] = (
                out["corrected_snv_pm10"] - out["corrected_left_flank_100_50"]
            )
        if "corrected_right_flank_50_100" in out.columns:
            out["corrected_snv_pm10_minus_right_flank"] = (
                out["corrected_snv_pm10"] - out["corrected_right_flank_50_100"]
            )

    out.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    print("\nWindow summary:")
    print(out)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    if args.flank_inner_bp >= args.flank_outer_bp:
        raise ValueError("--flank_inner_bp must be smaller than --flank_outer_bp")

    if args.flank_outer_bp > args.window_bp:
        raise ValueError("--flank_outer_bp must be <= --window_bp")

    print("Loading attention table...")
    df = load_attention_table(args.attention_csv)

    print(f"Building value table for: {args.value}")
    value_df = build_value_table(df, args.value)

    print("Computing category mean profiles...")
    profiles = mean_profiles(
        value_df=value_df,
        value_col=args.value,
        categories=categories,
        window_bp=args.window_bp,
    )

    print("Fitting/subtracting linear flank baseline...")
    corrected, baseline_df = add_linear_baseline(
        profiles=profiles,
        flank_inner_bp=args.flank_inner_bp,
        flank_outer_bp=args.flank_outer_bp,
        baseline_scope=args.baseline_scope,
    )

    corrected_path = output_dir / f"{args.value}_linear_flank_corrected_profiles.csv"
    baseline_path = output_dir / f"{args.value}_linear_flank_baseline_params.csv"
    window_summary_path = output_dir / f"{args.value}_window_summary.csv"

    corrected.to_csv(corrected_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)

    print(f"Saved {corrected_path}")
    print(f"Saved {baseline_path}")

    raw_plot_path = output_dir / f"{args.value}_raw_with_linear_flank_baseline_pm{args.window_bp}.png"
    corrected_plot_path = output_dir / f"{args.value}_linear_flank_corrected_pm{args.window_bp}.png"

    plot_raw_and_baseline(
        corrected=corrected,
        output_path=raw_plot_path,
        value_name=args.value,
        window_bp=args.window_bp,
        flank_inner_bp=args.flank_inner_bp,
        flank_outer_bp=args.flank_outer_bp,
        baseline_scope=args.baseline_scope,
        title_suffix=args.title_suffix,
        dpi=args.dpi,
    )

    plot_corrected(
        corrected=corrected,
        output_path=corrected_plot_path,
        value_name=args.value,
        window_bp=args.window_bp,
        flank_inner_bp=args.flank_inner_bp,
        flank_outer_bp=args.flank_outer_bp,
        baseline_scope=args.baseline_scope,
        title_suffix=args.title_suffix,
        dpi=args.dpi,
    )

    summarize_windows(
        corrected=corrected,
        output_path=window_summary_path,
    )

    config = vars(args)
    with open(output_dir / "linear_flank_correction_config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()