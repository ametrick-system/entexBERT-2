#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import numpy as np

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative TP/FP/FN/TN examples from predictions.csv."
    )

    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_category", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    df["example_index"] = np.arange(len(df))

    required = {
        "label",
        "pred_label",
        "prob_positive",
        "confusion_category",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = []

    for category in ["TP", "TN", "FP", "FN"]:
        group = df[df["confusion_category"] == category].copy()

        if group.empty:
            print(f"No examples for category {category}")
            continue

        # For hap_pair data, avoid selecting the same input pair multiple times
        if {"sequence1", "sequence2"}.issubset(group.columns):
            before = len(group)
            group = group.drop_duplicates(subset=["sequence1", "sequence2"]).copy()
            after = len(group)
            print(f"{category}: deduplicated hap-pairs {before} -> {after}")

        # For ref_single data, avoid selecting the same sequence multiple times
        elif "sequence" in group.columns:
            before = len(group)
            group = group.drop_duplicates(subset=["sequence"]).copy()
            after = len(group)
            print(f"{category}: deduplicated sequences {before} -> {after}")

        if category == "TP":
            # High-confidence true positives
            group = group.sort_values("prob_positive", ascending=False)

        elif category == "TN":
            # High-confidence true negatives
            group = group.sort_values("prob_positive", ascending=True)

        elif category == "FP":
            # Most confident false positives
            group = group.sort_values("prob_positive", ascending=False)

        elif category == "FN":
            # Most confident false negatives, i.e. model strongly said non-AS
            group = group.sort_values("prob_positive", ascending=True)

        group = group.head(args.n_per_category).copy()
        group["selection_rank_within_category"] = range(1, len(group) + 1)

        selected.append(group)

        out_path = output_dir / f"{category}_representative_examples.csv"
        group.to_csv(out_path, index=False)

        print(f"\n{category}: selected {len(group)} examples")
        print(group[["label", "pred_label", "prob_positive", "confusion_category"]].head())

    if selected:
        selected_df = pd.concat(selected, ignore_index=True)
        selected_df.to_csv(output_dir / "representative_examples_all.csv", index=False)

    print(f"\nSaved representative examples to: {output_dir}")


if __name__ == "__main__":
    main()