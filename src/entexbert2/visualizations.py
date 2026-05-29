import os
from typing import Optional, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset_csvs(
    data_dir: str,
    split_names: Sequence[str] = ("train", "dev", "test"),
) -> pd.DataFrame:
    """
    Load train/dev/test CSVs from a dataset directory and add a split column.

    Expected files:
        train.csv
        dev.csv
        test.csv
    """
    frames = []

    for split in split_names:
        path = os.path.join(data_dir, f"{split}.csv")

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)

    if not frames:
        raise ValueError(f"No split CSVs found in {data_dir}")

    return pd.concat(frames, ignore_index=True)


def plot_label_distribution(
    data,
    label_col: str = "label",
    output_path: Optional[str] = None,
    bins: int = 50,
    title: Optional[str] = None,
    by_split: bool = True,
    show: bool = False,
):
    """
    Plot the distribution of the target label.

    Args:
        data:
            Either a pandas DataFrame, a path to one CSV, or a directory
            containing train.csv/dev.csv/test.csv.
        label_col:
            Column containing the target label.
        output_path:
            Optional path to save the figure.
        bins:
            Number of histogram bins for continuous labels.
        title:
            Optional plot title.
        by_split:
            If True and a split column exists, overlay distributions by split.
        show:
            Whether to call plt.show().

    Returns:
        matplotlib Axes object.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif os.path.isdir(data):
        df = load_dataset_csvs(data)
    else:
        df = pd.read_csv(data)

    if label_col not in df.columns:
        raise ValueError(f"label_col {label_col!r} not found in data columns.")

    labels = pd.to_numeric(df[label_col], errors="coerce")
    finite_mask = np.isfinite(labels)

    df = df.loc[finite_mask].copy()
    df[label_col] = labels.loc[finite_mask].astype(float)

    if df.empty:
        raise ValueError("No finite label values available to plot.")

    fig, ax = plt.subplots(figsize=(7, 5))

    # If labels are low-cardinality integers, a bar chart is clearer.
    unique_vals = np.sort(df[label_col].unique())
    is_discrete = (
        len(unique_vals) <= 20
        and np.allclose(unique_vals, np.round(unique_vals))
    )

    if is_discrete:
        if by_split and "split" in df.columns:
            counts = (
                df.groupby(["split", label_col])
                .size()
                .reset_index(name="count")
            )

            for split_name, group in counts.groupby("split"):
                ax.plot(
                    group[label_col],
                    group["count"],
                    marker="o",
                    linestyle="-",
                    label=split_name,
                )
            ax.legend()
        else:
            counts = df[label_col].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values)

        ax.set_xlabel(label_col)
        ax.set_ylabel("Count")

    else:
        if by_split and "split" in df.columns:
            for split_name, group in df.groupby("split"):
                ax.hist(
                    group[label_col],
                    bins=bins,
                    alpha=0.5,
                    density=False,
                    label=split_name,
                )
            ax.legend()
        else:
            ax.hist(df[label_col], bins=bins)

        ax.set_xlabel(label_col)
        ax.set_ylabel("Count")

    if title is None:
        title = f"Distribution of {label_col}"

    ax.set_title(title)

    summary_text = (
        f"n = {len(df)}\n"
        f"mean = {df[label_col].mean():.4g}\n"
        f"std = {df[label_col].std():.4g}\n"
        f"min = {df[label_col].min():.4g}\n"
        f"max = {df[label_col].max():.4g}"
    )

    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    return ax


def plot_sequence_length_distribution(
    data,
    output_path: Optional[str] = None,
    bins: int = 50,
    title: str = "Sequence length distribution",
    show: bool = False,
):
    """
    Plot sequence length distribution for either single-sequence or pair-sequence CSVs.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif os.path.isdir(data):
        df = load_dataset_csvs(data)
    else:
        df = pd.read_csv(data)

    fig, ax = plt.subplots(figsize=(7, 5))

    if "sequence" in df.columns:
        lengths = df["sequence"].astype(str).str.len()
        ax.hist(lengths, bins=bins)
        ax.set_xlabel("Sequence length")

    elif "sequence1" in df.columns and "sequence2" in df.columns:
        lengths1 = df["sequence1"].astype(str).str.len()
        lengths2 = df["sequence2"].astype(str).str.len()

        ax.hist(lengths1, bins=bins, alpha=0.5, label="sequence1")
        ax.hist(lengths2, bins=bins, alpha=0.5, label="sequence2")
        ax.legend()
        ax.set_xlabel("Sequence length")

    else:
        raise ValueError("Expected either sequence or sequence1/sequence2 columns.")

    ax.set_ylabel("Count")
    ax.set_title(title)

    fig.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    return ax