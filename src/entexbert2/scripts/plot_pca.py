#!/usr/bin/env python3
"""
Plot PCA of pooled embeddings from analyze.py's pca.csv:
  Panel 1: colored by prediction category (TP/FP/TN/FN)
  Panel 2: colored by true label

Prediction category comes from pca.csv directly if present (new analyze.py runs),
otherwise it is joined from predictions.csv on example_id. Explained-variance
percentages come from pca_explained_variance.csv (auto-discovered next to pca.csv),
from --evr, or are omitted -- never fabricated.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colors so categories/labels never swap between panels or runs.
CATEGORY_COLORS = {
    "FN": "#1f77b4",  # blue
    "FP": "#ff7f0e",  # orange
    "TN": "#2ca02c",  # green
    "TP": "#d62728",  # red
}
CATEGORY_ORDER = ["FN", "FP", "TN", "TP"]

LABEL_COLORS = {0: "#1f77b4", 1: "#ff7f0e"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pca_csv", required=True, help="analyze.py pca.csv")
    p.add_argument("--predictions_csv", default=None,
                   help="Only needed if pca.csv lacks confusion_category (old runs); "
                        "joined on example_id.")
    p.add_argument("--evr_csv", default=None,
                   help="pca_explained_variance.csv. Auto-discovered next to pca.csv if absent.")
    p.add_argument("--evr", default=None,
                   help="Manual explained-variance ratios, e.g. '0.453,0.064'. Overrides --evr_csv.")
    p.add_argument("--output", default=None, help="Output PNG (default: pca_scatter.png next to pca.csv).")
    p.add_argument("--pc_x", type=int, default=1, help="PC for x axis (1-based).")
    p.add_argument("--pc_y", type=int, default=2, help="PC for y axis (1-based).")
    p.add_argument("--flip_pc_x", action="store_true", help="Negate the x PC (PCA sign is arbitrary).")
    p.add_argument("--flip_pc_y", action="store_true", help="Negate the y PC (PCA sign is arbitrary).")
    p.add_argument("--point_size", type=float, default=12.0)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title_prefix", default="PCA of pooled embeddings")
    return p.parse_args()


def load_evr(args):
    """Return list of explained-variance ratios, or None. Never invented."""
    if args.evr:
        return [float(x) for x in args.evr.split(",") if x.strip() != ""]
    path = args.evr_csv
    if path is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.pca_csv)),
                                 "pca_explained_variance.csv")
        path = candidate if os.path.exists(candidate) else None
    if path and os.path.exists(path):
        evr_df = pd.read_csv(path)
        return evr_df["explained_variance_ratio"].astype(float).tolist()
    return None


def axis_label(pc_1based, evr):
    if evr is not None and len(evr) >= pc_1based:
        return f"PC{pc_1based} ({evr[pc_1based - 1] * 100:.1f}%)"
    return f"PC{pc_1based}"


def attach_category(pca_df, args):
    """Ensure pca_df has confusion_category, joining from predictions.csv if needed."""
    if "confusion_category" in pca_df.columns:
        return pca_df, True

    if args.predictions_csv is None:
        print("WARNING: pca.csv has no 'confusion_category' and no --predictions_csv given; "
              "skipping the prediction-category panel.")
        return pca_df, False

    if "example_id" not in pca_df.columns:
        print("WARNING: pca.csv has no 'example_id' to join on; skipping the prediction-category panel.")
        return pca_df, False

    pred = pd.read_csv(args.predictions_csv)
    if "example_id" not in pred.columns or "confusion_category" not in pred.columns:
        print("WARNING: predictions.csv lacks example_id/confusion_category; "
              "skipping the prediction-category panel.")
        return pca_df, False

    key = pred[["example_id", "confusion_category"]].copy()
    if key["example_id"].duplicated().any():
        raise ValueError(
            "example_id is not unique in predictions.csv; refusing to merge "
            "(would fan out points and distort the plot). Deduplicate or fix ids first."
        )

    n_before = len(pca_df)
    merged = pca_df.merge(key, on="example_id", how="left")
    n_unmatched = int(merged["confusion_category"].isna().sum())
    if n_unmatched:
        print(f"WARNING: {n_unmatched}/{n_before} pca rows had no matching example_id "
              f"in predictions.csv; those points are dropped from the category panel.")
    return merged, True


def scatter_by_group(ax, df, x_col, y_col, group_col, color_map, order, legend_title):
    present = [g for g in order if g in set(df[group_col].dropna().unique())]
    # extras not in the predefined order (kept stable, appended)
    extras = [g for g in df[group_col].dropna().unique() if g not in present]
    groups = present + sorted(extras, key=str)

    # Draw largest group first so smaller groups are not buried.
    sizes = {g: int((df[group_col] == g).sum()) for g in groups}
    draw_order = sorted(groups, key=lambda g: sizes[g], reverse=True)

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, g in enumerate(draw_order):
        sub = df[df[group_col] == g]
        color = color_map.get(g, default_cycle[i % len(default_cycle)] if default_cycle else None)
        ax.scatter(sub[x_col], sub[y_col], s=args_point_size, alpha=args_alpha,
                   linewidths=0, color=color, label=None)

    # Legend in stable order (not draw order), with full-opacity handles.
    handles = []
    for g in groups:
        color = color_map.get(g)
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                  markersize=6, color=color, label=str(g)))
    ax.legend(handles=handles, title=legend_title, loc="best", frameon=True)


def main():
    global args_point_size, args_alpha
    args = parse_args()
    args_point_size = args.point_size
    args_alpha = args.alpha

    pca_df = pd.read_csv(args.pca_csv)

    x_col = f"PC{args.pc_x}"
    y_col = f"PC{args.pc_y}"
    for col in (x_col, y_col):
        if col not in pca_df.columns:
            raise ValueError(f"{col} not in pca.csv (columns: {list(pca_df.columns)}).")

    if args.flip_pc_x:
        pca_df[x_col] = -pca_df[x_col]
    if args.flip_pc_y:
        pca_df[y_col] = -pca_df[y_col]

    evr = load_evr(args)
    if evr is None:
        print("Note: no explained-variance ratios found (no sidecar / --evr); axis labels omit %.")

    pca_df, have_category = attach_category(pca_df, args)

    if "target" not in pca_df.columns:
        raise ValueError("pca.csv has no 'target' column for the true-label panel.")

    # Decide whether the true label is categorical (classification) or continuous (regression).
    target = pca_df["target"]
    uniq = pd.unique(target.dropna())
    is_categorical = (len(uniq) <= 10) and np.all(np.equal(np.mod(np.asarray(uniq, dtype=float), 1), 0))

    n_panels = 2 if have_category else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 5.2 * n_panels))
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    if have_category:
        ax = next(ax_iter)
        scatter_by_group(ax, pca_df, x_col, y_col, "confusion_category",
                         CATEGORY_COLORS, CATEGORY_ORDER, legend_title=None)
        ax.set_title(f"{args.title_prefix} by prediction category")
        ax.set_xlabel(axis_label(args.pc_x, evr))
        ax.set_ylabel(axis_label(args.pc_y, evr))

    ax = next(ax_iter)
    if is_categorical:
        lab = pca_df.copy()
        lab["__label"] = lab["target"].astype(int)
        label_order = sorted(int(v) for v in uniq)
        color_map = {v: LABEL_COLORS.get(v) for v in label_order}
        scatter_by_group(ax, lab, x_col, y_col, "__label",
                         color_map, label_order, legend_title=None)
        # relabel legend entries as label=v
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              color=color_map.get(v), label=f"label={v}") for v in label_order]
        ax.legend(handles=handles, loc="best", frameon=True)
    else:
        sc = ax.scatter(pca_df[x_col], pca_df[y_col], c=pca_df["target"],
                        s=args_point_size, alpha=args_alpha, linewidths=0, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="true value")
    ax.set_title(f"{args.title_prefix} by true label")
    ax.set_xlabel(axis_label(args.pc_x, evr))
    ax.set_ylabel(axis_label(args.pc_y, evr))

    fig.tight_layout()

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.pca_csv)), "pca_scatter.png")
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
