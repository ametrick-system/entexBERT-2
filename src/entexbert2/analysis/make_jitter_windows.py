#!/usr/bin/env python
"""
Build jittered copies of binding windows for the ISM position-control experiment.

Selects the top-N peak windows (same deterministic top-N-by-rank selection as ism_saliency),
then for each jitter offset delta CIRCULARLY ROLLS the sequence by delta positions. A roll is a
pure translation: identical bases and base-composition, only the anchor/motif POSITION changes
(center -> center+delta). No padding, no new bases, no edge artifact (motif stays clear of the
wrap boundary for |delta| <= ~60 in a 257bp window).

Writes one CSV per delta: <outdir>/<prefix>_jit{delta}.csv, each with the SAME columns as the
input (seq_col rolled; feature_col / rank_col preserved) so ism_saliency.py runs on it unchanged.

Purpose: run trunk ISM on each -> the saliency peak should travel to center+delta, showing the
model localizes the motif wherever it sits (center-token mean-pool is NOT pinning attribution).
"""
import argparse, os
import numpy as np
import pandas as pd


def roll_seq(s, delta):
    """Circular roll of a DNA string by delta (positive = shift right/downstream)."""
    s = str(s)
    L = len(s)
    delta = delta % L
    if delta == 0:
        return s
    return s[-delta:] + s[:-delta]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--feature_col", default="feature_type")
    ap.add_argument("--feature_keep", default="peak")
    ap.add_argument("--feature_exact", action="store_true")
    ap.add_argument("--rank_col", default="binding_label_raw")
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--deltas", default="-60,-30,0,30,60",
                    help="comma list of jitter offsets in bp")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="jitter")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.windows_csv)
    # SAME selection as ism_saliency: filter feature, sort by rank, take top-N -- ONCE,
    # so every delta jitters the identical set of windows (a controlled within-window shift).
    if a.feature_col in df.columns:
        if a.feature_exact:
            df = df[df[a.feature_col].astype(str).str.strip() == a.feature_keep]
        else:
            df = df[df[a.feature_col].astype(str).str.contains(a.feature_keep, case=False, na=False)]
    if a.rank_col in df.columns:
        df = df.sort_values(a.rank_col, ascending=False)
    df = (df if a.n_windows <= 0 else df.head(a.n_windows)).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("no windows selected -- check --seq_col/--feature_col/--rank_col")

    L = len(str(df[a.seq_col].iloc[0]))
    deltas = [int(x) for x in a.deltas.split(",")]
    for dlt in deltas:
        out = df.copy()
        out[a.seq_col] = out[a.seq_col].map(lambda s: roll_seq(s, dlt))
        # sanity: every rolled sequence keeps its length
        assert (out[a.seq_col].str.len() == L).all(), f"roll changed length at delta={dlt}"
        path = os.path.join(a.outdir, f"{a.prefix}_jit{dlt}.csv")
        out.to_csv(path, index=False)
        print(f"[jitter] delta={dlt:+d}  ->  {path}  ({len(out)} windows, L={L})")


if __name__ == "__main__":
    main()