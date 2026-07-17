#!/usr/bin/env python3
"""
audit_proximity_leakage.py -- quantify PROXIMITY (near-duplicate) leakage between
train and test splits, the mechanism distinct from exact-sequence leakage.

Exact-sequence auditors catch byte-identical windows. They MISS the leakage that
comes from two DIFFERENT SNVs whose +/-flank windows OVERLAP: if a test SNV sits
within ~one window-width of a train SNV on the same chromosome, their sequences are
near-identical and (usually) share a label, so the model has effectively seen the
test example. This is the failure mode of splitting by EXACT coordinate (which keeps
same-SNV copies together but scatters neighbouring SNVs across splits) with NO
chromosome / locus / bin holdout.

For each TEST SNV this computes the distance to the nearest TRAIN SNV on the same
chromosome, then reports:
  - fraction of test SNVs with a train SNV closer than the window width (windows overlap)
  - fraction closer than HALF the window width (>50% overlap -- strong leakage)
  - among the overlapping pairs, LABEL CONCORDANCE (overlap only leaks if labels agree)
  - a within-vs-across-split contrast: nearest-train distance vs nearest-OTHER-TEST
    distance (if test SNVs are as close to each other as to train, the split simply
    reflects dense sampling, not a train/test-specific leak)

Input: train/dev/test files (CSV or TSV) that carry a chromosome column and a single
SNV coordinate column (e.g. the builder's `chr` + `SNV`, or `.meta.csv`). Column names
and delimiter are auto-detected; override with flags. A label column enables the
concordance metric.

Usage:
  python audit_proximity_leakage.py --train train.csv --test test.csv \
      [--dev dev.csv] [--chrom_col chr] [--pos_col SNV] [--label_col label] \
      [--window_bp 512] [--out proximity_leakage.csv] [--plot prox.png]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

CHROM_CANDS = ["chr", "chrom", "chromosome", "seqnames", "#chr"]
POS_CANDS   = ["SNV", "snv_pos", "pos", "position", "snp_pos", "variant_pos", "start"]
LABEL_CANDS = ["label", "y", "imbalance_significance", "target"]


def _read_any(path):
    """Delimiter-agnostic reader: try comma, then tab, then whitespace. Header assumed."""
    for sep in (",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _pick(df, override, cands, role):
    if override:
        if override not in df.columns:
            sys.exit(f"ERROR: --{role} '{override}' not in columns {list(df.columns)[:12]}")
        return override
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _nearest_same_chrom(test_chrom, test_pos, ref_chrom, ref_pos, exclude_self=False):
    """For each (test_chrom,test_pos) return distance to nearest (ref_chrom,ref_pos)
    on the SAME chromosome. exclude_self drops the zero-distance self-match when ref
    IS the test set (nearest-other-in-set contrast)."""
    ref_by_chrom = {}
    for c, p in zip(ref_chrom, ref_pos):
        ref_by_chrom.setdefault(c, []).append(p)
    for c in ref_by_chrom:
        ref_by_chrom[c] = np.sort(np.asarray(ref_by_chrom[c], dtype=np.int64))
    out = np.full(len(test_pos), np.iinfo(np.int64).max, dtype=np.int64)
    for i, (c, p) in enumerate(zip(test_chrom, test_pos)):
        arr = ref_by_chrom.get(c)
        if arr is None or len(arr) == 0:
            continue
        j = np.searchsorted(arr, p)
        best = np.iinfo(np.int64).max
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(arr):
                d = abs(int(arr[k]) - int(p))
                if exclude_self and d == 0:
                    # skip exactly one self-match; keep genuine duplicates handled below
                    continue
                best = min(best, d)
        out[i] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--dev", default=None, help="Optional; audited against train too.")
    ap.add_argument("--chrom_col", default=None)
    ap.add_argument("--pos_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--window_bp", type=int, default=512,
                    help="Full window width (2*flank). Windows within this distance overlap. Default 512.")
    ap.add_argument("--out", default="proximity_leakage.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--dataset_name", default="dataset")
    args = ap.parse_args()

    train = _read_any(args.train)
    test = _read_any(args.test)
    chrom_c = _pick(train, args.chrom_col, CHROM_CANDS, "chrom_col")
    pos_c   = _pick(train, args.pos_col, POS_CANDS, "pos_col")
    if chrom_c is None or pos_c is None:
        sys.exit(f"ERROR: could not find chrom/pos columns. Have {list(train.columns)[:12]}. "
                 f"Pass --chrom_col/--pos_col.")
    label_c = _pick(train, args.label_col, LABEL_CANDS, "label_col")

    def prep(df):
        d = df[[chrom_c, pos_c] + ([label_c] if label_c and label_c in df.columns else [])].copy()
        d[chrom_c] = d[chrom_c].astype(str)
        d[pos_c] = pd.to_numeric(d[pos_c], errors="coerce")
        d = d.dropna(subset=[pos_c])
        d[pos_c] = d[pos_c].astype(np.int64)
        return d

    train, test = prep(train), prep(test)
    W, H = args.window_bp, args.window_bp // 2

    rows = []
    splits = [("test", test)] + ([("dev", prep(_read_any(args.dev)))] if args.dev else [])
    for split_name, sdf in splits:
        d_train = _nearest_same_chrom(sdf[chrom_c].to_numpy(), sdf[pos_c].to_numpy(),
                                      train[chrom_c].to_numpy(), train[pos_c].to_numpy())
        d_self = _nearest_same_chrom(sdf[chrom_c].to_numpy(), sdf[pos_c].to_numpy(),
                                     sdf[chrom_c].to_numpy(), sdf[pos_c].to_numpy(),
                                     exclude_self=True)
        finite = d_train[d_train < np.iinfo(np.int64).max]
        n = len(sdf)
        overlap = d_train < W        # windows overlap at all
        strong  = d_train < H        # >50% window overlap
        exact0  = d_train == 0       # same coordinate present in train

        r = {
            "dataset": args.dataset_name, "split": split_name, "n": n,
            "window_bp": W,
            "frac_window_overlap": float(overlap.mean()),
            "frac_strong_overlap_50pct": float(strong.mean()),
            "frac_exact_coord_in_train": float(exact0.mean()),
            "median_nearest_train_bp": float(np.median(finite)) if len(finite) else np.nan,
            "median_nearest_othertest_bp": float(np.median(d_self[d_self < np.iinfo(np.int64).max]))
                if (d_self < np.iinfo(np.int64).max).any() else np.nan,
        }
        # label concordance among overlapping pairs (needs nearest-train label)
        if label_c and label_c in sdf.columns:
            # recompute nearest-train INDEX to fetch its label, for overlapping test rows
            tb = {}
            for c, p, l in zip(train[chrom_c], train[pos_c], train[label_c]):
                tb.setdefault(c, []).append((p, l))
            for c in tb:
                tb[c].sort()
            conc = []
            for c, p, l in zip(sdf[chrom_c], sdf[pos_c], sdf[label_c]):
                arr = tb.get(c)
                if not arr:
                    continue
                ps = np.array([a[0] for a in arr]); ls = [a[1] for a in arr]
                j = np.searchsorted(ps, p); best=None; bd=W
                for k in (j-1, j, j+1):
                    if 0 <= k < len(ps):
                        dd = abs(int(ps[k]) - int(p))
                        if dd < bd:
                            bd = dd; best = ls[k]
                if best is not None:  # within window
                    conc.append(int(best == l))
            r["n_overlapping_pairs"] = len(conc)
            r["label_concordance_overlap"] = float(np.mean(conc)) if conc else np.nan
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))

    if args.plot and len(splits):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            sdf = splits[0][1]
            d_train = _nearest_same_chrom(sdf[chrom_c].to_numpy(), sdf[pos_c].to_numpy(),
                                          train[chrom_c].to_numpy(), train[pos_c].to_numpy())
            d = d_train[d_train < np.iinfo(np.int64).max]
            d = np.clip(d, 1, 10**7)
            ax.hist(np.log10(d), bins=60, color="#4C72B0", alpha=0.85)
            ax.axvline(np.log10(W), color="#C44E52", lw=2, label=f"window width ({W} bp)")
            ax.axvline(np.log10(H), color="#DD8452", lw=1.5, ls="--", label=f"half window ({H} bp)")
            ax.set_xlabel("distance to nearest TRAIN SNV, same chrom (log10 bp)")
            ax.set_ylabel("test SNVs")
            ax.set_title("Proximity of test SNVs to training SNVs")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"wrote {args.plot}")
        except Exception as ex:
            print(f"plot skipped: {ex}")


if __name__ == "__main__":
    main()
