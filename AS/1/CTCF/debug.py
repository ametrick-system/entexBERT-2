#!/usr/bin/env python3
"""
Offline diagnosis of why the signed-regression twin sits at Spearman ~= 0.
No GPU, no model -- reads the rich <split>.meta.csv sidecars (which carry label, total_reads,
tissue, locus_id, ref_allele_ratio, imbalance_significance).

Answers three questions:
  (1) DEPTH NOISE  -- is the label mostly determined by read depth, and does its signal survive
      a stricter depth floor?  (corr(label, depth); |label| vs depth; variance under floors)
  (2) CONFLICT CEILING -- how much label variance is WITHIN locus (same sequence, different
      tissues -> unlearnable) vs BETWEEN loci (learnable)?  This bounds achievable R^2/Spearman.
  (3) TWIN WIRING SANITY -- a paired CSV must have sequence1 != sequence2 (ref vs alt windows);
      if they're identical the contrast is empty regardless of the model.

It does NOT touch the model. A separate one-batch print (see --emit_twin_assert) is the way to
confirm input_ids_alt actually reaches forward at train time.
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="dataset input/ dir (has train/dev/test .meta.csv)")
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--label_col", default="label")
    p.add_argument("--depth_floors", default="20,30,50,100,200",
                   help="comma list of min_total_reads floors to re-evaluate label variance under")
    return p.parse_args()


def load_split(data_dir, split):
    meta = os.path.join(data_dir, f"{split}.meta.csv")
    base = os.path.join(data_dir, f"{split}.csv")
    if os.path.exists(meta):
        df = pd.read_csv(meta)
        print(f"Loaded {meta}  ({len(df)} rows, {df.shape[1]} cols)")
    else:
        df = pd.read_csv(base)
        print(f"NOTE: no {split}.meta.csv; loaded {base} (depth/tissue/locus checks need the sidecar)")
    return df


def section(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def q1_depth_noise(df, label_col, floors):
    section("(1) DEPTH NOISE: is the label driven by read depth, and does signal survive a floor?")
    y = pd.to_numeric(df[label_col], errors="coerce")
    if "total_reads" not in df.columns:
        print("  no 'total_reads' column; cannot run depth diagnosis."); return
    depth = pd.to_numeric(df["total_reads"], errors="coerce")
    ok = y.notna() & depth.notna()
    y, depth = y.to_numpy()[ok.values], depth.to_numpy()[ok.values]
    absy = np.abs(y)

    from scipy import stats
    print(f"  n={len(y)}  label mean={y.mean():+.4f} std={y.std():.4f}  "
          f"frac|label|<0.05={np.mean(absy < 0.05):.3f}  frac==0={np.mean(y == 0):.3f}")
    if len(y) > 2 and y.std() > 0:
        r_sd, _ = stats.spearmanr(depth, y)
        r_abs, _ = stats.spearmanr(depth, absy)
        print(f"  spearman(depth, label)      = {r_sd:+.3f}   (sign should NOT depend on depth -> ~0 good)")
        print(f"  spearman(depth, |label|)    = {r_abs:+.3f}   (strong negative => small |effect| is just high-depth balance)")
    # The key test: does label VARIANCE collapse as we raise the floor? If most variance lives at
    # low depth, it's sampling noise and a higher floor (or depth-weighted loss) is required.
    print("\n  label variance + size as the depth floor rises (variance should be STABLE if real):")
    print(f"    {'floor':>6} {'n':>9} {'label_std':>10} {'mean|label|':>11} {'frac|y|<0.05':>13}")
    base_std = None
    for f in floors:
        m = depth >= f
        if m.sum() < 50:
            print(f"    {f:>6} {m.sum():>9}  (too few rows)"); continue
        s = y[m].std()
        base_std = base_std if base_std is not None else s
        print(f"    {f:>6} {m.sum():>9} {s:>10.4f} {np.abs(y[m]).mean():>11.4f} {np.mean(np.abs(y[m])<0.05):>13.3f}")
    print("  READ: if label_std rises sharply with the floor, low-depth rows were adding noise, not signal;")
    print("        if it's flat, depth isn't the problem and the signal (if any) is depth-independent.")


def q2_conflict(df, label_col):
    section("(2) CONFLICT CEILING: within-locus (cross-tissue) vs between-locus label variance")
    if "locus_id" not in df.columns:
        print("  no 'locus_id' column; cannot run conflict diagnosis."); return
    y = pd.to_numeric(df[label_col], errors="coerce")
    d = df.assign(_y=y).dropna(subset=["_y"])
    g = d.groupby("locus_id")["_y"]
    n_loci = g.ngroups
    sizes = g.size()
    multi = sizes[sizes > 1]
    total_var = d["_y"].var()
    # within-locus variance (mean of per-locus variances, weighted by rows) = irreducible by sequence
    within = g.transform("mean")
    within_var = float(((d["_y"] - within) ** 2).mean())
    between_var = float(total_var - within_var) if total_var is not None else float("nan")
    frac_within = within_var / total_var if total_var and total_var > 0 else float("nan")
    print(f"  loci={n_loci}  multi-row loci={len(multi)} ({len(multi)/max(n_loci,1):.1%})  "
          f"max rows/locus={int(sizes.max())}")
    print(f"  total label variance      = {total_var:.4f}")
    print(f"  within-locus variance     = {within_var:.4f}   (cross-tissue conflict on the SAME sequence)")
    print(f"  between-locus variance    = {between_var:.4f}   (the part sequence CAN explain)")
    print(f"  fraction within-locus     = {frac_within:.3f}")
    # An optimistic ceiling: best possible R^2 if the model predicted each locus's mean perfectly.
    print(f"\n  OPTIMISTIC R^2 CEILING (predict per-locus mean) = {1 - frac_within:.3f}")
    print("  READ: if 'fraction within-locus' is high (say >0.5), most variance is cross-tissue conflict")
    print("        on identical sequences -> Spearman is capped low and pooling tissues is the problem.")
    if "tissue" in d.columns:
        nt = d["tissue"].nunique()
        # per-tissue label std, to see if a single tissue is cleaner
        print(f"\n  tissues pooled = {nt}; per-tissue label std (a single tissue avoids the conflict):")
        for t, sub in d.groupby("tissue"):
            print(f"    {str(t)[:34]:34s} n={len(sub):7d}  std={sub['_y'].std():.4f}  "
                  f"mean={sub['_y'].mean():+.4f}")


def q3_twin_sanity(data_dir, split):
    section("(3) TWIN WIRING SANITY: do the two windows actually differ (ref vs alt)?")
    base = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(base)
    if not {"sequence1", "sequence2"}.issubset(df.columns):
        print("  CSV has no sequence1/sequence2 -> NOT a paired build. The twin needs ref_alt_pair.")
        return
    s1, s2 = df["sequence1"].astype(str), df["sequence2"].astype(str)
    identical = (s1 == s2)
    print(f"  paired rows={len(df)}  sequence1==sequence2: {identical.sum()} ({identical.mean():.3%})")
    if identical.mean() > 0.001:
        print("  WARNING: some ref/alt windows are IDENTICAL -> empty contrast for those rows "
              "(homozygous-looking; should have been dropped).")
    # how many bases differ on a sample (expect ~1 for a single SNV, more with jitter overlap)
    diffs = []
    for a, b in zip(s1.head(2000), s2.head(2000)):
        if len(a) == len(b):
            diffs.append(sum(c1 != c2 for c1, c2 in zip(a, b)))
    if diffs:
        diffs = np.array(diffs)
        print(f"  base differences per pair (first {len(diffs)}): mean={diffs.mean():.2f} "
              f"median={np.median(diffs):.0f} max={diffs.max()}  (expect ~1 for a clean single-SNV contrast)")
    print("  NOTE: this confirms the DATA carries a contrast. To confirm the MODEL uses it, add a")
    print("        one-batch print in finetune.forward: print('twin?', input_ids_alt is not None) -- it")
    print("        must be True during training, else the Trainer is dropping the alt tensors.")


def main():
    args = parse_args()
    floors = [int(x) for x in args.depth_floors.split(",") if x.strip()]
    df = load_split(args.data_dir, args.split)
    q1_depth_noise(df, args.label_col, floors)
    q2_conflict(df, args.label_col)
    q3_twin_sanity(args.data_dir, args.split)
    section("SUMMARY -> what to change")
    print("  - q1 label_std rises with floor  => raise min_total_reads and/or depth-weighted loss")
    print("  - q2 fraction within-locus high  => train on ONE tissue (or per-locus aggregate), not pooled")
    print("  - q3 sequences identical / 0-diff=> data/contrast bug; twin can't work on it")
    print("  - all clean but model still flat => add the forward twin-print to confirm wiring, then")
    print("    if wired & label learnable & still flat: real negative result -> gkm-SVM on same split")


if __name__ == "__main__":
    main()