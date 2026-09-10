#!/usr/bin/env python3
"""
select_heldout_chroms.py — reproducible, TF-agnostic held-out chromosome selection.

Pure function of a per-chromosome positive-count table plus four documented,
genome-level policy parameters. Re-run on any TF's variant set to regenerate
its held-out test set with zero manual choices, and a logged rationale.

  python select_heldout_chroms.py --eval_csv /home/asm242/palmer_scratch/entexBERT_2-experiments/ADASTRA/ctcf_adastra_evalset.csv \
      --out_prefix ctcf   [--label_col label --target 0.10 --top_train_spare 2 --max_set 2 --ci_floor 100]

Emits: <out_prefix>_per_chrom_counts.csv  and prints the chosen held-out set + CI flag.
"""
import argparse, itertools, sys
import pandas as pd, numpy as np

# genome-level policy (TF-INDEPENDENT): special-regime chromosomes to always exclude.
SPECIAL_DEFAULT = ["chrX", "chrY", "chr6",                    # sex + MHC/HLA
                   "chr13", "chr14", "chr15", "chr21", "chr22"]  # acrocentric (rDNA/satellite)

def natural_key(c):
    x = c.replace("chr", "")
    return (0, int(x)) if x.isdigit() else (1, x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True,
                    help="variant table with a 'chr' column and a binary positive-label column")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--label_col", default="label",
                    help="name of the 0/1 positive-label column "
                         "(ADASTRA: 'label'; EN-TEx hetSNV: 'imbalance_significance')")
    ap.add_argument("--target", type=float, default=0.10,
                    help="coverage-target fraction of total positives")
    ap.add_argument("--top_train_spare", type=int, default=2,
                    help="spare the N largest chromosomes (by variant count) from holdout")
    ap.add_argument("--max_set", type=int, default=2,
                    help="max chromosomes in the held-out set")
    ap.add_argument("--ci_floor", type=int, default=100,
                    help="min positives in the chosen set; below this, warn + fall back to hash-bin only")
    ap.add_argument("--special", nargs="+", default=SPECIAL_DEFAULT)
    a = ap.parse_args()

    df = pd.read_csv(a.eval_csv, usecols=lambda c: c in ("chr", a.label_col))
    if a.label_col not in df.columns:
        sys.exit(f"ERROR: label column {a.label_col!r} not found in {a.eval_csv}")
    df["_lab"] = df[a.label_col].astype(int)
    g = (df.groupby("chr")["_lab"].agg(n="size", pos="sum").reset_index())
    g["k"] = g["chr"].map(natural_key)
    g = g.sort_values("k").drop(columns="k").reset_index(drop=True)
    tot = int(g["pos"].sum())
    g["pos_frac"] = g["pos"] / tot
    g.drop(columns="pos_frac").assign(pos_frac=g["pos_frac"]) \
        .to_csv(f"{a.out_prefix}_per_chrom_counts.csv", index=False)

    # spare the N largest chromosomes by VARIANT COUNT (n) — data-driven per TF
    by_size = g.sort_values("n", ascending=False)["chr"].tolist()
    spare_big = set(by_size[:a.top_train_spare])
    excl = set(a.special) | spare_big
    elig = g[~g["chr"].isin(excl)]
    posd = dict(zip(elig["chr"], elig["pos"]))

    cands = []
    for r in range(1, a.max_set + 1):
        for cmb in itertools.combinations(sorted(posd, key=natural_key), r):
            p = sum(posd[c] for c in cmb)
            cands.append((cmb, p, p / tot))
    # closest to target; tie-break by natural chromosome order (deterministic)
    cands.sort(key=lambda x: (abs(x[2] - a.target), [natural_key(c) for c in x[0]]))
    chosen, cpos, cfrac = cands[0]

    print(f"total positives = {tot}  over {len(g)} chromosomes")
    print(f"largest-{a.top_train_spare} by variant count (spared from holdout): "
          f"{sorted(spare_big, key=natural_key)}")
    print(f"special-regime excluded: {sorted(set(a.special), key=natural_key)}")
    print(f"\nHELD-OUT SET: {'+'.join(chosen)}  pos={cpos}  frac={cfrac:.4f} "
          f"(target {a.target:.0%}, dev {abs(cfrac-a.target):.4f})")
    if cpos < a.ci_floor:
        print(f"\n!! CI-VIABILITY WARNING: {cpos} positives < floor {a.ci_floor}. "
              f"Whole-chromosome tier too sparse for this TF — report the hash-bin "
              f"leak-free number only, not a whole-chromosome CI.")
    else:
        print(f"CI-viability OK ({cpos} >= {a.ci_floor} positives).")
    print(f"\nwrote {a.out_prefix}_per_chrom_counts.csv")

if __name__ == "__main__":
    main()
