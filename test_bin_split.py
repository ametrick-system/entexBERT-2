#!/usr/bin/env python3
"""
test_bin_split.py — unit test for the pure genomic-bin 3-way split + boundary exclusion.

Run at the entexBERT-2 repo root (where `import entexbert2` resolves), in the eb2 env:
    python test_bin_split.py

Tests (pure numpy/pandas — no model, no GPU, runs in seconds):
  T1  plain pure-bin 3-way ratios ~ 80/10/10
  T2  determinism (same input -> same split)
  T3  order/donor invariance (shuffle rows -> same locus keeps its split)
  T4  co-bin integrity: every (chrom, bin) chunk is entirely in ONE split
  T5  boundary exclusion drops ~2*boundary_bp/bin_size of loci
  T6  after exclusion, NO kept locus's window straddles a bin edge
  T7  backward-compat: bin_test_frac=0 -> chromosome-holdout path unchanged (test == held-out chroms)
"""
import sys, numpy as np, pandas as pd

try:
    from entexbert2.utils import PartitionSpec, assign_split_column, genomic_bin_id
except Exception as e:
    print(f"[FATAL] cannot import entexbert2.utils ({e}); run from repo root in the eb2 env.")
    sys.exit(2)

FAIL = []
def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond: FAIL.append(name)

# ---- synthetic loci: 60k across 22 chromosomes ----
rng = np.random.default_rng(0)
N, NCHR, BS, BP = 60000, 22, 100_000, 256
df = pd.DataFrame({
    "chr": [f"chr{rng.integers(1, NCHR+1)}" for _ in range(N)],
    "SNV": rng.integers(1_000, 50_000_000, size=N),
})

# T1 — plain 3-way ratios
s1 = PartitionSpec(enabled=True, bin_size=BS, bin_test_frac=0.10, bin_dev_frac=0.10)
a1 = assign_split_column(df, s1)
vc = pd.Series(a1).value_counts(normalize=True)
check("T1 plain 3-way ratios ~80/10/10",
      abs(vc.get("train",0)-0.80)<0.02 and abs(vc.get("dev",0)-0.10)<0.02 and abs(vc.get("test",0)-0.10)<0.02,
      f"{vc.round(4).to_dict()}")

# T2 — determinism
check("T2 deterministic", bool((a1 == assign_split_column(df, s1)).all()))

# T3 — order/donor invariance
perm = rng.permutation(N)
a1p = assign_split_column(df.iloc[perm].reset_index(drop=True), s1)
check("T3 order/donor-invariant", bool((a1[perm] == a1p).all()))

# T4 — co-bin integrity
b = [genomic_bin_id(c,p,BS) for c,p in zip(df["chr"], df["SNV"])]
nsplit = pd.DataFrame({"bin":b, "s":a1}).groupby("bin")["s"].nunique()
check("T4 every bin entirely in one split", int((nsplit>1).sum())==0,
      f"bins_with_multiple_splits={int((nsplit>1).sum())}")

# T5 / T6 — boundary exclusion
s2 = PartitionSpec(enabled=True, bin_size=BS, bin_test_frac=0.10, bin_dev_frac=0.10,
                   exclude_boundary=True, boundary_bp=BP)
a2 = assign_split_column(df, s2)
excl = (a2=="__exclude__").mean()
check("T5 boundary exclusion fraction ~2*bp/bin", abs(excl - 2*BP/BS) < 0.002,
      f"excluded={excl:.4f} expected={2*BP/BS:.4f}")
p = df["SNV"].to_numpy(); home = p//BS
straddle = ((p-BP)//BS != home) | ((p+BP)//BS != home)
kept = a2 != "__exclude__"
check("T6 no kept locus straddles a bin edge", int((kept & straddle).sum())==0,
      f"kept_straddlers={int((kept & straddle).sum())}")

# T7 — backward compat: chromosome holdout unchanged
s0 = PartitionSpec(enabled=True, bin_size=BS,
                   fold_assignment={"chr5":0,"chr12":0,"chr21":0}, fold_id=0)
a0 = assign_split_column(df, s0)
test_chroms = set(df["chr"][a0=="test"])
no_exclude = "__exclude__" not in set(a0)
check("T7 chrom-holdout path: test == held-out chroms only, no exclusion",
      test_chroms=={"chr5","chr12","chr21"} and no_exclude,
      f"test_chroms={sorted(test_chroms)}")

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} check(s): {FAIL}")
    sys.exit(1)
print("ALL CHECKS PASSED — bin-split logic is correct; safe to build inputs.")
