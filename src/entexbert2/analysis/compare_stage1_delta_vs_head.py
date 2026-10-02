#!/usr/bin/env python
"""compare_stage1_delta_vs_head.py -- sanity check: does the Stage-2 ASB head beat the Stage-1 trunk's
allelic delta at predicting the held-out ASB set?

For each cell, on the SAME Stage-2 test.csv loci (sequence1/sequence2 = ref/alt or hap1/hap2,
label = AS/non-AS -- the exact split eval_results.json uses):
  stage1_delta : run the Stage-1 binding TRUNK on BOTH alleles -> |mu1 - mu2|, AUROC vs label
                 (the "binding just changed" baseline; no ASB head)
  stage2_head  : the trained ASB head's ell, AUROC vs label
Prints a table sorted WEAKEST-Stage-1-first and writes a tidy CSV (one row/cell) for a later figure.
If stage2_head > stage1_delta even where stage1_delta ~ 0.5, the head adds ASB-specific signal beyond
a raw binding-magnitude change -- which is the whole point of Stage 2.

  python compare_stage1_delta_vs_head.py --exp_root experiments --donor ENC-002 --arm ref \
      --assays CTCF,EP300,POLR2A,ATAC,H3K4me3,H3K27ac,H3K9me3,H3K36me3,H3K27me3,POLR2AphosphoS5 \
      --n_boot 1000 --out_csv stage1delta_vs_head_ENC-002.csv
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from entexbert2.model_io import run_inference


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")


def trunk_mu(trunk_dir, seqs, bs, dev):
    arr = np.asarray(run_inference(trunk_dir, list(seqs), bs, dev)[0])
    return arr[:, 0] if arr.ndim > 1 else arr.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--donor", default="ENC-002")
    ap.add_argument("--arm", choices=["ref", "personal"], default="ref")
    ap.add_argument("--assays", required=True, help="comma-separated")
    ap.add_argument("--s", type=int, default=20); ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_boot", type=int, default=0, help="paired bootstrap on the improvement (0 = skip)")
    ap.add_argument("--out_csv", default="stage1delta_vs_head.csv")
    a = ap.parse_args()

    rows = []
    for assay in [x for x in a.assays.split(",") if x]:
        cell = os.path.join(a.exp_root, f"{assay.lower()}_{a.donor}")
        test = os.path.join(cell, f"stage2_{a.arm}_data", "test.csv")
        trunk = os.path.join(cell, "stage1_trunk", "runs", "reg")
        head = os.path.join(cell, f"stage2_{a.arm}", "runs", f"clf_s{a.s}_seed{a.seed}")
        miss = [p for p in (test, trunk, head) if not os.path.exists(p)]
        if miss:
            print(f"[skip] {assay}/{a.donor}: missing {miss}"); continue
        df = pd.read_csv(test)
        if not {"sequence1", "sequence2", "label"}.issubset(df.columns):
            print(f"[skip] {assay}: test.csv cols {list(df.columns)[:6]}"); continue
        y = df["label"].astype(int).to_numpy()
        s1 = df["sequence1"].astype(str).tolist(); s2 = df["sequence2"].astype(str).tolist()

        mu1 = trunk_mu(trunk, s1, a.batch_size, a.device)     # Stage-1 binding, allele 1
        mu2 = trunk_mu(trunk, s2, a.batch_size, a.device)     # Stage-1 binding, allele 2
        d1 = np.abs(mu1 - mu2)                                 # |Delta mu| -- Stage-1-only ASB baseline
        ell = np.asarray(run_inference(head, [[x, z] for x, z in zip(s1, s2)],
                                       a.batch_size, a.device)[0]).reshape(-1)   # Stage-2 head

        au1, au2 = auroc(y, d1), auroc(y, ell)
        row = dict(assay=assay, donor=a.donor, arm=a.arm, n=len(y), n_pos=int(y.sum()),
                   auroc_stage1_delta=au1, auroc_stage2_head=au2, improvement=au2 - au1)
        if a.n_boot and y.min() != y.max():
            rng = np.random.default_rng(0); diffs = []
            for _ in range(a.n_boot):
                idx = rng.integers(0, len(y), len(y))
                if y[idx].min() == y[idx].max():
                    continue
                diffs.append(auroc(y[idx], ell[idx]) - auroc(y[idx], d1[idx]))
            diffs = np.array(diffs)
            row["boot_frac_head_better"] = float((diffs > 0).mean())
            row["boot_ci_lo"] = float(np.percentile(diffs, 2.5))
            row["boot_ci_hi"] = float(np.percentile(diffs, 97.5))
        rows.append(row)
        print(f"[{assay}] n={len(y)} pos={int(y.sum())} | stage1delta={au1:.3f}  head={au2:.3f}  "
              f"improvement={au2 - au1:+.3f}")

    if not rows:
        print("[warn] no cells scored"); return
    out = pd.DataFrame(rows).sort_values("auroc_stage1_delta").reset_index(drop=True)  # weakest Stage-1 first
    out.to_csv(a.out_csv, index=False)
    print("\n=== sorted weakest Stage-1 first ===")
    print(out.to_string(index=False))
    print(f"\n[write] {a.out_csv}")
    print(f"head beats stage1-delta in {int((out.improvement > 0).sum())}/{len(out)} cells; "
          f"mean improvement {out.improvement.mean():+.3f}")


if __name__ == "__main__":
    main()
