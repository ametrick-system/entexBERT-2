#!/usr/bin/env python
"""score_crossdonor.py -- score ONE trained ASB head on every donor's Stage-2 test.csv split, plain
roc_auc on the head logit ell. This is the EXACT metric + split that eval_results.json uses
(trainer.evaluate on test.csv -> roc_auc_score(y, ell)), just with THIS head instead of that donor's own
head -- so the cross-donor number is directly comparable to the within-individual (matched) diagonal you
read from eval_results.json. Leak-clean by construction: the Stage-2 fold is the shared per-assay fold,
so every donor's test.csv holds the same held-out coordinates this head never trained on.

Writes ONE per-head CSV (no shared append -> no array race): assay, train_donor, eval_donor, arm,
donor_kind, auroc, n, n_pos. The matched row (eval_donor == train_donor) is included as a free sanity
check against eval_results.json; the figure uses eval_results.json for the diagonal.

  python -m entexbert2.analysis.score_crossdonor \\
      --checkpoint_dir experiments/ctcf_ENC-001/stage2_ref/runs/clf_s20_seed20 \\
      --assay CTCF --train_donor ENC-001 --arm ref --exp_root experiments \\
      --donors ENC-001 ENC-002 ENC-003 ENC-004 --out experiments/ctcf_ENC-001/eval/crossdonor_ref.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from entexbert2.model_io import run_inference


def score_on(head_dir, test_csv, batch_size, device):
    df = pd.read_csv(test_csv)
    if "sequence1" not in df.columns or "sequence2" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{test_csv}: need sequence1,sequence2,label; got {list(df.columns)[:6]}")
    pairs = [[s1, s2] for s1, s2 in zip(df["sequence1"].astype(str), df["sequence2"].astype(str))]
    logits, _emb, _rc = run_inference(head_dir, pairs, batch_size, device)   # classification -> ell
    ell = np.asarray(logits).reshape(-1)
    y = df["label"].astype(int).to_numpy()
    auroc = float(roc_auc_score(y, ell)) if y.min() != y.max() else float("nan")
    return auroc, len(y), int(y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the trained head (train_donor's arm)")
    ap.add_argument("--assay", required=True)
    ap.add_argument("--train_donor", required=True)
    ap.add_argument("--arm", choices=["ref", "personal"], default="ref")
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--donors", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="per-head output CSV (overwritten, not appended)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = []
    for ed in a.donors:
        tcsv = os.path.join(a.exp_root, f"{a.assay.lower()}_{ed}", f"stage2_{a.arm}_data", "test.csv")
        if not os.path.exists(tcsv):
            print(f"[skip] {a.assay}/{ed} {a.arm}: no test.csv at {tcsv}")
            continue
        au, n, npos = score_on(a.checkpoint_dir, tcsv, a.batch_size, a.device)
        kind = "matched" if ed == a.train_donor else "cross-donor"
        rows.append(dict(assay=a.assay, train_donor=a.train_donor, eval_donor=ed, arm=a.arm,
                         donor_kind=kind, auroc=au, n=n, n_pos=npos))
        print(f"  [{a.assay}] {a.train_donor} -> {ed} ({kind}) {a.arm}: "
              f"AUROC={au:.4f} n={n} n_pos={npos}")
    if not rows:
        print("[warn] no rows produced (no test.csv found)"); return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[write] {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
