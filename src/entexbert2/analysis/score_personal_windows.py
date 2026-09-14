#!/usr/bin/env python
"""
Score a trained ASB head on its PERSONAL twin windows -> a perVariant table for the personal
confusion grid. score_asb scores REFERENCE windows only (hg38 substitution); this scores the SAME
personal windows the personal ISM used (<tf>_asb_ism_windows_entex_personal.csv), so the personal
panel's PREDICTION (delta) and its SALIENCY share the personal-sequence context. Reuses
model_io.run_inference (the identical scoring path score_asb / ism use).

Output columns match what fig_ism_confusion_grid expects: chr, ref_start (=hg38 anchor, the join key),
delta (ASB logit), abs_delta, as_label, total_reads, donor_kind.

  python score_personal_windows.py --checkpoint_dir <cell>/stage2_personal/runs/clf_s20_seed20 \
      --windows_csv <cell>/ism/<tf>_asb_ism_windows_entex_personal.csv \
      --out <cell>/eval/score_personal_hetsnv_perVariant.csv.gz
"""
import argparse
import numpy as np, pandas as pd
from entexbert2.model_io import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the trained PERSONAL ASB head")
    ap.add_argument("--windows_csv", required=True, help="*_entex_personal.csv (sequence1/2, as_label, chr, anchor)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor", default="ENC-002", help="stamped into the summary for the honest table")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    df = pd.read_csv(a.windows_csv)
    pairs = [[s1, s2] for s1, s2 in zip(df["sequence1"].astype(str), df["sequence2"].astype(str))]
    logits, _emb, _rc = run_inference(a.checkpoint_dir, pairs,
                                      batch_size=a.batch_size, device=a.device)
    delta = np.asarray(logits).reshape(-1)                 # ASB logit l = a*||z1-z2|| + b (classification)
    out = pd.DataFrame(dict(
        chr=df["chr"].astype(str),
        ref_start=df["anchor"].astype(int),                # hg38 0-based anchor = the grid's join key
        delta=delta, abs_delta=np.abs(delta),
        as_label=df["as_label"].astype(str),
        total_reads=df["total_reads"] if "total_reads" in df.columns else 0.0,
        donor_kind="matched"))                             # personal = same-donor
    out.to_csv(a.out, index=False, compression="gzip")
    print(f"[scored] {len(out)} personal loci -> {a.out}  "
          f"delta range [{delta.min():.3f}, {delta.max():.3f}]  "
          f"AS/nonAS {out.as_label.value_counts().to_dict()}")

    # Honest AUROC, symmetric with score_asb's ref summary. These personal windows are per-locus
    # (any-sig as_label, tissue-pooled) and leak-free BY CONSTRUCTION (built with --restrict_coords
    # off the ref leak-free set), so the number is directly comparable to ref's leak_free[+covmatch].
    # classification score = delta (ell); label = (as_label == "AS").
    from entexbert2.score_asb import balanced_auroc
    y = (out["as_label"].values == "AS").astype(int)
    if int(y.sum()) >= 1 and int((y == 0).sum()) >= 1:
        cov = out["total_reads"].to_numpy(float)
        rows = []
        for tag, cover in [("leak_free", None), ("leak_free", cov)]:
            pt, aupr, (lo, hi), npos = balanced_auroc(delta, y, cover=cover)
            cm = cover is not None
            rows.append(dict(donor=a.donor, donor_kind="matched", regime=tag, cover_matched=cm,
                             tissue="ALL", arm="personal", auroc=pt, auroc_lo=lo, auroc_hi=hi,
                             auprc=aupr, n_pos=npos))
            print(f"  [personal:{tag}{'+covmatch' if cm else ''}] AUROC={pt:.4f} "
                  f"CI[{lo:.4f},{hi:.4f}] AUPRC={aupr:.4f} n_pos={npos}")
        summ = a.out.replace("_perVariant.csv.gz", "_summary.csv")
        pd.DataFrame(rows).to_csv(summ, index=False)
        print(f"[write] {summ}")
    else:
        print("  [personal] only one class present -> skipping AUROC summary")


if __name__ == "__main__":
    main()
