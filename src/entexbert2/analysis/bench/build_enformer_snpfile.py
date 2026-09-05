#!/usr/bin/env python
"""
Build the SNP input + label sidecar for re-scoring a benchmark DNN (Enformer/Sei/DeepSEA/DeepFun)
on entexBERT-2's OWN leak-free ADASTRA test variants, so every model is judged on the identical
variant set. Emits:
  --out_snp    : one `chr_pos_ref_alt` per line (exactly the format hdm2020/benchmark's
                 <model>.predict.py expects; hg38 coords -> point predict.py at hg38.fa, no liftover).
  --out_labels : snp,chr,pos,ref,alt,label,total_cover  (for score_enformer_asb.py).

The leak filter and coordinate convention are IMPORTED from entexbert2.score_asb, so the variant
set is exactly the one entexBERT-2's leak-free AUROC was computed on (no drift).
"""
import argparse, numpy as np, pandas as pd
from asb_stat import load_adastra, seen_bins_from_meta, flag_leaky   # vendored copy of score_asb stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True, help="ADASTRA evalset (chr,pos,ref,alt,snp,total_cover,label)")
    ap.add_argument("--train_coords", nargs="*", default=[], help="train.meta.csv dev.meta.csv (leak filter)")
    ap.add_argument("--bin_size", type=int, default=100_000)
    ap.add_argument("--neg_ratio", type=float, default=3.0, help="negatives per positive to score (pool for balancing/covmatch)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_snp", required=True)
    ap.add_argument("--out_labels", required=True)
    args = ap.parse_args()

    ev = load_adastra(args.eval_csv)
    seen = seen_bins_from_meta(args.train_coords, args.bin_size)
    leaky = flag_leaky(ev, seen, args.bin_size, "chr", "pos", pos_is_1based=True)  # ADASTRA is 1-based
    ev = ev.loc[~leaky].copy()
    print(f"[snpfile] leak-free rows: {len(ev)} (pos={int((ev.label==1).sum())}, neg={int((ev.label==0).sum())})")

    pos = ev[ev.label == 1]
    neg = ev[ev.label == 0]
    n_neg = min(len(neg), int(round(args.neg_ratio * len(pos))))
    neg = neg.sample(n=n_neg, random_state=args.seed)
    out = pd.concat([pos, neg], ignore_index=True)
    # 'snp' is already chr_pos_ref_alt; rebuild defensively so it can't drift from chr/pos/ref/alt
    out["snp"] = out.chr.astype(str) + "_" + out.pos.astype(int).astype(str) + "_" + out.ref.astype(str) + "_" + out.alt.astype(str)
    out = out.drop_duplicates("snp")
    print(f"[snpfile] scoring set: {len(out)} variants ({int((out.label==1).sum())} pos + {int((out.label==0).sum())} neg)")

    with open(args.out_snp, "w") as f:
        f.write("\n".join(out["snp"].tolist()) + "\n")
    cols = ["snp", "chr", "pos", "ref", "alt", "label"] + (["total_cover"] if "total_cover" in out.columns else [])
    out[cols].to_csv(args.out_labels, index=False)
    print(f"[snpfile] wrote {args.out_snp} and {args.out_labels}")

if __name__ == "__main__":
    main()
