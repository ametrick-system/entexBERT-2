#!/usr/bin/env python3
"""
build_adastra_pairs.py — turn the ADASTRA CTCF evalset into a PAIRED CSV that
dump_contrast_embeddings.py can consume, so the contrast diagnostic figure can be
made with ADASTRA as the input instead of the EN-TEx within-dataset test.

The distance head compares two windows. On EN-TEx those are hap1/hap2; on ADASTRA
the canonical pairing (matching how the model is SCORED on ADASTRA) is:
    sequence1 = hg38 REFERENCE window (128/128, SNV at center)
    sequence2 = ALT window (same window, alt base substituted at center)
    label     = ADASTRA binary ASB label
So the head distance s = ||P(ref) - P(alt)|| and ell = a*s + b are computed on the
SAME ref/alt pairs the ADASTRA AUROC is measured on -> the figure's ell AUROC will
match the ADASTRA balanced AUROC (prevalence-invariant), making it the visual
companion to the 0.72 headline.

Window construction is copied byte-for-byte from score_ctcf_adastra.build_windows
so the pairs are identical to the scoring pipeline.

LEAK-FREE: pass --train_coords <train.meta.csv> <dev.meta.csv> (this checkpoint's own
sidecars) with --drop_leaky to write only variants whose 100kb bin was NOT seen in
train/dev -- matching the leak-free ADASTRA number. Omit for the full set.

  python build_adastra_pairs.py \
    --eval_csv ctcf_adastra_evalset.csv.gz \
    --ref_fasta $HOME/reference_genome/hg38.fa \
    --out adastra_pairs.csv \
    [--train_coords .../inputs_clf/train.meta.csv .../inputs_clf/dev.meta.csv --drop_leaky]

Then feed adastra_pairs.csv to dump_contrast_embeddings.py (--data_csv) and plot as usual.
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta


def build_windows(eval_df, ref_fasta, left_bp, right_bp):
    """Return (ref_seqs, alt_seqs, keep_mask). pos is 1-based (ADASTRA/VCF).
    Identical to score_ctcf_adastra.build_windows."""
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    ref_seqs, alt_seqs, keep = [], [], []
    n_oob = n_refmismatch = n_badchrom = 0
    for chrom, pos1, ref_a, alt_a in zip(
        eval_df["chr"], eval_df["pos"], eval_df["ref"], eval_df["alt"]
    ):
        if chrom not in fa:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_badchrom += 1; continue
        p0 = int(pos1) - 1
        start = p0 - left_bp
        end = p0 + right_bp + 1
        if start < 0 or end > len(fa[chrom]):
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != win:
            keep.append(False); ref_seqs.append(""); alt_seqs.append("")
            n_oob += 1; continue
        center = left_bp
        if seq[center] != str(ref_a).upper():
            n_refmismatch += 1
        alt_seq = seq[:center] + str(alt_a).upper() + seq[center + 1:]
        ref_seqs.append(seq); alt_seqs.append(alt_seq); keep.append(True)
    print(f"[windows] built {sum(keep)}/{len(eval_df)}  "
          f"(dropped: {n_oob} out-of-bounds, {n_badchrom} bad-chrom; "
          f"ref-base!=hg38 on {n_refmismatch} kept rows)")
    return ref_seqs, alt_seqs, np.array(keep, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_csv", required=True, help="ctcf_adastra_evalset.csv.gz")
    ap.add_argument("--ref_fasta", required=True, help="hg38.fa")
    ap.add_argument("--out", required=True, help="output paired CSV (sequence1,sequence2,label)")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="this checkpoint's own train.meta.csv dev.meta.csv (for --drop_leaky)")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--drop_leaky", action="store_true",
                    help="keep only variants whose 100kb bin was NOT seen in train/dev")
    args = ap.parse_args()

    ev = pd.read_csv(args.eval_csv)
    print(f"[load] ADASTRA eval: {len(ev)} rows "
          f"(pos={int((ev.label==1).sum())}, neg={int((ev.label==0).sum())})")

    ref_seqs, alt_seqs, keep = build_windows(ev, args.ref_fasta, args.left_bp, args.right_bp)
    ev = ev.loc[keep].reset_index(drop=True)
    ref_seqs = list(np.asarray(ref_seqs)[keep])
    alt_seqs = list(np.asarray(alt_seqs)[keep])

    # optional leak-free filter against this checkpoint's own meta sidecars
    if args.train_coords:
        seen = set()
        for path in args.train_coords:
            if not os.path.exists(path):
                print(f"[leakage] WARNING: {path} not found; skipping."); continue
            tc = pd.read_csv(path)
            chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
            pos_col = next((c for c in ("SNV", "pos", "anchor") if c in tc.columns), None)
            if pos_col is None:
                print(f"[leakage] {path}: no SNV/pos/anchor col; skip."); continue
            before = len(seen)
            seen |= set(zip(tc[chrom_col].astype(str), (tc[pos_col].astype(int) // args.bin_size)))
            print(f"[leakage] {os.path.basename(path)}: +{len(seen)-before} bins, {len(seen)} total")
        ev_bins = list(zip(ev["chr"].astype(str), ((ev["pos"].astype(int) - 1) // args.bin_size)))
        leaky = np.array([b in seen for b in ev_bins])
        n_pos_leak = int(leaky[ev["label"].to_numpy() == 1].sum())
        print(f"[leakage] {leaky.sum()}/{len(ev)} in a seen bin ({100*leaky.mean():.2f}%); "
              f"{n_pos_leak} ASB-positive.")
        if args.drop_leaky:
            keep2 = ~leaky
            ev = ev.loc[keep2].reset_index(drop=True)
            ref_seqs = list(np.asarray(ref_seqs)[keep2])
            alt_seqs = list(np.asarray(alt_seqs)[keep2])
            print(f"[leakage] --drop_leaky -> {len(ev)} leak-free rows kept "
                  f"(pos={int((ev.label==1).sum())})")

    out = pd.DataFrame({"sequence1": ref_seqs, "sequence2": alt_seqs,
                        "label": ev["label"].astype(int).to_numpy()})
    out.to_csv(args.out, index=False)
    print(f"[write] {args.out}: {len(out)} pairs "
          f"(pos={int((out.label==1).sum())}, neg={int((out.label==0).sum())})")


if __name__ == "__main__":
    main()
