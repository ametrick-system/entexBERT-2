#!/usr/bin/env python3
"""
score_asb.py -- score a trained entexBERT-2 model on an ASB eval set by the twin contrast
    mu = head(window1) - head(window2)
and report balanced AUROC (negatives subsampled to #positives, seed=1) with a bootstrap CI.

Two eval sets (--eval):
  adastra : the ADASTRA benchmark (Han et al. 2024, Fig 3A). window1 = ref allele, window2 =
            alt allele; |mu| vs the binary ASB label, placed against the published models.
  hetsnv  : the EN-TEx het-SNV set, reported PER DONOR and PER TISSUE. window1 = hap1 allele,
            window2 = hap2 allele; keeps the tissue dimension and a continuous effect size
            (signed_log_count_ratio) so we also get magnitude calibration Spearman(|mu|, |lfc|)
            and signed calibration Spearman(mu, lfc).

SIGN CONVENTION (matches model.py / model_io.py / the training label): window1 is ref/hap1,
window2 is alt/hap2, so mu = logit P(hap1). For hetsnv the signed Spearman(mu, signed_log_count_
ratio) is therefore expected POSITIVE (both are hap1-vs-hap2 in the same direction).

LEAKAGE: pass --train_coords fold0/train.meta.csv fold0/dev.meta.csv to flag/drop eval variants
whose 100kb bin was seen in training (bins are GENOMIC/reference-based, so a seen bin leaks at
that locus regardless of donor). Do NOT pass test.meta.csv.

--dump_embeddings writes {out}_pools.npz (snp/id, pool_ref, pool_alt, label, leaky) -- the raw
per-window pools for the frozen-trunk pre-check (probe_frozen_trunk.py). An MLP head cannot be
rebuilt from the contrast alone (g(a)-g(b) != g(a-b)), so both pools are dumped.

Cluster inputs (all small; no big ADASTRA download for --eval adastra):
  --checkpoint_dir  trained model .../runs/reg (has run_config.json)
  --eval_csv        ctcf_adastra_evalset.csv.gz          (--eval adastra)
  --hetsnv_tsv      /home/asm242/entex_data/hetSNVs.tsv  (--eval hetsnv)
  --ref_fasta       /home/asm242/reference_genome/hg38.fa
  --reference_csv   ctcf_benchmark_reference_auroc.csv   (--eval adastra, for placement)
Run from the repo root in the eb2 env (needs entexbert2 + pyfaidx + sklearn + scipy).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

from entexbert2.model_io import run_inference

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}


# ----------------------------------------------------------------------
# Shared: window construction. window1 = allele1 base, window2 = allele2 base,
# both substituted at the SNV center of the hg38 window. Returns row-aligned
# (seqs1, seqs2, keep_mask). pos_is_1based toggles ADASTRA (1-based) vs hetSNV
# (0-based ref_start).
# ----------------------------------------------------------------------
def build_windows(df, ref_fasta, left_bp, right_bp,
                  chrom_col, pos_col, refbase_col, a1_col, a2_col, pos_is_1based):
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    seqs1, seqs2, keep = [], [], []
    n_oob = n_badchrom = n_refmismatch = 0
    for chrom, posv, ref_a, a1, a2 in zip(
        df[chrom_col], df[pos_col], df[refbase_col], df[a1_col], df[a2_col]
    ):
        if chrom not in fa:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_badchrom += 1; continue
        p0 = (int(posv) - 1) if pos_is_1based else int(posv)   # 0-based SNV position
        start = p0 - left_bp
        end = p0 + right_bp + 1                                 # half-open; length = win
        clen = len(fa[chrom])
        if start < 0 or end > clen:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != win:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_oob += 1; continue
        center = left_bp
        if seq[center] != str(ref_a).upper():
            n_refmismatch += 1
        seqs1.append(seq[:center] + str(a1).upper() + seq[center + 1:])
        seqs2.append(seq[:center] + str(a2).upper() + seq[center + 1:])
        keep.append(True)
    print(f"[windows] built {sum(keep)}/{len(df)}  "
          f"(dropped: {n_oob} out-of-bounds, {n_badchrom} bad-chrom; "
          f"hg38-base!=ref on {n_refmismatch} kept rows)")
    return seqs1, seqs2, np.array(keep, dtype=bool)


# ----------------------------------------------------------------------
# Shared: balanced AUROC (subsample the larger class to the smaller, seed=1),
# with a bootstrap CI over the balanced set.
# ----------------------------------------------------------------------
def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    m = min(len(pos_idx), len(neg_idx))
    if m < 10:
        return np.nan, np.nan, (np.nan, np.nan), m
    rng = np.random.default_rng(seed)
    if len(neg_idx) >= len(pos_idx):
        sel_neg = rng.choice(neg_idx, size=m, replace=False); sel_pos = pos_idx
    else:
        sel_pos = rng.choice(pos_idx, size=m, replace=False); sel_neg = neg_idx
    idx = np.concatenate([sel_pos, sel_neg])
    y = label[idx]; s = score[idx]
    point = roc_auc_score(y, s)
    aupr = average_precision_score(y, s)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return point, aupr, (lo, hi), m


# ----------------------------------------------------------------------
# Shared: leakage — collect the SEEN (chr, bin) set from meta sidecars.
# ----------------------------------------------------------------------
def seen_bins_from_meta(coord_files, bin_size):
    seen = set()
    for path in coord_files or []:
        if not os.path.exists(path):
            print(f"[leakage] WARNING: {path} not found; skipping."); continue
        tc = pd.read_csv(path)
        chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
        pos_col = ("SNV" if "SNV" in tc.columns else "pos" if "pos" in tc.columns
                   else "anchor" if "anchor" in tc.columns else None)
        if pos_col is None:
            print(f"[leakage] {path}: no SNV/pos/anchor column ({list(tc.columns)[:6]}...); skipping.")
            continue
        before = len(seen)
        seen |= set(zip(tc[chrom_col].astype(str), (tc[pos_col].astype(int) // bin_size)))
        print(f"[leakage] {os.path.basename(path)}: +{len(seen)-before} bins, {len(seen)} seen total")
    return seen


def flag_leaky(df, seen, bin_size, chrom_col, pos_col, pos_is_1based):
    if not seen:
        return np.zeros(len(df), dtype=bool)
    if pos_is_1based:
        bins = ((df[pos_col].astype(int) - 1) // bin_size)
    else:
        bins = (df[pos_col].astype(int) // bin_size)
    pairs = list(zip(df[chrom_col].astype(str), bins))
    return np.array([b in seen for b in pairs])


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_adastra(eval_csv):
    ev = pd.read_csv(eval_csv)
    ev["label"] = ev["label"].astype(int)
    print(f"[load] ADASTRA eval: {len(ev)} rows "
          f"(pos={int((ev.label==1).sum())}, neg={int((ev.label==0).sum())})")
    return ev


def load_hetsnv(path, assay, min_total_reads):
    usecols = ["chr", "ref_start", "ref_end", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "tissue", "assay", "cA", "cC", "cG", "cT",
               "ref_allele_ratio", "p_betabinom", "imbalance_significance"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    df = df.reset_index(drop=True)

    def base_count(row, allele_col):
        col = _BASECOL.get(str(row[allele_col]).upper())
        return float(row[col]) if col in row and pd.notna(row[col]) else 0.0

    df["hap1_count"] = df.apply(lambda r: base_count(r, "hap1_allele"), axis=1)
    df["hap2_count"] = df.apply(lambda r: base_count(r, "hap2_allele"), axis=1)
    df["total_reads"] = df["hap1_count"] + df["hap2_count"]
    df["signed_log_count_ratio"] = np.log2((df["hap1_count"] + 0.5) / (df["hap2_count"] + 0.5))
    if min_total_reads:
        n0 = len(df)
        df = df[df["total_reads"] >= min_total_reads].reset_index(drop=True)
        print(f"[filter] total_reads>={min_total_reads}: {len(df)}/{n0} rows kept")
    df["label"] = df["imbalance_significance"].astype(int)
    return df


# ----------------------------------------------------------------------
# Scoring helper: run twin inference, attach delta/abs_delta, optionally dump pools.
# window1 = allele1 window (ref/hap1), window2 = allele2 window (alt/hap2).
# ----------------------------------------------------------------------
def score_pairs(df, seqs1, seqs2, keep, args, snp_col):
    df = df.loc[keep].reset_index(drop=True)
    pairs = [[s1, s2] for s1, s2 in zip(np.asarray(seqs1)[keep], np.asarray(seqs2)[keep])]
    print(f"[score] running twin inference on {len(pairs)} variants "
          f"(dump_embeddings={args.dump_embeddings})...")
    if args.dump_embeddings:
        logits, _emb, pool_ref, pool_alt, run_config = run_inference(
            args.checkpoint_dir, pairs, args.batch_size, args.device,
            json.loads(args.overrides) if args.overrides else {}, dump_pools=True)
    else:
        logits, _emb, run_config = run_inference(
            args.checkpoint_dir, pairs, args.batch_size, args.device,
            json.loads(args.overrides) if args.overrides else {})
        pool_ref = pool_alt = None
    delta = np.asarray(logits, dtype=float).reshape(len(pairs), -1)[:, 0]
    df["delta"] = delta
    df["abs_delta"] = np.abs(delta)
    return df, run_config, pool_ref, pool_alt


def dump_pools_npz(out, df, pool_ref, pool_alt, id_col):
    path = f"{out}_pools.npz"
    np.savez_compressed(
        path,
        id=df[id_col].astype(str).to_numpy(),
        pool_ref=pool_ref.astype(np.float32),
        pool_alt=pool_alt.astype(np.float32),
        label=df["label"].astype(int).to_numpy(),
        leaky=df["leaky"].astype(bool).to_numpy(),
    )
    print(f"[dump] wrote {path}  (pool_ref {pool_ref.shape}, pool_alt {pool_alt.shape})")


# ----------------------------------------------------------------------
# ADASTRA eval
# ----------------------------------------------------------------------
def eval_adastra(args):
    ev = load_adastra(args.eval_csv)
    seqs1, seqs2, keep = build_windows(
        ev, args.ref_fasta, args.left_bp, args.right_bp,
        chrom_col="chr", pos_col="pos", refbase_col="ref",
        a1_col="ref", a2_col="alt", pos_is_1based=True)
    ev, run_config, pool_ref, pool_alt = score_pairs(ev, seqs1, seqs2, keep, args, "snp")

    seen = seen_bins_from_meta(args.train_coords, args.bin_size)
    leaky = flag_leaky(ev, seen, args.bin_size, "chr", "pos", pos_is_1based=True)
    ev["leaky"] = leaky
    if seen:
        n_pos_leak = int(leaky[ev["label"].to_numpy() == 1].sum())
        print(f"[leakage] {leaky.sum()}/{len(ev)} eval variants ({100*leaky.mean():.2f}%) "
              f"in a seen bin; {n_pos_leak} ASB-positive. Dropped for leak_free.")
    else:
        print("[leakage] no --train_coords; full-set number only. "
              "For leak-free, pass fold0/train.meta.csv fold0/dev.meta.csv.")

    def report(tag, sub):
        pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(), sub["label"].to_numpy())
        print(f"[AUROC:{tag}] balanced {m} pos + {m} neg  AUROC={pt:.4f} "
              f"95%CI[{lo:.4f},{hi:.4f}]  AUPRC={aupr:.4f}")
        return {"regime": tag, "auroc": pt, "auroc_lo": lo, "auroc_hi": hi,
                "auprc": aupr, "n_pos": int(m)}

    results = [report("full", ev)]
    if leaky.any():
        results.append(report("leak_free", ev.loc[~leaky]))

    if args.reference_csv and os.path.exists(args.reference_csv):
        ref = pd.read_csv(args.reference_csv).sort_values("CTCF_AUROC", ascending=False)
        eb2 = results[-1]["auroc"]
        print(f"\n=== entexBERT-2 vs benchmark models (CTCF AUROC, {len(ref)} with valid CTCF) ===")
        placed = False
        for _, r in ref.iterrows():
            if not placed and eb2 >= r["CTCF_AUROC"]:
                print(f"  >>> entexBERT-2 (this run)   {eb2:.4f}  <<<"); placed = True
            print(f"      {r['model']:20s} {r['family']:9s} {r['CTCF_AUROC']:.4f}")
        if not placed:
            print(f"  >>> entexBERT-2 (this run)   {eb2:.4f}  (below all listed) <<<")

    pv_cols = ["chr", "pos", "ref", "alt", "snp", "label", "delta", "abs_delta", "leaky"]
    ev[pv_cols].to_csv(f"{args.out}_perVariant.csv.gz", index=False, compression="gzip")
    with open(f"{args.out}_metrics.json", "w") as f:
        json.dump({"eval": "adastra", "results": results,
                   "checkpoint_dir": args.checkpoint_dir,
                   "run_config_task": run_config.get("task"),
                   "n_scored": int(len(ev)),
                   "left_bp": args.left_bp, "right_bp": args.right_bp,
                   "leaky_excluded": bool(leaky.any())}, f, indent=2)
    if args.dump_embeddings:
        dump_pools_npz(args.out, ev, pool_ref, pool_alt, "snp")
    print(f"\n[done] wrote {args.out}_metrics.json + {args.out}_perVariant.csv.gz")


# ----------------------------------------------------------------------
# hetSNV eval  (per donor, per tissue)
# ----------------------------------------------------------------------
def eval_hetsnv(args):
    full = load_hetsnv(args.hetsnv_tsv, args.assay, args.min_total_reads)
    print(f"[load] {len(full)} hetSNV rows over donors={sorted(full.donor.unique())}")
    seen = seen_bins_from_meta(args.train_coords, args.bin_size)

    all_rows, summary = [], []
    for donor in args.donors:
        d = full[full["donor"] == donor].reset_index(drop=True)
        if len(d) == 0:
            print(f"\n[{donor}] no rows; skipping."); continue
        print(f"\n===== donor {donor}: {len(d)} rows "
              f"(pos={int((d.label==1).sum())}, neg={int((d.label==0).sum())}) =====")
        seqs1, seqs2, keep = build_windows(
            d, args.ref_fasta, args.left_bp, args.right_bp,
            chrom_col="chr", pos_col="ref_start", refbase_col="ref_allele",
            a1_col="hap1_allele", a2_col="hap2_allele", pos_is_1based=False)
        d, _cfg, pool_ref, pool_alt = score_pairs(d, seqs1, seqs2, keep, args, "chr")
        d["donor"] = donor
        d["donor_kind"] = "matched" if donor == args.matched_donor else "cross-donor"
        leaky = flag_leaky(d, seen, args.bin_size, "chr", "ref_start", pos_is_1based=False)
        d["leaky"] = leaky
        if seen:
            kind = d["donor_kind"].iloc[0]
            print(f"[leakage] {leaky.sum()}/{len(d)} {donor} ({kind}) variants in a seen bin "
                  f"({int(leaky[d.label.to_numpy()==1].sum())} positive).")

        def rep(tag, sub):
            pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(), sub["label"].to_numpy())
            mag = (spearmanr(sub["abs_delta"], sub["signed_log_count_ratio"].abs()).correlation
                   if len(sub) > 10 else np.nan)
            sgn = (spearmanr(sub["delta"], sub["signed_log_count_ratio"]).correlation
                   if len(sub) > 10 else np.nan)
            print(f"  [{donor}:{tag}] AUROC={pt:.4f} CI[{lo:.4f},{hi:.4f}] AUPRC={aupr:.4f} "
                  f"n_pos={m} | mag_Spearman={mag:.4f} signed_Spearman={sgn:.4f}")
            summary.append(dict(donor=donor, donor_kind=d["donor_kind"].iloc[0], regime=tag,
                                tissue="ALL", auroc=pt, auroc_lo=lo, auroc_hi=hi, auprc=aupr,
                                n_pos=m, mag_spearman=mag, signed_spearman=sgn))
        rep("full", d)
        if leaky.any():
            rep("leak_free", d.loc[~leaky])

        d_tis = d.loc[~leaky] if leaky.any() else d
        tis_regime = "leak_free" if leaky.any() else "full"
        print(f"  --- per-tissue [{tis_regime}] (>= {args.min_tissue_pos} pos & neg) ---")
        for tis, sub in d_tis.groupby("tissue"):
            npos = int((sub.label == 1).sum()); nneg = int((sub.label == 0).sum())
            if npos < args.min_tissue_pos or nneg < args.min_tissue_pos:
                continue
            pt, aupr, (lo, hi), m = balanced_auroc(sub["abs_delta"].to_numpy(), sub["label"].to_numpy())
            mag = spearmanr(sub["abs_delta"], sub["signed_log_count_ratio"].abs()).correlation
            print(f"    {tis:32s} AUROC={pt:.4f} n_pos={m} mag_Sp={mag:.4f}")
            summary.append(dict(donor=donor, donor_kind=d["donor_kind"].iloc[0], regime=tis_regime,
                                tissue=tis, auroc=pt, auroc_lo=lo, auroc_hi=hi, auprc=aupr,
                                n_pos=m, mag_spearman=mag, signed_spearman=np.nan))
        all_rows.append((d, pool_ref, pool_alt))

    if all_rows:
        alld = pd.concat([r[0] for r in all_rows], ignore_index=True)
        pv_cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
                   "donor", "donor_kind", "tissue", "label", "total_reads",
                   "signed_log_count_ratio", "delta", "abs_delta", "leaky"]
        alld[pv_cols].to_csv(f"{args.out}_perVariant.csv.gz", index=False, compression="gzip")
        print(f"\n[write] {args.out}_perVariant.csv.gz  ({len(alld)} scored variants)")
        if args.dump_embeddings:
            pr = np.concatenate([r[1] for r in all_rows], axis=0)
            pa = np.concatenate([r[2] for r in all_rows], axis=0)
            alld["_id"] = (alld["chr"].astype(str) + ":" + alld["ref_start"].astype(str)
                           + "_" + alld["donor"].astype(str))
            dump_pools_npz(args.out, alld, pr, pa, "_id")

    sm = pd.DataFrame(summary)
    sm.to_csv(f"{args.out}_summary.csv", index=False)
    print(f"[write] {args.out}_summary.csv")
    show = sm[sm.tissue == "ALL"] if len(sm) else sm
    if len(show):
        print("\n=== donor-level (ALL-tissue) summary ===")
        print(show[["donor", "donor_kind", "regime", "auroc", "n_pos",
                    "mag_spearman", "signed_spearman"]].to_string(index=False))


# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", choices=["adastra", "hetsnv"], required=True)
    ap.add_argument("--checkpoint_dir", required=True, help=".../runs/reg (has run_config.json)")
    ap.add_argument("--ref_fasta", required=True, help="hg38.fa")
    # adastra
    ap.add_argument("--eval_csv", default=None, help="ctcf_adastra_evalset.csv.gz (--eval adastra)")
    ap.add_argument("--reference_csv", default=None, help="ctcf_benchmark_reference_auroc.csv")
    # hetsnv
    ap.add_argument("--hetsnv_tsv", default=None, help="hetSNVs.tsv (--eval hetsnv)")
    ap.add_argument("--assay", default="CTCF")
    ap.add_argument("--donors", nargs="+", default=["ENC-001", "ENC-002", "ENC-003", "ENC-004"])
    ap.add_argument("--matched_donor", default="ENC-002")
    ap.add_argument("--min_total_reads", type=int, default=20)
    ap.add_argument("--min_tissue_pos", type=int, default=20)
    # shared
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="fold0/train.meta.csv fold0/dev.meta.csv (leak filter; NOT test.meta.csv)")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--drop_leaky", action="store_true")
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overrides", default=None, help="JSON dict of run_config overrides")
    ap.add_argument("--dump_embeddings", action="store_true",
                    help="also write {out}_pools.npz (raw ref/alt pools for probe_frozen_trunk.py)")
    ap.add_argument("--out", default=None, help="output prefix (default derived from --eval)")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"entexbert2_{args.eval}_result"
    if args.eval == "adastra" and not args.eval_csv:
        ap.error("--eval adastra requires --eval_csv")
    if args.eval == "hetsnv" and not args.hetsnv_tsv:
        ap.error("--eval hetsnv requires --hetsnv_tsv")
    return args


def main():
    args = parse_args()
    if args.eval == "adastra":
        eval_adastra(args)
    else:
        eval_hetsnv(args)


if __name__ == "__main__":
    main()
