#!/usr/bin/env python
"""eval_utils.py -- shared post-hoc EVAL/ANALYSIS helpers: hetSNV loading + the leak filter.
Imported by score_asb (the eval engine) and emit_leakfree_loci (the 3_ism allow-list preflight), so
both use the IDENTICAL leak filter -> ISM loci and honest-eval loci stay aligned. Leaf module
(pandas/numpy/os only, no entexbert2 imports) so it can be imported from anywhere without a cycle.
Add further shared analysis helpers here as they come up.
"""
import os
import json
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score, average_precision_score

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}


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
# window building, scoring, pooling, AUROC (shared by score_individual + score_crossdonor)
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


def score_pairs(df, seqs1, seqs2, keep, args, snp_col):
    from entexbert2.model_io import run_inference   # lazy: keeps eval_utils torch-free for build_asb_ism_windows
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


def pool_hetsnv_tissues(df):
    """Pool hetSNV rows per LOCUS across tissues to match the tissue-pooled TRAINING label (hap_counts):
    sum hap1/hap2 counts, recompute signed_log_count_ratio, label = any-sig (max over the locus's
    tissues), total_reads = summed. One row per locus (tissue='pooled'). The head emits ONE
    tissue-agnostic score per locus, so this is the train-matched eval; per-tissue rows cap a
    per-locus model near chance when ASB varies by tissue."""
    keys = [k for k in ["chr", "ref_start", "ref_end", "ref_allele", "hap1_allele",
                        "hap2_allele", "donor", "assay"] if k in df.columns]
    g = df.groupby(keys, sort=False).agg(
        hap1_count=("hap1_count", "sum"), hap2_count=("hap2_count", "sum"),
        imbalance_significance=("imbalance_significance", "max"),
        n_tissues=("tissue", "nunique")).reset_index()
    g["total_reads"] = g["hap1_count"] + g["hap2_count"]
    g["signed_log_count_ratio"] = np.log2((g["hap1_count"] + 0.5) / (g["hap2_count"] + 0.5))
    g["label"] = g["imbalance_significance"].astype(int)
    g["tissue"] = "pooled"
    return g


def balanced_auroc(score, label, seed=1, n_boot=1000, cover=None, n_qbins=20):
    # cover=None -> random balance (original behavior). cover given -> COVERAGE-MATCHED balance:
    # subsample the larger class so its coverage (depth) distribution matches the smaller class,
    # removing the depth/coverage confound (coverage-alone AUROC -> ~0.5). Any AUROC left is real
    # allelic skill, not a depth shortcut.
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    m = min(len(pos_idx), len(neg_idx))
    if m < 10:
        return np.nan, np.nan, (np.nan, np.nan), m
    rng = np.random.default_rng(seed)
    if cover is None:
        if len(neg_idx) >= len(pos_idx):
            sel_neg = rng.choice(neg_idx, size=m, replace=False); sel_pos = pos_idx
        else:
            sel_pos = rng.choice(pos_idx, size=m, replace=False); sel_neg = neg_idx
    else:
        # match the LARGER class's log-coverage distribution to the SMALLER class (quantile bins)
        lc = np.log1p(np.clip(np.asarray(cover, dtype=float), 0, None))
        small = pos_idx if len(pos_idx) <= len(neg_idx) else neg_idx
        large = neg_idx if len(pos_idx) <= len(neg_idx) else pos_idx
        edges = np.quantile(lc[small], np.linspace(0, 1, n_qbins + 1))
        edges[0] -= 1e-6; edges[-1] += 1e-6
        sb = np.digitize(lc[small], edges); lb = np.digitize(lc[large], edges)
        picks = []
        for b in np.unique(sb):
            pool = large[lb == b]; want = int((sb == b).sum())
            if len(pool):
                picks.append(rng.choice(pool, size=want, replace=len(pool) < want))
        sel_large = np.concatenate(picks) if picks else large[:0]
        if len(pos_idx) <= len(neg_idx):
            sel_pos, sel_neg = small, sel_large
        else:
            sel_pos, sel_neg = sel_large, small
    m = min(len(sel_pos), len(sel_neg))
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
