"""
asb_stat.py -- VERBATIM copy of the ASB scoring statistic from entexbert2/score_asb.py
(balanced_auroc, seen_bins_from_meta, flag_leaky, load_adastra), vendored here so the
benchmark re-scoring harness runs in a lean env (numpy/pandas/sklearn only) WITHOUT importing
the full entexbert2 model stack. This guarantees every re-scored model uses the IDENTICAL
statistic entexBERT-2 was evaluated with. Keep in sync with score_asb.py if that changes.
Source: github.com/ametrick-system/entexBERT-2 src/entexbert2/score_asb.py
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

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

