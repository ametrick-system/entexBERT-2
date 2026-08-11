#!/usr/bin/env python3
"""
probe_frozen_trunk.py -- Tier-0 LOCAL pre-check for the Stage-2 ASB twin head.

Before committing GPU time to Stage-2 fine-tuning, this answers the cheap question: does a
precision-weighted head on top of the FROZEN binding trunk already recover allelic signal?
It is a generalized post-hoc probe -- the same thing Stage-2a (freeze trunk, train head only)
would learn, but fit in closed-ish form on CPU from dumped embeddings, with zero collapse risk.

INPUTS (both produced by one cluster inference pass; nothing else to download):
  --pools     {prefix}_pools.npz from `score_asb.py --eval adastra --dump_embeddings`
              arrays: id (=snp), pool_ref (N,H), pool_alt (N,H), label (N,), leaky (N,)
  --evalset   ctcf_adastra_evalset.csv.gz -- joined by snp for total_cover (depth n) and chr
              (chr drives the leak-free chromosome split; total_cover is the privileged weight n).

WHAT IT DOES:
  target  y = logit((k+0.5)/(n+1)) is NOT available here (ADASTRA has no per-allele k), so the
            probe trains on the BINARY ASB label with a weighted logistic head -- the Tier-0
            decision is "does depth-weighted allelic signal separate ASB from non-ASB better than
            the plain trunk contrast?". (Stage-2 proper trains on the continuous logit target from
            hetSNV k,n; this pre-check only gates whether that is worth running.)
  feature   the twin contrast x = pool_alt - pool_ref  (H-dim), matching the model's twin.
            Two heads are fit and compared:
              linear : logistic regression on x            (== what a 1-layer head can learn)
              mlp    : 1-hidden-layer MLP on [pool_ref, pool_alt]  (2H-dim; an MLP head cannot be
                       rebuilt from the contrast alone, g(a)-g(b) != g(a-b), so it needs both pools)
  weight    w = n_eff(n; s) = n(1+s)/(n+s), normalized to mean 1 (privileged precision weighting).
            Swept over --neff_s (default 0 20 50 100; s=0 => unweighted baseline).
  split     leak-free: fit on all chroms EXCEPT the held-out set, evaluate on the held-out set,
            with trained-bin variants (leaky=True) dropped from BOTH. Reports balanced AUROC
            (negatives subsampled to #pos, seed=1) + bootstrap CI, vs the ~0.68 plain-|contrast|
            probe baseline and the 0.50 collapse floor.

Local, CPU, minutes. No GPU, no cluster. Decision gate: if the best weighted head clears the
~0.68 |Delta| probe by a margin outside its CI, Stage-2 fine-tuning is worth the GPU time.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def neff(n, s):
    """Effective sample size n_eff = n(1+s)/(n+s); saturates at 1+s. s=0 -> all weights 1."""
    n = np.asarray(n, dtype=float)
    if s <= 0:
        return np.ones_like(n)
    return n * (1.0 + s) / (n + s)


def balanced_auroc(score, label, seed=1, n_boot=1000):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos = np.where(label == 1)[0]
    neg = np.where(label == 0)[0]
    m = min(len(pos), len(neg))
    if m < 10:
        return np.nan, np.nan, (np.nan, np.nan), m
    rng = np.random.default_rng(seed)
    if len(neg) >= len(pos):
        sel_neg = rng.choice(neg, size=m, replace=False); sel_pos = pos
    else:
        sel_pos = rng.choice(pos, size=m, replace=False); sel_neg = neg
    idx = np.concatenate([sel_pos, sel_neg])
    y = label[idx]; s = score[idx]
    pt = roc_auc_score(y, s); ap = average_precision_score(y, s)
    boots = []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return pt, ap, (lo, hi), m


def load_joined(pools_path, evalset_path):
    z = np.load(pools_path, allow_pickle=True)
    df = pd.DataFrame({
        "snp": z["id"].astype(str),
        "label": z["label"].astype(int),
        "leaky": z["leaky"].astype(bool),
    })
    pool_ref = z["pool_ref"].astype(np.float64)
    pool_alt = z["pool_alt"].astype(np.float64)
    assert len(df) == len(pool_ref) == len(pool_alt), "pools npz arrays are not row-aligned"

    ev = pd.read_csv(evalset_path, usecols=["snp", "chr", "total_cover"])
    ev["snp"] = ev["snp"].astype(str)
    merged = df.merge(ev, on="snp", how="left")
    n_missing = int(merged["total_cover"].isna().sum())
    if n_missing:
        print(f"[join] WARNING: {n_missing}/{len(merged)} variants had no total_cover/chr match; dropping them.")
        keep = ~merged["total_cover"].isna()
        merged = merged.loc[keep].reset_index(drop=True)
        pool_ref = pool_ref[keep.to_numpy()]
        pool_alt = pool_alt[keep.to_numpy()]
    merged["n"] = merged["total_cover"].astype(float)
    print(f"[load] {len(merged)} variants  (pos={int(merged.label.sum())}, "
          f"leaky={int(merged.leaky.sum())}, chroms={merged.chr.nunique()})")
    return merged, pool_ref, pool_alt


def fit_eval_head(kind, Xtr, ytr, wtr, Xte, seed):
    """Fit a weighted head on (Xtr,ytr,wtr), return P(ASB) on Xte. kind in {linear, mlp}."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)
    if kind == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr_s, ytr, sample_weight=wtr)
    elif kind == "mlp":
        # sklearn MLPClassifier has no sample_weight; approximate precision weighting by
        # resampling the training set with probability proportional to w (deterministic seed).
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True,
                            random_state=seed)
        rng = np.random.default_rng(seed)
        p = wtr / wtr.sum()
        idx = rng.choice(len(Xtr_s), size=len(Xtr_s), replace=True, p=p)
        clf.fit(Xtr_s[idx], ytr[idx])
    else:
        raise ValueError(kind)
    return clf.predict_proba(Xte_s)[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", required=True, help="{prefix}_pools.npz from score_asb --dump_embeddings")
    ap.add_argument("--evalset", required=True, help="ctcf_adastra_evalset.csv.gz (join for depth + chr)")
    ap.add_argument("--heldout_chroms", nargs="+", default=["chr5", "chr12"],
                    help="chromosomes held out for evaluation (fit on the rest)")
    ap.add_argument("--neff_s", nargs="+", type=float, default=[0.0, 20.0, 50.0, 100.0],
                    help="saturation caps to sweep; s=0 is the unweighted baseline")
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp"], choices=["linear", "mlp"])
    ap.add_argument("--probe_baseline", type=float, default=0.68,
                    help="the plain binding-trunk |Delta| probe AUROC to beat (reference line)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="frozen_trunk_probe")
    args = ap.parse_args()

    df, pool_ref, pool_alt = load_joined(args.pools, args.evalset)
    contrast = pool_alt - pool_ref

    heldout = set(args.heldout_chroms)
    is_test = df["chr"].isin(heldout).to_numpy()
    clean = ~df["leaky"].to_numpy()             # drop trained-bin variants from BOTH sides
    tr = (~is_test) & clean
    te = is_test & clean
    print(f"[split] leak-free: train {int(tr.sum())} (pos {int(df.label[tr].sum())}) on "
          f"non-{sorted(heldout)}; test {int(te.sum())} (pos {int(df.label[te].sum())}) on {sorted(heldout)}")

    y = df["label"].to_numpy()
    n = df["n"].to_numpy()

    # reference: plain |contrast| ranker on the held-out set (the ~0.68 probe analog)
    absc = np.linalg.norm(contrast, axis=1)  # magnitude of the twin contrast
    pt0, ap0, ci0, m0 = balanced_auroc(absc[te], y[te], seed=args.seed)
    print(f"\n[reference] |contrast| ranker (held-out, leak-free): "
          f"AUROC={pt0:.4f} CI[{ci0[0]:.4f},{ci0[1]:.4f}] n_pos={m0}  "
          f"(probe baseline to beat ~{args.probe_baseline})")

    feats = {"linear": contrast, "mlp": np.hstack([pool_ref, pool_alt])}
    results = [{"head": "abs_contrast", "s": None, "auroc": pt0,
                "auroc_lo": ci0[0], "auroc_hi": ci0[1], "n_pos": int(m0)}]
    print(f"\n{'head':8s} {'s':>6s}  {'AUROC':>7s}  {'95% CI':>17s}  n_pos")
    for head in args.heads:
        X = feats[head]
        for s in args.neff_s:
            w = neff(n[tr], s)
            w = w / w.mean()
            proba = fit_eval_head(head, X[tr], y[tr], w, X[te], args.seed)
            pt, apr, ci, m = balanced_auroc(proba, y[te], seed=args.seed)
            print(f"{head:8s} {s:6.0f}  {pt:7.4f}  [{ci[0]:6.4f},{ci[1]:6.4f}]  {m}")
            results.append({"head": head, "s": s, "auroc": pt,
                            "auroc_lo": ci[0], "auroc_hi": ci[1], "n_pos": int(m)})

    best = max((r for r in results if r["head"] != "abs_contrast"),
               key=lambda r: (r["auroc"] if not np.isnan(r["auroc"]) else -1))
    print(f"\n[best weighted head] {best['head']} s={best['s']}: AUROC={best['auroc']:.4f} "
          f"CI[{best['auroc_lo']:.4f},{best['auroc_hi']:.4f}]")
    verdict = ("GO: clears the probe baseline lower-bound" if best["auroc_lo"] > args.probe_baseline
               else "MARGINAL: overlaps the probe baseline -- Stage-2 may not add much"
               if best["auroc"] >= args.probe_baseline
               else "STOP: does not reach the probe baseline")
    print(f"[verdict] {verdict}  (baseline {args.probe_baseline}, |contrast| ref {pt0:.4f})")

    with open(f"{args.out}_results.json", "w") as f:
        json.dump({"heldout_chroms": sorted(heldout), "probe_baseline": args.probe_baseline,
                   "abs_contrast_auroc": pt0, "results": results,
                   "best": best, "verdict": verdict}, f, indent=2)
    pd.DataFrame(results).to_csv(f"{args.out}_results.csv", index=False)
    print(f"\n[done] wrote {args.out}_results.json + {args.out}_results.csv")


if __name__ == "__main__":
    main()
