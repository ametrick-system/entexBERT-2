#!/usr/bin/env python
"""
Fair-comparison re-scorer for a benchmark DNN (Enformer here) on entexBERT-2's OWN ADASTRA test set.

Takes the per-variant, per-track delta TSV from hdm2020/benchmark's enformer.predict.py and computes
AUROC under a 2x2 decomposition that isolates WHY the benchmark's headline number is higher than a
fair number would be:

  track choice x depth control
  ------------ x -------------
  (A) best-track-on-test   x  no covmatch   <- reproduces the benchmark's inflated method (winner's curse)
  (B) best-track-on-test   x  covmatch
  (C) pre-specified mean   x  no covmatch   <- honest track choice (no test peeking)
  (D) pre-specified mean   x  covmatch      <- the fully fair number to put next to entexBERT-2

All AUROCs use entexbert2.score_asb.balanced_auroc (IDENTICAL statistic to entexBERT-2's own eval);
the leak filter uses the same flag_leaky/seen_bins_from_meta. Report full + leak_free regimes.

Delta TSV format (from enformer.predict.py, header=False): col0 = snp (chr_pos_ref_alt),
cols 1..5313 = per-track delta for Enformer human track index 0..5312.
"""
import argparse, json, numpy as np, pandas as pd
from asb_stat import balanced_auroc, seen_bins_from_meta, flag_leaky   # vendored copy of score_asb stats

def enformer_tf_tracks(targets_xlsx, tf):
    """Return the 0-based Enformer human-track indices that are ChIP-TF for `tf` (e.g. CTCF, EP300).
    TF name is parsed from `description` exactly as hdm2020/benchmark's TFrename.R does:
    filter assay_subtype=='ChIP-TF', TF = description.split(':')[1]."""
    tg = pd.read_excel(targets_xlsx)
    chip = tg[tg["assay_subtype"].astype(str) == "ChIP-TF"].copy()
    def tf_of(desc):
        parts = str(desc).split(":")
        return parts[1].strip() if len(parts) > 1 else ""
    chip["TF"] = chip["description"].map(tf_of)
    chip["TF"] = chip["TF"].str.replace("eGFP-", "", regex=False).str.replace("3xFLAG-", "", regex=False)
    hits = chip[chip["TF"].str.upper() == tf.upper()]
    idx = sorted(int(i) for i in hits["index"].tolist())
    print(f"[tracks] {tf}: {len(idx)} ChIP-TF Enformer tracks -> indices {idx[:8]}{'...' if len(idx)>8 else ''}")
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_tsv", required=True, help="enformer.predict.py output (snp + 5313 track deltas, no header)")
    ap.add_argument("--targets_xlsx", required=True, help="Enformer.targets.human.xlsx")
    ap.add_argument("--labels_csv", required=True, help="from build_enformer_snpfile.py: snp,chr,pos,ref,alt,label,total_cover")
    ap.add_argument("--tf", required=True, help="CTCF | EP300")
    ap.add_argument("--train_coords", nargs="*", default=[])
    ap.add_argument("--bin_size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap iters for reported CIs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- load deltas: col0=snp, cols 1..N = track 0..N-1
    d = pd.read_csv(args.delta_tsv, sep="\t", header=None)
    d = d.rename(columns={0: "snp"})
    track_cols = {c - 1: c for c in d.columns if c != "snp"}   # track index -> dataframe column
    lab = pd.read_csv(args.labels_csv)
    df = lab.merge(d, on="snp", how="inner")
    print(f"[score] merged {len(df)}/{len(lab)} labelled variants with Enformer deltas "
          f"(pos={int((df.label==1).sum())}, neg={int((df.label==0).sum())})")

    tf_idx = enformer_tf_tracks(args.targets_xlsx, args.tf)
    tf_idx = [i for i in tf_idx if i in track_cols]
    if not tf_idx:
        raise SystemExit(f"[score] no {args.tf} tracks present in the delta TSV")
    cols = [track_cols[i] for i in tf_idx]
    absA = df[cols].abs().to_numpy()                 # (n_var, n_tracks)
    label = df["label"].to_numpy().astype(int)
    cover = df["total_cover"].to_numpy() if "total_cover" in df.columns else None
    mean_track = absA.mean(axis=1)                    # pre-specified aggregate (no test peeking)

    # ---- leak flag
    seen = seen_bins_from_meta(args.train_coords, args.bin_size)
    leaky = flag_leaky(df, seen, args.bin_size, "chr", "pos", pos_is_1based=True)
    regimes = [("full", np.ones(len(df), bool))]
    if leaky.any():
        regimes.append(("leak_free", ~leaky))
        print(f"[leakage] {leaky.sum()}/{len(df)} in a seen bin ({int(leaky[label==1].sum())} pos)")

    def au(score, mask, covmatch, n_boot):
        cov = cover[mask] if (covmatch and cover is not None) else None
        pt, aupr, (lo, hi), m = balanced_auroc(score[mask], label[mask], seed=args.seed, cover=cov, n_boot=n_boot)
        return pt, lo, hi, m

    rows = []
    for rname, mask in regimes:
        for covmatch in ([False, True] if cover is not None else [False]):
            # (C/D) pre-specified mean track -- full bootstrap for its CI
            pt, lo, hi, m = au(mean_track, mask, covmatch, args.n_boot)
            rows.append(dict(tf=args.tf, regime=rname, covmatch=covmatch, track_choice="prespecified_mean",
                             auroc=pt, lo=lo, hi=hi, n_pos=m))
            # (A/B) best-track-on-test: scan point-AUROC (no bootstrap) to select ON this eval set,
            # then bootstrap ONLY the winning track for its CI (fast + the CI belongs to the reported track).
            best_pt, best_j, best_i = -1.0, -1, -1
            for j, i in enumerate(tf_idx):
                pt_j, _, _, _ = au(absA[:, j], mask, covmatch, 0)
                if pt_j == pt_j and pt_j > best_pt:
                    best_pt, best_j, best_i = pt_j, j, i
            bpt, blo, bhi, bm = au(absA[:, best_j], mask, covmatch, args.n_boot)
            rows.append(dict(tf=args.tf, regime=rname, covmatch=covmatch, track_choice=f"best_on_test(idx{best_i})",
                             auroc=bpt, lo=blo, hi=bhi, n_pos=bm))

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Enformer fair-comparison decomposition ===")
    print(res.to_string(index=False))
    print("\nKey rows: 'best_on_test' + covmatch=False  ~ the benchmark's method (inflated);")
    print("          'prespecified_mean' + covmatch=True (leak_free)  = the fair number vs entexBERT-2.")

if __name__ == "__main__":
    main()
