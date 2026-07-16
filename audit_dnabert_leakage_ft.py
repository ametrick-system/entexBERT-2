#!/usr/bin/env python3
"""
audit_dnabert_leakage_ft.py — exact-sequence leakage audit for the DNABERT FINETUNING data tree
(data/ft), which is the data actually used to finetune. Layout:

    <base_dir>/<k>/<assay>/<train_indiv>/
        train.txt          # k-mer lines: 'GCT CTC ...\t<label>'  (may have a 'sequence\tlabel' header)
        val.txt            # the dev/validation split (NOT named dev.txt here)
        test.txt           # the primary test set
        1/test.txt ... N/test.txt   # N reshuffled test resamples (each a separate random test draw)
        cached_*_dnaprom   # DNABERT tokenized caches — IGNORED (derived from the .txt)

For each (k, assay, train_indiv) it reconstructs the nucleotide sequence from the k-mers, hashes it,
and measures how much of each held-out split (test.txt and every <n>/test.txt) is an EXACT DUPLICATE
of a train+val sequence. Aggregates to one CSV + a per-k heatmap (assay x train_indiv).

Usage
-----
  python audit_dnabert_leakage_ft.py \
      --base_dir /home/asm242/palmer_scratch/as/ccre/dnabert_model/data/ft \
      --k_values 3 4 5 6 \
      --out_dir leakage_audit_ft --out_prefix entex_ft
"""
import argparse, glob, hashlib, os, re, sys
import numpy as np
import pandas as pd


def kmer_line_to_seq(line):
    line = line.rstrip("\n").rstrip("\r")
    if "\t" not in line:
        return None
    kmers_str, label = line.rsplit("\t", 1)
    kmers = kmers_str.split()
    if not kmers:
        return None
    try:
        lab = int(str(label).strip())
    except ValueError:
        return None
    seq = kmers[0] + "".join(km[-1] for km in kmers[1:])
    return seq, lab


def load_split(path):
    recs = []
    with open(path) as fh:
        for i, ln in enumerate(fh):
            if i == 0 and ln.lower().startswith("sequence"):   # tolerate a header line
                continue
            r = kmer_line_to_seq(ln)
            if r is None:
                continue
            recs.append((hashlib.sha1(r[0].encode()).hexdigest(), r[1]))
    if not recs:
        return None
    return pd.DataFrame(recs, columns=["hash", "label"])


def test_files_for(indiv_dir):
    """primary test.txt + every immediate numeric-subdir test.txt, sorted."""
    tests = []
    primary = os.path.join(indiv_dir, "test.txt")
    if os.path.exists(primary):
        tests.append(primary)
    for sub in sorted(glob.glob(os.path.join(indiv_dir, "[0-9]*")),
                      key=lambda p: (len(os.path.basename(p)), os.path.basename(p))):
        t = os.path.join(sub, "test.txt")
        if os.path.isdir(sub) and os.path.exists(t):
            tests.append(t)
    return tests


def audit_indiv(indiv_dir):
    train_p = os.path.join(indiv_dir, "train.txt")
    val_p   = os.path.join(indiv_dir, "val.txt")
    if not (os.path.exists(train_p) and os.path.exists(val_p)):
        return None
    test_ps = test_files_for(indiv_dir)
    if not test_ps:
        return None
    train = load_split(train_p); val = load_split(val_p)
    if train is None or val is None:
        return None
    train_set = set(train["hash"]); val_set = set(val["hash"])
    trainval = train_set | val_set

    frames = [train.assign(split="train"), val.assign(split="val")]
    test_dfs = {}
    for tp in test_ps:
        t = load_split(tp)
        if t is not None:
            rel = os.path.relpath(tp, indiv_dir)
            test_dfs[rel] = t
            frames.append(t.assign(split=rel))
    if not test_dfs:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    nlab = all_df.groupby("hash")["label"].nunique()
    conflict_hashes = set(nlab[nlab > 1].index)

    # test_files_for returns test.txt FIRST, then numeric subdirs 1..N in order, so the
    # first entry is the within-individual primary test and the rest are the paper's
    # cross-donor resamples (SNPs exclusive to other donors), each EVALUATED SEPARATELY.
    names = list(test_dfs.keys())
    fracs, rowcounts, conflicts = [], [], []
    per_set = {}          # name -> leaked fraction, kept individually
    for name in names:
        t = test_dfs[name]
        leaked = int(t["hash"].isin(trainval).sum())
        frac = leaked / max(len(t), 1)
        fracs.append(frac); rowcounts.append(len(t))
        conflicts.append(len(set(t["hash"]) & trainval & conflict_hashes))
        per_set[name] = frac
    val_leak = int(val["hash"].isin(train_set).sum()) / max(len(val), 1)

    # split primary (within-individual) from the cross-donor resample sets
    primary_name = names[0] if names and os.path.basename(names[0]) == "test.txt" \
                   and os.path.dirname(names[0]) == "" else None
    if primary_name is not None:
        primary_leak = per_set[primary_name]
        cross = [per_set[n] for n in names if n != primary_name]
    else:                                   # fallback: no bare test.txt found
        primary_leak = float(np.min(fracs))
        cross = fracs

    out = {
        "train_rows": len(train), "train_unique": train["hash"].nunique(),
        "val_rows": len(val), "val_unique": val["hash"].nunique(),
        "n_test_files": len(test_dfs), "n_crossdonor_sets": len(cross),
        "test_rows_mean": float(np.mean(rowcounts)),
        "val_leaked_frac": val_leak,
        # within-individual held-out test.txt (matches the paper's ~0.79 regime)
        "primary_test_leaked_frac": float(primary_leak),
        # per-set-then-average across the cross-donor resamples (matches the paper's
        # 0.6876 regime: 10 sets each evaluated separately, then averaged)
        "crossdonor_leaked_frac_mean": float(np.mean(cross)),
        "crossdonor_leaked_frac_std": float(np.std(cross, ddof=1)) if len(cross) > 1 else 0.0,
        "crossdonor_leaked_frac_min": float(np.min(cross)),
        "crossdonor_leaked_frac_max": float(np.max(cross)),
        # kept for backward-compat / spread across ALL 11 files
        "test_leaked_frac_mean": float(np.mean(fracs)),
        "test_leaked_frac_min": float(np.min(fracs)),
        "test_leaked_frac_max": float(np.max(fracs)),
        "train_within_dup_frac": (len(train) - train["hash"].nunique()) / max(len(train), 1),
        "label_conflict_seqs_in_test_mean": float(np.mean(conflicts)),
    }
    # each individual cross-donor set as its own column (crossdonor_set_01 ...)
    cross_names = [n for n in names if n != primary_name] if primary_name is not None else names
    for i, n in enumerate(cross_names, 1):
        out[f"crossdonor_set_{i:02d}"] = float(per_set[n])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True, help="data/ft dir with <k>/<assay>/<train_indiv> layout")
    ap.add_argument("--k_values", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--out_dir", default="leakage_audit_ft")
    ap.add_argument("--out_prefix", default="entex_ft")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for k in args.k_values:
        kdir = os.path.join(args.base_dir, str(k))
        if not os.path.isdir(kdir):
            print(f"[skip] {kdir} not found"); continue
        for assay in sorted(os.listdir(kdir)):
            adir = os.path.join(kdir, assay)
            if not os.path.isdir(adir):
                continue
            for indiv in sorted(os.listdir(adir)):
                idir = os.path.join(adir, indiv)
                if not os.path.isdir(idir):
                    continue
                res = audit_indiv(idir)
                if res is None:
                    continue
                res.update({"k": k, "assay": assay, "train_indiv": indiv})
                rows.append(res)
                print(f"  k={k} {assay:16s} indiv={indiv:3s} "
                      f"primary {res['primary_test_leaked_frac']*100:5.1f}%  "
                      f"cross-donor {res['crossdonor_leaked_frac_mean']*100:5.1f}%"
                      f"(+/-{res['crossdonor_leaked_frac_std']*100:.0f})  "
                      f"val {res['val_leaked_frac']*100:5.1f}%  "
                      f"(train {res['train_rows']}/{res['train_unique']}u, {res['n_crossdonor_sets']} cross sets)")

    if not rows:
        sys.exit("No datasets audited — check --base_dir / layout (expects <k>/<assay>/<indiv>/train.txt,val.txt,test.txt).")
    df = pd.DataFrame(rows)
    lead = ["k", "assay", "train_indiv", "train_rows", "train_unique", "val_rows", "val_unique",
            "n_test_files", "n_crossdonor_sets", "test_rows_mean", "val_leaked_frac",
            "primary_test_leaked_frac", "crossdonor_leaked_frac_mean", "crossdonor_leaked_frac_std",
            "crossdonor_leaked_frac_min", "crossdonor_leaked_frac_max",
            "test_leaked_frac_mean", "test_leaked_frac_min", "test_leaked_frac_max",
            "train_within_dup_frac", "label_conflict_seqs_in_test_mean"]
    set_cols = sorted([c for c in df.columns if c.startswith("crossdonor_set_")])
    cols = lead + set_cols
    df = df[[c for c in cols if c in df.columns]].sort_values(["k", "assay", "train_indiv"])
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_leakage_by_dataset.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== OVERALL (two regimes, matched to the paper) ===")
    print(f"datasets audited: {len(df)}")
    print(f"WITHIN-INDIVIDUAL primary test.txt leak: {df['primary_test_leaked_frac'].mean()*100:.1f}%  "
          f"(range {df['primary_test_leaked_frac'].min()*100:.0f}-{df['primary_test_leaked_frac'].max()*100:.0f}%)")
    print(f"CROSS-DONOR 10-set average leak (per-set-then-average): "
          f"{df['crossdonor_leaked_frac_mean'].mean()*100:.1f}%  "
          f"(range {df['crossdonor_leaked_frac_mean'].min()*100:.0f}-{df['crossdonor_leaked_frac_mean'].max()*100:.0f}%)")
    print("\n=== per-assay (avg over k and train_indiv): within | cross-donor ===")
    pa = (df.groupby("assay")[["primary_test_leaked_frac", "crossdonor_leaked_frac_mean"]]
            .mean().sort_values("crossdonor_leaked_frac_mean", ascending=False) * 100)
    for a, r in pa.iterrows():
        print(f"  {a:16s} within {r['primary_test_leaked_frac']:5.1f}%   cross-donor {r['crossdonor_leaked_frac_mean']:5.1f}%")
    print(f"\nwrote {out_csv}")

    # heatmap: one panel per k, assay (rows) x train_indiv (cols)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    ks = sorted(df["k"].unique())
    indivs = sorted(df["train_indiv"].unique(), key=lambda x: (len(x), x))
    assays = sorted(df["assay"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.0*len(ks)+1.2, 0.42*len(assays)+1.6),
                             squeeze=False, constrained_layout=True)
    metric = "crossdonor_leaked_frac_mean"
    vmax = max(1.0, df[metric].max()*100)
    im = None
    for j, k in enumerate(ks):
        ax = axes[0][j]; sub = df[df["k"] == k]
        M = np.full((len(assays), len(indivs)), np.nan)
        for _, r in sub.iterrows():
            M[assays.index(r["assay"]), indivs.index(r["train_indiv"])] = r[metric]*100
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(indivs))); ax.set_xticklabels(indivs, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(assays)))
        ax.set_yticklabels(assays if j == 0 else [], fontsize=6)
        ax.set_title(f"k={k}", fontsize=8)
        for a in range(len(assays)):
            for d in range(len(indivs)):
                if not np.isnan(M[a, d]):
                    ax.text(d, a, f"{M[a,d]:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if M[a, d] > vmax*0.6 else "black")
    fig.suptitle("Cross-donor exact-sequence leakage in FINETUNING data "
                 "(mean over 10 sets, each evaluated separately; % of test rows also in train+val)",
                 y=1.02, fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("% leaked", fontsize=7)
    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_leakage_heatmap.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
