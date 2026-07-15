#!/usr/bin/env python3
"""
audit_dnabert_leakage_batch.py — quantify EXACT-SEQUENCE leakage across ALL DNABERT
train/dev/test splits in an EN-TEx generate_seq output tree, for every (k, donor, assay).

Directory layout expected (the paper's generate_seq/output):
    <base_dir>/<k>/<donor>_<assay>_train.txt
                   <donor>_<assay>_dev.txt
                   <donor>_<assay>_test.txt, _test2.txt ... _test10.txt
                   <donor>_<assay>_sequences.tsv   (ignored)

For each (k, donor, assay) it reconstructs the nucleotide sequence from the k-mers, hashes it,
and measures how much of each held-out split is an EXACT DUPLICATE of a train(+dev) sequence —
the overlap that turns a reported test metric into memorization recall. Aggregates everything into
one master CSV and a per-k heatmap (assay x donor) of mean test-leakage %.

Usage
-----
  python audit_dnabert_leakage_batch.py \
      --base_dir /home/asm242/palmer_scratch/as/ccre/dnabert_preprocessing/generate_seq/output \
      --k_values 3 4 5 6 \
      --out_dir leakage_audit --out_prefix entex

Outputs (under --out_dir)
  <out_prefix>_leakage_by_dataset.csv   one row per (k, donor, assay): sizes + leakage fractions
  <out_prefix>_leakage_heatmap.png      per-k heatmaps: mean % of test rows also in train(+dev)
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
            if i == 0 and ln.lower().startswith("sequence"):
                continue
            r = kmer_line_to_seq(ln)
            if r is None:
                continue
            seq, lab = r
            recs.append((hashlib.sha1(seq.encode()).hexdigest(), lab))
    if not recs:
        return None
    return pd.DataFrame(recs, columns=["hash", "label"])


def discover_prefixes(kdir):
    """Return sorted list of '<donor>_<assay>' prefixes that have a _train.txt in kdir."""
    prefixes = []
    for p in sorted(glob.glob(os.path.join(kdir, "*_train.txt"))):
        base = os.path.basename(p)[:-len("_train.txt")]
        prefixes.append(base)
    return prefixes


def test_files_for(kdir, prefix):
    hits = set(glob.glob(os.path.join(kdir, f"{prefix}_test.txt")))
    hits |= set(glob.glob(os.path.join(kdir, f"{prefix}_test[0-9]*.txt")))
    return sorted(hits)


def audit_one(kdir, prefix):
    """Return a dict of leakage metrics for one (donor,assay) prefix, or None if incomplete."""
    train_p = os.path.join(kdir, f"{prefix}_train.txt")
    dev_p   = os.path.join(kdir, f"{prefix}_dev.txt")
    test_ps = test_files_for(kdir, prefix)
    if not (os.path.exists(train_p) and os.path.exists(dev_p) and test_ps):
        return None
    train = load_split(train_p); dev = load_split(dev_p)
    if train is None or dev is None:
        return None
    train_set = set(train["hash"]); dev_set = set(dev["hash"])
    traindev = train_set | dev_set

    # label conflicts across everything
    frames = [train.assign(split="train"), dev.assign(split="dev")]
    test_dfs = {}
    for tp in test_ps:
        t = load_split(tp)
        if t is not None:
            test_dfs[os.path.basename(tp)] = t
            frames.append(t.assign(split=os.path.basename(tp)))
    if not test_dfs:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    nuniq_lab = all_df.groupby("hash")["label"].nunique()
    conflict_hashes = set(nuniq_lab[nuniq_lab > 1].index)

    # per-test-file leaked fraction vs train+dev
    test_leak_fracs, test_rowcounts, test_conflicts = [], [], []
    for name, t in test_dfs.items():
        leaked_rows = int(t["hash"].isin(traindev).sum())
        test_leak_fracs.append(leaked_rows / max(len(t), 1))
        test_rowcounts.append(len(t))
        test_conflicts.append(len(set(t["hash"]) & traindev & conflict_hashes))
    dev_leak = int(dev["hash"].isin(train_set).sum()) / max(len(dev), 1)

    return {
        "donor": prefix.split("_", 1)[0],
        "assay": prefix.split("_", 1)[1] if "_" in prefix else prefix,
        "prefix": prefix,
        "train_rows": len(train), "train_unique": train["hash"].nunique(),
        "dev_rows": len(dev), "dev_unique": dev["hash"].nunique(),
        "n_test_files": len(test_dfs),
        "test_rows_mean": float(np.mean(test_rowcounts)),
        "dev_leaked_frac": dev_leak,
        "test_leaked_frac_mean": float(np.mean(test_leak_fracs)),
        "test_leaked_frac_min": float(np.min(test_leak_fracs)),
        "test_leaked_frac_max": float(np.max(test_leak_fracs)),
        "train_within_dup_frac": (len(train) - train["hash"].nunique()) / max(len(train), 1),
        "label_conflict_seqs_in_test_mean": float(np.mean(test_conflicts)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True, help="generate_seq/output dir containing <k>/ subfolders")
    ap.add_argument("--k_values", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--out_dir", default="leakage_audit")
    ap.add_argument("--out_prefix", default="entex")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for k in args.k_values:
        kdir = os.path.join(args.base_dir, str(k))
        if not os.path.isdir(kdir):
            print(f"[skip] {kdir} not found"); continue
        prefixes = discover_prefixes(kdir)
        print(f"[k={k}] {len(prefixes)} (donor,assay) prefixes in {kdir}")
        for pref in prefixes:
            res = audit_one(kdir, pref)
            if res is None:
                print(f"  [warn] incomplete: {pref}"); continue
            res["k"] = k
            rows.append(res)
            print(f"  k={k} {pref:28s} test-leak {res['test_leaked_frac_mean']*100:5.1f}%  "
                  f"dev-leak {res['dev_leaked_frac']*100:5.1f}%  "
                  f"(train {res['train_rows']}/{res['train_unique']}u)")

    if not rows:
        sys.exit("No datasets audited — check --base_dir / layout.")
    df = pd.DataFrame(rows)
    cols = ["k", "donor", "assay", "prefix", "train_rows", "train_unique", "dev_rows", "dev_unique",
            "n_test_files", "test_rows_mean", "test_leaked_frac_mean", "test_leaked_frac_min",
            "test_leaked_frac_max", "dev_leaked_frac", "train_within_dup_frac",
            "label_conflict_seqs_in_test_mean"]
    df = df[cols].sort_values(["k", "assay", "donor"])
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_leakage_by_dataset.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== OVERALL ===")
    print(f"datasets audited: {len(df)}")
    print(f"mean test-leakage across all: {df['test_leaked_frac_mean'].mean()*100:.1f}%")
    print(f"min / max dataset test-leakage: {df['test_leaked_frac_mean'].min()*100:.1f}% / "
          f"{df['test_leaked_frac_mean'].max()*100:.1f}%")
    print(f"wrote {out_csv}")

    # ---- heatmaps: one panel per k, assay (rows) x donor (cols), colour = mean test-leak % ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})

    ks = sorted(df["k"].unique())
    donors = sorted(df["donor"].unique())
    assays = sorted(df["assay"].unique())
    ncol = len(ks)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol + 1.2, 0.42 * len(assays) + 1.6),
                             squeeze=False, constrained_layout=True)
    vmax = max(1.0, df["test_leaked_frac_mean"].max() * 100)
    im = None
    for j, k in enumerate(ks):
        ax = axes[0][j]
        sub = df[df["k"] == k]
        M = np.full((len(assays), len(donors)), np.nan)
        for _, r in sub.iterrows():
            M[assays.index(r["assay"]), donors.index(r["donor"])] = r["test_leaked_frac_mean"] * 100
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(donors))); ax.set_xticklabels(donors, rotation=45, ha="right", fontsize=6)
        if j == 0:
            ax.set_yticks(range(len(assays))); ax.set_yticklabels(assays, fontsize=6)
        else:
            ax.set_yticks(range(len(assays))); ax.set_yticklabels([])
        ax.set_title(f"k={k}", fontsize=8)
        for a in range(len(assays)):
            for d in range(len(donors)):
                if not np.isnan(M[a, d]):
                    ax.text(d, a, f"{M[a,d]:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if M[a, d] > vmax*0.6 else "black")
    fig.suptitle("Exact-sequence test-set leakage (% of test rows also in train+dev)", y=1.02, fontsize=9)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("% leaked", fontsize=7)
    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_leakage_heatmap.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
