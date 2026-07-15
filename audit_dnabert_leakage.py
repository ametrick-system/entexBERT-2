#!/usr/bin/env python3
"""
audit_dnabert_leakage.py — quantify EXACT-SEQUENCE leakage across DNABERT train/dev/test splits.

The DNABERT k-mer format is space-separated k-mers + a tab + a binary label:
    GCT CTC TCG ... \t 1
This tool reconstructs the underlying nucleotide sequence from the k-mers (first k-mer in full,
then the last base of each subsequent k-mer), hashes it, and measures how many sequences are
SHARED across splits. A sequence appearing in both train and test means the model can memorize it
in training and be scored on the identical input at test -> inflated test metrics.

It reports, for every split PAIR, the overlap count and — the number that matters — the fraction
of the TEST set that also appears in TRAIN (train u dev). It also flags LABEL CONFLICTS (same
sequence, different label) and handles the paper's multiple test files (test.txt, test2..testN.txt).

Usage
-----
  # explicit files:
  python audit_dnabert_leakage.py --train train.txt --dev dev.txt --test test.txt [test2.txt ...] \
      --out_prefix enc001_ctcf

  # or a directory of one (donor,assay) prefix (auto-discovers *_train/_dev/_test*.txt):
  python audit_dnabert_leakage.py --dir output/3 --prefix enc001_CTCF --out_prefix enc001_ctcf

Outputs (under --out_dir, default '.')
  <out_prefix>_leakage_summary.csv   per-pair overlap counts + fractions
  <out_prefix>_leakage.png           bar plot: fraction of each split leaked from train(+dev)
"""
import argparse, glob, hashlib, os, sys
import numpy as np
import pandas as pd


def kmer_line_to_seq(line):
    """'GCT CTC ...\\t1' -> (nucleotide_seq, label:int) or None. Reconstruct: kmers[0] + last base of each next."""
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
    """Return DataFrame[hash, label, seqlen] for one file; skips a header line if present."""
    recs = []
    with open(path) as fh:
        for i, ln in enumerate(fh):
            if i == 0 and ln.lower().startswith("sequence"):   # tolerate a 'sequence\tlabel' header
                continue
            r = kmer_line_to_seq(ln)
            if r is None:
                continue
            seq, lab = r
            recs.append((hashlib.sha1(seq.encode()).hexdigest(), lab, len(seq)))
    if not recs:
        raise ValueError(f"No parseable rows in {path}")
    return pd.DataFrame(recs, columns=["hash", "label", "seqlen"])


def discover(dir_, prefix):
    """Find <prefix>_train.txt / _dev.txt / _test*.txt in a directory."""
    def find(suffixglob):
        hits = sorted(glob.glob(os.path.join(dir_, f"{prefix}*{suffixglob}")))
        return hits
    train = find("_train.txt")
    dev   = find("_dev.txt")
    tests = sorted(set(find("_test.txt") + find("_test[0-9]*.txt")))
    if not (train and dev and tests):
        raise SystemExit(f"Could not find train/dev/test for prefix {prefix!r} in {dir_} "
                         f"(train={train}, dev={dev}, tests={tests})")
    return train[0], dev[0], tests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train"); ap.add_argument("--dev"); ap.add_argument("--test", nargs="+")
    ap.add_argument("--dir"); ap.add_argument("--prefix")
    ap.add_argument("--out_dir", default="."); ap.add_argument("--out_prefix", default="dnabert")
    args = ap.parse_args()

    if args.dir and args.prefix:
        train_p, dev_p, test_ps = discover(args.dir, args.prefix)
    elif args.train and args.dev and args.test:
        train_p, dev_p, test_ps = args.train, args.dev, args.test
    else:
        ap.error("provide either --dir + --prefix, or --train + --dev + --test <files...>")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"train: {train_p}\ndev:   {dev_p}\ntest:  {test_ps}\n")

    train = load_split(train_p); dev = load_split(dev_p)
    tests = {os.path.basename(p): load_split(p) for p in test_ps}

    train_set = set(train["hash"]); dev_set = set(dev["hash"])
    traindev = train_set | dev_set

    # label map for conflict detection (first-seen label per hash across ALL splits)
    all_df = pd.concat([train.assign(split="train"), dev.assign(split="dev")]
                       + [t.assign(split=name) for name, t in tests.items()], ignore_index=True)
    lab_by_hash = all_df.groupby("hash")["label"].agg(lambda s: s.nunique())
    conflict_hashes = set(lab_by_hash[lab_by_hash > 1].index)

    rows = []
    def pair(name_a, set_a, df_b, name_b):
        hb = set(df_b["hash"])
        inter = set_a & hb
        n_b = len(df_b)
        # rows of b that are leaked (count rows, not unique, since duplicates inflate per-row)
        leaked_rows = int(df_b["hash"].isin(set_a).sum())
        confl = len(inter & conflict_hashes)
        rows.append({
            "from": name_a, "to": name_b,
            "to_rows": n_b, "to_unique": df_b["hash"].nunique(),
            "shared_unique_seqs": len(inter),
            "leaked_rows": leaked_rows,
            "leaked_frac_of_rows": leaked_rows / max(n_b, 1),
            "label_conflict_seqs": confl,
        })

    # within-split duplication (memorization surface even inside one file)
    for name, df in [("train", train), ("dev", dev)] + list(tests.items()):
        n, u = len(df), df["hash"].nunique()
        rows.append({"from": name, "to": name+" (within)", "to_rows": n, "to_unique": u,
                     "shared_unique_seqs": n-u, "leaked_rows": n-u,
                     "leaked_frac_of_rows": (n-u)/max(n,1), "label_conflict_seqs": np.nan})

    # cross-split — the leakage that inflates the test metric
    pair("train", train_set, dev, "dev")
    for name, t in tests.items():
        pair("train", train_set, t, name)                     # test leaked from train
        pair("train+dev", traindev, t, name + " (vs train+dev)")

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_leakage_summary.csv")
    summary.to_csv(out_csv, index=False)

    # headline numbers
    print("=== SPLIT SIZES ===")
    print(f"  train {len(train):7d} rows / {train['hash'].nunique():7d} unique")
    print(f"  dev   {len(dev):7d} rows / {dev['hash'].nunique():7d} unique")
    for name, t in tests.items():
        print(f"  {name:14s} {len(t):7d} rows / {t['hash'].nunique():7d} unique")
    print("\n=== TEST LEAKED FROM TRAIN(+DEV) — the metric-inflating number ===")
    tl = summary[summary["to"].str.contains(r"vs train\+dev")]
    for _, r in tl.iterrows():
        print(f"  {r['to']:32s} {r['leaked_rows']:6d}/{r['to_rows']:6d} test rows "
              f"= {r['leaked_frac_of_rows']*100:5.1f}% leaked  "
              f"(label-conflicting: {int(r['label_conflict_seqs'])})")

    # ---- plot: fraction of each TEST file leaked from train+dev, plus dev-from-train ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FOCAL, GREY = "#c1442e", "#9a9a9a"

    plot_df = summary[summary["to"].str.contains(r"vs train\+dev")].copy()
    plot_df["label"] = plot_df["to"].str.replace(r" \(vs train\+dev\)", "", regex=True)
    dev_row = summary[(summary["from"]=="train") & (summary["to"]=="dev")]
    labels = list(plot_df["label"]) + ["dev"]
    fracs  = list(plot_df["leaked_frac_of_rows"]) + [float(dev_row["leaked_frac_of_rows"].iloc[0])]
    fracs_pct = [f*100 for f in fracs]

    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(labels)+2), 4))
    colors = [FOCAL if "test" in l else GREY for l in labels]
    bars = ax.bar(range(len(labels)), fracs_pct, color=colors, width=0.7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("% of split's rows also in train(+dev)")
    ax.set_title("Exact-sequence leakage into held-out splits", loc="left")
    ax.set_ylim(0, max(fracs_pct + [1]) * 1.18)
    for b, v in zip(bars, fracs_pct):
        ax.text(b.get_x()+b.get_width()/2, v + max(fracs_pct+[1])*0.02, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=6)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_leakage.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out_csv}\nwrote {out_png}")


if __name__ == "__main__":
    main()
