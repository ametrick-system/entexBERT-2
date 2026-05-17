#!/usr/bin/env python

import argparse
import random
from pathlib import Path

import pandas as pd
import pysam
import hashlib

def seq_to_kmers(seq, k):
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

def clean_offset_name(offset):
    if offset == "random":
        return "offset_random"
    if offset == "jitter":
        return "offset_jitter"
    offset = int(offset)
    if offset < 0:
        return f"offset_m{abs(offset)}"
    return f"offset_{offset}"


def resolve_chrom(fasta, chrom):
    refs = set(fasta.references)

    if chrom in refs:
        return chrom

    if chrom.startswith("chr") and chrom[3:] in refs:
        return chrom[3:]

    if not chrom.startswith("chr") and f"chr{chrom}" in refs:
        return f"chr{chrom}"

    return None

def extract_window(fasta, chrom, snp_pos, window_size, desired_snp_index):
    chrom2 = resolve_chrom(fasta, chrom)
    if chrom2 is None:
        return None

    start = int(snp_pos) - desired_snp_index
    end = start + window_size

    if start < 0:
        return None

    chrom_len = fasta.get_reference_length(chrom2)
    if end > chrom_len:
        return None

    seq = fasta.fetch(chrom2, start, end).upper()

    if len(seq) != window_size:
        return None

    return seq

def maybe_inject_allele(seq, desired_snp_index, allele):
    allele = str(allele).upper()

    if allele not in {"A", "C", "G", "T"}:
        return None

    seq = list(seq)
    seq[desired_snp_index] = allele
    return "".join(seq)

def assign_group_splits(df, seed, train_frac=0.8, val_frac=0.1):
    """
    Assign train/val/test splits by genomic coordinate group, not by row.
    This prevents the same SNV coordinate from appearing in multiple splits.
    """
    rng = random.Random(seed)

    groups = (
        df[["chr", "ref_start", "ref_end"]]
        .astype(str)
        .agg(":".join, axis=1)
        .unique()
        .tolist()
    )

    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = int(train_frac * n_groups)
    n_val = int(val_frac * n_groups)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    splits = []

    for _, row in df.iterrows():
        gid = f"{row['chr']}:{row['ref_start']}:{row['ref_end']}"

        if gid in train_groups:
            splits.append("train")
        elif gid in val_groups:
            splits.append("val")
        elif gid in test_groups:
            splits.append("test")
        else:
            raise RuntimeError(f"Group {gid} was not assigned to a split.")

    return splits

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--as_tsv", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--assay", default=None)
    parser.add_argument("--donor", default=None)
    parser.add_argument("--tissue", default=None)

    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--offsets", default="0,16,32,64,-16,-32,-64")
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--random_margin", type=int, default=16)
    parser.add_argument("--include_jitter", action="store_true")
    parser.add_argument("--jitter_max_offset", type=int, default=64)

    parser.add_argument("--min_total_reads", type=int, default=10)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--allele_mode",
        choices=["reference", "hap1", "hap2"],
        default="reference",
        help="reference = use hg38 sequence as-is; hap1/hap2 = inject that allele at the SNV position"
    )

    parser.add_argument("--keep_N", action="store_true")

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Reading:", args.as_tsv)
    df = pd.read_csv(args.as_tsv, sep="\t")

    # Basic sanity
    df["ref_start"] = df["ref_start"].astype(int)
    df["ref_end"] = df["ref_end"].astype(int)
    df = df[df["ref_end"] - df["ref_start"] == 1].copy()

    if args.assay is not None:
        df = df[df["assay"] == args.assay].copy()

    if args.donor is not None:
        df = df[df["donor"] == args.donor].copy()

    if args.tissue is not None:
        df = df[df["tissue"] == args.tissue].copy()

    for col in ["cA", "cC", "cG", "cT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["total_reads"] = df[["cA", "cC", "cG", "cT"]].sum(axis=1)
    df = df[df["total_reads"] >= args.min_total_reads].copy()

    df["label"] = df["imbalance_significance"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    print(f"After filters: positives={len(pos)}, negatives={len(neg)}")

    n = min(len(pos), len(neg))
    if args.max_per_class is not None:
        n = min(n, args.max_per_class)

    if n == 0:
        raise ValueError("No balanced examples available after filtering.")

    pos = pos.sample(n=n, random_state=args.seed)
    neg = neg.sample(n=n, random_state=args.seed)

    data = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    data["example_id"] = [
        f"{r['chr']}:{r['ref_start']}:{r['ref_allele']}:{r['hap1_allele']}:{r['hap2_allele']}:{i}"
        for i, r in data.iterrows()
    ]
    data["split"] = assign_group_splits(data, args.seed)

    coord_cols = ["chr", "ref_start", "ref_end"]

    split_counts = data.groupby(coord_cols)["split"].nunique()
    n_leaky_coords = int((split_counts > 1).sum())

    print(f"Coordinate leakage check: {n_leaky_coords} coordinates appear in multiple splits")

    if n_leaky_coords > 0:
        raise RuntimeError("Coordinate leakage detected: same SNV appears in multiple splits.")

    print(f"Selected balanced dataset: {len(data)} examples total ({n} per class)")
    print(data["split"].value_counts())
    print("Label counts by split:")
    print(pd.crosstab(data["split"], data["label"]))

    offsets = [x.strip() for x in args.offsets.split(",") if x.strip()]

    if args.include_random:
        offsets.append("random")

    if args.include_jitter:
        offsets.append("jitter")

    fasta = pysam.FastaFile(args.ref_fasta)

    center_idx = args.window_size // 2
    rng = random.Random(args.seed)

    for offset in offsets:
        offset_dir = out_root / clean_offset_name(offset)
        offset_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "train": open(offset_dir / "train.txt", "w"),
            "val": open(offset_dir / "val.txt", "w"),
            "test": open(offset_dir / "test.txt", "w"),
        }

        meta = open(offset_dir / "metadata.tsv", "w")
        meta.write(
            "example_id\tsplit\tchr\tref_start\tref_end\tlabel\t"
            "desired_snp_index\toffset\tallele_mode\tsequence\n"
        )

        counts = {"train": 0, "val": 0, "test": 0}
        skipped = 0

        for _, row in data.iterrows():
            if offset == "random":
                # SNP can appear almost anywhere in the window, excluding margins.
                per_example_rng = random.Random( # makes same random examples per random seed for reproducibility
                    int(hashlib.md5(f"{args.seed}:{row['example_id']}:random".encode()).hexdigest(), 16)
                )
                desired_snp_index = per_example_rng.randint(
                    args.random_margin,
                    args.window_size - args.random_margin - 1
                )
                actual_offset = desired_snp_index - center_idx

            elif offset == "jitter":
                # SNP appears at a random position within +/- jitter_max_offset of center.
                per_example_rng = random.Random(
                    int(hashlib.md5(f"{args.seed}:{row['example_id']}:jitter".encode()).hexdigest(), 16)
                )
                actual_offset = per_example_rng.randint(
                    -args.jitter_max_offset,
                    args.jitter_max_offset
                )
                desired_snp_index = center_idx + actual_offset

            else:
                actual_offset = int(offset)
                desired_snp_index = center_idx + actual_offset

            if desired_snp_index < 0 or desired_snp_index >= args.window_size:
                skipped += 1
                continue

            seq = extract_window(
                fasta=fasta,
                chrom=row["chr"],
                snp_pos=int(row["ref_start"]),
                window_size=args.window_size,
                desired_snp_index=desired_snp_index,
            )

            if seq is None:
                skipped += 1
                continue

            if not args.keep_N and "N" in seq:
                skipped += 1
                continue

            if args.allele_mode == "hap1":
                seq = maybe_inject_allele(seq, desired_snp_index, row["hap1_allele"])
            elif args.allele_mode == "hap2":
                seq = maybe_inject_allele(seq, desired_snp_index, row["hap2_allele"])

            if seq is None:
                skipped += 1
                continue

            kmer_seq = seq_to_kmers(seq, args.k)
            label = int(row["label"])
            split = row["split"]

            files[split].write(f"{kmer_seq}\t{label}\n")
            counts[split] += 1

            meta.write(
                f"{row['example_id']}\t{split}\t{row['chr']}\t{row['ref_start']}\t"
                f"{row['ref_end']}\t{label}\t{desired_snp_index}\t{actual_offset}\t"
                f"{args.allele_mode}\t{seq}\n"
            )

        for f in files.values():
            f.close()
        meta.close()

        print(f"\nWrote {offset_dir}")
        print("counts:", counts)
        print("skipped:", skipped)

    fasta.close()


if __name__ == "__main__":
    main()