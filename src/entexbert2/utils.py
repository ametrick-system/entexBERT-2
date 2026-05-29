import os
import math
import pandas as pd
import numpy as np
import pyBigWig

from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Any, Literal

######################
# Global Specs & Utils
######################

TaskType = Literal["classification", "regression"]
AuxTaskType = Literal["binary", "multiclass", "regression"]

@dataclass
class LabelSpec:
    """
    Specification for one label column

    - fn: takes in a row-like object and returns the raw label value
    - transform_fn: applied after fn
    """
    name: str
    fn: Callable[[pd.Series], float]
    task_type: str = "regression"
    transform_fn: Callable[[float], float] = lambda x: x

@dataclass
class SNVWindowSpec:
    """
    SNV-centered window specification

    - left_bp: number of nucleotides to the left of the SNV to include in the window
    - right_bp: number of nucleotides to the right of the SNV to include in the window
    """
    left_bp: int
    right_bp: int
    chrom_sizes_path: Optional[str] = None

@dataclass
class BalanceSpec:
    """
    Balancing/subsampling strategy:
        - "none": do not balance
        - "global_binary": balance by label_col across full dataset
        - "per_tissue_binary": balance by label_col separately within each tissue
    """
    strategy: str = "none"
    label_col: str = "imbalance_significance"
    random_state: int = 42

def log1p_transform(x: float) -> float:
    """
    Safe log1p transform for nonnegative signal.
    """
    x = float(x)

    if not np.isfinite(x):
        return 0.0

    if x < 0:
        x = 0.0

    return float(np.log1p(x))

def identity_transform(x: float) -> float:
    return float(x)

def load_chrom_sizes(chrom_sizes_path: Optional[str]) -> Dict[str, int]:
    """
    Load chromosome sizes from a two-column chrom.sizes file
    """
    chrom_sizes = {}

    if chrom_sizes_path is None:
        return chrom_sizes

    with open(chrom_sizes_path) as f:
        for line in f:
            if not line.strip():
                continue
            chrom, size = line.strip().split()[:2]
            chrom_sizes[chrom] = int(size)

    return chrom_sizes


def split_and_write_csvs(
    df: pd.DataFrame,
    output_dir: str,
    label_col: str,
    input_mode: str = "ref_single",
    aux_cols: Optional[List[str]] = None,
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    skip_ambiguous: bool = True,
    group_cols: Optional[List[str]] = None,
):
    """
    Write train/dev/test CSVs for entexBERT-2.

    If group_cols is provided, all rows with the same group key are assigned
    to the same split. This prevents duplicate sequence/window leakage across
    train/dev/test.
    """
    aux_cols = aux_cols or []
    group_cols = group_cols or []

    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}.")

    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair"}

    if input_mode in paired_modes:
        input_cols = ["sequence1", "sequence2"]
    else:
        input_cols = ["sequence"]

    required = input_cols + [label_col] + aux_cols + group_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for CSV writing: {missing}")

    out = df[required].copy()
    out = out.rename(columns={label_col: "label"})

    if skip_ambiguous:
        if input_mode in paired_modes:
            out = out[
                ~out["sequence1"].str.contains("N", regex=False)
                & ~out["sequence2"].str.contains("N", regex=False)
            ].copy()
        else:
            out = out[~out["sequence"].str.contains("N", regex=False)].copy()

    if out.empty:
        raise ValueError("No examples remain after filtering ambiguous sequences.")

    rng = np.random.default_rng(seed)

    if group_cols:
        out["_split_group"] = out[group_cols].astype(str).agg("|".join, axis=1)

        unique_groups = out["_split_group"].drop_duplicates().to_numpy()
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train_groups = int(split_ratio[0] * n_groups)
        n_dev_groups = int(split_ratio[1] * n_groups)

        train_groups = set(unique_groups[:n_train_groups])
        dev_groups = set(unique_groups[n_train_groups:n_train_groups + n_dev_groups])
        test_groups = set(unique_groups[n_train_groups + n_dev_groups:])

        splits = {
            "train.csv": out[out["_split_group"].isin(train_groups)].copy(),
            "dev.csv": out[out["_split_group"].isin(dev_groups)].copy(),
            "test.csv": out[out["_split_group"].isin(test_groups)].copy(),
        }

        for split_df in splits.values():
            split_df.drop(columns=["_split_group"], inplace=True)

        print(f"Group-aware split using group_cols={group_cols}")
        print(f"Unique groups: {n_groups}")
        print({
            "train_groups": len(train_groups),
            "dev_groups": len(dev_groups),
            "test_groups": len(test_groups),
        })

    else:
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

        total = len(out)
        n_train = int(split_ratio[0] * total)
        n_dev = int(split_ratio[1] * total)

        splits = {
            "train.csv": out.iloc[:n_train].copy(),
            "dev.csv": out.iloc[n_train:n_train + n_dev].copy(),
            "test.csv": out.iloc[n_train + n_dev:].copy(),
        }

    # Do not write group columns to final Trainer CSVs unless they are also input/aux columns.
    final_cols = input_cols + ["label"] + aux_cols

    os.makedirs(output_dir, exist_ok=True)

    for filename, split_df in splits.items():
        split_df = split_df[final_cols].copy()
        split_df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Saved {sum(len(v) for v in splits.values())} total examples to {output_dir}")
    print({name: len(split_df) for name, split_df in splits.items()})

    combined = pd.concat([split_df[final_cols] for split_df in splits.values()], ignore_index=True)

    print("\nMain label summary:")
    print(combined["label"].describe())

    if combined["label"].nunique() <= 20:
        print("\nMain label counts:")
        print(combined["label"].value_counts().sort_index())

    for aux_col in aux_cols:
        print(f"\nAux label summary: {aux_col}")
        print(combined[aux_col].describe())

######################
# AS Experiment Utils
######################

def get_base_count(row: pd.Series, allele: str) -> float:
    """
    Return count for allele A/C/G/T from EN-TEx AS columns cA/cC/cG/cT
    """
    allele = str(allele).upper()
    col = f"c{allele}"

    if col not in row:
        raise ValueError(f"Expected column {col!r} for allele {allele!r}.")

    return float(row[col])

def hap1_count(row: pd.Series) -> float:
    return get_base_count(row, row["hap1_allele"])

def hap2_count(row: pd.Series) -> float:
    return get_base_count(row, row["hap2_allele"])

def total_allele_reads(row: pd.Series) -> float:
    return hap1_count(row) + hap2_count(row)

def hap1_ratio(row: pd.Series) -> float:
    total = total_allele_reads(row)
    if total <= 0:
        return 0.0
    return hap1_count(row) / total

def signed_log_count_ratio(row: pd.Series, pseudocount: float = 1.0) -> float:
    """
    Positive means hap1 has more reads than hap2
    Negative means hap2 has more reads than hap1
    """
    return math.log(hap1_count(row) + pseudocount) - math.log(hap2_count(row) + pseudocount)

def abs_log_count_ratio(row: pd.Series, pseudocount: float = 1.0) -> float:
    return abs(signed_log_count_ratio(row, pseudocount=pseudocount))

def as_magnitude_from_ratio(row: pd.Series) -> float:
    """
    0 means perfectly balanced hap1/hap2
    1 means all reads are on one haplotype
    """
    return abs(hap1_ratio(row) - 0.5) * 2.0

def imbalance_significance_label(row: pd.Series) -> int:
    return int(row["imbalance_significance"])

def ref_allele_ratio_label(row: pd.Series) -> float:
    return float(row["ref_allele_ratio"])

def load_as_table(
    input_tsv: str,
    assay: str,
    donor: str,
    tissue: Optional[str] = None,
    min_total_reads: Optional[int] = None,
    chunksize: int = 100000,
) -> pd.DataFrame:
    """
    Load and filter EN-TEx hetSNVs_default_AS.tsv for one assay/donor, optionally one tissue

    Adds:
        total_reads
        hap1_count
        hap2_count
        hap1_ratio
        as_magnitude
        signed_log_count_ratio
        abs_log_count_ratio
    """
    usecols = [
        "chr", "ref_start", "ref_end",
        "ref_allele", "hap1_allele", "hap2_allele",
        "experiment_accession", "donor", "tissue", "assay",
        "cA", "cC", "cG", "cT",
        "ref_allele_ratio", "p_betabinom", "imbalance_significance",
    ]

    chunks = []

    for chunk in pd.read_csv(
        input_tsv,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
    ):
        mask = (chunk["assay"] == assay) & (chunk["donor"] == donor)

        if tissue is not None:
            mask &= chunk["tissue"].eq(tissue)

        sub = chunk[mask].copy()

        if sub.empty:
            continue

        sub["hap1_count"] = sub.apply(hap1_count, axis=1)
        sub["hap2_count"] = sub.apply(hap2_count, axis=1)
        sub["total_reads"] = sub["hap1_count"] + sub["hap2_count"]

        if min_total_reads is not None:
            sub = sub[sub["total_reads"] >= min_total_reads].copy()

        if sub.empty:
            continue

        sub["hap1_ratio"] = sub.apply(hap1_ratio, axis=1)
        sub["as_magnitude"] = sub.apply(as_magnitude_from_ratio, axis=1)
        sub["signed_log_count_ratio"] = sub.apply(signed_log_count_ratio, axis=1)
        sub["abs_log_count_ratio"] = sub.apply(abs_log_count_ratio, axis=1)

        chunks.append(sub)

    if not chunks:
        raise ValueError(
            f"No AS rows found for assay={assay}, donor={donor}, tissue={tissue}."
        )

    return pd.concat(chunks, ignore_index=True)

def balance_as_table(
    df: pd.DataFrame,
    balance_spec: BalanceSpec,
) -> pd.DataFrame:
    """
    Balance AS examples according to a strategy
    """
    strategy = balance_spec.strategy
    label_col = balance_spec.label_col
    random_state = balance_spec.random_state

    if strategy == "none":
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if label_col not in df.columns:
        raise ValueError(f"label_col {label_col!r} not found in dataframe.")

    if strategy == "global_binary":
        pos = df[df[label_col] == 1]
        neg = df[df[label_col] == 0]

        if pos.empty or neg.empty:
            raise ValueError("Need both positive and negative examples for global_binary balancing.")

        n = min(len(pos), len(neg))
        pos_sampled = pos.sample(n=n, random_state=random_state)
        neg_sampled = neg.sample(n=n, random_state=random_state)

        out = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
        return out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if strategy == "per_tissue_binary":
        balanced_groups = []

        for tissue, group in df.groupby("tissue"):
            pos = group[group[label_col] == 1]
            neg = group[group[label_col] == 0]

            if pos.empty or neg.empty:
                print(f"Skipping {tissue}: missing one class.")
                continue

            n = min(len(pos), len(neg))
            pos_sampled = pos.sample(n=n, random_state=random_state)
            neg_sampled = neg.sample(n=n, random_state=random_state)

            balanced = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
            balanced_groups.append(balanced)

            print(f"{tissue}: {n} positives, {n} negatives")

        if not balanced_groups:
            raise ValueError("No tissue produced a valid balanced dataset.")

        out = pd.concat(balanced_groups, ignore_index=True)
        return out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    raise ValueError(f"Unsupported balancing strategy: {strategy}")

def add_snv_windows(
    df: pd.DataFrame,
    window_spec: SNVWindowSpec,
) -> pd.DataFrame:
    """
    Add bed_start, bed_end, and SNV location columns for SNV-containing sequence windows
    """
    df = df.copy()

    chrom_sizes = load_chrom_sizes(window_spec.chrom_sizes_path)

    df["SNV"] = df["ref_start"].astype(int)
    df["bed_start"] = (df["SNV"] - window_spec.left_bp).clip(lower=0)
    df["bed_end"] = df["SNV"] + window_spec.right_bp + 1

    if chrom_sizes:
        df["bed_end"] = [
            min(end, chrom_sizes.get(chrom, end))
            for chrom, end in zip(df["chr"], df["bed_end"])
        ]

    # Ensure fixed length windows
    expected_len = window_spec.left_bp + 1 + window_spec.right_bp
    valid = (df["bed_end"] - df["bed_start"]) == expected_len
    df = df[valid].copy()

    return df.reset_index(drop=True)

def add_label_columns(
    df: pd.DataFrame,
    primary_label: LabelSpec,
    aux_labels: Optional[List[LabelSpec]] = None,
) -> pd.DataFrame:
    """
    Add primary and auxiliary label columns to the AS dataframe
    """
    df = df.copy()
    aux_labels = aux_labels or []

    all_specs = [primary_label] + aux_labels

    for spec in all_specs:
        values = []
        for _, row in df.iterrows():
            raw = spec.fn(row)
            values.append(spec.transform_fn(raw))
        df[spec.name] = values

    return df

def make_haplotype_sequence(
    ref_sequence: str,
    snv_offset: int,
    allele: str,
) -> str:
    """
    Replace the SNV position in a reference sequence with the requested allele
    (assumes allele is a single base)
    """
    ref_sequence = ref_sequence.upper()
    allele = str(allele).upper()

    if len(allele) != 1:
        raise ValueError(f"Only SNVs are supported, got allele={allele!r}.")

    if snv_offset < 0 or snv_offset >= len(ref_sequence):
        raise ValueError(
            f"snv_offset={snv_offset} out of bounds for sequence length {len(ref_sequence)}."
        )

    return ref_sequence[:snv_offset] + allele + ref_sequence[snv_offset + 1:]

def add_sequence_inputs(
    df: pd.DataFrame,
    ref_fasta,
    input_mode: str = "ref_single",
) -> pd.DataFrame:
    """
    Add sequence columns according to input_mode

    Requires ref_fasta to support:
        str(ref_fasta[chrom][start:end])
    as pyfaidx.Fasta does
    """

    single_modes = {"ref_single", "hap1_single", "hap2_single"}
    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair"}

    if input_mode not in single_modes and input_mode not in paired_modes:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    df = df.copy()

    sequences = []
    sequence1s = []
    sequence2s = []

    for _, row in df.iterrows():
        chrom = row["chr"]
        start = int(row["bed_start"])
        end = int(row["bed_end"])
        snv = int(row["SNV"])

        ref_seq = str(ref_fasta[chrom][start:end]).upper()

        # Sanity check that SNV is in the provided window (should be by construction)
        if snv < start or snv >= end:
            raise ValueError(
                f"SNV {chrom}:{snv} is outside extracted window {chrom}:{start}-{end}."
            )

        snv_offset = snv - start

        if len(ref_seq) != end - start:
            raise ValueError(
                f"Extracted sequence length mismatch at {chrom}:{start}-{end}. "
                f"Expected {end - start}, got {len(ref_seq)}."
            )

        # Sanity check: confirm that the reference FASTA base matches the AS table.
        ref_allele = str(row["ref_allele"]).upper()

        if len(ref_allele) == 1:
            observed_ref_base = ref_seq[snv_offset].upper()

            if observed_ref_base != ref_allele:
                raise ValueError(
                    f"Reference allele mismatch at {chrom}:{snv}. "
                    f"FASTA has {observed_ref_base!r}, but AS table has ref_allele={ref_allele!r}. "
                    f"This likely indicates a coordinate convention issue."
                )

        hap1_seq = make_haplotype_sequence(ref_seq, snv_offset, row["hap1_allele"])
        hap2_seq = make_haplotype_sequence(ref_seq, snv_offset, row["hap2_allele"])

        if input_mode == "ref_single":
            sequences.append(ref_seq)

        elif input_mode == "hap1_single":
            sequences.append(hap1_seq)

        elif input_mode == "hap2_single":
            sequences.append(hap2_seq)

        elif input_mode == "hap_pair":
            sequence1s.append(hap1_seq)
            sequence2s.append(hap2_seq)

        elif input_mode == "ref_hap1_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap1_seq)

        elif input_mode == "ref_hap2_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap2_seq)

        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

    if input_mode in single_modes:
        df["sequence"] = sequences
    elif input_mode in paired_modes:
        df["sequence1"] = sequence1s
        df["sequence2"] = sequence2s

    return df

###############################
# Sequence Window Overlap Utils
###############################

def summarize_overlaps(overlaps, qstart, qend, mode: str):
    """
    Summarize a list of overlaps according to mode
    overlaps should be [(ov_start, ov_end, score), ...]
    """
    if mode == "binary":
        return 1.0 if len(overlaps) > 0 else 0.0

    if mode == "count":
        return float(len(overlaps))

    if mode == "max_score":
        scores = [score for _, _, score in overlaps if score is not None]
        return max(scores) if scores else 0.0

    if mode == "sum_score":
        scores = [score for _, _, score in overlaps if score is not None]
        return float(sum(scores)) if scores else 0.0

    if mode == "frac_covered":
        if len(overlaps) == 0:
            return 0.0

        merged = []
        for ov_start, ov_end, _ in sorted(overlaps, key=lambda x: x[0]):
            if not merged or ov_start > merged[-1][1]:
                merged.append([ov_start, ov_end])
            else:
                merged[-1][1] = max(merged[-1][1], ov_end)

        covered = sum(seg_end - seg_start for seg_start, seg_end in merged)
        return covered / max(1, qend - qstart)

    raise ValueError(f"Unsupported overlap mode: {mode}")

def summarize_duplicate_as_windows(
    df: pd.DataFrame,
    label_col: str = "imbalance_significance",
    group_cols: Optional[List[str]] = None,
):
    """
    Print a summary of duplicate sequence/window groups and label conflicts.

    For all-tissue AS classification, duplicates often arise because the same
    SNV/window appears in multiple tissues.
    """
    if group_cols is None:
        group_cols = [
            "chr",
            "bed_start",
            "bed_end",
            "ref_allele",
            "hap1_allele",
            "hap2_allele",
        ]

    missing = [c for c in group_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for duplicate summary: {missing}")

    tmp = df.copy()
    tmp["_group_key"] = tmp[group_cols].astype(str).agg("|".join, axis=1)

    grouped = tmp.groupby("_group_key")[label_col].agg(
        n_rows="count",
        n_unique_labels="nunique",
        mean_label="mean",
        min_label="min",
        max_label="max",
    )

    duplicate_groups = grouped[grouped["n_rows"] > 1]
    conflicting_groups = grouped[grouped["n_unique_labels"] > 1]

    print("\nDuplicate AS/window summary")
    print(f"Rows: {len(tmp)}")
    print(f"Unique groups: {len(grouped)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Rows in duplicate groups: {int(duplicate_groups['n_rows'].sum()) if len(duplicate_groups) else 0}")
    print(f"Conflicting-label groups: {len(conflicting_groups)}")
    print(f"Rows in conflicting-label groups: {int(conflicting_groups['n_rows'].sum()) if len(conflicting_groups) else 0}")

    if len(conflicting_groups) > 0:
        print("\nExample conflicting groups:")
        print(conflicting_groups.head(10))

    tmp.drop(columns=["_group_key"], inplace=True)

    return grouped

def build_as_prediction_dataset(
    input_tsv: str,
    output_dir: str,
    ref_fasta,
    assay: str,
    donor: str,
    primary_label: LabelSpec,
    window_spec: SNVWindowSpec,
    input_mode: str = "hap_pair",
    tissue: Optional[str] = None,
    min_total_reads: Optional[int] = None,
    balance_spec: Optional[BalanceSpec] = None,
    aux_labels: Optional[List[LabelSpec]] = None,
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    chunksize: int = 100000,
    skip_ambiguous: bool = True,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    End-to-end AS dataset builder for entexBERT-2.

    Produces train/dev/test CSVs and returns the final dataframe.
    """
    balance_spec = balance_spec or BalanceSpec(strategy="none")
    aux_labels = aux_labels or []

    df = load_as_table(
        input_tsv=input_tsv,
        assay=assay,
        donor=donor,
        tissue=tissue,
        min_total_reads=min_total_reads,
        chunksize=chunksize,
    )

    df = balance_as_table(df, balance_spec)
    df = add_snv_windows(df, window_spec)
    df = add_label_columns(df, primary_label=primary_label, aux_labels=aux_labels)

    if group_cols is not None:
        summarize_duplicate_as_windows(
            df,
            label_col=primary_label.name,
            group_cols=group_cols,
        )

    df = add_sequence_inputs(df, ref_fasta=ref_fasta, input_mode=input_mode)

    split_and_write_csvs(
        df=df,
        output_dir=output_dir,
        label_col=primary_label.name,
        input_mode=input_mode,
        aux_cols=[spec.name for spec in aux_labels],
        split_ratio=split_ratio,
        seed=seed,
        skip_ambiguous=skip_ambiguous,
        group_cols=group_cols,
    )

    return df

#####################
# BigWig Signal Utils
#####################

def summarize_bigwig_values(
    values,
    mode: str = "mean",
    missing_value: float = 0.0,
) -> float:
    """
    Summarize per-base BigWig values

    Args:
        values:
            List/array of values from pyBigWig.values(...)
            Missing bases are usually NaN
        mode:
            One of:
                - "mean"
                - "max"
                - "min"
                - "sum"
                - "std"
                - "coverage"
                - "mean_nonzero"
                - "max_abs"
        missing_value:
            Value used when no finite values are available

    Returns:
        float summary value
    """
    arr = np.asarray(values, dtype=float)

    finite = np.isfinite(arr)
    finite_vals = arr[finite]

    if mode == "coverage":
        if len(arr) == 0:
            return 0.0
        return float(finite.sum() / len(arr))

    if finite_vals.size == 0:
        return float(missing_value)

    if mode == "mean":
        return float(np.mean(finite_vals))

    if mode == "max":
        return float(np.max(finite_vals))

    if mode == "min":
        return float(np.min(finite_vals))

    if mode == "sum":
        return float(np.sum(finite_vals))

    if mode == "std":
        return float(np.std(finite_vals))

    if mode == "mean_nonzero":
        nonzero = finite_vals[finite_vals != 0]
        if nonzero.size == 0:
            return float(missing_value)
        return float(np.mean(nonzero))

    if mode == "max_abs":
        return float(np.max(np.abs(finite_vals)))

    raise ValueError(f"Unsupported BigWig signal summary mode: {mode}")

class BigWigSignalAnnotator:
    """
    Annotator for extracting continuous signal from a BigWig file

    Use for targets like:
        - fold-change-over-control signal at the SNV
        - mean signal over the sequence window
        - max signal over the sequence window
        - local signal around the SNV

    Coordinates are expected to be 0-based half-open, matching BED-style coordinates and pyBigWig queries
    """

    def __init__(
        self,
        bigwig_path: str,
        mode: str = "mean",
        region: str = "window",
        radius_bp: int = 0,
        missing_value: float = 0.0,
        use_values: bool = True,
        exact: bool = True,
    ):
        """
        Args:
            bigwig_path:
                Path to BigWig file.
            mode:
                Signal summary mode. Supported:
                    "mean", "max", "min", "sum", "std", "coverage",
                    "mean_nonzero", "max_abs"
            region:
                Region to query:
                    - "window": use row["bed_start"] to row["bed_end"]
                    - "snv": use one base [SNV, SNV + 1)
                    - "snv_radius": use [SNV - radius_bp, SNV + radius_bp + 1)
            radius_bp:
                Used only when region="snv_radius".
            missing_value:
                Returned when no finite BigWig signal exists in the queried region.
            use_values:
                If True, use per-base bw.values(...) and summarize manually.
                This is safest and gives predictable nan handling.
            exact:
                If use_values=False, pass exact=exact to bw.stats(...).
        """
        if pyBigWig is None:
            raise ImportError(
                "pyBigWig is required for BigWigSignalAnnotator. "
                "Install with `pip install pyBigWig` or "
                "`conda install pybigwig -c conda-forge -c bioconda`."
            )

        self.bigwig_path = bigwig_path
        self.mode = mode
        self.region = region
        self.radius_bp = radius_bp
        self.missing_value = missing_value
        self.use_values = use_values
        self.exact = exact

        self.bw = pyBigWig.open(bigwig_path)

        if self.bw is None:
            raise ValueError(f"Could not open BigWig file: {bigwig_path}")

        if not self.bw.isBigWig():
            self.bw.close()
            raise ValueError(f"File is not a BigWig file: {bigwig_path}")

        self.chrom_sizes = dict(self.bw.chroms())

    def close(self):
        if getattr(self, "bw", None) is not None:
            self.bw.close()
            self.bw = None

    def _query_region_from_row(self, row: pd.Series):
        chrom = row["chr"]

        if chrom not in self.chrom_sizes:
            return chrom, None, None

        chrom_size = self.chrom_sizes[chrom]

        if self.region == "window":
            start = int(row["bed_start"])
            end = int(row["bed_end"])

        elif self.region == "snv":
            snv = int(row["SNV"])
            start = snv
            end = snv + 1

        elif self.region == "snv_radius":
            snv = int(row["SNV"])
            start = snv - self.radius_bp
            end = snv + self.radius_bp + 1

        else:
            raise ValueError(
                f"Unsupported BigWig query region {self.region!r}. "
                "Choose from {'window', 'snv', 'snv_radius'}."
            )

        start = max(0, start)
        end = min(chrom_size, end)

        if end <= start:
            return chrom, None, None

        return chrom, start, end

    def __call__(self, row: pd.Series) -> float:
        chrom, start, end = self._query_region_from_row(row)

        if start is None or end is None:
            return float(self.missing_value)

        if self.use_values:
            values = self.bw.values(chrom, start, end)
            return summarize_bigwig_values(
                values,
                mode=self.mode,
                missing_value=self.missing_value,
            )

        # Faster option for common summary stats
        # exact=True avoids zoom-level approximations
        if self.mode in {"mean", "max", "min", "std", "coverage"}:
            val = self.bw.stats(
                chrom,
                start,
                end,
                type=self.mode,
                exact=self.exact,
            )[0]

            if val is None or not np.isfinite(val):
                return float(self.missing_value)

            return float(val)

        # For sum/mean_nonzero/max_abs, fall back to per-base values
        values = self.bw.values(chrom, start, end)
        return summarize_bigwig_values(
            values,
            mode=self.mode,
            missing_value=self.missing_value,
        )

class TissueAwareBigWigSignalAnnotator:
    """
    Tissue-aware BigWig signal annotator

    Selects the BigWig file based on row["tissue"]
    """

    def __init__(
        self,
        bigwig_paths_by_tissue: Dict[str, str],
        mode: str = "mean",
        region: str = "window",
        radius_bp: int = 0,
        missing_value: float = 0.0,
        use_values: bool = True,
        exact: bool = True,
    ):
        self.bigwig_paths_by_tissue = bigwig_paths_by_tissue
        self.mode = mode
        self.region = region
        self.radius_bp = radius_bp
        self.missing_value = missing_value
        self.use_values = use_values
        self.exact = exact

        self.annotators = {}

        missing = []
        for tissue, path in bigwig_paths_by_tissue.items():
            if path is None or not os.path.exists(path):
                missing.append((tissue, path))
                continue

            self.annotators[tissue] = BigWigSignalAnnotator(
                bigwig_path=path,
                mode=mode,
                region=region,
                radius_bp=radius_bp,
                missing_value=missing_value,
                use_values=use_values,
                exact=exact,
            )

        if not self.annotators:
            raise ValueError("No valid BigWig files were provided for any tissue.")

        if missing:
            print("Warning: missing BigWigs for some tissues:")
            for tissue, path in missing:
                print(f"  {tissue}: {path}")

    def close(self):
        for annotator in self.annotators.values():
            annotator.close()

    def __call__(self, row: pd.Series) -> float:
        tissue = row["tissue"]

        if tissue not in self.annotators:
            return float(self.missing_value)

        return self.annotators[tissue](row)

def make_bigwig_label_spec(
    name: str,
    bigwig_path: str,
    mode: str = "mean",
    region: str = "window",
    radius_bp: int = 0,
    transform_fn: Callable[[float], float] = lambda x: x,
    missing_value: float = 0.0,
    use_values: bool = True,
    exact: bool = True,
) -> LabelSpec:
    """
    Build a regression LabelSpec from one BigWig file
    """
    annotator = BigWigSignalAnnotator(
        bigwig_path=bigwig_path,
        mode=mode,
        region=region,
        radius_bp=radius_bp,
        missing_value=missing_value,
        use_values=use_values,
        exact=exact,
    )

    return LabelSpec(
        name=name,
        fn=annotator,
        task_type="regression",
        transform_fn=transform_fn,
    )

def make_tissue_aware_bigwig_label_spec(
    name: str,
    bigwig_paths_by_tissue: Dict[str, str],
    mode: str = "mean",
    region: str = "window",
    radius_bp: int = 0,
    transform_fn: Callable[[float], float] = lambda x: x,
    missing_value: float = 0.0,
    use_values: bool = True,
    exact: bool = True,
) -> LabelSpec:
    """
    Build a regression LabelSpec from tissue-specific BigWig files
    """
    annotator = TissueAwareBigWigSignalAnnotator(
        bigwig_paths_by_tissue=bigwig_paths_by_tissue,
        mode=mode,
        region=region,
        radius_bp=radius_bp,
        missing_value=missing_value,
        use_values=use_values,
        exact=exact,
    )

    return LabelSpec(
        name=name,
        fn=annotator,
        task_type="regression",
        transform_fn=transform_fn,
    )