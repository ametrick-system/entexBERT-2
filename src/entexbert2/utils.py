import os
import math
import hashlib
import pandas as pd
import numpy as np
import pyBigWig

from dataclasses import dataclass, field
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
    - required_columns: columns the fn needs to be present on each row (validated at compose time against the row source's columns)
    """
    name: str
    fn: Callable[[pd.Series], float]
    task_type: str = "regression"
    transform_fn: Callable[[float], float] = lambda x: x
    required_columns: List[str] = field(default_factory=list)

@dataclass
class SNVWindowSpec:
    """
    SNV-centered window specification

    - left_bp: number of nucleotides to the left of the SNV to include in the window
    - right_bp: number of nucleotides to the right of the SNV to include in the window
    - snv_offset_mode: "fixed" (SNV always at offset left_bp) or "uniform" (SNV offset
      jittered uniformly within +/- jitter_max_bp of left_bp, per example)
    - jitter_max_bp: max +/- jitter of the SNV offset within the window (uniform mode only).
      Must be <= min(left_bp, right_bp) so the SNV always stays inside the fixed-length window.

    Window length is always fixed at left_bp + 1 + right_bp; jitter only slides the window
    relative to the SNV, it does not change the length. The realized per-example offset is
    recorded downstream as anchor_offset_seq1[/seq2].
    """
    left_bp: int
    right_bp: int
    chrom_sizes_path: Optional[str] = None
    snv_offset_mode: str = "fixed"
    jitter_max_bp: int = 0

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

@dataclass
class PartitionSpec:
    """
    Hybrid cross-individual split:
        - hold out whole chromosome(s) for TEST, bin the rest for TRAIN/DEV
        - bin_size / salt / ratios travel in the resolved config
        - K-fold-ready: fold_assignment maps every chromosome to a fold index and fold_id selects which fold is the current test set
    """
    enabled: bool = False
    bin_size: int = 100_000
    salt: str = "entexbert2_v1"
    fold_assignment: Dict[str, int] = field(default_factory=dict)   # chromosome -> fold index
    fold_id: int = 0                                                # which fold is TEST now
    train_frac_within_nontest: float = 8.0 / 9.0                    # 8:1 train:dev within non-test
    # --- pure genomic-bin 3-way split (used only when NO chromosome is held out) ---
    # When fold_assignment yields no test chromosomes AND bin_test_frac > 0, the bin hash assigns
    # test/dev/train directly (all chromosomes contribute to every split). Leaves the chromosome-
    # holdout path untouched (bin_test_frac == 0 -> old 2-way train/dev behavior).
    bin_test_frac: float = 0.0     # fraction of bins -> test  (e.g. 0.10)
    bin_dev_frac: float = 0.0      # fraction of bins -> dev   (e.g. 0.10); train = 1 - test - dev
    # --- boundary-leakage control: drop loci whose [pos-boundary_bp, pos+boundary_bp] window
    #     crosses a bin edge (near-identical windows can otherwise land in different splits) ---
    exclude_boundary: bool = False
    boundary_bp: int = 0           # half-window; set to max(left_bp, right_bp)


def genomic_bin_id(chrom: str, pos: int, bin_size: int) -> str:
    """
    Deterministic genomic-bin id: chrom + which fixed-width tile the position falls in
        - uses the genomic position (donor-invariant) so the same locus lands in the same bin for every individual
    """
    return f"{chrom}|{int(pos) // int(bin_size)}"

def assign_split_column(df: pd.DataFrame, spec: "PartitionSpec") -> np.ndarray:
    """
    Assign a 'train'/'dev'/'test' label to every row for the hybrid partition, in two stages:
      1. a locus on a held-out test chromosome (fold_assignment[chrom] == fold_id) -> 'test'
      2. otherwise, hash its genomic bin -- int(sha1(f"{salt}|{genomic_bin_id(...)}"), 16) % 1e6 / 1e6
         -- and send it to 'train' if that fraction < train_frac_within_nontest, else 'dev'

    Deterministic and donor-invariant: the split depends only on (chrom, genomic position), so the
    same locus lands in the same split across every (donor, assay) cell -> cross-individual
    train/dev vs test disjointness by construction

    Vectorized: sha1 is evaluated once per DISTINCT genomic bin (a few 1e4 genome-wide) rather than
    once per row, via np.unique over int64-packed (chrom, bin) keys. The hashed string is identical
    to the per-row form. Returns an object ndarray of length len(df)
    """
    pos_col = "SNV" if "SNV" in df.columns else "ref_start"
    chroms = df["chr"].astype(str).to_numpy()
    positions = df[pos_col].astype("int64").to_numpy()

    n = len(df)
    out = np.empty(n, dtype=object)

    # Stage 1 (vectorized): loci on a held-out test chromosome -> 'test'
    test_chroms = [c for c, f in spec.fold_assignment.items() if f == spec.fold_id]
    is_test = np.isin(chroms, test_chroms) if test_chroms else np.zeros(n, dtype=bool)
    out[is_test] = "test"

    # Stage 2 (vectorized): hash each UNIQUE genomic bin once, then map the fraction back to rows
    nontest = ~is_test
    if nontest.any():
        bin_idx = positions[nontest] // int(spec.bin_size)                 # int64 tile index
        chrom_codes, chrom_uniques = pd.factorize(chroms[nontest], sort=False)
        offset = int(bin_idx.max()) + 1                                    # pack (chrom, bin) -> 1 int64
        packed = chrom_codes.astype("int64") * offset + bin_idx
        uniq, inverse = np.unique(packed, return_inverse=True)             # unique bins + row->bin map
        uniq_frac = np.empty(uniq.shape[0], dtype="float64")
        for j, val in enumerate(uniq.tolist()):
            chrom = chrom_uniques[val // offset]
            b = val % offset
            # b == pos // bin_size, so genomic_bin_id(chrom, b*bin_size, bin_size) == f"{chrom}|{b}":
            # one definition of the bin convention, evaluated per distinct bin (not per row).
            bid = genomic_bin_id(chrom, int(b) * int(spec.bin_size), spec.bin_size)
            h = int(hashlib.sha1(f"{spec.salt}|{bid}".encode()).hexdigest(), 16)
            uniq_frac[j] = (h % 1_000_000) / 1_000_000.0
        fracs = uniq_frac[inverse]
        pure_bin = (len(test_chroms) == 0) and (spec.bin_test_frac > 0)
        if pure_bin:
            tf, dv = spec.bin_test_frac, spec.bin_dev_frac
            out[nontest] = np.where(
                fracs < tf, "test",
                np.where(fracs < tf + dv, "dev", "train"),
            )
        else:
            out[nontest] = np.where(fracs < spec.train_frac_within_nontest, "train", "dev")

    # Boundary exclusion: mark loci whose window straddles a bin edge for dropping. Applied only to
    # hashed (nontest) rows -- whole-chromosome test sets are boundary-free by construction. The
    # sentinel "__exclude__" is dropped in build_dataset before split_and_write_csvs (which validates
    # the split column against exactly {train,dev,test}).
    if spec.exclude_boundary and spec.boundary_bp > 0:
        bp = int(spec.boundary_bp); bs = int(spec.bin_size)
        home = positions // bs
        same_bin = ((positions - bp) // bs == home) & ((positions + bp) // bs == home)
        out[nontest & ~same_bin] = "__exclude__"

    return out

def log1p_transform(x: float) -> float:
    """
    log1p transform for nonnegative signal
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
    meta_cols: Optional[List[str]] = None,
    split_mode: str = "train_dev_test",
    exclude_loci: Optional[set] = None,
    dedup_sequences_across_splits: bool = True,
    balance_spec: Optional["BalanceSpec"] = None,
    balance_split: str = "all",
    split_col: Optional[str] = None,
    depth_col: Optional[str] = None,
):
    """
    Write entexBERT-2 dataset CSVs.

    For each split, writes TWO files:
      - <split>.csv       : minimal Trainer input (sequence(s), label, aux only)
      - <split>.meta.csv  : rich superset (minimal columns + meta_cols + a 'split' column),
                            consumed by analysis/eval/plotting. Row-aligned to <split>.csv.

    Splitting:
      - split_mode == "train_dev_test": group-aware split if group_cols is provided
        (rows sharing a group key go to one split, preventing window leakage), else a
        plain row-level shuffle split.
      - split_mode == "test_only": no split; all rows written to test.csv / test.meta.csv.

    exclude_loci: optional set of locus_id values to drop before writing (used to build a
    cross-individual test set that excludes a reference model's train+dev loci).
    """
    aux_cols = aux_cols or []
    group_cols = group_cols or []
    meta_cols = meta_cols or []

    if split_mode not in {"train_dev_test", "test_only"}:
        raise ValueError(f"Unsupported split_mode: {split_mode!r}")

    if split_mode == "train_dev_test" and not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}.")

    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}
    input_cols = ["sequence1", "sequence2"] if input_mode in paired_modes else ["sequence"]

    # Columns we need to carry through, de-duplicated and order-preserved
    requested = input_cols + [label_col] + aux_cols + group_cols + meta_cols
    if split_col is not None:
        requested = requested + [split_col]
    if depth_col is not None:
        requested = requested + [depth_col]

    required = list(dict.fromkeys(requested))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for CSV writing: {missing}")

    out = df[required].copy()
    out = out.rename(columns={label_col: "label"})
    if depth_col is not None and depth_col != "depth":
        out = out.rename(columns={depth_col: "depth"}) # rename to a stable name

    # meta_cols may reference label_col under its original name; after rename use "label".
    meta_cols = ["label" if c == label_col else c for c in meta_cols]

    if skip_ambiguous:
        if input_mode in paired_modes:
            out = out[
                ~out["sequence1"].str.contains("N", regex=False)
                & ~out["sequence2"].str.contains("N", regex=False)
            ].copy()
        else:
            out = out[~out["sequence"].str.contains("N", regex=False)].copy()

    if exclude_loci:
        if "locus_id" not in out.columns:
            raise ValueError("exclude_loci requires a 'locus_id' column (add it via meta_cols).")
        before = len(out)
        out = out[~out["locus_id"].isin(set(exclude_loci))].copy()
        print(f"exclude_loci: removed {before - len(out)} of {before} rows "
              f"({len(set(exclude_loci))} excluded loci).")

    if out.empty:
        raise ValueError("No examples remain after filtering.")

    rng = np.random.default_rng(seed)

    if split_col is not None:
        _vals = set(out[split_col].astype(str).unique())
        _bad = _vals - {"train", "dev", "test"}
        if _bad:
            raise ValueError(
                f"split_col {split_col!r} has unexpected values {sorted(_bad)}; "
                f"expected only 'train'/'dev'/'test'."
            )
        splits = { 
            "train.csv": out[out[split_col] == "train"].drop(columns=[split_col]).copy(),
            "dev.csv":   out[out[split_col] == "dev"].drop(columns=[split_col]).copy(),
            "test.csv":  out[out[split_col] == "test"].drop(columns=[split_col]).copy(),
        }
        _sizes = {k: len(v) for k, v in splits.items()}
        print(f"Precomputed split via column {split_col!r}: {_sizes}")

    elif split_mode == "test_only":
        splits = {"test.csv": out.copy()}
        print(f"test_only mode: {len(out)} examples -> test.csv")

    elif group_cols:
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

    # Cross-split sequence dedup: guarantee no identical input sequence appears in more than
    # one split (priority train > dev > test). Locus-grouping prevents the SAME variant from
    # crossing splits, but with window jitter on ref_single, DISTINCT nearby SNVs can produce
    # identical reference windows that would otherwise contaminate eval. Within-split
    # duplicates are left untouched (they don't inflate held-out metrics).
    if split_mode == "train_dev_test" and dedup_sequences_across_splits and len(splits) > 1:
        def _seq_key(frame):
            s = frame[input_cols[0]].astype(str)
            for c in input_cols[1:]:
                s = s + "|" + frame[c].astype(str)
            return s

        seen = set()
        total_dropped = 0
        for fn in ("train.csv", "dev.csv", "test.csv"):
            sdf = splits[fn]
            key = _seq_key(sdf)
            keep = ~key.isin(seen)
            dropped = int((~keep).sum())
            if dropped:
                total_dropped += dropped
                print(f"cross-split dedup: dropped {dropped} rows from {fn[:-4]} "
                      f"(sequence already present in an earlier split).")
            splits[fn] = sdf[keep].copy()
            seen.update(key[keep].tolist())
        if total_dropped:
            print(f"cross-split dedup: removed {total_dropped} cross-split duplicate-sequence rows total.")

    # Optionally balance ONLY the train split; dev/test stay at natural prevalence.
    # (Runs after dedup so we balance the final training rows. The label column is "label"
    # here, so we point the spec at it.)
    if (split_mode == "train_dev_test" and balance_split == "train"
            and balance_spec is not None and balance_spec.strategy != "none"
            and "train.csv" in splits and not splits["train.csv"].empty):
        from dataclasses import replace as _dc_replace
        train_df = splits["train.csv"]
        before, before_pos = len(train_df), int((train_df["label"] == 1).sum())
        splits["train.csv"] = balance_as_table(train_df, _dc_replace(balance_spec, label_col="label"))
        after = len(splits["train.csv"]); after_pos = int((splits["train.csv"]["label"] == 1).sum())
        print(f"train-only balance ({balance_spec.strategy}): train {before} -> {after} rows "
              f"(pos {before_pos} -> {after_pos}); dev/test left at natural prevalence.")

    # Minimal Trainer CSV columns vs. rich metadata CSV columns
    final_cols = input_cols + ["label"] + aux_cols
    if depth_col is not None:
        final_cols = final_cols + ["depth"]

    meta_out_cols = list(dict.fromkeys(final_cols + meta_cols + ["split"]))

    os.makedirs(output_dir, exist_ok=True)

    for filename, split_df in splits.items():
        split_name = filename[:-len(".csv")]
        split_df = split_df.copy()
        split_df["split"] = split_name

        # Minimal training file
        split_df[final_cols].to_csv(os.path.join(output_dir, filename), index=False)

        # Rich, row-aligned metadata sidecar
        meta_present = [c for c in meta_out_cols if c in split_df.columns]
        meta_name = filename.replace(".csv", ".meta.csv")
        split_df[meta_present].to_csv(os.path.join(output_dir, meta_name), index=False)

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

def log_total_count(row: pd.Series, pseudocount: float = 1.0) -> float:
    """
    Total binding depth at the SNV: log(ref_count + alt_count + 1) = log(total_allele_reads + 1).
    NOT an allelic contrast -- it's a per-locus magnitude. Used as a single-sequence CONTROL
    (ref_single input, no twin): if the backbone+head can't learn even this easy target, the
    failure is systemic (LR / frozen backbone / scaling), not "the allelic contrast is too subtle."
    """
    return math.log(total_allele_reads(row) + pseudocount)

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

def alt_allele_of(row: pd.Series) -> Optional[str]:
    """
    The non-reference allele at a het-SNV: whichever of hap1_allele/hap2_allele differs
    from ref_allele. Returns None for rows where alt is undefined (neither or both
    haplotype alleles equal ref -- homozygous-looking or multiallelic rows), so callers
    can skip them rather than emit a garbage contrast.
    """
    ref = str(row["ref_allele"]).upper()
    h1 = str(row["hap1_allele"]).upper()
    h2 = str(row["hap2_allele"]).upper()
    non_ref = [a for a in (h1, h2) if a != ref]
    if len(set(non_ref)) != 1:        # 0 (homozygous-ref) or 2 distinct (multiallelic) -> undefined
        return None
    return non_ref[0]

def ref_count(row: pd.Series) -> float:
    return get_base_count(row, row["ref_allele"])

def alt_count(row: pd.Series) -> float:
    alt = alt_allele_of(row)
    if alt is None:
        return float("nan")
    return get_base_count(row, alt)

def signed_log_alt_ref(row: pd.Series, pseudocount: float = 1.0) -> float:
    """
    deltaSVM-style signed allelic effect, REFERENCE-oriented:
        log(alt_count + pc) - log(ref_count + pc)
    Positive  -> the ALT allele binds more than REF (variant increases binding)
    Negative  -> the ALT allele binds less than REF (variant decreases binding)
    Returns NaN where alt is undefined (see alt_allele_of); such rows are dropped at
    label-attach time. Pair this ONLY with a ref-first input (ref_alt_pair) so the
    target sign and the input ordering agree.
    """
    alt = alt_allele_of(row)
    if alt is None:
        return float("nan")
    return math.log(get_base_count(row, alt) + pseudocount) - math.log(ref_count(row) + pseudocount)

def _neg_log10_bb(p, eps: float = 1e-300) -> float:
    return -math.log10(max(float(p), eps))

def neg_log10_p_betabinom(row: pd.Series) -> float:
    """Privileged LUPI significance target -- NaN-EXCLUDE variant. -log10(p_betabinom); returns
    NaN when p is missing so the row is DROPPED at label-attach (drop_aux_nan=True). This is the
    baseline policy: untested sites are excluded, not relabeled 'balanced'. p_betabinom is a
    per-SNV MEASUREMENT property (known at train, absent for a novel sequence) -> privileged.
    Never use as a MAIN-task input."""
    p = row["p_betabinom"]
    return float("nan") if pd.isna(p) else _neg_log10_bb(p)

def neg_log10_p_betabinom_fill0(row: pd.Series) -> float:
    """Privileged LUPI significance target -- KEEP variant. -log10(p_betabinom), or 0.0 when p is
    missing. Use ONLY together with the p_tested flag, so the model can distinguish a real 0
    (tested & balanced) from a placeholder 0 (untested). For the 'keep untested rows as
    negatives' policy."""
    p = row["p_betabinom"]
    return 0.0 if pd.isna(p) else _neg_log10_bb(p)

def p_tested(row: pd.Series) -> int:
    """Privileged binary flag: 1 if the site had a beta-binomial test (p_betabinom present) else 0.
    Binary-valued; use it as an aux head with aux_task_type='binary'. Pairs with
    neg_log10_p_betabinom_fill0 so 'testability' (a depth/power proxy) is its own privileged signal."""
    return int(not pd.isna(row["p_betabinom"]))

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
        # assay: a single value; "ALL"/None to pool every assay for the donor;
        # or a comma-separated string / list to pool a chosen subset.
        donor_mask = chunk["donor"] == donor
        if assay is None or (isinstance(assay, str) and assay.strip().upper() == "ALL"):
            mask = donor_mask
        elif isinstance(assay, (list, tuple, set)):
            mask = donor_mask & chunk["assay"].isin(list(assay))
        elif isinstance(assay, str) and "," in assay:
            wanted = [a.strip() for a in assay.split(",") if a.strip()]
            mask = donor_mask & chunk["assay"].isin(wanted)
        else:
            mask = donor_mask & (chunk["assay"] == assay)

        if tissue is not None and str(tissue).strip().upper() != "ALL":
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
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add bed_start, bed_end, SNV, and snv_window_offset columns for SNV-centered windows.

    Window length is always fixed at left_bp + 1 + right_bp. The SNV is placed at offset
    `snv_window_offset` within the window:
      - snv_offset_mode == "fixed":   offset is always left_bp.
      - snv_offset_mode == "uniform": offset is drawn uniformly per example from
                                      [left_bp - jitter_max_bp, left_bp + jitter_max_bp],
                                      then clamped to whatever the contig boundaries allow.

    Rows whose fixed-length window cannot fit within the contig (or, in fixed mode, whose
    single allowed offset is infeasible at a boundary) are dropped, with a count reported.
    """
    df = df.copy()

    chrom_sizes = load_chrom_sizes(window_spec.chrom_sizes_path)

    left_bp = window_spec.left_bp
    right_bp = window_spec.right_bp
    window_len = left_bp + 1 + right_bp

    mode = getattr(window_spec, "snv_offset_mode", "fixed")
    jitter = int(getattr(window_spec, "jitter_max_bp", 0) or 0)

    if mode not in {"fixed", "uniform"}:
        raise ValueError(f"Unsupported snv_offset_mode: {mode!r}")

    if mode == "uniform":
        if jitter < 0:
            raise ValueError("jitter_max_bp must be non-negative.")
        if jitter > min(left_bp, right_bp):
            raise ValueError(
                f"jitter_max_bp={jitter} exceeds min(left_bp, right_bp)="
                f"{min(left_bp, right_bp)}; the SNV could fall outside the window."
            )

    # Center on a generic "anchor" genomic position so non-SNV row sources (peaks, tiles)
    # can reuse this windowing. SNV sources set anchor = ref_start. The column is still
    # named "SNV" downstream for backward compatibility.
    anchor_col = "anchor" if "anchor" in df.columns else "ref_start"
    df["SNV"] = df[anchor_col].astype(int)

    rng = np.random.default_rng(seed)

    # Desired offset band (before boundary clamping). In fixed mode this is a single value.
    band_lo = left_bp - jitter if mode == "uniform" else left_bp
    band_hi = left_bp + jitter if mode == "uniform" else left_bp

    offsets = np.empty(len(df), dtype=np.int64)
    bed_starts = np.empty(len(df), dtype=np.int64)
    bed_ends = np.empty(len(df), dtype=np.int64)
    keep = np.zeros(len(df), dtype=bool)

    chroms = df["chr"].to_numpy()
    snvs = df["SNV"].to_numpy()

    for i in range(len(df)):
        snv = int(snvs[i])
        csize = chrom_sizes.get(chroms[i]) if chrom_sizes else None

        # Feasible offset range so the whole fixed-length window fits in the contig:
        #   bed_start = snv - o >= 0           -> o <= snv
        #   bed_end   = snv - o + window_len <= csize  -> o >= snv + window_len - csize
        # plus keep the SNV inside the window: 0 <= o <= window_len - 1.
        o_lo = max(band_lo, 0)
        o_hi = min(band_hi, window_len - 1, snv)
        if csize is not None:
            o_lo = max(o_lo, snv + window_len - csize)

        if o_lo > o_hi:
            keep[i] = False
            offsets[i] = -1
            bed_starts[i] = -1
            bed_ends[i] = -1
            continue

        if mode == "uniform":
            o = int(rng.integers(o_lo, o_hi + 1))
        else:
            o = left_bp  # fixed; guaranteed within [o_lo, o_hi] by the check above

        offsets[i] = o
        bed_starts[i] = snv - o
        bed_ends[i] = snv - o + window_len
        keep[i] = True

    df["bed_start"] = bed_starts
    df["bed_end"] = bed_ends
    df["snv_window_offset"] = offsets

    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"add_snv_windows: dropped {n_drop} of {len(df)} windows that did not fit "
            f"the contig at the requested offset (mode={mode}, jitter={jitter})."
        )

    df = df[keep].reset_index(drop=True)

    if df.empty:
        raise ValueError("No windows remain after fitting fixed-length windows to contigs.")

    return df

def add_label_columns(
    df: pd.DataFrame,
    primary_label: LabelSpec,
    aux_labels: Optional[List[LabelSpec]] = None,
    drop_aux_nan: bool = True,
) -> pd.DataFrame:
    """
    Add primary and auxiliary label columns to the AS dataframe.

    Rows whose PRIMARY label is NaN are always dropped (undefined target).
    If drop_aux_nan is True (default), rows with a NaN in ANY auxiliary label are also dropped --
    a NaN aux value would otherwise be written to the training CSV and yield a NaN loss/gradient.
    This is also the mechanism behind the 'exclude untested sites' p_betabinom policy
    (neg_log10_p_betabinom returns NaN for untested sites). Set drop_aux_nan=False to keep NaN aux
    rows (e.g. when using neg_log10_p_betabinom_fill0 + p_tested instead).
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

    # Drop rows whose PRIMARY label is undefined (e.g. signed_log_alt_ref returns NaN where
    # alt is undefined). Safety net; ref_alt_pair already pre-filters these in add_sequence_inputs.
    before = len(df)
    df = df[pd.notna(df[primary_label.name])].copy()
    dropped = before - len(df)
    if dropped:
        print(f"add_label_columns: dropped {dropped} rows with undefined primary label "
              f"'{primary_label.name}'.")

    # Drop rows with a NaN AUXILIARY label
    #(an NaN aux value would otherwise be written into the training CSV and produce a NaN loss/gradient)
    if drop_aux_nan and aux_labels:
        aux_names = [s.name for s in aux_labels]
        before = len(df)
        df = df[df[aux_names].notna().all(axis=1)].copy()
        dropped = before - len(df)
        if dropped:
            print(f"add_label_columns: dropped {dropped} rows with NaN aux label(s) "
                  f"{aux_names} (drop_aux_nan=True).")

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
    Add sequence columns according to input_mode, plus per-sequence anchor/extent metadata.

    Requires ref_fasta to support:
        str(ref_fasta[chrom][start:end])
    as pyfaidx.Fasta does

    Metadata columns added (0-based, in the stored sequence-string space):
        anchor_offset_seq1, feat_start_seq1, feat_end_seq1   (always)
        anchor_offset_seq2, feat_start_seq2, feat_end_seq2   (paired modes only)
        feature_type                                         ("snv")

    NOTE (substitution-only engine): make_haplotype_sequence currently enforces single-base
    alleles, so every sequence has identical length and the anchor offset is the same in seq1
    and seq2. The per-sequence columns are computed independently so that adding indel support
    later (length-changing haplotypes) only requires changing how each sequence's offset/extent
    is derived, not the downstream contract.
    """

    single_modes = {"ref_single", "hap1_single", "hap2_single"}
    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}

    if input_mode not in single_modes and input_mode not in paired_modes:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    needs_hap1 = input_mode in {"hap1_single", "hap_pair", "ref_hap1_pair"}
    needs_hap2 = input_mode in {"hap2_single", "hap_pair", "ref_hap2_pair"}

    has_alleles = ("hap1_allele" in df.columns) and ("hap2_allele" in df.columns)
    if (needs_hap1 or needs_hap2) and not has_alleles:
        raise ValueError(
            f"input_mode={input_mode!r} needs haplotype alleles, but the row source provides "
            f"no hap1_allele/hap2_allele columns (has_variants=False)."
        )

    has_ref_allele = "ref_allele" in df.columns

    df = df.copy()

    # ref_alt_pair builds a REFERENCE-vs-ALTERNATE contrast (sequence1=ref, sequence2=alt).
    # It needs ref + haplotype alleles, and we drop rows where alt is undefined (homozygous-ref
    # or multiallelic) so the contrast -- and the signed_log_alt_ref target -- stay well-posed.
    if input_mode == "ref_alt_pair":
        if not (has_alleles and has_ref_allele):
            raise ValueError("ref_alt_pair needs ref_allele + hap1_allele/hap2_allele columns.")
        alt_defined = df.apply(lambda r: alt_allele_of(r) is not None, axis=1)
        n_drop = int((~alt_defined).sum())
        if n_drop:
            print(f"ref_alt_pair: dropping {n_drop} rows with undefined alt "
                  f"(homozygous-ref / multiallelic).")
        df = df[alt_defined].copy()

    sequences = []
    sequence1s = []
    sequence2s = []

    anchor1s, fstart1s, fend1s = [], [], []
    anchor2s, fstart2s, fend2s = [], [], []

    for _, row in df.iterrows():
        chrom = row["chr"]
        start = int(row["bed_start"])
        end = int(row["bed_end"])
        snv = int(row["SNV"])

        ref_seq = str(ref_fasta[chrom][start:end]).upper()

        # Sanity check that the anchor is in the provided window (should be by construction)
        if snv < start or snv >= end:
            raise ValueError(
                f"Anchor {chrom}:{snv} is outside extracted window {chrom}:{start}-{end}."
            )

        snv_offset = snv - start

        if len(ref_seq) != end - start:
            raise ValueError(
                f"Extracted sequence length mismatch at {chrom}:{start}-{end}. "
                f"Expected {end - start}, got {len(ref_seq)}."
            )

        ref_allele = str(row["ref_allele"]).upper() if has_ref_allele else None

        # Sanity check: confirm the reference FASTA base matches the table's ref allele.
        # Only meaningful for single-base variant sources.
        if ref_allele is not None and len(ref_allele) == 1:
            observed_ref_base = ref_seq[snv_offset].upper()
            if observed_ref_base != ref_allele:
                raise ValueError(
                    f"Reference allele mismatch at {chrom}:{snv}. "
                    f"FASTA has {observed_ref_base!r}, but table has ref_allele={ref_allele!r}. "
                    f"This likely indicates a coordinate convention issue."
                )

        # Haplotype sequences are built lazily, only for the modes that need them.
        hap1_allele = str(row["hap1_allele"]).upper() if has_alleles else None
        hap2_allele = str(row["hap2_allele"]).upper() if has_alleles else None
        hap1_seq = make_haplotype_sequence(ref_seq, snv_offset, hap1_allele) if needs_hap1 else None
        hap2_seq = make_haplotype_sequence(ref_seq, snv_offset, hap2_allele) if needs_hap2 else None

        # Per-sequence offset/extent. Substitution-only for now, so the anchor offset is
        # snv_offset in every sequence; extent length is the length of that sequence's allele
        # (1 for a variant-free anchor such as a peak summit).
        def offset_extent(allele):
            ext = max(1, len(allele)) if allele else 1
            return snv_offset, snv_offset, snv_offset + ext

        if input_mode == "ref_single":
            sequences.append(ref_seq)
            a1, s1, e1 = offset_extent(ref_allele)

        elif input_mode == "hap1_single":
            sequences.append(hap1_seq)
            a1, s1, e1 = offset_extent(hap1_allele)

        elif input_mode == "hap2_single":
            sequences.append(hap2_seq)
            a1, s1, e1 = offset_extent(hap2_allele)

        elif input_mode == "hap_pair":
            sequence1s.append(hap1_seq)
            sequence2s.append(hap2_seq)
            a1, s1, e1 = offset_extent(hap1_allele)
            a2, s2, e2 = offset_extent(hap2_allele)

        elif input_mode == "ref_hap1_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap1_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(hap1_allele)

        elif input_mode == "ref_hap2_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap2_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(hap2_allele)

        elif input_mode == "ref_alt_pair":
            alt_allele = alt_allele_of(row)  # not None: undefined-alt rows dropped above
            alt_seq = make_haplotype_sequence(ref_seq, snv_offset, alt_allele)
            sequence1s.append(ref_seq)
            sequence2s.append(alt_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(alt_allele)

        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        anchor1s.append(a1)
        fstart1s.append(s1)
        fend1s.append(e1)

        if input_mode in paired_modes:
            anchor2s.append(a2)
            fstart2s.append(s2)
            fend2s.append(e2)

    if input_mode in single_modes:
        df["sequence"] = sequences
    elif input_mode in paired_modes:
        df["sequence1"] = sequence1s
        df["sequence2"] = sequence2s

    df["anchor_offset_seq1"] = anchor1s
    df["feat_start_seq1"] = fstart1s
    df["feat_end_seq1"] = fend1s

    if input_mode in paired_modes:
        df["anchor_offset_seq2"] = anchor2s
        df["feat_start_seq2"] = fstart2s
        df["feat_end_seq2"] = fend2s

    # Preserve a source-provided feature_type; default to "snv" for the AS/SNV source.
    if "feature_type" not in df.columns:
        df["feature_type"] = "snv"

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

def add_locus_and_example_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a jitter-invariant locus_id and a unique, content-based example_id.

    - locus_id = sha1(chr|SNV)[:16]: the grouping / leakage / cross-individual-exclusion key.
      All rows for the same SNV locus -- across tissues and across jitter draws -- share it.
    - example_id = "<chr>:<SNV>:<tissue>:<occurrence>": unique per row within a
      (donor, assay) dataset, stable and content-based.
    """
    df = df.copy()

    df["locus_id"] = [
        hashlib.sha1(f"{c}|{int(s)}".encode()).hexdigest()[:16]
        for c, s in zip(df["chr"], df["SNV"])
    ]

    tissue = df["tissue"].astype(str) if "tissue" in df.columns else pd.Series(["NA"] * len(df), index=df.index)
    base = df["chr"].astype(str) + ":" + df["SNV"].astype(str) + ":" + tissue
    occ = base.groupby(base).cumcount().astype(str)
    df["example_id"] = base + ":" + occ

    return df


#############################
# AS label-spec factories
#############################

# Continuous AS targets derived directly from the hetSNV table columns.
AS_REGRESSION_TARGETS: Dict[str, Callable[[pd.Series], float]] = {
    "hap1_ratio": hap1_ratio,
    "ref_allele_ratio": ref_allele_ratio_label,
    "signed_log_count_ratio": signed_log_count_ratio,
    "signed_log_alt_ref": signed_log_alt_ref,
    "log_total_count": log_total_count,
    "abs_log_count_ratio": abs_log_count_ratio,
    "as_magnitude": as_magnitude_from_ratio,
    "neg_log10_p_betabinom": neg_log10_p_betabinom,
    "neg_log10_p_betabinom_fill0": neg_log10_p_betabinom_fill0,
    "p_tested": p_tested, # binary-valued; use with aux_task_type='binary'
}

# Columns each target needs present on a row (for compose-time validation).
_AS_COUNT_COLS = ["cA", "cC", "cG", "cT", "hap1_allele", "hap2_allele"]
_AS_TARGET_REQUIREMENTS: Dict[str, List[str]] = {
    "hap1_ratio": _AS_COUNT_COLS,
    "signed_log_count_ratio": _AS_COUNT_COLS,
    "signed_log_alt_ref": _AS_COUNT_COLS + ["ref_allele"],
    "log_total_count": _AS_COUNT_COLS,
    "abs_log_count_ratio": _AS_COUNT_COLS,
    "as_magnitude": _AS_COUNT_COLS,
    "ref_allele_ratio": ["ref_allele_ratio"],
    "neg_log10_p_betabinom": ["p_betabinom"],
    "neg_log10_p_betabinom_fill0": ["p_betabinom"],
    "p_tested": ["p_betabinom"],
}


def make_as_class_label_spec(name: str = "imbalance_significance") -> LabelSpec:
    """Binary/multi-level AS classification from the precomputed imbalance_significance column."""
    return LabelSpec(
        name=name,
        fn=imbalance_significance_label,
        task_type="classification",
        required_columns=["imbalance_significance"],
    )


def make_as_regression_label_spec(
    target: str,
    name: Optional[str] = None,
    transform_fn: Callable[[float], float] = lambda x: x,
) -> LabelSpec:
    """Continuous AS regression target (allelic ratio, signed log count ratio, etc.)."""
    if target not in AS_REGRESSION_TARGETS:
        raise ValueError(
            f"Unsupported AS regression target {target!r}. "
            f"Choose from {sorted(AS_REGRESSION_TARGETS)}."
        )
    return LabelSpec(
        name=name or target,
        fn=AS_REGRESSION_TARGETS[target],
        task_type="regression",
        transform_fn=transform_fn,
        required_columns=list(_AS_TARGET_REQUIREMENTS[target]),
    )


#############################
# Row sources
#############################

class RowSource:
    """
    Base class for a dataset row source.

    A row source loads a DataFrame of anchored examples that the rest of the pipeline
    (windowing -> sequence building -> labels -> split) consumes. Subclasses must produce,
    at minimum, the columns: "chr", "anchor" (0-based genomic position to center on), and
    "donor"/"tissue"/"assay" provenance. Variant sources additionally provide
    ref_allele/hap1_allele/hap2_allele (and AS sources the cA/cC/cG/cT counts).

    Capability flags let build_dataset reject invalid compositions up front:
      - has_variants: whether haplotype substitution is possible
      - supported_input_modes: which sequence input_modes make sense for this source
    """

    source_type: str = "base"
    has_variants: bool = False
    supported_input_modes: set = {"ref_single"}

    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"source_type": self.source_type, "has_variants": self.has_variants}


class SNVRowSource(RowSource):
    """SNV row source: wraps load_as_table (one row per het SNV per tissue)."""

    source_type = "snv_tsv"
    has_variants = True
    supported_input_modes = {
        "ref_single", "hap1_single", "hap2_single",
        "hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair",
    }

    def __init__(
        self,
        input_tsv: str,
        assay: str,
        donor: str,
        tissue: Optional[str] = None,
        min_total_reads: Optional[int] = None,
        chunksize: int = 100000,
    ):
        self.input_tsv = input_tsv
        self.assay = assay
        self.donor = donor
        self.tissue = tissue
        self.min_total_reads = min_total_reads
        self.chunksize = chunksize

    def load(self) -> pd.DataFrame:
        df = load_as_table(
            input_tsv=self.input_tsv,
            assay=self.assay,
            donor=self.donor,
            tissue=self.tissue,
            min_total_reads=self.min_total_reads,
            chunksize=self.chunksize,
        )
        df["anchor"] = df["ref_start"].astype(int)
        return df

    def describe(self) -> dict:
        return {
            "source_type": self.source_type,
            "has_variants": self.has_variants,
            "input_tsv": self.input_tsv,
            "assay": self.assay,
            "donor": self.donor,
            "tissue": self.tissue,
            "min_total_reads": self.min_total_reads,
        }


#############################
# Source-agnostic builder
#############################

def build_dataset(
    row_source: RowSource,
    output_dir: str,
    ref_fasta,
    primary_label: LabelSpec,
    window_spec: SNVWindowSpec,
    input_mode: str = "hap_pair",
    balance_spec: Optional[BalanceSpec] = None,
    balance_split: str = "all",
    aux_labels: Optional[List[LabelSpec]] = None,
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    skip_ambiguous: bool = True,
    group_cols: Optional[List[str]] = None,
    split_mode: str = "train_dev_test",
    exclude_loci: Optional[set] = None,
    dedup_sequences_across_splits: bool = True,
    partition_spec: Optional["PartitionSpec"] = None,
    drop_aux_nan: bool = True,
    depth_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Source-agnostic dataset builder.

    Composes any RowSource with any LabelSpec(s), validating up front that:
      - input_mode is supported by the row source, and
      - every label's required_columns are provided by the source (post-windowing).

    Writes minimal Trainer CSVs + rich .meta.csv sidecars; returns the final DataFrame.
    Leakage prevention is on by default (group by locus_id); pass group_cols=[] to disable.

    If partition_spec is given and enabled, a deterministic donor-invariant train/dev/test column
    is computed (hold-out-chromosome TEST + hashed genomic bins for TRAIN/DEV) and takes priority
    over the group-shuffle split. drop_aux_nan (default True) drops rows with a NaN auxiliary label.
    """
    balance_spec = balance_spec or BalanceSpec(strategy="none")
    aux_labels = aux_labels or []
    if group_cols is None:
        group_cols = ["locus_id"]

    if input_mode not in row_source.supported_input_modes:
        raise ValueError(
            f"input_mode={input_mode!r} is not supported by row source "
            f"{row_source.source_type!r} (supports {sorted(row_source.supported_input_modes)})."
        )

    df = row_source.load()
    if balance_split == "all":
        df = balance_as_table(df, balance_spec)
    elif balance_split != "train":
        raise ValueError(f"balance_split must be 'all' or 'train', got {balance_split!r}.")
    df = add_snv_windows(df, window_spec, seed=seed)

    # Compose-time label validation (post-windowing, so window columns are available).
    all_labels = [primary_label] + list(aux_labels)
    needed = set()
    for spec in all_labels:
        needed.update(getattr(spec, "required_columns", []) or [])
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise ValueError(
            f"Label(s) require columns not provided by row source "
            f"{row_source.source_type!r}: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    df = add_label_columns(df, primary_label=primary_label, aux_labels=aux_labels, drop_aux_nan=drop_aux_nan)
    df = add_sequence_inputs(df, ref_fasta=ref_fasta, input_mode=input_mode)
    df = add_locus_and_example_ids(df)

    split_col = None
    if partition_spec is not None and partition_spec.enabled:
        df["_assigned_split"] = assign_split_column(df, partition_spec)
        split_col = "_assigned_split"
        _counts = pd.Series(df["_assigned_split"]).value_counts().to_dict()
        print(f"partition_spec enabled (bin_size={partition_spec.bin_size}, "
              f"fold_id={partition_spec.fold_id}): row-level split counts {_counts}")
        if (df["_assigned_split"] == "__exclude__").any():
            _n0 = len(df)
            df = df[df["_assigned_split"] != "__exclude__"].copy()
            print(f"  boundary exclusion: dropped {_n0 - len(df)} loci whose window "
                  f"straddled a {partition_spec.bin_size}bp bin edge "
                  f"(boundary_bp={partition_spec.boundary_bp})")

    if group_cols:
        summarize_duplicate_as_windows(
            df,
            label_col=primary_label.name,
            group_cols=group_cols,
        )

    paired = input_mode in {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}
    meta_cols = [
        "example_id", "locus_id", "feature_type",
        "anchor_offset_seq1", "feat_start_seq1", "feat_end_seq1",
    ]
    if paired:
        meta_cols += ["anchor_offset_seq2", "feat_start_seq2", "feat_end_seq2"]
    meta_cols += ["chr", "SNV", "ref_allele", "hap1_allele", "hap2_allele",
                  "tissue", "donor", "assay"]
    # Carry the AS-call columns so a regression run can be mapped back to imbalance_significance
    # post-hoc (threshold |prediction| -> AUPRC vs the binary call), and depth for stratification.
    meta_cols += ["imbalance_significance", "ref_allele_ratio", "total_reads"]
    meta_cols = [c for c in meta_cols if c in df.columns]

    split_and_write_csvs(
        df=df,
        output_dir=output_dir,
        label_col=primary_label.name,
        input_mode=input_mode,
        aux_cols=[spec.name for spec in aux_labels],
        split_col=split_col,
        split_ratio=split_ratio,
        seed=seed,
        skip_ambiguous=skip_ambiguous,
        group_cols=group_cols,
        meta_cols=meta_cols,
        split_mode=split_mode,
        exclude_loci=exclude_loci,
        dedup_sequences_across_splits=dedup_sequences_across_splits,
        balance_spec=balance_spec,
        balance_split=balance_split,
        depth_col=depth_col,
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
        required_columns=["chr"],
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
        required_columns=["chr", "tissue"],
    )

#####################
# Peak BED Signal Utils
#####################

class PeakBedAnnotator:
    """
    Annotator for deriving labels from a peak BED / ENCODE narrowPeak file.

    Supports:
        - "binary":       1.0 if the queried region overlaps any peak, else 0.0   (classification)
        - "count":        number of overlapping peaks
        - "max_score":    max peak score among overlaps (e.g. narrowPeak signalValue)
        - "sum_score":    sum of peak scores among overlaps
        - "frac_covered": fraction of the queried region covered by peaks

    Region (mirrors BigWigSignalAnnotator):
        - "window":     [bed_start, bed_end)
        - "snv":        [SNV, SNV + 1)
        - "snv_radius": [SNV - radius_bp, SNV + radius_bp + 1)

    Coordinates are 0-based half-open (BED), matching the SNV coordinates the engine
    has already validated against the reference FASTA.

    narrowPeak score fields (BED6+4): score=col5, signalValue=col7, pValue=col8, qValue=col9
    (0-based column indices 4/6/7/8). Generic BED uses col5 (index 4) if present.
    """

    NARROWPEAK_FIELDS = {"score": 4, "signalValue": 6, "pValue": 7, "qValue": 8}

    def __init__(
        self,
        bed_path: str,
        mode: str = "binary",
        region: str = "snv",
        radius_bp: int = 0,
        score_field: str = "signalValue",
        is_narrowpeak: bool = True,
        missing_value: float = 0.0,
    ):
        self.bed_path = bed_path
        self.mode = mode
        self.region = region
        self.radius_bp = radius_bp
        self.score_field = score_field
        self.is_narrowpeak = is_narrowpeak
        self.missing_value = missing_value
        self._load(bed_path)

    def _load(self, bed_path: str):
        df = pd.read_csv(
            bed_path, sep="\t", header=None, comment="#", compression="infer", dtype=str
        )
        if df.shape[1] < 3:
            raise ValueError(
                f"{bed_path}: expected at least 3 BED columns, got {df.shape[1]}."
            )

        chroms = df[0].astype(str).to_numpy()
        starts = pd.to_numeric(df[1], errors="coerce").to_numpy()
        ends = pd.to_numeric(df[2], errors="coerce").to_numpy()

        if self.is_narrowpeak:
            if self.score_field not in self.NARROWPEAK_FIELDS:
                raise ValueError(
                    f"Unsupported narrowPeak score_field {self.score_field!r}. "
                    f"Choose from {sorted(self.NARROWPEAK_FIELDS)}."
                )
            score_col = self.NARROWPEAK_FIELDS[self.score_field]
        else:
            score_col = 4  # generic BED score column

        if score_col is not None and score_col < df.shape[1]:
            scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy()
        else:
            scores = np.zeros(len(df), dtype=float)

        valid = np.isfinite(starts) & np.isfinite(ends)
        chroms = chroms[valid]
        starts = starts[valid].astype(np.int64)
        ends = ends[valid].astype(np.int64)
        scores = scores[valid].astype(float)

        self.n_peaks = int(len(starts))
        self.max_width = int((ends - starts).max()) if self.n_peaks else 0

        # Per-chrom arrays, sorted by start for searchsorted-based overlap queries.
        self.by_chrom = {}
        for c in np.unique(chroms):
            m = chroms == c
            cs, ce, csc = starts[m], ends[m], scores[m]
            order = np.argsort(cs, kind="stable")
            self.by_chrom[c] = (cs[order], ce[order], csc[order])

        # Sanity: warn if the chosen score column is degenerate (e.g. all-zero col5 score).
        if self.mode in {"max_score", "sum_score"} and self.n_peaks and np.allclose(scores, scores[0]):
            print(
                f"PeakBedAnnotator WARNING: score_field={self.score_field!r} is constant "
                f"({scores[0]}) across all peaks in {bed_path}; the resulting label will be "
                f"degenerate. Did you mean a different score_field?"
            )

    def _region_from_row(self, row: pd.Series):
        chrom = row["chr"]

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
                f"Unsupported peak query region {self.region!r}. "
                "Choose from {'window', 'snv', 'snv_radius'}."
            )

        start = max(0, start)
        if end <= start:
            return chrom, None, None
        return chrom, start, end

    def _overlaps(self, chrom, qstart, qend):
        """Peaks with start < qend and end > qstart. Returns [(start, end, score), ...]."""
        if chrom not in self.by_chrom:
            return []
        starts, ends, scores = self.by_chrom[chrom]
        # Candidates start in [qstart - max_width, qend); max_width bounds the lower edge so a
        # peak that starts well before qstart but extends into it is still considered.
        lo = int(np.searchsorted(starts, qstart - self.max_width, side="left"))
        hi = int(np.searchsorted(starts, qend, side="left"))
        out = []
        for i in range(lo, hi):
            if ends[i] > qstart:
                out.append((int(starts[i]), int(ends[i]), float(scores[i])))
        return out

    def __call__(self, row: pd.Series) -> float:
        chrom, qstart, qend = self._region_from_row(row)
        if qstart is None:
            # Degenerate region: no overlaps. summarize_overlaps handles the empty case.
            return float(summarize_overlaps([], 0, 1, self.mode))
        overlaps = self._overlaps(chrom, qstart, qend)
        return float(summarize_overlaps(overlaps, qstart, qend, self.mode))


def make_peak_bed_label_spec(
    name: str,
    bed_path: str,
    mode: str = "binary",
    region: str = "snv",
    radius_bp: int = 0,
    score_field: str = "signalValue",
    is_narrowpeak: bool = True,
    missing_value: float = 0.0,
    transform_fn: Callable[[float], float] = lambda x: x,
    task_type: Optional[str] = None,
) -> LabelSpec:
    """
    Build a LabelSpec from a peak BED / narrowPeak file.

    task_type defaults to "classification" for mode="binary", else "regression".
    """
    annotator = PeakBedAnnotator(
        bed_path=bed_path,
        mode=mode,
        region=region,
        radius_bp=radius_bp,
        score_field=score_field,
        is_narrowpeak=is_narrowpeak,
        missing_value=missing_value,
    )

    if task_type is None:
        task_type = "classification" if mode == "binary" else "regression"

    return LabelSpec(
        name=name,
        fn=annotator,
        task_type=task_type,
        transform_fn=transform_fn,
        required_columns=["chr"],
    )