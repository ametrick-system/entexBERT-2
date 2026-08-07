#!/usr/bin/env python3

"""
Config-driven entexBERT-2 dataset runner.

Each experiment is a declarative config (YAML or JSON); this runner composes a row source
with primary/aux labels and calls the source-agnostic build_dataset. New formats are added
by (1) implementing a RowSource / make_*_label_spec in entexbert2.utils and (2) registering
it in ROW_SOURCE_BUILDERS / LABEL_BUILDERS below. New *experiments* need no code at all.

Usage:
    python run_experiment.py configs/as_ctcf_binary_1024_center.yaml
    python run_experiment.py exp.yaml --ref_fasta /data/hg38.fa --output_dir runs/foo

Example config:

    experiment: as_ctcf_binary_1024_center
    ref_fasta: /data/hg38.fa
    output_dir: runs/as_ctcf_binary_1024_center
    row_source: {type: snv_tsv, path: hetSNVs_default_AS.tsv,
                 assay: TF-ChIP-seq_CTCF, donor: ENC-002, tissue: null, min_total_reads: 10}
    primary_label: {type: as_class}
    aux_labels: []
    sequence: {input_mode: ref_single}
    window: {left_bp: 512, right_bp: 511, snv_offset_mode: fixed, jitter_max_bp: 0}
    balance: {strategy: none, label_col: imbalance_significance}
    split: {mode: train_dev_test, ratio: [0.8, 0.1, 0.1], seed: 42, group: locus,
            skip_ambiguous: true, exclude_loci_meta: []}
    head: {task: classification, num_labels: 2, head_num_layers: 1, head_hidden_size: -1}

    # OPTIONAL hybrid cross-individual partition. Omit (or enabled: false) to use the split block
    # above. When enabled it OVERRIDES the group-shuffle split: the chromosome(s) whose
    # fold_assignment value == fold_id become TEST; the rest are hashed into train/dev. K-fold-ready
    # — a later sweep just re-runs with a different fold_id. Nothing dataset-specific is in code.
    partition: {enabled: true, bin_size: 100000, salt: entexbert2_v1, fold_id: 0,
                fold_assignment: {chr7: 0, chr14: 0}}

Label types (primary_label / each aux):
    as_class       : {type, name?}                                         (classification)
    as_regression  : {type, target, name?, transform?}                     (regression)
                     target in {hap1_ratio, ref_allele_ratio, signed_log_count_ratio,
                                abs_log_count_ratio, as_magnitude}
    bigwig         : {type, path, target_name|name, signal_mode?, region?, radius_bp?,
                      transform?, missing_value?}                          (regression)
    peak           : {type, path, target_name|name, peak_mode?, score_field?, region?,
                      radius_bp?, format?, transform?, missing_value?}
"""

import argparse
import dataclasses
import json
import os
import sys

from pyfaidx import Fasta

from entexbert2.utils import (
    BalanceSpec,
    PartitionSpec,
    SNVWindowSpec,
    SNVRowSource,
    PeakBedRowSource,
    MultiTissuePeakRowSource,
    BetabinomCountRowSource,
    build_dataset,
    make_as_class_label_spec,
    make_as_regression_label_spec,
    make_bigwig_label_spec,
    make_peak_bed_label_spec,
    make_column_label_spec,
    log1p_transform,
    identity_transform,
)

NONE_TISSUE_TOKENS = {None, "null", "NONE", "None", "none", "", "all", "ALL"}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transform_fn(name):
    name = (name or "identity").lower()
    if name == "log1p":
        return log1p_transform
    if name == "identity":
        return identity_transform
    raise ValueError(f"Unsupported transform: {name!r}")


# ---------------------------------------------------------------------------
# Row source registry  (extension point #1)
# ---------------------------------------------------------------------------

def _build_snv_source(cfg):
    tissue = cfg.get("tissue")
    if tissue in NONE_TISSUE_TOKENS:
        tissue = None
    return SNVRowSource(
        input_tsv=cfg["path"],
        assay=cfg["assay"],
        donor=cfg["donor"],
        tissue=tissue,
        min_total_reads=cfg.get("min_total_reads"),
        chunksize=cfg.get("chunksize", 100000),
    )

def _build_peak_bed_source(cfg):
    tissue = cfg.get("tissue")
    if tissue in NONE_TISSUE_TOKENS:
        tissue = None
    return PeakBedRowSource(
        peak_path=cfg["path"],
        assay=cfg["assay"],
        donor=cfg["donor"],
        tissue=tissue,
        genome_sizes_path=cfg.get("genome_sizes"),
        is_narrowpeak=(cfg.get("format", "narrowpeak") == "narrowpeak"),
        summit_mode=cfg.get("summit_mode", "summit"),
        background_ratio=cfg.get("background_ratio", 1.0),
        background_gap_bp=cfg.get("background_gap_bp", 1000),
        exclude_chroms=cfg.get("exclude_chroms"),
        seed=cfg.get("seed", 42),
    )

def _build_multi_tissue_peak_source(cfg):
    tracks = cfg["tissue_tracks"]          # [{tissue, peak_path, bigwig_path}, ...]
    return MultiTissuePeakRowSource(
        datasets=tracks,
        assay=cfg["assay"],
        donor=cfg["donor"],
        genome_sizes_path=cfg.get("genome_sizes"),
        is_narrowpeak=(cfg.get("format", "narrowpeak") == "narrowpeak"),
        summit_mode=cfg.get("summit_mode", "summit"),
        merge_window_bp=cfg.get("merge_window_bp", 100),
        label_radius_bp=cfg.get("label_radius_bp", 32),
        background_ratio=cfg.get("background_ratio", 1.0),
        background_gap_bp=cfg.get("background_gap_bp", 1000),
        exclude_chroms=cfg.get("exclude_chroms"),
        seed=cfg.get("seed", 42),
    )

def _build_betabinom_source(cfg):
    return BetabinomCountRowSource(
        counts_csv=cfg["path"],
        assay=cfg.get("assay"),
        donor=cfg.get("donor"),
    )

ROW_SOURCE_BUILDERS = {
    "snv_tsv": _build_snv_source,
    "peak_bed": _build_peak_bed_source,
    "multi_tissue_peak": _build_multi_tissue_peak_source,
    "betabinom_counts": _build_betabinom_source,
}


# ---------------------------------------------------------------------------
# Label registry  (extension point #2)
# ---------------------------------------------------------------------------

# Default post-fn transform per label type.
_DEFAULT_TRANSFORM = {
    "as_class": "identity",
    "as_regression": "identity",
    "bigwig": "log1p",
    "peak": "log1p",
}


def _label_name(cfg, fallback):
    return cfg.get("name") or cfg.get("target_name") or fallback


def _build_as_class(cfg, _tf):
    return make_as_class_label_spec(name=cfg.get("name", "imbalance_significance"))


def _build_as_regression(cfg, tf):
    return make_as_regression_label_spec(
        target=cfg["target"], name=cfg.get("name"), transform_fn=tf
    )


def _build_bigwig(cfg, tf):
    return make_bigwig_label_spec(
        name=_label_name(cfg, "bigwig"),
        bigwig_path=cfg["path"],
        mode=cfg.get("signal_mode", "max"),
        region=cfg.get("region", "snv_radius"),
        radius_bp=cfg.get("radius_bp", 20),
        transform_fn=tf,
        missing_value=cfg.get("missing_value", 0.0),
        use_values=True,
        exact=True,
    )


def _build_peak(cfg, tf):
    return make_peak_bed_label_spec(
        name=_label_name(cfg, "peak"),
        bed_path=cfg["path"],
        mode=cfg.get("peak_mode", "binary"),
        region=cfg.get("region", "snv"),
        radius_bp=cfg.get("radius_bp", 0),
        score_field=cfg.get("score_field", "signalValue"),
        is_narrowpeak=(cfg.get("format", "narrowpeak") == "narrowpeak"),
        missing_value=cfg.get("missing_value", 0.0),
        transform_fn=tf,
    )

def _build_column(cfg, tf):
    return make_column_label_spec(
        column=cfg["column"], name=cfg.get("name"), transform_fn=tf
    )

LABEL_BUILDERS = {
    "as_class": _build_as_class,
    "as_regression": _build_as_regression,
    "bigwig": _build_bigwig,
    "column": _build_column,
    "peak": _build_peak,
}


def build_label(cfg):
    ltype = cfg["type"]
    if ltype not in LABEL_BUILDERS:
        raise ValueError(f"Unknown label type {ltype!r}. Known: {sorted(LABEL_BUILDERS)}.")
    tf = get_transform_fn(cfg.get("transform", _DEFAULT_TRANSFORM.get(ltype, "identity")))
    return LABEL_BUILDERS[ltype](cfg, tf)


def build_source(cfg):
    stype = cfg["type"]
    if stype not in ROW_SOURCE_BUILDERS:
        raise ValueError(f"Unknown row_source type {stype!r}. Known: {sorted(ROW_SOURCE_BUILDERS)}.")
    return ROW_SOURCE_BUILDERS[stype](cfg)


# ---------------------------------------------------------------------------
# Config loading / helpers
# ---------------------------------------------------------------------------

def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise ImportError("PyYAML is required for YAML configs; use JSON or `pip install pyyaml`.") from e
            return yaml.safe_load(f)
        if ext == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config extension {ext!r}; use .yaml/.yml/.json.")


def load_exclude_loci(meta_paths):
    import pandas as pd
    loci = set()
    for p in meta_paths or []:
        df = pd.read_csv(p)
        if "locus_id" not in df.columns:
            raise ValueError(f"{p} has no 'locus_id' column (expected a *.meta.csv).")
        loci.update(df["locus_id"].astype(str).tolist())
    return loci

def build_partition_spec(cfg):
    """
    Build a PartitionSpec from an optional top-level `partition:` config block. Returns None when
    the block is absent or partition.enabled is false (in which case build_dataset falls back to
    the group-shuffle split — fully backward-compatible).

    Nothing about any specific dataset lives in code: the held-out test chromosome(s), bin size,
    salt, and active fold all come from the config. `fold_assignment` maps chromosome -> fold index;
    the chromosome whose fold index == fold_id becomes the TEST set, the rest are hashed into
    train/dev. Chromosome keys are coerced to str so YAML ints (e.g. `1:`) match the 'chr' column.

    Config block (all keys optional; shown with PartitionSpec defaults):
        partition:
            enabled: true
            bin_size: 100000
            salt: entexbert2_v1
            fold_id: 0
            train_frac_within_nontest: 0.8888888888888888   # 8:1 train:dev
            fold_assignment: {chr7: 0, chr14: 0}            # chrom -> fold index
    """
    pcfg = cfg.get("partition")
    if not pcfg or not pcfg.get("enabled", False):
        return None

    raw_assignment = pcfg.get("fold_assignment", {}) or {}
    fold_assignment = {str(k): int(v) for k, v in raw_assignment.items()}

    fold_id = int(pcfg.get("fold_id", 0))
    if fold_assignment and fold_id not in set(fold_assignment.values()):
        raise ValueError(
            f"partition.fold_id={fold_id} is not present in fold_assignment "
            f"(folds {sorted(set(fold_assignment.values()))}); the TEST set would be empty."
        )

    # boundary_bp defaults to the window half-width (max of left/right) when boundary exclusion is on
    _wcfg = cfg.get("window", {}) or {}
    _default_bp = max(int(_wcfg.get("left_bp", 0)), int(_wcfg.get("right_bp", 0)))
    return PartitionSpec(
        enabled=True,
        bin_size=int(pcfg.get("bin_size", 100_000)),
        salt=str(pcfg.get("salt", "entexbert2_v1")),
        fold_assignment=fold_assignment,
        fold_id=fold_id,
        train_frac_within_nontest=float(pcfg.get("train_frac_within_nontest", 8.0 / 9.0)),
        bin_test_frac=float(pcfg.get("bin_test_frac", 0.0)),
        bin_dev_frac=float(pcfg.get("bin_dev_frac", 0.0)),
        exclude_boundary=bool(pcfg.get("exclude_boundary", False)),
        boundary_bp=int(pcfg.get("boundary_bp", _default_bp)),
    )

def resolve_head(head, primary_label):
    """
    Derive/validate the finetune head config from the primary label's task_type so the two
    can't disagree. The config's `head` block only needs architecture (head_num_layers,
    head_hidden_size) and, for classification, num_labels (the class count).
    """
    head = dict(head or {})
    task = primary_label.task_type  # authoritative: comes from the label

    if "task" in head and head["task"] != task:
        raise ValueError(
            f"head.task={head['task']!r} disagrees with the primary label's task_type={task!r}. "
            f"Omit head.task (it is derived) or fix it."
        )
    head["task"] = task

    if task == "regression":
        if head.get("num_labels", 1) != 1:
            print(f"  note: regression forces num_labels=1 (config had {head.get('num_labels')}).")
        head["num_labels"] = 1
    else:
        if "num_labels" not in head:
            print("  note: classification head defaulting to num_labels=2; set head.num_labels "
                  "explicitly for multi-bin labels.")
        head.setdefault("num_labels", 2)

    head.setdefault("head_num_layers", 1)
    head.setdefault("head_hidden_size", -1)
    return head


def emit_finetune_settings(head, output_dir, primary_name, aux_names):
    """Print the finetune settings recorded for this dataset (verified flags only)."""
    print("\nFinetune settings (map to finetune_entexbert2.py):")
    print(f"  --task {head['task']}")
    print(f"  --main_num_labels {head['num_labels']}")
    print(f"  --head_num_layers {head['head_num_layers']}  (1 = linear, >1 = MLP)")
    if head["head_hidden_size"] != -1:
        print(f"  --head_hidden_size {head['head_hidden_size']}")
    if aux_names:
        print(f"  (aux heads: {aux_names} — set --num_aux_tasks / --aux_task_types / --aux_num_labels)")
    print(f"  trainer data path: {output_dir}")
    print(f"  primary label column: {primary_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run an entexBERT-2 dataset generation experiment from a config.")
    p.add_argument("config", help="Path to the experiment config (.yaml/.yml/.json)")
    p.add_argument("--ref_fasta", default=None, help="Override ref_fasta from the config")
    p.add_argument("--output_dir", default=None, help="Override output_dir from the config")
    return p.parse_args()


def run_from_config(cfg, ref_fasta=None, output_dir=None):
    """
    Build a dataset from a config dict. Importable for notebooks/tests.

    ref_fasta / output_dir override the config when provided.
    Returns the final DataFrame.
    """
    name = cfg.get("experiment", "experiment")
    ref_fasta_path = ref_fasta or cfg["ref_fasta"]
    output_dir = output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    source = build_source(cfg["row_source"])
    primary_label = build_label(cfg["primary_label"])
    aux_labels = [build_label(c) for c in cfg.get("aux_labels", [])]

    wcfg = cfg.get("window", {})
    window_spec = SNVWindowSpec(
        left_bp=wcfg["left_bp"],
        right_bp=wcfg["right_bp"],
        chrom_sizes_path=wcfg.get("chrom_sizes"),
        snv_offset_mode=wcfg.get("snv_offset_mode", "fixed"),
        jitter_max_bp=wcfg.get("jitter_max_bp", 0),
    )

    scfg = cfg.get("split", {})
    seed = scfg.get("seed", 42)
    split_mode = scfg.get("mode", "train_dev_test")
    group_cols = ["locus_id"] if scfg.get("group", "locus") == "locus" else []
    split_ratio = tuple(scfg.get("ratio", (0.8, 0.1, 0.1)))
    skip_ambiguous = scfg.get("skip_ambiguous", True)
    exclude_loci = load_exclude_loci(scfg.get("exclude_loci_meta")) or None
    dedup_across_splits = scfg.get("dedup_across_splits", True)

    bcfg = cfg.get("balance", {})
    balance_spec = BalanceSpec(
        strategy=bcfg.get("strategy", "none"),
        label_col=bcfg.get("label_col", "imbalance_significance"),
        random_state=seed,
    )
    balance_split = bcfg.get("apply_to", "all")  # "all" = balance before split; "train" = train only
    if balance_split not in {"all", "train"}:
        raise ValueError(f"balance.apply_to must be 'all' or 'train', got {balance_split!r}.")

    input_mode = cfg.get("sequence", {}).get("input_mode", "hap_pair")
    depth_col = cfg.get("depth_col") # e.g. "total_reads" for the heteroscedastic head
    count_cols = cfg.get("count_cols") # e.g. ["k", "n"] for the beta-binomial task

    # Optional hybrid cross-individual partition (held-out test chrom(s) + hashed genomic bins).
    # None => fall back to the group-shuffle split. Everything dataset-specific is in the config.
    partition_spec = build_partition_spec(cfg)

    # Head is derived from the label's task_type (single source of truth).
    head = resolve_head(cfg.get("head"), primary_label)

    print(f"Experiment: {name}")
    print(f"  source:     {source.source_type} (has_variants={source.has_variants})")
    print(f"  primary:    {primary_label.name} [{primary_label.task_type}]")
    print(f"  aux:        {[a.name for a in aux_labels] or 'none'}")
    print(f"  input_mode: {input_mode}")
    print(f"  window:     L{window_spec.left_bp}/R{window_spec.right_bp} "
          f"offset={window_spec.snv_offset_mode} jitter={window_spec.jitter_max_bp}")
    print(f"  split:      {split_mode} group={'locus' if group_cols else 'none'} "
          f"exclude={len(exclude_loci) if exclude_loci else 0} loci")

    if partition_spec is not None:
        _test_chroms = sorted(c for c, f in partition_spec.fold_assignment.items()
                              if f == partition_spec.fold_id)
        print(f"  partition:  hybrid bin_size={partition_spec.bin_size} "
              f"fold_id={partition_spec.fold_id} test_chroms={_test_chroms} "
              f"(overrides group-shuffle)")
    print(f"  output_dir: {output_dir}")

    # Self-documenting manifest = the resolved config + derived head + resolved partition.
    # partition_resolved captures the FULL fold_assignment actually used, so a run is re-derivable
    # and a later K-fold sweep just re-runs with a different fold_id.
    partition_resolved = dataclasses.asdict(partition_spec) if partition_spec is not None else None
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump({"experiment": name, "resolved": cfg, "head_resolved": head,
                   "partition_resolved": partition_resolved,
                   "ref_fasta": ref_fasta_path, "output_dir": output_dir}, f, indent=2)

    print("Loading reference FASTA...")
    ref = Fasta(ref_fasta_path)

    df = build_dataset(
        row_source=source,
        output_dir=output_dir,
        ref_fasta=ref,
        primary_label=primary_label,
        window_spec=window_spec,
        input_mode=input_mode,
        balance_spec=balance_spec,
        balance_split=balance_split,
        aux_labels=aux_labels,
        split_ratio=split_ratio,
        seed=seed,
        skip_ambiguous=skip_ambiguous,
        group_cols=group_cols,
        split_mode=split_mode,
        exclude_loci=exclude_loci,
        dedup_sequences_across_splits=dedup_across_splits,
        partition_spec=partition_spec,
        depth_col=depth_col,
        count_cols=count_cols,
    )

    print(f"\nDone. Final rows: {len(df)}")
    emit_finetune_settings(head, output_dir, primary_label.name, [a.name for a in aux_labels])
    return df


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("experiment", os.path.splitext(os.path.basename(args.config))[0])
    run_from_config(cfg, ref_fasta=args.ref_fasta, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
