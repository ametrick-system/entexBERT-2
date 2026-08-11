#!/usr/bin/env python3

"""
Config-driven entexBERT-2 dataset runner (2-stage ASB pipeline).

Each experiment is a declarative config (YAML or JSON); this runner composes a row source
with a primary label and calls the source-agnostic build_dataset in entexbert2.build_inputs.
New formats are added by (1) implementing a RowSource / make_*_label_spec in build_inputs and
(2) registering it in ROW_SOURCE_BUILDERS / LABEL_BUILDERS below. New *experiments* need no code.

Two live pipelines:

  Stage 1 (binding trunk):  row_source multi_tissue_peak + label bigwig (or column)
  Stage 2 (ASB twin head):  row_source betabinom_counts + label logit_ratio, with
                            depth_col: n  (n carried through as the privileged weight)

Usage:
    python run_experiment.py configs/stage2_ctcf_asb.yaml
    python run_experiment.py exp.yaml --ref_fasta /data/hg38.fa --output_dir runs/foo

Example Stage-2 config:

    experiment: stage2_ctcf_asb
    ref_fasta: /data/hg38.fa
    output_dir: runs/stage2_ctcf_asb
    row_source: {type: betabinom_counts, path: ctcf_betabinom_counts.csv, donor: null}
    primary_label: {type: logit_ratio}          # y = logit((k+0.5)/(n+1))
    sequence: {input_mode: hap_pair}
    window: {left_bp: 128, right_bp: 128, snv_offset_mode: fixed, jitter_max_bp: 0}
    balance: {strategy: none}
    split: {mode: train_dev_test, ratio: [0.8, 0.1, 0.1], seed: 42, group: locus}
    head: {task: regression, num_labels: 1, head_num_layers: 1, head_hidden_size: -1}
    depth_col: n                                 # n -> `depth` weight column (w = n_eff)
    partition: {enabled: true, bin_size: 100000, salt: entexbert2_v1, fold_id: 0,
                fold_assignment: {chr5: 0, chr12: 0}}

Label types:
    logit_ratio : {type, name?}                  Stage-2 ASB target logit((k+0.5)/(n+1))  (regression)
    bigwig      : {type, path, name?, signal_mode?, region?, radius_bp?, transform?}       (regression)
    column      : {type, column, name?, transform?}   precomputed source column            (regression)
"""

import argparse
import dataclasses
import json
import os
import sys

from pyfaidx import Fasta

from entexbert2.build_inputs import (
    BalanceSpec,
    PartitionSpec,
    SNVWindowSpec,
    MultiTissuePeakRowSource,
    BetabinomCountRowSource,
    build_dataset,
    make_logit_ratio_label_spec,
    make_bigwig_label_spec,
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
    "multi_tissue_peak": _build_multi_tissue_peak_source,   # Stage 1 (binding trunk)
    "betabinom_counts": _build_betabinom_source,            # Stage 2 (ASB twin head)
}


# ---------------------------------------------------------------------------
# Label registry  (extension point #2)
# ---------------------------------------------------------------------------

# Default post-fn transform per label type.
_DEFAULT_TRANSFORM = {
    "logit_ratio": "identity",
    "bigwig": "log1p",
    "column": "identity",
}


def _label_name(cfg, fallback):
    return cfg.get("name") or cfg.get("target_name") or fallback


def _build_logit_ratio(cfg, _tf):
    return make_logit_ratio_label_spec(name=cfg.get("name", "logit_ratio"))


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


def _build_column(cfg, tf):
    return make_column_label_spec(
        column=cfg["column"], name=cfg.get("name"), transform_fn=tf
    )

LABEL_BUILDERS = {
    "logit_ratio": _build_logit_ratio,     # Stage 2 ASB target
    "bigwig": _build_bigwig,               # Stage 1 binding signal
    "column": _build_column,               # precomputed source column (e.g. consensus fold-change)
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
            fold_assignment: {chr5: 0, chr12: 0}            # chrom -> fold index
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
    head_hidden_size). The 2-stage ASB model is regression-only (num_labels=1).
    """
    head = dict(head or {})
    task = primary_label.task_type  # authoritative: comes from the label

    if "task" in head and head["task"] != task:
        raise ValueError(
            f"head.task={head['task']!r} disagrees with the primary label's task_type={task!r}. "
            f"Omit head.task (it is derived) or fix it."
        )
    head["task"] = task

    if task != "regression":
        raise ValueError(
            f"The streamlined 2-stage model supports task='regression' only, got {task!r}."
        )
    if head.get("num_labels", 1) != 1:
        print(f"  note: regression forces num_labels=1 (config had {head.get('num_labels')}).")
    head["num_labels"] = 1

    head.setdefault("head_num_layers", 1)
    head.setdefault("head_hidden_size", -1)
    return head


def emit_finetune_settings(head, output_dir, primary_name, depth_col):
    """Print the finetune settings recorded for this dataset (verified flags only)."""
    print("\nFinetune settings (map to finetune_entexbert2.py):")
    print(f"  --task {head['task']}")
    print(f"  --main_num_labels {head['num_labels']}")
    print(f"  --head_num_layers {head['head_num_layers']}  (1 = linear, >1 = MLP)")
    if head["head_hidden_size"] != -1:
        print(f"  --head_hidden_size {head['head_hidden_size']}")
    if depth_col:
        print(f"  --hetero_loss precision_neff --neff_s <s>   (privileged weight from '{depth_col}' -> depth)")
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
    depth_col = cfg.get("depth_col")   # Stage 2: "n" -> privileged precision weight (w = n_eff)
    count_cols = cfg.get("count_cols") # optional: extra count columns to carry into train.csv

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
    if depth_col:
        print(f"  depth_col:  {depth_col} (-> 'depth' privileged weight)")

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
    emit_finetune_settings(head, output_dir, primary_label.name, depth_col)
    return df


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("experiment", os.path.splitext(os.path.basename(args.config))[0])
    run_from_config(cfg, ref_fasta=args.ref_fasta, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
