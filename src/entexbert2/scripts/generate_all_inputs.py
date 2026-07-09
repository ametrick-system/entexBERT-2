#!/usr/bin/env python3
"""
generate_all_inputs.py — one entry point that generates training inputs for MANY datasets
(donor x assay [x tissue]) from a single base config, reusing run_experiment.run_from_config.

Why one script: every dataset must be built with the SAME window/label/balance settings AND, above
all, the SAME partition (identical fold_assignment across every dataset). Holding the partition in one
shared base config and looping is what keeps the cross-individual test set honest — a locus lands in
the same split for every donor/assay, so the deferred cross-individual evaluation is leakage-free by
construction. Per-dataset copies override ONLY the row-source selectors and the output directory.

Nothing dataset-specific is hardcoded: the base config supplies ref_fasta, the TSV path, labels,
window, balance, head, and the `partition:` block; the dataset list supplies which (donor, assay,
tissue) triples to build. Both come from files.

Dataset list — provide exactly one source:
  --datasets_csv PATH   a CSV with columns donor, assay, and optionally tissue
                     (the per-dataset CSV that plan_inputs.py writes drops in directly)
  or a top-level `datasets:` list in the base config:
       datasets:
         - {donor: ENC-002, assay: TF-ChIP-seq_CTCF}
         - {donor: ENC-003, assay: TF-ChIP-seq_CTCF, tissue: liver}

Output layout (per dataset, fold-scoped so folds never collide on disk):
  <base output_dir>/<donor>__<assay>[__<tissue>]/fold<fold_id>/{train,dev,test}.csv + manifests
A top-level generate_all_inputs_manifest.json records the base config, the resolved shared
partition, and one row per dataset (status + resolved output_dir), so the whole batch is re-derivable
and a later K-fold sweep is the same call with a different partition.fold_id.

Usage:
  python generate_all_inputs.py base_config.yaml --datasets_csv input_plan/ctcf_datasets.csv
  python generate_all_inputs.py base_config.yaml --dry_run          # print the plan, build nothing
  python generate_all_inputs.py base_config.yaml --continue_on_error --skip_existing
"""

import argparse
import copy
import csv
import json
import os
import re
import sys
import traceback

# Reuse the single-experiment machinery — do NOT reimplement dataset building here
from entexbert2.run_experiment import load_config, run_from_config, build_partition_spec

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate entexBERT-2 training inputs for many datasets from one base config.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Base experiment config (.yaml/.yml/.json) — shared settings + partition.")
    p.add_argument("--datasets_csv", default=None,
                   help="CSV with columns donor, assay[, tissue]. Overrides a `datasets:` block in the config.")
    p.add_argument("--ref_fasta", default=None, help="Override ref_fasta from the config (applies to all datasets).")
    p.add_argument("--output_dir", default=None, help="Override the base output_dir from the config.")
    p.add_argument("--dry_run", action="store_true", help="Print the plan (datasets + partition) and build nothing.")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Keep going if a dataset fails; record the error in the manifest instead of aborting.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a dataset whose output dir already contains train.csv (idempotent re-runs).")
    p.add_argument("--limit", type=int, default=None, help="Only build the first N datasets (for testing).")
    return p.parse_args()


def _sanitize(s):
    """Filesystem-safe token: keep alnum/._-, collapse everything else to '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "NA"


def dataset_id(dataset):
    """donor__assay[__tissue], sanitized, for the per-dataset output directory."""
    parts = [dataset["donor"], dataset["assay"]]
    if dataset.get("tissue") not in (None, "", "ALL"):
        parts.append(dataset["tissue"])
    return "__".join(_sanitize(p) for p in parts)


def load_datasets(args, cfg):
    """Dataset list from --datasets_csv (preferred) or a `datasets:` block in the config. Exactly one required."""
    if args.datasets_csv:
        datasets = []
        with open(args.datasets_csv, newline="") as f:
            reader = csv.DictReader(f)
            if "donor" not in reader.fieldnames or "assay" not in reader.fieldnames:
                raise ValueError(f"{args.datasets_csv} must have at least 'donor' and 'assay' columns; "
                                 f"found {reader.fieldnames}.")
            for row in reader:
                dataset = {"donor": row["donor"].strip(), "assay": row["assay"].strip()}
                tissue = (row.get("tissue") or "").strip()
                if tissue:
                    dataset["tissue"] = tissue
                datasets.append(dataset)
        source = f"datasets_csv={args.datasets_csv}"
    else:
        datasets = cfg.get("datasets")
        if not datasets:
            raise ValueError("No datasets: pass --datasets_csv or add a top-level `datasets:` list to the config.")
        # normalize: require donor+assay per entry
        for c in datasets:
            if "donor" not in c or "assay" not in c:
                raise ValueError(f"Each `datasets:` entry needs 'donor' and 'assay'; got {c}.")
        source = "config `datasets:` block"

    # de-duplicate on the (donor, assay, tissue) identity while preserving order
    seen, uniq = set(), []
    for c in datasets:
        key = (c["donor"], c["assay"], c.get("tissue"))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if len(uniq) < len(datasets):
        print(f"Note: dropped {len(datasets) - len(uniq)} duplicate dataset(s).")
    return uniq, source


def make_dataset_cfg(base_cfg, dataset, base_output_dir, fold_id):
    """Deep-copy the base config, override ONLY the row-source selectors + output_dir. Partition untouched."""
    cfg = copy.deepcopy(base_cfg)
    cfg.pop("datasets", None)  # the per-dataset config describes a single dataset
    rs = cfg.setdefault("row_source", {})
    rs["donor"] = dataset["donor"]
    rs["assay"] = dataset["assay"]
    if "tissue" in dataset:
        rs["tissue"] = dataset["tissue"]
    cfg["experiment"] = dataset_id(dataset)
    cfg["output_dir"] = os.path.join(base_output_dir, dataset_id(dataset), f"fold{fold_id}")
    return cfg


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    base_output_dir = args.output_dir or base_cfg.get("output_dir")
    if not base_output_dir:
        raise ValueError("No base output_dir: set it in the config or pass --output_dir.")

    # The shared partition — resolved once from the base config, identical for every dataset.
    shared_partition = build_partition_spec(base_cfg)
    fold_id = shared_partition.fold_id if shared_partition is not None else 0
    if shared_partition is not None:
        test_chroms = sorted(c for c, f in shared_partition.fold_assignment.items()
                             if f == shared_partition.fold_id)
    else:
        test_chroms = None

    datasets, datasets_source = load_datasets(args, base_cfg)
    if args.limit is not None:
        datasets = datasets[:args.limit]

    print(f"generate_all_inputs: {len(datasets)} dataset(s) from {datasets_source}")
    print(f"  base output_dir: {base_output_dir}")
    if shared_partition is not None:
        print(f"  SHARED partition: bin_size={shared_partition.bin_size} fold_id={fold_id} "
              f"test_chroms={test_chroms} (identical across all datasets)")
    else:
        print("  partition: none (group-shuffle split per dataset — NOT cross-individual honest)")
    for c in datasets:
        print(f"    - {dataset_id(c)}  ->  {os.path.join(base_output_dir, dataset_id(c), f'fold{fold_id}')}")

    if args.dry_run:
        print("\n[dry run] built nothing.")
        return

    results = []
    for i, dataset in enumerate(datasets, 1):
        cid = dataset_id(dataset)
        dataset_cfg = make_dataset_cfg(base_cfg, dataset, base_output_dir, fold_id)
        out = dataset_cfg["output_dir"]
        if args.skip_existing and os.path.exists(os.path.join(out, "train.csv")):
            print(f"\n[{i}/{len(datasets)}] SKIP {cid} (train.csv exists)")
            results.append({"dataset": dataset, "dataset_id": cid, "output_dir": out, "status": "skipped"})
            continue
        print(f"\n[{i}/{len(datasets)}] BUILD {cid}")
        try:
            df = run_from_config(dataset_cfg, ref_fasta=args.ref_fasta)
            results.append({"dataset": dataset, "dataset_id": cid, "output_dir": out,
                            "status": "ok", "n_rows": int(len(df))})
        except Exception as e:
            if not args.continue_on_error:
                raise
            print(f"    ERROR ({type(e).__name__}): {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({"dataset": dataset, "dataset_id": cid, "output_dir": out,
                            "status": "error", "error": f"{type(e).__name__}: {e}"})

    # Top-level manifest: base config + shared partition + one row per dataset (re-derivable, K-fold-ready).
    os.makedirs(base_output_dir, exist_ok=True)
    manifest = {
        "base_config": args.config,
        "datasets_source": datasets_source,
        "fold_id": fold_id,
        "shared_partition_resolved": (
            {"enabled": shared_partition.enabled, "bin_size": shared_partition.bin_size,
             "salt": shared_partition.salt, "fold_assignment": shared_partition.fold_assignment,
             "fold_id": shared_partition.fold_id,
             "train_frac_within_nontest": shared_partition.train_frac_within_nontest}
            if shared_partition is not None else None),
        "datasets": results,
    }
    mpath = os.path.join(base_output_dir, "generate_all_inputs_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    n_ok = sum(r["status"] == "ok" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    n_err = sum(r["status"] == "error" for r in results)
    print(f"\nDone. ok={n_ok} skipped={n_skip} error={n_err}. Manifest: {mpath}")
    if n_err:
        sys.exit(1)

if __name__ == "__main__":
    main()