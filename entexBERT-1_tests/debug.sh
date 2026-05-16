#!/bin/bash
set -eo pipefail

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"
META="$PROJECT_DIR/data/shift_exp_10k/offset_0/metadata.tsv"

python - <<'PY'
import pandas as pd
from pathlib import Path

meta = pd.read_csv(Path.home() / "entexBERT-2/entexBERT-1_tests/data/shift_exp_10k/offset_0/metadata.tsv", sep="\t")

coord_cols = ["chr", "ref_start", "ref_end"]

dup_counts = meta.groupby(coord_cols).size()
print("num rows:", len(meta))
print("num unique coords:", len(dup_counts))
print("coords repeated >1:", (dup_counts > 1).sum())
print("max repeats for one coord:", dup_counts.max())

# Check whether any coordinate appears in multiple splits
split_counts = meta.groupby(coord_cols)["split"].nunique()
print("coords appearing in multiple splits:", (split_counts > 1).sum())
PY