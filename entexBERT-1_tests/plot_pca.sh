#!/bin/bash
set -eo pipefail

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

python plot_pca.py \
  --split val \
  --include_jitter \
  --out_png "$HOME/entexBERT-2/entexBERT-1_tests/results/ctcf_enc01_compare_uncapped/pca_attention_with_jitter.png" \
  --out_pdf "$HOME/entexBERT-2/entexBERT-1_tests/results/ctcf_enc01_compare_uncapped/pca_attention_with_jitter.pdf"