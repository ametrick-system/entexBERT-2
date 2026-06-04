module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/hap_pair_classification"
PATCHED_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"

LAYER=11
TOKEN="cls"

rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python $PROJECT_DIR/src/entexbert2/scripts/plot_attention_linear_flank_corrected.py \
  --attention_csv "$RUN_DIR/analysis/test/attention_profiles/${TOKEN}_layer${LAYER}_allheads_n100/attention_profiles_long.csv" \
  --output_dir "$RUN_DIR/analysis/test/attention_profiles/${TOKEN}_layer${LAYER}_allheads_n100/linear_flank_corrected_pm100" \
  --value hap1_plus_hap2 \
  --window_bp 256 \
  --flank_inner_bp 100 \
  --flank_outer_bp 256 \
  --baseline_scope per_category \
  --title_suffix "source=${TOKEN}, layers=${LAYER}, heads=all, n/category=100"