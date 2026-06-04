module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/hap_pair_classification"
PATCHED_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"

LAYER=0
TOKEN="cls"

rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python $PROJECT_DIR/src/entexbert2/scripts/plot_average_attention_profiles.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --examples_csv "$RUN_DIR/analysis/test/representative_examples/representative_examples_all.csv" \
  --output_dir "$RUN_DIR/analysis/test/attention_profiles/${TOKEN}_layer${LAYER}_allheads_n100" \
  --model_name_or_path "$PATCHED_MODEL" \
  --input_mode hap_pair \
  --source_token "$TOKEN" \
  --layers $LAYER \
  --heads all \
  --n_per_category 100 \
  --left_bp 256 \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --model_max_length 512