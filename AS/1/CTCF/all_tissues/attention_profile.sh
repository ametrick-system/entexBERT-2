module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

LAYERS=11
TOKEN="cls"
HEADS="all"
WINDOW_BP=256
NUM_EX=100

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/ref_single_classification"
PATCHED_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"
EXAMPLES_CSV="$RUN_DIR/analysis/test/representative_examples_${NUM_EX}/representative_examples_all.csv"
OUTPUT_DIR="$RUN_DIR/analysis/test/attention_profiles/full_${TOKEN}_layer${LAYER}_${HEADS}_alibi_removed_pm${WINDOW_BP}"

rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python $PROJECT_DIR/src/entexbert2/scripts/plot_attention_profiles.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --examples_csv $EXAMPLES_CSV \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path "$PATCHED_MODEL" \
  --input_mode ref_single \
  --source_token cls \
  --layers $LAYERS \
  --heads $HEADS \
  --remove_alibi \
  --n_per_category 100 \
  --left_bp 256 \
  --plot_window_bp $WINDOW_BP \
  --profile_correction none \
  --plot_values all \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --model_max_length 512