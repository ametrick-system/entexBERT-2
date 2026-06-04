module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/hap_pair_classification"
PATCHED_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"

rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python $PROJECT_DIR/src/entexbert2/scripts/bertviz_example.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --examples_csv "$RUN_DIR/analysis/test/representative_examples/representative_examples_all.csv" \
  --output_dir "$RUN_DIR/analysis/test/bertviz_examples" \
  --model_name_or_path "$PATCHED_MODEL" \
  --category TP \
  --rank 1 \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --model_max_length 512