module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$PROJECT_DIR/AS/1/CTCF/all_tissues/ref_single_classification"

python $PROJECT_DIR/src/entexbert2/scripts/select_representative_examples.py \
  --predictions_csv "$RUN_DIR/analysis/test/predictions.csv" \
  --output_dir "$RUN_DIR/analysis/test/representative_examples" \
  --n_per_category 100

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/ref_single_classification"

python $PROJECT_DIR/src/entexbert2/scripts/plot_ref_single_saliency.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --examples_csv "$RUN_DIR/analysis/test/representative_examples/representative_examples_all.csv" \
  --output_dir "$RUN_DIR/analysis/test/saliency_representative_examples_100" \
  --model_name_or_path "zhihan1996/DNABERT-2-117M" \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --left_bp 256 \
  --model_max_length 512 \
  --saliency_method grad_x_input \
  --normalize_per_example