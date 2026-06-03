module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$PROJECT_DIR/AS/1/CTCF/all_tissues/ref_single_classification"

python $PROJECT_DIR/src/entexbert2/scripts/analyze_classification_run.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --data_csv "$RUN_DIR/input/test.csv" \
  --output_dir "$RUN_DIR/analysis/test" \
  --model_name_or_path "zhihan1996/DNABERT-2-117M" \
  --input_mode ref_single \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --batch_size 16 \
  --model_max_length 512

conda deactivate