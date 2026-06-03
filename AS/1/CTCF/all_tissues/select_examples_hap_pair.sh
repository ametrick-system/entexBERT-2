module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/hap_pair_classification"

python $PROJECT_DIR/src/entexbert2/scripts/select_representative_examples.py \
  --predictions_csv "$RUN_DIR/analysis/test/predictions.csv" \
  --output_dir "$RUN_DIR/analysis/test/representative_examples" \
  --n_per_category 10