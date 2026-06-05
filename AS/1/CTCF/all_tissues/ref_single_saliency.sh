module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"

RUN_DIR="$PROJECT_DIR/AS/1/CTCF/all_tissues/ref_single_classification"
PATCHED_MODEL="$PROJECT_DIR/DNABERT-2-117M-attention"
MOTIFS_CSV="$PROJECT_DIR/motifs/ctcf.csv"

python $PROJECT_DIR/src/entexbert2/scripts/plot_saliency.py \
  --checkpoint_dir "$RUN_DIR/output" \
  --examples_csv "$RUN_DIR/analysis/test/representative_examples_100/representative_examples_all.csv" \
  --output_dir "$RUN_DIR/analysis/test/saliency_motif_full/positive_logit_pm256" \
  --model_name_or_path "$PATCHED_MODEL" \
  --main_task classification \
  --main_num_labels 2 \
  --input_mode ref_single \
  --categories TP,FP,TN,FN \
  --n_per_group 100 \
  --saliency_target positive_logit \
  --saliency_method l2 \
  --normalize_per_example max \
  --plot_values ref \
  --plot_window_bp 256 \
  --left_bp 256 \
  --center_label SNV \
  --overlay_motifs \
  --motifs_csv "$MOTIFS_CSV" \
  --motif_sigma 5 \
  --make_heatmaps \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --model_max_length 512