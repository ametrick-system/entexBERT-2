OUT="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/ref_single_classification_256_256_jitter50"

python $HOME/entexBERT-2/src/entexbert2/scripts/probe_embeddings.py \
  --pca_csv $OUT/analysis/pca.csv --predictions_csv $OUT/analysis/predictions.csv \
  --output_dir $OUT/analysis/embedding_probe --n_pcs 5 \
  --snv_pos_col snv_pos --left_bp 256 --right_bp 256       # only if you also pass tracks