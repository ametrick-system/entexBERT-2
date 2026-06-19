JITTER_DIR="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/ref_single_classification_256_256_jitter50"

# 1) SNV-focused ISM (cheap, decisive): a handful of forward passes per example
python $HOME/entexBERT-2/src/entexbert2/scripts/plot_attribution_profiles.py \
  --checkpoint_dir $JITTER_DIR/output \
  --examples_csv   $JITTER_DIR/analysis/representative_examples_all.csv \
  --output_dir     $JITTER_DIR/analysis/attribution \
  --method ism --ism_mode snv --score_mode margin

# 2) then add the windowed ISM profile + IG companion (heavier)
python $HOME/entexBERT-2/src/entexbert2/scripts/plot_attribution_profiles.py \
  --checkpoint_dir $JITTER_DIR/output \
  --examples_csv   $JITTER_DIR/analysis/representative_examples_all.csv \
  --output_dir     $JITTER_DIR/analysis/attribution \
  --method both --ism_mode both --ism_window_bp 100 --plot_window_bp 100