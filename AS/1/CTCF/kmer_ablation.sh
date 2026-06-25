OUT="$HOME/entexBERT-2/AS/1/CTCF/all_tissues/ref_single_classification_256_256_jitter50_baltrain"

python $HOME/entexBERT-2/src/entexbert2/scripts/plot_attribution_profiles.py \
  --checkpoint_dir $OUT/output \
  --examples_csv   $OUT/analysis/representative_examples_all.csv \
  --output_dir     $OUT/analysis/attribution \
  --input_mode ref_single \
  --method ism --ism_mode kmer \
  --kmer_sizes 3,6,10,15 --kmer_replacement dinuc --kmer_n_shuffles 3 \
  --ism_window_bp 100 --kmer_stride 1 \
  --batch_size 256 --n_per_category 100 --left_bp 256 --model_max_length 512 --device cuda