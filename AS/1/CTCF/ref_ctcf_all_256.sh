#!/bin/bash
set -euo pipefail

########################################
# Environment setup
########################################

module purge
module load miniconda
conda activate eb2

########################################
# Project paths
########################################

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"

# Make sure local src package imports work
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

########################################
# Experiment identity
########################################

DONOR="ENC-001"
DONOR_NUMBER="1"

TISSUE_ARG="ALL"
TISSUE_LABEL="all_tissues"

ASSAY="TF-ChIP-seq_CTCF"
ASSAY_NICKNAME="CTCF"

########################################
# Dataset parameters
########################################

MIN_TOTAL_READS=10

LEFT_BP=128
RIGHT_BP=128

INPUT_MODE="ref_single"

# Normal all-tissue AS classification:
# balance positives/negatives separately within each tissue
BALANCE_STRATEGY="per_tissue_binary"

SPLIT_TRAIN=0.8
SPLIT_DEV=0.1
SPLIT_TEST=0.1

SEED=42
CHUNKSIZE=100000

RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_AS_classification"

########################################
# Input files
########################################

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

########################################
# Output locations
########################################

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}"

DATA_DIR="${EXPERIMENT_DIR}/input"
OUTPUT_DIR="${EXPERIMENT_DIR}/output"
FIGURE_DIR="${EXPERIMENT_DIR}/figures"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$FIGURE_DIR" "$LOG_DIR"

LOGFILE="${LOG_DIR}/${RUN_NAME}.log"

# Save all stdout/stderr to logfile and terminal
exec > >(tee -a "$LOGFILE") 2>&1

SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

echo "========================================"
echo "Starting experiment: $RUN_NAME"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "========================================"

########################################
# Training parameters
########################################

MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"

POOLING_MODE="cls"

HEAD_NUM_LAYERS=1
HEAD_HIDDEN_SIZE=-1
HEAD_ACTIVATION="gelu"
HEAD_DROPOUT=0.1

PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=50

LOGGING_STEPS=50
EVAL_STEPS=100
SAVE_STEPS=100

########################################
# Sanity checks
########################################

echo
echo "Checking input files..."

for path in "$AS_TSV" "$REF_FASTA" "$CHROM_SIZES"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Missing required file: $path"
        exit 1
    fi
done

if [[ ! -f "$PROJECT_DIR/src/entexbert2/scripts/make_as_classification_dataset.py" ]]; then
    echo "ERROR: Missing dataset script:"
    echo "$PROJECT_DIR/src/entexbert2/scripts/make_as_classification_dataset.py"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/src/entexbert2/finetune_entexbert2.py" ]]; then
    echo "ERROR: Missing trainer script:"
    echo "$PROJECT_DIR/src/entexbert2/finetune_entexbert2.py"
    exit 1
fi

echo "All required files found."

echo
echo "GPU status:"
nvidia-smi || true

echo
echo "Python/package sanity check:"
python - <<'PY'
import sys
print("Python:", sys.executable)
import entexbert2
print("Imported entexbert2 from:", entexbert2.__file__)
PY

echo
echo "====================================="
echo "Building AS classification dataset..."
echo "====================================="

python $SCRIPTS_DIR/make_as_classification_dataset.py \
    --as_tsv "$AS_TSV" \
    --ref_fasta "$REF_FASTA" \
    --output_dir "$DATA_DIR" \
    --assay "$ASSAY" \
    --donor "$DONOR" \
    --tissue "$TISSUE_ARG" \
    --min_total_reads "$MIN_TOTAL_READS" \
    --left_bp "$LEFT_BP" \
    --right_bp "$RIGHT_BP" \
    --chrom_sizes "$CHROM_SIZES" \
    --input_mode "$INPUT_MODE" \
    --balance_strategy "$BALANCE_STRATEGY" \
    --split_ratio "$SPLIT_TRAIN" "$SPLIT_DEV" "$SPLIT_TEST" \
    --seed "$SEED" \
    --chunksize "$CHUNKSIZE"

echo
echo "Dataset files:"
ls -lh "$DATA_DIR"

echo
echo "===================================================="
echo "Checking duplicate sequence leakage across splits..."
echo "===================================================="

python - <<PY
import os
import sys
import pandas as pd

data_dir = "$DATA_DIR"

paths = {
    "train": os.path.join(data_dir, "train.csv"),
    "dev": os.path.join(data_dir, "dev.csv"),
    "test": os.path.join(data_dir, "test.csv"),
}

dfs = {}
for split, path in paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dfs[split] = pd.read_csv(path)

for split, df in dfs.items():
    if "sequence" not in df.columns:
        raise ValueError(f"{split}.csv does not contain a 'sequence' column. Got columns: {list(df.columns)}")

seq_sets = {split: set(df["sequence"].astype(str)) for split, df in dfs.items()}

overlaps = {
    "train_dev": seq_sets["train"] & seq_sets["dev"],
    "train_test": seq_sets["train"] & seq_sets["test"],
    "dev_test": seq_sets["dev"] & seq_sets["test"],
}

print("Unique sequences per split:")
for split, seqs in seq_sets.items():
    print(f"  {split}: {len(seqs)} unique sequences / {len(dfs[split])} rows")

has_leakage = False
for name, overlap in overlaps.items():
    print(f"{name} duplicate sequences across splits: {len(overlap)}")
    if len(overlap) > 0:
        has_leakage = True

if has_leakage:
    print("\\nERROR: Duplicate input sequences found across train/dev/test splits.")
    print("This can inflate evaluation metrics. Use group-aware splitting by window/SNV.")
    sys.exit(1)

print("No exact duplicate reference sequences found across splits.")
PY

echo
echo "==============================================="
echo "Training entexBERT-2 AS classification model..."
echo "==============================================="

python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --task classification \
    --main_num_labels 2 \
    --pooling_mode "$POOLING_MODE" \
    --head_num_layers "$HEAD_NUM_LAYERS" \
    --head_hidden_size "$HEAD_HIDDEN_SIZE" \
    --head_activation "$HEAD_ACTIVATION" \
    --head_dropout "$HEAD_DROPOUT" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model eval_auroc \
    --greater_is_better True \
    --save_total_limit 3 \
    --eval_and_save_results True \
    --save_model True \
    --fp16 True \
    --dataloader_pin_memory False \
    --seed "$SEED"

########################################
# Final summary
########################################

echo
echo "========================================"
echo "Experiment complete!"
echo "Date: $(date)"
echo "Run name: $RUN_NAME"
echo "Dataset dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Figure dir: $FIGURE_DIR"
echo "Log file: $LOGFILE"
echo "========================================"

if [[ -f "$OUTPUT_DIR/results/$RUN_NAME/eval_results.json" ]]; then
    echo
    echo "Final eval results:"
    cat "$OUTPUT_DIR/results/$RUN_NAME/eval_results.json"
else
    echo
    echo "No eval_results.json found at:"
    echo "$OUTPUT_DIR/results/$RUN_NAME/eval_results.json"
fi

echo
echo "================================================"
echo "Selecting TP/FP/TN/FN Representative Examples..."
echo "================================================"

python $PROJECT_DIR/src/entexbert2/scripts/analyze_classification_run.py \
  --checkpoint_dir "$EXPERIMENT_DIR/output" \
  --data_csv "$EXPERIMENT_DIR/input/test.csv" \
  --output_dir "$EXPERIMENT_DIR/analysis/" \
  --model_name_or_path "zhihan1996/DNABERT-2-117M" \
  --input_mode ref_single \
  --pooling_mode cls \
  --head_num_layers 1 \
  --head_hidden_size -1 \
  --head_activation gelu \
  --head_dropout 0.1 \
  --batch_size 16 \
  --model_max_length 257

TOKEN="cls"
LAYERS=11
HEADS="all"
NUM_EX=100

python $SCRIPTS_DIR/select_representative_examples.py \
  --predictions_csv "$EXPERIMENT_DIR/analysis/predictions.csv" \
  --output_dir "$EXPERIMENT_DIR/analysis/representative_examples_${NUM_EX}" \
  --n_per_category $NUM_EX

echo
echo "============================="
echo "Plotting Attention Profile..."
echo "============================="

INFERENCE_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"
EXAMPLES_CSV="$EXPERIMENT_DIR/analysis/representative_examples_${NUM_EX}/representative_examples_all.csv"
ATTENTION_OUTPUT="$EXPERIMENT_DIR/analysis/attention_profiles/full_${TOKEN}_layers(${LAYERS})_heads(${HEADS})_alibi_removed_left(${LEFT_BP})right(${RIGHT_BP})"

rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python $PROJECT_DIR/src/entexbert2/scripts/plot_attention_profiles.py \
  --checkpoint_dir "$EXPERIMENT_DIR/output" \
  --examples_csv $EXAMPLES_CSV \
  --output_dir $ATTENTION_OUTPUT \
  --model_name_or_path "$INFERENCE_MODEL" \
  --input_mode ref_single \
  --source_token $TOKEN \
  --layers $LAYERS \
  --heads $HEADS \
  --remove_alibi \
  --n_per_category $NUM_EX \
  --left_bp $LEFT_BP \
  --plot_window_bp $LEFT_BP \
  --profile_correction none \
  --plot_values all \
  --pooling_mode $POOLING_MODE \
  --head_num_layers $HEAD_NUM_LAYERS \
  --head_hidden_size $HEAD_HIDDEN_SIZE \
  --head_activation $HEAD_ACTIVATION \
  --head_dropout $HEAD_DROPOUT \
  --model_max_length 257