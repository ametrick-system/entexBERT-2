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

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

########################################
# Experiment identity
########################################

TARGET_NAME="CTCF"
DONOR="ENC-001"
DONOR_NUMBER="1"

TISSUE="sigmoid_colon"

ASSAY="TF-ChIP-seq_CTCF"
ASSAY_NICKNAME="CTCF"

########################################
# Dataset parameters
########################################

MIN_TOTAL_READS=10

LEFT_BP=256
RIGHT_BP=256

INPUT_MODE="hap_pair"

BALANCE_STRATEGY="global_binary"

SIGNAL_MODE="max"
SIGNAL_REGION="snv_radius"
SIGNAL_RADIUS_BP=20
TARGET_TRANSFORM="log1p"

AUX_NAME="ctcf_log1p_max_fc_snv_pm${SIGNAL_RADIUS_BP}"
LAMBDA_AUX=0.3

SPLIT_TRAIN=0.8
SPLIT_DEV=0.1
SPLIT_TEST=0.1

SEED=42
CHUNKSIZE=100000

RUN_NAME="${INPUT_MODE}_${TARGET_NAME}_${DONOR}_${TISSUE}_AS_classification_LUPI_${AUX_NAME}_lambda${LAMBDA_AUX}"

########################################
# Input files
########################################

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

BIGWIG="$HOME/entex_data/${DONOR}/${ASSAY_NICKNAME}_${TISSUE}_fold_change_over_control.bigWig"

########################################
# Output locations
########################################

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE}/${INPUT_MODE}_classification_lupi_bigwig"

DATA_DIR="${EXPERIMENT_DIR}/input"
OUTPUT_DIR="${EXPERIMENT_DIR}/output"
FIGURE_DIR="${EXPERIMENT_DIR}/figures"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$FIGURE_DIR" "$LOG_DIR"

LOGFILE="${LOG_DIR}/${RUN_NAME}.log"

exec > >(tee -a "$LOGFILE") 2>&1

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
LEARNING_RATE=5e-5
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

for path in "$AS_TSV" "$REF_FASTA" "$CHROM_SIZES" "$BIGWIG"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Missing required file: $path"
        exit 1
    fi
done

if [[ ! -f "$PROJECT_DIR/src/entexbert2/scripts/make_as_classification_lupi_dataset.py" ]]; then
    echo "ERROR: Missing dataset script:"
    echo "$PROJECT_DIR/src/entexbert2/scripts/make_as_classification_lupi_dataset.py"
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

########################################
# Step 1: Build LUPI dataset
########################################

echo
echo "========================================"
echo "Step 1: Building hap_pair AS classification + BigWig LUPI dataset"
echo "========================================"

python src/entexbert2/scripts/make_as_classification_lupi_dataset.py \
    --as_tsv "$AS_TSV" \
    --ref_fasta "$REF_FASTA" \
    --bigwig "$BIGWIG" \
    --output_dir "$DATA_DIR" \
    --assay "$ASSAY" \
    --donor "$DONOR" \
    --tissue "$TISSUE" \
    --min_total_reads "$MIN_TOTAL_READS" \
    --left_bp "$LEFT_BP" \
    --right_bp "$RIGHT_BP" \
    --chrom_sizes "$CHROM_SIZES" \
    --input_mode "$INPUT_MODE" \
    --balance_strategy "$BALANCE_STRATEGY" \
    --aux_name "$AUX_NAME" \
    --signal_mode "$SIGNAL_MODE" \
    --signal_region "$SIGNAL_REGION" \
    --signal_radius_bp "$SIGNAL_RADIUS_BP" \
    --target_transform "$TARGET_TRANSFORM" \
    --split_ratio "$SPLIT_TRAIN" "$SPLIT_DEV" "$SPLIT_TEST" \
    --seed "$SEED" \
    --chunksize "$CHUNKSIZE"

echo
echo "Dataset files:"
ls -lh "$DATA_DIR"

########################################
# Step 1.5: Check hap-pair leakage
########################################

echo
echo "========================================"
echo "Step 1.5: Checking duplicate hap-pair leakage across splits"
echo "========================================"

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

dfs = {split: pd.read_csv(path) for split, path in paths.items()}

for split, df in dfs.items():
    required = {"sequence1", "sequence2", "label", "$AUX_NAME"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split}.csv missing columns {missing}. Got {list(df.columns)}")

def pair_set(df):
    return set(zip(df["sequence1"].astype(str), df["sequence2"].astype(str)))

pair_sets = {split: pair_set(df) for split, df in dfs.items()}

overlaps = {
    "train_dev": pair_sets["train"] & pair_sets["dev"],
    "train_test": pair_sets["train"] & pair_sets["test"],
    "dev_test": pair_sets["dev"] & pair_sets["test"],
}

print("Rows and unique hap-pairs per split:")
for split, pairs in pair_sets.items():
    print(f"  {split}: {len(pairs)} unique pairs / {len(dfs[split])} rows")

has_leakage = False
for name, overlap in overlaps.items():
    print(f"{name} duplicate hap-pairs across splits: {len(overlap)}")
    if len(overlap) > 0:
        has_leakage = True

if has_leakage:
    print("\\nERROR: Duplicate hap-pair inputs found across train/dev/test splits.")
    sys.exit(1)

print("No exact duplicate hap-pair inputs found across splits.")

print("\\nLabel counts by split:")
for split, df in dfs.items():
    print(f"\\n{split}:")
    print(df["label"].value_counts().sort_index())

print("\\nAux label summary by split:")
for split, df in dfs.items():
    print(f"\\n{split}:")
    print(df["$AUX_NAME"].describe())
PY

########################################
# Step 2: Plot dataset distributions
########################################

echo
echo "========================================"
echo "Step 2: Plotting dataset distributions"
echo "========================================"

if [[ -f "$PROJECT_DIR/src/entexbert2/scripts/plot_dataset_distribution.py" ]]; then
    python src/entexbert2/scripts/plot_dataset_distribution.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$FIGURE_DIR" \
        --bins 50
else
    echo "plot_dataset_distribution.py not found; skipping plots."
fi

echo
echo "Figure files:"
ls -lh "$FIGURE_DIR" || true

########################################
# Step 3: Fine-tune entexBERT-2 with LUPI aux regression
########################################

echo
echo "========================================"
echo "Step 3: Training entexBERT-2 hap_pair AS classification model with BigWig LUPI"
echo "========================================"

python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --data_path "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --task classification \
    --main_num_labels 2 \
    --num_aux_tasks 1 \
    --aux_task_names "$AUX_NAME" \
    --aux_task_types regression \
    --aux_num_labels 1 \
    --lambda_aux "$LAMBDA_AUX" \
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