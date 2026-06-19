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

LEFT_BP=256
RIGHT_BP=256

# Center-jitter: per example, the SNV is randomly offset +/- JITTER_MAX_BP from the
# window center (uniform). Must be <= min(LEFT_BP, RIGHT_BP) so the SNV stays in-window.
# Set to 0 to recover a centered (non-jittered) run.
JITTER_MAX_BP=50

INPUT_MODE="ref_single"

# Normal all-tissue AS classification:
# balance positives/negatives separately within each tissue
BALANCE_STRATEGY="per_tissue_binary"

SPLIT_TRAIN=0.8
SPLIT_DEV=0.1
SPLIT_TEST=0.1

SEED=42
CHUNKSIZE=100000

RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_AS_classification_jitter${JITTER_MAX_BP}"

########################################
# Input files
########################################

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

########################################
# Output locations
########################################

HEAD_NUM_LAYERS=2

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}_MLP${HEAD_NUM_LAYERS}"

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
echo "Center jitter: +/- ${JITTER_MAX_BP} bp"
echo "========================================"

########################################
# Training parameters
########################################

MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"

POOLING_MODE="cls"

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

# Run the attention-profile step at the end? Leave "false" to finish at analysis;
# flip to "true" once the attention plotter is ready, then re-run (it picks up
# the already-trained checkpoint and the representative examples).
RUN_ATTENTION="false"

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

if [[ ! -f "$PROJECT_DIR/src/entexbert2/run_experiment.py" ]]; then
    echo "ERROR: Missing dataset runner:"
    echo "$PROJECT_DIR/src/entexbert2/run_experiment.py"
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
echo "Building AS classification dataset (center jitter)..."
echo "====================================="

# The dataset step is config-driven. We generate the config from the variables above
# (single source of truth). snv_offset_mode=uniform with jitter_max_bp jitters the SNV
# within +/- JITTER_MAX_BP of center per example; realized offsets are recorded in the
# *.meta.csv sidecars for the analysis/plotting stage. "ALL" tissue pools tissues;
# splitting is leakage-safe by locus (group: locus).
CONFIG_FILE="${EXPERIMENT_DIR}/dataset_config.yaml"

cat > "$CONFIG_FILE" <<EOF
experiment: ${RUN_NAME}
ref_fasta: ${REF_FASTA}
output_dir: ${DATA_DIR}

row_source:
  type: snv_tsv
  path: ${AS_TSV}
  assay: ${ASSAY}
  donor: ${DONOR}
  tissue: ${TISSUE_ARG}
  min_total_reads: ${MIN_TOTAL_READS}
  chunksize: ${CHUNKSIZE}

primary_label: {type: as_class}
aux_labels: []

sequence: {input_mode: ${INPUT_MODE}}

window:
  left_bp: ${LEFT_BP}
  right_bp: ${RIGHT_BP}
  chrom_sizes: ${CHROM_SIZES}
  snv_offset_mode: uniform
  jitter_max_bp: ${JITTER_MAX_BP}

balance: {strategy: ${BALANCE_STRATEGY}, label_col: imbalance_significance}

split:
  mode: train_dev_test
  ratio: [${SPLIT_TRAIN}, ${SPLIT_DEV}, ${SPLIT_TEST}]
  seed: ${SEED}
  group: locus
  skip_ambiguous: true

head: {num_labels: 2, head_num_layers: ${HEAD_NUM_LAYERS}, head_hidden_size: ${HEAD_HIDDEN_SIZE}}
EOF

echo "Wrote dataset config: $CONFIG_FILE"
python -m entexbert2.run_experiment "$CONFIG_FILE"

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