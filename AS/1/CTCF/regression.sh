#!/bin/bash
set -euo pipefail

################################################################################
# entexBERT-2 CONTROL: can the backbone + head learn ANY sequence-window -> scalar map?
#
# Target : log_total_count = log(ref_count + alt_count + 1)  (total binding DEPTH at the SNV)
# Input  : ref_single  (SINGLE sequence, NO twin)
#          -> total depth is a per-locus magnitude, not an allelic contrast; the twin would
#             predict head(alt)-head(ref) ~= 0 for it. So this control is deliberately single-seq.
# Arch   : center_mean pooling + 2-layer MLP head (matches the dead signed-effect run, minus twin)
#
# PURPOSE: diagnostic, not a deliverable. This is an EASY target (strong CTCF motif -> high signal).
#   - If eval_spearman / eval_r2 CLIMB here  => the model CAN learn from these windows; the dead
#     signed-effect result is a real localization ("resolves binding, not the 1-base allelic diff").
#   - If this is ALSO flat (eval_mse pinned, spearman ~0) => systemic failure (LR / frozen backbone
#     / label scaling / optimizer), and the signed-effect result was never a fair test.
#
# Run as a SHORT PROBE (max_steps 1500, eval every 300). The answer shows up fast; do not wait
# for full epochs. Analysis is OFF -- the answer is in the training eval log.
################################################################################

########################################
# Environment
########################################
module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

########################################
# CONFIG -- experiment identity (CTCF / individual 1)
########################################
DONOR="ENC-001"
DONOR_NUMBER="1"
TISSUE_ARG="ALL"
TISSUE_LABEL="all_tissues"
ASSAY="TF-ChIP-seq_CTCF"
ASSAY_NICKNAME="CTCF"

########################################
# CONFIG -- control target / input
########################################
REG_TARGET="log_total_count"      # total binding depth (easy, single-sequence target)
TARGET_TAG="logtotal"
INPUT_MODE="ref_single"           # SINGLE sequence -- no twin for a magnitude target

########################################
# CONFIG -- dataset / window (match the signed-effect run)
########################################
MIN_TOTAL_READS=20
LEFT_BP=256
RIGHT_BP=256
JITTER_MAX_BP=50
SPLIT_TRAIN=0.8; SPLIT_DEV=0.1; SPLIT_TEST=0.1
SEED=42
CHUNKSIZE=100000

########################################
# CONFIG -- model / head / training  (match the dead run: center pooling + 2-layer MLP)
########################################
MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"
POOLING_MODE="center_mean"        # center pooling; width MUST be odd
CENTER_POOL_WIDTH=5
HEAD_NUM_LAYERS=2                 # 2-layer MLP
HEAD_HIDDEN_SIZE=256
HEAD_ACTIVATION="gelu"
HEAD_DROPOUT=0.1

MODEL_MAX_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
LOGGING_STEPS=50
SELECT_METRIC="eval_spearman"

# PROBE controls -- short by design
MAX_STEPS=1500
EVAL_STEPS=300
SAVE_STEPS=1500

########################################
# Toggles
########################################
RUN_TRAINING="true"
RUN_ANALYSIS="false"              # answer is in the training eval log; no need to analyze a probe

########################################
# Input files
########################################
AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

########################################
# Derived paths
########################################
if [[ "$JITTER_MAX_BP" -gt 0 ]]; then
    WIN_TAG="${INPUT_MODE}_control_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}"
    OFFSET_MODE="uniform"
else
    WIN_TAG="${INPUT_MODE}_control_${LEFT_BP}_${RIGHT_BP}"
    OFFSET_MODE="fixed"
fi
RUN_SUBDIR="${WIN_TAG}_${TARGET_TAG}"
RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_CONTROL_${TARGET_TAG}"

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${RUN_SUBDIR}"
DATA_DIR="${EXPERIMENT_DIR}/input"
OUTPUT_DIR="${EXPERIMENT_DIR}/output"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

CONFIG_FILE="${EXPERIMENT_DIR}/dataset_config.yaml"
LOGFILE="${LOG_DIR}/${RUN_NAME}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "CONTROL: $RUN_NAME"
echo "Date: $(date)   Host: $(hostname)"
echo "Target=$REG_TARGET  input=$INPUT_MODE (single, no twin)  pooling=$POOLING_MODE/$CENTER_POOL_WIDTH"
echo "Probe: max_steps=$MAX_STEPS eval_steps=$EVAL_STEPS. Watch eval_spearman / eval_r2."
echo "========================================"

for path in "$AS_TSV" "$REF_FASTA" "$CHROM_SIZES"; do
    [[ -f "$path" ]] || { echo "ERROR: missing input file: $path"; exit 1; }
done

if [[ "$RUN_TRAINING" == "true" ]]; then
    echo; echo "===== 1. Building dataset (ref_single -> log_total_count) ====="
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

primary_label: {type: as_regression, target: ${REG_TARGET}}
aux_labels: []

sequence: {input_mode: ${INPUT_MODE}}

window:
  left_bp: ${LEFT_BP}
  right_bp: ${RIGHT_BP}
  chrom_sizes: ${CHROM_SIZES}
  snv_offset_mode: ${OFFSET_MODE}
  jitter_max_bp: ${JITTER_MAX_BP}

balance: {strategy: none}

split:
  mode: train_dev_test
  ratio: [${SPLIT_TRAIN}, ${SPLIT_DEV}, ${SPLIT_TEST}]
  seed: ${SEED}
  group: locus
  skip_ambiguous: true

head: {num_labels: 1, head_num_layers: ${HEAD_NUM_LAYERS}, head_hidden_size: ${HEAD_HIDDEN_SIZE}}
EOF
    echo "Wrote $CONFIG_FILE"
    python -m entexbert2.run_experiment "$CONFIG_FILE"

    echo; echo "===== 2. Target summary per split ====="
    python - <<PY
import os, pandas as pd
d = "$DATA_DIR"
for s in ["train","dev","test"]:
    df = pd.read_csv(os.path.join(d, f"{s}.csv"))
    y = df["label"].astype(float)
    print(f"  {s:5s}: n={len(df):7d}  log_total mean={y.mean():.3f} std={y.std():.3f} "
          f"min={y.min():.2f} max={y.max():.2f}")
PY

    echo; echo "===== 3. Train (single-seq control; probe ${MAX_STEPS} steps) ====="
    # NOTE: --evaluation_strategy may be --eval_strategy on newer transformers.
    python -m entexbert2.finetune_entexbert2 \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --data_path "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --task regression \
        --main_num_labels 1 \
        --pooling_mode "$POOLING_MODE" \
        --center_pool_width "$CENTER_POOL_WIDTH" \
        --head_num_layers "$HEAD_NUM_LAYERS" \
        --head_hidden_size "$HEAD_HIDDEN_SIZE" \
        --head_activation "$HEAD_ACTIVATION" \
        --head_dropout "$HEAD_DROPOUT" \
        --model_max_length "$MODEL_MAX_LENGTH" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --max_steps "$MAX_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --weight_decay "$WEIGHT_DECAY" \
        --warmup_ratio "$WARMUP_RATIO" \
        --logging_steps "$LOGGING_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --evaluation_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end True \
        --metric_for_best_model "$SELECT_METRIC" \
        --greater_is_better True \
        --save_total_limit 1 \
        --eval_and_save_results True \
        --save_model True \
        --fp16 True \
        --dataloader_pin_memory False \
        --seed "$SEED"
fi

echo
echo "========================================"
echo "CONTROL done: $RUN_NAME"
echo "  READ THE eval_spearman / eval_r2 TRAJECTORY ABOVE:"
echo "   - climbing (spearman > ~0.1, r2 > 0, eval_mse FALLING) => model CAN learn these windows;"
echo "       the dead signed-effect result is a real localization (binding yes, allelic-contrast no)."
echo "   - flat (eval_mse pinned, spearman ~0) => SYSTEMIC failure (LR/backbone/scaling); the"
echo "       signed-effect run was never a fair test -- fix the systemic issue first."
echo "  log: $LOGFILE"
echo "========================================"