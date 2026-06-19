#!/bin/bash
set -euo pipefail

########################################
# OFF-CENTER CROSS-EVALUATION
#
# Decisive control for "is the central attention SNV-locked or window-center-locked?"
# It feeds ONE trained checkpoint (MODEL_*) the off-center inputs of ANOTHER run
# (DATA_*), and realigns each example to its own SNV via anchor_offset_seq1.
#
#   - SNV-locked attention  -> sharp peak at position_relative_to_snv = 0
#   - center-locked (positional) attention -> the window center maps to a SPREAD of
#     relative positions (about +/- the data's jitter), so it shows up as a BROAD blob,
#     NOT a flat line. "Flat near 0" is the only reading that means "no SNV signal".
#
# This isolates INPUT ALIGNMENT from MODEL IDENTITY, which neither the centered nor the
# jitter plot can do alone (those are two different models).
########################################

module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

########################################
# Shared identity (model and data should be the SAME assay/donor/tissue/loci)
########################################

DONOR_NUMBER="1"
ASSAY_NICKNAME="CTCF"
TISSUE_LABEL="all_tissues"
INPUT_MODE="ref_single"

########################################
# MODEL side: the trained checkpoint to PROBE (e.g. the centered run)
########################################

MODEL_LEFT_BP=256
MODEL_RIGHT_BP=256
MODEL_JITTER=0          # 0 for a centered checkpoint

########################################
# DATA side: the off-center inputs to FEED it (e.g. the jitter run's test set)
########################################

DATA_LEFT_BP=256
DATA_RIGHT_BP=256
DATA_JITTER=50          # the inputs whose SNVs are off-center

########################################
# Analysis / attention parameters
########################################

NUM_EX=100
MODEL_MAX_LENGTH=512
DEVICE="cuda"
INFERENCE_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"
SOURCE_TOKEN="cls"
LAYERS="11"
HEADS="all"

########################################
# Reconstruct experiment directories (same convention as the training scripts)
########################################

subdir () {  # args: left right jitter
    if [[ "$3" -gt 0 ]]; then
        echo "${INPUT_MODE}_classification_${1}_${2}_jitter${3}"
    else
        echo "${INPUT_MODE}_classification_${1}_${2}"
    fi
}

BASE="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}"
MODEL_EXPERIMENT_DIR="$BASE/$(subdir "$MODEL_LEFT_BP" "$MODEL_RIGHT_BP" "$MODEL_JITTER")"
DATA_EXPERIMENT_DIR="$BASE/$(subdir "$DATA_LEFT_BP"  "$DATA_RIGHT_BP"  "$DATA_JITTER")"

MODEL_CHECKPOINT_DIR="$MODEL_EXPERIMENT_DIR/output"
DATA_TEST_CSV="$DATA_EXPERIMENT_DIR/input/test.csv"

DATA_TAG="$(subdir "$DATA_LEFT_BP" "$DATA_RIGHT_BP" "$DATA_JITTER")"
CROSS_DIR="$MODEL_EXPERIMENT_DIR/analysis_offcenter/on_${DATA_TAG}"
REP_CSV="$CROSS_DIR/representative_examples_all.csv"

# Universally-covered plot window for the DATA's jitter (avoids edge bias).
PLOT_WINDOW_BP=$(( DATA_LEFT_BP - DATA_JITTER ))

echo "========================================"
echo "OFF-CENTER CROSS-EVAL"
echo "  MODEL (probed):  $MODEL_EXPERIMENT_DIR"
echo "  DATA  (fed in):  $DATA_EXPERIMENT_DIR"
echo "  Output:          $CROSS_DIR"
echo "  Date: $(date)"
echo "========================================"

########################################
# Guardrails
########################################

if [[ ! -f "$MODEL_CHECKPOINT_DIR/run_config.json" ]]; then
    echo "ERROR: $MODEL_CHECKPOINT_DIR/run_config.json not found (need it to load the model)."
    exit 1
fi
if [[ ! -f "$DATA_TEST_CSV" ]]; then
    echo "ERROR: $DATA_TEST_CSV not found."
    exit 1
fi
if [[ ! -f "${DATA_TEST_CSV%.csv}.meta.csv" ]]; then
    echo "ERROR: ${DATA_TEST_CSV%.csv}.meta.csv not found."
    echo "       Without per-example anchor_offset_seq1 the realignment falls back to a constant"
    echo "       --left_bp, which DEFEATS the purpose of this off-center test. Aborting."
    exit 1
fi

if [[ "$MODEL_LEFT_BP" -ne "$DATA_LEFT_BP" || "$MODEL_RIGHT_BP" -ne "$DATA_RIGHT_BP" ]]; then
    echo
    echo "*** WARNING: model window (${MODEL_LEFT_BP}/${MODEL_RIGHT_BP}) != data window (${DATA_LEFT_BP}/${DATA_RIGHT_BP}). ***"
    echo "    This test is only clean when the two windows MATCH (so only the SNV centering differs)."
    echo "    With mismatched windows you are also changing sequence length and genomic content,"
    echo "    which confounds the result. Cleanest fix: generate a matched-window off-center test set, e.g."
    echo "      python -m entexbert2.run_experiment --config <a ${MODEL_LEFT_BP}/${MODEL_RIGHT_BP} jitter config>"
    echo "    and point DATA_* at that. Continuing, but interpret with this caveat."
    echo
fi

if [[ "$DATA_JITTER" -le 0 ]]; then
    echo "*** WARNING: DATA_JITTER=0 means the inputs are centered, so this is NOT an off-center test. ***"
fi

mkdir -p "$CROSS_DIR"

########################################
# Step 1: analyze the MODEL checkpoint on the DATA (off-center) inputs
########################################

echo
echo "Running analyze.py (MODEL checkpoint on DATA inputs)..."
echo "NOTE: the centered model has not seen off-center SNVs, so its predictions here are"
echo "      out-of-distribution; treat the TP/FP/TN/FN labels as approximate and lean on the"
echo "      pooled spatial profile for the locking question."

python -m entexbert2.analyze \
    --checkpoint_dir "$MODEL_CHECKPOINT_DIR" \
    --data_csv "$DATA_TEST_CSV" \
    --output_dir "$CROSS_DIR" \
    --n_per_category "$NUM_EX" \
    --batch_size 16 \
    --device "$DEVICE"

if [[ ! -f "$REP_CSV" ]]; then
    echo "ERROR: $REP_CSV not produced."
    exit 1
fi

########################################
# Step 2: attention profile, realigned to each example's SNV
########################################

ATTENTION_OUT="$CROSS_DIR/attention_profiles/${SOURCE_TOKEN}_layers(${LAYERS})_heads(${HEADS})_alibi_removed"
rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

echo
echo "Plotting attention (realigned to per-example SNV)..."
python "$SCRIPTS_DIR/plot_attention_profiles.py" \
    --checkpoint_dir "$MODEL_CHECKPOINT_DIR" \
    --examples_csv "$REP_CSV" \
    --output_dir "$ATTENTION_OUT" \
    --model_name_or_path "$INFERENCE_MODEL" \
    --input_mode "$INPUT_MODE" \
    --source_token "$SOURCE_TOKEN" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --remove_alibi \
    --n_per_category "$NUM_EX" \
    --left_bp "$DATA_LEFT_BP" \
    --plot_window_bp "$PLOT_WINDOW_BP" \
    --profile_correction none \
    --plot_values all \
    --model_max_length "$MODEL_MAX_LENGTH"

echo
echo "========================================"
echo "Done."
echo "  Decision rule: sharp peak at 0 => SNV-locked; broad ~+/-${DATA_JITTER}bp blob => center/positional."
echo "  Output: $ATTENTION_OUT"
echo "========================================"
