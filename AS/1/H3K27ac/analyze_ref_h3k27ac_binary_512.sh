#!/bin/bash
set -euo pipefail

########################################
# Post-training analysis + attention for an ALREADY-TRAINED entexBERT-2 run.
# Runs analyze.py (metrics + PCA + representative selection) then the attention plotter.
# Set the identity/window variables below to match the run you want to analyze.
########################################

module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

########################################
# Which run to analyze (must match the trained experiment's directory)
########################################

DONOR_NUMBER="1"
ASSAY_NICKNAME="H3K27ac"
TISSUE_LABEL="all_tissues"
INPUT_MODE="ref_single"

LEFT_BP=256
RIGHT_BP=256
JITTER_MAX_BP=0

# Reconstruct the experiment directory exactly as the training script created it.
if [[ "$JITTER_MAX_BP" -gt 0 ]]; then
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}"
else
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}"
fi
EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${RUN_SUBDIR}"

CHECKPOINT_DIR="${EXPERIMENT_DIR}/output"
TEST_CSV="${EXPERIMENT_DIR}/input/test.csv"
ANALYSIS_DIR="${EXPERIMENT_DIR}/analysis"
REP_CSV="${ANALYSIS_DIR}/representative_examples_all.csv"

########################################
# Analysis / attention parameters
########################################

NUM_EX=100                       # examples per category (analyze) and per category (attention)
MODEL_MAX_LENGTH=512             # must match run_config.json's model_max_length from training
DEVICE="cuda"

# Attention-extraction model (DNABERT-2 with attentions exposed) + attention options.
INFERENCE_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"
SOURCE_TOKEN="cls"
LAYERS="11"
HEADS="all"

# With jitter, only positions within +/- (LEFT_BP - JITTER_MAX_BP) are covered by every
# example, so plotting wider than that averages over a shrinking subset (edge bias).
PLOT_WINDOW_BP=$(( LEFT_BP - JITTER_MAX_BP ))

########################################
# Sanity checks
########################################

echo "========================================"
echo "Analyzing run: $EXPERIMENT_DIR"
echo "Date: $(date)"
echo "========================================"

for path in "$CHECKPOINT_DIR" "$TEST_CSV"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Missing expected path: $path"
        exit 1
    fi
done

if [[ ! -f "$CHECKPOINT_DIR/run_config.json" ]]; then
    echo "ERROR: $CHECKPOINT_DIR/run_config.json not found."
    echo "       analyze.py needs it (written by the updated trainer). If this run was trained"
    echo "       with the old trainer, either retrain or hand-write run_config.json."
    exit 1
fi

mkdir -p "$ANALYSIS_DIR"

########################################
# Step 1: analyze (metrics + PCA + representative selection)
########################################

echo
echo "====================================="
echo "Running analyze.py..."
echo "====================================="

python -m entexbert2.analyze \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_csv "$TEST_CSV" \
    --output_dir "$ANALYSIS_DIR" \
    --n_per_category "$NUM_EX" \
    --batch_size 16 \
    --device "$DEVICE"

echo
echo "Analysis outputs:"
ls -lh "$ANALYSIS_DIR"

if [[ ! -f "$REP_CSV" ]]; then
    echo "ERROR: $REP_CSV not produced by analyze.py."
    exit 1
fi

echo
echo "====================================="
echo "Plotting PCA..."
echo "====================================="

python $SCRIPTS_DIR/plot_pca.py --pca_csv $EXPERIMENT_DIR/analysis/pca.csv

########################################
# Step 2: attention profiles
########################################

echo
echo "====================================="
echo "Plotting attention profiles..."
echo "====================================="

ATTENTION_OUT="${ANALYSIS_DIR}/attention_profiles/full_${SOURCE_TOKEN}_layers(${LAYERS})_heads(${HEADS})_alibi_removed_left(${LEFT_BP})right(${RIGHT_BP})_jitter${JITTER_MAX_BP}"

# Clear any stale cached custom module for the attention model so it reloads cleanly.
rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention

python "$SCRIPTS_DIR/plot_attention_profiles.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --examples_csv "$REP_CSV" \
    --output_dir "$ATTENTION_OUT" \
    --model_name_or_path "$INFERENCE_MODEL" \
    --input_mode "$INPUT_MODE" \
    --source_token "$SOURCE_TOKEN" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --remove_alibi \
    --n_per_category "$NUM_EX" \
    --left_bp "$LEFT_BP" \
    --plot_window_bp "$PLOT_WINDOW_BP" \
    --profile_correction none \
    --plot_values all \
    --model_max_length "$MODEL_MAX_LENGTH"

echo
echo "========================================"
echo "Done."
echo "Analysis dir:  $ANALYSIS_DIR"
echo "Attention dir: $ATTENTION_OUT"
echo "========================================"
