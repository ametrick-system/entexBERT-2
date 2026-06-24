#!/bin/bash
set -euo pipefail

################################################################################
# CONFIDENCE-CONFOUND CONTROL: re-run SNV and k-mer ISM in PROBABILITY space.
#
# Your existing attribution run used --score_mode margin (unbounded logits), which
# conflates "FP relies on perturbable sequence" with "FP are just low-margin/boundary
# cases." This re-runs the same two ISM modes with --score_mode prob_pos (bounded
# softmax prob) into a SEPARATE dir so you can compare margin vs prob side-by-side.
#
# Read it as: if FP-fragility / TP-TN-flatness (and the SNV-vs-background gaps) PERSIST
# in both spaces -> real sequence-dependence asymmetry. If they collapse/flip -> it was
# largely a confidence artifact. Compare the *ordering and SNV-vs-background gap*, not
# the absolute magnitudes (units differ: [0,1] vs unbounded).
################################################################################

module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

########################################
# CONFIG -- point at the run you already analyzed (must match its directory)
########################################
DONOR_NUMBER="1"
ASSAY_NICKNAME="CTCF"
TISSUE_LABEL="all_tissues"
INPUT_MODE="ref_single"
LEFT_BP=256
RIGHT_BP=256
JITTER_MAX_BP=50
MODEL_MAX_LENGTH=512

# ISM params -- keep these IDENTICAL to your margin run so the only change is the score.
N_PER_CATEGORY=100
ISM_WINDOW_BP=100
N_CONTROLS=25
KMER_SIZES="6,10,15"             # drop k=3: too constrained for dinuc shuffle (near no-op)
KMER_REPLACEMENT="dinuc"
KMER_N_SHUFFLES=3
KMER_STRIDE=1
BATCH_SIZE=256                   # pure inference; big batch amortizes the CPU re-tokenization
DEVICE="cuda"

########################################
# Derived paths (same convention as the pipeline script)
########################################
if [[ "$JITTER_MAX_BP" -gt 0 ]]; then
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}"
else
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}"
fi
EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${RUN_SUBDIR}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/output"
ANALYSIS_DIR="${EXPERIMENT_DIR}/analysis"
REP_CSV="${ANALYSIS_DIR}/representative_examples_all.csv"
OUT_DIR="${ANALYSIS_DIR}/attribution_probpos"   # separate from the margin run

echo "========================================"
echo "prob_pos control on: $EXPERIMENT_DIR"
echo "Output: $OUT_DIR   (compare against $ANALYSIS_DIR/attribution)"
echo "Date: $(date)"
echo "========================================"

########################################
# Guardrails
########################################
[[ -f "$CHECKPOINT_DIR/run_config.json" ]] || { echo "ERROR: $CHECKPOINT_DIR/run_config.json missing."; exit 1; }
if [[ ! -f "$REP_CSV" ]]; then
    echo "ERROR: $REP_CSV not found. Run analyze.py first (it produces the representative examples)."
    exit 1
fi
mkdir -p "$OUT_DIR"

########################################
# 1. SNV ISM in probability space
########################################
echo; echo "===== prob_pos: SNV ISM ====="
python "$SCRIPTS_DIR/plot_attribution_profiles.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --examples_csv   "$REP_CSV" \
    --output_dir     "$OUT_DIR" \
    --input_mode "$INPUT_MODE" \
    --method ism --ism_mode snv \
    --score_mode prob_pos \
    --n_controls "$N_CONTROLS" \
    --ism_window_bp "$ISM_WINDOW_BP" \
    --n_per_category "$N_PER_CATEGORY" \
    --left_bp "$LEFT_BP" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

########################################
# 2. k-mer ISM in probability space
########################################
echo; echo "===== prob_pos: k-mer ISM ====="
python "$SCRIPTS_DIR/plot_attribution_profiles.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --examples_csv   "$REP_CSV" \
    --output_dir     "$OUT_DIR" \
    --input_mode "$INPUT_MODE" \
    --method ism --ism_mode kmer \
    --score_mode prob_pos \
    --kmer_sizes "$KMER_SIZES" \
    --kmer_replacement "$KMER_REPLACEMENT" \
    --kmer_n_shuffles "$KMER_N_SHUFFLES" \
    --kmer_stride "$KMER_STRIDE" \
    --ism_window_bp "$ISM_WINDOW_BP" \
    --n_per_category "$N_PER_CATEGORY" \
    --left_bp "$LEFT_BP" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo
echo "========================================"
echo "Done. prob_pos outputs in: $OUT_DIR"
echo "  SNV:   ism_snv_summary.csv / ism_snv_vs_background.png"
echo "  k-mer: ism_kmer_snv_vs_background.{csv,png} / ism_kmer_k*_profile.png"
echo "Compare against the margin run in: $ANALYSIS_DIR/attribution"
echo "Decision: pattern PERSISTS in both spaces -> real; collapses/flips -> confidence artifact."
echo "========================================"