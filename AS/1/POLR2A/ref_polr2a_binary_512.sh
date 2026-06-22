#!/bin/bash
set -euo pipefail

################################################################################
# entexBERT-2: ONE reproducible allele-specific BINARY classification experiment
# (no auxiliary targets) + the FULL analysis pipeline.
#
# Pipeline:
#   1. Build dataset           (run_experiment.py, config-driven)
#   2. Cross-split leakage check
#   3. Train                   (finetune_entexbert2.py; writes run_config.json)
#   4. Analyze                 (analyze.py: metrics, PCA, representative examples)
#   5. PCA scatter             (plot_pca.py)
#   6. Attention profiles      (plot_attention_profiles.py)   [needs attention model]
#   7. Attribution profiles    (plot_attribution_profiles.py: ISM core + IG companion)
#
# Everything is driven by the CONFIG block below (single source of truth).
# Set JITTER_MAX_BP=0 for a centered run; >0 to jitter the SNV within the window.
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
SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

########################################
# CONFIG -- experiment identity
########################################
DONOR="ENC-001"
DONOR_NUMBER="1"
TISSUE_ARG="ALL"                 # ALL = pool tissues (locus-grouped); or a tissue name
TISSUE_LABEL="all_tissues"
ASSAY="TF-ChIP-seq_POLR2A"
ASSAY_NICKNAME="POLR2A"

########################################
# CONFIG -- dataset / window
########################################
MIN_TOTAL_READS=10
LEFT_BP=256
RIGHT_BP=256
JITTER_MAX_BP=0                 # 0 = centered; >0 = uniform SNV jitter, must be <= min(LEFT_BP,RIGHT_BP)
INPUT_MODE="ref_single"
BALANCE_STRATEGY="per_tissue_binary"
SPLIT_TRAIN=0.8; SPLIT_DEV=0.1; SPLIT_TEST=0.1
SEED=42
CHUNKSIZE=100000

# Positive = AS in >=1 tissue, Negative = AS in no tissue (primary_label type as_class).
# NO auxiliary targets: aux_labels is empty and no aux flags are passed to the trainer.

########################################
# CONFIG -- model / head / training
########################################
MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"
POOLING_MODE="cls"
HEAD_NUM_LAYERS=1                # 1 = linear head; 2 = the MLP head (set HEAD_HIDDEN_SIZE>0 then)
HEAD_HIDDEN_SIZE=-1              # -1 with num_layers=1; e.g. 256 with num_layers=2
HEAD_ACTIVATION="gelu"
HEAD_DROPOUT=0.1

MODEL_MAX_LENGTH=512             # must be >= the window's token count; also recorded in run_config.json
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_STEPS=50
LOGGING_STEPS=50; EVAL_STEPS=100; SAVE_STEPS=100

########################################
# CONFIG -- analysis / interpretability
########################################
RUN_TRAINING="true"              # set "false" to re-run only the analysis on an existing checkpoint
RUN_ANALYSIS="true"
RUN_ATTENTION="true"             # auto-skipped if the attention-extraction model is absent
RUN_ATTRIBUTION="true"

N_PER_CATEGORY=100
ISM_WINDOW_BP=100
ATTR_SCORE_MODE="margin"         # margin | prob_pos | pos_logit  (prob_pos = the confidence-confound control)
# Attention needs a DNABERT-2 copy whose layers expose attentions (see your attention model setup).
INFERENCE_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"

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
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}_layers${HEAD_NUM_LAYERS}"
    RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_AS_classification_jitter${JITTER_MAX_BP}_layers${HEAD_NUM_LAYERS}"
else
    RUN_SUBDIR="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}"
    RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_AS_classification_layers${HEAD_NUM_LAYERS}"
fi

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${RUN_SUBDIR}"
DATA_DIR="${EXPERIMENT_DIR}/input"
OUTPUT_DIR="${EXPERIMENT_DIR}/output"
ANALYSIS_DIR="${EXPERIMENT_DIR}/analysis"
FIGURE_DIR="${EXPERIMENT_DIR}/figures"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$ANALYSIS_DIR" "$FIGURE_DIR" "$LOG_DIR"

REP_CSV="${ANALYSIS_DIR}/representative_examples_all.csv"
CONFIG_FILE="${EXPERIMENT_DIR}/dataset_config.yaml"
LOGFILE="${LOG_DIR}/${RUN_NAME}.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Universally-covered plot window (jitter shrinks the region every example spans).
PLOT_WINDOW_BP=$(( LEFT_BP - JITTER_MAX_BP ))

echo "========================================"
echo "Experiment: $RUN_NAME"
echo "Date: $(date)   Host: $(hostname)"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Jitter: +/- ${JITTER_MAX_BP} bp   Head layers: ${HEAD_NUM_LAYERS}"
echo "========================================"

########################################
# Sanity checks
########################################
for path in "$AS_TSV" "$REF_FASTA" "$CHROM_SIZES"; do
    [[ -f "$path" ]] || { echo "ERROR: missing input file: $path"; exit 1; }
done
for s in run_experiment.py finetune_entexbert2.py analyze.py model_io.py; do
    [[ -f "$PROJECT_DIR/src/entexbert2/$s" ]] || { echo "ERROR: missing $s"; exit 1; }
done
python - <<'PY'
import entexbert2; print("entexbert2 from:", entexbert2.__file__)
PY

if [[ "$RUN_TRAINING" == "true" ]]; then
    ####################################
    # 1. Build dataset (config-driven, no aux targets)
    ####################################
    echo; echo "===== 1. Building dataset ====="
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
  snv_offset_mode: $([[ "$JITTER_MAX_BP" -gt 0 ]] && echo uniform || echo fixed)
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
    echo "Wrote $CONFIG_FILE"
    python -m entexbert2.run_experiment "$CONFIG_FILE"
    ls -lh "$DATA_DIR"

    ####################################
    # 2. Cross-split leakage check
    ####################################
    echo; echo "===== 2. Leakage check ====="
    python - <<PY
import os, sys, pandas as pd
data_dir = "$DATA_DIR"
dfs = {s: pd.read_csv(os.path.join(data_dir, f"{s}.csv")) for s in ["train","dev","test"]}
seq = {s: set(df["sequence"].astype(str)) for s, df in dfs.items()}
ov = {"train_dev": seq["train"]&seq["dev"], "train_test": seq["train"]&seq["test"], "dev_test": seq["dev"]&seq["test"]}
for s in seq: print(f"  {s}: {len(seq[s])} unique / {len(dfs[s])} rows")
leak = False
for k, v in ov.items():
    print(f"{k} cross-split dups: {len(v)}")
    leak = leak or len(v) > 0
if leak:
    print("ERROR: cross-split sequence leakage."); sys.exit(1)
print("No cross-split sequence leakage.")
PY

    ####################################
    # 3. Train (binary classification, no aux heads)
    ####################################
    echo; echo "===== 3. Training ====="
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
        --model_max_length "$MODEL_MAX_LENGTH" \
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
fi

########################################
# run_config.json gate (needed by all analysis steps)
########################################
if [[ ! -f "$OUTPUT_DIR/run_config.json" ]]; then
    echo "ERROR: $OUTPUT_DIR/run_config.json not found."
    echo "       The analysis steps need it. It is written by the trainer at save time;"
    echo "       if this checkpoint predates that, hand-write run_config.json or retrain."
    exit 1
fi

if [[ "$RUN_ANALYSIS" == "true" ]]; then
    ####################################
    # 4. Analyze: metrics + PCA + representative examples
    ####################################
    echo; echo "===== 4. analyze.py ====="
    python -m entexbert2.analyze \
        --checkpoint_dir "$OUTPUT_DIR" \
        --data_csv "$DATA_DIR/test.csv" \
        --output_dir "$ANALYSIS_DIR" \
        --n_per_category "$N_PER_CATEGORY" \
        --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --device cuda
    [[ -f "$REP_CSV" ]] || { echo "ERROR: $REP_CSV not produced."; exit 1; }
    echo "Metrics:"; cat "$ANALYSIS_DIR/metrics.json" 2>/dev/null || true

    ####################################
    # 5. PCA scatter (category + true label; evr auto-loaded from sidecar)
    ####################################
    echo; echo "===== 5. plot_pca.py ====="
    python "$SCRIPTS_DIR/plot_pca.py" \
        --pca_csv "$ANALYSIS_DIR/pca.csv" \
        --output "$FIGURE_DIR/pca_scatter.png"

    ####################################
    # 6. Attention profiles (only if the attention-extraction model is present)
    ####################################
    if [[ "$RUN_ATTENTION" == "true" && -d "$INFERENCE_MODEL" ]]; then
        echo; echo "===== 6. plot_attention_profiles.py ====="
        rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention
        python "$SCRIPTS_DIR/plot_attention_profiles.py" \
            --checkpoint_dir "$OUTPUT_DIR" \
            --examples_csv "$REP_CSV" \
            --output_dir "$ANALYSIS_DIR/attention_profiles" \
            --model_name_or_path "$INFERENCE_MODEL" \
            --input_mode "$INPUT_MODE" \
            --source_token cls --layers 11 --heads all --remove_alibi \
            --n_per_category "$N_PER_CATEGORY" \
            --left_bp "$LEFT_BP" \
            --plot_window_bp "$PLOT_WINDOW_BP" \
            --profile_correction none --plot_values all \
            --model_max_length "$MODEL_MAX_LENGTH"
    else
        echo; echo "===== 6. attention SKIPPED (RUN_ATTENTION=$RUN_ATTENTION; model present: $([[ -d "$INFERENCE_MODEL" ]] && echo yes || echo no)) ====="
    fi

    ####################################
    # 7. Attribution profiles: ISM (faithful core) + IG (companion)
    ####################################
    if [[ "$RUN_ATTRIBUTION" == "true" ]]; then
        echo; echo "===== 7. plot_attribution_profiles.py (ISM + IG) ====="
        python "$SCRIPTS_DIR/plot_attribution_profiles.py" \
            --checkpoint_dir "$OUTPUT_DIR" \
            --examples_csv "$REP_CSV" \
            --output_dir "$ANALYSIS_DIR/attribution" \
            --input_mode "$INPUT_MODE" \
            --method both --ism_mode both \
            --ism_window_bp "$ISM_WINDOW_BP" \
            --plot_window_bp "$ISM_WINDOW_BP" \
            --score_mode "$ATTR_SCORE_MODE" \
            --ig_steps 32 --ig_baseline mask \
            --n_per_category "$N_PER_CATEGORY" \
            --left_bp "$LEFT_BP" \
            --model_max_length "$MODEL_MAX_LENGTH" \
            --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
            --device cuda
    fi
fi

########################################
# Summary
########################################
echo
echo "========================================"
echo "Done: $RUN_NAME"
echo "  data:        $DATA_DIR"
echo "  checkpoint:  $OUTPUT_DIR"
echo "  analysis:    $ANALYSIS_DIR"
echo "  figures:     $FIGURE_DIR  (pca_scatter.png)"
echo "  log:         $LOGFILE"
echo "========================================"