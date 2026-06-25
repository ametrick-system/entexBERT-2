#!/bin/bash
set -euo pipefail

################################################################################
# entexBERT-2: CTCF / individual-1 (ENC-001) allele-specific BINARY classification.
#
# Reflects this project's decided pipeline:
#   - dev/test ALWAYS at natural prevalence (never balanced) -> honest, suite-comparable metrics
#   - train handled per ARM (below): balanced-subsample, OR full data + class weighting
#   - selection metric = AUPRC (imbalance-aware; read it relative to the positive prevalence)
#
# Pipeline: build dataset -> leakage check -> prevalence readout -> train -> (analysis, gated).
# The final test set IS evaluated here (once) -- this is the experiment, not tuning.
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
# CONFIG -- experiment identity (CTCF / individual 1)
########################################
DONOR="ENC-001"
DONOR_NUMBER="1"
TISSUE_ARG="ALL"                 # ALL = pool tissues (locus-grouped)
TISSUE_LABEL="all_tissues"
ASSAY="TF-ChIP-seq_CTCF"
ASSAY_NICKNAME="CTCF"

########################################
# CONFIG -- TRAIN-BALANCING ARM  (dev/test are natural prevalence in ALL arms)
#   balanced_train   : subsample train to 50/50 (BALANCE applied to train split only)
#   full_weighted    : full natural-prevalence train + inverse-frequency class weighting
#   full_unweighted  : full train, no weighting (baseline; expected to skew toward negative)
########################################
ARM="balanced_train"
case "$ARM" in
  balanced_train)  BALANCE_STRATEGY="per_tissue_binary"; BALANCE_APPLY_TO="train"; CLASS_WEIGHTS="none";     ARM_TAG="baltrain" ;;
  full_weighted)   BALANCE_STRATEGY="none";              BALANCE_APPLY_TO="all";   CLASS_WEIGHTS="balanced"; ARM_TAG="fullw" ;;
  full_unweighted) BALANCE_STRATEGY="none";              BALANCE_APPLY_TO="all";   CLASS_WEIGHTS="none";     ARM_TAG="fullu" ;;
  *) echo "ERROR: unknown ARM '$ARM'"; exit 1 ;;
esac

########################################
# CONFIG -- dataset / window
########################################
MIN_TOTAL_READS=10
LEFT_BP=256
RIGHT_BP=256
JITTER_MAX_BP=50                 # 0 = centered; >0 = uniform SNV jitter (<= min(LEFT_BP,RIGHT_BP))
INPUT_MODE="ref_single"
SPLIT_TRAIN=0.8; SPLIT_DEV=0.1; SPLIT_TEST=0.1
SEED=42
CHUNKSIZE=100000

########################################
# CONFIG -- model / head / training  (set LR + epochs from your tune_hparams results)
########################################
MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"
POOLING_MODE="cls"
HEAD_NUM_LAYERS=1                # 1 = linear; 2 = MLP head (set HEAD_HIDDEN_SIZE=256). MUST match what you tuned.
HEAD_HIDDEN_SIZE=-1
HEAD_ACTIVATION="gelu"
HEAD_DROPOUT=0.1

MODEL_MAX_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
NUM_TRAIN_EPOCHS=4               # <-- from tuning
LEARNING_RATE=2e-5              # <-- from tuning
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
LOGGING_STEPS=50; EVAL_STEPS=200; SAVE_STEPS=200
SELECT_METRIC="eval_auprc"

########################################
# CONFIG -- analysis (dev-derived decision threshold; safe at natural prevalence)
#   analyze.py picks the TP/FP/TN/FN threshold from dev (max-F1 or Youden) instead of 0.5,
#   so the confusion categories -- and the per-category attention/ISM/IG plots that draw
#   from representative_examples_all.csv -- reflect a sensible operating point at ~4% positives.
########################################
RUN_TRAINING="false"             # checkpoint already trained; re-run analysis only
RUN_ANALYSIS="true"
RUN_ATTENTION="true"
RUN_ATTRIBUTION="true"
ANALYSIS_THRESHOLD="f1"          # f1 | youden | a float (e.g. 0.5); dev.csv supplies the operating point
N_PER_CATEGORY=100
ISM_WINDOW_BP=100
ATTR_SCORE_MODE="margin"
RUN_KMER_ABLATION="true"         # sliding k-bp ablation (redundancy / motif-width probe)
KMER_SIZES="6,10,15"             # k=3 dinuc-shuffle is a near no-op (too constrained); use mono/random for small k
KMER_REPLACEMENT="dinuc"         # dinuc | mono | random (dinuc preserves dinucleotide content)
KMER_N_SHUFFLES=3                # background shuffles per window; raise for a more stable k-mer background
KMER_STRIDE=1
INFERENCE_MODEL="$HOME/entexBERT-2/DNABERT-2-117M-attention"

########################################
# Input files
########################################
AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

########################################
# Derived paths (ARM in the path so the three arms don't overwrite each other)
########################################
if [[ "$JITTER_MAX_BP" -gt 0 ]]; then
    WIN_TAG="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}"
    OFFSET_MODE="uniform"
else
    WIN_TAG="${INPUT_MODE}_classification_${LEFT_BP}_${RIGHT_BP}"
    OFFSET_MODE="fixed"
fi
RUN_SUBDIR="${WIN_TAG}_${ARM_TAG}"
RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_AS_${ARM_TAG}"

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
PLOT_WINDOW_BP=$(( LEFT_BP - JITTER_MAX_BP ))

echo "========================================"
echo "Experiment: $RUN_NAME"
echo "Date: $(date)   Host: $(hostname)"
echo "ARM: $ARM  (balance=$BALANCE_STRATEGY apply_to=$BALANCE_APPLY_TO class_weights=$CLASS_WEIGHTS)"
echo "Dev/test: NATURAL prevalence.  Select: $SELECT_METRIC.  Analysis threshold: $ANALYSIS_THRESHOLD (dev-derived)."
echo "Dir: $EXPERIMENT_DIR"
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

if [[ "$RUN_TRAINING" == "true" ]]; then
    ####################################
    # 1. Build dataset
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
  snv_offset_mode: ${OFFSET_MODE}
  jitter_max_bp: ${JITTER_MAX_BP}

balance: {strategy: ${BALANCE_STRATEGY}, label_col: imbalance_significance, apply_to: ${BALANCE_APPLY_TO}}

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

    ####################################
    # 2. Leakage check + prevalence readout (dev/test should be natural; train per ARM)
    ####################################
    echo; echo "===== 2. Leakage check + class balance per split ====="
    python - <<PY
import os, sys, pandas as pd
d = "$DATA_DIR"
dfs = {s: pd.read_csv(os.path.join(d, f"{s}.csv")) for s in ["train","dev","test"]}
seq = {s: set(df["sequence"].astype(str)) for s, df in dfs.items()}
leak = False
for k, (a,b) in {"train_dev":("train","dev"),"train_test":("train","test"),"dev_test":("dev","test")}.items():
    n = len(seq[a] & seq[b]); leak = leak or n > 0
    print(f"  {k} cross-split dups: {n}")
if leak:
    print("ERROR: cross-split sequence leakage."); sys.exit(1)
print("No cross-split sequence leakage.")
print("Class balance per split:")
for s, df in dfs.items():
    n = len(df); pos = int((df["label"] == 1).sum())
    print(f"  {s:5s}: {n:7d} rows  {pos:7d} pos  ({(pos/n if n else 0):.3%})")
PY

    ####################################
    # 3. Train (binary; AUPRC selection; class weighting per ARM; test evaluated once at end)
    ####################################
    echo; echo "===== 3. Training (ARM=$ARM) ====="
    # NOTE: if your transformers is recent, --evaluation_strategy may be --eval_strategy.
    python -m entexbert2.finetune_entexbert2 \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --data_path "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --task classification \
        --main_num_labels 2 \
        --class_weights "$CLASS_WEIGHTS" \
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
        --warmup_ratio "$WARMUP_RATIO" \
        --logging_steps "$LOGGING_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --evaluation_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end True \
        --metric_for_best_model "$SELECT_METRIC" \
        --greater_is_better True \
        --save_total_limit 3 \
        --eval_and_save_results True \
        --save_model True \
        --fp16 True \
        --dataloader_pin_memory False \
        --seed "$SEED"
fi

########################################
# run_config.json gate
########################################
if [[ ! -f "$OUTPUT_DIR/run_config.json" ]]; then
    echo "ERROR: $OUTPUT_DIR/run_config.json not found (needed for analysis)."; exit 1
fi

if [[ "$RUN_ANALYSIS" == "true" ]]; then
    echo; echo "===== 4. analyze.py (threshold=$ANALYSIS_THRESHOLD, dev-derived) ====="
    python -m entexbert2.analyze \
        --checkpoint_dir "$OUTPUT_DIR" \
        --data_csv "$DATA_DIR/test.csv" \
        --dev_csv "$DATA_DIR/dev.csv" \
        --threshold "$ANALYSIS_THRESHOLD" \
        --output_dir "$ANALYSIS_DIR" \
        --n_per_category "$N_PER_CATEGORY" \
        --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --device cuda
    [[ -f "$REP_CSV" ]] || { echo "ERROR: $REP_CSV not produced."; exit 1; }
    echo "Metrics (note decision_threshold / threshold_source):"
    cat "$ANALYSIS_DIR/metrics.json" 2>/dev/null || true

    echo; echo "===== 5. plot_pca.py ====="
    python "$SCRIPTS_DIR/plot_pca.py" --pca_csv "$ANALYSIS_DIR/pca.csv" --output "$FIGURE_DIR/pca_scatter.png"

    if [[ "$RUN_ATTENTION" == "true" && -d "$INFERENCE_MODEL" ]]; then
        echo; echo "===== 6. plot_attention_profiles.py ====="
        rm -rf ~/.cache/huggingface/modules/transformers_modules/DNABERT-2-117M-attention
        python "$SCRIPTS_DIR/plot_attention_profiles.py" \
            --checkpoint_dir "$OUTPUT_DIR" --examples_csv "$REP_CSV" \
            --output_dir "$ANALYSIS_DIR/attention_profiles" --model_name_or_path "$INFERENCE_MODEL" \
            --input_mode "$INPUT_MODE" --source_token cls --layers 11 --heads all --remove_alibi \
            --n_per_category "$N_PER_CATEGORY" --left_bp "$LEFT_BP" --plot_window_bp "$PLOT_WINDOW_BP" \
            --profile_correction none --plot_values all --model_max_length "$MODEL_MAX_LENGTH"
    else
        echo; echo "===== 6. attention SKIPPED ====="
    fi

    if [[ "$RUN_ATTRIBUTION" == "true" ]]; then
        echo; echo "===== 7. plot_attribution_profiles.py (ISM + IG) ====="
        python "$SCRIPTS_DIR/plot_attribution_profiles.py" \
            --checkpoint_dir "$OUTPUT_DIR" --examples_csv "$REP_CSV" \
            --output_dir "$ANALYSIS_DIR/attribution" --input_mode "$INPUT_MODE" \
            --method both --ism_mode both --ism_window_bp "$ISM_WINDOW_BP" --plot_window_bp "$ISM_WINDOW_BP" \
            --score_mode "$ATTR_SCORE_MODE" --ig_steps 32 --ig_baseline mask \
            --n_per_category "$N_PER_CATEGORY" --left_bp "$LEFT_BP" --model_max_length "$MODEL_MAX_LENGTH" \
            --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" --device cuda
    fi

    ####################################
    # 7b. k-mer ablation (redundancy / motif-width probe)
    #   Ablates a sliding k-bp window (replaced by shuffled background) and measures the
    #   score drop -- catches distributed/redundant signal that single-base ISM misses.
    #   Separate run (own dir) so it doesn't re-run IG; writes ism_kmer_k{K}_profile.png +
    #   ism_kmer_snv_vs_background.png. Report the SNV-vs-background GAP, not absolute height
    #   (bigger k mechanically -> bigger drop).
    ####################################
    if [[ "$RUN_KMER_ABLATION" == "true" ]]; then
        echo; echo "===== 7b. k-mer ablation (sizes=$KMER_SIZES, repl=$KMER_REPLACEMENT) ====="
        python "$SCRIPTS_DIR/plot_attribution_profiles.py" \
            --checkpoint_dir "$OUTPUT_DIR" --examples_csv "$REP_CSV" \
            --output_dir "$ANALYSIS_DIR/attribution_kmer" --input_mode "$INPUT_MODE" \
            --method ism --ism_mode kmer \
            --kmer_sizes "$KMER_SIZES" --kmer_replacement "$KMER_REPLACEMENT" \
            --kmer_n_shuffles "$KMER_N_SHUFFLES" --kmer_stride "$KMER_STRIDE" \
            --plot_window_bp "$ISM_WINDOW_BP" --score_mode "$ATTR_SCORE_MODE" \
            --n_per_category "$N_PER_CATEGORY" --left_bp "$LEFT_BP" --model_max_length "$MODEL_MAX_LENGTH" \
            --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" --device cuda
    fi
fi

echo
echo "========================================"
echo "Done: $RUN_NAME  (ARM=$ARM)"
echo "  data:       $DATA_DIR"
echo "  checkpoint: $OUTPUT_DIR  (test metrics in results/.../eval_results.json)"
echo "  log:        $LOGFILE"
echo "========================================"