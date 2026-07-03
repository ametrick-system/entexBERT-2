#!/bin/bash
set -euo pipefail

################################################################################
# entexBERT-2: AS-SIGNIFICANCE (binary) from the TWIN, with a SYMMETRIC contrast head.
#
# Target : imbalance_significance (is this site allele-specific at all? 1/0)  -- SYMMETRIC
# Input  : ref_alt_pair (twin: ref window + alt window)
# Head   : contrast_mode=symmetric_abs  ->  logits = head(|pool(alt) - pool(ref)|)
#          NOT the signed head(alt)-head(ref): significance is direction-agnostic, so an
#          antisymmetric head would only detect alt-favored imbalance and miss ref-favored AS.
#
# This is a HEAD-TO-HEAD against your existing SINGLE-WINDOW significance classifier
# (ref_single -> imbalance_significance, AUPRC ~0.265 at ~3.9% prevalence). Same target, same
# split policy, same metric. The twin earns its place ONLY if it beats that single-window AUPRC.
#
# Also reports the DEPTH-CONFOUND correlation (predicted prob vs total_reads): AS calls are
# easier at high depth, so a classifier can ride coverage/power rather than allelic biology.
#
# REQUIRES (cluster code):
#   - finetune_entexbert2.py : contrast_mode field + model attr + symmetric branch in forward
#                              (see the by-content edits accompanying this script)
#   - model_io.py            : build_model sets model.contrast_mode; logits_and_embeddings
#                              does head(|pool(alt)-pool(ref)|) when contrast_mode=symmetric_abs
#   - utils.py               : ref_alt_pair input mode (already deployed)
################################################################################

module purge
module load miniconda
conda activate eb2

PROJECT_DIR="$HOME/entexBERT-2"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
SCRIPTS_DIR="$PROJECT_DIR/src/entexbert2/scripts"

########################################
# Experiment identity (CTCF / individual 1)  -- mirror the single-window classifier
########################################
DONOR="ENC-001"; DONOR_NUMBER="1"
TISSUE_ARG="ALL"; TISSUE_LABEL="all_tissues"
ASSAY="TF-ChIP-seq_CTCF"; ASSAY_NICKNAME="CTCF"

########################################
# Target / input / contrast
########################################
INPUT_MODE="ref_alt_pair"
CONTRAST_MODE="symmetric_abs"     # |pool(alt) - pool(ref)| -> head ; symmetric for significance

########################################
# Dataset / window  (MATCH the single-window classifier for a fair head-to-head)
########################################
MIN_TOTAL_READS=10                # same floor as the classification pipeline (not the reg's 20)
LEFT_BP=256; RIGHT_BP=256
JITTER_MAX_BP=50
SPLIT_TRAIN=0.8; SPLIT_DEV=0.1; SPLIT_TEST=0.1
SEED=42; CHUNKSIZE=100000

########################################
# Balancing / weighting ARM  (mirror the classifier arms; default = full + class-weighted)
########################################
ARM="${1:-full_weighted}"
case "$ARM" in
  balanced_train)  BALANCE_STRATEGY="per_tissue_binary"; BALANCE_APPLY_TO="train"; CLASS_WEIGHTS="none";     ARM_TAG="baltrain" ;;
  full_weighted)   BALANCE_STRATEGY="none";              BALANCE_APPLY_TO="all";   CLASS_WEIGHTS="balanced"; ARM_TAG="fullw" ;;
  full_unweighted) BALANCE_STRATEGY="none";              BALANCE_APPLY_TO="all";   CLASS_WEIGHTS="none";     ARM_TAG="fullu" ;;
  *) echo "Unknown ARM '$ARM' (balanced_train|full_weighted|full_unweighted)"; exit 1 ;;
esac

########################################
# Model / head / training  (CLS + linear head, to match the single-window classifier)
########################################
MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"
POOLING_MODE="cls"
CENTER_POOL_WIDTH=5
HEAD_NUM_LAYERS=1
HEAD_HIDDEN_SIZE=-1
HEAD_ACTIVATION="gelu"
HEAD_DROPOUT=0.1

MODEL_MAX_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01; WARMUP_RATIO=0.06
LOGGING_STEPS=50; EVAL_STEPS=200; SAVE_STEPS=200
SELECT_METRIC="eval_auprc"        # same as the single-window classifier
ANALYSIS_THRESHOLD="f1"           # dev-derived decision threshold

RUN_TRAINING="true"; RUN_ANALYSIS="true"
N_PER_CATEGORY=100

########################################
# Inputs
########################################
AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"
CHROM_SIZES="$HOME/reference_genome/hg38.chrom.sizes"

########################################
# Derived paths
########################################
if [[ "$JITTER_MAX_BP" -gt 0 ]]; then
    WIN_TAG="${INPUT_MODE}_signif_${LEFT_BP}_${RIGHT_BP}_jitter${JITTER_MAX_BP}"; OFFSET_MODE="uniform"
else
    WIN_TAG="${INPUT_MODE}_signif_${LEFT_BP}_${RIGHT_BP}"; OFFSET_MODE="fixed"
fi
RUN_SUBDIR="${WIN_TAG}_${CONTRAST_MODE}_${ARM_TAG}"
RUN_NAME="${INPUT_MODE}_${ASSAY_NICKNAME}_${DONOR}_${TISSUE_LABEL}_TWINsignif_${ARM_TAG}"

EXPERIMENT_DIR="$PROJECT_DIR/AS/${DONOR_NUMBER}/${ASSAY_NICKNAME}/${TISSUE_LABEL}/${RUN_SUBDIR}"
DATA_DIR="${EXPERIMENT_DIR}/input"; OUTPUT_DIR="${EXPERIMENT_DIR}/output"
ANALYSIS_DIR="${EXPERIMENT_DIR}/analysis"; LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$ANALYSIS_DIR" "$LOG_DIR"
CONFIG_FILE="${EXPERIMENT_DIR}/dataset_config.yaml"
LOGFILE="${LOG_DIR}/${RUN_NAME}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "TWIN-significance: $RUN_NAME"
echo "Date: $(date)  Host: $(hostname)"
echo "Target=imbalance_significance  input=$INPUT_MODE  contrast=$CONTRAST_MODE  ARM=$ARM"
echo "Head-to-head vs single-window classifier (AUPRC ~0.265). Select=$SELECT_METRIC."
echo "========================================"

for p in "$AS_TSV" "$REF_FASTA" "$CHROM_SIZES"; do [[ -f "$p" ]] || { echo "ERROR missing $p"; exit 1; }; done

if [[ "$RUN_TRAINING" == "true" ]]; then
    echo; echo "===== 1. Build dataset (ref_alt_pair -> imbalance_significance) ====="
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

balance: {strategy: ${BALANCE_STRATEGY}, apply_to: ${BALANCE_APPLY_TO}}

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

    echo; echo "===== 2. Leakage check (pair-aware) + prevalence per split ====="
    python - <<PY
import os, sys, pandas as pd
d = "$DATA_DIR"
dfs = {s: pd.read_csv(os.path.join(d, f"{s}.csv")) for s in ["train","dev","test"]}
def keyset(df, cols): return set(map(tuple, df[cols].astype(str).itertuples(index=False, name=None)))
pair_cols = ["sequence1","sequence2"] if "sequence1" in dfs["train"].columns else ["sequence"]
loc = "locus_id" in dfs["train"].columns
leak=False
for k,(a,b) in {"train_dev":("train","dev"),"train_test":("train","test"),"dev_test":("dev","test")}.items():
    pd_=len(keyset(dfs[a],pair_cols)&keyset(dfs[b],pair_cols))
    lo=len(keyset(dfs[a],["locus_id"])&keyset(dfs[b],["locus_id"])) if loc else -1
    print(f"  {k}: pair-dups={pd_} locus-overlap={lo}")
    leak = leak or pd_>0 or lo>0
if leak: print("ERROR: true cross-split leakage."); sys.exit(1)
print("No true cross-split leakage.")
for s,df in dfs.items():
    y=df["label"].astype(int)
    print(f"  {s:5s}: n={len(df):7d}  pos={int(y.sum())} ({y.mean():.3%})")
PY

    echo; echo "===== 3. Train (twin, symmetric_abs contrast; classification) ====="
    # NOTE: --evaluation_strategy may be --eval_strategy on newer transformers.
    python -m entexbert2.finetune_entexbert2 \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --data_path "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --task classification \
        --main_num_labels 2 \
        --contrast_mode "$CONTRAST_MODE" \
        --class_weights "$CLASS_WEIGHTS" \
        --pooling_mode "$POOLING_MODE" \
        --center_pool_width "$CENTER_POOL_WIDTH" \
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
        --save_total_limit 2 \
        --eval_and_save_results True \
        --save_model True \
        --fp16 True \
        --dataloader_pin_memory False \
        --seed "$SEED"
fi

if [[ ! -f "$OUTPUT_DIR/run_config.json" ]]; then echo "ERROR: no run_config.json"; exit 1; fi

if [[ "$RUN_ANALYSIS" == "true" ]]; then
    echo; echo "===== 4. analyze.py (AUPRC/AUROC + dev-derived threshold) ====="
    python -m entexbert2.analyze \
        --checkpoint_dir "$OUTPUT_DIR" \
        --data_csv "$DATA_DIR/test.csv" \
        --dev_csv "$DATA_DIR/dev.csv" \
        --output_dir "$ANALYSIS_DIR" \
        --n_per_category "$N_PER_CATEGORY" \
        --batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --threshold "$ANALYSIS_THRESHOLD" \
        --device cuda
    echo "Metrics:"; cat "$ANALYSIS_DIR/metrics.json" 2>/dev/null || true

    echo; echo "===== 5. Depth-confound check (pred prob vs total_reads) ====="
    python - <<PY
import os, json, numpy as np, pandas as pd
ad="$ANALYSIS_DIR"; dd="$DATA_DIR"
try:
    pred=pd.read_csv(os.path.join(ad,"predictions.csv"))
except Exception as e:
    print("  (no predictions.csv:", e, ")"); raise SystemExit
# probability column: prefer an explicit positive-class prob, else derive
pcol=[c for c in pred.columns if c.lower() in ("prob","prob_1","p_pos","score","pred_prob")]
prob = pred[pcol[0]] if pcol else (pred["pred_value"] if "pred_value" in pred.columns else None)
meta=os.path.join(dd,"test.meta.csv"); base=os.path.join(dd,"test.csv")
m=pd.read_csv(meta) if os.path.exists(meta) else pd.read_csv(base)
if prob is None or "total_reads" not in m.columns or len(m)!=len(pred):
    print("  cannot align prob & total_reads; skipping (cols:",list(pred.columns),")")
else:
    from scipy import stats
    depth=pd.to_numeric(m["total_reads"],errors="coerce").to_numpy()
    p=pd.to_numeric(prob,errors="coerce").to_numpy()
    ok=~np.isnan(depth)&~np.isnan(p)
    rho,_=stats.spearmanr(depth[ok],p[ok])
    print(f"  spearman(predicted prob, total_reads) = {rho:+.3f}")
    print("  READ: strongly positive => the classifier is partly riding the depth/power confound,")
    print("        not purely allelic biology. Compare this between twin and single-window models.")
PY
fi

echo
echo "========================================"
echo "Done: $RUN_NAME"
echo "  COMPARE: this twin AUPRC vs the single-window classifier AUPRC (~0.265, same split policy)."
echo "    twin >  single-window  => the allelic contrast adds significance signal (good, publishable)"
echo "    twin ~= single-window  => significance is already in the reference window; contrast adds little"
echo "    twin <  single-window  => contrast noise > added signal"
echo "  metrics: $ANALYSIS_DIR/metrics.json   log: $LOGFILE"
echo "========================================"
