#!/bin/bash
set -eo pipefail

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"

DATA_ROOT="$PROJECT_DIR/data/jitter_exp_10k"
RESULT_ROOT="$PROJECT_DIR/results/cls_control_10k"

DNABERT_EXAMPLES="$PROJECT_DIR/external/DNABERT/examples"
PRETRAINED_MODEL="$HOME/pretrained_models/DNABERT_6"

SEED=42
WINDOW_SIZE=256
EPOCHS=5
BATCH_SIZE=16
LR=2e-5

JITTER_EXPS=("jitter_16" "jitter_32" "jitter_64")

echo "============================================================"
echo "Running DNABERT1 CLS baseline on jitter datasets"
echo "DATA_ROOT=$DATA_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "PRETRAINED_MODEL=$PRETRAINED_MODEL"
echo "============================================================"

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
import tensorflow as tf
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("tensorflow:", tf.__version__)
PY

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT"; exit 1; }
[[ -d "$DNABERT_EXAMPLES" ]] || { echo "Missing DNABERT_EXAMPLES: $DNABERT_EXAMPLES"; exit 1; }
[[ -d "$PRETRAINED_MODEL" ]] || { echo "Missing PRETRAINED_MODEL: $PRETRAINED_MODEL"; exit 1; }

rm -rf "$RESULT_ROOT"
mkdir -p "$RESULT_ROOT"

SUMMARY_TSV="$RESULT_ROOT/summary.tsv"
echo -e "experiment\tmodel_type\ttrain_n\tval_n\ttest_n\tacc\tauc\taupr\tf1\tmcc\tprecision\trecall\tmodel_dir\tlog_file" > "$SUMMARY_TSV"

cd "$DNABERT_EXAMPLES"

for EXP_NAME in "${JITTER_EXPS[@]}"
do
    DATA_SUBDIR="$DATA_ROOT/$EXP_NAME/offset_jitter"

    EXP_RESULT_DIR="$RESULT_ROOT/$EXP_NAME"
    MODEL_OUT="$EXP_RESULT_DIR/model"
    PREDICT_DIR="$EXP_RESULT_DIR/predict"
    LOGFILE="$EXP_RESULT_DIR/run.log"

    [[ -d "$DATA_SUBDIR" ]] || { echo "Missing DATA_SUBDIR: $DATA_SUBDIR"; exit 1; }

    mkdir -p "$MODEL_OUT" "$PREDICT_DIR"

    TRAIN_N=$(wc -l < "$DATA_SUBDIR/train.txt")
    VAL_N=$(wc -l < "$DATA_SUBDIR/val.txt")
    TEST_N=$(wc -l < "$DATA_SUBDIR/test.txt")

    echo "============================================================"
    echo "Experiment: $EXP_NAME"
    echo "Model: DNABERT1 standard CLS classifier"
    echo "train=$TRAIN_N val=$VAL_N test=$TEST_N"
    echo "DATA_SUBDIR=$DATA_SUBDIR"
    echo "MODEL_OUT=$MODEL_OUT"
    echo "============================================================"

    python entexbert_ft.py \
      --model_type dna \
      --tokenizer_name dna6 \
      --model_name_or_path "$PRETRAINED_MODEL" \
      --task_name dnaprom \
      --do_train \
      --do_eval \
      --do_predict \
      --data_dir "$DATA_SUBDIR" \
      --predict_dir "$PREDICT_DIR" \
      --max_seq_length "$WINDOW_SIZE" \
      --per_gpu_train_batch_size "$BATCH_SIZE" \
      --per_gpu_eval_batch_size "$BATCH_SIZE" \
      --per_gpu_pred_batch_size "$BATCH_SIZE" \
      --learning_rate "$LR" \
      --num_train_epochs "$EPOCHS" \
      --output_dir "$MODEL_OUT" \
      --evaluate_during_training \
      --logging_steps 100 \
      --save_steps 100000 \
      --warmup_percent 0.1 \
      --hidden_dropout_prob 0.1 \
      --overwrite_output \
      --weight_decay 0.01 \
      --n_process 8 \
      --seed "$SEED" \
      2>&1 | tee "$LOGFILE"

    python - <<PY >> "$SUMMARY_TSV"
import re

experiment = "$EXP_NAME"
model_type = "dna_cls"
train_n = "$TRAIN_N"
val_n = "$VAL_N"
test_n = "$TEST_N"
model_dir = "$MODEL_OUT"
log_file = "$LOGFILE"

metrics = {
    "acc": "",
    "auc": "",
    "aupr": "",
    "f1": "",
    "mcc": "",
    "precision": "",
    "recall": "",
}

in_pred = False
pattern = re.compile(r"- INFO - __main__ -\\s+([a-zA-Z0-9_]+) = ([^\\s]+)")

with open(log_file) as f:
    for line in f:
        if "***** Pred results" in line:
            in_pred = True
            continue

        if in_pred:
            m = pattern.search(line)
            if m:
                key, value = m.group(1), m.group(2)
                if key in metrics:
                    metrics[key] = value

print(
    "\\t".join([
        experiment,
        model_type,
        train_n,
        val_n,
        test_n,
        metrics["acc"],
        metrics["auc"],
        metrics["aupr"],
        metrics["f1"],
        metrics["mcc"],
        metrics["precision"],
        metrics["recall"],
        model_dir,
        log_file,
    ])
)
PY

    echo "Finished $EXP_NAME"
    echo
done

echo "============================================================"
echo "DNABERT1 CLS baseline finished."
echo "Summary:"
echo "$SUMMARY_TSV"
echo "============================================================"

column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"