#!/bin/bash
set -eo pipefail

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"

DATA_DIR="$PROJECT_DIR/data/shift_exp_10k"
RESULT_DIR="$PROJECT_DIR/results/shift_exp_10k"

DNABERT_EXAMPLES="$PROJECT_DIR/external/DNABERT/examples"
PRETRAINED_MODEL="$HOME/pretrained_models/DNABERT_6"

EPOCHS=5
BATCH_SIZE=16
LR=2e-5
SEED=42

echo "Project dir: $PROJECT_DIR"
echo "Data dir: $DATA_DIR"
echo "Result dir: $RESULT_DIR"
echo "Pretrained model: $PRETRAINED_MODEL"

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
import tensorflow as tf
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("tensorflow:", tf.__version__)
PY

# ---------------------------------------------------------------------
# 1. Generate shifted datasets
# ---------------------------------------------------------------------

rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

python "$PROJECT_DIR/format_data.py" \
  --as_tsv "$AS_TSV" \
  --ref_fasta "$REF_FASTA" \
  --out_dir "$DATA_DIR" \
  --assay "TF-ChIP-seq_CTCF" \
  --donor "ENC-002" \
  --window_size 256 \
  --k 6 \
  --offsets "0,16,32,64,-16,-32,-64" \
  --include_random \
  --min_total_reads 10 \
  --max_per_class 10000 \
  --seed "$SEED" \
  --allele_mode reference

# ---------------------------------------------------------------------
# 2. Fine-tune one entexBERT model per shifted dataset
# ---------------------------------------------------------------------

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

SUMMARY_TSV="$RESULT_DIR/summary.tsv"
echo -e "offset\ttrain_n\tval_n\ttest_n\tacc\tauc\taupr\tf1\tmcc\tprecision\trecall\tmodel_dir\tlog_file" > "$SUMMARY_TSV"

OFFSETS=(
  "offset_0"
  "offset_16"
  "offset_32"
  "offset_64"
  "offset_m16"
  "offset_m32"
  "offset_m64"
  "offset_random"
)

cd "$DNABERT_EXAMPLES"

for OFFSET in "${OFFSETS[@]}"
do
    DATA_SUBDIR="$DATA_DIR/$OFFSET"
    OFFSET_RESULT_DIR="$RESULT_DIR/$OFFSET"
    MODEL_OUT="$OFFSET_RESULT_DIR/model"
    PREDICT_DIR="$OFFSET_RESULT_DIR/predict"
    LOGFILE="$OFFSET_RESULT_DIR/run.log"

    mkdir -p "$MODEL_OUT" "$PREDICT_DIR"

    TRAIN_N=$(wc -l < "$DATA_SUBDIR/train.txt")
    VAL_N=$(wc -l < "$DATA_SUBDIR/val.txt")
    TEST_N=$(wc -l < "$DATA_SUBDIR/test.txt")

    echo "============================================================"
    echo "Running entexBERT dnasnp for $OFFSET"
    echo "train=$TRAIN_N val=$VAL_N test=$TEST_N"
    echo "DATA_SUBDIR=$DATA_SUBDIR"
    echo "MODEL_OUT=$MODEL_OUT"
    echo "============================================================"

    python entexbert_ft.py \
      --model_type dnasnp \
      --tokenizer_name dna6 \
      --model_name_or_path "$PRETRAINED_MODEL" \
      --task_name dnaprom \
      --do_train \
      --do_eval \
      --do_predict \
      --data_dir "$DATA_SUBDIR" \
      --predict_dir "$PREDICT_DIR" \
      --max_seq_length 256 \
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
      --pred_layer 11 \
      --seed "$SEED" \
      2>&1 | tee "$LOGFILE"

    # -----------------------------------------------------------------
    # 3. Extract final prediction metrics from this offset's log
    # -----------------------------------------------------------------

    python - <<PY >> "$SUMMARY_TSV"
import re

offset = "$OFFSET"
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

pattern = re.compile(r"- INFO - __main__ -\s+([a-zA-Z0-9_]+) = ([^\\s]+)")

with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            key, value = m.group(1), m.group(2)
            if key in metrics:
                metrics[key] = value

print(
    "\\t".join([
        offset,
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

    echo "Finished $OFFSET"
    echo
done

echo "============================================================"
echo "All experiments finished."
echo "Summary:"
echo "$SUMMARY_TSV"
echo "============================================================"

column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"