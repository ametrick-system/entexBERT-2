#!/bin/bash
set -eo pipefail

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------

PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"

DATA_ROOT="$PROJECT_DIR/data/jitter_exp_10k"
RESULT_ROOT="$PROJECT_DIR/results/jitter_exp_10k"

DNABERT_EXAMPLES="$PROJECT_DIR/external/DNABERT/examples"
PRETRAINED_MODEL="$HOME/pretrained_models/DNABERT_6"

SEED=42
MAX_PER_CLASS=10000

WINDOW_SIZE=256
KMER=6

EPOCHS=5
BATCH_SIZE=16
LR=2e-5

# One experiment per jitter radius.
# jitter_16 means each example has SNP offset sampled from [-16, +16].
JITTER_MAX_OFFSETS=(16 32 64)

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

echo "============================================================"
echo "Running entexBERT jitter experiment"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "AS_TSV=$AS_TSV"
echo "REF_FASTA=$REF_FASTA"
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

test -f "$AS_TSV"
test -f "$REF_FASTA"
test -d "$DNABERT_EXAMPLES"
test -d "$PRETRAINED_MODEL"
test -f "$PROJECT_DIR/format_data.py"

# Make sure format_data.py has the jitter options.
if ! python "$PROJECT_DIR/format_data.py" --help | grep -q -- "--include_jitter"; then
    echo "ERROR: format_data.py does not appear to support --include_jitter yet."
    echo "Add --include_jitter and --jitter_max_offset before running this script."
    exit 1
fi

# ---------------------------------------------------------------------
# Output setup
# ---------------------------------------------------------------------

rm -rf "$DATA_ROOT"
rm -rf "$RESULT_ROOT"

mkdir -p "$DATA_ROOT"
mkdir -p "$RESULT_ROOT"

SUMMARY_TSV="$RESULT_ROOT/summary.tsv"

echo -e "experiment\tjitter_max_offset\ttrain_n\tval_n\ttest_n\toffset_min\toffset_max\toffset_mean\toffset_std\tacc\tauc\taupr\tf1\tmcc\tprecision\trecall\tmodel_dir\tlog_file" > "$SUMMARY_TSV"

# ---------------------------------------------------------------------
# Run one jitter experiment per jitter radius
# ---------------------------------------------------------------------

for JITTER_MAX in "${JITTER_MAX_OFFSETS[@]}"
do
    EXP_NAME="jitter_${JITTER_MAX}"

    DATA_OUT="$DATA_ROOT/$EXP_NAME"
    DATA_SUBDIR="$DATA_OUT/offset_jitter"

    EXP_RESULT_DIR="$RESULT_ROOT/$EXP_NAME"
    MODEL_OUT="$EXP_RESULT_DIR/model"
    PREDICT_DIR="$EXP_RESULT_DIR/predict"
    LOGFILE="$EXP_RESULT_DIR/run.log"

    mkdir -p "$DATA_OUT"
    mkdir -p "$MODEL_OUT" "$PREDICT_DIR"

    echo "============================================================"
    echo "Generating data for $EXP_NAME"
    echo "SNP offset sampled from [-$JITTER_MAX, +$JITTER_MAX]"
    echo "DATA_OUT=$DATA_OUT"
    echo "============================================================"

    python "$PROJECT_DIR/format_data.py" \
      --as_tsv "$AS_TSV" \
      --ref_fasta "$REF_FASTA" \
      --out_dir "$DATA_OUT" \
      --assay "TF-ChIP-seq_CTCF" \
      --donor "ENC-002" \
      --window_size "$WINDOW_SIZE" \
      --k "$KMER" \
      --offsets "" \
      --include_jitter \
      --jitter_max_offset "$JITTER_MAX" \
      --min_total_reads 10 \
      --max_per_class "$MAX_PER_CLASS" \
      --seed "$SEED" \
      --allele_mode reference

    if [[ ! -d "$DATA_SUBDIR" ]]; then
        echo "ERROR: Expected jitter data directory missing: $DATA_SUBDIR"
        echo "Check that clean_offset_name('jitter') returns offset_jitter in format_data.py."
        exit 1
    fi

    TRAIN_N=$(wc -l < "$DATA_SUBDIR/train.txt")
    VAL_N=$(wc -l < "$DATA_SUBDIR/val.txt")
    TEST_N=$(wc -l < "$DATA_SUBDIR/test.txt")

    OFFSET_STATS=$(python - <<PY
import pandas as pd
meta = pd.read_csv("$DATA_SUBDIR/metadata.tsv", sep="\t")
print(
    f"{int(meta['offset'].min())}\\t"
    f"{int(meta['offset'].max())}\\t"
    f"{meta['offset'].mean():.6f}\\t"
    f"{meta['offset'].std():.6f}"
)
PY
)

    echo "============================================================"
    echo "Fine-tuning entexBERT dnasnp for $EXP_NAME"
    echo "train=$TRAIN_N val=$VAL_N test=$TEST_N"
    echo "offset stats: $OFFSET_STATS"
    echo "DATA_SUBDIR=$DATA_SUBDIR"
    echo "MODEL_OUT=$MODEL_OUT"
    echo "LOGFILE=$LOGFILE"
    echo "============================================================"

    cd "$DNABERT_EXAMPLES"

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
      --pred_layer 11 \
      --seed "$SEED" \
      2>&1 | tee "$LOGFILE"

    # -----------------------------------------------------------------
    # Extract final prediction metrics.
    # -----------------------------------------------------------------

    python - <<PY >> "$SUMMARY_TSV"
import re

experiment = "$EXP_NAME"
jitter_max = "$JITTER_MAX"
train_n = "$TRAIN_N"
val_n = "$VAL_N"
test_n = "$TEST_N"
offset_stats = "$OFFSET_STATS".split("\\t")
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
        jitter_max,
        train_n,
        val_n,
        test_n,
        offset_stats[0],
        offset_stats[1],
        offset_stats[2],
        offset_stats[3],
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

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------

echo "============================================================"
echo "All jitter experiments finished."
echo "Summary saved to:"
echo "$SUMMARY_TSV"
echo "============================================================"

column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"