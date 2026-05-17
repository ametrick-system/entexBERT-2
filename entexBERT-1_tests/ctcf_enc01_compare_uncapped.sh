

# ---------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------

PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"

AS_TSV="$HOME/entex_data/hetSNVs.tsv"
REF_FASTA="$HOME/reference_genome/hg38.fa"

PRETRAINED_MODEL="$HOME/pretrained_models/DNABERT_3"
DNABERT_EXAMPLES="$PROJECT_DIR/external/DNABERT/examples"

DATA_DIR="$PROJECT_DIR/data/ctcf_enc01_compare_uncapped"
RESULT_DIR="$PROJECT_DIR/results/ctcf_enc01_compare_uncapped"

DONOR="ENC-001"
ASSAY="TF-ChIP-seq_CTCF"

SEED=42

WINDOW_SIZE=256
KMER=3
JITTER_MAX_OFFSET=64

EPOCHS=5
BATCH_SIZE=16
LR=5e-5

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

echo "============================================================"
echo "Paper-style CTCF Individual 1 comparison"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "AS_TSV=$AS_TSV"
echo "REF_FASTA=$REF_FASTA"
echo "PRETRAINED_MODEL=$PRETRAINED_MODEL"
echo "DONOR=$DONOR"
echo "ASSAY=$ASSAY"
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

[[ -f "$AS_TSV" ]] || { echo "Missing AS_TSV: $AS_TSV"; exit 1; }
[[ -f "$REF_FASTA" ]] || { echo "Missing REF_FASTA: $REF_FASTA"; exit 1; }
[[ -d "$PRETRAINED_MODEL" ]] || { echo "Missing PRETRAINED_MODEL: $PRETRAINED_MODEL"; exit 1; }
[[ -d "$DNABERT_EXAMPLES" ]] || { echo "Missing DNABERT_EXAMPLES: $DNABERT_EXAMPLES"; exit 1; }
[[ -f "$PROJECT_DIR/format_data.py" ]] || { echo "Missing format_data.py: $PROJECT_DIR/format_data.py"; exit 1; }

python -m py_compile "$PROJECT_DIR/format_data.py"

if ! python "$PROJECT_DIR/format_data.py" --help | grep -q -- "--include_jitter"; then
    echo "ERROR: format_data.py does not support --include_jitter."
    exit 1
fi

# ---------------------------------------------------------------------
# 1. Generate centered + jittered datasets from the same selected examples
# ---------------------------------------------------------------------

echo "============================================================"
echo "Generating data"
echo "============================================================"

rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

python "$PROJECT_DIR/format_data.py" \
  --as_tsv "$AS_TSV" \
  --ref_fasta "$REF_FASTA" \
  --out_dir "$DATA_DIR" \
  --assay "$ASSAY" \
  --donor "$DONOR" \
  --window_size "$WINDOW_SIZE" \
  --k "$KMER" \
  --offsets "0" \
  --include_jitter \
  --jitter_max_offset "$JITTER_MAX_OFFSET" \
  --min_total_reads 10 \
  --seed "$SEED" \
  --allele_mode reference

CENTERED_DATA="$DATA_DIR/offset_0"
JITTER_DATA="$DATA_DIR/offset_jitter"

[[ -d "$CENTERED_DATA" ]] || { echo "Missing centered data: $CENTERED_DATA"; exit 1; }
[[ -d "$JITTER_DATA" ]] || { echo "Missing jitter data: $JITTER_DATA"; exit 1; }

echo "Centered data counts:"
wc -l "$CENTERED_DATA"/train.txt "$CENTERED_DATA"/val.txt "$CENTERED_DATA"/test.txt

echo "Jitter data counts:"
wc -l "$JITTER_DATA"/train.txt "$JITTER_DATA"/val.txt "$JITTER_DATA"/test.txt

echo "Checking coordinate leakage:"
python - <<PY
import pandas as pd
from pathlib import Path

for name, path in [
    ("centered", Path("$CENTERED_DATA") / "metadata.tsv"),
    ("jitter", Path("$JITTER_DATA") / "metadata.tsv"),
]:
    meta = pd.read_csv(path, sep="\t")
    coord_cols = ["chr", "ref_start", "ref_end"]
    split_counts = meta.groupby(coord_cols)["split"].nunique()
    print(name)
    print("  rows:", len(meta))
    print("  unique coords:", meta.groupby(coord_cols).ngroups)
    print("  coords in multiple splits:", int((split_counts > 1).sum()))
    print(pd.crosstab(meta["split"], meta["label"]))
    if name == "jitter":
        print("  jitter offset min/max/mean/std:",
              int(meta["offset"].min()),
              int(meta["offset"].max()),
              float(meta["offset"].mean()),
              float(meta["offset"].std()))
PY

# ---------------------------------------------------------------------
# 2. Run the three model/data comparisons
# ---------------------------------------------------------------------

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

SUMMARY_TSV="$RESULT_DIR/summary.tsv"
echo -e "experiment\tmodel_type\tdata_variant\ttrain_n\tval_n\ttest_n\tacc\tauc\taupr\tf1\tmcc\tprecision\trecall\tmodel_dir\tlog_file" > "$SUMMARY_TSV"

run_experiment () {
    EXPERIMENT="$1"
    MODEL_TYPE="$2"
    DATA_SUBDIR="$3"
    DATA_VARIANT="$4"

    EXP_RESULT_DIR="$RESULT_DIR/$EXPERIMENT"
    MODEL_OUT="$EXP_RESULT_DIR/model"
    PREDICT_DIR="$EXP_RESULT_DIR/predict"
    LOGFILE="$EXP_RESULT_DIR/run.log"

    mkdir -p "$MODEL_OUT" "$PREDICT_DIR"

    TRAIN_N=$(wc -l < "$DATA_SUBDIR/train.txt")
    VAL_N=$(wc -l < "$DATA_SUBDIR/val.txt")
    TEST_N=$(wc -l < "$DATA_SUBDIR/test.txt")

    echo "============================================================"
    echo "Experiment: $EXPERIMENT"
    echo "MODEL_TYPE=$MODEL_TYPE"
    echo "DATA_VARIANT=$DATA_VARIANT"
    echo "DATA_SUBDIR=$DATA_SUBDIR"
    echo "train=$TRAIN_N val=$VAL_N test=$TEST_N"
    echo "MODEL_OUT=$MODEL_OUT"
    echo "LOGFILE=$LOGFILE"
    echo "============================================================"

    cd "$DNABERT_EXAMPLES"

    python entexbert_ft.py \
      --model_type "$MODEL_TYPE" \
      --tokenizer_name dna3 \
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

    python - <<PY >> "$SUMMARY_TSV"
import re

experiment = "$EXPERIMENT"
model_type = "$MODEL_TYPE"
data_variant = "$DATA_VARIANT"
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
        data_variant,
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
}

# DNABERT1 standard CLS control on centered SNP windows
run_experiment "dnabert1_cls_centered" "dna" "$CENTERED_DATA" "centered"

# entexBERT center-pooling head on the exact same centered SNP windows
run_experiment "entexbert_dnasnp_centered" "dnasnp" "$CENTERED_DATA" "centered"

# entexBERT center-pooling head on same examples/splits, but jittered SNP positions
run_experiment "entexbert_dnasnp_jitter64" "dnasnp" "$JITTER_DATA" "jitter64"

# ---------------------------------------------------------------------
# Final display
# ---------------------------------------------------------------------

echo "============================================================"
echo "Finished comparison."
echo "Summary saved to:"
echo "$SUMMARY_TSV"
echo "============================================================"

column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"