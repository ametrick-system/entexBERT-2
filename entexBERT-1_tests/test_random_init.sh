module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate entexbert_dnabert1
set -u

export PYTHONNOUSERSITE=1

export PROJECT_DIR="$HOME/entexBERT-2/entexBERT-1_tests"
export DNABERT_EXAMPLES="$PROJECT_DIR/external/DNABERT/examples"

export KMER=3
export DATA_PATH="$PROJECT_DIR/data/shift_exp_10k/offset_0"

export MODEL_PATH="$HOME/pretrained_models/DNABERT_3"

LOGFILE="$PROJECT_DIR/random_init.log"
exec &> >(tee -a "$LOGFILE")

cd "$DNABERT_EXAMPLES"

python entexbert_ft.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 257 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PROJECT_DIR \
    --n_process 48