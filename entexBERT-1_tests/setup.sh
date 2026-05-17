#!/bin/bash
set -eo pipefail
export PYTHONNOUSERSITE=1

# Create virtual environment
cd ~/

module load miniconda

conda create -n entexbert_dnabert1 python=3.6 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate entexbert_dnabert1
set -u

conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge -y

cd ~/entexBERT-2/entexBERT-1_tests/external/DNABERT

python3 -m pip install --editable .

cd examples
python3 -m pip install -r requirements.txt
conda install -y -c defaults "mkl<2024" "intel-openmp<2024" # downgrade mkl for compatibility
python -m pip install "tensorflow==1.15.5" "protobuf<3.20" # install tensorflow

# Move .py files from entexBERT into DNABERT directory
cp ~/entexBERT-2/entexBERT-1_tests/external/entexBERT-paper/*.py ~/entexBERT-2/entexBERT-1_tests/external/DNABERT/examples/

# Check setup
cd ~/entexBERT-2/entexBERT-1_tests/external/DNABERT/examples

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

import tensorflow as tf
print("tensorflow:", tf.__version__)

import entexbert_ft
import finetune_models
import utils_glue

print("entexBERT import OK")
PY

# download DNABERT with k=6
MODEL_DIR="$HOME/pretrained_models/DNABERT_6"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/config.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/pytorch_model.bin
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/vocab.txt
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/special_tokens_map.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/tokenizer_config.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/configuration_bert.py
wget -nc https://huggingface.co/zhihan1996/DNA_bert_6/resolve/main/dnabert_layer.py

# check that download was successful
ls -lh "$MODEL_DIR"

# check that entexBERT can access pretrained model
cd ~/entexBERT-2/entexBERT-1_tests/external/DNABERT/examples

python - <<'PY'
import os
import torch
from transformers import BertConfig, BertForMaskedLM, DNATokenizer

model_dir = os.path.expanduser("~/pretrained_models/DNABERT_6")

print("model_dir:", model_dir)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

config = BertConfig.from_pretrained(model_dir)
print("config hidden size:", config.hidden_size)
print("config num layers:", config.num_hidden_layers)

tokenizer = DNATokenizer.from_pretrained(model_dir)
print("tokenizer vocab size:", len(tokenizer))

model = BertForMaskedLM.from_pretrained(model_dir, config=config)
print("model loaded OK")
PY