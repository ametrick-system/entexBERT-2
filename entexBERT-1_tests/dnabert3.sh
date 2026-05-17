#!/bin/bash
set -eo pipefail
export PYTHONNOUSERSITE=1

# Create virtual environment
module load miniconda


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate entexbert_dnabert1
set -u

# download DNABERT with k=6
MODEL_DIR="$HOME/pretrained_models/DNABERT_3"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/config.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/pytorch_model.bin
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/vocab.txt
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/special_tokens_map.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/tokenizer_config.json
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/configuration_bert.py
wget -nc https://huggingface.co/zhihan1996/DNA_bert_3/resolve/main/dnabert_layer.py

# check that download was successful
ls -lh "$MODEL_DIR"

# check that entexBERT can access pretrained model
cd ~/entexBERT-2/entexBERT-1_tests/external/DNABERT/examples

python - <<'PY'
import os
import torch
from transformers import BertConfig, BertForMaskedLM, DNATokenizer

model_dir = os.path.expanduser("~/pretrained_models/DNABERT_3")

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