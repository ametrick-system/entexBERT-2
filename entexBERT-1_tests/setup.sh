# Create virtual environment
cd ~/

module load miniconda

conda create -n entexbert_dnabert1 python=3.6 -y
conda activate entexbert_dnabert1

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