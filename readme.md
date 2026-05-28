# entexBERT-2

## Summary
Fine-tuning a Transformer model on EN-TEx data built off a modified DNABERT-2 backbone

## Environment Setup
```bash
# Step 1: create virtual environment (in home directory)
module load miniconda
conda create -n dnabert2 python=3.8

# Step 2: activate virtual environment & install base software
conda activate dnabert2
git clone https://github.com/openai/triton.git
cd triton/python
pip install cmake

# Step 3: download DNABERT2 code and install other requirements/dependencies (back in home directory)
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
cd DNABERT_2
python3 -m pip install -r requirements.txt
pip install biopython

# Step 4: install other useful packages
pip install edlib # for sequence alignment
pip install pyBigWig # for processing bigWig files in python
```