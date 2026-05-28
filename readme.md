# entexBERT-2

## Summary
Fine-tuning a Transformer model on EN-TEx data built off a modified DNABERT-2 backbone

## Environment Setup
```bash
# Step 1: create virtual environment (in home directory)
module load miniconda
conda create -n eb2 python=3.8

# Step 2: activate virtual environment & install DNABERT-2 software
conda activate eb2
git clone https://github.com/MAGICS-LAB/DNABERT_2.git
cd DNABERT_2
python3 -m pip install -r requirements.txt

# Step 3: [in home directory] install entexBERT-2 as a package with its dependencies
git clone https://github.com/ametrick-system/entexBERT-2.git
pip install .
```