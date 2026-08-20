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
pip install h5py logomaker modisco-lite
```
## Quickstart

The pipeline is **config-driven**. `run_experiment.py` reads a YAML config, builds the
windowed dataset (`train/dev/test.csv` + `.meta.csv`), and `finetune_entexbert2.py` trains from it.

```bash
# 1. Stage-1: build + train the binding trunk (regression)
python -m entexbert2.run_experiment configs/ctcf_stage1_binding.yaml \
    --ref_fasta $REF --output_dir $BUILD_S1
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path DNABERT-2-117M-attention --data_path $BUILD_S1 \
    --task regression --input_mode ref_single \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 1 --model_max_length 512 \
    --num_train_epochs 3 --learning_rate 1e-3 --bf16 \
    --output_dir $TRUNK

# 2. Stage-2: build + train the ASB head on the FROZEN trunk (classification)
python -m entexbert2.run_experiment configs/ctcf_stage2_asb.yaml \
    --ref_fasta $REF --output_dir $BUILD_S2
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path DNABERT-2-117M-attention --data_path $BUILD_S2 \
    --task classification --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 2 --head_hidden_size 128 --proj_dim 128 \
    --balanced_sampler True --neff_s 20 \
    --init_backbone_from $TRUNK/pytorch_model.bin --freeze_backbone True \
    --num_train_epochs 3 --learning_rate 1e-3 --bf16 \
    --output_dir $HEAD

# 3. Score against ADASTRA (leak-free) and EN-TEx hetSNVs
python -m entexbert2.score_asb --eval adastra --checkpoint_dir $HEAD \
    --ref_fasta $REF --eval_csv $ADASTRA --assay CTCF \
    --train_coords $BUILD_S2/train.meta.csv $BUILD_S2/dev.meta.csv --drop_leaky \
    --left_bp 128 --right_bp 128 --out $OUT/score_adastra
```

---

## Model arguments (`finetune_entexbert2.py`)

| Argument | Values | Meaning |
|---|---|---|
| `--task` | `regression` \| `classification` | Stage-1 binding trunk vs Stage-2 ASB head. Selects head topology and loss. |
| `--input_mode` | `ref_single` \| `hap_pair` | One reference window (Stage 1) vs the (hap1, hap2) twin (Stage 2). |
| `--pooling_mode` | `center_mean` | Mean-pool the central tokens of the window into the sequence representation. |
| `--center_pool_width` | int (default 5) | Number of central tokens averaged by `center_mean`. |
| `--proj_dim` | int (default 128) | Width of the shared projection φ: 768 → d. |
| `--head_num_layers` | int | 1 = linear head; 2 = one hidden layer (`--head_hidden_size`). |
| `--head_hidden_size` | int | Hidden width when `head_num_layers ≥ 2` (−1 = none). |
| `--balanced_sampler` | bool | Class-balanced sampler at train time (Stage-2 ASB; do **not** also subsample the data). |
| `--neff_s` | float | Saturation cap `s` for the privileged depth weight `w = n(1+s)/(n+s)`. Typical 20–50. |
| `--init_backbone_from` | path | Initialize the backbone from a checkpoint (Stage-2 loads the Stage-1 trunk here). |
| `--freeze_backbone` | bool | Freeze the backbone (True for the frozen-trunk Stage-2 arm). |

Everything else is standard HuggingFace `TrainingArguments`
(`--learning_rate`, `--num_train_epochs`, `--per_device_train_batch_size`, `--bf16`,
`--metric_for_best_model auroc`, …).