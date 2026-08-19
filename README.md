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
pip install h5py logomaker modisco-lite # TODO: FOLD INTO DEPENDENCIES
```

## `finetune_entexbert2.py` — CLI options

Run with `python -m entexbert2.finetune_entexbert2 <flags>` (parsed by `transformers.HfArgumentParser` over three dataclasses). Any [`transformers.TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) flag not listed below is also accepted (e.g. `--bf16`, `--report_to`, `--lr_scheduler_type`).

### Model / architecture (`ModelArguments`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--model_name_or_path` | str | `facebook/opt-125m` | Backbone checkpoint (for entexBERT-2, the DNABERT-2 path, e.g. `$HOME/entexBERT-2/DNABERT-2-117M-attention`). |
| `--pooling_mode` | str | `center_mean` | Token→sequence pooling: `center_mean` (mean over the center window) or `cls`. |
| `--center_pool_width` | int | `5` | Odd width of the center-mean pool window (ignored for `cls`). |
| `--head_num_layers` | int | `1` | Twin head `g_φ` depth: `1` = linear, `≥2` = MLP. |
| `--head_hidden_size` | int | `-1` | MLP hidden size (`-1` = auto; ignored when `head_num_layers=1`). |
| `--head_activation` | str | `gelu` | Head activation: `gelu`/`relu`/`tanh`/`silu`. |
| `--head_dropout` | float | `0.1` | Dropout in the head. |
| `--neff_s` | float | `50.0` | Privileged precision-weight saturation cap `s` in `w = n_eff(n) = n(1+s)/(n+s)`. `0` = unweighted MSE. |
| `--init_backbone_from` | str | `None` | Path to a Stage-1 checkpoint; loads **backbone.\*** weights only (partial, `strict=False`). Stage-2 transfer. |
| `--freeze_backbone` | bool | `False` | Freeze the backbone (Stage 2a: train the twin head alone). Omit for Stage 2b. |
| `--use_lora` | bool | `False` | Wrap the model in LoRA. |
| `--lora_r` | int | `8` | LoRA rank. |
| `--lora_alpha` | int | `32` | LoRA alpha. |
| `--lora_dropout` | float | `0.05` | LoRA dropout. |
| `--lora_target_modules` | str | `query,value` | Comma-separated modules to adapt. |

### Data (`DataArguments`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--data_path` | str | `None` | Directory containing `train.csv` / `dev.csv` / `test.csv` (built by `run_experiment.py`). |
| `--kmer` | int | `-1` | k-mer input encoding; `-1` = raw BPE (DNABERT-2 default). |
| `--task` | str | `regression` | Only `regression` is supported in the streamlined 2-stage model. |
| `--input_mode` | str | `hap_pair` | `hap_pair` (twin: two windows per example) or `single`. |

### Training (`TrainingArguments`, extends `transformers.TrainingArguments`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--output_dir` | str | `output` | Where checkpoints, `run_config.json`, and results are written. |
| `--run_name` | str | `run` | Run name (also the results subdir). |
| `--num_train_epochs` | int | `1` | Training epochs. |
| `--per_device_train_batch_size` | int | `1` | Train batch size per device. |
| `--per_device_eval_batch_size` | int | `1` | Eval batch size per device. |
| `--gradient_accumulation_steps` | int | `1` | Grad-accum steps (effective batch = train_bs × accum × #devices). |
| `--learning_rate` | float | `1e-4` | Peak LR. For Stage 2b gentle unfreeze, set this low (~1/10–1/100 of the head LR). |
| `--warmup_steps` | int | `50` | LR warmup steps. |
| `--weight_decay` | float | `0.01` | Weight decay. |
| `--model_max_length` | int | `512` | Max token length. |
| `--optim` | str | `adamw_torch` | Optimizer. |
| `--fp16` | bool | `False` | FP16 training. (Use `--bf16` for bf16 — inherited from the base class.) |
| `--evaluation_strategy` | str | `steps` | When to evaluate. |
| `--eval_steps` | int | `100` | Eval interval (steps). |
| `--save_steps` | int | `100` | Checkpoint interval (steps). |
| `--logging_steps` | int | `100` | Logging interval (steps). |
| `--save_total_limit` | int | `3` | Max checkpoints kept. |
| `--load_best_model_at_end` | bool | `True` | Restore the best checkpoint at the end. |
| `--metric_for_best_model` | str | `spearman` | Selection metric (regression: Spearman calibration). |
| `--greater_is_better` | bool | `True` | Higher metric = better (matches `spearman`). |
| `--save_model` | bool | `False` | Save the final model with `safe_save_model_for_hf_trainer`. |
| `--eval_and_save_results` | bool | `True` | Evaluate on `test.csv` and write `eval_results.json` at the end. |
| `--seed` | int | `42` | RNG seed. |
| `--cache_dir` | str | `None` | HF cache dir. |
| `--dataloader_pin_memory` | bool | `False` | Pin dataloader memory. |
| `--find_unused_parameters` | bool | `False` | DDP find-unused-parameters. |

> The trainer sets `remove_unused_columns=False` internally so the twin's second window
> (`input_ids_alt`) and the privileged `depth` column survive to `model.forward`. The reported
> eval metrics are `mse`, `pearson`, `spearman`, and a threshold-free direction `auroc`
> (does `μ` rank hap1-favored loci, `y>0`, above hap2-favored).

### Example: Stage-2 frozen trunk

```bash
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path $HOME/entexBERT-2/DNABERT-2-117M-attention \
    --data_path experiments/stage2_CTCF/inputs \
    --task regression --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 1 --neff_s 50 \
    --init_backbone_from experiments/binding_reg_ENC-002_CTCF/runs/reg \
    --freeze_backbone True \
    --model_max_length 512 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --learning_rate 3e-5 --weight_decay 0.01 --warmup_steps 50 \
    --bf16 --metric_for_best_model spearman --greater_is_better True \
    --save_model True --output_dir experiments/stage2_CTCF/runs/asb_2a
```

### Example: Stage-2 gentle unfreeze (low LR, no `--freeze_backbone`)

```bash
python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path $HOME/entexBERT-2/DNABERT-2-117M-attention \
    --data_path experiments/stage2_CTCF/inputs \
    --task regression --input_mode hap_pair \
    --head_num_layers 1 --neff_s 50 \
    --init_backbone_from experiments/binding_reg_ENC-002_CTCF/runs/reg \
    --model_max_length 512 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 8 \
    --num_train_epochs 2 --learning_rate 3e-6 --weight_decay 0.01 --warmup_steps 50 \
    --bf16 --metric_for_best_model spearman --greater_is_better True \
    --save_model True --output_dir experiments/stage2_CTCF/runs/asb_2b
```
