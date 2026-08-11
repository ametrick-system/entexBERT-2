import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

## NEW IMPORTS #################################################
from scipy.stats import pearsonr, spearmanr # regression metrics
from entexbert2.model import entexBERT2ForSequencePrediction 
################################################################


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    # NEW: entexBERT-2 head / pooling architecture (used by entexBERT2ForSequencePrediction) ################
    pooling_mode: str = field(default="center_mean", metadata={"help": "'center_mean' or 'cls'"})
    center_pool_width: int = field(default=5, metadata={"help": "odd width of the center-mean pool window"})
    head_num_layers: int = field(default=1, metadata={"help": "1 = linear head, >=2 = MLP head g_phi"})
    head_hidden_size: int = field(default=-1, metadata={"help": "MLP hidden size (-1 = auto)"})
    head_activation: str = field(default="gelu")
    head_dropout: float = field(default=0.1)
    # NEW: privileged precision weighting w =  n(1+s)/(n+s) #################################################
    neff_s: float = field(default=50.0, metadata={"help": "n_eff saturation cap s (0 = unweighted)"})
    # NEW: 2-stage transfer learning ########################################################################
    init_backbone_from: Optional[str] = field(default=None,
        metadata={"help": "path to a Stage-1 checkpoint; loads backbone.* weights only"})
    freeze_backbone: bool = field(default=False,
        metadata={"help": "freeze the backbone (Stage 2: train the twin head only)"})
    #########################################################################################################

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    # NEW: regression task capabilities and streamline haplotype pair input format ########################
    task: str = field(default="regression", metadata={"help": "'regression' (only mode supported)"})
    input_mode: str = field(default="hap_pair", metadata={"help": "'hap_pair' (twin) or 'single'"})
    #######################################################################################################

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    # NEW: regression selects the best checkpoint by Spearman, greater is better!
    metric_for_best_model: str = field(default="spearman")
    greater_is_better: bool = field(default=True)
    #############################################################################

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1,
                 input_mode: str = "hap_pair"): # NEW: hap_pair reads two windows per example

        super(SupervisedDataset, self).__init__()

        # MODIFIED: load data from the disk ##############################################
             # read by column name: sequence1,sequence2,label[,depth])
             # original code inferred  single-vs-paired sequence input from the column count
        with open(data_path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{data_path} is empty.")
        cols = rows[0].keys()

        if input_mode == "hap_pair":
            if "sequence1" not in cols or "sequence2" not in cols:
                raise ValueError(f"hap_pair needs sequence1,sequence2 columns; got {list(cols)}.")
            logging.warning("Perform hap_pair twin regression...")
            texts = [[r["sequence1"], r["sequence2"]] for r in rows]
        else:
            seqcol = "sequence" if "sequence" in cols else "sequence1"
            logging.warning("Perform single-sequence regression...")
            texts = [r[seqcol] for r in rows]

        # NEW: float labels for regression tasks
        labels = [float(r["label"]) for r in rows]
        # NEW: optional privileged depth column n (the precision weight)
        self.depth = [float(r["depth"]) for r in rows] if "depth" in cols else None
        ############################################################################################

        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")

            # MODIFIED: k-merize *each* window of the pair (or the single sequence)
            if input_mode == "hap_pair":
                flat = [s for pair in texts for s in pair]
                flat = load_or_generate_kmer(data_path, flat, kmer)
                texts = [[flat[2 * i], flat[2 * i + 1]] for i in range(len(texts))]
            else:
                texts = load_or_generate_kmer(data_path, texts, kmer)
            #######################################################################

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        # MODIFIED: for hap_pair, tokenize the two windows SEPARATELY rather than [SEP]-concatenating
        self.input_mode = input_mode
        if input_mode == "hap_pair":
            enc1 = tokenizer([t[0] for t in texts], return_tensors="pt", padding="longest",
                             max_length=tokenizer.model_max_length, truncation=True)
            enc2 = tokenizer([t[1] for t in texts], return_tensors="pt", padding="longest",
                             max_length=tokenizer.model_max_length, truncation=True)
            self.input_ids = enc1["input_ids"]
            self.attention_mask = enc1["attention_mask"]
            self.input_ids_alt = enc2["input_ids"]
            self.attention_mask_alt = enc2["attention_mask"]
        else:
            enc = tokenizer(texts, return_tensors="pt", padding="longest",
                            max_length=tokenizer.model_max_length, truncation=True)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.input_ids_alt = None
            self.attention_mask_alt = None

        self.labels = labels
        self.num_labels = 1 # NEW: regression -> single output
        ############################################################################################

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # MODIFIED: emit the twin pair + float label (+ depth weight when present)
        item = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.input_ids_alt is not None:
            item["input_ids_alt"] = self.input_ids_alt[i]
        if self.depth is not None:
            item["depth"] = self.depth[i]
        return item
        ###########################################################################

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # MODIFIED: float labels (regression target) instead of .long() class ids
        batch = dict(
            input_ids=input_ids,
            labels=torch.tensor(labels, dtype=torch.float),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # NEW: pad + attach the second (alt) window when present (twin input)
        if "input_ids_alt" in instances[0]:
            alt = torch.nn.utils.rnn.pad_sequence(
                [instance["input_ids_alt"] for instance in instances],
                batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["input_ids_alt"] = alt
            batch["attention_mask_alt"] = alt.ne(self.tokenizer.pad_token_id)
        # NEW: carry the privileged depth weight through to the loss (crucially NOT the model input)
        if "depth" in instances[0]:
            batch["depth"] = torch.tensor([instance["depth"] for instance in instances],
                                          dtype=torch.float)
        return batch

# """
# Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
# """
# def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
#     valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
#     valid_predictions = predictions[valid_mask]
#     valid_labels = labels[valid_mask]
#     return {
#         "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
#         "f1": sklearn.metrics.f1_score(
#             valid_labels, valid_predictions, average="macro", zero_division=0
#         ),
#         "matthews_correlation": sklearn.metrics.matthews_corrcoef(
#             valid_labels, valid_predictions
#         ),
#         "precision": sklearn.metrics.precision_score(
#             valid_labels, valid_predictions, average="macro", zero_division=0
#         ),
#         "recall": sklearn.metrics.recall_score(
#             valid_labels, valid_predictions, average="macro", zero_division=0
#         ),
#     }

# NEW: regression metrics ######################################################################
def calculate_regression_metrics(predictions: np.ndarray, labels: np.ndarray):
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    out = {
        "mse": float(np.mean((pred - y) ** 2)),
        "pearson": float(pearsonr(pred, y)[0]) if len(y) > 2 else float("nan"),
        "spearman": float(spearmanr(pred, y).correlation) if len(y) > 2 else float("nan"),
    }
    direction = (y > 0).astype(int) # hap1-favored vs hap2-favored
    if 0 < direction.sum() < len(direction):
        out["auroc"] = float(sklearn.metrics.roc_auc_score(direction, pred))
    else:
        out["auroc"] = float("nan")
    return out
##############################################################################################

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
# def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
#     if isinstance(logits, tuple):  # Unpack logits if it's a tuple
#         logits = logits[0]

#     if logits.ndim == 3:
#         # Reshape logits to 2D if needed
#         logits = logits.reshape(-1, logits.shape[-1])

#     return torch.argmax(logits, dim=-1)

# MODIFIED: for regression, logits ARE the prediction
def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.reshape(-1) # flatten: (N, 1) -> (N,)

"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_regression_metrics(predictions, labels) # MODIFIED: regression instead of classification

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NEW: the twin/depth columns are non-standard Trainer inputs -- make sure we keep them!
    training_args.remove_unused_columns = False
    ########################################################################################

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator [MODIFIED: added input_mode=data_args.input_mode for hap_pair functionality]]
        train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=os.path.join(data_args.data_path, "train.csv"),
                                      kmer=data_args.kmer, input_mode=data_args.input_mode)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer, input_mode=data_args.input_mode)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer, input_mode=data_args.input_mode)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # MODIFIED: load entexBERT-2 model
    model = entexBERT2ForSequencePrediction(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        pooling_mode=model_args.pooling_mode,
        center_pool_width=model_args.center_pool_width,
        head_num_layers=model_args.head_num_layers,
        head_hidden_size=model_args.head_hidden_size,
        head_activation=model_args.head_activation,
        head_dropout=model_args.head_dropout,
        neff_s=model_args.neff_s,
    )

    # NEW: 2-stage transfer ################################################################################
        # Load a Stage-1 binding trunk (backbone only) -> optionally freeze it for Stage-2 (twin head train)
    if model_args.init_backbone_from:
        model.init_backbone_from(model_args.init_backbone_from)
    if model_args.freeze_backbone:
        model.freeze_backbone()
    ########################################################################################################

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # NEW: persist the architecture so model_io.py can rebuild the model exactly for scoring ###
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "run_config.json"), "w") as f:
        json.dump({
            "model_name_or_path": model_args.model_name_or_path,
            "cache_dir": training_args.cache_dir,
            "model_max_length": training_args.model_max_length,
            "pooling_mode": model_args.pooling_mode,
            "center_pool_width": model_args.center_pool_width,
            "head_num_layers": model_args.head_num_layers,
            "head_hidden_size": model_args.head_hidden_size,
            "head_activation": model_args.head_activation,
            "head_dropout": model_args.head_dropout,
            "neff_s": model_args.neff_s,
            "task": data_args.task,
            "input_mode": data_args.input_mode,
            "use_lora": model_args.use_lora,
        }, f, indent=2)
    ##############################################################################################

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()