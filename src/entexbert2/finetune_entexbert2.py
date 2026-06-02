'''
This script is a modified version of the DNABERT-2 finetuning script,
found at https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py

This modified version supports:
- Continuous label prediction via a linear regression head
- Incorporation of auxiliary tasks during training following the LUPI framework

All modifications to the original script are wrapped in comments in the following format:
### NEW: [description of modification] ###
...
##########################################

Last modified: 6/2/2026 by Amy Metrick
'''

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

############### NEW: IMPORTS ###############
from functools import partial
from scipy.stats import pearsonr, spearmanr
from transformers.modeling_outputs import SequenceClassifierOutput
############################################

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


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

    ######################### NEW: MAIN TASK (with regression and MLP support) #######################
    task: str = field(
        default="regression",
        metadata={"help": "Main task type: 'classification' or 'regression'"}
    )

    main_num_labels: int = field(
        default=1,
        metadata={
            "help": (
                "Number of output labels for the main head. "
                "Use 1 for regression, 1 for binary BCE classification, "
                "2 for binary softmax classification, and >2 for multiclass."
            )
        }
    )

    head_num_layers: int = field(
        default=1,
        metadata={
            "help": (
                "Total number of Linear layers in the main prediction head. "
                "1 means a simple linear head. >1 means an MLP head."
            )
        },
    )

    head_hidden_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Hidden size for MLP prediction head. "
                "If -1, use the backbone hidden size."
            )
        },
    )

    head_activation: str = field(
        default="gelu",
        metadata={
            "help": "Activation function for MLP prediction head: 'gelu', 'relu', 'tanh', or 'silu'."
        },
    )

    head_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability used inside the MLP prediction head."
        },
    )
    ##################################################################################################

    ############################## NEW: LUPI / AUX TASKS ################################
    num_aux_tasks: int = field(
        default=0,
        metadata={"help": "Number of auxiliary tasks. 0 means no auxiliary supervision."}
    )

    aux_task_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Names of auxiliary tasks, one per auxiliary head."}
    )

    aux_task_types: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Type of each auxiliary task. "
                "Supported values per head: 'binary', 'multiclass', 'regression'."
            )
        }
    )

    aux_num_labels: List[int] = field(
        default_factory=list,
        metadata={
            "help": (
                "Number of output labels/classes for each auxiliary head. "
                "Use 1 for regression, 1 for binary BCE-style classification, "
                "and >1 for multiclass classification."
            )
        }
    )

    lambda_aux: List[float] = field(
        default_factory=list,
        metadata={
            "help": "Loss weight for each auxiliary task, one per auxiliary head."
        }
    )
    #####################################################################################

    ############################## NEW: ENABLE CENTER/MEAN POOLING ###############################
    pooling_mode: str = field(
        default="cls",
        metadata={"help": "How to pool token embeddings: 'cls' or 'center_mean'."}
    )

    center_pool_width: int = field(
        default=5,
        metadata={"help": "Number of center tokens to mean-pool when pooling_mode='center_mean'."}
    )
    ###############################################################################################

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
    #### EDITED (indicated with NEW) to support optional LUPI auxiliary labels ####

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
        task: str = "classification",
        aux_task_names: Optional[List[str]] = None,
        aux_task_types: Optional[List[str]] = None,
    ): # NEW: ADD TASK, AUX INFO

        super(SupervisedDataset, self).__init__()

        ##################### NEW: INITIALIZE TASK, AUX INFO ########################
        self.task = task
        self.aux_task_names = aux_task_names or []
        self.aux_task_types = aux_task_types or []

        # ensure task name list is same size as task type list
        if len(self.aux_task_names) != len(self.aux_task_types):
            raise ValueError(
                f"aux_task_names and aux_task_types must have the same length, got "
                f"{len(self.aux_task_names)} and {len(self.aux_task_types)}"
            )
        #############################################################################

        # load data from the disk
        with open(data_path, "r") as f:
            reader = csv.DictReader(f) # NEW: use DictReader to parse column names
            data = list(reader)
        
        ##### NEW: THE BELOW STRUCTURE IS REWORKED USING DictReader TO HANDLE NEW INPUT CSV FORMAT #####
        if len(data) == 0:
            raise ValueError(f"No rows found in {data_path}")
        
        # detect whether this is single-sequence or sequence-pair
        has_sequence = "sequence" in data[0]
        has_sequence_pair = "sequence1" in data[0] and "sequence2" in data[0]

        if has_sequence:
            logging.warning("Perform single-sequence prediction...")
            texts = [row["sequence"] for row in data]
        elif has_sequence_pair:
            logging.warning("Perform sequence-pair prediction...")
            texts = [[row["sequence1"], row["sequence2"]] for row in data]
        else:
            raise ValueError(
                "CSV must contain either 'sequence' column or both 'sequence1' and 'sequence2' columns."
            )
        
        # main labels
        if "label" not in data[0]:
            raise ValueError("CSV must contain a 'label' column for the main task.")

        if task == "regression":
            labels = [float(row["label"]) for row in data]
        elif task == "classification":
            labels = [int(row["label"]) for row in data]
        else:
            raise ValueError(f"Unsupported main task: {task}")
        
        # auxiliary labels
        aux_labels = []
        for row in data:
            row_aux = []
            for aux_name, aux_type in zip(self.aux_task_names, self.aux_task_types):
                if aux_name not in row:
                    raise ValueError(
                        f"Auxiliary task column '{aux_name}' not found in {data_path}"
                    )

                raw_val = row[aux_name]

                if aux_type == "regression":
                    row_aux.append(float(raw_val))
                elif aux_type == "binary":
                    # BCEWithLogitsLoss expects float labels
                    row_aux.append(float(raw_val))
                elif aux_type == "multiclass":
                    row_aux.append(int(raw_val))
                else:
                    raise ValueError(
                        f"Unsupported aux task type '{aux_type}' for aux task '{aux_name}'"
                    )
            aux_labels.append(row_aux)
        ########################################################################################
        
        if kmer != -1:
            if torch.distributed.is_available() and torch.distributed.is_initialized(): # NEW: added for robustness in case of using multiple GPUs
                if torch.distributed.get_rank() not in [0, -1]:
                    torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_available() and torch.distributed.is_initialized(): # NEW (see above)
                if torch.distributed.get_rank() == 0:
                    torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.aux_labels = aux_labels # NEW

        ######### NEW: SET NUM_LABELS TO 1 FOR REGRESSION TASK ##########
        # Stored only for dataset inspection/debugging
        # The model head size is controlled by training_args.main_num_labels
        self.num_labels = 1 if task == "regression" else len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Any]: # NEW: change return type to Any
        #### NEW: SUPPORT FOR AUX LABELS ####
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }

        if len(self.aux_task_names) > 0:
            item["aux_labels"] = self.aux_labels[i]

        return item

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    #### EDITED (indicated with NEW) to support optional LUPI auxiliary labels ####

    tokenizer: transformers.PreTrainedTokenizer
    ######### NEW: KEEP TRACK OF MAIN & AUX TASK TYPES #############
    main_task: str = "classification"
    aux_task_types: Optional[List[str]] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]: # NEW: use Any, since batch will contain tensors
        # NOTE: altered structure to support aux functionality
        aux_task_types = self.aux_task_types or [] # NEW
        
        # input_ids
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # NEW: ATTENTION MASK PADDING TO ALLOW ATTENTION MASK TO BE PASSED THROUGH __getitem__
        attention_mask = [instance["attention_mask"] for instance in instances]
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        # main labels
        labels = [instance["labels"] for instance in instances]
        main_dtype = torch.float if self.main_task == "regression" else torch.long
        labels = torch.tensor(labels, dtype=main_dtype)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        # auxiliary labels
        if len(aux_task_types) > 0:
            if "aux_labels" not in instances[0]:
                raise ValueError(
                    "aux_task_types were provided to the collator, "
                    "but dataset instances do not contain 'aux_labels'."
                )

            num_aux_tasks = len(aux_task_types)
            batched_aux_labels = []

            for aux_idx in range(num_aux_tasks):
                aux_values = [instance["aux_labels"][aux_idx] for instance in instances]
                aux_type = aux_task_types[aux_idx]

                if aux_type in {"binary", "regression"}:
                    aux_tensor = torch.tensor(aux_values, dtype=torch.float)
                elif aux_type == "multiclass":
                    aux_tensor = torch.tensor(aux_values, dtype=torch.long)
                else:
                    raise ValueError(f"Unsupported auxiliary task type: {aux_type}")

                batched_aux_labels.append(aux_tensor)

            batch["aux_labels"] = batched_aux_labels

        return batch

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    ##### EDITED TO BE ROBUST TO DIFFERENT SHAPES #####
    # ensure correct shapes
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)

    # exclude padding tokens if present
    valid_mask = labels != -100

    if valid_mask.sum() == 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "matthews_correlation": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    # ensure integer type for classification metrics
    valid_predictions = valid_predictions.astype(int)
    valid_labels = valid_labels.astype(int)
    ####################################################

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# (original version) from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
# NEW: EDITED FOR STABILITY IN AUX TASK ARCHITECTURE
def preprocess_logits_for_metrics(
    main_task: str,
    logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
    labels,
):
    # If model output is a tuple, take the first element as the main logits
    if isinstance(logits, tuple):
        logits = logits[0]

    if main_task == "regression":
        # expected shape: [batch] or [batch, 1]
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits
    
    elif main_task == "classification":
        # Sequence-labeling-style logits, if ever present
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])

        # Single-logit binary classification: make shape [batch] so compute_metrics uses the sigmoid/AUROC path
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return logits

    else:
        raise ValueError(f"Unsupported main task: {main_task}")

"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(main_task: str, eval_pred):
    predictions, labels = eval_pred

    # Hugging Face may pass multiple label arrays as a tuple
    # when the batch contains both labels and aux_labels.
    # We only want the main task labels.
    if isinstance(labels, tuple):
        labels = labels[0]

    # Safely squeeze trailing singleton dimensions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim > 1 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if labels.ndim > 1 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)

    # NEW: REGRESSION METRICS
    if main_task == "regression":
        labels = np.asarray(labels, dtype=float)
        predictions = np.asarray(predictions, dtype=float)

        # Ignore NaNs and infs if they appear
        finite_mask = np.isfinite(labels) & np.isfinite(predictions)

        if finite_mask.sum() == 0:
            return {
                "mse": float("nan"),
                "r2": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
            }

        labels = labels[finite_mask]
        predictions = predictions[finite_mask]

        mse = sklearn.metrics.mean_squared_error(labels, predictions)
        r2 = sklearn.metrics.r2_score(labels, predictions)

        # Guard against constant arrays / tiny eval sets
        if len(labels) > 1 and np.std(labels) > 0 and np.std(predictions) > 0:
            pearson = pearsonr(labels, predictions)[0]
            spearman = spearmanr(labels, predictions)[0]
        else:
            pearson = 0.0
            spearman = 0.0

        return {
            "mse": mse,
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
        }

    elif main_task == "classification":
        labels = np.asarray(labels)

        # binary / multiclass logits
        if predictions.ndim == 2:
            pred_classes = np.argmax(predictions, axis=-1)
            metrics = calculate_metric_with_sklearn(pred_classes, labels)

            # add AUROC for binary classification only
            if predictions.shape[1] == 2:
                probs = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                probs = probs / np.sum(probs, axis=1, keepdims=True)
                pos_probs = probs[:, 1]

                if len(np.unique(labels)) == 2:
                    metrics["auroc"] = sklearn.metrics.roc_auc_score(labels, pos_probs)
                else:
                    metrics["auroc"] = 0.0

            return metrics

        # single-logit binary classification
        elif predictions.ndim == 1:
            pos_probs = 1.0 / (1.0 + np.exp(-predictions))
            pred_classes = (pos_probs >= 0.5).astype(int)
            metrics = calculate_metric_with_sklearn(pred_classes, labels)

            if len(np.unique(labels)) == 2:
                metrics["auroc"] = sklearn.metrics.roc_auc_score(labels, pos_probs)
            else:
                metrics["auroc"] = 0.0

            return metrics

        else:
            raise ValueError(f"Unexpected classification prediction shape: {predictions.shape}")

    else:
        raise ValueError(f"Unsupported main task: {main_task}")

############################### NEW: HELPER FUNCTIONS TO CONSTRUCT MLP HEAD ################################
def get_activation_module(name: str) -> torch.nn.Module:
    """
    Return activation module from string name
    """
    name = name.lower()

    if name == "gelu":
        return torch.nn.GELU()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(
            f"Unsupported activation {name!r}. "
            "Choose from {'gelu', 'relu', 'tanh', 'silu'}."
        )

def build_prediction_head(
    input_size: int,
    output_size: int,
    num_layers: int = 1,
    hidden_size: int = -1,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    Build either a linear prediction head or an MLP prediction head

    num_layers counts total Linear layers.
      - num_layers=1: Linear(input_size -> output_size)
      - num_layers=2: Linear(input_size -> hidden_size) + activation/dropout + Linear(hidden_size -> output_size)
      - num_layers=3+: deeper MLP
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

    if hidden_size == -1:
        hidden_size = input_size

    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive or -1, got {hidden_size}.")

    if dropout < 0 or dropout >= 1:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    # Original behavior: simple linear head.
    if num_layers == 1:
        return torch.nn.Linear(input_size, output_size)

    layers = []

    # First hidden layer
    layers.append(torch.nn.Linear(input_size, hidden_size))
    layers.append(get_activation_module(activation))
    layers.append(torch.nn.Dropout(dropout))

    # Additional hidden layers
    for _ in range(num_layers - 2):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(get_activation_module(activation))
        layers.append(torch.nn.Dropout(dropout))

    # Final output layer
    layers.append(torch.nn.Linear(hidden_size, output_size))

    return torch.nn.Sequential(*layers)
############################################################################################################

####### NEW: FUNCTION TO ENSURE ALL TRAINING ARGS ARE VALID #######
def validate_training_args(training_args):
    allowed_main_tasks = {"classification", "regression"}
    allowed_aux_task_types = {"binary", "multiclass", "regression"}
    allowed_pooling_modes = {"cls", "center_mean"}
    allowed_head_activations = {"gelu", "relu", "tanh", "silu"}

    if training_args.task not in allowed_main_tasks:
        raise ValueError(
            f"Unsupported main task {training_args.task!r}. "
            f"Choose from {allowed_main_tasks}."
        )

    if training_args.task == "regression" and training_args.main_num_labels != 1:
        raise ValueError("For regression, main_num_labels must be 1.")

    if training_args.task == "classification" and training_args.main_num_labels < 1:
        raise ValueError("For classification, main_num_labels must be >= 1.")

    # Validate pooling regardless of whether LUPI is used.
    if training_args.pooling_mode not in allowed_pooling_modes:
        raise ValueError(
            f"Unsupported pooling_mode {training_args.pooling_mode!r}. "
            f"Choose from {allowed_pooling_modes}."
        )

    if training_args.center_pool_width < 1:
        raise ValueError("center_pool_width must be >= 1.")

    if training_args.pooling_mode == "center_mean" and training_args.center_pool_width % 2 == 0:
        raise ValueError(
            "center_pool_width should be odd for symmetric center pooling."
        )

    if training_args.head_num_layers < 1:
        raise ValueError("head_num_layers must be >= 1.")

    if training_args.head_hidden_size != -1 and training_args.head_hidden_size <= 0:
        raise ValueError("head_hidden_size must be positive or -1.")

    if training_args.head_activation.lower() not in allowed_head_activations:
        raise ValueError(
            f"Unsupported head_activation {training_args.head_activation!r}. "
            f"Choose from {allowed_head_activations}."
        )

    if training_args.head_dropout < 0 or training_args.head_dropout >= 1:
        raise ValueError("head_dropout must be in [0, 1).")

    if training_args.num_aux_tasks < 0:
        raise ValueError("num_aux_tasks must be >= 0.")

    if training_args.num_aux_tasks == 0:
        if any([
            training_args.aux_task_names,
            training_args.aux_task_types,
            training_args.aux_num_labels,
            training_args.lambda_aux,
        ]):
            raise ValueError(
                "num_aux_tasks is 0, but auxiliary task configuration lists are non-empty."
            )
        return

    if len(training_args.aux_task_names) != training_args.num_aux_tasks:
        raise ValueError(
            f"num_aux_tasks={training_args.num_aux_tasks}, but got "
            f"{len(training_args.aux_task_names)} aux_task_names."
        )

    if len(training_args.aux_task_types) != training_args.num_aux_tasks:
        raise ValueError(
            f"num_aux_tasks={training_args.num_aux_tasks}, but got "
            f"{len(training_args.aux_task_types)} aux_task_types."
        )

    if len(training_args.aux_num_labels) != training_args.num_aux_tasks:
        raise ValueError(
            f"num_aux_tasks={training_args.num_aux_tasks}, but got "
            f"{len(training_args.aux_num_labels)} aux_num_labels."
        )

    if len(training_args.lambda_aux) != training_args.num_aux_tasks:
        raise ValueError(
            f"num_aux_tasks={training_args.num_aux_tasks}, but got "
            f"{len(training_args.lambda_aux)} lambda_aux values."
        )

    for i, (task_type, num_labels, weight) in enumerate(
        zip(
            training_args.aux_task_types,
            training_args.aux_num_labels,
            training_args.lambda_aux,
        )
    ):
        if task_type not in allowed_aux_task_types:
            raise ValueError(
                f"Unsupported aux_task_types[{i}]={task_type!r}. "
                f"Choose from {allowed_aux_task_types}."
            )

        if task_type in {"binary", "regression"} and num_labels != 1:
            raise ValueError(
                f"Aux task {i} has type {task_type!r}, so aux_num_labels[{i}] must be 1, got {num_labels}."
            )

        if task_type == "multiclass" and num_labels < 2:
            raise ValueError(
                f"Aux task {i} is multiclass, so aux_num_labels[{i}] must be >= 2, got {num_labels}."
            )

        if weight < 0:
            raise ValueError(f"lambda_aux[{i}] must be nonnegative, got {weight}.")

####### NEW: FUNCTION TO PRINT NUMBER OF TRAINABLE PARAMETERS #######
def print_trainable_parameters(model):
    """
    Print the number and percentage of trainable parameters.
    Useful for checking whether full fine-tuning vs. LoRA is configured correctly.
    """
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

# !!! NEW: entexBERT-2 training class !!!
class entexBERT2ForSequencePrediction(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        main_task: str = "classification",
        main_num_labels: int = 2,
        aux_task_names: Optional[List[str]] = None,
        aux_task_types: Optional[List[str]] = None,
        aux_num_labels: Optional[List[int]] = None,
        lambda_aux: Optional[List[float]] = None,
        pooling_mode: str = "cls",
        center_pool_width: int = 5,
        head_num_layers: int = 1,
        head_hidden_size: int = -1,
        head_activation: str = "gelu",
        head_dropout: float = 0.1,
    ):
        super().__init__()

        self.main_task = main_task
        self.main_num_labels = main_num_labels

        self.aux_task_names = aux_task_names or []
        self.aux_task_types = aux_task_types or []
        self.aux_num_labels = aux_num_labels or []
        self.lambda_aux = lambda_aux or []

        self.pooling_mode = pooling_mode
        self.center_pool_width = center_pool_width

        self.head_num_layers = head_num_layers
        self.head_hidden_size = head_hidden_size
        self.head_activation = head_activation
        self.head_dropout = head_dropout

        if not (
            len(self.aux_task_names)
            == len(self.aux_task_types)
            == len(self.aux_num_labels)
            == len(self.lambda_aux)
        ):
            raise ValueError(
                "aux_task_names, aux_task_types, aux_num_labels, and lambda_aux "
                "must all have the same length."
            )

        # Shared pretrained backbone
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)

        # Main head
        self.main_head = build_prediction_head(
            input_size=hidden_size,
            output_size=main_num_labels,
            num_layers=head_num_layers,
            hidden_size=head_hidden_size,
            activation=head_activation,
            dropout=head_dropout,
        )

        # Auxiliary heads
        self.aux_heads = torch.nn.ModuleDict()
        for name, num_labels in zip(self.aux_task_names, self.aux_num_labels):
            self.aux_heads[name] = torch.nn.Linear(hidden_size, num_labels)
    
    def _pool_sequence_representation(self, backbone_outputs, attention_mask=None):
        """
        Pool token-level representations.
        Supported modes:
          - cls: use first token
          - center_mean: mean-pool a window around the center of the sequence
        """

        # Extract token-level hidden states
        if hasattr(backbone_outputs, "last_hidden_state") and backbone_outputs.last_hidden_state is not None:
            sequence_output = backbone_outputs.last_hidden_state

        elif isinstance(backbone_outputs, dict) and "last_hidden_state" in backbone_outputs:
            sequence_output = backbone_outputs["last_hidden_state"]

        elif isinstance(backbone_outputs, (tuple, list)):
            first = backbone_outputs[0]
            if torch.is_tensor(first) and first.ndim == 3:
                sequence_output = first
            else:
                raise ValueError(f"Cannot extract token-level hidden states from output type {type(backbone_outputs)}")

        else:
            raise ValueError(f"Cannot extract token-level hidden states from output type {type(backbone_outputs)}")

        if self.pooling_mode == "cls":
            return sequence_output[:, 0, :]

        elif self.pooling_mode == "center_mean":
            if self.center_pool_width % 2 == 0:
                raise ValueError("center_pool_width should be odd for symmetric center pooling.")

            batch_size, max_seq_len, hidden_size = sequence_output.shape
            half = self.center_pool_width // 2

            pooled_outputs = []

            for b in range(batch_size):
                if attention_mask is not None:
                    # Number of non-padding tokens for this example
                    valid_len = int(attention_mask[b].sum().item())
                else:
                    valid_len = max_seq_len

                valid_len = max(valid_len, 1) # To be extra safe :)

                center = valid_len // 2
                start = max(0, center - half)
                end = min(valid_len, center + half + 1)

                pooled_b = sequence_output[b, start:end, :].mean(dim=0)
                pooled_outputs.append(pooled_b)

            return torch.stack(pooled_outputs, dim=0)

        else:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")

    def _compute_main_loss(self, logits, labels):
        if self.main_task == "regression":
            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            labels = labels.float()
            return torch.nn.functional.mse_loss(logits, labels)

        elif self.main_task == "classification":
            labels = labels.long()

            # Binary classification with single logit
            if self.main_num_labels == 1:
                if logits.ndim > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                labels = labels.float()
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            # Multiclass classification
            return torch.nn.functional.cross_entropy(logits, labels)

        else:
            raise ValueError(f"Unsupported main task: {self.main_task}")

    def _compute_aux_loss(self, aux_name, aux_type, aux_logits, aux_labels):
        if aux_type == "regression":
            if aux_logits.ndim > 1 and aux_logits.shape[-1] == 1:
                aux_logits = aux_logits.squeeze(-1)
            aux_labels = aux_labels.float()
            return torch.nn.functional.mse_loss(aux_logits, aux_labels)

        elif aux_type == "binary":
            if aux_logits.ndim > 1 and aux_logits.shape[-1] == 1:
                aux_logits = aux_logits.squeeze(-1)
            aux_labels = aux_labels.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(aux_logits, aux_labels)

        elif aux_type == "multiclass":
            aux_labels = aux_labels.long()
            return torch.nn.functional.cross_entropy(aux_logits, aux_labels)

        else:
            raise ValueError(f"Unsupported auxiliary task type for {aux_name}: {aux_type}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        aux_labels=None,
        **kwargs,
    ):
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        pooled = self._pool_sequence_representation(
            backbone_outputs,
            attention_mask=attention_mask,
        )
        pooled = self.dropout(pooled)

        # Main task logits
        logits = self.main_head(pooled)

        # Auxiliary logits
        aux_logits_dict = {}
        for name in self.aux_task_names:
            aux_logits_dict[name] = self.aux_heads[name](pooled)

        total_loss = None
        loss_dict = {}

        # Main loss
        if labels is not None:
            main_loss = self._compute_main_loss(logits, labels)
            total_loss = main_loss
            loss_dict["main_loss"] = main_loss.detach()

        # Auxiliary losses (for training ONLY)
        if aux_labels is not None and self.training:
            if len(aux_labels) != len(self.aux_task_names):
                raise ValueError(
                    f"Expected {len(self.aux_task_names)} auxiliary label tensors, "
                    f"but got {len(aux_labels)}."
                )

            for i, (name, task_type, weight) in enumerate(
                zip(self.aux_task_names, self.aux_task_types, self.lambda_aux)
            ):
                this_aux_logits = aux_logits_dict[name]
                this_aux_labels = aux_labels[i]

                aux_loss = self._compute_aux_loss(
                    aux_name=name,
                    aux_type=task_type,
                    aux_logits=this_aux_logits,
                    aux_labels=this_aux_labels,
                )

                loss_dict[f"{name}_loss"] = aux_loss.detach()

                weighted_aux_loss = weight * aux_loss
                total_loss = weighted_aux_loss if total_loss is None else total_loss + weighted_aux_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
        )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    validate_training_args(training_args) # NEW
    training_args.label_names = ["labels"] # NEW

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

    # define datasets and data collator
    # NEW: modified to include main task, aux task info
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=data_args.kmer,
        task=training_args.task,
        aux_task_names=training_args.aux_task_names,
        aux_task_types=training_args.aux_task_types,
    )

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer,
        task=training_args.task,
        aux_task_names=training_args.aux_task_names,
        aux_task_types=training_args.aux_task_types,
    )

    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
        task=training_args.task,
        aux_task_names=training_args.aux_task_names,
        aux_task_types=training_args.aux_task_types,
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        main_task=training_args.task,
        aux_task_types=training_args.aux_task_types,
    )

    #### NEW: ALTERED FORMAT FOR MODEL LOADING ####

    # main head output dimension
    if training_args.task == "regression":
        main_num_labels = 1
    else:
        main_num_labels = training_args.main_num_labels

    # load entexBERT-2 model
    model = entexBERT2ForSequencePrediction(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        main_task=training_args.task,
        main_num_labels=main_num_labels,
        aux_task_names=training_args.aux_task_names,
        aux_task_types=training_args.aux_task_types,
        aux_num_labels=training_args.aux_num_labels,
        lambda_aux=training_args.lambda_aux,
        pooling_mode=training_args.pooling_mode,
        center_pool_width=training_args.center_pool_width,
        head_num_layers=training_args.head_num_layers,
        head_hidden_size=training_args.head_hidden_size,
        head_activation=training_args.head_activation,
        head_dropout=training_args.head_dropout,
    )

    # configure LoRA on the backbone only
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
        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.print_trainable_parameters()
    
    print_trainable_parameters(model) # NEW: print trainable parameters in full model after LoRA wrapping

    print(
        f"main task={training_args.task!r}, "
        f"num_aux_tasks={training_args.num_aux_tasks}, "
        f"aux_task_names={training_args.aux_task_names}, "
        f"pooling_mode={training_args.pooling_mode!r}, "
        f"center_pool_width={training_args.center_pool_width}, "
        f"head_num_layers={training_args.head_num_layers}, "
        f"head_hidden_size={training_args.head_hidden_size}, "
        f"head_activation={training_args.head_activation!r}, "
        f"head_dropout={training_args.head_dropout}"
    )

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics, training_args.task),
                                   compute_metrics=partial(compute_metrics, training_args.task),
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