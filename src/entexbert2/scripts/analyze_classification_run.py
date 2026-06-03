#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze an entexBERT-2 classification run: predictions + pooled embeddings + PCA."
    )

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--model_name_or_path",
        default="zhihan1996/DNABERT-2-117M",
    )

    parser.add_argument("--input_mode", default="hap_pair", choices=["ref_single", "hap_pair"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_max_length", type=int, default=512)

    # Must match training run
    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--center_pool_width", type=int, default=5)
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def find_best_or_final_model_file(checkpoint_dir):
    """
    Find model weights saved by Hugging Face Trainer.

    Checks:
      1. trainer_state.json best_model_checkpoint
      2. checkpoint_dir/model.safetensors or pytorch_model.bin
      3. latest checkpoint-* directory
    """
    checkpoint_dir = Path(checkpoint_dir)

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state = json.load(f)

        best_ckpt = state.get("best_model_checkpoint")
        if best_ckpt is not None:
            best_ckpt = Path(best_ckpt)
            for fname in ["model.safetensors", "pytorch_model.bin"]:
                candidate = best_ckpt / fname
                if candidate.exists():
                    return candidate

    for fname in ["model.safetensors", "pytorch_model.bin"]:
        candidate = checkpoint_dir / fname
        if candidate.exists():
            return candidate

    checkpoint_paths = []
    for p in checkpoint_dir.glob("checkpoint-*"):
        match = re.search(r"checkpoint-(\d+)$", str(p))
        if match:
            checkpoint_paths.append((int(match.group(1)), p))

    if checkpoint_paths:
        checkpoint_paths.sort()
        latest = checkpoint_paths[-1][1]
        for fname in ["model.safetensors", "pytorch_model.bin"]:
            candidate = latest / fname
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir}")


def load_model_weights(model, model_file, device):
    model_file = Path(model_file)

    if model_file.name.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed. "
                "Install with `pip install safetensors`."
            ) from exc

        state_dict = load_file(str(model_file), device=device)

    else:
        state_dict = torch.load(str(model_file), map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {model_file}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])


def get_activation_module(name):
    name = name.lower()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def get_sequence_output(backbone_outputs):
    if isinstance(backbone_outputs, (tuple, list)):
        return backbone_outputs[0]
    return backbone_outputs.last_hidden_state


def pool_sequence(model, sequence_output, attention_mask):
    """
    Reproduce the model's pooling behavior for analysis.
    """
    if model.pooling_mode == "cls":
        return sequence_output[:, 0, :]

    if model.pooling_mode == "center_mean":
        batch_size, max_seq_len, hidden_size = sequence_output.shape
        half = model.center_pool_width // 2

        pooled_outputs = []

        for b in range(batch_size):
            if attention_mask is not None:
                valid_len = int(attention_mask[b].sum().item())
            else:
                valid_len = max_seq_len

            valid_len = max(valid_len, 1)

            center = valid_len // 2
            start = max(0, center - half)
            end = min(valid_len, center + half + 1)

            pooled_b = sequence_output[b, start:end, :].mean(dim=0)
            pooled_outputs.append(pooled_b)

        return torch.stack(pooled_outputs, dim=0)

    raise ValueError(f"Unsupported pooling mode: {model.pooling_mode}")


def tokenize_batch(tokenizer, df_batch, input_mode, model_max_length, device):
    if input_mode == "hap_pair":
        enc = tokenizer(
            df_batch["sequence1"].astype(str).tolist(),
            df_batch["sequence2"].astype(str).tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_length,
        )
    else:
        enc = tokenizer(
            df_batch["sequence"].astype(str).tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_length,
        )

    return {k: v.to(device) for k, v in enc.items()}


def confusion_category(y_true, y_pred):
    cats = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            cats.append("TP")
        elif t == 0 and p == 0:
            cats.append("TN")
        elif t == 0 and p == 1:
            cats.append("FP")
        elif t == 1 and p == 0:
            cats.append("FN")
        else:
            cats.append("NA")
    return cats


def compute_metrics(labels, probs, preds):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }

    try:
        metrics["auroc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["aupr"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["aupr"] = float("nan")

    return metrics


def plot_pca(embeddings, pred_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    pca_df = pred_df.copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]

    pca_df.to_csv(output_dir / "pca_coordinates.csv", index=False)

    # Plot by true label
    fig, ax = plt.subplots(figsize=(7, 6))

    for label_value, group in pca_df.groupby("label"):
        ax.scatter(
            group["PC1"],
            group["PC2"],
            s=18,
            alpha=0.75,
            label=f"label={label_value}",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA of pooled embeddings by true label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_true_label.png", dpi=300)
    plt.close(fig)

    # Plot by confusion category
    fig, ax = plt.subplots(figsize=(7, 6))

    for cat, group in pca_df.groupby("confusion_category"):
        ax.scatter(
            group["PC1"],
            group["PC2"],
            s=18,
            alpha=0.75,
            label=cat,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA of pooled embeddings by prediction category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_confusion_category.png", dpi=300)
    plt.close(fig)

    return pca_df, pca.explained_variance_ratio_


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    print("Initializing model...")
    model = entexBERT2ForSequencePrediction(
        model_name_or_path=args.model_name_or_path,
        main_task="classification",
        main_num_labels=2,
        pooling_mode=args.pooling_mode,
        center_pool_width=args.center_pool_width,
        head_num_layers=args.head_num_layers,
        head_hidden_size=args.head_hidden_size,
        head_activation=args.head_activation,
        head_dropout=args.head_dropout,
    )

    model_file = find_best_or_final_model_file(args.checkpoint_dir)
    model.to(device)
    load_model_weights(model, model_file, device)
    model.eval()

    print(f"Reading data: {args.data_csv}")
    df = pd.read_csv(args.data_csv)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in data CSV.")

    if args.input_mode == "hap_pair":
        required = {"sequence1", "sequence2", "label"}
    else:
        required = {"sequence", "label"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_logits = []
    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start:start + args.batch_size]
            batch = tokenize_batch(
                tokenizer,
                batch_df,
                args.input_mode,
                args.model_max_length,
                device,
            )

            backbone_inputs = {
                k: v for k, v in batch.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
            }

            backbone_outputs = model.backbone(
                **backbone_inputs,
                return_dict=False,
            )

            sequence_output = get_sequence_output(backbone_outputs)
            pooled = pool_sequence(
                model,
                sequence_output,
                attention_mask=batch.get("attention_mask"),
            )

            pooled = model.dropout(pooled)
            logits = model.main_head(pooled)

            all_logits.append(logits.detach().cpu())
            all_embeddings.append(pooled.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    prob_positive = probs[:, 1]
    preds = np.argmax(probs, axis=-1)

    labels = df["label"].astype(int).to_numpy()
    cats = confusion_category(labels, preds)

    pred_df = df.copy()
    pred_df["logit_0"] = logits[:, 0]
    pred_df["logit_1"] = logits[:, 1]
    pred_df["prob_0"] = probs[:, 0]
    pred_df["prob_1"] = probs[:, 1]
    pred_df["prob_positive"] = prob_positive
    pred_df["pred_label"] = preds
    pred_df["confusion_category"] = cats

    metrics = compute_metrics(labels, prob_positive, preds)
    cm = confusion_matrix(labels, preds).tolist()

    pred_path = output_dir / "predictions.csv"
    emb_path = output_dir / "embeddings.npy"
    metrics_path = output_dir / "metrics.json"

    pred_df.to_csv(pred_path, index=False)
    np.save(emb_path, embeddings)

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "confusion_matrix": cm,
                "n_examples": int(len(df)),
                "model_file": str(model_file),
            },
            f,
            indent=2,
        )

    pca_df, explained = plot_pca(embeddings, pred_df, output_dir)

    print("\nDone.")
    print(f"Predictions: {pred_path}")
    print(f"Embeddings:  {emb_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"PCA figures: {output_dir}")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nPCA explained variance:")
    print(explained)


if __name__ == "__main__":
    main()