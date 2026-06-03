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

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute gradient x input saliency profiles for representative ref_single examples."
    )

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--model_name_or_path", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--left_bp", type=int, default=256)

    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--center_pool_width", type=int, default=5)
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument(
        "--saliency_method",
        default="grad_x_input",
        choices=["grad_x_input", "grad_norm"],
    )
    parser.add_argument(
        "--normalize_per_example",
        action="store_true",
        help="Normalize each saliency profile to sum to 1.",
    )

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def find_best_or_final_model_file(checkpoint_dir):
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
        from safetensors.torch import load_file
        state_dict = load_file(str(model_file), device=str(device))
    else:
        state_dict = torch.load(str(model_file), map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {model_file}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def map_token_scores_to_bases(
    token_scores,
    offset_mapping,
    sequence_ids,
    sequence_length,
    normalize=False,
):
    base_scores = np.zeros(sequence_length, dtype=float)
    counts = np.zeros(sequence_length, dtype=float)

    for tok_idx, (start, end) in enumerate(offset_mapping):
        seq_id = sequence_ids[tok_idx]

        if seq_id is None:
            continue

        start = int(start)
        end = int(end)

        if end <= start:
            continue

        start = max(0, start)
        end = min(sequence_length, end)

        if end <= start:
            continue

        base_scores[start:end] += float(token_scores[tok_idx])
        counts[start:end] += 1.0

    valid = counts > 0
    base_scores[valid] /= counts[valid]

    if normalize and base_scores.sum() > 0:
        base_scores = base_scores / base_scores.sum()

    return base_scores


def compute_one_example_saliency(
    model,
    tokenizer,
    sequence,
    device,
    model_max_length,
    saliency_method,
    normalize,
):
    encoding = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model_max_length,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding.pop("offset_mapping")[0].cpu().numpy()
    sequence_ids = encoding.sequence_ids(0)

    batch = {k: v.to(device) for k, v in encoding.items()}

    model.zero_grad(set_to_none=True)

    embedding_layer = model.backbone.get_input_embeddings()
    saved = {}

    def embedding_hook(module, inputs, output):
        saved["embeddings"] = output
        output.retain_grad()

    handle = embedding_layer.register_forward_hook(embedding_hook)

    outputs = model(**batch)
    logits = outputs.logits

    # AS-vs-non-AS score
    score = logits[0, 1] - logits[0, 0]
    score.backward()

    handle.remove()

    embeddings = saved["embeddings"]
    grad = embeddings.grad

    if grad is None:
        raise RuntimeError("Embedding gradients were not captured.")

    emb = embeddings.detach()[0]
    grad = grad.detach()[0]

    if saliency_method == "grad_x_input":
        token_scores = (grad * emb).norm(dim=-1)
    elif saliency_method == "grad_norm":
        token_scores = grad.norm(dim=-1)
    else:
        raise ValueError(f"Unsupported saliency method: {saliency_method}")

    token_scores = token_scores.cpu().numpy()

    base_scores = map_token_scores_to_bases(
        token_scores=token_scores,
        offset_mapping=offset_mapping,
        sequence_ids=sequence_ids,
        sequence_length=len(sequence),
        normalize=normalize,
    )

    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    return {
        "base_scores": base_scores,
        "logit_0": float(logits[0, 0].detach().cpu()),
        "logit_1": float(logits[0, 1].detach().cpu()),
        "prob_0": float(probs[0]),
        "prob_1": float(probs[1]),
        "score": float(score.detach().cpu()),
    }


def plot_average_saliency(saliency_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for category, group in saliency_df.groupby("confusion_category"):
        mean_profile = group.groupby("position_relative_to_snv")["saliency"].mean()
        ax.plot(
            mean_profile.index,
            mean_profile.values,
            label=category,
            linewidth=2,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel("Average saliency")
    ax.set_title("Average ref_single saliency by confusion category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "average_saliency_ref_single_by_confusion_category.png", dpi=300)
    plt.close(fig)


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

    examples = pd.read_csv(args.examples_csv)

    required = {"sequence", "label", "pred_label", "prob_positive", "confusion_category"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Examples CSV missing required columns: {missing}")

    saliency_rows = []
    summary_rows = []

    for local_idx, row in examples.iterrows():
        sequence = str(row["sequence"])

        result = compute_one_example_saliency(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            device=device,
            model_max_length=args.model_max_length,
            saliency_method=args.saliency_method,
            normalize=args.normalize_per_example,
        )

        example_id = row.get("example_index", local_idx)
        category = row["confusion_category"]

        for pos, val in enumerate(result["base_scores"]):
            saliency_rows.append({
                "example_id": example_id,
                "local_idx": local_idx,
                "confusion_category": category,
                "label": int(row["label"]),
                "pred_label": int(row["pred_label"]),
                "prob_positive_original": float(row["prob_positive"]),
                "prob_positive_recomputed": result["prob_1"],
                "position": pos,
                "position_relative_to_snv": pos - args.left_bp,
                "saliency": float(val),
            })

        summary_rows.append({
            "example_id": example_id,
            "local_idx": local_idx,
            "confusion_category": category,
            "label": int(row["label"]),
            "pred_label": int(row["pred_label"]),
            "prob_positive_original": float(row["prob_positive"]),
            "prob_positive_recomputed": result["prob_1"],
            "score": result["score"],
            "saliency_sum": float(result["base_scores"].sum()),
            "saliency_max": float(result["base_scores"].max()),
        })

        print(
            f"{local_idx + 1}/{len(examples)} "
            f"{category} label={row['label']} pred={row['pred_label']} "
            f"prob={result['prob_1']:.4f}"
        )

    saliency_df = pd.DataFrame(saliency_rows)
    summary_df = pd.DataFrame(summary_rows)

    saliency_path = output_dir / "saliency_long.csv"
    summary_path = output_dir / "saliency_summary.csv"

    saliency_df.to_csv(saliency_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_average_saliency(saliency_df, output_dir)

    print("\nDone.")
    print(f"Saved saliency long table: {saliency_path}")
    print(f"Saved saliency summary:    {summary_path}")
    print(f"Saved plots to:            {output_dir}")


if __name__ == "__main__":
    main()