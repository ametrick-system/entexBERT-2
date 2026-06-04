#!/usr/bin/env python3

import argparse
import os
import re
import json
from pathlib import Path

import pandas as pd
import torch
import transformers

from bertviz import head_view, model_view

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction


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

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {model_file}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("First missing:", missing[:10])
    if unexpected:
        print("First unexpected:", unexpected[:10])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--model_name_or_path", default="DNABERT-2-117M-attention")
    parser.add_argument("--model_max_length", type=int, default=512)

    parser.add_argument("--category", default="TP", choices=["TP", "TN", "FP", "FN"])
    parser.add_argument("--rank", type=int, default=1)

    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    df = pd.read_csv(args.examples_csv)

    sub = df[df["confusion_category"] == args.category].copy()
    if sub.empty:
        raise ValueError(f"No examples found for category {args.category}")

    if "selection_rank_within_category" in sub.columns:
        chosen = sub[sub["selection_rank_within_category"] == args.rank]
        if chosen.empty:
            raise ValueError(
                f"No {args.category} example with selection_rank_within_category={args.rank}"
            )
        row = chosen.iloc[0]
    else:
        row = sub.iloc[args.rank - 1]

    seq1 = str(row["sequence1"])
    seq2 = str(row["sequence2"])

    print("Selected example:")
    print(row[["label", "pred_label", "prob_positive", "confusion_category"]])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    model = entexBERT2ForSequencePrediction(
        model_name_or_path=args.model_name_or_path,
        main_task="classification",
        main_num_labels=2,
        pooling_mode=args.pooling_mode,
        head_num_layers=args.head_num_layers,
        head_hidden_size=args.head_hidden_size,
        head_activation=args.head_activation,
        head_dropout=args.head_dropout,
    )

    model_file = find_best_or_final_model_file(args.checkpoint_dir)
    model.to(device)
    load_model_weights(model, model_file, device)
    model.eval()

    inputs = tokenizer(
        seq1,
        seq2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.model_max_length,
    )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

    print("Logits:", outputs.logits.detach().cpu().numpy())
    print("Prob non-AS:", float(probs[0]))
    print("Prob AS:", float(probs[1]))
    print("Num tokens:", len(tokens))
    print("Num attention layers:", len(outputs.attentions))
    print("Attention[0] shape:", tuple(outputs.attentions[0].shape))

    # BertViz expects attention tensors on CPU.
    attentions = tuple(attn.detach().cpu() for attn in outputs.attentions)

    # Save token list for debugging.
    with open(output_dir / f"{args.category}_rank{args.rank}_tokens.txt", "w") as f:
        for i, tok in enumerate(tokens):
            f.write(f"{i}\t{tok}\n")

    # Save a compact metadata file.
    meta = {
        "category": args.category,
        "rank": args.rank,
        "label": int(row["label"]),
        "pred_label": int(row["pred_label"]),
        "prob_positive_original": float(row["prob_positive"]),
        "prob_positive_recomputed": float(probs[1]),
        "num_tokens": len(tokens),
        "model_file": str(model_file),
    }

    with open(output_dir / f"{args.category}_rank{args.rank}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Create BertViz HTML files.
    # Model view: all layers. This is the broad overview.
    html = model_view(
        attentions,
        tokens,
        html_action="return",
    )
    with open(output_dir / f"{args.category}_rank{args.rank}_model_view.html", "w") as f:
        f.write(html.data)

    # Head view: final layer only, as a lighter first inspection.
    final_layer = len(attentions) - 1
    html = head_view(
        attentions,
        tokens,
        include_layers=[final_layer],
        html_action="return",
    )
    with open(output_dir / f"{args.category}_rank{args.rank}_head_view_final_layer.html", "w") as f:
        f.write(html.data)

    # Head view: all layers
    all_layers = list(range(len(attentions)))
    html = head_view(
        attentions,
        tokens,
        include_layers=all_layers,
        html_action="return",
    )
    with open(output_dir / f"{args.category}_rank{args.rank}_head_view_all_layers.html", "w") as f:
        f.write(html.data)

    print(f"\nSaved BertViz outputs to: {output_dir}")
    print("Open the .html files in a browser or download them from the cluster.")


if __name__ == "__main__":
    main()