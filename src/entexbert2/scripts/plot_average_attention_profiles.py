#!/usr/bin/env python3

import argparse
import json
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
        description=(
            "Plot average DNABERT-2 attention profiles by TP/FP/TN/FN category. "
            "Supports choosing layer(s), head(s), and source token."
        )
    )

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to patched local DNABERT-2 model directory.",
    )

    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--left_bp", type=int, default=256)

    parser.add_argument(
        "--n_per_category",
        type=int,
        default=100,
        help="Top N examples per confusion category.",
    )

    parser.add_argument(
        "--categories",
        default="TP,FP,TN,FN",
        help="Comma-separated categories to include.",
    )

    parser.add_argument(
        "--layers",
        default="11",
        help="Comma-separated layer indices, e.g. '11' or '8,9,10,11' or 'all'.",
    )

    parser.add_argument(
        "--heads",
        default="all",
        help="Comma-separated head indices, e.g. '0,3,7' or 'all'.",
    )

    parser.add_argument(
        "--source_token",
        default="cls",
        choices=["cls", "hap1_snv", "hap2_snv", "token_index", "hap1_position", "hap2_position"],
        help=(
            "Token from which attention is taken. "
            "'cls' uses token 0. "
            "'hap1_snv' or 'hap2_snv' use the token containing the SNV. "
            "'token_index' uses --source_token_index. "
            "'hap1_position'/'hap2_position' use --source_base_pos."
        ),
    )

    parser.add_argument(
        "--source_token_index",
        type=int,
        default=None,
        help="Used only when --source_token token_index.",
    )

    parser.add_argument(
        "--source_base_pos",
        type=int,
        default=None,
        help="Used only when --source_token hap1_position or hap2_position. Defaults to --left_bp.",
    )

    parser.add_argument(
        "--input_mode",
        default="hap_pair",
        choices=["hap_pair", "ref_single"],
    )

    # Must match training run
    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def parse_index_list(spec, max_value=None):
    if spec == "all":
        if max_value is None:
            return "all"
        return list(range(max_value))

    values = []
    for x in spec.split(","):
        x = x.strip()
        if x:
            values.append(int(x))

    return values


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
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])


def select_examples(df, categories, n_per_category):
    selected = []

    for category in categories:
        group = df[df["confusion_category"] == category].copy()

        if group.empty:
            print(f"Warning: no examples found for {category}")
            continue

        if "selection_rank_within_category" in group.columns:
            group = group.sort_values("selection_rank_within_category")
        elif "prob_positive" in group.columns:
            if category in {"TP", "FP"}:
                group = group.sort_values("prob_positive", ascending=False)
            else:
                group = group.sort_values("prob_positive", ascending=True)

        group = group.head(n_per_category).copy()
        print(f"{category}: using {len(group)} examples")
        selected.append(group)

    if not selected:
        raise ValueError("No examples selected.")

    return pd.concat(selected, ignore_index=True)


def find_token_containing_position(sequence_ids, offsets, target_sequence_id, base_pos):
    hits = []

    for i, (sid, offset) in enumerate(zip(sequence_ids, offsets)):
        start, end = offset
        if sid == target_sequence_id and start <= base_pos < end:
            hits.append(i)

    if not hits:
        return None

    return hits[0]


def get_source_token_index(
    source_token,
    source_token_index,
    source_base_pos,
    left_bp,
    sequence_ids,
    offsets,
):
    snv_pos = left_bp
    base_pos = source_base_pos if source_base_pos is not None else snv_pos

    if source_token == "cls":
        return 0

    if source_token == "token_index":
        if source_token_index is None:
            raise ValueError("--source_token_index is required when --source_token token_index")
        return int(source_token_index)

    if source_token == "hap1_snv":
        idx = find_token_containing_position(sequence_ids, offsets, target_sequence_id=0, base_pos=snv_pos)
        if idx is None:
            raise ValueError("Could not find hap1 SNV token.")
        return idx

    if source_token == "hap2_snv":
        idx = find_token_containing_position(sequence_ids, offsets, target_sequence_id=1, base_pos=snv_pos)
        if idx is None:
            raise ValueError("Could not find hap2 SNV token.")
        return idx

    if source_token == "hap1_position":
        idx = find_token_containing_position(sequence_ids, offsets, target_sequence_id=0, base_pos=base_pos)
        if idx is None:
            raise ValueError(f"Could not find hap1 token containing base position {base_pos}.")
        return idx

    if source_token == "hap2_position":
        idx = find_token_containing_position(sequence_ids, offsets, target_sequence_id=1, base_pos=base_pos)
        if idx is None:
            raise ValueError(f"Could not find hap2 token containing base position {base_pos}.")
        return idx

    raise ValueError(f"Unsupported source_token: {source_token}")


def token_scores_to_base_profiles(
    token_scores,
    sequence_ids,
    offsets,
    seq1_len,
    seq2_len=None,
):
    hap1_scores = np.zeros(seq1_len, dtype=float)
    hap1_counts = np.zeros(seq1_len, dtype=float)

    hap2_scores = None
    hap2_counts = None

    if seq2_len is not None:
        hap2_scores = np.zeros(seq2_len, dtype=float)
        hap2_counts = np.zeros(seq2_len, dtype=float)

    for tok_idx, score in enumerate(token_scores):
        sid = sequence_ids[tok_idx]
        start, end = offsets[tok_idx]

        if sid is None:
            continue
        if end <= start:
            continue

        score = float(score)

        if sid == 0:
            start = max(0, int(start))
            end = min(seq1_len, int(end))
            if end > start:
                hap1_scores[start:end] += score
                hap1_counts[start:end] += 1.0

        elif sid == 1 and seq2_len is not None:
            start = max(0, int(start))
            end = min(seq2_len, int(end))
            if end > start:
                hap2_scores[start:end] += score
                hap2_counts[start:end] += 1.0

    valid1 = hap1_counts > 0
    hap1_scores[valid1] /= hap1_counts[valid1]

    if seq2_len is not None:
        valid2 = hap2_counts > 0
        hap2_scores[valid2] /= hap2_counts[valid2]

    return hap1_scores, hap2_scores


def compute_attention_profile_for_example(
    model,
    tokenizer,
    row,
    args,
    layer_indices,
    head_indices,
    device,
):
    if args.input_mode == "hap_pair":
        seq1 = str(row["sequence1"])
        seq2 = str(row["sequence2"])

        encoding = tokenizer(
            seq1,
            seq2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.model_max_length,
            return_offsets_mapping=True,
        )

    else:
        seq1 = str(row["sequence"])
        seq2 = None

        encoding = tokenizer(
            seq1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.model_max_length,
            return_offsets_mapping=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids(0)

    source_idx = get_source_token_index(
        source_token=args.source_token,
        source_token_index=args.source_token_index,
        source_base_pos=args.source_base_pos,
        left_bp=args.left_bp,
        sequence_ids=sequence_ids,
        offsets=offsets,
    )

    source_token_label = tokens[source_idx]

    model_inputs = {
        k: v.to(device)
        for k, v in encoding.items()
        if k != "offset_mapping"
    }

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            output_hidden_states=False,
        )

    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    if layer_indices == "all":
        layer_indices = list(range(len(attentions)))

    num_heads = attentions[0].shape[1]
    if head_indices == "all":
        head_indices = list(range(num_heads))

    # Average attention from source token over selected layers and heads.
    # Each layer tensor shape: [batch, heads, seq_len, seq_len]
    selected = []
    for layer_idx in layer_indices:
        attn_layer = attentions[layer_idx][0]  # [heads, seq, seq]
        attn_selected_heads = attn_layer[head_indices, source_idx, :]  # [heads, seq]
        selected.append(attn_selected_heads.mean(dim=0))

    token_scores = torch.stack(selected, dim=0).mean(dim=0).detach().cpu().numpy()

    hap1_scores, hap2_scores = token_scores_to_base_profiles(
        token_scores=token_scores,
        sequence_ids=sequence_ids,
        offsets=offsets,
        seq1_len=len(seq1),
        seq2_len=len(seq2) if seq2 is not None else None,
    )

    info = {
        "source_token_index": source_idx,
        "source_token": source_token_label,
        "num_tokens": len(tokens),
        "prob_positive": float(torch.softmax(outputs.logits, dim=-1)[0, 1].detach().cpu()),
    }

    return hap1_scores, hap2_scores, info


def make_long_rows(example_id, row, hap_name, scores, left_bp, info):
    rows = []
    for pos, value in enumerate(scores):
        rows.append({
            "example_id": example_id,
            "confusion_category": row["confusion_category"],
            "label": int(row["label"]),
            "pred_label": int(row["pred_label"]),
            "hap": hap_name,
            "position": pos,
            "position_relative_to_snv": pos - left_bp,
            "attention": float(value),
            "source_token_index": info["source_token_index"],
            "source_token": info["source_token"],
            "num_tokens": info["num_tokens"],
            "prob_positive_recomputed": info["prob_positive"],
        })
    return rows


def plot_profiles(attn_df, output_dir, title_suffix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hap in sorted(attn_df["hap"].unique()):
        sub = attn_df[attn_df["hap"] == hap].copy()

        fig, ax = plt.subplots(figsize=(8, 5))

        for category, group in sub.groupby("confusion_category"):
            mean_profile = (
                group.groupby("position_relative_to_snv")["attention"]
                .mean()
                .sort_index()
            )

            ax.plot(
                mean_profile.index,
                mean_profile.values,
                linewidth=2,
                label=category,
            )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Position relative to SNV")
        ax.set_ylabel("Average attention")
        ax.set_title(f"{hap} average attention by confusion category\n{title_suffix}")
        ax.legend()
        fig.tight_layout()

        out_path = output_dir / f"average_attention_{hap}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    # If hap-pair, also plot hap1 + hap2 and hap1 - hap2.
    if {"hap1", "hap2"}.issubset(set(attn_df["hap"].unique())):
        pivot = attn_df.pivot_table(
            index=["example_id", "confusion_category", "position_relative_to_snv"],
            columns="hap",
            values="attention",
        ).reset_index()

        pivot["hap1_plus_hap2"] = pivot["hap1"] + pivot["hap2"]
        pivot["hap1_minus_hap2"] = pivot["hap1"] - pivot["hap2"]

        for value_col in ["hap1_plus_hap2", "hap1_minus_hap2"]:
            fig, ax = plt.subplots(figsize=(8, 5))

            for category, group in pivot.groupby("confusion_category"):
                mean_profile = (
                    group.groupby("position_relative_to_snv")[value_col]
                    .mean()
                    .sort_index()
                )

                ax.plot(
                    mean_profile.index,
                    mean_profile.values,
                    linewidth=2,
                    label=category,
                )

            ax.axvline(0, linestyle="--", linewidth=1)
            if value_col.endswith("minus_hap2"):
                ax.axhline(0, linewidth=0.75)

            ax.set_xlabel("Position relative to SNV")
            ax.set_ylabel("Average attention")
            ax.set_title(f"{value_col} average attention\n{title_suffix}")
            ax.legend()
            fig.tight_layout()

            out_path = output_dir / f"average_attention_{value_col}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved {out_path}")

        pivot.to_csv(output_dir / "attention_pair_summary_long.csv", index=False)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    print("Loading examples...")
    df = pd.read_csv(args.examples_csv)
    selected_df = select_examples(df, categories, args.n_per_category)

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    print("Loading model...")
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

    layer_indices = parse_index_list(args.layers)
    head_indices = parse_index_list(args.heads)

    all_rows = []
    source_info_rows = []

    print("Computing attention profiles...")
    for local_idx, row in selected_df.iterrows():
        hap1_scores, hap2_scores, info = compute_attention_profile_for_example(
            model=model,
            tokenizer=tokenizer,
            row=row,
            args=args,
            layer_indices=layer_indices,
            head_indices=head_indices,
            device=device,
        )

        example_id = row.get("example_index", local_idx)

        all_rows.extend(
            make_long_rows(
                example_id=example_id,
                row=row,
                hap_name="hap1" if args.input_mode == "hap_pair" else "ref",
                scores=hap1_scores,
                left_bp=args.left_bp,
                info=info,
            )
        )

        if hap2_scores is not None:
            all_rows.extend(
                make_long_rows(
                    example_id=example_id,
                    row=row,
                    hap_name="hap2",
                    scores=hap2_scores,
                    left_bp=args.left_bp,
                    info=info,
                )
            )

        source_info_rows.append({
            "example_id": example_id,
            "confusion_category": row["confusion_category"],
            "label": int(row["label"]),
            "pred_label": int(row["pred_label"]),
            "source_token_index": info["source_token_index"],
            "source_token": info["source_token"],
            "num_tokens": info["num_tokens"],
            "prob_positive_recomputed": info["prob_positive"],
        })

        if (local_idx + 1) % 25 == 0 or local_idx == len(selected_df) - 1:
            print(f"  processed {local_idx + 1}/{len(selected_df)}")

    attn_df = pd.DataFrame(all_rows)
    source_df = pd.DataFrame(source_info_rows)

    attn_path = output_dir / "attention_profiles_long.csv"
    source_path = output_dir / "attention_source_tokens.csv"

    attn_df.to_csv(attn_path, index=False)
    source_df.to_csv(source_path, index=False)

    title_suffix = (
        f"source={args.source_token}, layers={args.layers}, heads={args.heads}, "
        f"n/category={args.n_per_category}"
    )

    plot_profiles(attn_df, output_dir, title_suffix=title_suffix)

    config = vars(args)
    config["model_file"] = str(model_file)

    with open(output_dir / "attention_profile_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Saved long attention table: {attn_path}")
    print(f"Saved source token table:   {source_path}")
    print(f"Saved plots/config to:      {output_dir}")


if __name__ == "__main__":
    main()