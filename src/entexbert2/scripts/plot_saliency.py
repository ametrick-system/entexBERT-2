#!/usr/bin/env python3

import argparse
import csv
import io
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

import requests
from Bio import motifs
from Bio.Seq import Seq

import transformers
from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "General entexBERT-2 saliency/motif plotting script. Supports "
            "classification/regression, confusion-category splits, sequence pairs, "
            "single sequence input, and optional JASPAR motif overlays."
        )
    )

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)

    parser.add_argument("--main_task", required=True, choices=["classification", "regression"])
    parser.add_argument("--main_num_labels", type=int, default=None)

    parser.add_argument("--input_mode", choices=["ref_single", "hap_pair"], default="ref_single")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--left_bp", type=int, default=256)

    # Classification selection
    parser.add_argument("--categories", default="TP,FP,TN,FN")
    parser.add_argument("--n_per_group", type=int, default=100)
    parser.add_argument("--force_recompute_predictions", action="store_true")

    # Regression selection
    parser.add_argument(
        "--regression_subsets",
        default="top,bottom",
        help="For regression: comma-separated from top,bottom,all,random.",
    )
    parser.add_argument(
        "--regression_sort_by",
        default="label",
        choices=["label", "pred", "abs_error"],
    )

    # Saliency objective
    parser.add_argument(
        "--saliency_target",
        default="auto",
        choices=[
            "auto",
            "positive_logit",
            "positive_prob",
            "predicted_logit",
            "true_logit",
            "margin",
            "loss",
            "regression_output",
        ],
    )
    parser.add_argument(
        "--saliency_method",
        default="abs_sum",
        choices=["abs_sum", "l2", "signed_sum"],
    )
    parser.add_argument(
        "--normalize_per_example",
        default="max",
        choices=["max", "sum", "none"],
    )

    # Plotting
    parser.add_argument(
        "--plot_values",
        default="all",
        help=(
            "For ref_single: all,ref. "
            "For hap_pair: all,hap1,hap2,hap1_plus_hap2,hap1_minus_hap2."
        ),
    )
    parser.add_argument("--plot_window_bp", type=int, default=-1)
    parser.add_argument("--make_heatmaps", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)

    # Motifs
    parser.add_argument(
        "--motifs_csv",
        default=None,
        help="Optional CSV: motif_name,jaspar_id[,color].",
    )
    parser.add_argument("--motif_sigma", type=float, default=5.0)
    parser.add_argument("--overlay_motifs", action="store_true")

    # Reference markers
    parser.add_argument("--center_label", default="SNV")
    parser.add_argument("--jitter_bp", type=int, default=None)

    # Model args matching fine-tuning run
    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

# -------------------------
# Checkpoint/model helpers
# -------------------------
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


def load_tokenizer_and_model(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if args.main_num_labels is None:
        main_num_labels = 1 if args.main_task == "regression" else 2
    else:
        main_num_labels = args.main_num_labels

    model = entexBERT2ForSequencePrediction(
        model_name_or_path=args.model_name_or_path,
        main_task=args.main_task,
        main_num_labels=main_num_labels,
        pooling_mode=args.pooling_mode,
        head_num_layers=args.head_num_layers,
        head_hidden_size=args.head_hidden_size,
        head_activation=args.head_activation,
        head_dropout=args.head_dropout,
    )

    model_file = find_best_or_final_model_file(args.checkpoint_dir)
    device = torch.device(args.device)
    model.to(device)
    load_model_weights(model, model_file, device)
    model.eval()

    return tokenizer, model, device, model_file


def get_embedding_layer(model):
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "backbone") and hasattr(model.backbone, "get_input_embeddings"):
        emb = model.backbone.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "backbone") and hasattr(model.backbone, "embeddings"):
        return model.backbone.embeddings.word_embeddings

    raise AttributeError("Could not find input embedding layer.")


# -------------------------
# Prediction / selection
# -------------------------
def tokenize_row(tokenizer, row, args, return_offsets_mapping=False):
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
            return_offsets_mapping=return_offsets_mapping,
        )
        return encoding, seq1, seq2

    seq = str(row["sequence"])
    encoding = tokenizer(
        seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.model_max_length,
        return_offsets_mapping=return_offsets_mapping,
    )
    return encoding, seq, None


def predict_one(model, tokenizer, row, args, device):
    encoding, _, _ = tokenize_row(tokenizer, row, args, return_offsets_mapping=False)
    model_inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)

    logits = outputs.logits.detach().cpu()

    if args.main_task == "classification":
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
        prob_positive = float(probs[1].item()) if probs.numel() > 1 else float(probs[0].item())
        return {
            "pred_label": pred_label,
            "prob_positive": prob_positive,
            "logit_0": float(logits[0, 0].item()),
            "logit_1": float(logits[0, 1].item()) if logits.shape[1] > 1 else np.nan,
        }

    pred = float(logits.view(-1)[0].item())
    return {"pred": pred}


def add_predictions_if_needed(df, model, tokenizer, args, device):
    df = df.copy()

    need_predictions = args.force_recompute_predictions

    if args.main_task == "classification":
        needed = {"pred_label", "prob_positive", "confusion_category"}
        if not needed.issubset(df.columns):
            need_predictions = True

    else:
        if "pred" not in df.columns:
            need_predictions = True

    if not need_predictions:
        print("Using existing predictions/groups from examples CSV.")
        return df

    print("Computing predictions for selection/grouping...")
    preds = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        preds.append(predict_one(model, tokenizer, row, args, device))

    pred_df = pd.DataFrame(preds)
    for col in pred_df.columns:
        df[col] = pred_df[col].values

    if args.main_task == "classification":
        if "label" not in df.columns:
            raise ValueError("Classification selection requires a label column.")

        labels = df["label"].astype(int)
        pred_labels = df["pred_label"].astype(int)

        df["confusion_category"] = np.select(
            [
                (labels == 1) & (pred_labels == 1),
                (labels == 0) & (pred_labels == 0),
                (labels == 0) & (pred_labels == 1),
                (labels == 1) & (pred_labels == 0),
            ],
            ["TP", "TN", "FP", "FN"],
            default="OTHER",
        )

    else:
        if "label" in df.columns:
            df["abs_error"] = (df["pred"] - df["label"]).abs()

    return df


def select_classification_examples(df, args):
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    selected = []

    for category in categories:
        group = df[df["confusion_category"] == category].copy()
        if group.empty:
            print(f"Warning: no examples for {category}")
            continue

        if "selection_rank_within_category" in group.columns:
            group = group.sort_values("selection_rank_within_category")
        elif "prob_positive" in group.columns:
            if category in {"TP", "FP"}:
                group = group.sort_values("prob_positive", ascending=False)
            elif category in {"TN", "FN"}:
                group = group.sort_values("prob_positive", ascending=True)

        group = group.head(args.n_per_group).copy()
        group["saliency_group"] = category
        print(f"{category}: selected {len(group)} examples")
        selected.append(group)

    if not selected:
        raise ValueError("No classification examples selected.")

    return pd.concat(selected, ignore_index=True)


def select_regression_examples(df, args):
    subsets = [x.strip() for x in args.regression_subsets.split(",") if x.strip()]
    selected = []

    if args.regression_sort_by not in df.columns:
        raise ValueError(f"Regression sort column not found: {args.regression_sort_by}")

    for subset in subsets:
        if subset == "top":
            group = df.sort_values(args.regression_sort_by, ascending=False).head(args.n_per_group).copy()
        elif subset == "bottom":
            group = df.sort_values(args.regression_sort_by, ascending=True).head(args.n_per_group).copy()
        elif subset == "random":
            group = df.sample(n=min(args.n_per_group, len(df)), random_state=args.seed).copy()
        elif subset == "all":
            group = df.copy()
        else:
            raise ValueError(f"Unsupported regression subset: {subset}")

        group["saliency_group"] = subset
        print(f"{subset}: selected {len(group)} examples")
        selected.append(group)

    return pd.concat(selected, ignore_index=True)


def select_examples(df, model, tokenizer, args, device):
    df = add_predictions_if_needed(df, model, tokenizer, args, device)

    if args.main_task == "classification":
        return select_classification_examples(df, args)

    return select_regression_examples(df, args)


# -------------------------
# Saliency
# -------------------------
def choose_objective(outputs, row, args, device):
    logits = outputs.logits

    if args.main_task == "regression":
        return logits.view(-1)[0]

    target = args.saliency_target
    if target == "auto":
        target = "positive_logit"

    if target == "positive_logit":
        return logits[0, 1]

    if target == "positive_prob":
        return torch.softmax(logits, dim=-1)[0, 1]

    if target == "predicted_logit":
        pred = int(torch.argmax(logits, dim=-1)[0].item())
        return logits[0, pred]

    if target == "true_logit":
        if "label" not in row:
            raise ValueError("true_logit target requires label column.")
        label = int(row["label"])
        return logits[0, label]

    if target == "margin":
        return logits[0, 1] - logits[0, 0]

    if target == "loss":
        if "label" not in row:
            raise ValueError("loss target requires label column.")
        label = torch.tensor([int(row["label"])], dtype=torch.long, device=device)
        return nn.CrossEntropyLoss()(logits, label)

    raise ValueError(f"Unsupported saliency_target for classification: {target}")


def token_saliency_from_embedding_grad(grad_tensor, args):
    # grad_tensor: [1, seq_len, hidden_dim]
    grad = grad_tensor.squeeze(0)

    if args.saliency_method == "abs_sum":
        sal = grad.abs().sum(dim=-1)
    elif args.saliency_method == "l2":
        sal = torch.sqrt((grad ** 2).sum(dim=-1))
    elif args.saliency_method == "signed_sum":
        sal = grad.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported saliency_method: {args.saliency_method}")

    sal = sal.detach().cpu().numpy()
    return sal


def token_scores_to_base_profiles(token_scores, sequence_ids, offsets, seq1_len, seq2_len=None):
    seq1_scores = np.zeros(seq1_len, dtype=float)
    seq1_counts = np.zeros(seq1_len, dtype=float)

    seq2_scores = None
    seq2_counts = None

    if seq2_len is not None:
        seq2_scores = np.zeros(seq2_len, dtype=float)
        seq2_counts = np.zeros(seq2_len, dtype=float)

    for tok_idx, score in enumerate(token_scores):
        sid = sequence_ids[tok_idx]
        start, end = offsets[tok_idx]

        if sid is None:
            continue
        if end <= start:
            continue

        if sid == 0:
            start = max(0, int(start))
            end = min(seq1_len, int(end))
            if end > start:
                seq1_scores[start:end] += float(score)
                seq1_counts[start:end] += 1.0

        elif sid == 1 and seq2_len is not None:
            start = max(0, int(start))
            end = min(seq2_len, int(end))
            if end > start:
                seq2_scores[start:end] += float(score)
                seq2_counts[start:end] += 1.0

    valid1 = seq1_counts > 0
    seq1_scores[valid1] /= seq1_counts[valid1]

    if seq2_len is not None:
        valid2 = seq2_counts > 0
        seq2_scores[valid2] /= seq2_counts[valid2]

    return seq1_scores, seq2_scores


def normalize_profile(x, mode):
    x = np.asarray(x, dtype=float)

    if mode == "none":
        return x

    if mode == "max":
        denom = np.nanmax(np.abs(x))
        if denom > 0:
            return x / denom
        return x

    if mode == "sum":
        denom = np.nansum(np.abs(x))
        if denom > 0:
            return x / denom
        return x

    raise ValueError(f"Unsupported normalize mode: {mode}")


def compute_saliency_for_example(model, tokenizer, embedding_layer, row, args, device):
    encoding, seq1, seq2 = tokenize_row(
        tokenizer,
        row,
        args,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids(0)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    model_inputs = {
        k: v.to(device)
        for k, v in encoding.items()
        if k != "offset_mapping"
    }

    grads = []

    def save_grads(module, grad_in, grad_out):
        grads.append(grad_out[0])

    handle = embedding_layer.register_full_backward_hook(save_grads)

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(**model_inputs)
        objective = choose_objective(outputs, row, args, device)
        objective.backward()
    finally:
        handle.remove()

    if not grads:
        raise RuntimeError("No embedding gradients captured.")

    token_sal = token_saliency_from_embedding_grad(grads[0], args)

    seq1_scores, seq2_scores = token_scores_to_base_profiles(
        token_scores=token_sal,
        sequence_ids=sequence_ids,
        offsets=offsets,
        seq1_len=len(seq1),
        seq2_len=len(seq2) if seq2 is not None else None,
    )

    seq1_scores = normalize_profile(seq1_scores, args.normalize_per_example)
    if seq2_scores is not None:
        seq2_scores = normalize_profile(seq2_scores, args.normalize_per_example)

    with torch.no_grad():
        logits = outputs.logits.detach().cpu()
        if args.main_task == "classification":
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = int(torch.argmax(probs).item())
            prob_positive = float(probs[1].item())
            pred_value = np.nan
        else:
            pred_label = np.nan
            prob_positive = np.nan
            pred_value = float(logits.view(-1)[0].item())

    info = {
        "num_tokens": len(tokens),
        "objective_value": float(objective.detach().cpu().item()),
        "pred_label_recomputed": pred_label,
        "prob_positive_recomputed": prob_positive,
        "pred_recomputed": pred_value,
    }

    if args.input_mode == "hap_pair":
        profiles = {"hap1": seq1_scores, "hap2": seq2_scores}
    else:
        profiles = {"ref": seq1_scores}

    return profiles, info


# -------------------------
# Motifs
# -------------------------
def load_motifs_csv(path):
    motif_ids = {}
    colors = {}

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            name = row[0].strip()
            jaspar_id = row[1].strip()
            motif_ids[name] = jaspar_id
            if len(row) >= 3:
                colors[name] = row[2].strip()

    return motif_ids, colors


def download_jaspar_pssm(jaspar_id):
    url = f"https://jaspar.genereg.net/api/v1/matrix/{jaspar_id}.pfm"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    m = motifs.read(io.StringIO(response.text), "jaspar")
    pwm = m.counts.normalize(pseudocounts=0.8)
    return pwm.log_odds()


def scan_motif_mean_profile(sequences, pssm, sigma):
    tracks = []

    for s in sequences:
        seq = Seq(str(s).upper())
        scores_fwd = np.array(pssm.calculate(seq))
        scores_rev = np.array(pssm.calculate(seq.reverse_complement()))[::-1]
        scores = np.maximum(scores_fwd, scores_rev)

        scores = scores - np.nanmin(scores)
        denom = np.nanmax(scores)
        if denom > 0:
            scores = scores / denom

        tracks.append(scores)

    min_len = min(map(len, tracks))
    arr = np.vstack([t[:min_len] for t in tracks])
    mean_track = arr.mean(axis=0)

    if sigma is not None and sigma > 0:
        mean_track = gaussian_filter1d(mean_track, sigma=sigma)

    return mean_track


def build_motif_profiles(selected_df, args):
    if args.motifs_csv is None or not args.overlay_motifs:
        return {}, {}

    motif_ids, colors = load_motifs_csv(args.motifs_csv)
    print(f"Loaded motifs: {motif_ids}")

    profile_sequences = {}

    if args.input_mode == "hap_pair":
        profile_sequences["hap1"] = selected_df["sequence1"].astype(str).tolist()
        profile_sequences["hap2"] = selected_df["sequence2"].astype(str).tolist()
    else:
        profile_sequences["ref"] = selected_df["sequence"].astype(str).tolist()

    motif_profiles = {}

    for motif_name, jaspar_id in motif_ids.items():
        print(f"Downloading/scanning motif {motif_name} ({jaspar_id})...")
        pssm = download_jaspar_pssm(jaspar_id)

        for profile_name, seqs in profile_sequences.items():
            key = (profile_name, motif_name)
            motif_profiles[key] = scan_motif_mean_profile(
                sequences=seqs,
                pssm=pssm,
                sigma=args.motif_sigma,
            )

    return motif_profiles, colors


# -------------------------
# Dataframe / plotting
# -------------------------
def add_saliency_rows(all_rows, example_id, row, profile_name, scores, args, info):
    for pos, val in enumerate(scores):
        all_rows.append({
            "example_id": example_id,
            "saliency_group": row["saliency_group"],
            "confusion_category": row.get("confusion_category", np.nan),
            "label": row.get("label", np.nan),
            "pred_label": row.get("pred_label", np.nan),
            "pred": row.get("pred", np.nan),
            "prob_positive": row.get("prob_positive", np.nan),
            "profile": profile_name,
            "position": pos,
            "position_relative_to_center": pos - args.left_bp,
            "saliency": float(val),
            **info,
        })


def build_value_tables(sal_df, args):
    available = set(sal_df["profile"].unique())

    if args.plot_values == "all":
        if args.input_mode == "hap_pair":
            values = ["hap1", "hap2", "hap1_plus_hap2", "hap1_minus_hap2"]
        else:
            values = ["ref"]
    else:
        values = [x.strip() for x in args.plot_values.split(",") if x.strip()]

    tables = {}

    for value in values:
        if value in {"hap1", "hap2", "ref"}:
            if value not in available:
                print(f"Skipping {value}: not available.")
                continue
            sub = sal_df[sal_df["profile"] == value].copy()
            sub = sub.rename(columns={"saliency": "value"})
            tables[value] = sub[
                ["example_id", "saliency_group", "position_relative_to_center", "value"]
            ].copy()

        elif value in {"hap1_plus_hap2", "hap1_minus_hap2"}:
            if not {"hap1", "hap2"}.issubset(available):
                print(f"Skipping {value}: hap1/hap2 not both available.")
                continue

            pivot = sal_df.pivot_table(
                index=["example_id", "saliency_group", "position_relative_to_center"],
                columns="profile",
                values="saliency",
                aggfunc="mean",
            ).reset_index()

            pivot = pivot.dropna(subset=["hap1", "hap2"]).copy()

            if value == "hap1_plus_hap2":
                pivot["value"] = pivot["hap1"] + pivot["hap2"]
            else:
                pivot["value"] = pivot["hap1"] - pivot["hap2"]

            tables[value] = pivot[
                ["example_id", "saliency_group", "position_relative_to_center", "value"]
            ].copy()

        else:
            raise ValueError(f"Unsupported plot value: {value}")

    return tables


def filter_window(df, args):
    if args.plot_window_bp is None or args.plot_window_bp < 0:
        return df.copy()

    return df[
        df["position_relative_to_center"].between(
            -args.plot_window_bp,
            args.plot_window_bp,
        )
    ].copy()


def mean_profile(value_df, args):
    sub = filter_window(value_df, args)

    return (
        sub.groupby(["saliency_group", "position_relative_to_center"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_saliency", "std": "std_saliency"})
    )


def get_motif_overlay_for_value(value_name, motif_profiles):
    overlays = {}

    if value_name in {"ref", "hap1", "hap2"}:
        for (profile_name, motif_name), track in motif_profiles.items():
            if profile_name == value_name:
                overlays[motif_name] = track

    elif value_name in {"hap1_plus_hap2", "hap1_minus_hap2"}:
        motif_names = sorted({motif_name for _, motif_name in motif_profiles.keys()})

        for motif_name in motif_names:
            h1 = motif_profiles.get(("hap1", motif_name))
            h2 = motif_profiles.get(("hap2", motif_name))

            if h1 is None or h2 is None:
                continue

            min_len = min(len(h1), len(h2))
            if value_name == "hap1_plus_hap2":
                track = (h1[:min_len] + h2[:min_len]) / 2.0
            else:
                track = h1[:min_len] - h2[:min_len]

            track = track - np.nanmin(track)
            denom = np.nanmax(np.abs(track))
            if denom > 0:
                track = track / denom

            overlays[motif_name] = track

    return overlays


def plot_mean_saliency(mean_df, value_name, output_dir, args, motif_profiles, motif_colors):
    fig, ax = plt.subplots(figsize=(10, 4))

    for group, gdf in mean_df.groupby("saliency_group"):
        gdf = gdf.sort_values("position_relative_to_center")
        ax.plot(
            gdf["position_relative_to_center"],
            gdf["mean_saliency"],
            linewidth=2,
            label=str(group),
        )

    ax.axvline(0, linestyle="--", linewidth=1.5, label=args.center_label)

    if args.jitter_bp is not None:
        ax.axvspan(
            -args.jitter_bp,
            args.jitter_bp,
            alpha=0.12,
            label=f"±{args.jitter_bp} bp",
        )

    overlays = get_motif_overlay_for_value(value_name, motif_profiles)
    for motif_name, track in overlays.items():
        x = np.arange(len(track)) - args.left_bp

        if args.plot_window_bp is not None and args.plot_window_bp >= 0:
            keep = (x >= -args.plot_window_bp) & (x <= args.plot_window_bp)
            x = x[keep]
            track = track[keep]

        ax.plot(
            x,
            track,
            linestyle="--",
            linewidth=1.8,
            color=motif_colors.get(motif_name, None),
            label=f"{motif_name} motif",
            alpha=0.85,
        )

    window_label = "full" if args.plot_window_bp < 0 else f"pm{args.plot_window_bp}"
    ax.set_title(
        f"{value_name} saliency profile ({window_label})\n"
        f"task={args.main_task}, target={args.saliency_target}, n/group={args.n_per_group}"
    )
    ax.set_xlabel(f"Position relative to {args.center_label}")
    ax.set_ylabel("Mean saliency")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(output_dir) / f"{value_name}_saliency_profile_{window_label}.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmaps(value_df, value_name, output_dir, args):
    if not args.make_heatmaps:
        return

    sub = filter_window(value_df, args)

    for group, gdf in sub.groupby("saliency_group"):
        pivot = gdf.pivot_table(
            index="example_id",
            columns="position_relative_to_center",
            values="value",
            aggfunc="mean",
        )

        pivot = pivot.sort_index(axis=1)
        arr = pivot.values

        height = max(3, min(12, arr.shape[0] / 12))
        fig, ax = plt.subplots(figsize=(10, height))
        im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="magma")
        fig.colorbar(im, ax=ax, label="Saliency")

        positions = pivot.columns.values
        zero_idx = np.argmin(np.abs(positions))
        ax.axvline(zero_idx, linestyle="--", linewidth=1.2)

        ax.set_title(f"{value_name} saliency heatmap: {group}")
        ax.set_xlabel(f"Position relative to {args.center_label}")
        ax.set_ylabel("Example")

        # A few readable ticks
        tick_positions = np.linspace(0, len(positions) - 1, num=min(7, len(positions))).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(positions[i])) for i in tick_positions])

        fig.tight_layout()

        safe_group = str(group).replace("/", "_")
        out_path = Path(output_dir) / f"{value_name}_saliency_heatmap_{safe_group}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out_path}")


def compute_motif_correlations(mean_df, value_name, output_dir, args, motif_profiles):
    overlays = get_motif_overlay_for_value(value_name, motif_profiles)
    if not overlays:
        return

    rows = []

    overall = (
        mean_df.groupby("position_relative_to_center")["mean_saliency"]
        .mean()
        .sort_index()
    )

    sal_x = overall.index.values
    sal_y = overall.values

    for motif_name, track in overlays.items():
        motif_x = np.arange(len(track)) - args.left_bp

        common_min = max(sal_x.min(), motif_x.min())
        common_max = min(sal_x.max(), motif_x.max())

        keep_sal = (sal_x >= common_min) & (sal_x <= common_max)
        keep_motif = (motif_x >= common_min) & (motif_x <= common_max)

        sal_aligned = sal_y[keep_sal]
        motif_aligned = track[keep_motif]

        n = min(len(sal_aligned), len(motif_aligned))
        if n < 3:
            continue

        r, p = pearsonr(sal_aligned[:n], motif_aligned[:n])
        rows.append({
            "value": value_name,
            "motif": motif_name,
            "pearson_r": r,
            "p_value": p,
            "n_positions": n,
        })

    if rows:
        out = pd.DataFrame(rows)
        out_path = Path(output_dir) / f"{value_name}_motif_correlations.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model/tokenizer...")
    tokenizer, model, device, model_file = load_tokenizer_and_model(args)
    embedding_layer = get_embedding_layer(model)

    print("Loading examples...")
    df = pd.read_csv(args.examples_csv)

    selected_df = select_examples(df, model, tokenizer, args, device)
    selected_path = output_dir / "selected_examples.csv"
    selected_df.to_csv(selected_path, index=False)
    print(f"Saved {selected_path}")

    motif_profiles, motif_colors = build_motif_profiles(selected_df, args)

    print("Computing saliency...")
    all_rows = []
    info_rows = []

    for local_idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Saliency"):
        profiles, info = compute_saliency_for_example(
            model=model,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            row=row,
            args=args,
            device=device,
        )

        example_id = row.get("example_index", local_idx)

        for profile_name, scores in profiles.items():
            add_saliency_rows(
                all_rows=all_rows,
                example_id=example_id,
                row=row,
                profile_name=profile_name,
                scores=scores,
                args=args,
                info=info,
            )

        info_rows.append({
            "example_id": example_id,
            "saliency_group": row["saliency_group"],
            "label": row.get("label", np.nan),
            "confusion_category": row.get("confusion_category", np.nan),
            **info,
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    sal_df = pd.DataFrame(all_rows)
    info_df = pd.DataFrame(info_rows)

    sal_path = output_dir / "saliency_profiles_long.csv"
    info_path = output_dir / "saliency_example_info.csv"

    sal_df.to_csv(sal_path, index=False)
    info_df.to_csv(info_path, index=False)

    print(f"Saved {sal_path}")
    print(f"Saved {info_path}")

    value_tables = build_value_tables(sal_df, args)

    for value_name, value_df in value_tables.items():
        print(f"Plotting {value_name}...")

        mean_df = mean_profile(value_df, args)
        mean_path = output_dir / f"{value_name}_mean_saliency.csv"
        mean_df.to_csv(mean_path, index=False)
        print(f"Saved {mean_path}")

        plot_mean_saliency(
            mean_df=mean_df,
            value_name=value_name,
            output_dir=output_dir,
            args=args,
            motif_profiles=motif_profiles,
            motif_colors=motif_colors,
        )

        plot_heatmaps(
            value_df=value_df,
            value_name=value_name,
            output_dir=output_dir,
            args=args,
        )

        compute_motif_correlations(
            mean_df=mean_df,
            value_name=value_name,
            output_dir=output_dir,
            args=args,
            motif_profiles=motif_profiles,
        )

    config = vars(args)
    config["model_file"] = str(model_file)

    with open(output_dir / "saliency_motif_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()