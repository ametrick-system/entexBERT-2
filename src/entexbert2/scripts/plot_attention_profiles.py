#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction

# -----------------
# Argument parsing
# -----------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot entexBERT-2 attention profiles. "
            "Supports sequence-pair or single-sequence input, chosen source token, "
            "chosen layers/heads, exact ALiBi removal, zooming, and profile correction."
        )
    )

    # Required paths
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--examples_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to local inference DNABERT-2 model directory (with attention extracted from each layer)",
    )

    # Input/data options
    parser.add_argument(
        "--input_mode",
        default="ref_single",
        choices=["ref_single", "hap_pair", "ref_hap1_pair", "ref_hap2_pair"],
    )
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument(
        "--left_bp",
        type=int,
        default=256,
        help="Left offset coordinate of sequence element of interest (e.g. SNV or ChIP-seq peak) within the raw input sequence",
    )

    parser.add_argument(
        "--categories",
        default="TP,FP,TN,FN",
        help="Comma-separated confusion categories to include.",
    )
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=100,
        help="Top N examples per confusion category.",
    )
    parser.add_argument(
        "--deduplicate_inputs",
        action="store_true",
        help="Drop duplicate sequence inputs within each category before selecting top N.",
    )

    # Attention source/options
    parser.add_argument(
        "--source_token",
        default="cls",
        choices=[
            "cls",
            "snv",
            "ref_snv",
            "hap1_snv",
            "hap2_snv",
            "token_index",
            "ref_position",
            "hap1_position",
            "hap2_position",
        ],
        help=(
            "Token from which attention is taken. "
            "For ref_single, choose from cls/snv/ref_snv/token_index/ref_position; "
            "For hap_pair, choose from cls/hap1_snv/hap2_snv/token_index/hap1_position/hap2_position; "
            "For ref_hap1_pair choose from cls/hap1_snv/ref_snv/token_index/hap1_position/ref_position; "
            "For ref_hap2_pair choose from cls/hap2_snv/ref_snv/token_index/hap2_position/ref_position."
        ),
    )
    parser.add_argument(
        "--source_token_index",
        type=int,
        default=None,
        help="Used only with --source_token token_index.",
    )
    parser.add_argument(
        "--source_base_pos",
        type=int,
        default=None,
        help=(
            "Used with --source_token ref_position/hap1_position/hap2_position. "
            "Defaults to --left_bp."
        ),
    )

    parser.add_argument(
        "--layers",
        default="last",
        help=(
            "Layers to average. Examples: 'last', '11', '8,9,10,11', '8-11', 'all'. "
            "Uses zero-based layer indices."
        ),
    )
    parser.add_argument(
        "--heads",
        default="all",
        help=(
            "Heads to average. Examples: 'all', '0', '0,3,7', '0-3'. "
            "Uses zero-based head indices."
        ),
    )
    parser.add_argument(
        "--remove_alibi",
        action="store_true",
        help=(
            "Exactly remove head-specific ALiBi linear distance penalties from attention scores before averaging."
        ),
    )

    # Plotting/options
    parser.add_argument(
        "--plot_values",
        default="all",
        help=(
            "Comma-separated profiles to plot. For hap_pair: all,hap1,hap2,hap1_plus_hap2,hap1_minus_hap2. "
            "For ref_single: all,ref."
        ),
    )
    parser.add_argument(
        "--plot_window_bp",
        type=int,
        default=100,
        help="Plot only +/- this many bp around SNV. Use -1 for full sequence.",
    )
    parser.add_argument(
        "--profile_correction",
        default="none",
        choices=[
            "none",
            "position_mean",
            "linear_flank_per_category",
            "linear_flank_global",
        ],
        help=(
            "Optional correction applied after mapping attention to base positions. "
            "This is separate from exact ALiBi removal."
        ),
    )
    parser.add_argument(
        "--flank_inner_bp",
        type=int,
        default=50,
        help="Inner edge of flank region for linear flank correction.",
    )
    parser.add_argument(
        "--flank_outer_bp",
        type=int,
        default=100,
        help="Outer edge of flank region for linear flank correction.",
    )
    parser.add_argument("--dpi", type=int, default=300)

    # Model/training args that must match fine-tuning run
    parser.add_argument("--pooling_mode", default="cls")
    parser.add_argument("--head_num_layers", type=int, default=1)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--head_activation", default="gelu")
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()

# -------------------------
# Helpers
# -------------------------
def sanitize_for_filename(s):
    return str(s).replace(",", "_").replace("-", "to").replace(" ", "")


def parse_index_spec(spec, max_value):
    """
    Parse layer/head specs:
      'all'
      'last'
      '11'
      '8,9,10,11'
      '8-11'
      '-1'
    """
    spec = str(spec).strip()

    if spec == "all":
        return list(range(max_value))

    if spec == "last":
        return [max_value - 1]

    values = []

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if re.fullmatch(r"\d+\-\d+", part):
            a, b = part.split("-")
            a, b = int(a), int(b)
            values.extend(list(range(a, b + 1)))
        else:
            x = int(part)
            if x < 0:
                x = max_value + x
            values.append(x)

    values = sorted(set(values))

    for x in values:
        if x < 0 or x >= max_value:
            raise ValueError(f"Index {x} out of range for max_value={max_value}")

    return values


def get_alibi_head_slopes(n_heads):
    """
    Exact copy of the DNABERT-2 / MosaicBERT ALiBi slope logic.
    """

    def get_slopes_power_of_2(n_heads_inner):
        start = 2 ** (-(2 ** -(math.log2(n_heads_inner) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n_heads_inner)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes_a = get_slopes_power_of_2(closest_power_of_2)
    slopes_b = get_alibi_head_slopes(2 * closest_power_of_2)
    slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
    return slopes_a + slopes_b


def remove_alibi_from_attention_probs(
    attn_selected_heads,
    head_indices,
    source_idx,
    valid_key_mask,
    n_total_heads,
):
    """
    Remove ALiBi exactly from already-softmaxed attention probabilities.

    Model computes:
        p_j = softmax(content_j - slope_h * abs(j - source_idx))

    Therefore:
        content_softmax_j ∝ p_j * exp(slope_h * abs(j - source_idx))

    This must be done per head before averaging heads.
    """
    device = attn_selected_heads.device
    dtype = attn_selected_heads.dtype

    slopes = torch.tensor(
        get_alibi_head_slopes(n_total_heads),
        device=device,
        dtype=dtype,
    )
    selected_slopes = slopes[head_indices].unsqueeze(1)

    seq_len = attn_selected_heads.shape[-1]
    key_positions = torch.arange(seq_len, device=device, dtype=dtype)
    distances = torch.abs(key_positions - float(source_idx)).unsqueeze(0)

    eps = torch.finfo(dtype).tiny
    log_probs = torch.log(attn_selected_heads.clamp_min(eps))

    debiased_logits = log_probs + selected_slopes * distances

    valid_key_mask = torch.as_tensor(valid_key_mask, device=device, dtype=torch.bool)
    debiased_logits = debiased_logits.masked_fill(
        ~valid_key_mask.unsqueeze(0),
        -1e9,
    )

    return torch.softmax(debiased_logits, dim=-1)


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


def select_examples(df, args):
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
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

        if args.deduplicate_inputs:
            before = len(group)
            if args.input_mode == "hap_pair":
                group = group.drop_duplicates(subset=["sequence1", "sequence2"]).copy()
            else:
                group = group.drop_duplicates(subset=["sequence"]).copy()
            after = len(group)
            print(f"{category}: deduplicated {before} -> {after}")

        group = group.head(args.n_per_category).copy()
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

    return hits[0] if hits else None


def get_source_token_index(args, sequence_ids, offsets):
    snv_pos = args.left_bp
    base_pos = args.source_base_pos if args.source_base_pos is not None else snv_pos

    if args.source_token == "cls":
        return 0

    if args.source_token == "token_index":
        if args.source_token_index is None:
            raise ValueError("--source_token_index is required with --source_token token_index")
        return int(args.source_token_index)

    if args.source_token in {"snv", "ref_snv"}:
        idx = find_token_containing_position(sequence_ids, offsets, 0, snv_pos)
        if idx is None:
            raise ValueError("Could not find SNV token in sequence 0.")
        return idx

    if args.source_token == "hap1_snv":
        idx = find_token_containing_position(sequence_ids, offsets, 0, snv_pos)
        if idx is None:
            raise ValueError("Could not find hap1 SNV token.")
        return idx

    if args.source_token == "hap2_snv":
        idx = find_token_containing_position(sequence_ids, offsets, 1, snv_pos)
        if idx is None:
            raise ValueError("Could not find hap2 SNV token.")
        return idx

    if args.source_token in {"ref_position", "hap1_position"}:
        idx = find_token_containing_position(sequence_ids, offsets, 0, base_pos)
        if idx is None:
            raise ValueError(f"Could not find sequence-0 token containing base {base_pos}.")
        return idx

    if args.source_token == "hap2_position":
        idx = find_token_containing_position(sequence_ids, offsets, 1, base_pos)
        if idx is None:
            raise ValueError(f"Could not find hap2 token containing base {base_pos}.")
        return idx

    raise ValueError(f"Unsupported source_token: {args.source_token}")


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

        if sid is None or end <= start:
            continue

        score = float(score)

        if sid == 0:
            start = max(0, int(start))
            end = min(seq1_len, int(end))
            if end > start:
                seq1_scores[start:end] += score
                seq1_counts[start:end] += 1.0

        elif sid == 1 and seq2_len is not None:
            start = max(0, int(start))
            end = min(seq2_len, int(end))
            if end > start:
                seq2_scores[start:end] += score
                seq2_counts[start:end] += 1.0

    valid1 = seq1_counts > 0
    seq1_scores[valid1] /= seq1_counts[valid1]

    if seq2_len is not None:
        valid2 = seq2_counts > 0
        seq2_scores[valid2] /= seq2_counts[valid2]

    return seq1_scores, seq2_scores


def compute_attention_profile_for_example(model, tokenizer, row, args, device):
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

    source_idx = get_source_token_index(args, sequence_ids, offsets)

    if source_idx < 0 or source_idx >= len(tokens):
        raise ValueError(f"source_idx={source_idx} out of range for {len(tokens)} tokens")

    valid_key_mask = encoding["attention_mask"][0].bool()

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

    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions.")

    attentions = outputs.attentions
    layer_indices = parse_index_spec(args.layers, max_value=len(attentions))

    n_total_heads = attentions[0].shape[1]
    head_indices = parse_index_spec(args.heads, max_value=n_total_heads)

    selected_layer_scores = []

    for layer_idx in layer_indices:
        # [batch, heads, seq, seq] -> [heads, seq, seq]
        attn_layer = attentions[layer_idx][0]

        # attention FROM source_idx TO all tokens
        attn_selected_heads = attn_layer[head_indices, source_idx, :]

        if args.remove_alibi:
            attn_selected_heads = remove_alibi_from_attention_probs(
                attn_selected_heads=attn_selected_heads,
                head_indices=head_indices,
                source_idx=source_idx,
                valid_key_mask=valid_key_mask.to(device),
                n_total_heads=n_total_heads,
            )

        # Average selected heads for this layer
        selected_layer_scores.append(attn_selected_heads.mean(dim=0))

    # Average selected layers
    token_scores = (
        torch.stack(selected_layer_scores, dim=0)
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    seq1_scores, seq2_scores = token_scores_to_base_profiles(
        token_scores=token_scores,
        sequence_ids=sequence_ids,
        offsets=offsets,
        seq1_len=len(seq1),
        seq2_len=len(seq2) if seq2 is not None else None,
    )

    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

    source_start, source_end = offsets[source_idx]

    info = {
        "source_token_index": int(source_idx),
        "source_token": str(tokens[source_idx]),
        "source_offset_start": int(source_start),
        "source_offset_end": int(source_end),
        "num_tokens": int(len(tokens)),
        "num_layers": int(len(attentions)),
        "num_heads": int(n_total_heads),
        "selected_layers": ",".join(map(str, layer_indices)),
        "selected_heads": ",".join(map(str, head_indices)),
        "prob_positive_recomputed": float(probs[1]),
        "logit_0": float(outputs.logits[0, 0].detach().cpu()),
        "logit_1": float(outputs.logits[0, 1].detach().cpu()),
    }

    if args.input_mode == "hap_pair":
        profiles = {
            "hap1": seq1_scores,
            "hap2": seq2_scores,
        }
    else:
        profiles = {
            "ref": seq1_scores,
        }

    return profiles, info


def add_profile_rows(all_rows, example_id, row, profile_name, scores, args, info):
    for pos, value in enumerate(scores):
        all_rows.append({
            "example_id": example_id,
            "confusion_category": row["confusion_category"],
            "label": int(row["label"]),
            "pred_label": int(row["pred_label"]),
            "profile": profile_name,
            "position": pos,
            "position_relative_to_snv": pos - args.left_bp,
            "attention": float(value),
            "source_token_index": info["source_token_index"],
            "source_token": info["source_token"],
            "source_offset_start": info["source_offset_start"],
            "source_offset_end": info["source_offset_end"],
            "num_tokens": info["num_tokens"],
            "prob_positive_recomputed": info["prob_positive_recomputed"],
        })


def build_value_tables(attn_df, args):
    available_profiles = set(attn_df["profile"].unique())

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
            if value not in available_profiles:
                print(f"Skipping {value}: not available.")
                continue

            sub = attn_df[attn_df["profile"] == value].copy()
            sub = sub.rename(columns={"attention": "value"})
            tables[value] = sub[
                [
                    "example_id",
                    "confusion_category",
                    "position_relative_to_snv",
                    "value",
                ]
            ].copy()

        elif value in {"hap1_plus_hap2", "hap1_minus_hap2"}:
            if not {"hap1", "hap2"}.issubset(available_profiles):
                print(f"Skipping {value}: hap1/hap2 not both available.")
                continue

            pivot = attn_df.pivot_table(
                index=["example_id", "confusion_category", "position_relative_to_snv"],
                columns="profile",
                values="attention",
                aggfunc="mean",
            ).reset_index()

            pivot = pivot.dropna(subset=["hap1", "hap2"]).copy()

            if value == "hap1_plus_hap2":
                pivot["value"] = pivot["hap1"] + pivot["hap2"]
            else:
                pivot["value"] = pivot["hap1"] - pivot["hap2"]

            tables[value] = pivot[
                [
                    "example_id",
                    "confusion_category",
                    "position_relative_to_snv",
                    "value",
                ]
            ].copy()

        else:
            raise ValueError(f"Unsupported plot value: {value}")

    return tables


def filter_plot_window(df, plot_window_bp):
    if plot_window_bp is None or plot_window_bp < 0:
        return df.copy()

    return df[
        df["position_relative_to_snv"].between(-plot_window_bp, plot_window_bp)
    ].copy()


def compute_mean_profiles(value_df, categories, plot_window_bp):
    sub = value_df[value_df["confusion_category"].isin(categories)].copy()
    sub = filter_plot_window(sub, plot_window_bp)

    profiles = (
        sub.groupby(["confusion_category", "position_relative_to_snv"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_attention"})
    )

    return profiles


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise ValueError("Need at least two finite points to fit line.")

    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def apply_profile_correction(profiles, args):
    profiles = profiles.copy()

    if args.profile_correction == "none":
        profiles["plot_attention"] = profiles["mean_attention"]
        profiles["baseline"] = 0.0
        return profiles, pd.DataFrame()

    if args.profile_correction == "position_mean":
        baseline = (
            profiles.groupby("position_relative_to_snv")["mean_attention"]
            .mean()
            .rename("baseline")
            .reset_index()
        )

        profiles = profiles.merge(baseline, on="position_relative_to_snv")
        profiles["plot_attention"] = profiles["mean_attention"] - profiles["baseline"]
        baseline_params = pd.DataFrame([{
            "correction": "position_mean",
            "description": "Subtract all-category positional mean at each position",
        }])
        return profiles, baseline_params

    is_flank = profiles["position_relative_to_snv"].abs().between(
        args.flank_inner_bp,
        args.flank_outer_bp,
        inclusive="both",
    )
    flank_df = profiles[is_flank].copy()

    if flank_df.empty:
        raise ValueError(
            f"No flank points found for {args.flank_inner_bp}-{args.flank_outer_bp} bp."
        )

    corrected_parts = []
    baseline_rows = []

    if args.profile_correction == "linear_flank_global":
        slope, intercept = fit_line(
            flank_df["position_relative_to_snv"],
            flank_df["mean_attention"],
        )

        for category, group in profiles.groupby("confusion_category"):
            group = group.copy()
            group["baseline"] = slope * group["position_relative_to_snv"] + intercept
            group["plot_attention"] = group["mean_attention"] - group["baseline"]
            corrected_parts.append(group)

        baseline_rows.append({
            "correction": args.profile_correction,
            "confusion_category": "GLOBAL",
            "slope": slope,
            "intercept": intercept,
            "n_flank_points": int(len(flank_df)),
        })

    elif args.profile_correction == "linear_flank_per_category":
        for category, group in profiles.groupby("confusion_category"):
            group = group.copy()
            flank_group = group[
                group["position_relative_to_snv"].abs().between(
                    args.flank_inner_bp,
                    args.flank_outer_bp,
                    inclusive="both",
                )
            ].copy()

            if len(flank_group) < 2:
                print(f"Warning: not enough flank points for {category}; skipping.")
                continue

            slope, intercept = fit_line(
                flank_group["position_relative_to_snv"],
                flank_group["mean_attention"],
            )

            group["baseline"] = slope * group["position_relative_to_snv"] + intercept
            group["plot_attention"] = group["mean_attention"] - group["baseline"]
            corrected_parts.append(group)

            baseline_rows.append({
                "correction": args.profile_correction,
                "confusion_category": category,
                "slope": slope,
                "intercept": intercept,
                "n_flank_points": int(len(flank_group)),
            })

    else:
        raise ValueError(f"Unsupported correction: {args.profile_correction}")

    corrected = pd.concat(corrected_parts, ignore_index=True)
    baseline_params = pd.DataFrame(baseline_rows)
    return corrected, baseline_params


def plot_profiles(profiles, value_name, output_dir, args, corrected=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    y_col = "plot_attention" if corrected else "mean_attention"

    for category, group in profiles.groupby("confusion_category"):
        group = group.sort_values("position_relative_to_snv")
        ax.plot(
            group["position_relative_to_snv"],
            group[y_col],
            linewidth=2,
            label=category,
        )

    ax.axvline(0, linestyle="--", linewidth=1)

    if corrected:
        ax.axhline(0, linewidth=0.75)

    if corrected and args.profile_correction.startswith("linear_flank"):
        ax.axvspan(-args.flank_outer_bp, -args.flank_inner_bp, alpha=0.08)
        ax.axvspan(args.flank_inner_bp, args.flank_outer_bp, alpha=0.08)

    window_label = (
        "full"
        if args.plot_window_bp is None or args.plot_window_bp < 0
        else f"pm{args.plot_window_bp}"
    )

    alibi_label = "alibi_removed" if args.remove_alibi else "alibi_on"
    correction_label = args.profile_correction if corrected else "raw"

    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel(
        "Attention" if not corrected else "Attention after profile correction"
    )

    title = (
        f"{value_name} {correction_label} attention ({window_label})\n"
        f"source={args.source_token}, layers={args.layers}, heads={args.heads}, "
        f"{alibi_label}, n/category={args.n_per_category}"
    )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_name = (
        f"{value_name}_{correction_label}_{alibi_label}_"
        f"layers_{sanitize_for_filename(args.layers)}_"
        f"heads_{sanitize_for_filename(args.heads)}_"
        f"source_{sanitize_for_filename(args.source_token)}_"
        f"{window_label}.png"
    )

    out_path = Path(output_dir) / out_name
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {out_path}")


def summarize_windows(profiles, value_name, output_dir):
    df = profiles.copy()

    def assign_region(pos):
        if -10 <= pos <= 10:
            return "snv_pm10"
        if -25 <= pos <= 25:
            return "snv_pm25"
        if -100 <= pos <= -50:
            return "left_flank_100_50"
        if 50 <= pos <= 100:
            return "right_flank_50_100"
        return "other"

    df["region"] = df["position_relative_to_snv"].apply(assign_region)
    df = df[df["region"] != "other"].copy()

    if df.empty:
        return

    agg = (
        df.groupby(["confusion_category", "region"])
        .agg(
            raw_mean_attention=("mean_attention", "mean"),
            plotted_mean_attention=("plot_attention", "mean"),
        )
        .reset_index()
    )

    raw = agg.pivot(
        index="confusion_category",
        columns="region",
        values="raw_mean_attention",
    )
    plotted = agg.pivot(
        index="confusion_category",
        columns="region",
        values="plotted_mean_attention",
    )

    raw.columns = [f"raw_{c}" for c in raw.columns]
    plotted.columns = [f"plotted_{c}" for c in plotted.columns]

    out = pd.concat([raw, plotted], axis=1).reset_index()

    out_path = Path(output_dir) / f"{value_name}_window_summary.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    if args.profile_correction.startswith("linear_flank"):
        if args.flank_inner_bp >= args.flank_outer_bp:
            raise ValueError("--flank_inner_bp must be smaller than --flank_outer_bp")
        if args.plot_window_bp >= 0 and args.flank_outer_bp > args.plot_window_bp:
            raise ValueError("--flank_outer_bp must be <= --plot_window_bp")

    print("Loading examples...")
    df = pd.read_csv(args.examples_csv)
    selected_df = select_examples(df, args)

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

    all_rows = []
    source_rows = []

    print("Computing attention profiles...")
    for local_idx, row in selected_df.iterrows():
        profiles, info = compute_attention_profile_for_example(
            model=model,
            tokenizer=tokenizer,
            row=row,
            args=args,
            device=device,
        )

        example_id = row.get("example_index", local_idx)

        for profile_name, scores in profiles.items():
            add_profile_rows(
                all_rows=all_rows,
                example_id=example_id,
                row=row,
                profile_name=profile_name,
                scores=scores,
                args=args,
                info=info,
            )

        source_rows.append({
            "example_id": example_id,
            "confusion_category": row["confusion_category"],
            "label": int(row["label"]),
            "pred_label": int(row["pred_label"]),
            **info,
        })

        if (local_idx + 1) % 25 == 0 or local_idx == len(selected_df) - 1:
            print(f"  processed {local_idx + 1}/{len(selected_df)}")

    attn_df = pd.DataFrame(all_rows)
    source_df = pd.DataFrame(source_rows)

    attn_path = output_dir / "attention_profiles_long.csv"
    source_path = output_dir / "attention_source_tokens.csv"

    attn_df.to_csv(attn_path, index=False)
    source_df.to_csv(source_path, index=False)

    print(f"Saved {attn_path}")
    print(f"Saved {source_path}")

    value_tables = build_value_tables(attn_df, args)

    for value_name, value_df in value_tables.items():
        print(f"\nPlotting value: {value_name}")

        mean_df = compute_mean_profiles(
            value_df=value_df,
            categories=categories,
            plot_window_bp=args.plot_window_bp,
        )

        mean_path = output_dir / f"{value_name}_mean_profiles.csv"
        mean_df.to_csv(mean_path, index=False)
        print(f"Saved {mean_path}")

        plot_profiles(
            profiles=mean_df.assign(plot_attention=mean_df["mean_attention"]),
            value_name=value_name,
            output_dir=output_dir,
            args=args,
            corrected=False,
        )

        corrected_df, baseline_params = apply_profile_correction(mean_df, args)

        corrected_path = output_dir / f"{value_name}_corrected_profiles.csv"
        baseline_path = output_dir / f"{value_name}_baseline_params.csv"

        corrected_df.to_csv(corrected_path, index=False)
        baseline_params.to_csv(baseline_path, index=False)

        print(f"Saved {corrected_path}")
        print(f"Saved {baseline_path}")

        if args.profile_correction != "none":
            plot_profiles(
                profiles=corrected_df,
                value_name=value_name,
                output_dir=output_dir,
                args=args,
                corrected=True,
            )

        summarize_windows(
            profiles=corrected_df,
            value_name=value_name,
            output_dir=output_dir,
        )

    config = vars(args)
    config["model_file"] = str(model_file)

    with open(output_dir / "attention_profile_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()