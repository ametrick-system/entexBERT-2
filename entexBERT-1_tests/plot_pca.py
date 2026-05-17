#!/usr/bin/env python

"""
Make paper-style PCA plots for DNABERT/entexBERT fine-tuned models.

Panels:
  left:  true labels, AS vs non-AS
  right: prediction categories, TP/FP/FN/TN

Default representation:
  attention-score vectors, following the entexBERT visualize() logic:
  last-layer CLS attention to k-mer tokens -> projected back to base positions.
"""

from __future__ import print_function

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Defaults matched to your current run
# ---------------------------------------------------------------------

DEFAULT_PROJECT_DIR = os.path.expanduser("~/entexBERT-2/entexBERT-1_tests")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_PROJECT_DIR, "data", "ctcf_enc01_compare_uncapped")
DEFAULT_RESULT_DIR = os.path.join(DEFAULT_PROJECT_DIR, "results", "ctcf_enc01_compare_uncapped")


# ---------------------------------------------------------------------
# Import local DNABERT / entexBERT code
# ---------------------------------------------------------------------

def add_repo_paths(project_dir):
    dnabert_root = os.path.join(project_dir, "external", "DNABERT")
    examples_dir = os.path.join(dnabert_root, "examples")
    src_dir = os.path.join(dnabert_root, "src")

    sys.path.insert(0, examples_dir)
    sys.path.insert(0, src_dir)


# ---------------------------------------------------------------------
# Data loading through original entexBERT/DNABERT utilities
# ---------------------------------------------------------------------

def make_load_args(data_dir, checkpoint_dir, model_type, max_seq_length, split):
    # load_and_cache_examples uses:
    # local_rank, data_dir, model_name_or_path, max_seq_length, task_name,
    # do_predict, overwrite_cache, n_process, model_type.
    return SimpleNamespace(
        local_rank=-1,
        data_dir=data_dir,
        model_name_or_path=checkpoint_dir,
        max_seq_length=max_seq_length,
        task_name="dnaprom",
        do_predict=True,
        overwrite_cache=True,
        n_process=1,
        model_type=model_type,
    )


def split_to_flags(split):
    if split == "val":
        return True, False
    if split == "test":
        return False, True
    if split == "train":
        return False, False
    raise ValueError("split must be one of: train, val, test")


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model_and_tokenizer(exp, max_seq_length):
    from entexbert_ft import MODEL_CLASSES

    model_type = exp["model_type"]
    checkpoint_dir = exp["checkpoint_dir"]
    tokenizer_name = exp["tokenizer_name"]
    kmer = exp["kmer"]

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else checkpoint_dir,
        do_lower_case=False,
    )

    config = config_class.from_pretrained(
        checkpoint_dir,
        num_labels=2,
        finetuning_task="dnaprom",
    )

    # Important for BertForSNPClassification.
    config.k = kmer

    # Important for attention PCA.
    config.output_attentions = True

    # Keep these consistent with your fine-tune config.
    if not hasattr(config, "split"):
        config.split = int(max_seq_length / 512)
    if not hasattr(config, "rnn"):
        config.rnn = "lstm"
    if not hasattr(config, "num_rnn_layer"):
        config.num_rnn_layer = 2
    if not hasattr(config, "rnn_dropout"):
        config.rnn_dropout = 0.0
    if not hasattr(config, "rnn_hidden"):
        config.rnn_hidden = 768

    model = model_class.from_pretrained(checkpoint_dir, config=config)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------
# Attention-vector extraction
# ---------------------------------------------------------------------

def attention_to_base_scores(attention_score, kmer):
    """
    Match the original entexBERT visualize() logic.

    attention_score shape:
        num_heads x seq_len x seq_len

    It takes attention_score[:, 0, i], i.e. all-head attention
    from CLS/source position 0 to token position i.
    Then it projects overlapping k-mer scores back to base positions.
    """

    attn_score = []

    # This mirrors:
    # for i in range(1, attention_score.shape[-1] - kmer + 2):
    #     attn_score.append(float(attention_score[:, 0, i].sum()))
    for i in range(1, attention_score.shape[-1] - kmer + 2):
        attn_score.append(float(attention_score[:, 0, i].sum()))

    # Original code zeroes at the first apparent padding transition.
    for i in range(len(attn_score) - 1):
        if attn_score[i + 1] == 0:
            attn_score[i] = 0
            break

    counts = np.zeros([len(attn_score) + kmer - 1], dtype=np.float32)
    real_scores = np.zeros([len(attn_score) + kmer - 1], dtype=np.float32)

    for i, score in enumerate(attn_score):
        for j in range(kmer):
            counts[i + j] += 1.0
            real_scores[i + j] += score

    counts[counts == 0] = 1.0
    real_scores = real_scores / counts

    norm = np.linalg.norm(real_scores)
    if norm > 0:
        real_scores = real_scores / norm

    return real_scores


@torch.no_grad()
def extract_attention_pca_inputs(exp, split, max_seq_length, batch_size, device):
    from entexbert_ft import load_and_cache_examples, TOKEN_ID_GROUP

    model, tokenizer = load_model_and_tokenizer(exp, max_seq_length)
    model.to(device)

    evaluate, test = split_to_flags(split)

    load_args = make_load_args(
        data_dir=exp["data_dir"],
        checkpoint_dir=exp["checkpoint_dir"],
        model_type=exp["model_type"],
        max_seq_length=max_seq_length,
        split=split,
    )

    dataset = load_and_cache_examples(
        load_args,
        "dnaprom",
        tokenizer,
        evaluate=evaluate,
        test=test,
    )

    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    all_scores = []
    all_labels = []
    all_preds = []
    all_probs = []

    softmax = torch.nn.Softmax(dim=1)

    for batch in loader:
        batch = tuple(t.to(device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }

        if exp["model_type"] in TOKEN_ID_GROUP:
            inputs["token_type_ids"] = batch[2]
        else:
            inputs["token_type_ids"] = None

        outputs = model(**inputs)
        logits = outputs[1]

        # With output_attentions=True, outputs[-1] is the tuple of layer attentions.
        # The original entexBERT visualize() uses outputs[-1][-1], the last layer.
        last_layer_attention = outputs[-1][-1]
        # shape: batch x heads x seq_len x seq_len

        probs = softmax(logits)[:, 1]
        preds = torch.argmax(logits, dim=1)

        attn_np = last_layer_attention.detach().cpu().numpy()
        for i in range(attn_np.shape[0]):
            scores = attention_to_base_scores(attn_np[i], exp["kmer"])
            all_scores.append(scores)

        all_labels.append(batch[3].detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    X = np.vstack(all_scores)
    y = np.concatenate(all_labels)
    yhat = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    return X, y, yhat, probs


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

COLORS = {
    "AS": "#7a1f8a",       # purple
    "non-AS": "#00cfe8",   # cyan
    "TP": "#0000ff",       # blue
    "FP": "#087a08",       # green
    "FN": "#d99a00",       # golden/orange
    "TN": "#ff0000",       # red
}


def confusion_categories(y, yhat):
    out = []
    for yt, yp in zip(y, yhat):
        if yt == 1 and yp == 1:
            out.append("TP")
        elif yt == 0 and yp == 1:
            out.append("FP")
        elif yt == 1 and yp == 0:
            out.append("FN")
        else:
            out.append("TN")
    return np.array(out)


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#666666")
    ax.grid(False)


def plot_label_panel(ax, Z, y):
    pos = y == 1
    neg = y == 0

    ax.scatter(
        Z[pos, 0], Z[pos, 1],
        s=24, alpha=0.28, c=COLORS["AS"], label="AS",
        edgecolors="none",
    )
    ax.scatter(
        Z[neg, 0], Z[neg, 1],
        s=24, alpha=0.28, c=COLORS["non-AS"], label="non-AS",
        edgecolors="none",
    )

    ax.legend(loc="upper right", frameon=True, fontsize=9)
    style_axis(ax)


def plot_confusion_panel(ax, Z, y, yhat):
    cats = confusion_categories(y, yhat)

    for name in ["TP", "FP", "FN", "TN"]:
        mask = cats == name
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=24, alpha=0.28, c=COLORS[name], label=name,
            edgecolors="none",
        )

    ax.legend(loc="upper right", frameon=True, fontsize=9)
    style_axis(ax)


def add_row_titles(fig, axes, titles):
    # Call after tight_layout so axes positions are final.
    for i, title in enumerate(titles):
        left_box = axes[i, 0].get_position()
        right_box = axes[i, 1].get_position()
        x = 0.5 * (left_box.x0 + right_box.x1)
        y = max(left_box.y1, right_box.y1) + 0.025
        fig.text(x, y, title, ha="center", va="bottom", fontsize=15, color="#555555")


def maybe_subsample(X, y, yhat, probs, max_points, seed):
    if max_points is None or max_points <= 0 or X.shape[0] <= max_points:
        return X, y, yhat, probs

    rng = np.random.RandomState(seed)
    idx = rng.choice(np.arange(X.shape[0]), size=max_points, replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx], yhat[idx], probs[idx]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_jitter", action="store_true")

    parser.add_argument(
        "--out_png",
        default=os.path.join(DEFAULT_RESULT_DIR, "pca_attention_like_paper.png"),
    )
    parser.add_argument(
        "--out_pdf",
        default=os.path.join(DEFAULT_RESULT_DIR, "pca_attention_like_paper.pdf"),
    )

    args = parser.parse_args()

    add_repo_paths(args.project_dir)

    # Import after paths are added.
    import entexbert_ft  # noqa: F401

    # Default: exactly paper-like two-row layout.
    # Add --include_jitter for a third row.
    experiments = [
        {
            "row_title": "CTCF, Individual 1 (entexBERT)",
            "model_type": "dnasnp",
            "checkpoint_dir": os.path.join(args.result_dir, "entexbert_dnasnp_centered", "model"),
            "data_dir": os.path.join(args.data_dir, "offset_0"),
            "tokenizer_name": "dna3",
            "kmer": 3,
        },
        {
            "row_title": "CTCF, Individual 1 (DNABERT)",
            "model_type": "dna",
            "checkpoint_dir": os.path.join(args.result_dir, "dnabert1_cls_centered", "model"),
            "data_dir": os.path.join(args.data_dir, "offset_0"),
            "tokenizer_name": "dna3",
            "kmer": 3,
        },
    ]

    if args.include_jitter:
        experiments.append(
            {
                "row_title": "CTCF, Individual 1 (entexBERT, jitter64)",
                "model_type": "dnasnp",
                "checkpoint_dir": os.path.join(args.result_dir, "entexbert_dnasnp_jitter64", "model"),
                "data_dir": os.path.join(args.data_dir, "offset_jitter"),
                "tokenizer_name": "dna3",
                "kmer": 3,
            }
        )

    for exp in experiments:
        if not os.path.isdir(exp["checkpoint_dir"]):
            raise RuntimeError("Missing checkpoint_dir: {}".format(exp["checkpoint_dir"]))
        if not os.path.isdir(exp["data_dir"]):
            raise RuntimeError("Missing data_dir: {}".format(exp["data_dir"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    nrows = len(experiments)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(12.0, 4.8 * nrows),
        squeeze=False,
    )

    row_titles = []

    for row_idx, exp in enumerate(experiments):
        print("=" * 80)
        print("Experiment:", exp["row_title"])
        print("model_type:", exp["model_type"])
        print("checkpoint:", exp["checkpoint_dir"])
        print("data:", exp["data_dir"])
        print("split:", args.split)

        X, y, yhat, probs = extract_attention_pca_inputs(
            exp=exp,
            split=args.split,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            device=device,
        )

        X, y, yhat, probs = maybe_subsample(
            X, y, yhat, probs,
            max_points=args.max_points,
            seed=args.seed,
        )

        print("num examples:", X.shape[0])
        print("attention vector dim:", X.shape[1])
        print("true positive fraction:", float(np.mean(y)))
        print("pred positive fraction:", float(np.mean(yhat)))

        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(X)
        print("PCA explained variance:", pca.explained_variance_ratio_)

        plot_label_panel(axes[row_idx, 0], Z, y)
        plot_confusion_panel(axes[row_idx, 1], Z, y, yhat)

        row_titles.append(exp["row_title"])

    # Big panel label like the paper.
    fig.text(0.018, 0.985, "A", ha="left", va="top", fontsize=26, fontweight="bold", color="#555555")

    plt.tight_layout(rect=[0.03, 0.02, 0.995, 0.96], h_pad=4.0, w_pad=2.0)
    add_row_titles(fig, axes, row_titles)

    out_dir = os.path.dirname(args.out_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")

    print("=" * 80)
    print("Saved:")
    print(args.out_png)
    print(args.out_pdf)


if __name__ == "__main__":
    main()