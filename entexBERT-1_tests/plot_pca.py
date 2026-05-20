#!/usr/bin/env python
# coding: utf-8

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
import matplotlib.patches as mpatches


DEFAULT_PROJECT_DIR = os.path.expanduser("~/entexBERT-2/entexBERT-1_tests")
DEFAULT_DATA_DIR = os.path.join(DEFAULT_PROJECT_DIR, "data", "ctcf_enc01_compare_10k")
DEFAULT_RESULT_DIR = os.path.join(DEFAULT_PROJECT_DIR, "results", "ctcf_enc01_compare_10k")


def add_repo_paths(project_dir):
    dnabert_root = os.path.join(project_dir, "external", "DNABERT")
    sys.path.insert(0, os.path.join(dnabert_root, "examples"))
    sys.path.insert(0, os.path.join(dnabert_root, "src"))


def split_to_flags(split):
    if split == "val":
        return True, False
    if split == "test":
        return False, True
    if split == "train":
        return False, False
    raise ValueError("split must be train, val, or test")


def make_loader_args(data_dir, checkpoint_dir, model_type, max_seq_length):
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


def load_model_and_tokenizer(exp, max_seq_length):
    from entexbert_ft import MODEL_CLASSES

    config_class, model_class, tokenizer_class = MODEL_CLASSES[exp["model_type"]]

    tokenizer = tokenizer_class.from_pretrained(
        exp["tokenizer_name"],
        do_lower_case=False,
    )

    config = config_class.from_pretrained(
        exp["checkpoint_dir"],
        num_labels=2,
        finetuning_task="dnaprom",
    )

    config.k = exp["kmer"]

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

    model = model_class.from_pretrained(exp["checkpoint_dir"], config=config)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def extract_pooled_embeddings(exp, split, max_seq_length, batch_size, device):
    from entexbert_ft import load_and_cache_examples, TOKEN_ID_GROUP

    model, tokenizer = load_model_and_tokenizer(exp, max_seq_length)
    model.to(device)

    evaluate, test = split_to_flags(split)
    loader_args = make_loader_args(
        data_dir=exp["data_dir"],
        checkpoint_dir=exp["checkpoint_dir"],
        model_type=exp["model_type"],
        max_seq_length=max_seq_length,
    )

    dataset = load_and_cache_examples(
        loader_args,
        "dnaprom",
        tokenizer,
        evaluate=evaluate,
        test=test,
    )

    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
    )

    all_embeddings = []
    all_labels = []
    all_preds = []
    all_probs = []

    softmax = torch.nn.Softmax(dim=1)

    for batch in loader:
        batch = tuple(t.to(device) for t in batch)

        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[2]
        labels = batch[3]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if exp["model_type"] in TOKEN_ID_GROUP:
            model_inputs["token_type_ids"] = token_type_ids
            bert_token_type_ids = token_type_ids
        else:
            model_inputs["token_type_ids"] = None
            bert_token_type_ids = None

        # Predictions use the actual fine-tuned classifier head.
        outputs = model(**model_inputs)
        logits = outputs[1]
        probs = softmax(logits)[:, 1]
        preds = torch.argmax(logits, dim=1)

        # Embeddings mimic paper's embeddings.py: BertModel output[1].
        bert_outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        pooled_embedding = bert_outputs[1]

        all_embeddings.append(pooled_embedding.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    X = np.vstack(all_embeddings)
    y = np.concatenate(all_labels)
    yhat = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    return X, y, yhat, probs


def confusion_labels(y, yhat):
    labels = []
    for true, pred in zip(y, yhat):
        if int(true) == 1 and int(pred) == 1:
            labels.append("TP")
        elif int(true) == 0 and int(pred) == 1:
            labels.append("FP")
        elif int(true) == 1 and int(pred) == 0:
            labels.append("FN")
        else:
            labels.append("TN")
    return np.array(labels)


def plot_true_labels(ax, Z, y):
    color_map = {
        1: "purple",
        0: "#00e4ff",
    }

    colors = [color_map[int(label)] for label in y]

    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.15, s=24, c=colors, edgecolors="none")

    handles = [
        mpatches.Patch(color=color_map[1], label="AS"),
        mpatches.Patch(color=color_map[0], label="non-AS"),
    ]

    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)
    ax.grid(False)


def plot_confusion(ax, Z, y, yhat):
    labels_4 = confusion_labels(y, yhat)

    color_map_4 = {
        "TP": "blue",
        "FP": "green",
        "FN": "goldenrod",
        "TN": "red",
    }

    colors = [color_map_4[label] for label in labels_4]

    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.15, s=24, c=colors, edgecolors="none")

    handles = [
        mpatches.Patch(color=color_map_4[label], label=label)
        for label in ["TP", "FP", "FN", "TN"]
    ]

    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)
    ax.grid(False)


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#555555")
    ax.tick_params(axis="both", labelsize=9)


def add_row_titles(fig, axes, row_titles):
    for i, title in enumerate(row_titles):
        left_box = axes[i, 0].get_position()
        right_box = axes[i, 1].get_position()
        x = 0.5 * (left_box.x0 + right_box.x1)
        y = max(left_box.y1, right_box.y1) + 0.030
        fig.text(x, y, title, ha="center", va="bottom", fontsize=15, color="#555555")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--include_jitter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out_png",
        default=os.path.join(DEFAULT_RESULT_DIR, "embedding_pca_like_paper.png"),
    )
    parser.add_argument(
        "--out_pdf",
        default=os.path.join(DEFAULT_RESULT_DIR, "embedding_pca_like_paper.pdf"),
    )

    args = parser.parse_args()

    add_repo_paths(args.project_dir)

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
        print(exp["row_title"])
        print("model_type:", exp["model_type"])
        print("checkpoint:", exp["checkpoint_dir"])
        print("data:", exp["data_dir"])
        print("split:", args.split)

        X, y, yhat, probs = extract_pooled_embeddings(
            exp=exp,
            split=args.split,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            device=device,
        )

        print("num examples:", len(y))
        print("positive fraction:", float(np.mean(y)))
        print("pred positive fraction:", float(np.mean(yhat)))

        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(X)

        print("PCA explained variance:", pca.explained_variance_ratio_)

        plot_true_labels(axes[row_idx, 0], Z, y)
        plot_confusion(axes[row_idx, 1], Z, y, yhat)

        style_axes(axes[row_idx, 0])
        style_axes(axes[row_idx, 1])

        row_titles.append(exp["row_title"])

    fig.text(
        0.018,
        0.985,
        "A",
        ha="left",
        va="top",
        fontsize=26,
        fontweight="bold",
        color="#555555",
    )

    plt.tight_layout(rect=[0.03, 0.02, 0.995, 0.95], h_pad=5.0, w_pad=2.0)
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