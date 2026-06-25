#!/usr/bin/env python3

"""
entexBERT-2 analysis: inference -> metrics -> embeddings/PCA -> representative selection.

Works for binary classification, multiclass classification, and regression. The task and
head config are read from run_config.json (via model_io), so nothing is hardcoded. Outputs:
  - predictions.csv                 (per-example predictions + category + rank + passthrough)
  - metrics.json                    (task-appropriate metrics)
  - pca.csv                         (pooled-embedding PCA coords + target + category)
  - representative_examples_all.csv (+ per-category CSVs) for the attention/saliency plotters

Selection (`--selection_mode auto` picks by task):
  binary       -> confusion: TP/FP/TN/FN, ranked by prediction confidence
  multiclass   -> per_class  : correct_<c>/error_<c>   (good for few classes)   [--multiclass_category_mode]
                  correct_incorrect: correct/incorrect (good for many classes)
                  confusion_cells  : <true>-><pred>
  regression   -> value_extremes : top/bottom by TRUE value (default; entexBERT-R style)
                  residual_extremes: best/worst by |residual|
                  quantile_bins    : bins of TRUE value

torch + model_io are imported lazily inside run_inference, so the analysis functions here are
importable/testable without a GPU stack.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def task_flavor(task, num_labels):
    if task == "regression":
        return "regression"
    return "binary" if num_labels <= 2 else "multiclass"


def detect_input_mode(columns):
    if "sequence1" in columns and "sequence2" in columns:
        return "pair"
    if "sequence" in columns:
        return "single"
    raise ValueError("Data must have a 'sequence' column or both 'sequence1' and 'sequence2'.")


# ---------------------------------------------------------------------------
# Per-example predictions
# ---------------------------------------------------------------------------

def compute_predictions(task, num_labels, logits, labels, threshold=0.5):
    """
    logits: np.ndarray [N, C] (C=1 for regression / single-logit binary).
    labels: np.ndarray [N] (float for regression, int otherwise).
    threshold: decision threshold for binary pred_label (default 0.5).
    Returns a dict of new columns + a hidden 'rep_score' (higher = more representative).
    """
    flavor = task_flavor(task, num_labels)
    out = {}

    if flavor == "regression":
        pred = logits.reshape(len(logits), -1)[:, 0]
        out["pred_value"] = pred
        out["true_value"] = labels.astype(float)
        out["residual"] = pred - labels.astype(float)
        out["abs_residual"] = np.abs(out["residual"])

    elif flavor == "binary":
        if logits.shape[1] == 1:
            prob_pos = _sigmoid(logits[:, 0])
        else:
            prob_pos = _softmax(logits)[:, 1]
        pred_label = (prob_pos >= threshold).astype(int)
        true = labels.astype(int)
        cat = np.where(
            (true == 1) & (pred_label == 1), "TP",
            np.where((true == 0) & (pred_label == 1), "FP",
            np.where((true == 0) & (pred_label == 0), "TN", "FN")))
        out["prob_positive"] = prob_pos
        out["pred_label"] = pred_label
        out["confusion_category"] = cat

    else:  # multiclass
        probs = _softmax(logits)
        pred_class = probs.argmax(axis=1)
        for c in range(probs.shape[1]):
            out[f"prob_{c}"] = probs[:, c]
        out["pred_class"] = pred_class
        out["max_prob"] = probs.max(axis=1)
        out["correct"] = (pred_class == labels.astype(int)).astype(int)

    return out


def pick_threshold(prob, labels, mode="f1"):
    """
    Choose a binary decision threshold on (prob, labels) by maximizing F1 or Youden's J.
    Used to derive the operating point from the DEV set so TP/FP/TN/FN on the (imbalanced,
    natural-prevalence) test set aren't all swept to negative by a fixed 0.5 cut.
    Returns (threshold, dev_score).
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        print("  pick_threshold: dev set has a single class; falling back to 0.5.")
        return 0.5, float("nan")

    if mode == "f1":
        prec, rec, thr = skm.precision_recall_curve(y, prob)
        if len(thr) == 0:
            return 0.5, float("nan")
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
        f1t = f1[:-1]  # precision_recall_curve: thr has len = len(prec) - 1
        i = int(np.argmax(f1t))
        return float(thr[i]), float(f1t[i])

    if mode == "youden":
        fpr, tpr, thr = skm.roc_curve(y, prob)
        j = tpr - fpr
        i = int(np.argmax(j))
        t = float(thr[i])
        if not np.isfinite(t):  # roc_curve may emit +inf as the first threshold
            t = 1.0
        return t, float(j[i])

    raise ValueError(f"unknown threshold mode {mode!r} (use 'f1' or 'youden').")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(task, num_labels, labels, pred):
    flavor = task_flavor(task, num_labels)
    m = {"task": task, "flavor": flavor, "n_examples": int(len(labels))}

    if flavor == "regression":
        y, yhat = labels.astype(float), pred["pred_value"]
        m["r2"] = float(skm.r2_score(y, yhat))
        m["mse"] = float(skm.mean_squared_error(y, yhat))
        m["rmse"] = float(np.sqrt(m["mse"]))
        m["mae"] = float(skm.mean_absolute_error(y, yhat))
        if len(y) > 1 and np.std(y) > 0 and np.std(yhat) > 0:
            r, p = stats.pearsonr(y, yhat)
            rho, _ = stats.spearmanr(y, yhat)
            m["pearson_r"], m["pearson_p"], m["spearman_r"] = float(r), float(p), float(rho)
        else:
            m["pearson_r"] = m["pearson_p"] = m["spearman_r"] = None

    elif flavor == "binary":
        y, prob, yhat = labels.astype(int), pred["prob_positive"], pred["pred_label"]
        m["accuracy"] = float(skm.accuracy_score(y, yhat))
        m["f1"] = float(skm.f1_score(y, yhat, zero_division=0))
        m["precision"] = float(skm.precision_score(y, yhat, zero_division=0))
        m["recall"] = float(skm.recall_score(y, yhat, zero_division=0))
        m["n_pos"], m["n_neg"] = int((y == 1).sum()), int((y == 0).sum())
        if m["n_pos"] > 0 and m["n_neg"] > 0:
            m["auroc"] = float(skm.roc_auc_score(y, prob))
            m["auprc"] = float(skm.average_precision_score(y, prob))
        else:
            m["auroc"] = m["auprc"] = None
        tn, fp, fn, tp = skm.confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        m["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    else:  # multiclass
        y, yhat = labels.astype(int), pred["pred_class"]
        m["accuracy"] = float(skm.accuracy_score(y, yhat))
        m["macro_f1"] = float(skm.f1_score(y, yhat, average="macro", zero_division=0))
        m["micro_f1"] = float(skm.f1_score(y, yhat, average="micro", zero_division=0))
        per_class = skm.f1_score(y, yhat, average=None,
                                 labels=list(range(num_labels)), zero_division=0)
        m["per_class_f1"] = {int(c): float(v) for c, v in enumerate(per_class)}
        m["confusion"] = skm.confusion_matrix(
            y, yhat, labels=list(range(num_labels))).tolist()

    return m


# ---------------------------------------------------------------------------
# Category assignment + representative ranking
# ---------------------------------------------------------------------------

def assign_categories(task, num_labels, df, selection_mode,
                      multiclass_category_mode="per_class", n_quantiles=4):
    """
    Adds 'category' and a hidden 'rep_score' (higher = more representative within category),
    then computes 'selection_rank_within_category' (0 = most representative). Returns the
    ordered list of categories that representative selection should draw from.
    """
    flavor = task_flavor(task, num_labels)
    df = df.copy()

    if flavor == "binary":
        df["category"] = df["confusion_category"]
        # representative = confident in the predicted direction
        pos = df["prob_positive"].to_numpy()
        rep = np.where(df["category"].isin(["TP", "FP"]), pos, 1.0 - pos)
        df["rep_score"] = rep
        sel_cats = [c for c in ["TP", "FP", "TN", "FN"] if (df["category"] == c).any()]

    elif flavor == "multiclass":
        if multiclass_category_mode == "correct_incorrect":
            df["category"] = np.where(df["correct"] == 1, "correct", "incorrect")
        elif multiclass_category_mode == "confusion_cells":
            df["category"] = (df["pred_class"].astype(str)
                              .radd(df["true_class"].astype(str) + "->"))
        else:  # per_class
            df["category"] = np.where(
                df["correct"] == 1,
                "correct_" + df["true_class"].astype(str),
                "error_" + df["true_class"].astype(str))
        df["rep_score"] = df["max_prob"]  # most confident = most representative
        sel_cats = sorted(df["category"].unique().tolist())

    else:  # regression
        true = df["true_value"].to_numpy()
        if selection_mode in ("auto", "value_extremes"):
            # Split into top/bottom halves by true value; within each, rank by extremeness
            # so selection's head(n_per_category) yields the n highest and n lowest values
            # (entexBERT-R "top/bottom by true label").
            df = _regression_extremes(df, true, key="value")
            sel_cats = ["top", "bottom"]
        elif selection_mode == "residual_extremes":
            df = _regression_extremes(df, df["abs_residual"].to_numpy(), key="residual")
            sel_cats = ["worst", "best"]
        elif selection_mode == "quantile_bins":
            bins = pd.qcut(true, q=n_quantiles, labels=[f"bin_{i}" for i in range(n_quantiles)],
                           duplicates="drop")
            df["category"] = bins.astype(str)
            df["rep_score"] = -true  # order within bin by ascending true value
            sel_cats = sorted(df["category"].unique().tolist())
        else:
            raise ValueError(f"Unsupported regression selection_mode: {selection_mode}")

    # within-category rank: 0 = most representative (highest rep_score)
    df["selection_rank_within_category"] = (
        df.groupby("category")["rep_score"].rank(ascending=False, method="first").astype(int) - 1
    )
    df = df.drop(columns=[c for c in ("_rank_desc", "_rank_asc") if c in df.columns])
    return df, sel_cats


def _regression_extremes(df, key_values, key="value"):
    """Assign two extreme categories for regression based on a key array."""
    df = df.copy()
    if key == "value":
        # top = highest values, bottom = lowest values
        df["category"] = "mid"
        df["rep_score"] = 0.0
        df.loc[:, "_kv"] = key_values
        top_mask = df["_kv"] >= df["_kv"].median()
        df.loc[top_mask, "category"] = "top"
        df.loc[~top_mask, "category"] = "bottom"
        # rep_score: extremeness within each side
        df.loc[df["category"] == "top", "rep_score"] = df.loc[df["category"] == "top", "_kv"]
        df.loc[df["category"] == "bottom", "rep_score"] = -df.loc[df["category"] == "bottom", "_kv"]
        df = df.drop(columns=["_kv"])
    else:  # residual: worst = largest |resid|, best = smallest
        df["category"] = "mid"
        df["rep_score"] = 0.0
        df.loc[:, "_kv"] = key_values
        med = df["_kv"].median()
        worst_mask = df["_kv"] >= med
        df.loc[worst_mask, "category"] = "worst"
        df.loc[~worst_mask, "category"] = "best"
        df.loc[df["category"] == "worst", "rep_score"] = df.loc[df["category"] == "worst", "_kv"]
        df.loc[df["category"] == "best", "rep_score"] = -df.loc[df["category"] == "best", "_kv"]
        df = df.drop(columns=["_kv"])
    return df


# ---------------------------------------------------------------------------
# Representative selection (folds in the old select_representative_examples)
# ---------------------------------------------------------------------------

def select_representatives(df, selection_categories, n_per_category,
                           input_mode, deduplicate_inputs=True):
    dedup_cols = (["sequence1", "sequence2"] if input_mode == "pair" else ["sequence"])
    dedup_cols = [c for c in dedup_cols if c in df.columns]

    chosen = []
    for cat in selection_categories:
        group = df[df["category"] == cat].copy()
        if group.empty:
            print(f"Warning: no examples for category {cat}")
            continue
        group = group.sort_values("selection_rank_within_category")
        if deduplicate_inputs and dedup_cols:
            before = len(group)
            group = group.drop_duplicates(subset=dedup_cols)
            if before != len(group):
                print(f"{cat}: deduplicated {before} -> {len(group)}")
        group = group.head(n_per_category)
        print(f"{cat}: using {len(group)} examples")
        chosen.append(group)

    if not chosen:
        raise ValueError("No representative examples selected.")
    return pd.concat(chosen, ignore_index=True)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(embeddings, n_components, max_examples=None, seed=0):
    X = embeddings
    idx = np.arange(len(X))
    if max_examples is not None and len(X) > max_examples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=max_examples, replace=False))
        X = X[idx]
    k = min(n_components, X.shape[1], max(2, X.shape[0]))
    pca = PCA(n_components=k, random_state=seed)
    coords = pca.fit_transform(X)
    return idx, coords, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Inference (lazy torch / model_io)
# ---------------------------------------------------------------------------

def run_inference(checkpoint_dir, texts, batch_size, device, overrides):
    import torch  # noqa
    from entexbert2 import model_io

    model, tokenizer, run_config = model_io.load_model_and_tokenizer(
        checkpoint_dir, device=device, overrides=overrides)

    all_logits, all_emb = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding="longest",
                        max_length=run_config.get("model_max_length", 512), truncation=True)
        logits, pooled = model_io.logits_and_embeddings(
            model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_emb.append(pooled.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_emb, axis=0), run_config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(data_csv):
    """Read the prediction CSV and merge the row-aligned .meta.csv sidecar if present."""
    df = pd.read_csv(data_csv)
    meta_path = data_csv[:-4] + ".meta.csv" if data_csv.endswith(".csv") else None
    if meta_path and os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if len(meta) == len(df):
            new_cols = [c for c in meta.columns if c not in df.columns]
            df = pd.concat([df.reset_index(drop=True), meta[new_cols].reset_index(drop=True)], axis=1)
            print(f"Merged {len(new_cols)} metadata columns from {meta_path}")
        else:
            print(f"Warning: {meta_path} has {len(meta)} rows vs {len(df)}; not merging.")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="entexBERT-2 analysis / PCA / representative selection.")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--data_csv", required=True)
    p.add_argument("--output_dir", required=True)
    # task/head come from run_config.json; these override if given
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--num_labels", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    # binary decision threshold for TP/FP/TN/FN (matters at natural prevalence)
    p.add_argument("--threshold", default="0.5",
                   help="Binary decision threshold: a float (e.g. 0.5), or 'f1'/'youden' to "
                        "derive it from the dev set (max-F1 or Youden's J).")
    p.add_argument("--dev_csv", default=None,
                   help="Dev CSV for --threshold f1/youden. If omitted and data_csv ends in "
                        "'test.csv', the sibling 'dev.csv' is used.")
    # selection
    p.add_argument("--selection_mode", default="auto",
                   choices=["auto", "confusion", "value_extremes", "residual_extremes", "quantile_bins"])
    p.add_argument("--multiclass_category_mode", default="per_class",
                   choices=["per_class", "correct_incorrect", "confusion_cells"])
    p.add_argument("--n_per_category", type=int, default=100)
    p.add_argument("--n_quantiles", type=int, default=4)
    p.add_argument("--deduplicate_inputs", dest="deduplicate_inputs", action="store_true", default=True,
                   help="Drop duplicate input sequences before selecting top-N per category (default: on).")
    p.add_argument("--no_deduplicate_inputs", dest="deduplicate_inputs", action="store_false",
                   help="Disable deduplication of input sequences.")
    # pca
    p.add_argument("--pca_components", type=int, default=10)
    p.add_argument("--max_pca_examples", type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.data_csv)
    input_mode = detect_input_mode(df.columns)
    if input_mode == "pair":
        texts = [[a, b] for a, b in zip(df["sequence1"], df["sequence2"])]
    else:
        texts = df["sequence"].astype(str).tolist()

    overrides = {"model_name_or_path": args.model_name_or_path,
                 "task": args.task, "main_num_labels": args.num_labels}
    logits, embeddings, run_config = run_inference(
        args.checkpoint_dir, texts, args.batch_size, args.device, overrides)

    task = run_config["task"]
    num_labels = run_config["main_num_labels"]
    flavor = task_flavor(task, num_labels)
    print(f"task={task} num_labels={num_labels} flavor={flavor} input_mode={input_mode}")

    labels = (df["label"].astype(float).to_numpy() if flavor == "regression"
              else df["label"].astype(int).to_numpy())
    if flavor == "multiclass":
        df["true_class"] = labels.astype(int)

    # Resolve the binary decision threshold (only meaningful for binary).
    threshold, thr_source = 0.5, "fixed_0.5"
    ts = str(args.threshold).strip().lower()
    if flavor == "binary":
        if ts in ("f1", "youden"):
            dev_csv = args.dev_csv
            if dev_csv is None and args.data_csv.endswith("test.csv"):
                dev_csv = args.data_csv[:-len("test.csv")] + "dev.csv"
            if not dev_csv or not os.path.exists(dev_csv):
                raise ValueError(f"--threshold {ts} needs a dev set; pass --dev_csv "
                                 f"(inferred {dev_csv!r} not found).")
            dev_df = load_data(dev_csv)
            dtexts = ([[a, b] for a, b in zip(dev_df["sequence1"], dev_df["sequence2"])]
                      if input_mode == "pair" else dev_df["sequence"].astype(str).tolist())
            dev_logits, _, _ = run_inference(args.checkpoint_dir, dtexts,
                                             args.batch_size, args.device, overrides)
            dev_prob = (_sigmoid(dev_logits[:, 0]) if dev_logits.shape[1] == 1
                        else _softmax(dev_logits)[:, 1])
            threshold, score = pick_threshold(dev_prob, dev_df["label"].astype(int).to_numpy(), ts)
            thr_source = f"dev_{ts}"
            print(f"Dev-derived threshold ({ts}): {threshold:.4f} (dev score={score:.4f}) from {dev_csv}")
        else:
            threshold = float(ts)
            thr_source = f"fixed_{threshold:g}"
            print(f"Using fixed decision threshold: {threshold:.4f}")
    elif ts in ("f1", "youden"):
        print(f"NOTE: --threshold {ts} ignored for non-binary task ({flavor}).")

    pred = compute_predictions(task, num_labels, logits, labels, threshold=threshold)
    for k, v in pred.items():
        df[k] = v

    metrics = compute_metrics(task, num_labels, labels, pred)
    if flavor == "binary":
        metrics["decision_threshold"] = float(threshold)
        metrics["threshold_source"] = thr_source
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", json.dumps(metrics, indent=2))

    # PCA over pooled embeddings
    target = (df["true_value"] if flavor == "regression" else labels)
    idx, coords, evr = run_pca(embeddings, args.pca_components, args.max_pca_examples)
    pca_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
    pca_df["target"] = np.asarray(target)[idx]
    # Attach the per-example id and prediction category so pca.csv is self-sufficient
    # for the PCA plots (no join back to predictions.csv needed for future runs).
    for col in ("example_id", "confusion_category", "pred_label", "pred_class", "true_class"):
        if col in df.columns:
            pca_df[col] = df[col].to_numpy()[idx]
    pca_df.to_csv(os.path.join(args.output_dir, "pca.csv"), index=False)
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": np.round(np.asarray(evr, dtype=float), 6),
    }).to_csv(os.path.join(args.output_dir, "pca_explained_variance.csv"), index=False)
    print("PCA explained variance ratio:", np.round(evr, 4).tolist())

    # categories + selection
    sel_mode = args.selection_mode
    if sel_mode == "auto":
        sel_mode = {"binary": "confusion", "multiclass": "class",
                    "regression": "value_extremes"}[flavor]
    df, sel_cats = assign_categories(
        task, num_labels, df, sel_mode,
        multiclass_category_mode=args.multiclass_category_mode, n_quantiles=args.n_quantiles)

    df.drop(columns=["rep_score"]).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    rep = select_representatives(df.drop(columns=["rep_score"]), sel_cats,
                                 args.n_per_category, input_mode, args.deduplicate_inputs)
    rep.to_csv(os.path.join(args.output_dir, "representative_examples_all.csv"), index=False)
    for cat, grp in rep.groupby("category"):
        grp.to_csv(os.path.join(args.output_dir, f"{cat}.csv"), index=False)

    print(f"\nWrote predictions.csv, metrics.json, pca.csv, representative_examples_all.csv "
          f"({len(rep)} examples across {len(sel_cats)} categories) to {args.output_dir}")


if __name__ == "__main__":
    main()
