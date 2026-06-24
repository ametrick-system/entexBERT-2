#!/usr/bin/env python3
"""
Attribution profiles for entexBERT-2: in-silico mutagenesis (ISM, the faithful core)
and Integrated Gradients (IG, the cheap gradient companion).

ISM is base-resolution: it mutates the actual nucleotide and re-tokenizes, so it sidesteps
the BPE token-smearing that limits attention/saliency/IG. It is the primary, intervention-
based evidence. IG is a one-backward-pass approximation for triangulation; its baseline is
unavoidably awkward for a BPE transformer (see --ig_baseline), so treat it as a second
opinion to be confirmed by targeted ISM, not as primary evidence.

Profiles are realigned per example to the SNV via anchor_offset_seq1 (jitter-correct), the
same convention as the attention plotter. Uses the model's OWN trained backbone via model_io
(no attention-extraction model needed).

Currently supports input_mode=ref_single (the CTCF/H3K27ac AS runs). hap_pair raises.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from entexbert2 import model_io

BASES = ["A", "C", "G", "T"]
# candidate column names for the alternate allele in the examples/meta CSV
ALT_COL_CANDIDATES = ["alt", "alt_allele", "ALT", "snv_alt", "variant_alt", "alt_base"]


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--examples_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name_or_path", default=None,
                   help="Optional backbone override. Default: the trained backbone from run_config "
                        "(ISM/IG do NOT need the attention-extraction model).")

    p.add_argument("--input_mode", default="ref_single", choices=["ref_single", "hap_pair"])
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--left_bp", type=int, default=256,
                   help="Fallback SNV offset when anchor_offset_seq1 is absent (centered runs).")

    p.add_argument("--categories", default="TP,FP,TN,FN")
    p.add_argument("--n_per_category", type=int, default=100)
    p.add_argument("--deduplicate_inputs", dest="deduplicate_inputs", action="store_true", default=True)
    p.add_argument("--no_deduplicate_inputs", dest="deduplicate_inputs", action="store_false")

    p.add_argument("--method", default="both", choices=["ism", "ig", "both"])
    p.add_argument("--ism_mode", default="both", choices=["snv", "window", "kmer", "both", "all"],
                   help="snv/window = single-base; kmer = sliding k-bp ablation (redundancy probe); "
                        "both = snv+window; all = snv+window+kmer.")
    p.add_argument("--ism_window_bp", type=int, default=100,
                   help="Half-width of the windowed-ISM / k-mer region around the SNV.")
    p.add_argument("--ism_reduce", default="max", choices=["max", "mean"],
                   help="Reduce the 3 alternate-base effects per position to one importance.")
    p.add_argument("--n_controls", type=int, default=25,
                   help="Random non-SNV positions per example for the SNV-vs-background comparison.")

    # k-mer / motif-width ablation (length-preserving substitution; NO indels)
    p.add_argument("--kmer_sizes", default="3,6,10,15",
                   help="Comma-separated k-mer widths to sweep for --ism_mode kmer/all.")
    p.add_argument("--kmer_replacement", default="dinuc", choices=["dinuc", "mono", "random"],
                   help="How to ablate a k-mer block: dinuc/mono shuffle (on-manifold, composition-"
                        "preserving) or random bases (off-manifold).")
    p.add_argument("--kmer_n_shuffles", type=int, default=3,
                   help="Ablations averaged per position (reduces single-shuffle noise).")
    p.add_argument("--kmer_stride", type=int, default=1,
                   help="Step between ablated centers across the window (>1 to cut cost).")

    p.add_argument("--score_mode", default="margin", choices=["margin", "pos_logit", "prob_pos"],
                   help="Scalar the attribution is computed against. margin = logit1 - logit0.")

    p.add_argument("--ig_steps", type=int, default=32)
    p.add_argument("--ig_baseline", default="mask", choices=["mask", "pad", "zero"],
                   help="Per-token embedding baseline. 'zero' is off-manifold (discouraged); "
                        "'mask'/'pad' use that token's embedding. None is ideal for BPE -- this is "
                        "the IG limitation, which is exactly why ISM is the primary method.")

    p.add_argument("--plot_window_bp", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------------------------------------------------------
# Example selection (mirrors the attention plotter)
# -------------------------------------------------------------------
def select_examples(df, args):
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    out = []
    for c in cats:
        g = df[df["confusion_category"] == c].copy()
        if g.empty:
            print(f"Warning: no examples for {c}")
            continue
        if "selection_rank_within_category" in g.columns:
            g = g.sort_values("selection_rank_within_category")
        if args.deduplicate_inputs and "sequence" in g.columns:
            g = g.drop_duplicates(subset=["sequence"]).copy()
        g = g.head(args.n_per_category).copy()
        print(f"{c}: using {len(g)} examples")
        out.append(g)
    if not out:
        raise ValueError("No examples selected.")
    return pd.concat(out, ignore_index=True)


def example_snv_offset(row, fallback_left_bp):
    if "anchor_offset_seq1" in row and pd.notna(row["anchor_offset_seq1"]):
        return int(row["anchor_offset_seq1"])
    return int(fallback_left_bp)


def example_alt_base(row):
    for col in ALT_COL_CANDIDATES:
        if col in row and isinstance(row[col], str) and row[col].strip().upper() in BASES:
            return row[col].strip().upper()
    return None


# -------------------------------------------------------------------
# Torch scorers (the only parts that need the model)
# -------------------------------------------------------------------
def make_score_fn(model, tokenizer, max_len, batch_size, device, score_mode):
    """Return score_fn(list_of_sequences) -> np.ndarray of scalar scores (no grad)."""
    def _score(seqs):
        scores = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            logits, _ = model_io.logits_and_embeddings(model, input_ids, attn)
            scores.append(target_score(logits, score_mode).detach().cpu().numpy())
        return np.concatenate(scores) if scores else np.array([])
    return _score


def target_score(logits, score_mode):
    if logits.shape[-1] < 2:
        # single-logit head: treat as the score directly
        return logits[:, 0]
    if score_mode == "margin":
        return logits[:, 1] - logits[:, 0]
    if score_mode == "pos_logit":
        return logits[:, 1]
    if score_mode == "prob_pos":
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(score_mode)


# -------------------------------------------------------------------
# ISM orchestration (pure: takes an injected score_fn -> unit-testable)
# -------------------------------------------------------------------
def _mutants_at(seq, pos):
    ref = seq[pos].upper()
    return [(b, seq[:pos] + b + seq[pos + 1:]) for b in BASES if b != ref]


def ism_window_importance(ref_seq, snv_pos, window_bp, score_fn, reduce="max", ref_score=None):
    """Per-position importance over [snv-window, snv+window]. Base-resolution. Pure orchestration."""
    n = len(ref_seq)
    lo = max(0, snv_pos - window_bp)
    hi = min(n, snv_pos + window_bp + 1)
    positions = list(range(lo, hi))

    # Build all mutant sequences for this example, remember which position each belongs to.
    mut_seqs, mut_pos = [], []
    for pos in positions:
        for _, ms in _mutants_at(ref_seq, pos):
            mut_seqs.append(ms)
            mut_pos.append(pos)

    if ref_score is None:
        ref_score = float(score_fn([ref_seq])[0])
    mut_scores = score_fn(mut_seqs) if mut_seqs else np.array([])

    by_pos = {pos: [] for pos in positions}
    for pos, sc in zip(mut_pos, mut_scores):
        by_pos[pos].append(abs(float(sc) - ref_score))

    importance = {}
    for pos in positions:
        deltas = by_pos[pos]
        if not deltas:
            importance[pos] = 0.0
        else:
            importance[pos] = max(deltas) if reduce == "max" else float(np.mean(deltas))
    return importance, ref_score


def ism_snv_importance(ref_seq, snv_pos, score_fn, alt_base=None, n_controls=25,
                       window_bp=100, rng=None, ref_score=None):
    """SNV importance (max over 3 alts), true-alt signed delta if known, and a background
    distribution from random non-SNV positions. Pure orchestration."""
    if rng is None:
        rng = np.random.default_rng(0)
    if ref_score is None:
        ref_score = float(score_fn([ref_seq])[0])

    # SNV: all 3 alternate bases
    snv_muts = _mutants_at(ref_seq, snv_pos)
    snv_scores = score_fn([m for _, m in snv_muts])
    per_alt = {b: float(s) - ref_score for (b, _), s in zip(snv_muts, snv_scores)}
    snv_max_abs = max(abs(v) for v in per_alt.values()) if per_alt else 0.0
    true_alt_delta = per_alt.get(alt_base.upper()) if isinstance(alt_base, str) else None

    # Background: random non-SNV positions within the same window
    n = len(ref_seq)
    lo = max(0, snv_pos - window_bp)
    hi = min(n, snv_pos + window_bp + 1)
    candidates = [p for p in range(lo, hi) if p != snv_pos]
    rng.shuffle(candidates)
    ctrl_positions = candidates[:min(n_controls, len(candidates))]

    ctrl_seqs, ctrl_owner = [], []
    for pos in ctrl_positions:
        for _, ms in _mutants_at(ref_seq, pos):
            ctrl_seqs.append(ms)
            ctrl_owner.append(pos)
    ctrl_scores = score_fn(ctrl_seqs) if ctrl_seqs else np.array([])
    ctrl_by_pos = {pos: [] for pos in ctrl_positions}
    for pos, sc in zip(ctrl_owner, ctrl_scores):
        ctrl_by_pos[pos].append(abs(float(sc) - ref_score))
    ctrl_max_abs = [max(v) for v in ctrl_by_pos.values() if v]
    ctrl_mean = float(np.mean(ctrl_max_abs)) if ctrl_max_abs else float("nan")

    return {
        "snv_max_abs_delta": snv_max_abs,
        "snv_true_alt_delta": true_alt_delta,
        "snv_per_alt": per_alt,
        "control_mean_abs_delta": ctrl_mean,
        "control_n": len(ctrl_max_abs),
        "ref_score": ref_score,
    }


# -------------------------------------------------------------------
# k-mer / motif-width ablation (length-preserving; NO indels)
# -------------------------------------------------------------------
def mono_shuffle(block, rng):
    """Permute the bases in the block (preserves single-base composition)."""
    chars = list(block)
    rng.shuffle(chars)
    return "".join(chars)


def random_replace(block, rng):
    """Replace with i.i.d. uniform ACGT (off-manifold; changes composition)."""
    return "".join(rng.choice(BASES, size=len(block)))


def dinuc_shuffle(block, rng):
    """
    Dinucleotide-preserving shuffle (Altschul-Erikson / Eulerian-walk), keeping the
    first and last base fixed and preserving all adjacent-pair (dinucleotide) counts.
    Falls back to mono_shuffle for blocks too short to shuffle meaningfully.
    """
    s = block.upper()
    if len(s) <= 3:
        return s  # too constrained to dinuc-shuffle with fixed ends; identity preserves counts
    first, last = s[0], s[-1]
    verts = set(s)
    # adjacency multiset
    edges = {v: [] for v in verts}
    for a, b in zip(s[:-1], s[1:]):
        edges[a].append(b)

    # Find a "last edge" per vertex (!= last) so that following last-edges always
    # reaches `last` without a cycle (an arborescence into `last`). Retry on failure.
    for _attempt in range(50):
        for v in verts:
            rng.shuffle(edges[v])
        last_edge = {v: edges[v][-1] for v in verts if v != last and edges[v]}
        ok = True
        for v in verts:
            if v == last:
                continue
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in last_edge:
                    ok = False
                    break
                seen.add(cur)
                cur = last_edge[cur]
            if not ok:
                break
        if ok:
            break
    else:
        return s  # no valid arborescence found; identity preserves dinucleotide counts

    # Per-vertex traversal order: shuffled edges with the reserved last-edge at the end.
    order = {}
    for v in verts:
        lst = list(edges[v])
        if v != last and v in last_edge:
            lst.remove(last_edge[v])
            lst.append(last_edge[v])
        order[v] = lst

    result = [first]
    idx = {v: 0 for v in verts}
    cur = first
    for _ in range(len(s) - 1):
        if idx[cur] >= len(order[cur]):
            return s  # defensive; identity preserves dinucleotide counts
        nxt = order[cur][idx[cur]]
        idx[cur] += 1
        result.append(nxt)
        cur = nxt
    return "".join(result)


REPLACERS = {"dinuc": dinuc_shuffle, "mono": mono_shuffle, "random": random_replace}


def kmer_ablation_window(ref_seq, snv_pos, window_bp, k, score_fn, replace_fn,
                         n_shuffles, stride, rng, ref_score=None):
    """Slide a length-k ablated block across the window; importance at each center =
    mean |Delta score| over n_shuffles ablations. Centers are anchored on the SNV so
    relative position 0 is always sampled. Pure orchestration (injected score_fn)."""
    n = len(ref_seq)
    if ref_score is None:
        ref_score = float(score_fn([ref_seq])[0])

    half = window_bp
    centers = sorted(set(
        c for c in range(snv_pos - half, snv_pos + half + 1, stride) if 0 <= c < n
    ) | {snv_pos})

    mut_seqs, owners = [], []
    for c in centers:
        start = c - k // 2
        start = max(0, min(start, n - k)) if n >= k else 0
        end = min(n, start + k)
        for _ in range(n_shuffles):
            new_block = replace_fn(ref_seq[start:end], rng)
            mut_seqs.append(ref_seq[:start] + new_block + ref_seq[end:])
            owners.append(c)

    scores = score_fn(mut_seqs) if mut_seqs else np.array([])
    by_c = {c: [] for c in centers}
    for c, sc in zip(owners, scores):
        by_c[c].append(abs(float(sc) - ref_score))
    return {c: (float(np.mean(by_c[c])) if by_c[c] else 0.0) for c in centers}


# -------------------------------------------------------------------
# Integrated Gradients (companion). Token-level; baseline is per-token embedding.
# -------------------------------------------------------------------
def _find_word_embedding(backbone):
    """Heuristic: the word-embedding nn.Embedding has the largest num_embeddings (the vocab)."""
    best = None
    for _, mod in backbone.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            if best is None or mod.num_embeddings > best.num_embeddings:
                best = mod
    return best


def integrated_gradients_tokens(model, tokenizer, seq, steps, baseline, score_mode,
                                device, max_len):
    """Per-token IG attribution for one sequence. Returns (token_attr, offsets, sequence_ids, tokens)."""
    emb_layer = _find_word_embedding(model.backbone)
    if emb_layer is None:
        raise RuntimeError("Could not locate the word-embedding module for IG.")

    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len,
                    return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()
    sequence_ids = enc.sequence_ids(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        input_emb = emb_layer(input_ids).detach()  # [1, T, H]
        if baseline == "zero":
            base_emb = torch.zeros_like(input_emb)
        else:
            tok = tokenizer.mask_token_id if baseline == "mask" else tokenizer.pad_token_id
            if tok is None:
                tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            base_ids = torch.full_like(input_ids, int(tok))
            base_emb = emb_layer(base_ids).detach()

    holder = {}

    def hook(_module, _inp, _out):
        return holder["emb"]

    handle = emb_layer.register_forward_hook(hook)
    total_grad = torch.zeros_like(input_emb)
    try:
        for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
            emb_a = (base_emb + alpha * (input_emb - base_emb)).detach().requires_grad_(True)
            holder["emb"] = emb_a
            backbone_out = model.backbone(input_ids=input_ids, attention_mask=attn, return_dict=True)
            pooled = model._pool_sequence_representation(backbone_out, attention_mask=attn)
            logits = model.main_head(pooled)
            score = target_score(logits, score_mode).sum()
            grad = torch.autograd.grad(score, emb_a)[0]
            total_grad = total_grad + grad.detach()
    finally:
        handle.remove()

    avg_grad = total_grad / steps
    ig = ((input_emb - base_emb) * avg_grad).sum(dim=-1)[0].detach().cpu().numpy()  # [T]
    return ig, offsets, sequence_ids, tokens


def token_attr_to_base_profile(token_attr, sequence_ids, offsets, seq_len):
    """Distribute per-token attribution across the bases it covers (mean), seq 0 only."""
    scores = np.zeros(seq_len, dtype=float)
    counts = np.zeros(seq_len, dtype=float)
    for i, a in enumerate(token_attr):
        sid = sequence_ids[i]
        if sid != 0:
            continue
        s, e = offsets[i]
        s, e = max(0, int(s)), min(seq_len, int(e))
        if e > s:
            scores[s:e] += float(a)
            counts[s:e] += 1.0
    valid = counts > 0
    scores[valid] /= counts[valid]
    return scores


# -------------------------------------------------------------------
# Aggregation + plotting
# -------------------------------------------------------------------
def mean_profile(long_df, value_col, plot_window_bp):
    sub = long_df
    if plot_window_bp is not None and plot_window_bp >= 0:
        sub = sub[sub["position_relative_to_snv"].between(-plot_window_bp, plot_window_bp)]
    return (sub.groupby(["confusion_category", "position_relative_to_snv"])[value_col]
            .mean().reset_index())


def plot_profile(mean_df, value_col, title, out_path, dpi):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat, g in mean_df.groupby("confusion_category"):
        g = g.sort_values("position_relative_to_snv")
        ax.plot(g["position_relative_to_snv"], g[value_col], linewidth=1.6, label=cat)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Position relative to SNV")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_snv_vs_background(snv_df, out_path, dpi):
    cats = list(snv_df["confusion_category"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cats))
    snv_means = [snv_df.loc[snv_df.confusion_category == c, "snv_max_abs_delta"].mean() for c in cats]
    ctrl_means = [snv_df.loc[snv_df.confusion_category == c, "control_mean_abs_delta"].mean() for c in cats]
    ax.bar(x - 0.2, snv_means, width=0.4, label="SNV |Δ|")
    ax.bar(x + 0.2, ctrl_means, width=0.4, label="background |Δ| (random positions)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("mean |Δ score| (max over alt bases)")
    ax.set_title("ISM: SNV effect vs background, per category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_kmer_snv_vs_background(bar_df, out_path, dpi):
    ks = sorted(bar_df["kmer_size"].unique())
    cats = sorted(bar_df["confusion_category"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 4), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    x = np.arange(len(cats))
    for ax, k in zip(axes, ks):
        sub = bar_df[bar_df.kmer_size == k].set_index("confusion_category")
        snv = [sub.loc[c, "snv_kmer_mean"] if c in sub.index else np.nan for c in cats]
        bg = [sub.loc[c, "background_kmer_mean"] if c in sub.index else np.nan for c in cats]
        ax.bar(x - 0.2, snv, width=0.4, label="SNV-centered k-mer")
        ax.bar(x + 0.2, bg, width=0.4, label="size-matched background")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_title(f"k={k}")
    axes[0].set_ylabel("mean |Δ score| (k-mer ablation)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("k-mer ablation: SNV-centered vs size-matched background")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    if args.input_mode == "hap_pair":
        raise NotImplementedError(
            "hap_pair attribution isn't wired up yet (two sequences + native allele contrast). "
            "Use ref_single, or ask to add hap_pair."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.examples_csv)
    sel = select_examples(df, args)

    overrides = {"model_name_or_path": args.model_name_or_path} if args.model_name_or_path else {}
    model, tokenizer, run_config = model_io.load_model_and_tokenizer(
        args.checkpoint_dir, device=args.device, overrides=overrides)
    model.eval()

    score_fn = make_score_fn(model, tokenizer, args.model_max_length, args.batch_size,
                             args.device, args.score_mode)

    do_ism = args.method in ("ism", "both")
    do_ig = args.method in ("ig", "both")
    want_window = args.ism_mode in ("window", "both", "all")
    want_snv = args.ism_mode in ("snv", "both", "all")
    want_kmer = args.ism_mode in ("kmer", "all")
    kmer_sizes = [int(x) for x in str(args.kmer_sizes).split(",") if x.strip()]
    replace_fn = REPLACERS[args.kmer_replacement]

    ism_window_rows, snv_rows, ig_rows, kmer_rows = [], [], [], []

    for local_idx, row in sel.iterrows():
        seq = str(row["sequence"])
        snv_off = example_snv_offset(row, args.left_bp)
        if not (0 <= snv_off < len(seq)):
            print(f"Skipping example {local_idx}: snv_offset {snv_off} outside sequence len {len(seq)}")
            continue
        ex_id = row.get("example_id", row.get("example_index", local_idx))
        cat = row["confusion_category"]
        ref_score = float(score_fn([seq])[0])

        if do_ism and want_window:
            imp, _ = ism_window_importance(seq, snv_off, args.ism_window_bp, score_fn,
                                           reduce=args.ism_reduce, ref_score=ref_score)
            for pos, val in imp.items():
                ism_window_rows.append({
                    "example_id": ex_id, "confusion_category": cat,
                    "position_relative_to_snv": pos - snv_off, "ism_importance": val})

        if do_ism and want_snv:
            res = ism_snv_importance(seq, snv_off, score_fn, alt_base=example_alt_base(row),
                                     n_controls=args.n_controls, window_bp=args.ism_window_bp,
                                     rng=rng, ref_score=ref_score)
            snv_rows.append({
                "example_id": ex_id, "confusion_category": cat,
                "snv_max_abs_delta": res["snv_max_abs_delta"],
                "snv_true_alt_delta": res["snv_true_alt_delta"],
                "control_mean_abs_delta": res["control_mean_abs_delta"],
                "control_n": res["control_n"], "ref_score": res["ref_score"]})

        if do_ism and want_kmer:
            for k in kmer_sizes:
                prof = kmer_ablation_window(seq, snv_off, args.ism_window_bp, k, score_fn,
                                            replace_fn, args.kmer_n_shuffles, args.kmer_stride,
                                            rng, ref_score=ref_score)
                for c, val in prof.items():
                    kmer_rows.append({
                        "example_id": ex_id, "confusion_category": cat, "kmer_size": k,
                        "position_relative_to_snv": c - snv_off, "kmer_importance": val})

        if do_ig:
            ig_tok, offsets, sequence_ids, _ = integrated_gradients_tokens(
                model, tokenizer, seq, args.ig_steps, args.ig_baseline, args.score_mode,
                args.device, args.model_max_length)
            base_prof = token_attr_to_base_profile(ig_tok, sequence_ids, offsets, len(seq))
            for pos, val in enumerate(base_prof):
                ig_rows.append({
                    "example_id": ex_id, "confusion_category": cat,
                    "position_relative_to_snv": pos - snv_off, "ig_attr": float(val)})

        if (local_idx + 1) % 10 == 0 or local_idx == len(sel) - 1:
            print(f"  processed {local_idx + 1}/{len(sel)}")

    # ---- write + plot ----
    if ism_window_rows:
        w = pd.DataFrame(ism_window_rows)
        w.to_csv(os.path.join(args.output_dir, "ism_window_long.csv"), index=False)
        m = mean_profile(w, "ism_importance", args.plot_window_bp)
        m.to_csv(os.path.join(args.output_dir, "ism_window_mean.csv"), index=False)
        plot_profile(m, "ism_importance",
                     "ISM importance (base-resolution, max over alt bases)",
                     os.path.join(args.output_dir, "ism_window_profile.png"), args.dpi)

    if snv_rows:
        s = pd.DataFrame(snv_rows)
        s.to_csv(os.path.join(args.output_dir, "ism_snv_per_example.csv"), index=False)
        summ = s.groupby("confusion_category").agg(
            snv_mean=("snv_max_abs_delta", "mean"),
            snv_median=("snv_max_abs_delta", "median"),
            background_mean=("control_mean_abs_delta", "mean"),
            true_alt_mean_signed=("snv_true_alt_delta", "mean"),
            n=("snv_max_abs_delta", "size")).reset_index()
        summ.to_csv(os.path.join(args.output_dir, "ism_snv_summary.csv"), index=False)
        plot_snv_vs_background(s, os.path.join(args.output_dir, "ism_snv_vs_background.png"), args.dpi)
        print("\nISM SNV-vs-background summary:")
        print(summ.to_string(index=False))

    if kmer_rows:
        kdf = pd.DataFrame(kmer_rows)
        kdf.to_csv(os.path.join(args.output_dir, "ism_kmer_long.csv"), index=False)

        # one realigned profile per k
        for k in sorted(kdf["kmer_size"].unique()):
            sub = kdf[kdf["kmer_size"] == k]
            m = mean_profile(sub.rename(columns={"kmer_importance": "v"}), "v", args.plot_window_bp)
            m.to_csv(os.path.join(args.output_dir, f"ism_kmer_k{k}_mean.csv"), index=False)
            plot_profile(m, "v",
                         f"k-mer ablation importance (k={k}, {args.kmer_replacement}, "
                         f"{args.kmer_n_shuffles} shuffles)",
                         os.path.join(args.output_dir, f"ism_kmer_k{k}_profile.png"), args.dpi)

        # SNV-centered k-mer vs size-matched background (centers whose block can't overlap the SNV),
        # derived from the same profile: SNV = importance at relative 0; background = mean over |rel| >= k.
        bar_rows = []
        for (cat, k), g in kdf.groupby(["confusion_category", "kmer_size"]):
            per_ex_snv, per_ex_bg = [], []
            for _ex, gex in g.groupby("example_id"):
                at0 = gex.loc[gex["position_relative_to_snv"] == 0, "kmer_importance"]
                far = gex.loc[gex["position_relative_to_snv"].abs() >= k, "kmer_importance"]
                if len(at0):
                    per_ex_snv.append(float(at0.mean()))
                if len(far):
                    per_ex_bg.append(float(far.mean()))
            bar_rows.append({
                "confusion_category": cat, "kmer_size": k,
                "snv_kmer_mean": float(np.mean(per_ex_snv)) if per_ex_snv else float("nan"),
                "background_kmer_mean": float(np.mean(per_ex_bg)) if per_ex_bg else float("nan"),
                "n": len(per_ex_snv)})
        bar_df = pd.DataFrame(bar_rows).sort_values(["kmer_size", "confusion_category"])
        bar_df.to_csv(os.path.join(args.output_dir, "ism_kmer_snv_vs_background.csv"), index=False)
        plot_kmer_snv_vs_background(bar_df, os.path.join(args.output_dir, "ism_kmer_snv_vs_background.png"), args.dpi)
        print("\nk-mer SNV-vs-background (mean |Delta| at SNV vs non-overlapping background):")
        print(bar_df.to_string(index=False))

    if ig_rows:
        g = pd.DataFrame(ig_rows)
        g.to_csv(os.path.join(args.output_dir, "ig_long.csv"), index=False)
        # plot |attr| profile (sign depends on score direction; magnitude is the comparable quantity)
        g["ig_abs"] = g["ig_attr"].abs()
        m = mean_profile(g, "ig_abs", args.plot_window_bp)
        m.to_csv(os.path.join(args.output_dir, "ig_mean.csv"), index=False)
        plot_profile(m, "ig_abs",
                     "Integrated Gradients |attribution| (token-level, baseline=%s)" % args.ig_baseline,
                     os.path.join(args.output_dir, "ig_profile.png"), args.dpi)

    with open(os.path.join(args.output_dir, "attribution_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
