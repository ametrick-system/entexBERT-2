#!/usr/bin/env python3

"""
Shared model I/O for entexBERT-2 evaluation, scoring, and the frozen-trunk pre-check.

The single place that knows how to rebuild a trained entexBERT-2 model and run inference.
score_asb.py and probe_frozen_trunk.py import it, so the architecture is never re-specified.

The architecture is read from run_config.json (written by the trainer at save time), so the
model is reconstructed exactly as trained. CLI flags can override individual fields, but the
default is to trust run_config.json.

The model itself lives in entexbert2.model (backbone f_theta + task-selected Stage-2 head +
LUPI-weighted loss). This module only handles config/weights loading and the inference helper.

TWO inference paths, selected by the trained model's `task` (read from run_config):
  * regression     : single-window binding score  mu = head(pool)   (Stage-1 trunk)
  * classification : symmetric contrast  ell = a*||P(pool1) - P(pool2)|| + b = logit P(ASB)
Both match entexbert2.model.forward exactly (eval mode -> dropout is identity).
"""

import glob
import json
import os
from typing import Optional

import torch
import transformers

# The trained model class (importing entexbert2.model is side-effect-free).
from entexbert2.model import entexBERT2ForSequencePrediction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_run_config(checkpoint_dir: str) -> dict:
    """Read run_config.json written by the trainer. Errors clearly if absent."""
    path = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No run_config.json in {checkpoint_dir}. It is written by the trainer at save "
            f"time; either re-run training with the current finetune script, or pass the "
            f"task/head fields explicitly."
        )
    with open(path) as f:
        return json.load(f)


def apply_overrides(run_config: dict, overrides: dict) -> dict:
    """Override run_config fields with any non-None values (e.g. from CLI)."""
    rc = dict(run_config)
    for k, v in (overrides or {}).items():
        if v is not None:
            rc[k] = v
    return rc


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def find_weights_file(checkpoint_dir: str) -> str:
    """
    Locate the saved weights. Prefers the top-level save (best model, since the trainer
    runs with load_best_model_at_end), then falls back to the latest checkpoint-* dir.
    """
    candidates = [
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "model.safetensors"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )
    for ckpt in reversed(ckpts):
        for name in ("pytorch_model.bin", "model.safetensors"):
            c = os.path.join(ckpt, name)
            if os.path.exists(c):
                return c

    raise FileNotFoundError(
        f"No pytorch_model.bin / model.safetensors found in {checkpoint_dir} "
        f"or its checkpoint-* subdirectories."
    )


def _load_state_dict(weights_path: str) -> dict:
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")


def load_model_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Load a state dict into model, reporting missing/unexpected keys (don't fail silently)."""
    state_dict = _load_state_dict(weights_path)
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    print(f"Loaded weights from {weights_path}")
    print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("  first missing:", missing[:10])
    if unexpected:
        print("  first unexpected:", unexpected[:10])

    # Heuristic guard: if the trained head didn't load, the analysis would be meaningless.
    # The head params differ by task: regression -> main_head.*, classification -> proj.*/dist_*.
    task = getattr(model, "task", "regression")
    head_prefixes = ("main_head",) if task == "regression" else ("proj", "dist_a", "dist_b")
    head_missing = [k for k in missing if k.startswith(head_prefixes)]
    if head_missing:
        raise RuntimeError(
            f"{task} head weights did not load ({head_missing[:5]}...). The checkpoint and the "
            f"run_config architecture likely disagree. Refusing to run on an untrained head."
        )
    return model


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_model(run_config: dict, device: str = "cpu") -> torch.nn.Module:
    """Instantiate the trained architecture from a (possibly overridden) run_config."""
    if run_config.get("use_lora"):
        raise NotImplementedError(
            "run_config has use_lora=True; LoRA checkpoints need adapter handling that isn't "
            "wired up here yet. Train without LoRA or extend build_model."
        )

    model = entexBERT2ForSequencePrediction(
        model_name_or_path=run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        pooling_mode=run_config["pooling_mode"],
        center_pool_width=run_config["center_pool_width"],
        head_num_layers=run_config["head_num_layers"],
        head_hidden_size=run_config["head_hidden_size"],
        head_activation=run_config.get("head_activation", "gelu"),
        head_dropout=run_config.get("head_dropout", 0.1),
        neff_s=run_config.get("neff_s", 50.0),
        task=run_config.get("task", "regression"),
        proj_dim=run_config.get("proj_dim", 128),
        interaction=run_config.get("interaction", "none"),          # NEW
        x_dim=run_config.get("x_dim", 256),                          # NEW
        x_heads=run_config.get("x_heads", 4),                        # NEW
        x_dropout=run_config.get("x_dropout", 0.1),                  # NEW
        x_readout=run_config.get("x_readout", "mean"),               # NEW
        x_width=run_config.get("x_width", 2),                        # NEW
    )
    return model.to(device)


def load_tokenizer(run_config: dict):
    return transformers.AutoTokenizer.from_pretrained(
        run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        model_max_length=run_config.get("model_max_length", 512),
        trust_remote_code=True,
    )


def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cpu", overrides: dict = None):
    """
    One call: read run_config.json, build the model, load weights, set eval mode, and load
    the matching tokenizer. Returns (model, tokenizer, run_config).
    """
    run_config = apply_overrides(load_run_config(checkpoint_dir), overrides or {})
    model = build_model(run_config, device=device)
    load_model_weights(model, find_weights_file(checkpoint_dir))
    model.eval()
    tokenizer = load_tokenizer(run_config)
    return model, tokenizer, run_config


# ---------------------------------------------------------------------------
# Inference helper (reuses the model's own backbone + pooling -> no drift)
# ---------------------------------------------------------------------------

@torch.no_grad()
def logits_and_embeddings(model, input_ids, attention_mask,
                          input_ids_alt=None, attention_mask_alt=None,
                          return_pools=False, var_tok_idx=None):    # NEW: var_tok_idx (cross_attn)
    """
    Score sequence(s) with the model's own backbone + pooling + head (eval mode -> dropout is
    identity, so this matches the trained forward exactly). Task-aware.

    regression (Stage-1 binding trunk): single-window score mu = head(pool1). Embedding = pool1.
    (Alt inputs are ignored for regression; the signed-twin path has been removed.)

    classification (symmetric contrast): REQUIRES paired inputs. The P(ASB) logit
    ell = a*||z1 - z2|| + b (a = softplus(dist_a) > 0), z = P(pool), matches model.forward.
    Symmetric in the two windows. Embedding = projected contrast z1 - z2.

    return_pools: also return the RAW per-window pools (pool1, pool2). These are what the
    frozen-trunk pre-check dumps; a projection cannot be reconstructed from the contrast alone,
    so probe_frozen_trunk needs both pools.

    Returns:
        return_pools=False:  (logits, pooled_contrast)
        return_pools=True:   (logits, pooled_contrast, pool1, pool2)   (pool2=None if single-seq)
    """
    task = getattr(model, "task", "regression")

    # Route through model._pool_one so pooling matches training exactly. eval mode -> dropout is
    # identity, so this matches model.forward exactly.
    def _pool(ids, mask):
        return model._pool_one(ids, mask)

    interaction = getattr(model, "interaction", "none")             # NEW
    # NEW: cross_attn computes its pools inside the classification branch (needs the alt window);
    # the single-window pool1 is only used by the bi-encoder / regression paths.
    pool1 = None if (task == "classification" and interaction == "cross_attn") else _pool(input_ids, attention_mask)
    pool2 = None

    if task == "classification":
        if input_ids_alt is None:
            raise ValueError(
                "classification scoring requires paired inputs (hap1, hap2); got a single sequence."
            )
        if interaction == "cross_attn":                              # NEW cross-encoder path
            from entexbert2.cross_allele_head import pool as _xpool
            H1 = model._encode_one(input_ids, attention_mask)
            H2 = model._encode_one(input_ids_alt, attention_mask_alt)
            H1x, H2x = model.xattn(H1, H2, attention_mask, attention_mask_alt)
            z1 = model.proj(_xpool(H1x, attention_mask, model.x_readout, var_tok_idx, model.x_width))
            z2 = model.proj(_xpool(H2x, attention_mask_alt, model.x_readout, var_tok_idx, model.x_width))
        else:                                                        # bi-encoder path (unchanged)
            pool2 = _pool(input_ids_alt, attention_mask_alt)
            z1 = model.proj(pool1)
            z2 = model.proj(pool2)
        pooled = z1 - z2                                                 # projected contrast (for probes)
        # Mirror model.forward's classification head EXACTLY: ell = a*||z1 - z2|| + b.
        s = torch.linalg.vector_norm(z1 - z2, dim=-1, keepdim=True)      # (N,1), >= 0
        a = torch.nn.functional.softplus(model.dist_a)
        logits = a * s + model.dist_b                                    # (N,1) = ell = logit P(ASB)
        if return_pools:
            return logits, pooled, pool1, pool2
        return logits, pooled

    # regression (Stage-1 binding trunk): single-window score mu = head(pool1)
    logits = model.main_head(pool1)
    pooled = pool1

    if return_pools:
        return logits, pooled, pool1, pool2
    return logits, pooled


# ---------------------------------------------------------------------------
# Batched inference over a list of sequences (or [ref, alt] pairs)
# ---------------------------------------------------------------------------

def run_inference(checkpoint_dir, texts, batch_size=64, device="cpu",
                  overrides=None, dump_pools=False):
    """
    Load a checkpoint and score a list of inputs. For regression (Stage-1 binding) each input
    is a single sequence -> single-window score. For classification each input is a
    [window1, window2] pair (hap_pair / ref_alt) tokenized SEPARATELY and combined by the
    projected-distance head -- never [SEP]-concatenated, which would bury the single-base
    allelic difference.

    dump_pools=False (default):
        returns (logits, emb, run_config)
          logits : (N, 1)  regression: binding score mu ; classification: P(ASB) logit ell
          emb    : (N, D)  regression: pool1 (H) ; classification: proj contrast (proj_dim)
    dump_pools=True (pair inputs only):
        returns (logits, emb, pool_ref, pool_alt, run_config)
          pool_ref = pool(window1), pool_alt = pool(window2)  -- RAW per-window pools for the
          frozen-trunk pre-check (a head/projection can't be rebuilt from the contrast alone).

    Convention: window1 is the ref / hap1 window, window2 is the alt / hap2 window.
    """
    model, tokenizer, run_config = load_model_and_tokenizer(
        checkpoint_dir, device=device, overrides=overrides)

    is_pair = len(texts) > 0 and isinstance(texts[0], (list, tuple))
    if dump_pools and not is_pair:
        raise ValueError("dump_pools=True requires paired inputs ([window1, window2]).")
    if getattr(model, "task", "regression") == "classification" and not is_pair:
        raise ValueError("classification scoring requires paired inputs ([window1, window2]).")
    mml = run_config.get("model_max_length", 512)
    interaction = run_config.get("interaction", "none")              # NEW

    all_logits, all_emb, all_ref, all_alt = [], [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if is_pair:
            ref = [b[0] for b in batch]
            alt = [b[1] for b in batch]
            if interaction == "cross_attn":                          # NEW: anchored aligned grid
                from entexbert2.anchor_tokenize import anchor_tokenize
                idr, ida, vtx = [], [], []
                for rseq, aseq in zip(ref, alt):
                    dif = [k for k in range(min(len(rseq), len(aseq))) if rseq[k] != aseq[k]]
                    vp = dif[0] if len(dif) == 1 else len(rseq) // 2   # single-base diff (jitter-safe); else center
                    a = anchor_tokenize(tokenizer, rseq, aseq, vp)     # variable length; batch-padded below
                    idr.append(torch.tensor(a["input_ids_ref"]))
                    ida.append(torch.tensor(a["input_ids_alt"]))
                    vtx.append(a["var_tok_idx"])
                ii_r = torch.nn.utils.rnn.pad_sequence(idr, batch_first=True, padding_value=tokenizer.pad_token_id)
                ii_a = torch.nn.utils.rnn.pad_sequence(ida, batch_first=True, padding_value=tokenizer.pad_token_id)
                out = logits_and_embeddings(
                    model, ii_r.to(device), ii_r.ne(tokenizer.pad_token_id).to(device),
                    input_ids_alt=ii_a.to(device),
                    attention_mask_alt=ii_a.ne(tokenizer.pad_token_id).to(device),
                    return_pools=dump_pools, var_tok_idx=torch.tensor(vtx).to(device))
            else:
                enc_r = tokenizer(ref, return_tensors="pt", padding="longest",
                                  max_length=mml, truncation=True)
                enc_a = tokenizer(alt, return_tensors="pt", padding="longest",
                                  max_length=mml, truncation=True)
                out = logits_and_embeddings(
                    model, enc_r["input_ids"].to(device), enc_r["attention_mask"].to(device),
                    input_ids_alt=enc_a["input_ids"].to(device),
                    attention_mask_alt=enc_a["attention_mask"].to(device),
                    return_pools=dump_pools)
            if dump_pools:
                logits, pooled, pool1, pool2 = out
                all_ref.append(pool1.detach().cpu().numpy())
                all_alt.append(pool2.detach().cpu().numpy())
            else:
                logits, pooled = out
        else:
            enc = tokenizer(batch, return_tensors="pt", padding="longest",
                            max_length=mml, truncation=True)
            logits, pooled = logits_and_embeddings(
                model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_emb.append(pooled.detach().cpu().numpy())

    import numpy as np
    logits_out = np.concatenate(all_logits, axis=0)
    emb_out = np.concatenate(all_emb, axis=0)
    if dump_pools:
        return (logits_out, emb_out,
                np.concatenate(all_ref, axis=0), np.concatenate(all_alt, axis=0),
                run_config)
    return logits_out, emb_out, run_config
