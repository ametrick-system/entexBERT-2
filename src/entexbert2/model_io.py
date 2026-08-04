#!/usr/bin/env python3

"""
Shared model I/O for entexBERT-2 evaluation and plotting.

This is the single place that knows how to rebuild a trained entexBERT-2 model and run
inference. analyze.py and both plotters import it, so the architecture is never re-specified

The architecture/task config is read from run_config.json (written by the trainer at save
time), so the model is reconstructed exactly as trained. CLI flags can override individual
fields, but the default is to trust run_config.json.
"""

import glob
import json
import os
from typing import Optional

import torch
import transformers

# The trained model class. Importing the trainer module is side-effect-free
# (train() is guarded by __main__), so this just pulls in the class definition.
from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction

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

    # Heuristic guard: if the head didn't load, the analysis would be meaningless.
    head_missing = [k for k in missing if k.startswith("main_head")]
    if head_missing:
        raise RuntimeError(
            f"main_head weights did not load ({head_missing[:5]}...). The checkpoint and the "
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

    aux_names = run_config.get("aux_task_names") or []
    aux_types = run_config.get("aux_task_types") or []
    aux_num = run_config.get("aux_num_labels") or []

    model = entexBERT2ForSequencePrediction(
        model_name_or_path=run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        main_task=run_config["task"],
        main_num_labels=run_config["main_num_labels"],
        aux_task_names=aux_names,
        aux_task_types=aux_types,
        aux_num_labels=aux_num,
        lambda_aux=[1.0] * len(aux_names),  # unused at inference; length must match
        pooling_mode=run_config["pooling_mode"],
        center_pool_width=run_config["center_pool_width"],
        head_num_layers=run_config["head_num_layers"],
        head_hidden_size=run_config["head_hidden_size"],
        head_activation=run_config["head_activation"],
        head_dropout=run_config["head_dropout"],
        predict_sigma=run_config.get("predict_sigma", False),
        sigma_logvar_clamp=run_config.get("sigma_logvar_clamp", 10.0),
    )
    # Twin contrast mode (signed = head(alt)-head(ref); symmetric_abs = head(|pool(alt)-pool(ref)|)).
    # Set as an attribute so logits_and_embeddings mirrors training even if the constructor
    # predates this field.
    model.contrast_mode = run_config.get("contrast_mode", "signed")
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
                          input_ids_alt=None, attention_mask_alt=None):
    """
    Score a sequence with the model's own backbone + pooling + head (eval mode -> dropout is
    identity, so this matches the trained forward exactly).

    TWIN / score-and-subtract: if alt inputs are given (ref_alt_pair), the prediction is the
    SIGNED contrast head(alt) - head(ref), and the returned embedding is the contrast
    pool(alt) - pool(ref) -- the representation the head's prediction is actually based on.
    Single-sequence inference (alt=None) is unchanged.
    """
    predict_sigma = getattr(model, "predict_sigma", False) and getattr(model, "sigma_head", None) is not None
    clamp = getattr(model, "sigma_logvar_clamp", 10.0)

    def _score(ids, mask):
        backbone_outputs = model.backbone(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled = model._pool_sequence_representation(backbone_outputs, attention_mask=mask)
        return model.main_head(pooled), pooled

    logits, pooled = _score(input_ids, attention_mask)
    log_var = None

    if input_ids_alt is not None:
        logits_alt, pooled_alt = _score(input_ids_alt, attention_mask_alt)
        if getattr(model, "contrast_mode", "signed") == "symmetric_abs":
            # Symmetric target: prediction is head(|pool(alt)-pool(ref)|); sigma reads the SAME
            # contrast vector (identical to training forward()).
            contrast = torch.abs(pooled_alt - pooled)
            logits = model.main_head(contrast)
            if predict_sigma:
                log_var = model.sigma_head(contrast).clamp(-clamp, clamp)
            pooled = contrast
        else:
            # Signed twin: Var(Delta)=Var(ref)+Var(alt) -> logsumexp(lv_ref, lv_alt), computed
            # from the raw ref/alt pools BEFORE pooled is overwritten (identical to training).
            if predict_sigma:
                lv_ref = model.sigma_head(pooled).clamp(-clamp, clamp)
                lv_alt = model.sigma_head(pooled_alt).clamp(-clamp, clamp)
                log_var = torch.logsumexp(torch.cat([lv_ref, lv_alt], dim=-1),
                                          dim=-1, keepdim=True).clamp(-clamp, clamp)
            logits = logits_alt - logits
            pooled = pooled_alt - pooled
    
    elif predict_sigma:
        # single-sequence inference
        log_var = model.sigma_head(pooled).clamp(-clamp, clamp)
    if predict_sigma:
        # pack [mu, log_var] so the caller can score Delta/sigma; sigma = exp(0.5*log_var)
        mu = logits if logits.ndim > 1 else logits.unsqueeze(-1)
        logits = torch.cat([mu, log_var], dim=-1)

    return logits, pooled