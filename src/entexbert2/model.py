"""
entexbert2.model — the streamlined 2-stage ASB model.

ONE model class, ONE task: a DNABERT-2 trunk (f_theta) with a twin prediction head
(g_phi) that scores allele-specific binding as a signed contrast between the two
haplotype windows, trained with a privileged read-count precision weight.

    mu = g_phi(f_theta(seq_hap1)) - g_phi(f_theta(seq_hap2))            # predicted allelic log-odds

At inference mu is produced from sequence alone (no counts) -> fully deployable.
The read counts enter ONLY through the loss weight (LUPI: privileged at train, absent
at test). Everything that served the abandoned approaches -- classification, the sigma
head, the LUPI aux-task multi-head machinery, the depth-tied heteroscedastic NLL, the
betabinomial task, the symmetric_abs contrast -- is intentionally absent.
"""

from typing import Optional

import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput


# ---------------------------------------------------------------------------
# Prediction head (g_phi): linear (num_layers=1) or MLP (num_layers>=2)
# ---------------------------------------------------------------------------

def get_activation_module(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported head activation {name!r} (gelu|relu|tanh|silu).")


def build_prediction_head(
    input_size: int,
    output_size: int = 1,
    num_layers: int = 1,
    hidden_size: int = -1,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    num_layers counts total Linear layers:
      1  -> Linear(input_size -> output_size)                 (linear head)
      2  -> Linear(input->hidden) + act + drop + Linear(hidden->output)
      3+ -> deeper MLP
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    if hidden_size == -1:
        hidden_size = input_size
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive or -1, got {hidden_size}.")
    if not (0 <= dropout < 1):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    if num_layers == 1:
        return torch.nn.Linear(input_size, output_size)

    layers = [torch.nn.Linear(input_size, hidden_size),
              get_activation_module(activation),
              torch.nn.Dropout(dropout)]
    for _ in range(num_layers - 2):
        layers += [torch.nn.Linear(hidden_size, hidden_size),
                   get_activation_module(activation),
                   torch.nn.Dropout(dropout)]
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# The model: f_theta (shared trunk) + g_phi (twin head) + precision-weighted loss
# ---------------------------------------------------------------------------

class entexBERT2ForSequencePrediction(torch.nn.Module):
    """
    Stage-1 trunk (fine-tuned on binding affinity) + Stage-2 twin ASB head.

    forward scores both haplotype windows through the SAME trunk and head and returns
    the signed contrast mu = head(pool(seq1)) - head(pool(seq2)). With seq1 = hap1 and
    seq2 = hap2, mu is the predicted logit P(hap1): mu > 0  <=>  P(hap1) > 1/2.

    Loss (when labels given): precision-weighted MSE on the logit scale,
        L = sum_i w_i (mu_i - y_i)^2 / sum_i w_i,
        w_i = n_eff(n_i) = n_i (1 + s) / (n_i + s),  normalized to mean 1,
    where y_i is the (build-time, Jeffreys-smoothed) observed log-odds label and n_i is
    the privileged total read depth (passed in as `depth`). w saturates at 1 + s so
    ultra-deep loci cannot dominate the gradient; low-depth loci are down-weighted, not
    discarded. s = neff_s is a fixed hyperparameter (dispersion of the beta-binomial).
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        pooling_mode: str = "center_mean",
        center_pool_width: int = 5,
        head_num_layers: int = 1,
        head_hidden_size: int = -1,
        head_activation: str = "gelu",
        head_dropout: float = 0.1,
        neff_s: float = 50.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if pooling_mode not in ("cls", "center_mean"):
            raise ValueError(f"pooling_mode must be 'cls' or 'center_mean', got {pooling_mode!r}.")
        if pooling_mode == "center_mean" and center_pool_width % 2 == 0:
            raise ValueError("center_pool_width must be odd for symmetric center pooling.")
        if neff_s <= 0:
            raise ValueError(f"neff_s (beta-binomial concentration) must be > 0, got {neff_s}.")

        self.pooling_mode = pooling_mode
        self.center_pool_width = int(center_pool_width)
        self.neff_s = float(neff_s)

        # Shared pretrained trunk f_theta.
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)

        # Twin head g_phi: scalar output (a per-window binding logit).
        self.main_head = build_prediction_head(
            input_size=hidden_size, output_size=1, num_layers=head_num_layers,
            hidden_size=head_hidden_size, activation=head_activation, dropout=head_dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

    # ---- Stage-1 -> Stage-2 transfer ---------------------------------------
    def freeze_backbone(self):
        """Stage 2a: freeze the trunk so only g_phi trains (no collapse risk, reuses Stage 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def init_backbone_from(self, checkpoint_path: str, map_location="cpu") -> int:
        """
        Load ONLY the `backbone.*` weights from a Stage-1 checkpoint state_dict into this
        (fresh-head) model. The old Stage-1 head keys are dropped; g_phi keeps its random
        init. Returns the number of backbone tensors loaded. strict=False by design.
        """
        sd = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd and "backbone.embeddings" not in str(sd.keys()):
            sd = sd["state_dict"]
        backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        if not backbone_sd:
            raise ValueError(
                f"No 'backbone.*' keys found in {checkpoint_path}; got prefixes "
                f"{sorted({k.split('.')[0] for k in sd})}. Is this an entexBERT2 checkpoint?"
            )
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        # 'missing' will list main_head.* (expected: fresh head); 'unexpected' should be empty.
        if unexpected:
            raise ValueError(f"Unexpected keys while loading backbone: {unexpected[:5]} ...")
        return len(backbone_sd)

    # ---- Pooling -----------------------------------------------------------
    def _pool(self, backbone_outputs, attention_mask=None):
        if hasattr(backbone_outputs, "last_hidden_state") and backbone_outputs.last_hidden_state is not None:
            seq = backbone_outputs.last_hidden_state
        elif isinstance(backbone_outputs, dict) and "last_hidden_state" in backbone_outputs:
            seq = backbone_outputs["last_hidden_state"]
        elif isinstance(backbone_outputs, (tuple, list)) and torch.is_tensor(backbone_outputs[0]) \
                and backbone_outputs[0].ndim == 3:
            seq = backbone_outputs[0]
        else:
            raise ValueError(f"Cannot extract token hidden states from {type(backbone_outputs)}.")

        if self.pooling_mode == "cls":
            return seq[:, 0, :]

        # center_mean: mean-pool center_pool_width tokens around the middle of each valid window
        bsz, max_len, _ = seq.shape
        half = self.center_pool_width // 2
        pooled = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else max_len
            valid = max(valid, 1)
            center = valid // 2
            start = max(0, center - half)
            end = min(valid, center + half + 1)
            pooled.append(seq[b, start:end, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _score_one(self, input_ids, attention_mask, **kwargs):
        """One window -> scalar binding logit + pooled representation."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            return_dict=True, **kwargs)
        pooled = self.dropout(self._pool(out, attention_mask=attention_mask))
        return self.main_head(pooled).squeeze(-1), pooled

    # ---- Forward: twin contrast + precision-weighted loss ------------------
    def forward(
        self,
        input_ids=None,            # hap1 window  (sequence1)
        attention_mask=None,
        input_ids_alt=None,        # hap2 window  (sequence2)
        attention_mask_alt=None,
        labels=None,               # y = logit((k+0.5)/(n+1))  (build-time, Jeffreys)
        depth=None,                # n = total read depth (privileged; weight only)
        **kwargs,
    ):
        if input_ids_alt is None:
            raise RuntimeError(
                "entexBERT2 ASB model requires a paired (hap1, hap2) batch, but input_ids_alt "
                "did not reach forward. Set remove_unused_columns=False and use the twin collator."
            )

        logit_hap1, _ = self._score_one(input_ids, attention_mask, **kwargs)
        logit_hap2, _ = self._score_one(input_ids_alt, attention_mask_alt, **kwargs)
        mu = logit_hap1 - logit_hap2                       # signed contrast = logit P(hap1)

        loss = None
        if labels is not None:
            y = labels.float().view_as(mu)
            if depth is None:
                # fall back to unweighted MSE (still valid; no privileged weighting)
                loss = torch.nn.functional.mse_loss(mu, y)
            else:
                n = depth.float().view_as(mu).clamp(min=1.0)
                s = self.neff_s
                w = n * (1.0 + s) / (n + s)               # n_eff, saturates at 1 + s
                w = w / w.mean()                          # normalize to mean 1 (LR-invariant)
                loss = (w * (mu - y) ** 2).mean()

        return SequenceClassifierOutput(loss=loss, logits=mu.unsqueeze(-1))
