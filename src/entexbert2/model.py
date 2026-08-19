"""
entexbert2.model — the streamlined 2-stage ASB model.

ONE trunk (f_theta = DNABERT-2), TWO Stage-2 heads selectable by `task`:

  * task="regression"  (signed twin):  mu = g(f(seq_hap1)) - g(f(seq_hap2))
        predicts the signed allelic log-odds; precision-weighted MSE on the
        Jeffreys logit-ratio label. Direction-aware, deployable. (Also the
        single-sequence Stage-1 binding trunk: input_ids_alt=None -> mu = g(f(seq)).)

  * task="classification"  (symmetric contrast):  s = ||P(h1) - P(h2)||,
        p = sigmoid(a*s + b); predicts P(ASB) from the DISTANCE between the two
        haplotype representations in a learned projection. Symmetric by
        construction (swapping alleles leaves s, hence p, unchanged) -- correct
        for the symmetric AS label. Precision-weighted BCE.

Both Stage-2 heads are trained with the LUPI read-count precision weight
w = n_eff(n) = n(1+s)/(n+s): privileged at train, absent at test (mu / s are
sequence-only -> fully deployable). Stage 2a freezes the trunk; 2b unfreezes it.

The regression head is the antisymmetric head-then-subtract comparator (encodes
DIRECTION). The classification head is a distance in a shared projection (encodes
MAGNITUDE only) -- the two topologies are different because the ASB label is
symmetric (imbalanced regardless of which allele wins) but the logit-ratio label
is signed. `task` therefore selects the head TOPOLOGY, not just the loss.
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
# The model: f_theta (shared trunk) + task-selected Stage-2 head + LUPI loss
# ---------------------------------------------------------------------------

class entexBERT2ForSequencePrediction(torch.nn.Module):
    """
    Stage-1 trunk (fine-tuned on binding affinity) + Stage-2 head (task-selected).

    task="regression":
        forward scores both windows through the SAME trunk and head and returns the
        signed contrast mu = head(pool(seq1)) - head(pool(seq2)); with seq1=hap1,
        seq2=hap2, mu is the predicted logit P(hap1). Single-window (input_ids_alt=None)
        -> mu = head(pool(seq)) is the Stage-1 binding score.
        Loss: precision-weighted MSE, L = sum_i w_i (mu_i - y_i)^2 / sum_i w_i.

    task="classification":
        forward projects each window's pooled representation with a SHARED projection
        P (768 -> proj_dim), forms the distance s = ||P(h1) - P(h2)||_2, and returns
        the logit ell = a*s + b so that p = sigmoid(ell) = P(ASB). Symmetric in the two
        haplotypes by construction.
        Loss: precision-weighted BCE, L = sum_i w_i BCE(sigmoid(ell_i), y_i) / sum_i w_i.

    In both cases w_i = n_eff(n_i) = n_i (1 + s) / (n_i + s), normalized to mean 1,
    with n_i the privileged total read depth (passed as `depth`). w saturates at 1 + s
    so ultra-deep loci cannot dominate; low-depth loci are down-weighted, not discarded.
    For classification this down-weights UNDERPOWERED NEGATIVES (loci called non-AS only
    because the test lacked depth) -- the main label-contamination source. s = neff_s.
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
        task: str = "regression",                 # NEW: 'regression' | 'classification'
        num_labels: int = 1,                       # NEW: regression head width T (multi-track Stage-1 = #tissues; 1 = scalar)
        proj_dim: int = 128,                       # NEW: classification projection dim d
        learned_metric: bool = False,              # NEW: classification -- s = ||L(z1-z2)|| (Mahalanobis) instead of ||z1-z2||
    ):
        super().__init__()
        if pooling_mode not in ("cls", "center_mean"):
            raise ValueError(f"pooling_mode must be 'cls' or 'center_mean', got {pooling_mode!r}.")
        if pooling_mode == "center_mean" and center_pool_width % 2 == 0:
            raise ValueError("center_pool_width must be odd for symmetric center pooling.")
        if neff_s <= 0:
            raise ValueError(f"neff_s (beta-binomial concentration) must be > 0, got {neff_s}.")
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}.")
        # NEW: num_labels is the regression head width T (multi-track Stage-1 predicts one
        # fold-change per tissue). T=1 is the original scalar path. Only regression uses it;
        # the classification contrast head always emits a single P(ASB) logit.
        num_labels = int(num_labels)
        if num_labels < 1:
            raise ValueError(f"num_labels must be >= 1, got {num_labels}.")
        if task == "classification" and num_labels != 1:
            raise ValueError(
                f"classification contrast head emits one P(ASB) logit; num_labels must be 1, "
                f"got {num_labels}."
            )

        self.pooling_mode = pooling_mode
        self.center_pool_width = int(center_pool_width)
        self.neff_s = float(neff_s)
        self.task = task                                    # NEW
        self.num_labels = num_labels                        # NEW: regression head width T

        # Shared pretrained trunk f_theta.
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)

        head_input = hidden_size

        if task == "regression":
            # Twin head g_phi: per-window binding logit(s). output_size = num_labels:
            #   T=1  -> scalar (original signed-twin ASB + single-track Stage-1 binding)
            #   T>1  -> one binding logit per tissue track (multi-track Stage-1 supervision).
            # Subtracted across haplotypes elementwise in the Stage-2 twin path.
            self.main_head = build_prediction_head(
                input_size=head_input, output_size=self.num_labels, num_layers=head_num_layers,
                hidden_size=head_hidden_size, activation=head_activation, dropout=head_dropout,
            )
            self.proj = None
            self.dist_a = None
            self.dist_b = None
            self.learned_metric = False
            self.metric_map = None
        else:
            # NEW: classification contrast head.
            # Shared projection P: hidden -> proj_dim (num_layers>=2 => nonlinear).
            if proj_dim <= 0:
                raise ValueError(f"proj_dim must be > 0 for classification, got {proj_dim}.")
            self.proj = build_prediction_head(
                input_size=head_input, output_size=proj_dim, num_layers=head_num_layers,
                hidden_size=head_hidden_size, activation=head_activation, dropout=head_dropout,
            )
            # logit = a * s + b ; a>0 enforced via softplus at forward time.
            # a_raw init so softplus(a_raw) ~ 1.0; b init 0 -> p starts ~ 0.5 at small distances.
            self.dist_a = torch.nn.Parameter(torch.tensor(0.5413))   # softplus(0.5413) ~ 1.0
            self.dist_b = torch.nn.Parameter(torch.tensor(0.0))
            # NEW (#1 learned-metric head): s = ||L (z1 - z2)|| with a learned square map L (d x d),
            # i.e. a Mahalanobis distance d^T M d, M = L^T L >= 0. Weights the ASB-relevant directions
            # instead of the isotropic Euclidean norm. Initialized to IDENTITY so at step 0 it is
            # byte-identical to the plain-distance head (s = ||z1-z2||) -- the change is a pure superset
            # and only departs from Euclidean as L trains. When learned_metric=False, metric_map stays
            # None and the forward path is exactly the original.
            self.learned_metric = bool(learned_metric)
            if self.learned_metric:
                self.metric_map = torch.nn.Linear(proj_dim, proj_dim, bias=False)
                with torch.no_grad():
                    self.metric_map.weight.copy_(torch.eye(proj_dim))   # identity at init
            else:
                self.metric_map = None
            self.main_head = None

        if freeze_backbone:
            self.freeze_backbone()

    # ---- Stage-1 -> Stage-2 transfer ---------------------------------------
    def freeze_backbone(self):
        """Stage 2a: freeze the trunk so only the head trains (no collapse risk, reuses Stage 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def init_backbone_from(self, checkpoint_path: str, map_location="cpu") -> int:
        """
        Load ONLY the `backbone.*` weights from a Stage-1 checkpoint state_dict into this
        (fresh-head) model. The Stage-1 head keys are dropped; the Stage-2 head keeps its
        random init. Returns the number of backbone tensors loaded. strict=False by design.
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
        # 'missing' lists the fresh head params (expected); 'unexpected' should be empty.
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

    def _pool_one(self, input_ids, attention_mask, **kwargs):
        """One window -> pooled representation h in R^hidden."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            return_dict=True, **kwargs)
        return self.dropout(self._pool(out, attention_mask=attention_mask))

    def _score_one(self, input_ids, attention_mask, **kwargs):
        """One window -> binding logit(s) + pooled representation (regression head).

        Returns (B,) when num_labels==1 (scalar path) and (B, T) when num_labels>1
        (multi-track Stage-1). We squeeze the last dim ONLY for the scalar head so the
        downstream signed-twin contrast and scalar loss are unchanged.
        """
        pooled = self._pool_one(input_ids, attention_mask, **kwargs)
        out = self.main_head(pooled)                      # (B, T)
        if self.num_labels == 1:
            out = out.squeeze(-1)                          # (B,)  scalar path (unchanged)
        return out, pooled

    # ---- LUPI precision weight --------------------------------------------
    def _neff_weight(self, depth, like):
        """w_i = n_eff(n_i) normalized to mean 1; falls back to ones if depth is None."""
        if depth is None:
            return torch.ones_like(like)
        n = depth.float().view_as(like).clamp(min=1.0)
        s = self.neff_s
        w = n * (1.0 + s) / (n + s)          # n_eff, saturates at 1 + s
        return w / w.mean()                  # normalize to mean 1 (LR-invariant)

    # ---- Forward -----------------------------------------------------------
    def forward(
        self,
        input_ids=None,            # hap1 window  (sequence1)
        attention_mask=None,
        input_ids_alt=None,        # hap2 window  (sequence2)
        attention_mask_alt=None,
        labels=None,               # regression: Jeffreys logit-ratio (or (B,T) per-track binding) ; classification: 0/1 AS
        depth=None,                # n = total read depth (privileged; weight only)
        label_mask=None,           # NEW: (B,T) 0/1 observed-tissue mask for multi-track Stage-1 (None => all observed)
        **kwargs,
    ):
        # ================= classification: symmetric contrast (distance) =================
        if self.task == "classification":
            if input_ids_alt is None:
                raise RuntimeError(
                    "classification (contrast) head requires a paired (hap1, hap2) batch, but "
                    "input_ids_alt did not reach forward. Use --input_mode hap_pair and the twin collator."
                )
            h1 = self._pool_one(input_ids, attention_mask, **kwargs)
            h2 = self._pool_one(input_ids_alt, attention_mask_alt, **kwargs)
            z1 = self.proj(h1)
            z2 = self.proj(h2)
            d = z1 - z2                                            # symmetric under swap up to sign
            if self.metric_map is not None:
                d = self.metric_map(d)                             # learned metric: L(z1 - z2)
            s = torch.linalg.vector_norm(d, dim=-1)                # >= 0, symmetric in (h1,h2)
            a = torch.nn.functional.softplus(self.dist_a)          # a > 0: distance monotone in P(ASB)
            ell = a * s + self.dist_b                              # logit of P(ASB)

            loss = None
            if labels is not None:
                y = labels.float().view_as(ell)
                w = self._neff_weight(depth, ell)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(ell, y, reduction="none")
                loss = (w * bce).mean()
            return SequenceClassifierOutput(loss=loss, logits=ell.unsqueeze(-1))

        # ================= regression: signed twin (+ single-seq Stage-1) =================
        # single-sequence branch = Stage-1 binding regression (one window -> one score).
        # Stage-2 ASB still passes hap2 (input_ids_alt) and gets the signed twin contrast.
        logit_hap1, _ = self._score_one(input_ids, attention_mask, **kwargs)
        if input_ids_alt is None:
            mu = logit_hap1                                # single-window binding score (B,) or (B,T)
        else:
            logit_hap2, _ = self._score_one(input_ids_alt, attention_mask_alt, **kwargs)
            mu = logit_hap1 - logit_hap2                   # signed contrast = logit P(hap1)

        # ---- Multi-track Stage-1 (num_labels > 1): masked multi-output MSE ----------------
        # mu is (B, T). labels is (B, T) per-tissue log1p fold-change; label_mask is (B, T) in
        # {0,1} marking tissues actually assayed at each locus. Only observed (mask=1) entries
        # contribute; the loss is the mean over observed entries so it is invariant to how many
        # tissues a locus has. depth-based n_eff weighting is not used for the binding trunk.
        if self.num_labels > 1:
            loss = None
            if labels is not None:
                y = labels.float().view_as(mu)
                if label_mask is not None:
                    m = label_mask.float().view_as(mu)
                else:
                    m = torch.ones_like(mu)
                sq = m * (mu - y) ** 2
                denom = m.sum().clamp(min=1.0)
                loss = sq.sum() / denom
            return SequenceClassifierOutput(loss=loss, logits=mu)   # (B, T), no unsqueeze

        # ---- Scalar path (num_labels == 1): unchanged signed-twin / single-track ----------
        loss = None
        if labels is not None:
            y = labels.float().view_as(mu)
            w = self._neff_weight(depth, mu)
            loss = (w * (mu - y) ** 2).mean()

        return SequenceClassifierOutput(loss=loss, logits=mu.unsqueeze(-1))
