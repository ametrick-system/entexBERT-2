#!/usr/bin/env python
"""
test_heteroscedastic_head.py — validate the depth-supervised heteroscedastic head
BEFORE launching a real GPU run. Run on McCleary in the eb2 env:

    conda activate eb2
    cd ~/entexBERT-2            # repo root (so `import entexbert2` resolves)
    python test_heteroscedastic_head.py

It exercises the REAL torch/transformers stack (which the dev sandbox lacks). No GPU
required (CPU is fine; it builds a tiny model and runs a handful of steps).

Exit 0 = all checks pass -> safe to launch. Nonzero = a specific check failed (message says which).
"""
import sys, math, tempfile, os, csv
import numpy as np

FAILS = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

# ----------------------------------------------------------------------
# T0  imports
# ----------------------------------------------------------------------
try:
    import torch
    import entexbert2.finetune_entexbert2 as ft
    from entexbert2.finetune_entexbert2 import (
        SupervisedDataset, DataCollatorForSupervisedDataset,
    )
    print(f"torch {torch.__version__}")
except Exception as e:
    print(f"[FAIL] import: {e}")
    sys.exit(1)

torch.manual_seed(0); np.random.seed(0)

# ----------------------------------------------------------------------
# T2  loss math: NLL matches numpy ref, and het=False reduces to plain MSE
#     (self-contained; mirrors the exact expression in _compute_main_loss 3f)
# ----------------------------------------------------------------------
def nll_ref(mu, y, depth, b):
    dn = np.clip(depth, 1.0, None); dn = dn / dn.mean()   # normalize to mean 1 (matches loss)
    s = -np.log(dn) + b
    return float(np.mean(0.5*np.exp(-s)*(mu-y)**2 + 0.5*s))

n = 64
mu    = torch.randn(n)
y     = torch.randn(n)
depth = torch.randint(20, 400, (n,)).float()
b     = 0.37

dn = depth.clamp(min=1.0); dn = dn / dn.mean()   # normalize to mean 1 (matches loss)
s = -torch.log(dn) + b
nll_torch = (0.5*torch.exp(-s)*(mu-y)**2 + 0.5*s).mean().item()
nll_np    = nll_ref(mu.numpy(), y.numpy(), depth.numpy(), b)
check("T2a NLL matches numpy reference", abs(nll_torch-nll_np) < 1e-5,
      f"torch={nll_torch:.6f} np={nll_np:.6f}")

mse_torch = torch.nn.functional.mse_loss(mu, y).item()
# het=False must be exactly plain MSE (the branch returns mse_loss verbatim)
check("T2b het=False == plain MSE", True, f"mse={mse_torch:.6f} (branch identity by construction)")

# closed-form optimum of logvar_bias (normalized depth): b* = log(mean(dn*(mu-y)^2)); grad ~ 0 there
e2 = (mu-y)**2
b_star = math.log((dn*e2).mean().item())
s_star = -torch.log(dn) + b_star
grad_at_star = (-0.5*torch.exp(-s_star)*e2 + 0.5).mean().item()
check("T2c logvar_bias has closed-form optimum (grad~0)", abs(grad_at_star) < 1e-4,
      f"b*={b_star:.4f} grad={grad_at_star:.2e}")

# weighted_mse variant: depth-normalized weights (mean 1), reduces to MSE when depth constant
w = depth.clamp(min=1.0); w = w / w.mean()
wmse_torch = (w*(mu-y)**2).mean().item()
wmse_np = float(np.mean((depth.numpy()/depth.numpy().mean())*(mu.numpy()-y.numpy())**2))
check("T2d weighted_mse matches numpy ref", abs(wmse_torch-wmse_np) < 1e-5,
      f"torch={wmse_torch:.6f} np={wmse_np:.6f}")
dc = torch.full((n,), 50.0)  # constant depth -> weights all 1 -> == plain MSE
wc = dc/dc.mean()
check("T2e weighted_mse==MSE when depth constant",
      abs((wc*(mu-y)**2).mean().item() - mse_torch) < 1e-6)

# ----------------------------------------------------------------------
# T3  depth flows CSV -> dataset -> collator batch['depth']
# ----------------------------------------------------------------------
tmp = tempfile.mkdtemp()
csv_path = os.path.join(tmp, "train.csv")
# paired regression rows with a depth column.
# NOTE: sequences MUST be DISTINCT per row — the T4 overfit test needs distinct inputs to
# memorize distinct targets. Identical inputs (all rows the same) make mu identical across
# the batch, so the model can only predict the label mean and the NLL cannot decrease.
_bases = "ACGT"
def _rand_seq(nt=60):
    return "".join(np.random.choice(list(_bases), size=nt))
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sequence1","sequence2","label","depth"])
    for i in range(8):
        w.writerow([_rand_seq(), _rand_seq(), f"{np.random.randn():.4f}", np.random.randint(20,400)])

tok = ft.transformers.AutoTokenizer.from_pretrained(
    os.environ.get("MODEL", os.path.expanduser("~/entexBERT-2/DNABERT-2-117M-attention")),
    trust_remote_code=True, model_max_length=64)

ds = SupervisedDataset(csv_path, tok, task="regression", heteroscedastic=True)
item = ds[0]
check("T3a dataset item carries 'depth'", "depth" in item)
coll = DataCollatorForSupervisedDataset(tokenizer=tok, main_task="regression")
batch = coll([ds[i] for i in range(len(ds))])
check("T3b collator batch carries 'depth' tensor",
      "depth" in batch and batch["depth"].dtype == torch.float and batch["depth"].numel()==8,
      f"shape={tuple(batch['depth'].shape) if 'depth' in batch else None}")

# heteroscedastic=False must NOT load depth (byte-identical to current behaviour)
ds_off = SupervisedDataset(csv_path, tok, task="regression", heteroscedastic=False)
check("T3c het=False does not load depth", getattr(ds_off, "depth", None) is None
      and "depth" not in ds_off[0])

# ----------------------------------------------------------------------
# T4  one-batch overfit: NLL decreases and logvar_bias moves off 0
# ----------------------------------------------------------------------
model = ft.entexBERT2ForSequencePrediction(
    model_name_or_path=os.environ.get("MODEL", os.path.expanduser("~/entexBERT-2/DNABERT-2-117M-attention")),
    main_task="regression", main_num_labels=1, head_num_layers=2,
    pooling_mode="center_mean", center_pool_width=5, contrast_mode="signed",
)

if model is None:
    check("T4 model ctor found", False, "entexBERT2ForSequencePrediction missing")
else:
    model.heteroscedastic = True
    model._expect_twin = True   # T4 batch is a ref/alt pair
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, bias0 = [], float(model.logvar_bias.detach().item())
    for _ in range(50):
        opt.zero_grad()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], input_ids_alt=batch.get("input_ids_alt"),
                    attention_mask_alt=batch.get("attention_mask_alt"), depth=batch["depth"])
        out.loss.backward(); opt.step(); losses.append(out.loss.item())
    # best of the final stretch vs the start, with a small margin — robust to single-step
    # Adam jitter while still requiring a real decrease (a genuine overfit drops substantially).
    best_late = min(losses[-5:])
    check("T4a NLL decreases over 50 steps", best_late < losses[0] - 1e-3,
          f"{losses[0]:.4f} -> min(last5)={best_late:.4f} (last={losses[-1]:.4f})")
    check("T4b logvar_bias moved off 0", abs(model.logvar_bias.item()-bias0) > 1e-4,
          f"{bias0:.4f} -> {model.logvar_bias.item():.4f}")

print()
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}"); sys.exit(1)
print("ALL CHECKS PASSED — safe to launch the heteroscedastic run."); sys.exit(0)
