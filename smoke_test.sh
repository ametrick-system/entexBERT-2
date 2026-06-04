python - <<'PY'
import torch
import transformers
import pandas as pd
import numpy as np
from pathlib import Path

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction

OFFICIAL_MODEL = "zhihan1996/DNABERT-2-117M"
PATCHED_MODEL = "DNABERT-2-117M-attention"

RUN_DIR = Path("/home/asm242/entexBERT-2/AS/1/CTCF/all_tissues/hap_pair_classification")
CHECKPOINT = RUN_DIR / "output/checkpoint-8600/pytorch_model.bin"
TEST_CSV = RUN_DIR / "input/test.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(TEST_CSV).head(16)

official_tokenizer = transformers.AutoTokenizer.from_pretrained(
    OFFICIAL_MODEL,
    trust_remote_code=True,
    model_max_length=512,
)

patched_tokenizer = transformers.AutoTokenizer.from_pretrained(
    PATCHED_MODEL,
    trust_remote_code=True,
    model_max_length=512,
)

official_model = entexBERT2ForSequencePrediction(
    model_name_or_path=OFFICIAL_MODEL,
    main_task="classification",
    main_num_labels=2,
    pooling_mode="cls",
    head_num_layers=1,
    head_hidden_size=-1,
    head_activation="gelu",
    head_dropout=0.1,
)

patched_model = entexBERT2ForSequencePrediction(
    model_name_or_path=PATCHED_MODEL,
    main_task="classification",
    main_num_labels=2,
    pooling_mode="cls",
    head_num_layers=1,
    head_hidden_size=-1,
    head_activation="gelu",
    head_dropout=0.1,
)

state = torch.load(CHECKPOINT, map_location="cpu")

missing_official, unexpected_official = official_model.load_state_dict(state, strict=False)
missing_patched, unexpected_patched = patched_model.load_state_dict(state, strict=False)

print("Official missing:", len(missing_official))
print("Official unexpected:", len(unexpected_official))
print("Patched missing:", len(missing_patched))
print("Patched unexpected:", len(unexpected_patched))

if missing_patched:
    print("First patched missing:", missing_patched[:10])
if unexpected_patched:
    print("First patched unexpected:", unexpected_patched[:10])

official_model.to(DEVICE).eval()
patched_model.to(DEVICE).eval()

seq1 = df["sequence1"].astype(str).tolist()
seq2 = df["sequence2"].astype(str).tolist()

official_inputs = official_tokenizer(
    seq1,
    seq2,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

patched_inputs = patched_tokenizer(
    seq1,
    seq2,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

official_inputs = {k: v.to(DEVICE) for k, v in official_inputs.items()}
patched_inputs = {k: v.to(DEVICE) for k, v in patched_inputs.items()}

with torch.no_grad():
    official_outputs = official_model(**official_inputs)
    patched_outputs_no_attn = patched_model(**patched_inputs)
    patched_outputs_attn = patched_model(
        **patched_inputs,
        output_attentions=True,
        output_hidden_states=True,
    )

official_logits = official_outputs.logits.detach().cpu()
patched_logits_no_attn = patched_outputs_no_attn.logits.detach().cpu()
patched_logits_attn = patched_outputs_attn.logits.detach().cpu()

official_probs = torch.softmax(official_logits, dim=-1)[:, 1].numpy()
patched_probs_no_attn = torch.softmax(patched_logits_no_attn, dim=-1)[:, 1].numpy()
patched_probs_attn = torch.softmax(patched_logits_attn, dim=-1)[:, 1].numpy()

print("\nMax abs logit diff, official vs patched no-attn:")
print(torch.max(torch.abs(official_logits - patched_logits_no_attn)).item())

print("\nMax abs prob diff, official vs patched no-attn:")
print(np.max(np.abs(official_probs - patched_probs_no_attn)))

print("\nMax abs prob diff, patched no-attn vs patched attn:")
print(np.max(np.abs(patched_probs_no_attn - patched_probs_attn)))

print("\nFirst 10 official probs:")
print(official_probs[:10])

print("\nFirst 10 patched no-attn probs:")
print(patched_probs_no_attn[:10])

print("\nFirst 10 patched attn probs:")
print(patched_probs_attn[:10])

print("\nPatched attentions is None:", patched_outputs_attn.attentions is None)
if patched_outputs_attn.attentions is not None:
    print("num attention layers:", len(patched_outputs_attn.attentions))
    print("attention[0] shape:", patched_outputs_attn.attentions[0].shape)
PY