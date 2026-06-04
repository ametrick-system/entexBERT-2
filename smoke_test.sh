python - <<'PY'
import torch
import transformers

MODEL_PATH = "DNABERT-2-117M-attention"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    model_max_length=512,
)

model = transformers.AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model.eval()

seq1 = "A" * 256 + "C" + "G" * 256
seq2 = "A" * 256 + "T" + "G" * 256

inputs = tokenizer(
    seq1,
    seq2,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

with torch.no_grad():
    outputs = model(
        **inputs,
        return_dict=True,
        output_attentions=True,
        output_hidden_states=True,
    )

print("Output type:", type(outputs))
print("last_hidden_state:", outputs.last_hidden_state.shape)
print("pooler_output:", None if outputs.pooler_output is None else outputs.pooler_output.shape)

print("hidden_states is None:", outputs.hidden_states is None)
if outputs.hidden_states is not None:
    print("num hidden states:", len(outputs.hidden_states))
    print("hidden_states[0]:", outputs.hidden_states[0].shape)
    print("hidden_states[-1]:", outputs.hidden_states[-1].shape)

print("attentions is None:", outputs.attentions is None)
if outputs.attentions is not None:
    print("num attention layers:", len(outputs.attentions))
    print("attention[0]:", outputs.attentions[0].shape)
    print("attention[-1]:", outputs.attentions[-1].shape)

    row_sums = outputs.attentions[-1][0, 0].sum(dim=-1)
    print("last layer/head 0 row sums, first 10:", row_sums[:10])
PY