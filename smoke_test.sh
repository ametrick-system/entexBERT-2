python - <<'PY'
import torch
import transformers

from entexbert2.finetune_entexbert2 import entexBERT2ForSequencePrediction

model_name = "zhihan1996/DNABERT-2-117M"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    model_max_length=512,
)

model = entexBERT2ForSequencePrediction(
    model_name_or_path=model_name,
    main_task="classification",
    main_num_labels=2,
)

model.eval()

# Force config flags too, in case the custom DNABERT-2 model reads config
model.backbone.config.output_attentions = True
model.backbone.config.output_hidden_states = True

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
    backbone_outputs = model.backbone(
        **inputs,
        return_dict=False,
        output_attentions=True,
        output_hidden_states=True,
    )

print("Backbone output type:", type(backbone_outputs))
print("Tuple length:", len(backbone_outputs))

for i, item in enumerate(backbone_outputs):
    print(f"\nItem {i}: type={type(item)}")

    if torch.is_tensor(item):
        print("  tensor shape:", tuple(item.shape))

    elif isinstance(item, (tuple, list)):
        print("  nested length:", len(item))

        if len(item) > 0:
            first = item[0]
            print("  first nested item type:", type(first))

            if torch.is_tensor(first):
                print("  first nested item shape:", tuple(first.shape))

            if len(item) > 1 and torch.is_tensor(item[-1]):
                print("  last nested item shape:", tuple(item[-1].shape))

    else:
        print("  value:", item)
PY