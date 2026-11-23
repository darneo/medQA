---
base_model: unsloth/Qwen3-4B-unsloth-bnb-4bit
library_name: peft
license: apache-2.0
language:
- en
- ru
- kk
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/Qwen3-4B-unsloth-bnb-4bit
- lora
- qlora
- sft
- unsloth
- medical
- kazakh
---

# medbot — LoRA adapter for Qwen3-4B, medical questions (en / ru / kk)

A LoRA adapter (33M parameters, 0.8% of the model, ~130 MB) fine-tuned with **QLoRA** on
[MedQuAD](https://github.com/abachaa/MedQuAD) (NIH medical Q&A; Russian and Kazakh are machine translations).
The base model is downloaded automatically. The ready merged model is in the root of
`dARYNel/qwen3-4b-medQA-kklang`.

> **Warning.** This is an experimental student model. Its medical accuracy has not been reviewed by a doctor.
> **Do not use it for diagnosis, treatment or emergencies.** In a life-threatening situation call your local
> emergency number (103 / 112 in Kazakhstan and Russia).

## Training

QLoRA (Unsloth): r=16, alpha=16, dropout 0, layers q/k/v/o + gate/up/down; 12,931 × 3 = 38,793 dialogues;
1 epoch (4,850 steps), lr 2e-4 cosine, batch 2 × 4, max length 1024, AdamW 8-bit; RTX 5060 8 GB.
Loss: train ~0.58, validation 0.627.

### How it was trained

1. **Data.** Each MedQuAD row is one question and answer in three languages; rows missing a translation were dropped
   (14,368 remain).
2. **Split.** 10% of the rows were held out for validation **before** expanding to languages, so translations of the
   same question never appear in both training and validation.
3. **Dialogues.** Each row becomes three dialogues (en, ru, kk): system prompt "you are a careful medical assistant…
   advise seeing a doctor" → question → reference answer. Qwen3 chat template, without thinking.
4. **QLoRA.** The base model is frozen in 4 bits; only the LoRA matrices are trained — the model learns to predict
   the reference answer token by token (SFT).

## Limitations

For urgent questions in Kazakh the model does not advise calling an ambulance; it may invent facts.

## Usage

```python
from unsloth import FastModel

model, tok = FastModel.from_pretrained("dARYNel/qwen3-4b-medQA-kklang", subfolder="lora_adapter",
                                       max_seq_length=1024, load_in_4bit=True)
FastModel.for_inference(model)
messages = [{"role": "system", "content": "You are a careful medical assistant. Answer in English with general "
                                          "information. Encourage consulting a qualified clinician for personal "
                                          "diagnosis or treatment."},
            {"role": "user", "content": "What is asthma?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=384, repetition_penalty=1.1)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

Without Unsloth: `peft.AutoPeftModelForCausalLM.from_pretrained("dARYNel/qwen3-4b-medQA-kklang", subfolder="lora_adapter")`.
System prompts for ru and kk are listed in the main model card.

## License

Apache 2.0, same as Qwen3. MedQuAD data: CC BY 4.0 (Ben Abacha & Demner-Fushman, 2019).
