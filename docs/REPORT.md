# medQA — trilingual medical question answering (EN / RU / KK)

Project report for the NLP project defense. Every number in this report comes from a file in the repository; the
file is named next to it. Experiments that were not run are marked **not measured**.

## 1. Problem and task

**Task:** open-ended (generative) question answering in the consumer-health domain, in English, Russian and Kazakh.

| | |
|---|---|
| Input | a health question in one of three languages ("What are the symptoms of asthma?", "Диабет дегеніміз не?") |
| Output | a short, careful answer in the **same language**, with general information and advice to see a doctor |
| Target (training / evaluation) | the reference answer from MedQuAD (EN original, RU / KK machine-translated) |
| Intended users | people looking for general health information in Kazakh or Russian, where few medical resources exist |
| Success criteria | answer in the right language; closer to the reference than the base model on unseen questions (chrF, ROUGE-L, semantic similarity); no loss of general medical knowledge (multiple-choice benchmarks); advises a doctor; local emergency numbers (103 / 112) |

The system is **not** a diagnostic tool: the metrics measure similarity to a reference, not clinical correctness.

## 2. Data

Full details: [`docs/DATASET_CARD.md`](DATASET_CARD.md).

- MedQuAD (NIH), 14 368 complete question–answer rows × 3 languages = 43 104 examples; 5 114 diseases / topics.
- Split by row (before expanding to languages): train 12 931 rows · held-out 1 437 rows (seed 42).
- Test set for the headline result: 1 344 held-out rows after removing 93 rows whose question also appears in train;
  0 answers are shared between train and test.
- External benchmarks for general knowledge: MedQA-USMLE, MedMCQA, PubMedQA, MMLU medical (EN); MMedBench-RU,
  KazMMLU Medicine (RU), KazMMLU Biology (RU, KK).

**Preprocessing and tokenization.** Text is only cleaned (empty values removed, whitespace stripped); no stemming or
stop-word removal, because the model works on subwords. Each example is rendered with the Qwen3 chat template:

```
<|im_start|>system  <careful-assistant prompt in the question's language>  <|im_end|>
<|im_start|>user    <question>                                              <|im_end|>
<|im_start|>assistant <think>\n\n</think>\n\n <reference answer>            <|im_end|>
```

The Qwen3 tokenizer (byte-level BPE, 151 k vocabulary) turns it into token IDs; sequences are truncated to
1 024 tokens; batches are padded with an attention mask. The loss is computed **only on the assistant answer**
(`train_on_responses_only`), so the model is not trained to repeat the prompt or the question.

## 3. Models

| system | type | what is trained |
|---|---|---|
| **base** (baseline) | Qwen3-4B, 4-bit (`unsloth/Qwen3-4B-unsloth-bnb-4bit`), same prompt | nothing |
| **medbot_old** (proposed) | Qwen3-4B + LoRA adapter (QLoRA) | adapter, on 12 931 train rows × 3 languages |
| medbot_full | medbot_old adapter trained further on the 1 437 held-out rows | adapter (no clean test set left) |
| + RAG | any of the above + 3 most similar QA pairs from the dataset in the system prompt (bge-m3 embeddings, cosine) | nothing (retrieval) |
| Saluki 27B, LingSante 124B-MoE | larger ready-made models (GGUF, 2-bit) via llama.cpp, with / without RAG | nothing |
| Hopfield-QA | modern Hopfield network over bge-m3 embeddings: retrieves a stored answer instead of generating | projections W_Q, W_K and β |

**Why QLoRA on a 4B model.** The only GPU is an RTX 5060 with 8 GB VRAM. A 27B model needs 24 GB+ for fine-tuning.
Qwen3-4B in 4 bits takes ~3 GB, leaving room for activations; LoRA trains 33 M parameters (0.8 % of the model,
adapter 132 MB) while the 4-bit base stays frozen.

**Architecture.** Qwen3-4B is a decoder-only transformer (36 layers, grouped-query attention, SwiGLU MLP, RoPE).
LoRA adds low-rank matrices B·A (rank r = 16, α = 16, no dropout) to every attention projection (q, k, v, o) and
MLP projection (gate, up, down); the output is `W x + (α/r)·B A x`. The prediction head is the frozen LM head
(softmax over the vocabulary); training minimises next-token cross-entropy on answer tokens.

## 4. Training

| | stage 1 — medbot_old | stage 2 — medbot_full |
|---|---|---|
| data | 12 931 rows × 3 = 38 793 examples | 1 437 rows × 3 = 4 311 examples |
| epochs / steps | 1 / 4 850 | 1 / 539 |
| optimizer | AdamW 8-bit | AdamW 8-bit |
| learning rate | 2e-4, 3 % warmup, cosine decay | 1e-4, 3 % warmup, decay |
| batch | 2 × 4 gradient accumulation = 8 | 8 |
| max sequence length | 1 024 | 1 024 |
| regularization | none beyond 1 epoch (LoRA dropout 0) | — |
| seed | 42 (Python, NumPy, torch, data split) | 42 |
| hardware / time | RTX 5060 8 GB, Unsloth, PyTorch 2.11 + CUDA 12.8 | 48 min (`models/medbot_train_full/train_report.json`) |
| train loss | 2.17 → 0.90 (step ~300) → 0.58 (end) | 0.722 (mean) |
| validation loss | **0.627** on the held-out 1 437 rows (`results/medbot/validation_metrics.json`) | — |

Figures: [`docs/figures/medbot_old_loss.png`](figures/medbot_old_loss.png) (only the recorded checkpoints: the
step-by-step log of stage 1 was not saved), [`docs/figures/medbot_full_training.png`](figures/medbot_full_training.png),
[`docs/figures/hopfield_training.png`](figures/hopfield_training.png).

**Over / underfitting.** Validation loss after stage 1 (0.627) is close to the final train loss (~0.58): no strong
overfitting after one epoch. No checkpoint selection was done on the held-out rows (single epoch, final adapter),
and no hyperparameter was tuned on them, so they remain a valid test set for medbot_old.

## 5. Evaluation

_Filled from `results/harness/SUMMARY.md` when all runs finish._
