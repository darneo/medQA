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

All automatic evaluation runs through **lm-evaluation-harness 0.4.13** (EleutherAI) with project tasks in
`harness/tasks/` and the runner `scripts/run_harness.py`; the summary is `results/harness/SUMMARY.md`
(built by `scripts/harness_report.py`), every answer is in `results/harness/<model>/*/samples_*.jsonl`.

### 5.1 Protocol

| | |
|---|---|
| Evaluation set (MedQuAD) | 150 held-out test rows × 3 languages = 450 questions (first 150 of the seed-42 shuffled test rows) |
| Prompt | the same system prompt medbot was trained with, Qwen3 chat template, thinking disabled |
| Decoding | greedy, max 512 new tokens, repetition penalty 1.1 (`harness/tasks/medquad/_medquad_base.yaml`) |
| Baseline | the same Qwen3-4B 4-bit model without the adapter (`base`) |
| Metrics | chrF (sacrebleu corpus chrF, char 6-grams, β = 2) — fairer than word overlap for Russian / Kazakh morphology; ROUGE-L F1 (word LCS, mean over questions); semantic similarity (cosine of bge-m3 embeddings); right language; repetition (share of repeated 4-grams); advises doctor (keyword check) |
| Uncertainty | 95 % bootstrap CI of the per-question difference to base (2 000 resamples) |

The similarity metrics compare to the dataset reference; they do **not** measure medical correctness — that is
covered by manual annotation (§6).

### 5.2 Held-out test: baseline vs fine-tuned (n = 150 per language)

| lang | model | chrF | ROUGE-L | semantic | repetition | advises doctor | words |
|---|---|---|---|---|---|---|---|
| EN | base | 34.3 | 0.149 | 0.800 | 1 % | 17 % | 241 |
| EN | medbot_old | **46.1** | **0.362** | **0.871** | 2 % | 8 % | 150 |
| RU | base | 28.2 | 0.095 | 0.780 | 0 % | 49 % | 155 |
| RU | medbot_old | **38.7** | **0.283** | **0.841** | 3 % | 29 % | 116 |
| KK | base | 17.9 | 0.064 | 0.706 | 29 % | 19 % | 96 |
| KK | medbot_old | **30.6** | **0.219** | **0.813** | 9 % | 27 % | 81 |

Difference medbot_old − base on the same questions (mean, 95 % CI):

| lang | Δ chrF | Δ ROUGE-L | Δ semantic |
|---|---|---|---|
| EN | +11.7 [+8.4, +15.0] | +0.214 [+0.174, +0.256] | +0.071 [+0.056, +0.086] |
| RU | +10.1 [+6.9, +13.7] | +0.188 [+0.149, +0.230] | +0.061 [+0.044, +0.077] |
| KK | +14.2 [+11.8, +16.8] | +0.155 [+0.126, +0.185] | +0.107 [+0.089, +0.124] |
| all | +12.0 [+10.2, +13.9] | +0.186 [+0.163, +0.207] | +0.079 [+0.070, +0.089] |

All intervals exclude zero. On the 90 test questions about diseases that never appeared in training,
medbot_old still improves (sentence chrF 30.1 → 46.8, semantic 0.784 → 0.875), so the gain is not only memorised
disease facts.

### 5.3 General knowledge: multiple-choice benchmarks (never trained on), accuracy %

| benchmark | n | base | medbot_old | medbot_new |
|---|---|---|---|---|
| MMedBench-RU (yes / no) | 256 | 77.3 ± 2.6 | 75.4 ± 2.7 | 74.2 ± 2.7 |
| KazMMLU Medicine (RU) | 297 | 43.4 ± 2.9 | 40.7 ± 2.9 | 40.7 ± 2.9 |
| KazMMLU Biology (RU) | 1 065 | 75.8 ± 1.3 | 73.0 ± 1.4 | 73.6 ± 1.4 |
| KazMMLU Biology (KK) | 1 022 | 52.8 ± 1.6 | 48.8 ± 1.6 | 48.9 ± 1.6 |
| MedQA-USMLE 4 options (EN) | 200 | 57.5 ± 3.5 | 57.5 ± 3.5 | 57.0 ± 3.5 |
| MedMCQA (EN) | 200 | 54.0 ± 3.5 | 48.5 ± 3.5 | 47.0 ± 3.5 |
| MMLU anatomy | 135 | 63.0 ± 4.2 | 65.2 ± 4.1 | 63.7 ± 4.2 |
| MMLU clinical knowledge | 200 | 73.5 ± 3.1 | 71.5 ± 3.2 | 70.5 ± 3.2 |
| MMLU college medicine | 173 | 71.1 ± 3.5 | 69.9 ± 3.5 | 70.5 ± 3.5 |
| MMLU medical genetics | 100 | 73.0 ± 4.5 | 73.0 ± 4.5 | 75.0 ± 4.4 |
| MMLU professional medicine | 200 | 71.0 ± 3.2 | 66.0 ± 3.4 | 66.0 ± 3.4 |

Zero-shot, log-likelihood of the option letter, no chat template; English tasks limited to the first 200 items.
PubMedQA was planned but could not be loaded (its loader is a dataset script, no longer supported by `datasets` 4.x)
— **not measured**. Fine-tuning leaves multiple-choice knowledge roughly unchanged: same on MedQA, +1…+2 on anatomy
and genetics, −2…−7 on MedMCQA, professional medicine and the RU / KK sets; most differences are within 1–2 standard
errors. A mild loss of multiple-choice ability is the price of learning the open-answer style.

### 5.4 Final model (medbot_new)

medbot_new is the model trained on **all** rows (teacher-approved). On the same 450 questions (which it saw in
training) it scores only slightly above medbot_old (which did not):

| lang | base | medbot_old | medbot_new |
|---|---|---|---|
| EN chrF / semantic | 34.3 / 0.800 | 46.1 / 0.871 | 46.9 / 0.874 |
| RU chrF / semantic | 28.2 / 0.780 | 38.7 / 0.841 | 39.8 / 0.845 |
| KK chrF / semantic | 17.9 / 0.706 | 30.6 / 0.813 | 30.8 / 0.824 |

The small gap (+0.2…+1.1 chrF) after seeing the answers shows that one epoch of LoRA learned the style, language
and format, not the reference texts — no sign of memorisation.

### 5.5 Robustness to rewording (paraphrased questions)

20 test questions × 3 languages rewritten by hand (`data/paraphrased_questions.json`; e.g. *"What is the best way to
treat Guillain-Barre syndrome?"*, *"I noticed blood in my urine. What does that mean?"*). Same reference answer.

| model | chrF original → paraphrase | semantic original → paraphrase |
|---|---|---|
| base | 31.1 → 30.3 | 0.778 → 0.763 |
| medbot_old | 38.9 → 36.3 | 0.839 → 0.811 |
| medbot_new | 40.2 → 36.2 | 0.847 → 0.808 |

The fine-tuned models lose ~3–4 chrF when the wording changes (part of their advantage is matching MedQuAD phrasing),
but stay clearly above base on paraphrased questions. n = 60 per model, indicative only.

### 5.6 Attempts to improve the scores

**Blocking repetition** (`no_repeat_ngram_size=4`, same 50 test questions per language):

| model | lang | chrF default → no-repeat | ROUGE-L | repetition |
|---|---|---|---|---|
| base | KK | 22.7 → 24.1 | 0.066 → 0.046 | 29 % → 0 % |
| medbot_new | EN | 48.9 → 47.8 | 0.379 → 0.351 | 2 % → 0 % |
| medbot_new | RU | 42.5 → 38.9 | 0.299 → 0.236 | 4 % → 0 % |
| medbot_new | KK | 36.1 → 31.6 | 0.222 → 0.120 | 9 % → 0 % |

Loops disappear, but quality drops: legitimate repeats (disease names, fixed phrases, Kazakh word forms) are blocked
too. Default decoding is kept.

**RAG** (3 most similar QA pairs in the system prompt, the question's own answer removed; 50 questions per language):

| model | lang | chrF no RAG → RAG | ROUGE-L |
|---|---|---|---|
| base | EN | 36.3 → 37.3 | 0.152 → 0.189 |
| base | RU | 32.3 → 32.7 | 0.100 → 0.128 |
| base | KK | 22.7 → 27.3 | 0.066 → 0.098 |
| medbot_new | EN | 48.9 → 43.0 | 0.379 → 0.312 |
| medbot_new | RU | 42.5 → 38.2 | 0.299 → 0.249 |
| medbot_new | KK | 36.1 → 34.0 | 0.222 → 0.210 |

Retrieval helps the base model (most in Kazakh) but hurts medbot, which was never trained with retrieved context and
gets distracted by neighbouring answers. Fine-tuning alone (36–49 chrF) beats base + RAG (27–37). Training medbot
with RAG-style prompts is a next step.

**Decoding settings chosen on a dev split.** Held-out rows 150–249 (100 per language, disjoint from the 150 test
rows) were used to choose generation settings for medbot (`scripts/decoding_sweep.py`, medbot_old, 50 per language;
selection rule fixed in advance: highest mean sentence chrF). Results (`results/harness/decoding_sweep.md`):

| setting | chrF (EN / RU / KK) | semantic | loops > 20 % |
|---|---|---|---|
| **max 768 new tokens** | **42.5** (48.2 / 40.7 / 38.8) | 0.846 | 9 % |
| default: repetition penalty 1.1, max 512 | 41.0 (47.3 / 39.7 / 36.1) | 0.846 | 5 % |
| repetition penalty 1.2 | 40.2 (46.0 / 41.0 / 33.7) | 0.811 | 1 % |
| no_repeat_ngram_size 8 | 39.3 (47.1 / 38.5 / 32.2) | 0.828 | 0 % |
| repetition penalty 1.05 | 39.2 (44.4 / 38.6 / 34.7) | 0.842 | 17 % |
| repetition penalty 1.0 | 37.0 (41.4 / 38.0 / 31.6) | 0.836 | 35 % |
| repetition penalty 1.3 | 32.9 (37.9 / 34.1 / 26.9) | 0.737 | 1 % |

Penalty 1.1 is the best trade-off; stronger penalties or n-gram blocking remove loops but damage the answers. The
only gain is a larger length budget. The winner was then run **once** on the test set (medbot_new, 450 questions):

| lang | chrF max 512 → 768 | Δ (95 % CI) | ROUGE-L | loops > 20 % |
|---|---|---|---|---|
| EN | 48.1 → 48.7 | +0.6 [+0.1, +1.2] | 0.384 → 0.389 | 1 % → 2 % |
| RU | 42.1 → 43.2 | +1.1 [+0.4, +1.8] | 0.291 → 0.298 | 7 % → 11 % |
| KK | 35.9 → 38.6 | +2.7 [+1.6, +3.9] | 0.223 → 0.240 | 17 % → 28 % |
| all | 42.0 → 43.5 | +1.4 [+1.0, +2.0] | 0.299 → 0.309 | 8 % → 14 % |

A small but significant gain, largest in Kazakh, because long answers are no longer cut off; the cost is that an
answer that does start looping now loops longer. Recommended setting for the final system: max 768 new tokens.

The baseline does not profit from the longer budget (base chrF 29.4 → 29.8), so with the same 768-token setting the
gap grows: **medbot_new − base = +13.7 chrF [+11.7, +15.7]** over all 450 test questions (EN +13.0, RU +11.6,
KK +16.6), ROUGE-L 0.099 → 0.309; answers with loops: base 22 % (KK 67 %), medbot_new 14 % (KK 28 %).

**Hybrid router (medbot + Hopfield-QA)** — answer from stored, verified text when the question is already in the
knowledge base (`scripts/hybrid_router.py`, `results/hybrid/`). Threshold chosen on dev so that ≤ 5 % of new
questions are routed to memory. It does not work well: MedQuAD has 5–10 near-identical questions per disease
("What is X?", "What are the symptoms of X?"), so question similarity cannot tell "the same question" from "the
same disease, another aspect" (median confidence 0.85 for new questions vs 0.87 for paraphrased known ones). Only
3–6 % of questions were routed; gain on paraphrases EN chrF 38.1 → 41.6, none on new questions. **Negative result.**

**By question type** (test, chrF base → medbot_new): symptoms 26.1 → 55.1, treatment 27.3 → 53.6, frequency
24.5 → 39.0, inheritance 31.6 → 42.8, genetic changes 34.0 → 38.2, "what is X" 32.3 → 34.2, causes 31.3 → 30.1.
Fine-tuning helps most where MedQuAD answers follow a fixed template and least for free-form definitions and causes.

### 5.7 Larger ready-made models

Ready-made models without fine-tuning, 2-bit GGUF via llama.cpp (`llama-server`, OpenAI-compatible API, same
prompt and metrics through lm-eval `local-chat-completions`), on the same 50 test questions per language:

| model | params | chrF EN / RU / KK | ROUGE-L EN / RU / KK | semantic EN / RU / KK | repetition KK | advises doctor RU |
|---|---|---|---|---|---|---|
| base (Qwen3-4B) | 4 B | 36.3 / 32.3 / 22.7 | 0.152 / 0.100 / 0.066 | 0.815 / 0.793 / 0.721 | 29 % | 49 % |
| Saluki 27B | 27 B | 36.1 / 33.7 / 24.4 | 0.151 / 0.100 / 0.074 | 0.819 / 0.805 / 0.760 | 24 % | 58 % |
| LingSante-3.1-air (MoE) | 124 B (5.5 B active) | 33.4 / 31.5 / 25.2 | 0.161 / 0.097 / 0.077 | 0.835 / 0.795 / 0.751 | 5 % | 84 % |
| **medbot_new** (QLoRA 4B) | 4 B | **48.9 / 42.5 / 36.1** | **0.379 / 0.299 / 0.222** | **0.871 / 0.857 / 0.831** | 9 % | 31 % |

The fine-tuned 4B model is far closer to the references than models 7–30× larger: domain and language adaptation
matter more than size for this task. The large models are not "wrong" — LingSante has the highest semantic similarity
among the non-fine-tuned models, the fewest loops and advises a doctor most often — but they answer in their own
style and length, which the overlap metrics penalise (see §6). Saluki answers often hit the 512-token limit.
Saluki / LingSante numbers use the first 50 test questions per language (generation is 10–30× slower on 8 GB VRAM).

## 6. Error analysis

Details: [`docs/ERROR_ANALYSIS_EXAMPLES.md`](ERROR_ANALYSIS_EXAMPLES.md) (examples selected by rule) and
[`docs/ANNOTATION.md`](ANNOTATION.md) (manual rubric).

- medbot_old is closer in meaning to the reference than base on 78 % of test questions (EN 79 %, RU 72 %, KK 81 %).
- Degenerate answers (repetition loop, cut at the limit): base 85 / 450, medbot_old 44 / 450 — mostly Kazakh
  (79 → 27).
- Manual annotation of 30 random test answers per system: base — 17 % correct, 10 % partially, **47 % hallucination**,
  7 % off-topic, 20 % degenerate; medbot_old — 30 % correct, 37 % partially, **33 % hallucination**, 0 % degenerate.
  Single LLM annotator, not a physician.
- Typical failures: base invents genes and disease definitions; medbot_old copies MedQuAD templates (symptom tables
  with "90 %" frequencies) and fills them with wrong items when it does not know the disease; both fail on misleading
  names (mycosis fungoides is a lymphoma; Wagner syndrome is an eye disease).
- Wording vs meaning: on the 24 questions of `scripts/test_all_models.py`, LingSante has the lowest chrF (21.8) but a
  semantic similarity close to medbot (0.767 vs 0.796) — it answers in its own words, which overlap metrics punish
  ([`docs/figures/phrasing_vs_meaning.png`](figures/phrasing_vs_meaning.png)).

## 7. Limitations

- References for Russian and Kazakh are machine translations.
- Similarity metrics are not correctness; the manual annotation is small (60 answers) and done by an LLM, not a doctor.
- medbot_new was trained on all rows, so its MedQuAD scores measure fit, not generalisation; generalisation of the
  method is shown by medbot_old on held-out rows and by paraphrased questions.
- After fine-tuning the model mentions a doctor less often in EN / RU (it imitates reference texts).
- Not a medical device: answers must not be used for diagnosis or treatment.
