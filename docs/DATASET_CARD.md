# Dataset card — MedQuAD EN / RU / KK

File: `data/final/qa_en_ru_kk.parquet` · loader: `medqa/data.py` (`load_dataset`) · statistics below are measured
from the file with that loader.

## Source and languages

| | |
|---|---|
| Source | **MedQuAD** (Ben Abacha & Demner-Fushman, 2019): consumer health questions and answers collected from 9 NIH websites |
| English | original MedQuAD text |
| Russian, Kazakh | **machine translations** of the English question and answer (translation errors are noise in references) |
| Domain | consumer health: diseases, genetic conditions, symptoms, treatment, inheritance |
| Task | open-ended (generative) question answering: input = question, target = reference answer |

## Size and cleaning

| | rows |
|---|---|
| rows in file | 14 453 |
| complete rows (question + answer non-empty in all 3 languages) — used everywhere | **14 368** |
| examples for training (row × 3 languages) | 43 104 |
| distinct diseases / topics (`focus`) | 5 114 |
| duplicate English questions (same wording, different source/answer) | 594 |
| duplicate answers | 0 |

Cleaning: `fillna("")`, strip whitespace, drop incomplete rows. No stemming / stop-word removal — the models are
subword-tokenised LLMs (Qwen3 BPE tokenizer) and bge-m3 (XLM-R SentencePiece), which take raw text.

## Composition

Source websites (`category`): GARD 5 321 · GHR 4 743 · NINDS 1 003 · MedlinePlus 940 · NIDDK 801 · CancerGov 633 ·
NHLBI 475 · NIHSeniorHealth 288 · CDC 249.

Question type (the `qtype` column is empty — "Unknown" — so it is derived from the English question template by
`medqa.data.question_type`): information 3 734 · symptoms 2 673 · treatment 2 317 · frequency 1 105 ·
genetic changes 1 079 · inheritance 950 · causes 653 · exams and tests 613 · outlook 348 · research 340 ·
susceptibility 245 · prevention 187 · stages 76 · complications 45 · support 3.

Length in words (median / mean / 95th percentile):

| | question | answer |
|---|---|---|
| EN | 7 / 7 / 11 | 141 / 204 / 518 |
| RU | 5 / 6 / 10 | 122 / 175 / 451 |
| KK | 6 / 6 / 10 | 113 / 161 / 414 |

## Splits

The split is made **by row before expanding to languages**, so the three translations of one question are always on
the same side.

| split | rows | how | used for |
|---|---|---|---|
| train (medbot_old) | 12 931 | `permutation(seed=42)`, last 90 % | QLoRA stage 1 |
| held-out (`first_adapter_unseen`) | 1 437 | first 10 % of the same permutation | validation loss of stage 1 (no checkpoint selection, no tuning) |
| **test (harness `medquad_heldout_*`)** | **1 344** | held-out rows minus 93 whose question text also appears in train | headline evaluation: base vs medbot_old |

Leakage checks (normalised text: lower-case, punctuation removed; `harness/tasks/medquad/utils.py:heldout_rows`):

- answers shared between train and held-out: **0**;
- held-out questions whose exact wording appears in train: 93 → removed from the test set;
- test rows whose disease (`focus`) appears in train with a *different* question type: 1 095 of 1 344 (81 %).
  This is topic overlap, not answer overlap; results are also reported for the 249 rows with unseen diseases.

`medbot_full` continued training on the 1 437 held-out rows, so **it has no clean test set** — its MedQuAD scores
measure memorisation only. The RAG index and Hopfield memory contain all 14 368 rows by design (they are the
knowledge base); their honest evaluation hides the row's own answer (`raghidden`, `hopfield_hidden`).

## External benchmarks (evaluation only, never trained on)

| benchmark | language | items | format | license |
|---|---|---|---|---|
| MedQA-USMLE 4-options, MedMCQA, PubMedQA, MMLU medical subsets | EN | 200 per task (`--limit`) | multiple choice | per dataset |
| MMedBench (Russian test) | RU | 256 | yes / no | CC BY-NC 4.0 |
| KazMMLU Medicine | RU | 297 | 4–5 options | CC BY-NC 4.0 |
| KazMMLU Biology (high school) | RU / KK | 1 065 / 1 022 | 4–5 options | CC BY-NC 4.0 |

Downloaded by `scripts/get_benchmarks.py`; not stored in git (non-commercial license).

## Known limitations

- RU / KK references are machine-translated: a correct answer can score lower because the reference itself is imperfect.
- Answers are long (median 141 words) and come from several NIH sites with different styles, so n-gram overlap
  metrics (ROUGE-L, chrF) are low even for good answers; semantic similarity (bge-m3) is reported alongside.
- Content is from US sources (e.g. emergency number 911); the model output replaces it with 103 / 112 for RU / KK.
