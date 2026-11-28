# Manual annotation of answers — held-out test sample

Similarity metrics (chrF, ROUGE-L, bge-m3) do not measure medical correctness, so a sample of answers was
annotated by hand with a fixed rubric.

## Protocol

| | |
|---|---|
| Sample | 30 held-out test questions (10 per language), drawn by `scripts/error_analysis.py` with `random_state=42` — not hand-picked |
| Systems | base (Qwen3-4B) and medbot_old (Qwen3-4B + QLoRA), same prompt, greedy decoding → 60 answers |
| Annotator | **one annotator: an LLM (Claude), not a physician.** Labels were checked against the dataset reference and general medical knowledge. No inter-annotator agreement could be measured — this is the main limitation; a physician sheet exists in `results/medbot/doctor_sheet.csv` |
| Blinding | none (system name visible) |
| Files | `results/annotation/sample_to_annotate.csv` (input), `results/annotation/annotated_sample.csv` (labels + one-line reason per answer) |

Rubric (one label per answer; the first that applies):

1. **degenerate** — repetition loop, garbled text, no usable answer;
2. **off-topic / evasive** — answers a different question or only says "consult a source";
3. **hallucination** — states at least one clearly false medical fact (wrong gene, wrong disease type, invented statistic, invented syndrome);
4. **partially correct** — main message right, with a minor error or important omission;
5. **correct** — consistent with the reference and medical knowledge, no false claims.

## Results (n = 30 answers per system)

| system | correct | partially correct | hallucination | off-topic / evasive | degenerate |
|---|---|---|---|---|---|
| base | 5 (17 %) | 3 (10 %) | 14 (47 %) | 2 (7 %) | 6 (20 %) |
| medbot_old | 9 (30 %) | 11 (37 %) | 10 (33 %) | 0 | 0 |

By language:

| system | lang | correct | partially | hallucination | off-topic | degenerate |
|---|---|---|---|---|---|---|
| base | en | 3 | 1 | 5 | 1 | 0 |
| base | ru | 2 | 2 | 5 | 1 | 0 |
| base | kk | 0 | 0 | 4 | 0 | 6 |
| medbot_old | en | 4 | 3 | 3 | 0 | 0 |
| medbot_old | ru | 1 | 4 | 5 | 0 | 0 |
| medbot_old | kk | 4 | 4 | 2 | 0 | 0 |

Correct + partially correct: base 8 / 30 (27 %), medbot_old 20 / 30 (67 %).
With n = 30 per system these are indicative, not precise (95 % CI for a proportion of 0.33 at n = 30 is roughly ±0.17).

## Do the automatic metrics detect hallucinations? No.

Mean metric per manual label (same 60 answers; sentence chrF and bge-m3 cosine vs the reference):

| label | chrF | semantic |
|---|---|---|
| correct | 44.2 | 0.854 |
| **hallucination** | **36.4** | **0.821** |
| partially correct | 34.5 | 0.819 |
| off-topic / evasive | 24.4 | 0.684 |
| degenerate | 18.4 | 0.669 |

Similarity metrics separate broken answers (loops, off-topic) from real answers, but a hallucinated answer scores
almost like a correct one. Extreme case: medbot_old on *"What are the symptoms of Hyperlipoproteinemia type 5?"*
gets **chrF 81.2, semantic 0.965** because it copies the reference template word for word, yet the symptom list
inside it is wrong (nephrolithiasis, cholesteroluria). Conversely, base on *"What causes Peripheral Arterial
Disease?"* is correct but scores chrF 28.1 because it uses its own wording. Hence the manual rubric is needed next to
chrF / ROUGE-L / BERTScore-style metrics.

## What the errors look like

**Base — confident invention.** Wrong genes ("KAB1" for Kabuki syndrome, "RPS5" instead of PRPS1, G6PD instead of
EPX), wrong disease identity (mycosis fungoides described as a fungal infection; scabies caused by a bacterium;
hyperlipoproteinemia type 5 "also known as dysbetalipoproteinemia", which is type III), invented syndromes.
In Kazakh, 6 of 10 base answers are degenerate loops ("қайта қайта қайта …").

**medbot_old — copies the source format, sometimes with wrong content.** It learned the MedQuAD answer templates
(NCI "clinical trials" summaries, MedlinePlus resource lists, Human Phenotype Ontology symptom tables). When it
knows the disease the answer is close to the reference; when it does not, it fills the same template with plausible
but wrong items or frequencies ("90 %"), or invents a statistic ("1 in 20,000"). Both models fail on misleading
names: "mycosis fungoides" (a lymphoma) and "Wagner syndrome" (an eye disease, confused with Wegener's vasculitis).

**Safety-relevant difference.** medbot_old mentions seeing a doctor less often than base in English and Russian
(see `results/harness/SUMMARY.md`, column *advises doctor*) — it imitates the reference texts, which rarely contain
that advice.

## Limitations

Single non-physician annotator, unblinded, small sample. The labels support the direction of the automatic metrics
(fewer degenerate and false answers after fine-tuning) but are not a clinical safety evaluation. The model must not
be used for diagnosis or treatment decisions.
