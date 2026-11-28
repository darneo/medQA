"""Hybrid medbot + Hopfield-QA router -> results/hybrid/hybrid_router.{json,md}.

    .venv\\Scripts\\python.exe scripts\\hybrid_router.py

Idea: if Hopfield-QA is confident that the question is already in the knowledge base (MedQuAD), return the stored,
verified NIH answer; otherwise let medbot_new generate. Stored answers cannot hallucinate.

Protocol (no tuning on the test set):
  - confidence = cosine between the question and the retrieved stored question (bge-m3);
  - threshold τ is chosen on the DEV rows (held-out rows 150-249) with the row's own answer HIDDEN, i.e. for
    questions that are NOT in the knowledge base: τ = 95th percentile of Hopfield confidence there, so at most ~5 %
    of genuinely new questions are wrongly answered from memory;
  - evaluated on the TEST rows (first 150 held-out rows per language) in two scenarios, using medbot_new answers
    already produced by the harness:
      new question      — own answer hidden from memory (question not in the base);
      paraphrased known — hand-paraphrased questions (data/paraphrased_questions.json), own answer in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from harness_report import semantic_scores, set_dir  # noqa: E402
from medqa.hopfield import HopfieldQA, load_cache  # noqa: E402

OUT = ROOT / "results" / "hybrid"
LANGS = ("en", "ru", "kk")


def heldout_order() -> list[int]:
    sys.path.insert(0, str(ROOT / "harness" / "tasks" / "medquad"))
    import utils  # noqa: E402
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"test": str(ROOT / "data/final/qa_en_ru_kk.parquet")})["test"]
    return list(utils._process(ds, "en", heldout=True)["row"])


def medbot_answers(pattern: str) -> dict[tuple[str, int], dict]:
    sys.path.insert(0, str(ROOT / "harness" / "tasks" / "medquad"))
    import utils  # noqa: E402

    out = {}
    for f in sorted((ROOT / "results/harness/medbot_full").glob(f"*/samples_{pattern}_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            s = json.loads(line)
            d = s["doc"]
            out[(d["lang"], d["row"])] = {"answer": utils.clean(s["resps"][0][0], d["lang"]), "ref": d["ref"], "q": d["q"]}
    return out


def main() -> None:
    set_dir(ROOT / "results" / "harness")
    OUT.mkdir(parents=True, exist_ok=True)
    order = heldout_order()
    dev_rows, test_rows = order[150:250], order[:150]

    cache = load_cache()
    p = cache["passages"]
    pos = {(r, l): i for i, (r, l) in enumerate(zip(p["row"], p["lang"]))}
    qa = HopfieldQA(device="cpu")  # GPU is busy with the decoding sweep

    def top1(vectors, langs, hide=None):
        """Top-1 of Hopfield; confidence = cosine(query, retrieved question). The Hopfield softmax saturates at 1.0
        even for questions that are not in memory, so it cannot be thresholded."""
        hits = [h[0] for h in qa.search_vectors(vectors, langs, k=1, hide=hide)]
        for v, h, lang in zip(vectors, hits, langs):
            h["score"] = float(np.asarray(v, dtype=np.float32) @ cache["questions"][pos[(h["row"], lang)]].astype(np.float32))
        return hits

    # --- threshold on DEV, own answer hidden (questions not in the base)
    keys = [(r, l) for l in LANGS for r in dev_rows]
    idx = np.array([pos[k] for k in keys])
    dev_hits = top1(cache["questions"][idx], [l for _, l in keys], hide=idx)
    dev_conf = np.array([h["score"] for h in dev_hits])
    tau = float(np.percentile(dev_conf, 95))
    own_vis = top1(cache["questions"][idx], [l for _, l in keys])
    print(f"DEV: τ = {tau:.3f} (95th pct of confidence, own hidden); own-visible exact recall "
          f"{np.mean([h['row'] == r for h, (r, _) in zip(own_vis, keys)]):.3f}")

    rows = []
    # --- scenario 1: new questions (own answer hidden), test rows, medbot_new answers on medquad_heldout
    mb = medbot_answers("medquad_heldout_??")
    keys = [(l, r) for l in LANGS for r in test_rows if (l, r) in mb]
    idx = np.array([pos[(r, l)] for l, r in keys])
    hits = top1(cache["questions"][idx], [l for l, _ in keys], hide=idx)
    for (l, r), h in zip(keys, hits):
        rows.append({"scenario": "new question (not in base)", "lang": l, "row": r, "conf": h["score"],
                     "routed": h["score"] >= tau, "medbot": mb[(l, r)]["answer"], "memory": h["answer"],
                     "ref": mb[(l, r)]["ref"]})

    # --- scenario 2: paraphrased known questions (own answer visible)
    para = medbot_answers("medquad_paraphrase_??")
    keys = sorted(para)
    vecs = qa.model.encode([para[k]["q"] for k in keys], normalize_embeddings=True, convert_to_numpy=True)
    hits = top1(vecs, [l for l, _ in keys])
    for (l, r), h in zip(keys, hits):
        rows.append({"scenario": "paraphrased known question", "lang": l, "row": r, "conf": h["score"],
                     "routed": h["score"] >= tau, "medbot": para[(l, r)]["answer"], "memory": h["answer"],
                     "ref": para[(l, r)]["ref"], "memory_is_own": h["row"] == r})

    df = pd.DataFrame(rows)
    df["hybrid"] = np.where(df.routed, df.memory, df.medbot)
    for col in ("medbot", "hybrid"):
        df[f"chrf_{col}"] = [sacrebleu.sentence_chrf(a, [ref]).score for a, ref in zip(df[col], df.ref)]
        df[f"sem_{col}"] = semantic_scores(list(zip(df[col], df.ref)))

    lines = ["| scenario | lang | n | routed to memory | chrF medbot → hybrid | semantic medbot → hybrid |",
             "|---|---|---|---|---|---|"]
    summary = []
    for (sc, l), g in df.groupby(["scenario", "lang"], sort=False):
        rec = {"scenario": sc, "lang": l, "n": len(g), "routed": float(g.routed.mean()),
               "chrf_medbot": g.chrf_medbot.mean(), "chrf_hybrid": g.chrf_hybrid.mean(),
               "sem_medbot": g.sem_medbot.mean(), "sem_hybrid": g.sem_hybrid.mean()}
        summary.append(rec)
        lines.append(f"| {sc} | {l} | {rec['n']} | {rec['routed']:.0%} | {rec['chrf_medbot']:.1f} → {rec['chrf_hybrid']:.1f} | "
                     f"{rec['sem_medbot']:.3f} → {rec['sem_hybrid']:.3f} |")
    para_routed = df[(df.scenario.str.startswith("paraphrased")) & df.routed]
    note = (f"\nτ = {tau:.3f} (chosen on dev rows, own answer hidden). Paraphrased questions routed to memory that "
            f"retrieved their own answer: {para_routed.memory_is_own.mean():.0%} of {len(para_routed)}.")
    text = "\n".join(lines) + "\n" + note + "\n"
    (OUT / "hybrid_router.md").write_text(text, encoding="utf-8")
    (OUT / "hybrid_router.json").write_text(json.dumps({"tau": tau, "summary": summary}, indent=2), encoding="utf-8")
    df.drop(columns=["medbot", "memory", "hybrid", "ref"]).to_csv(OUT / "hybrid_per_question.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
