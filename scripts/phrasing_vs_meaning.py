"""Surface overlap (chrF, ROUGE-L) vs meaning (bge-m3 cosine) for every model on the same 24 questions
(results/all_models, scripts/test_all_models.py) -> results/all_models/phrasing_vs_meaning.{csv,md} + docs/figures/.

    .venv\\Scripts\\python.exe scripts\\phrasing_vs_meaning.py

A model that answers correctly in its own words gets low chrF / ROUGE-L but high semantic similarity.
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
from medqa.evaluation import rouge_l  # noqa: E402
from medqa.text import tokens  # noqa: E402

SRC = ROOT / "results" / "all_models"
MODELS = {"hopfield_hidden": "Hopfield (own answer hidden)", "medbot_old": "medbot_old (QLoRA 4B)",
          "medbot_full": "medbot_full (QLoRA 4B)", "saluki_rag": "Saluki 27B + RAG", "lingsante": "LingSante 124B-MoE",
          "lingsante_rag": "LingSante + RAG"}


def main() -> None:
    set_dir(ROOT / "results" / "harness")  # shared bge-m3 cache
    rows = []
    for key, name in MODELS.items():
        path = SRC / f"{key}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        q = [json.loads(line) for line in (SRC / "questions.jsonl").open(encoding="utf-8")]
        for r, qq in zip(recs, q):
            r.setdefault("reference", qq["reference"])
            r.setdefault("lang", qq["lang"])
        sem = semantic_scores([(r["answer"], r["reference"]) for r in recs])
        for r, s in zip(recs, sem):
            rows.append({"model": name, "lang": r["lang"], "chrF": sacrebleu.sentence_chrf(r["answer"], [r["reference"]]).score,
                         "ROUGE-L": rouge_l(tokens(r["reference"]), tokens(r["answer"])), "semantic": s,
                         "answer words": len(tokens(r["answer"]))})
    df = pd.DataFrame(rows)
    agg = df.groupby("model", sort=False)[["chrF", "ROUGE-L", "semantic", "answer words"]].mean()
    agg["n"] = df.groupby("model", sort=False).size()
    agg.round(3).to_csv(SRC / "phrasing_vs_meaning.csv")
    lines = ["| model | n | chrF | ROUGE-L | semantic (bge-m3) | answer words |", "|---|---|---|---|---|---|"]
    lines += [f"| {m} | {r.n} | {r.chrF:.1f} | {r['ROUGE-L']:.3f} | {r.semantic:.3f} | {r['answer words']:.0f} |"
              for m, r in agg.iterrows()]
    (SRC / "phrasing_vs_meaning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (m, r) in enumerate(agg.iterrows()):
        sel = df[df.model == m]
        ax.scatter(sel["chrF"], sel["semantic"], s=14, alpha=0.35, color=f"C{i}")
        ax.scatter(r.chrF, r.semantic, s=90, color=f"C{i}", edgecolor="black", label=m, zorder=3)
    ax.set_xlabel("chrF (surface overlap with reference)")
    ax.set_ylabel("semantic similarity (bge-m3 cosine)")
    ax.set_title("Same 24 questions: wording vs meaning (big dot = mean)", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "phrasing_vs_meaning.png", dpi=160)
    print("figure: docs/figures/phrasing_vs_meaning.png", np.round(agg["semantic"].to_numpy(), 3))


if __name__ == "__main__":
    main()
