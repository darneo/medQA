"""Сводка всех прогонов lm-evaluation-harness: results/harness/SUMMARY.md.

    .venv\\Scripts\\python.exe scripts\\harness_report.py
    .venv\\Scripts\\python.exe scripts\\harness_report.py --no-semantic   # без bge-m3 (быстрее)

Для задач с генерацией (medquad*) дополнительно считается смысловая близость ответа к эталону:
косинус эмбеддингов bge-m3 (0–1). В отличие от ROUGE-L и chrF она не штрафует верный ответ другими словами.
Для тестов с выбором ответа — точность (acc) ± стандартная ошибка из harness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "harness"
CACHE = OUT / "semantic_cache.json"
MODEL_ORDER = ["base", "medbot_old", "medbot_full"]
GEN_METRICS = {"chrf": "chrF", "rougeL": "ROUGE-L", "semantic": "смысл (bge-m3)", "lang_ok": "язык верный",
               "repeats": "повторы", "advises_doctor": "совет к врачу", "answer_words": "слов в ответе"}


def set_dir(path: Path) -> None:
    global OUT, CACHE
    OUT, CACHE = Path(path), Path(path) / "semantic_cache.json"


def _load_utils():
    spec = importlib.util.spec_from_file_location("medquad_utils", ROOT / "harness/tasks/medquad/utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latest_runs() -> dict[tuple[str, str, int], dict]:
    """(модель, задача, n-shot) -> метрики и путь к samples последнего прогона."""
    runs: dict[tuple[str, str, int], dict] = {}
    for path in sorted(OUT.glob("*/*/results_*.json")):  # имена с датой: сортировка = по времени
        model = path.parent.parent.name
        d = json.loads(path.read_text(encoding="utf-8"))
        stamp = path.stem.removeprefix("results_")
        for task, res in d["results"].items():
            if task not in d.get("n-shot", {}):
                continue  # строка группы, не задачи
            shot = int(d["n-shot"][task])
            samples = path.parent / f"samples_{task}_{stamp}.jsonl"
            runs[(model, task, shot)] = {"res": res, "n": d["n-samples"][task]["effective"],
                                         "samples": samples if samples.exists() else None}
    return runs


def semantic_scores(pairs: list[tuple[str, str]]) -> list[float]:
    """Косинус bge-m3 между ответом и эталоном; результаты кэшируются в semantic_cache.json."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    keys = [hashlib.sha1(f"{a}\x00{r}".encode()).hexdigest() for a, r in pairs]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        model.max_seq_length = 512
        texts = [t for i in todo for t in pairs[i]]
        emb = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        for j, i in enumerate(todo):
            cache[keys[i]] = float(emb[2 * j] @ emb[2 * j + 1])
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[k] for k in keys]


def generative_table(runs: dict, semantic: bool) -> pd.DataFrame:
    utils = _load_utils()
    rows = []
    for (model, task, shot), r in runs.items():
        if not task.startswith("medquad"):
            continue
        variant, lang = task.rsplit("_", 1)
        row = {"модель": model, "вариант": {"medquad": "без RAG", "medquad_rag": "RAG",
                                             "medquad_raghidden": "RAG, свой ответ скрыт"}[variant],
               "язык": lang, "n": r["n"]}
        for m in GEN_METRICS:
            if f"{m},none" in r["res"]:
                row[m] = r["res"][f"{m},none"]
        if semantic and r["samples"]:
            samples = [json.loads(line) for line in r["samples"].open(encoding="utf-8")]
            pairs = [(utils.clean(s["resps"][0][0], s["doc"]["lang"]), s["doc"]["ref"]) for s in samples]
            row["semantic"] = float(np.mean(semantic_scores(pairs)))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_m"] = df["модель"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["_l"] = df["язык"].map({"en": 0, "ru": 1, "kk": 2})
    return df.sort_values(["вариант", "_m", "_l"]).drop(columns=["_m", "_l"])


def mc_table(runs: dict) -> pd.DataFrame:
    rows = {}
    for (model, task, shot), r in runs.items():
        if task.startswith("medquad") or "acc,none" not in r["res"]:
            continue
        key = (task, shot)
        rows.setdefault(key, {"задача": task, "shot": shot, "n": r["n"]})
        rows[key][model] = f"{100 * r['res']['acc,none']:.1f} ± {100 * r['res']['acc_stderr,none']:.1f}"
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    cols = ["задача", "shot", "n"] + [m for m in MODEL_ORDER if m in df] + [m for m in df if m not in
                                                                             MODEL_ORDER + ["задача", "shot", "n"]]
    return df[cols].sort_values(["задача", "shot"])


def to_markdown(df: pd.DataFrame) -> str:
    fmt = df.copy()
    for c in fmt.columns:
        if c in ("lang_ok", "repeats", "advises_doctor"):
            fmt[c] = (100 * fmt[c]).map("{:.0f} %".format)
        elif c in ("chrf", "answer_words"):
            fmt[c] = fmt[c].map("{:.1f}".format)
        elif c in ("rougeL", "semantic"):
            fmt[c] = fmt[c].map("{:.3f}".format)
    fmt = fmt.rename(columns=GEN_METRICS)
    lines = ["| " + " | ".join(map(str, fmt.columns)) + " |", "|" + "---|" * len(fmt.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in fmt.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--dir", type=Path, default=OUT, help="папка с прогонами (по умолчанию results/harness)")
    a = p.parse_args()
    set_dir(a.dir)
    runs = latest_runs()
    gen, mc = generative_table(runs, not a.no_semantic), mc_table(runs)
    text = ["# Результаты lm-evaluation-harness", "",
            "Собрано `scripts/harness_report.py` из `results/harness/*/*/results_*.json` (последний прогон "
            "каждой задачи).", ""]
    if not gen.empty:
        text += ["## MedQuAD: ответы своими словами (en / ru / kk)", "",
                 "ROUGE-L, chrF, смысл — похожесть на эталон датасета (выше — лучше); повторы — ниже лучше.", "",
                 to_markdown(gen), ""]
    if not mc.empty:
        text += ["## Тесты с выбором ответа: точность, %", "", to_markdown(mc), ""]
    (OUT / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
