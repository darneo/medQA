"""Задачи MedQuAD (en / ru / kk) для lm-evaluation-harness.

lm-eval загружает этот файл сам (через `!function utils.…` в YAML), поэтому путь к пакету medqa
добавляется здесь: метрики и очистка ответа — те же, что в ноутбуках (medqa/evaluation.py, medqa/text.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medqa.config import SEED  # noqa: E402
from medqa.data import REQUIRED_COLUMNS, col  # noqa: E402
from medqa.evaluation import repeated_share, rouge_l  # noqa: E402
from medqa.text import advises_help, detect_language, localize_emergency, strip_think, tokens  # noqa: E402


def _complete(doc: dict) -> bool:
    """Тот же фильтр, что в medqa.data.load_dataset: вопрос и ответ есть на всех трёх языках."""
    return all(str(doc.get(c) or "").strip() for c in REQUIRED_COLUMNS)


def _process(dataset, lang: str):
    """Полные строки, перемешанные с SEED: --limit N берёт одни и те же N случайных строк для всех моделей.

    row — номер строки в medqa.data.load_dataset() (тот же фильтр и порядок), по нему RAG прячет свой ответ.
    """
    dataset = dataset.filter(_complete).map(lambda d, i: {"row": i}, with_indices=True).shuffle(seed=SEED)
    q, a = col("question", lang), col("answer", lang)
    return dataset.map(lambda d: {"row": d["row"], "lang": lang, "q": str(d[q]).strip(), "ref": str(d[a]).strip()},
                       remove_columns=dataset.column_names)


def process_docs_en(dataset):
    return _process(dataset, "en")


def process_docs_ru(dataset):
    return _process(dataset, "ru")


def process_docs_kk(dataset):
    return _process(dataset, "kk")


# ---------------------------------------------------------------- RAG: справка из датасета в системной подсказке

RAG_K = 3
RAG_ROWS = int(os.environ.get("MEDQUAD_RAG_ROWS", "300"))  # поиск только для первых N строк (больше --limit не нужно)
_RAG = None


def _with_rag(dataset, lang: str, hide_own: bool):
    """Добавляет поле system: подсказка medbot + k похожих пар из индекса RAG (medqa/rag.py).

    hide_own=True — свой ответ из справки убран (как для нового вопроса, которого нет в базе):
    модель видит только соседние вопросы. hide_own=False — свой ответ может попасть в справку
    (вопрос есть в базе знаний — так RAG работает в реальном использовании).
    """
    global _RAG
    from medqa.prompts import build_messages
    from medqa.rag import RAGIndex, format_context

    dataset = _process(dataset, lang).select(range(min(RAG_ROWS, len(dataset))))
    if _RAG is None:
        _RAG = RAGIndex(device="cpu")  # видеокарта занята моделью, которую оценивает harness
    hits = _RAG.search_many([(q, lang) for q in dataset["q"]], k=RAG_K + 1)
    systems = []
    for row, q, h in zip(dataset["row"], dataset["q"], hits):
        h = [x for x in h if not (hide_own and x["row"] == row)][:RAG_K]
        systems.append(build_messages(q, lang, context=format_context(h))[0]["content"])
    return dataset.add_column("system", systems)


def process_docs_rag_en(dataset):
    return _with_rag(dataset, "en", hide_own=False)


def process_docs_rag_ru(dataset):
    return _with_rag(dataset, "ru", hide_own=False)


def process_docs_rag_kk(dataset):
    return _with_rag(dataset, "kk", hide_own=False)


def process_docs_raghidden_en(dataset):
    return _with_rag(dataset, "en", hide_own=True)


def process_docs_raghidden_ru(dataset):
    return _with_rag(dataset, "ru", hide_own=True)


def process_docs_raghidden_kk(dataset):
    return _with_rag(dataset, "kk", hide_own=True)


def clean(text: str, lang: str) -> str:
    """Как medqa.medbot.generate: без <think>, 911 -> местные номера для ru / kk."""
    return localize_emergency(strip_think(text), lang)


def process_results(doc: dict, results: list[str]) -> dict:
    """Метрики одного ответа. Те же, что в medqa/evaluation.py (похожесть на эталон, не медицинская правильность)."""
    answer = clean(results[0], doc["lang"])
    words = tokens(answer)
    return {
        "rougeL": rouge_l(tokens(doc["ref"]), words),
        "chrf": (doc["ref"], answer),  # корпусный chrF (0–100) — агрегация chrf из lm-eval
        "lang_ok": float(detect_language(answer) == doc["lang"]),
        "repeats": repeated_share(words),
        "advises_doctor": float(advises_help(answer)),
        "answer_words": float(len(words)),
    }
