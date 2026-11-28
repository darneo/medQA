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


def heldout_rows() -> dict[int, bool]:
    """Отложенная выборка: строки, которых не было в обучении medbot_old (medqa.data.first_adapter_unseen,
    10 %, seed 42), без 93 строк, чей вопрос дословно встречается в обучающих 90 %.

    Значение — встречалась ли эта болезнь (focus) в обучении с вопросом другого типа (для анализа ошибок).
    """
    import re

    import numpy as np

    from medqa.data import load_dataset

    df = load_dataset()
    order = np.random.default_rng(SEED).permutation(len(df))
    n = round(len(df) * 0.1)
    test, train = order[:n], order[n:]
    norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()  # noqa: E731
    train_q = {norm(df.question[i]) for i in train}
    train_a = {norm(df.answer[i]) for i in train}
    train_focus = {df.focus[i] for i in train}
    return {int(i): bool(df.focus[i] in train_focus) for i in test
            if norm(df.question[i]) not in train_q and norm(df.answer[i]) not in train_a}


def _process(dataset, lang: str, heldout: bool = False):
    """Полные строки, перемешанные с SEED: --limit N берёт одни и те же N случайных строк для всех моделей.

    row — номер строки в medqa.data.load_dataset() (тот же фильтр и порядок), по нему RAG прячет свой ответ.
    heldout=True — только отложенная выборка (см. heldout_rows), поле seen_focus для анализа ошибок.
    """
    dataset = dataset.filter(_complete).map(lambda d, i: {"row": i}, with_indices=True)
    if heldout:
        rows = heldout_rows()
        dataset = dataset.filter(lambda d: d["row"] in rows)
    dataset = dataset.shuffle(seed=SEED)
    q, a = col("question", lang), col("answer", lang)

    def fields(d):
        out = {"row": d["row"], "lang": lang, "q": str(d[q]).strip(), "ref": str(d[a]).strip()}
        if heldout:
            out["seen_focus"] = rows[d["row"]]
        return out

    return dataset.map(fields, remove_columns=dataset.column_names)


# ---------------------------------------------------------------- RAG: справка из датасета в системной подсказке

RAG_K = 3
RAG_ROWS = int(os.environ.get("MEDQUAD_RAG_ROWS", "300"))  # поиск только для первых N строк (больше --limit не нужно)
_RAG = None


def _with_rag(dataset, lang: str, hide_own: bool, heldout: bool = False):
    """Добавляет поле system: подсказка medbot + k похожих пар из индекса RAG (medqa/rag.py).

    hide_own=True — свой ответ из справки убран (как для нового вопроса, которого нет в базе):
    модель видит только соседние вопросы. hide_own=False — свой ответ может попасть в справку
    (вопрос есть в базе знаний — так RAG работает в реальном использовании).
    """
    global _RAG
    from medqa.prompts import build_messages
    from medqa.rag import RAGIndex, format_context

    dataset = _process(dataset, lang, heldout)
    dataset = dataset.select(range(min(RAG_ROWS, len(dataset))))
    if _RAG is None:
        _RAG = RAGIndex(device="cpu")  # видеокарта занята моделью, которую оценивает harness
    hits = _RAG.search_many([(q, lang) for q in dataset["q"]], k=RAG_K + 1)
    systems = []
    for row, q, h in zip(dataset["row"], dataset["q"], hits):
        h = [x for x in h if not (hide_own and x["row"] == row)][:RAG_K]
        systems.append(build_messages(q, lang, context=format_context(h))[0]["content"])
    return dataset.add_column("system", systems)


# process_docs_<вариант>_<язык> для YAML: medquad[_heldout][_rag|_raghidden]_{en,ru,kk}
def _make(lang: str, heldout: bool, rag: str | None):
    if rag is None:
        return lambda dataset: _process(dataset, lang, heldout)
    return lambda dataset: _with_rag(dataset, lang, hide_own=(rag == "raghidden"), heldout=heldout)


for _lang in ("en", "ru", "kk"):
    for _held in (False, True):
        for _rag in (None, "rag", "raghidden"):
            _name = "_".join(x for x in ("process_docs", "heldout" if _held else "", _rag or "", _lang) if x)
            globals()[_name] = _make(_lang, _held, _rag)


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
