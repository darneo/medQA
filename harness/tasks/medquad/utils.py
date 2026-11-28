"""Задачи MedQuAD (en / ru / kk) для lm-evaluation-harness.

lm-eval загружает этот файл сам (через `!function utils.…` в YAML), поэтому путь к пакету medqa
добавляется здесь: метрики и очистка ответа — те же, что в ноутбуках (medqa/evaluation.py, medqa/text.py).
"""

from __future__ import annotations

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
    """Полные строки, перемешанные с SEED: --limit N берёт одни и те же N случайных строк для всех моделей."""
    dataset = dataset.filter(_complete).shuffle(seed=SEED)
    q, a = col("question", lang), col("answer", lang)
    return dataset.map(lambda d: {"lang": lang, "q": str(d[q]).strip(), "ref": str(d[a]).strip()},
                       remove_columns=dataset.column_names)


def process_docs_en(dataset):
    return _process(dataset, "en")


def process_docs_ru(dataset):
    return _process(dataset, "ru")


def process_docs_kk(dataset):
    return _process(dataset, "kk")


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
