"""Тесты с выбором ответа на русском и казахском (данные — scripts/get_benchmarks.py).

Формат как у MMLU в lm-eval: вопрос, варианты «A. …», строка «Ответ:» / «Жауап:»; модель оценивается по тому,
какая буква для неё вероятнее (acc), без генерации текста.
"""

from __future__ import annotations

LETTERS = "ABCDE"
ANSWER_WORD = {"ru": "Ответ", "kk": "Жауап"}


def _text(doc: dict, lang: str) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(doc["choices"]))
    return f"{doc['question']}\n{options}\n{ANSWER_WORD[lang]}:"


def doc_to_text_ru(doc: dict) -> str:
    return _text(doc, "ru")


def doc_to_text_kk(doc: dict) -> str:
    return _text(doc, "kk")


def doc_to_choice(doc: dict) -> list[str]:
    return list(LETTERS[:len(doc["choices"])])
