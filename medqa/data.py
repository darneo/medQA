"""Загрузка датасета MedQuAD (en/ru/kk).

Отдельной validation-выборки нет: RAG, Хопфилд и medbot используют весь датасет.
Фильтр строк менять нельзя — от него зависят номера строк в индексах.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATASET, LANGS, SEED


def col(base: str, lang: str) -> str:
    """Имя колонки: question / answer для en, question_ru / answer_kk и т.д."""
    return base if lang == "en" else f"{base}_{lang}"


REQUIRED_COLUMNS = [col(base, lang) for lang in LANGS for base in ("question", "answer")]


def load_dataset(path: Path | str | None = None, verbose: bool = False) -> pd.DataFrame:
    """Читает parquet и оставляет строки, где есть вопрос и ответ на всех трёх языках."""
    path = Path(path) if path is not None else DATASET
    if not path.is_file():
        raise FileNotFoundError(f"Нет датасета: {path}")
    df = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"В parquet не хватает колонок: {missing}")

    df = df.reset_index(drop=True)
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    complete = (df[REQUIRED_COLUMNS].apply(lambda s: s.str.len()) > 0).all(axis=1)
    if verbose:
        print(f"Строк в файле: {len(df)}; неполных отброшено: {int((~complete).sum())}; осталось: {int(complete.sum())}")
    return df[complete].reset_index(drop=True)


def first_adapter_unseen(df: pd.DataFrame) -> pd.DataFrame:
    """Строки, которых НЕ было в первом обучении medbot (1437 строк, 10 %, seed 42).

    Нужна только для дообучения medbot «на остальных данных» и для старых ноутбуков сравнения.
    """
    order = np.random.default_rng(SEED).permutation(len(df))
    return df.iloc[order[:round(len(df) * 0.1)]].reset_index(drop=True)


QUESTION_TYPES = [  # (тип, шаблон английского вопроса MedQuAD) — порядок важен: сначала частные
    ("symptoms", r"symptom"),
    ("genetic changes", r"genetic changes"),
    ("inheritance", r"inherit"),
    ("frequency", r"how many people|how common"),
    ("outlook", r"outlook"),
    ("research", r"research|clinical trial"),
    ("susceptibility", r"at risk"),
    ("prevention", r"prevent"),
    ("stages", r"stages"),
    ("complications", r"complication"),
    ("support", r"support"),
    ("exams and tests", r"diagnos"),
    ("treatment", r"treat|what to do for"),
    ("causes", r"\bcause"),
    ("information", r"^what (is|are)\b|information about"),
]


def question_type(question_en: str) -> str:
    """Тип вопроса по шаблону английской формулировки (в датасете колонка qtype пустая — «Unknown»)."""
    q = question_en.lower()
    for name, pattern in QUESTION_TYPES:
        if re.search(pattern, q):
            return name
    return "other"


def to_examples(frame: pd.DataFrame, langs: Iterable[str] = LANGS) -> list[dict]:
    """Разворачивает строки на языки: одна строка -> по примеру на каждый язык."""
    examples = []
    for _, row in frame.iterrows():
        for lang in langs:
            examples.append({"lang": lang, "question": row[col("question", lang)],
                             "answer": row[col("answer", lang)]})
    return examples
