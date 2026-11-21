"""Загрузка датасета MedQuAD (en/ru/kk) и разбиение на train / validation.

Фильтр и разбиение менять нельзя: адаптер medbot обучен именно на этом
train, а validation (1437 строк) должен остаться теми же вопросами.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATASET, LANGS, SEED, VAL_RATIO


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


def split_train_val(df: pd.DataFrame, val_ratio: float = VAL_RATIO,
                    seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит ИСХОДНЫЕ строки на train и validation.

    Делим до разворачивания на языки, поэтому переводы одного вопроса никогда
    не попадают одновременно в обучение и проверку. Алгоритм совпадает с тем,
    на котором обучен адаптер medbot (validation = 1437 строк).
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio должен быть между 0 и 1")
    order = np.random.default_rng(seed).permutation(len(df))
    val_n = max(1, round(len(df) * val_ratio))
    val_ids = set(order[:val_n].tolist())
    train = df.iloc[[i for i in range(len(df)) if i not in val_ids]].reset_index(drop=True)
    val = df.iloc[order[:val_n]].reset_index(drop=True)
    return train, val


def to_examples(frame: pd.DataFrame, langs: Iterable[str] = LANGS) -> list[dict]:
    """Разворачивает строки на языки: одна строка -> по примеру на каждый язык."""
    examples = []
    for _, row in frame.iterrows():
        for lang in langs:
            examples.append({"lang": lang, "question": row[col("question", lang)],
                             "answer": row[col("answer", lang)]})
    return examples
