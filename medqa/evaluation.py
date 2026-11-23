"""Метрики для сравнения ответов двух вариантов модели (в ноутбуках notebooks/compare_*.ipynb).

Все метрики — похожесть на эталон и простые проверки текста, а НЕ медицинская правильность.

| метрика            | что значит                                                     | лучше |
|--------------------|----------------------------------------------------------------|-------|
| язык верный        | ответ на языке вопроса                                         | выше  |
| ROUGE-L            | похожесть на эталон по словам (0–1), только если эталон есть   | выше  |
| chrF               | похожесть на эталон по буквам (0–100), честнее для ru / kk     | выше  |
| обрезан            | ответ упёрся в лимит токенов                                   | ниже  |
| повторы            | доля повторяющихся 4-словных фраз — зацикливание               | ниже  |
| совет к врачу / 103| в ответе есть совет обратиться к врачу или вызвать скорую      | выше  |
"""

from __future__ import annotations

import pandas as pd

from .text import advises_help, detect_language, tokens

HIGHER_IS_BETTER = {"язык верный": True, "ROUGE-L": True, "chrF": True, "обрезан": False, "повторы": False,
                    "совет к врачу / 103": True}
PERCENT = ("язык верный", "обрезан", "повторы", "совет к врачу / 103")


def rouge_l(ref: list[str], pred: list[str]) -> float:
    """ROUGE-L F1 по словам: длина наибольшей общей подпоследовательности."""
    if not ref or not pred:
        return 0.0
    prev = [0] * (len(pred) + 1)
    for x in ref:
        cur = [0]
        for j, y in enumerate(pred, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(ref)
    return 2 * p * r / (p + r)


def repeated_share(words: list[str], n: int = 4) -> float:
    """Доля n-словных фраз, которые уже встречались в этом же ответе."""
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонки метрик. Нужны колонки answer, lang, reference (или None), truncated (bool)."""
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None
    df = df.copy()
    df["язык верный"] = [detect_language(a) == lang for a, lang in zip(df["answer"], df["lang"])]
    df["обрезан"] = df["truncated"].astype(bool)
    df["повторы"] = [repeated_share(tokens(a)) for a in df["answer"]]
    df["совет к врачу / 103"] = [advises_help(a) for a in df["answer"]]
    has_ref = df["reference"].notna()
    refs, answers = df.loc[has_ref, "reference"], df.loc[has_ref, "answer"]
    df.loc[has_ref, "ROUGE-L"] = [rouge_l(tokens(r), tokens(a)) for r, a in zip(refs, answers)]
    if sacrebleu is not None:
        df.loc[has_ref, "chrF"] = [sacrebleu.sentence_chrf(a, [r]).score for r, a in zip(refs, answers)]
    return df


def summary_table(df: pd.DataFrame, by: str, a: str, b: str) -> pd.DataFrame:
    """Средние метрики двух вариантов (a, b) и кто лучше. Доли — в процентах."""
    metrics = [m for m in HIGHER_IS_BETTER if m in df]
    table = df.groupby(by)[metrics].mean().T[[a, b]]
    diff = table[b] - table[a]
    table[f"разница ({b} − {a})"] = diff
    table["кто лучше"] = [("=" if abs(d) < 1e-9 else (b if (d > 0) == HIGHER_IS_BETTER[m] else a))
                          for m, d in diff.items()]
    for m in PERCENT:
        if m in table.index:
            table.loc[m, [a, b, f"разница ({b} − {a})"]] *= 100
    table.index = [f"{m}, %" if m in PERCENT else m for m in table.index]
    return table.round(3)
