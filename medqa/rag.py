"""RAG: поиск похожих пар «вопрос–ответ» в датасете, чтобы подсказать их модели.

Индекс — TF-IDF по символьным n-граммам: устойчив к словоформам en/ru/kk и не
требует GPU. При оценке строка самого вопроса исключается (exclude_row), чтобы
модель не видела эталонный ответ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LANGS
from .data import col, load_dataset


class MedQuadRAG:
    def __init__(self, data: pd.DataFrame | Path | str | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.df = data if isinstance(data, pd.DataFrame) else load_dataset(data)
        self._vectorizers = {}
        self._matrices = {}
        for lang in LANGS:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self._matrices[lang] = vectorizer.fit_transform(self.df[col("question", lang)].tolist())
            self._vectorizers[lang] = vectorizer

    def search(self, question: str, lang: str, k: int = 3, exclude_row: int | None = None) -> list[dict]:
        """Top-k похожих пар на том же языке: [{row, score, question, answer}]."""
        if lang not in self._vectorizers:
            raise ValueError(f"Неизвестный язык: {lang}")
        if k < 1:
            raise ValueError("k должен быть >= 1")
        query = self._vectorizers[lang].transform([question])
        sims = (self._matrices[lang] @ query.T).toarray().ravel()
        top = min(k + 1, len(sims))  # +1 на случай исключённой строки
        candidates = np.argpartition(-sims, top - 1)[:top]
        hits = []
        for idx in candidates[np.argsort(-sims[candidates])]:
            idx = int(idx)
            if exclude_row is not None and idx == int(exclude_row):
                continue
            hits.append({"row": idx, "score": float(sims[idx]),
                         "question": self.df.at[idx, col("question", lang)],
                         "answer": self.df.at[idx, col("answer", lang)]})
            if len(hits) == k:
                break
        return hits

    def context(self, question: str, lang: str, k: int = 3, exclude_row: int | None = None,
                max_answer_chars: int = 1000) -> tuple[str, list[dict]]:
        """Текст справки для системной подсказки и сами найденные пары."""
        hits = self.search(question, lang, k, exclude_row)
        blocks = [f"[похожесть {h['score']:.2f}] Вопрос: {h['question']}\nОтвет: {h['answer'][:max_answer_chars]}"
                  for h in hits]
        return "\n\n".join(blocks), hits
