"""RAG: найти в датасете похожие вопросы с ответами и подсказать их модели.

Как это работает:

1. **Индекс** (`RAGIndex.build`, один раз). Каждая пара «вопрос + начало ответа» из **train**-части
   датасета — на каждом из трёх языков — превращается моделью эмбеддингов bge-m3 в вектор смысла
   (1024 числа). Векторы нормированы и сохраняются в data/rag_index/.
   Validation в индекс не попадает: при проверке на validation модель не может подсмотреть эталон.
2. **Поиск** (`search`). Вопрос тоже превращается в вектор той же моделью. Среди пар того же языка
   ищутся k с наибольшим косинусным сходством (у нормированных векторов это скалярное произведение).
3. **Справка** (`format_context`). Найденные ответы, обрезанные до max_chars, склеиваются в текст,
   который вставляется в системную подсказку модели (см. prompts.build_messages(context=...)).

    python -m medqa rag build                        # собрать индекс (видеокарта ускоряет в ~20 раз)
    python -m medqa rag search "Что такое астма?" --lang ru
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LANGS, RAG_EMBED_MODEL, RAG_INDEX_DIR
from .data import col, load_dataset, split_train_val

ANSWER_CHARS_IN_INDEX = 600   # сколько символов ответа добавлять к вопросу при построении вектора


def _encoder(model_name: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()
    return model


class RAGIndex:
    """Индекс эмбеддингов по train-части датасета. Поиск — внутри одного языка."""

    def __init__(self, index_dir: Path | str = RAG_INDEX_DIR, device: str = "cpu"):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Нет индекса RAG в {self.index_dir} — соберите: python -m medqa rag build")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.passages = pd.read_parquet(self.index_dir / "passages.parquet")
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype(np.float32)
        self.device = device
        self._model = None
        self._by_lang = {lang: np.flatnonzero(self.passages["lang"].to_numpy() == lang) for lang in LANGS}

    @property
    def model(self):
        if self._model is None:   # модель эмбеддингов грузится при первом поиске
            self._model = _encoder(self.meta["model"], self.device)
        return self._model

    # ---------------------------------------------------------------- построение

    @staticmethod
    def build(df: pd.DataFrame | None = None, index_dir: Path | str = RAG_INDEX_DIR,
              model_name: str = RAG_EMBED_MODEL, device: str | None = None, batch_size: int = 64) -> Path:
        """Собрать индекс по train-части (validation исключена) и сохранить в index_dir."""
        train, _ = split_train_val(df if df is not None else load_dataset())
        passages = pd.DataFrame([{"train_row": i, "lang": lang, "question": r[col("question", lang)],
                                  "answer": r[col("answer", lang)]}
                                 for i, r in train.iterrows() for lang in LANGS])
        texts = (passages["question"] + "\n" + passages["answer"].str[:ANSWER_CHARS_IN_INDEX]).tolist()
        t0 = time.time()
        model = _encoder(model_name, device)
        print(f"Эмбеддинги: {len(texts)} пар ({len(train)} строк train × {len(LANGS)} языка), модель {model_name}, "
              f"{model.device}", flush=True)
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        passages.to_parquet(index_dir / "passages.parquet", index=False)
        np.save(index_dir / "embeddings.npy", emb.astype(np.float16))
        meta = {"model": model_name, "passages": len(passages), "train_rows": len(train), "dim": int(emb.shape[1]),
                "answer_chars_in_index": ANSWER_CHARS_IN_INDEX, "split": "train (validation исключена)",
                "built_seconds": round(time.time() - t0)}
        (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Индекс сохранён в {index_dir} за {meta['built_seconds']} с", flush=True)
        return index_dir

    # ---------------------------------------------------------------- поиск

    def search_many(self, queries: list[tuple[str, str]], k: int = 3) -> list[list[dict]]:
        """Для каждого (вопрос, язык) — k самых похожих пар того же языка: [{score, question, answer, train_row}]."""
        vectors = self.model.encode([q for q, _ in queries], normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=32).astype(np.float32)
        results = []
        for vec, (_, lang) in zip(vectors, queries):
            if lang not in self._by_lang:
                raise ValueError(f"Неизвестный язык: {lang}")
            idx = self._by_lang[lang]
            scores = self.embeddings[idx] @ vec
            top = np.argsort(-scores)[:k]
            results.append([{"score": float(scores[t]), **self.passages.iloc[idx[t]][["question", "answer",
                                                                                     "train_row"]].to_dict()}
                            for t in top])
        return results

    def search(self, question: str, lang: str, k: int = 3) -> list[dict]:
        return self.search_many([(question, lang)], k)[0]


def format_context(hits: list[dict], max_chars: int = 700) -> str:
    """Справка для модели: найденные пары, ответы обрезаны до max_chars по границе слова."""
    blocks = []
    for n, h in enumerate(hits, 1):
        answer = h["answer"]
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(" ", 1)[0] + " …"
        blocks.append(f"[{n}] {h['question']}\n{answer}")
    return "\n\n".join(blocks)
