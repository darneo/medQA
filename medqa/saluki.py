"""Saluki 27B: прогон вопросов датасета и новых вопросов, запись ответов с возобновлением.

Каждый ответ — одна строка JSONL. В запись сохраняется режим генерации (RAG,
краткость, лимит токенов), чтобы при возобновлении не смешать в одном файле
ответы разных режимов.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .config import LANGS
from .data import col
from .llama_server import chat
from .prompts import build_messages
from .text import localize_emergency, strip_think


@dataclass(frozen=True)
class GenConfig:
    """Режим генерации — то, от чего зависит ответ."""
    rag: bool = False
    rag_k: int | None = None
    concise: bool = True
    max_tokens: int = 384
    thinking: bool = False

    @classmethod
    def make(cls, rag: bool = False, rag_k: int = 3, concise: bool = True, max_tokens: int = 384,
             thinking: bool = False) -> "GenConfig":
        if rag and rag_k < 1:
            raise ValueError("rag_k должен быть >= 1")
        return cls(rag=rag, rag_k=rag_k if rag else None, concise=concise, max_tokens=max_tokens, thinking=thinking)


CONFIG_KEYS = tuple(GenConfig.__dataclass_fields__)


def answer_one(session, base_url: str, question: str, lang: str, config: GenConfig, *, row: int | None = None,
               rag=None, temperature: float = 0.2, top_p: float = 0.9, timeout_s: float = 900,
               retries: int = 2, log: Callable[[str], None] = print) -> dict:
    """Один вопрос -> одна запись. Ошибки не бросает, а пишет в поле error.

    При RAG строка `row` исключается из поиска, чтобы модель не видела эталонный ответ.
    """
    context, top_score = None, None
    if config.rag:
        if rag is None:
            raise ValueError("Включён RAG, но индекс не передан")
        context, hits = rag.context(question, lang, config.rag_k, exclude_row=row)
        top_score = round(hits[0]["score"], 3) if hits else None
    messages = build_messages(question, lang, concise=config.concise, context=context)
    rec = {"row": row, "lang": lang, "question": question, "answer": "", "finish_reason": None,
           "prompt_tokens": None, "completion_tokens": None, "latency_s": None, "error": None,
           "rag_top_score": top_score, **asdict(config)}
    for attempt in range(1, retries + 2):
        t0 = time.time()
        try:
            result = chat(session, base_url, messages, max_tokens=config.max_tokens, temperature=temperature,
                          top_p=top_p, thinking=config.thinking, timeout_s=timeout_s)
            result["answer"] = localize_emergency(strip_think(result["answer"]), lang)
            rec.update(result, error=None, latency_s=round(time.time() - t0, 2))
            return rec
        except Exception as exc:  # noqa: BLE001 — батч должен продолжаться
            rec.update(error=f"{type(exc).__name__}: {exc}", latency_s=round(time.time() - t0, 2))
            if attempt <= retries:
                log(f"  повтор {attempt}/{retries} после ошибки: {exc}")
                time.sleep(min(5 * attempt, 30))
    return rec


def append_jsonl(out, rec: dict) -> None:
    """Дописать запись и сразу сбросить на диск: прогон можно прервать в любой момент."""
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())


def iter_jsonl(path: Path | str):
    """Записи JSONL по одной (как есть, без NaN для отсутствующих полей); битые строки пропускаются."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):  # строка, число и т.п. — не запись ответа
                yield rec


def read_jsonl(path: Path | str) -> pd.DataFrame:
    """JSONL -> DataFrame."""
    return pd.DataFrame(list(iter_jsonl(path)))


def load_answers(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else read_jsonl(path)


def load_done(path: Path, config: GenConfig) -> set[tuple[int, str]]:
    """Пары (row, lang), уже успешно отвеченные в файле.

    Если в файле есть ответы другого режима, останавливаемся: иначе такие вопросы
    молча пропустятся и результаты смешаются. Старые записи без полей режима совместимы.
    """
    current = asdict(config)
    done, conflicts = set(), set()
    for rec in iter_jsonl(path):
        for key in CONFIG_KEYS:
            if key in rec and rec[key] != current[key]:
                conflicts.add(f"{key}: в файле {rec[key]!r}, сейчас {current[key]!r}")
        if not rec.get("error") and rec.get("row") is not None and "lang" in rec:
            done.add((int(rec["row"]), str(rec["lang"])))
    if conflicts:
        raise ValueError(f"В {Path(path).name} уже есть ответы с другими настройками:\n  "
                         + "\n  ".join(sorted(conflicts))
                         + "\nУкажите другой файл (--output), чтобы не смешивать режимы.")
    return done


def select_rows(n_rows: int, limit: int | None = None, offset: int = 0, shuffle: bool = False,
                seed: int = 42) -> list[int]:
    rows = list(range(n_rows))
    if shuffle:
        random.Random(seed).shuffle(rows)
    rows = rows[offset:]
    return rows[:limit] if limit is not None else rows


def run_batch(df: pd.DataFrame, rows: Iterable[int], langs: Iterable[str], output: Path, config: GenConfig, *,
              session, base_url: str, rag=None, temperature: float = 0.2, top_p: float = 0.9,
              timeout_s: float = 900, retries: int = 2, progress: Callable[[int, int, dict], None] | None = None,
              log: Callable[[str], None] = print) -> dict:
    """Задаёт вопросы (row × lang), которых ещё нет в output. Возвращает сводку прогона."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, langs = list(rows), list(langs)
    done = load_done(output, config)
    pending = [(row, lang) for row in rows for lang in langs if (row, lang) not in done]
    total = len(rows) * len(langs)
    log(f"Задач: {total}; уже готово: {total - len(pending)}; осталось: {len(pending)}")
    errors, latencies, started = 0, [], time.time()
    with output.open("a", encoding="utf-8") as out:
        for n, (row, lang) in enumerate(pending, 1):
            rec = answer_one(session, base_url, df.at[row, col("question", lang)], lang, config, row=row, rag=rag,
                             temperature=temperature, top_p=top_p, timeout_s=timeout_s, retries=retries, log=log)
            append_jsonl(out, rec)
            if rec["error"]:
                errors += 1
            else:
                latencies.append(rec["latency_s"])
            if progress:
                progress(n, len(pending), rec)
    return {"done": len(pending), "errors": errors, "minutes": round((time.time() - started) / 60, 1),
            "avg_latency_s": round(sum(latencies) / len(latencies), 1) if latencies else None}


def export(src: Path, dst: Path) -> pd.DataFrame:
    """JSONL -> parquet; для повторов (row, lang) остаётся последняя попытка."""
    df = read_jsonl(src)
    if df.empty:
        raise FileNotFoundError(f"Нет ответов в {src}")
    if {"row", "lang"} <= set(df.columns):
        df = df.sort_values(["row", "lang"]).drop_duplicates(["row", "lang"], keep="last").reset_index(drop=True)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return df


def ask_questions(questions: list[dict], output: Path, config: GenConfig, *, session, base_url: str, rag=None,
                  langs: Iterable[str] = LANGS, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Новые вопросы (не из датасета): каждый на каждом языке. Пишет JSONL и parquet рядом."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    langs = list(langs)
    total, n = len(questions) * len(langs), 0
    with output.open("w", encoding="utf-8") as out:
        for qid, q in enumerate(questions):
            for lang in langs:
                n += 1
                rec = answer_one(session, base_url, q[lang], lang, config, rag=rag, log=log)
                rec = {"id": qid, **{k: v for k, v in rec.items() if k != "row"}}
                append_jsonl(out, rec)
                status = f"ОШИБКА {rec['error']}" if rec["error"] else f"{len(rec['answer'])} символов"
                log(f"[{n}/{total}] id={qid} {lang} {rec['latency_s']} с, {status}")
    df = read_jsonl(output)
    df.to_parquet(output.with_suffix(".parquet"), index=False)
    return df
