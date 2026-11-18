"""Метрики похожести ответов модели на эталоны MedQuAD.

ROUGE-1/2/L, BLEU, chrF, Jaccard, покрытие эталона, язык ответа, длина.
Это похожесть на эталон, а не медицинская точность: эталоны длинные и
переведены машинно, а модель отвечает своими словами.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .data import col
from .text import tokens


def _ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def _f1(overlap: float, pred_len: int, ref_len: int) -> float:
    precision = overlap / pred_len if pred_len else 0.0
    recall = overlap / ref_len if ref_len else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_n(ref: list[str], pred: list[str], n: int) -> float:
    ref_counts, pred_counts = Counter(_ngrams(ref, n)), Counter(_ngrams(pred, n))
    overlap = sum((ref_counts & pred_counts).values())
    return _f1(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def rouge_l(ref: list[str], pred: list[str]) -> float:
    """ROUGE-L F1 через длину наибольшей общей подпоследовательности (память O(len(pred)))."""
    if not ref or not pred:
        return 0.0
    prev = [0] * (len(pred) + 1)
    for x in ref:
        cur = [0]
        for j, y in enumerate(pred, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return _f1(prev[-1], len(pred), len(ref))


def _text(value) -> str:
    """None / NaN -> "" (иначе str(nan) превращается в слово "nan")."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def score_pair(pred: str, ref: str, lang: str) -> dict:
    import langid
    import sacrebleu

    ref_tok, pred_tok = tokens(ref), tokens(pred)
    pred_set, ref_set = set(pred_tok), set(ref_tok)
    union = pred_set | ref_set
    detected = langid.classify(pred)[0] if pred.strip() else ""
    return {
        "rouge1_f1": rouge_n(ref_tok, pred_tok, 1),
        "rouge2_f1": rouge_n(ref_tok, pred_tok, 2),
        "rougeL_f1": rouge_l(ref_tok, pred_tok),
        "bleu": sacrebleu.sentence_bleu(pred, [ref], tokenize="intl").score,
        "chrf": sacrebleu.sentence_chrf(pred, [ref]).score,
        "jaccard": len(pred_set & ref_set) / len(union) if union else 0.0,
        "reference_coverage": len(pred_set & ref_set) / len(ref_set) if ref_set else 0.0,
        "answer_chars": len(pred),
        "reference_chars": len(ref),
        "length_ratio": len(pred) / len(ref) if ref else 0.0,
        "detected_lang": detected,
        "language_match": detected == lang,
    }


def evaluate_answers(answers: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Сравнивает ответы (колонки row, lang, answer, ...) с эталонами датасета.

    Если пара (row, lang) встречается несколько раз, берётся последняя попытка.
    Средние метрики качества считаются только по успешным ответам.
    """
    if answers.empty:
        raise ValueError("Нет ответов для оценки")
    if not {"row", "lang"} <= set(answers.columns):
        raise ValueError("В ответах нужны колонки row и lang")
    answers = answers.drop_duplicates(["row", "lang"], keep="last")

    rows = []
    for rec in answers.to_dict("records"):
        row, lang = int(rec["row"]), str(rec["lang"])
        if not 0 <= row < len(reference):
            raise ValueError(f"row={row} вне датасета ({len(reference)} строк): ответы от другого parquet?")
        pred = _text(rec.get("answer"))
        ref = str(reference.at[row, col("answer", lang)])
        rows.append({
            "row": row, "lang": lang,
            "question": str(reference.at[row, col("question", lang)]),
            "reference": ref, "answer": pred,
            **score_pair(pred, ref, lang),
            "finish_reason": rec.get("finish_reason"),
            "completion_tokens": rec.get("completion_tokens"),
            "latency_s": rec.get("latency_s"),
            "error": _text(rec.get("error")) or None,
        })
    metrics = pd.DataFrame(rows)
    ok = metrics[metrics["error"].isna() & (metrics["answer_chars"] > 0)]
    if ok.empty:
        raise ValueError("Нет ни одного успешного ответа для оценки")
    summary = {
        "answers": len(metrics),
        "evaluated": len(ok),
        "errors": int(metrics["error"].notna().sum()),
        "empty_answers": int((metrics["answer_chars"] == 0).sum()),
        "truncated": int((metrics["finish_reason"] == "length").sum()),
        "language_match_pct": round(100 * ok["language_match"].mean(), 1),
        "rouge1_f1_mean": round(ok["rouge1_f1"].mean(), 3),
        "rouge2_f1_mean": round(ok["rouge2_f1"].mean(), 3),
        "rougeL_f1_mean": round(ok["rougeL_f1"].mean(), 3),
        "rougeL_f1_median": round(ok["rougeL_f1"].median(), 3),
        "bleu_mean": round(ok["bleu"].mean(), 1),
        "chrf_mean": round(ok["chrf"].mean(), 1),
        "jaccard_mean": round(ok["jaccard"].mean(), 3),
        "reference_coverage_mean": round(ok["reference_coverage"].mean(), 3),
        "length_ratio_median": round(ok["length_ratio"].median(), 2),
        "avg_latency_s": round(pd.to_numeric(ok["latency_s"]).mean(), 1),
        "avg_completion_tokens": round(pd.to_numeric(ok["completion_tokens"]).mean(), 0),
    }
    return metrics, summary


def save_metrics(metrics: pd.DataFrame, summary: dict, prefix: Path) -> tuple[Path, Path]:
    """Пишет <prefix>.parquet (по строкам) и <prefix>.json (сводка)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_parquet = prefix.parent / f"{prefix.name}.parquet"
    out_json = prefix.parent / f"{prefix.name}.json"
    metrics.to_parquet(out_parquet, index=False)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_parquet, out_json
