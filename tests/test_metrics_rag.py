import json

import numpy as np
import pandas as pd
import pytest

from medqa.config import SALUKI_RESULTS
from medqa.metrics import evaluate_answers, rouge_l, rouge_n
from medqa.rag import MedQuadRAG


@pytest.mark.parametrize("answers, saved", [("answers", "metrics"), ("answers_long", "metrics_long"),
                                            ("answers_rag", "metrics_rag")])
def test_metrics_reproduce_saved_results(dataset, answers, saved):
    _, summary = evaluate_answers(pd.read_parquet(SALUKI_RESULTS / f"{answers}.parquet"), dataset)
    expected = json.loads((SALUKI_RESULTS / f"{saved}.json").read_text(encoding="utf-8"))
    assert {k: summary[k] for k in expected} == expected


def test_errors_do_not_lower_quality(dataset):
    answers = pd.read_parquet(SALUKI_RESULTS / "answers.parquet").head(3).copy()
    answers.loc[0, "answer"] = None
    answers.loc[0, "error"] = "ConnectionError: boom"
    metrics, summary = evaluate_answers(answers, dataset)
    assert (summary["errors"], summary["evaluated"]) == (1, 2)
    assert "nan" not in metrics.loc[0, "answer"]


def test_rouge_basics():
    a = "сердце болит сильно".split()
    assert rouge_l(a, a) == 1.0 and rouge_n(a, a, 2) == 1.0
    assert rouge_l(a, []) == 0.0
    assert rouge_l("a b c d".split(), "a c d".split()) == pytest.approx(2 * 1.0 * 0.75 / 1.75)


@pytest.fixture(scope="module")
def rag(dataset):
    return MedQuadRAG(dataset)


def test_rag_excludes_query_row(rag, dataset):
    question = dataset.at[5, "question_ru"]
    assert rag.search(question, "ru", k=1)[0]["row"] == 5
    hits = rag.search(question, "ru", k=3, exclude_row=5)
    assert len(hits) == 3 and 5 not in [h["row"] for h in hits]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_rag_top_k_equals_full_sort(rag):
    question = "Что вызывает глаукому?"
    query = rag._vectorizers["ru"].transform([question])
    sims = (rag._matrices["ru"] @ query.T).toarray().ravel()
    expected = sorted(sims, reverse=True)[:5]
    assert [h["score"] for h in rag.search(question, "ru", k=5)] == pytest.approx(expected)
    assert np.isfinite(expected).all()
