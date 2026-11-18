import json

import pytest
import requests

from medqa.rag import MedQuadRAG
from medqa.saluki import GenConfig, ask_questions, export, load_done, read_jsonl, run_batch


def _run(dataset, url, output, config, rows=(0, 1), langs=("ru", "kk"), rag=None):
    with requests.Session() as session:
        return run_batch(dataset, rows, langs, output, config, session=session, base_url=url, rag=rag,
                         log=lambda *_: None)


def test_batch_writes_and_resumes(dataset, fake_server, tmp_path):
    out = tmp_path / "answers.jsonl"
    config = GenConfig.make()
    assert _run(dataset, fake_server, out, config)["done"] == 4
    assert _run(dataset, fake_server, out, config, rows=(0, 1, 2))["done"] == 2  # только недостающие
    recs = read_jsonl(out)
    assert len(recs) == 6 and recs["error"].isna().all()
    assert not recs["answer"].str.contains("<think>").any()  # блок рассуждений вырезан
    assert recs["concise"].all() and not recs["rag"].any()  # режим записан в каждую строку


def test_mixing_modes_is_refused(dataset, fake_server, tmp_path):
    out = tmp_path / "answers.jsonl"
    _run(dataset, fake_server, out, GenConfig.make())
    with pytest.raises(ValueError, match="другими настройками"):
        load_done(out, GenConfig.make(rag=True))


def test_old_records_without_mode_are_compatible(tmp_path):
    out = tmp_path / "old.jsonl"
    out.write_text(json.dumps({"row": 3, "lang": "ru", "answer": "x", "error": None}) + "\n", encoding="utf-8")
    assert load_done(out, GenConfig.make(rag=True)) == {(3, "ru")}


def test_rag_gives_context_and_excludes_row(dataset, fake_server, tmp_path):
    out = tmp_path / "rag.jsonl"
    _run(dataset, fake_server, out, GenConfig.make(rag=True), rows=(0,), langs=("kk",), rag=MedQuadRAG(dataset))
    rec = read_jsonl(out).iloc[0]
    assert "rag=True" in rec["answer"] and rec["rag_top_score"] > 0 and rec["rag_k"] == 3


def test_export_keeps_last_attempt(tmp_path):
    src = tmp_path / "a.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"row": 1, "lang": "ru", "answer": "", "error": "boom"},
        {"row": 1, "lang": "ru", "answer": "ок", "error": None},
        "битая строка"]) + "\n{не json\n", encoding="utf-8")
    df = export(src, tmp_path / "a.parquet")
    assert len(df) == 1 and df.iloc[0]["answer"] == "ок"


def test_new_questions_and_emergency_number(fake_server, tmp_path):
    questions = [{"en": "What is asthma?", "ru": "У меня боль в груди, что делать?", "kk": "Астма дегеніміз не?"}]
    with requests.Session() as session:
        df = ask_questions(questions, tmp_path / "new.jsonl", GenConfig.make(), session=session,
                           base_url=fake_server, log=lambda *_: None)
    assert len(df) == 3 and "row" not in df and (tmp_path / "new.parquet").exists()
    assert df.set_index("lang").loc["ru", "answer"] == "Позвоните 103 или 112."  # не американский 911


def test_gen_config_validation():
    assert GenConfig.make(rag=False, rag_k=0).rag_k is None
    with pytest.raises(ValueError):
        GenConfig.make(rag=True, rag_k=0)
