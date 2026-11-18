"""Команда python -m medqa: подсказки, сообщения и коды выхода."""

import json
import subprocess
import sys

from medqa.config import ROOT, SALUKI_RESULTS


def run(*args):
    r = subprocess.run([sys.executable, "-m", "medqa", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def test_help_is_russian():
    code, out = run("--help")
    assert code == 0 and "Медицинский QA" in out and "saluki" in out and "medbot" in out
    assert "использование:" in out and "positional arguments" not in out and "show this help" not in out
    code, out = run("saluki", "run", "--help")
    assert code == 0 and "просить ответ в 3–5 предложений" in out
    code, out = run("rag", "вопрос", "--lang", "de")
    assert code == 2 and "недопустимое значение" in out


def test_data_and_rag():
    code, out = run("data")
    assert code == 0 and "осталось: 14368" in out and "validation: 1437" in out
    code, out = run("rag", "Что такое глаукома?", "--k", "2")
    assert code == 0 and out.count("[строка") == 2


def test_evaluate_writes_metrics(tmp_path):
    code, out = run("evaluate", str(SALUKI_RESULTS / "answers_rag.parquet"), "--out-prefix", str(tmp_path / "m"))
    assert code == 0
    saved = json.loads((SALUKI_RESULTS / "metrics_rag.json").read_text(encoding="utf-8"))
    got = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert {k: got[k] for k in saved} == saved


def test_saluki_run_against_fake_server(fake_server, tmp_path):
    out = tmp_path / "a.jsonl"
    code, log = run("saluki", "run", "--server-url", fake_server, "--limit", "2", "--langs", "kk",
                    "--output", str(out))
    assert code == 0, log
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2 and out.with_suffix(".parquet").exists()
    code, log = run("saluki", "run", "--server-url", fake_server, "--limit", "2", "--langs", "kk", "--rag",
                    "--output", str(out))
    assert code == 1 and "другими настройками" in log  # режимы не смешиваются


def test_friendly_errors():
    code, out = run("medbot", "chat", "--adapter", "models/нет_такого")
    assert code == 1 and "Адаптер не найден" in out
    code, out = run("evaluate", "нет_файла.parquet")
    assert code != 0
