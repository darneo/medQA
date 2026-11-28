"""Скачивает русские и казахские медицинские бенчмарки для harness и приводит их к одному формату.

    .venv\\Scripts\\python.exe scripts\\get_benchmarks.py

Результат — data/benchmarks/<имя>/{test,dev}.jsonl, в каждой строке {"question", "choices": [...], "answer": i}.
Лицензия обоих наборов CC BY-NC 4.0 (только некоммерческое использование), поэтому в git они не хранятся.

  mmedbench_ru         MMedBench (Henrychur/MMedBench), русская часть: 256 вопросов да/нет (из RuMedDaNet)
  kazmmlu_medicine_ru  KazMMLU (MBZUAI/KazMMLU), «Medicine (Professional & University in rus)»
  kazmmlu_biology_ru   KazMMLU, «Biology (High School in rus)»
  kazmmlu_biology_kk   KazMMLU, «Biology (High School in kaz)» — медицины на казахском в KazMMLU нет
"""

from __future__ import annotations

import ast
import io
import json
import random
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmarks"
HF = "https://huggingface.co/datasets"
KAZMMLU = {
    "kazmmlu_medicine_ru": "Medicine (Professional & University in rus)",
    "kazmmlu_biology_ru": "Biology (High School in rus)",
    "kazmmlu_biology_kk": "Biology (High School in kaz)",
}
LETTERS = "ABCDE"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "medqa-benchmarks"})) as r:
        return r.read()


def save(name: str, split: str, rows: list[dict]) -> None:
    path = OUT / name / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}: {len(rows)}")


def mmedbench() -> None:
    print("MMedBench (русский)")
    z = zipfile.ZipFile(io.BytesIO(fetch(f"{HF}/Henrychur/MMedBench/resolve/main/MMedBench.zip")))

    def rows(member: str) -> list[dict]:
        out = []
        for line in z.read(member).decode("utf-8").splitlines():
            d = json.loads(line)
            options = d["options"] if isinstance(d["options"], dict) else ast.literal_eval(d["options"])
            keys = sorted(options)
            out.append({"question": d["question"].strip(), "choices": [str(options[k]).strip() for k in keys],
                        "answer": keys.index(d["answer_idx"].strip())})
        return out

    save("mmedbench_ru", "test", rows("MMedBench/Test/Russian.jsonl"))
    train = rows("MMedBench/Train/Russian.jsonl")
    save("mmedbench_ru", "dev", random.Random(42).sample(train, 20))  # примеры для few-shot


def kazmmlu() -> None:
    for name, subset in KAZMMLU.items():
        print(f"KazMMLU: {subset}")
        for split in ("test", "dev"):
            url = f"{HF}/MBZUAI/KazMMLU/resolve/main/{urllib.parse.quote(subset)}/{split}.csv"
            df = pd.read_csv(io.BytesIO(fetch(url)), dtype=str).fillna("")
            rows = []
            for _, r in df.iterrows():
                choices = [r[f"Option {c}"].strip() for c in LETTERS if r.get(f"Option {c}", "").strip()]
                key = r["Answer Key"].strip().upper()
                if not r["Question"].strip() or key not in LETTERS[:len(choices)]:
                    continue  # битые строки: нет вопроса или ответ указывает на пустой вариант
                rows.append({"question": r["Question"].strip(), "choices": choices, "answer": LETTERS.index(key)})
            save(name, split, rows)


if __name__ == "__main__":
    mmedbench()
    kazmmlu()
