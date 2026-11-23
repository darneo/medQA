"""Тест всех моделей на одних и тех же случайных вопросах из датасета.

    .venv\\Scripts\\python.exe scripts\\test_all_models.py                 # 8 строк × 3 языка = 24 вопроса
    .venv\\Scripts\\python.exe scripts\\test_all_models.py --rows 4 --only hopfield medbot_full

Модели (по очереди, чтобы хватило 8 ГБ видеопамяти):
  hopfield        — обученная сеть Хопфилда, свой ответ в памяти (как для вопроса из датасета)
  hopfield_hidden — то же, но свой ответ скрыт (как для нового вопроса, которого нет в датасете)
  medbot_old    — Qwen3-4B + первый адаптер (обучен на 90 % строк)
  medbot_full   — Qwen3-4B + адаптер, дообученный на оставшихся 10 % (видел весь датасет)
  saluki_rag    — Saluki 27B + RAG по всему датасету (k=3)
  lingsante     — LingSante-3.1-air-VL (124B MoE, 5.5B активных, GGUF 2 бита), без RAG
  lingsante_rag — то же + RAG по всему датасету (k=3)

Все вопросы взяты из датасета, поэтому это проверка того, как модели воспроизводят известные
ответы, а не обобщение на новые вопросы. Результаты: results/all_models/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from medqa.config import LANGS, MEDBOT_ADAPTER, RESULTS_DIR  # noqa: E402
from medqa.data import col, load_dataset  # noqa: E402

OUT = RESULTS_DIR / "all_models"
MODELS = ["hopfield", "hopfield_hidden", "medbot_old", "medbot_full", "saluki_rag", "lingsante", "lingsante_rag"]
FULL_ADAPTER = ROOT / "models" / "medbot_train_full" / "lora_adapter"


def pick_questions(rows: int, seed: int) -> pd.DataFrame:
    df = load_dataset()
    idx = np.random.default_rng(seed).choice(len(df), rows, replace=False)
    return pd.DataFrame([{"row": int(i), "lang": lang, "question": df.iloc[i][col("question", lang)],
                          "reference": df.iloc[i][col("answer", lang)], "focus": df.iloc[i].get("focus"),
                          "qtype": df.iloc[i].get("qtype")}
                         for i in idx for lang in LANGS])


def run_hopfield(qs: pd.DataFrame, hide_own: bool) -> list[dict]:
    from medqa.hopfield import HopfieldQA, load_cache

    cache = load_cache()
    p = cache["passages"]
    pos = {(r, l): i for i, (r, l) in enumerate(zip(p["row"], p["lang"]))}
    own = np.array([pos[(r, l)] for r, l in zip(qs["row"], qs["lang"])])
    qa = HopfieldQA()
    hits = qa.search_vectors(cache["questions"][own], qs["lang"].tolist(), k=1, hide=own if hide_own else None)
    return [{"answer": h[0]["answer"], "truncated": False, "confidence": round(h[0]["score"], 3),
             "matched_question": h[0]["question"], "same_disease": h[0]["focus"] == f,
             "same_type": h[0]["focus"] == f and h[0]["qtype"] == t}
            for h, f, t in zip(hits, p["focus"].to_numpy()[own], p["qtype"].to_numpy()[own])]


def run_medbot(qs: pd.DataFrame, adapter: Path, max_new_tokens: int) -> list[dict]:
    import gc

    import torch

    from medqa.medbot import generate, load_model

    model, tokenizer = load_model(adapter)
    out = []
    for n, (_, q) in enumerate(qs.iterrows(), 1):
        t0 = time.time()
        answer = generate(model, tokenizer, q.question, q.lang, max_new_tokens=max_new_tokens, temperature=0)
        n_tok = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
        out.append({"answer": answer, "truncated": n_tok >= max_new_tokens - 5, "seconds": round(time.time() - t0, 1)})
        print(f"  {n}/{len(qs)} [{q.lang}] {time.time() - t0:.0f} с", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_llama(qs: pd.DataFrame, max_new_tokens: int, *, use_rag: bool, config=None) -> list[dict]:
    """Модель через llama-server (Saluki по умолчанию, или config с другой GGUF), с RAG или без."""
    from medqa.rag import RAGIndex, format_context
    from medqa.saluki import SalukiServer, ask

    hits = [None] * len(qs)
    if use_rag:
        hits = RAGIndex().search_many(list(zip(qs["question"], qs["lang"])), k=3)
    out = []
    with SalukiServer(config) as server:
        for n, ((_, q), h) in enumerate(zip(qs.iterrows(), hits), 1):
            t0 = time.time()
            r = ask(q.question, q.lang, url=server.url, context=format_context(h) if h else None,
                    max_tokens=max_new_tokens)
            row = {"answer": r["answer"], "truncated": r["finish_reason"] == "length",
                   "completion_tokens": r["completion_tokens"], "seconds": round(time.time() - t0, 1)}
            if use_rag:
                row["rag_found_own"] = any(x["row"] == q.row for x in h)
            out.append(row)
            print(f"  {n}/{len(qs)} [{q.lang}] {time.time() - t0:.0f} с, {r['completion_tokens']} ток.", flush=True)
    return out


def lingsante_config():
    """LingSante-3.1-air-VL: все эксперты MoE в ОЗУ (--cpu-moe), остальные слои на видеокарте."""
    from medqa.config import LINGSANTE_MODEL
    from medqa.saluki import ServerConfig

    return ServerConfig(model=LINGSANTE_MODEL, port=8081, ctx=8192, parallel=1, gpu_layers=99,
                        extra=["--cpu-moe"])


TITLES = {"hopfield": "Хопфилд", "hopfield_hidden": "Хопфилд (свой ответ скрыт)", "medbot_old": "medbot старый", "medbot_full": "medbot новый",
          "saluki_rag": "Saluki + RAG", "lingsante": "LingSante", "lingsante_rag": "LingSante + RAG"}


def side_by_side(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица для просмотра (Excel): вопрос, эталон и ответы всех моделей в одной строке."""
    first = df[df["model"] == df["model"].iloc[0]].reset_index(drop=True)
    out = first[["row", "lang", "focus", "qtype", "question", "reference"]].copy()
    out.insert(0, "id", [f"row{r}_{lang}" for r, lang in zip(out["row"], out["lang"])])
    for name, title in TITLES.items():
        d = df[df["model"] == name].reset_index(drop=True)
        if d.empty:
            continue
        out[f"{title}: ответ"] = d["answer"]
        out[f"{title}: chrF"] = d["chrF"].round(1)
        if name.startswith("hopfield"):
            out[f"{title}: вспомненный вопрос"] = d["matched_question"]
            out[f"{title}: болезнь и тип совпали"] = d["same_type"].map({True: "да", False: "нет"})
        if name.endswith("_rag"):
            out[f"{title}: RAG нашёл эталон"] = d["rag_found_own"].map({True: "да", False: "нет"})
    return out.rename(columns={"row": "строка", "lang": "язык", "focus": "болезнь", "qtype": "тип вопроса",
                               "question": "вопрос", "reference": "эталон (ответ из датасета)"})


def main() -> int:
    from medqa.evaluation import add_metrics

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=8, help="случайных строк датасета (× 3 языка)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--only", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--redo", action="store_true", help="пересчитать модели, у которых уже есть результат")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    qs = pick_questions(args.rows, args.seed)
    qs.to_json(OUT / "questions.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Вопросов: {len(qs)} (строки датасета {sorted(qs['row'].unique().tolist())})", flush=True)

    runners = {"hopfield": lambda: run_hopfield(qs, hide_own=False),
               "hopfield_hidden": lambda: run_hopfield(qs, hide_own=True),
               "medbot_old": lambda: run_medbot(qs, MEDBOT_ADAPTER, args.max_new_tokens),
               "medbot_full": lambda: run_medbot(qs, FULL_ADAPTER, args.max_new_tokens),
               "saluki_rag": lambda: run_llama(qs, args.max_new_tokens, use_rag=True),
               "lingsante": lambda: run_llama(qs, args.max_new_tokens, use_rag=False, config=lingsante_config()),
               "lingsante_rag": lambda: run_llama(qs, args.max_new_tokens, use_rag=True, config=lingsante_config())}
    for name in args.only:
        path = OUT / f"{name}.jsonl"
        if path.exists() and not args.redo:
            print(f"{name}: уже есть {path.name}, пропускаю", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        res = pd.DataFrame(runners[name]())
        pd.concat([qs.reset_index(drop=True), res], axis=1).to_json(path, orient="records", lines=True,
                                                                    force_ascii=False)
        print(f"{name}: готово за {(time.time() - t0) / 60:.1f} мин", flush=True)

    frames = []
    for name in MODELS:
        path = OUT / f"{name}.jsonl"
        if path.exists():
            frames.append(pd.read_json(path, lines=True).assign(model=name))
    df = add_metrics(pd.concat(frames, ignore_index=True))
    metrics = ["язык верный", "ROUGE-L", "chrF", "обрезан", "повторы"]
    table = df.groupby("model", sort=False)[metrics].mean()
    by_lang = df.pivot_table(index="model", columns="lang", values="chrF", aggfunc="mean", sort=False)
    table[[f"chrF {lang}" for lang in by_lang.columns]] = by_lang.to_numpy()
    for m in ("язык верный", "обрезан", "повторы"):
        table[m] *= 100
    table = table.round(2)
    print("\nИтог (язык/обрезан/повторы — %, ROUGE-L 0–1, chrF 0–100 — похожесть на эталон):")
    print(table.to_string())
    table.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    df.to_json(OUT / "all_answers.jsonl", orient="records", lines=True, force_ascii=False)
    side_by_side(df).to_csv(OUT / "answers_side_by_side.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(table.to_dict(orient="index"), ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\nСохранено в {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
