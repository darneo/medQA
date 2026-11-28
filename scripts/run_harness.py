"""Оценка моделей через lm-evaluation-harness (EleutherAI, pip install "lm-eval[hf]").

    .venv\\Scripts\\python.exe scripts\\run_harness.py                          # medbot_full, MedQuAD en/ru/kk, 50 строк
    .venv\\Scripts\\python.exe scripts\\run_harness.py --models base medbot_old medbot_full --limit 100
    .venv\\Scripts\\python.exe scripts\\run_harness.py --suite bench --limit 200  # стандартные медицинские бенчмарки
    .venv\\Scripts\\python.exe scripts\\run_harness.py --dry-run                  # только показать команды

Наборы задач (--suite):
  medquad — свои задачи harness/tasks/medquad: вопросы датасета на en / ru / kk, ответ генерируется,
            метрики как в ноутбуках (ROUGE-L, chrF, язык, повторы, совет к врачу). Все строки датасета
            были в обучении medbot, так что это проверка воспроизведения, а не обобщения.
  medquad_rag       — то же + справка RAG (3 похожие пары из датасета) в системной подсказке;
                      свой ответ может попасть в справку (вопрос есть в базе знаний).
  medquad_raghidden — то же, но свой ответ из справки убран (как для нового вопроса) — честный вариант.
  heldout[_rag|_raghidden] — то же на ОТЛОЖЕННОЙ выборке: 1344 строки, которых medbot_old не видел при обучении
                      (10 % датасета, seed 42, без дословных повторов вопросов из обучающей части).
                      Главный честный тест: base против medbot_old. medbot_full эти строки видел.
  bench   — готовые англоязычные бенчмарки с выбором варианта ответа (acc): MedQA-USMLE (4 варианта),
            MedMCQA, PubMedQA и медицинские разделы MMLU. Их модели не видели — честное сравнение
            с базовой Qwen3-4B и с цифрами из статей.
  ru_kk   — тесты с выбором ответа на русском и казахском: MMedBench-ru (да/нет), KazMMLU «Медицина» (ru),
            «Биология» (ru, kk). Данные сначала скачать: python scripts/get_benchmarks.py

Модели (--models) запускаются по очереди, чтобы хватило 8 ГБ видеопамяти. Результаты: results/harness/<модель>/.
Saluki и LingSante (GGUF через llama-server) сюда не входят: harness их тоже умеет (--model local-completions),
но для них нужен запущенный сервер — это отдельный шаг.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "harness" / "tasks"
OUT = ROOT / "results" / "harness"

BASE = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MODELS = {
    "base": {"pretrained": BASE},
    "medbot_old": {"pretrained": BASE, "peft": "models/medbot_qwen3_4b/lora_adapter"},
    "medbot_full": {"pretrained": BASE, "peft": "models/medbot_train_full/lora_adapter"},
}
SUITES = {
    "medquad": ["medquad_en", "medquad_ru", "medquad_kk"],
    "medquad_rag": ["medquad_rag_en", "medquad_rag_ru", "medquad_rag_kk"],
    "medquad_raghidden": ["medquad_raghidden_en", "medquad_raghidden_ru", "medquad_raghidden_kk"],
    "heldout": ["medquad_heldout_en", "medquad_heldout_ru", "medquad_heldout_kk"],
    "heldout_rag": ["medquad_heldout_rag_en", "medquad_heldout_rag_ru", "medquad_heldout_rag_kk"],
    "heldout_raghidden": ["medquad_heldout_raghidden_en", "medquad_heldout_raghidden_ru",
                          "medquad_heldout_raghidden_kk"],
    "bench": ["medqa_4options", "medmcqa", "pubmedqa", "mmlu_anatomy", "mmlu_clinical_knowledge",
              "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine"],
    "ru_kk": ["mmedbench_ru", "kazmmlu_medicine_ru", "kazmmlu_biology_ru", "kazmmlu_biology_kk"],
}
GENERATIVE = ("medquad", "medquad_rag", "medquad_raghidden", "heldout", "heldout_rag", "heldout_raghidden")


def command(model: str, suite: str, limit: int | None, batch_size: str, device: str,
            num_fewshot: int | None = None) -> list[str]:
    args = {**MODELS[model], "dtype": "bfloat16"}
    if "peft" in args:
        args["tokenizer"] = args["peft"]  # чат-шаблон и токенизатор сохранены вместе с адаптером
    # enable_thinking=False: Qwen3 отвечает сразу, без блока <think> (как в medqa.medbot.chat_text)
    args["enable_thinking"] = "False"
    cmd = [sys.executable, "-m", "lm_eval", "run",
           "--model", "hf", "--model_args", ",".join(f"{k}={v}" for k, v in args.items()),
           "--tasks", ",".join(SUITES[suite]), "--include_path", str(TASKS_DIR),
           "--device", device, "--batch_size", batch_size,
           "--output_path", str(OUT / model), "--seed", "42", "--log_samples"]  # ответы — в samples_*.jsonl
    if suite in GENERATIVE:
        cmd += ["--apply_chat_template"]
    if limit:
        cmd += ["--limit", str(limit)]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODELS, default=["medbot_full"])
    p.add_argument("--suite", choices=SUITES, default="medquad")
    p.add_argument("--limit", type=int, default=50, help="строк на задачу (0 — все)")
    p.add_argument("--batch-size", default="1", help="для bench можно auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-fewshot", type=int, default=None, help="примеры с ответами в промпте (для bench / ru_kk)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    for model in a.models:
        cmd = command(model, a.suite, a.limit or None, a.batch_size, a.device, a.num_fewshot)
        print(f"\n=== {model} / {a.suite} ===\n" + subprocess.list2cmdline(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
