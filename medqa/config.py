"""Пути проекта и общие константы. Все остальные модули берут пути только отсюда."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# Путь к датасету можно переопределить переменной окружения (удобно в облаке).
DATASET = Path(os.environ.get("MEDQA_DATASET", DATA_DIR / "final" / "qa_en_ru_kk.parquet"))
NEW_QUESTIONS = DATA_DIR / "new_questions.json"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
TOOLS_DIR = ROOT / "tools"

# Saluki 27B: GGUF-файл и сборка llama.cpp
SALUKI_REPO = "ConwayResearch/Underdog-Saluki-27B-1.0"
SALUKI_FILE = "Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
SALUKI_SIZE = 7_898_369_152
SALUKI_MODEL = MODELS_DIR / "saluki" / SALUKI_FILE
LLAMA_CPP_DIR = TOOLS_DIR / "llama_cpp"
SALUKI_RESULTS = RESULTS_DIR / "saluki"

# medbot: Qwen3-4B + LoRA-адаптер
MEDBOT_BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MEDBOT_DIR = MODELS_DIR / "medbot_qwen3_4b"
MEDBOT_ADAPTER = MEDBOT_DIR / "lora_adapter"
MEDBOT_RESULTS = RESULTS_DIR / "medbot"

LANGS: tuple[str, ...] = ("en", "ru", "kk")
LANG_NAMES = {"en": "английский", "ru": "русский", "kk": "казахский"}
SEED = 42
VAL_RATIO = 0.1
