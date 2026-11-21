"""Пути проекта и общие константы. Все остальные модули берут пути только отсюда."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# Путь к датасету можно переопределить переменной окружения (удобно в облаке).
DATASET = Path(os.environ.get("MEDQA_DATASET", DATA_DIR / "final" / "qa_en_ru_kk.parquet"))

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# medbot: Qwen3-4B + LoRA-адаптер
MEDBOT_BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MEDBOT_DIR = MODELS_DIR / "medbot_qwen3_4b"
MEDBOT_ADAPTER = MEDBOT_DIR / "lora_adapter"
MEDBOT_RESULTS = RESULTS_DIR / "medbot"

LANGS: tuple[str, ...] = ("en", "ru", "kk")
LANG_NAMES = {"en": "английский", "ru": "русский", "kk": "казахский"}
SEED = 42
VAL_RATIO = 0.1
