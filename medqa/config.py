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

# Saluki 27B: готовая модель (GGUF, 2 бита), запускается через llama.cpp (llama-server)
SALUKI_REPO = "ConwayResearch/Underdog-Saluki-27B-1.0"
SALUKI_FILE = "Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
SALUKI_MODEL = MODELS_DIR / "saluki" / SALUKI_FILE
LLAMA_CPP_DIR = ROOT / "tools" / "llama_cpp"
SALUKI_RESULTS = RESULTS_DIR / "saluki"

# RAG: индекс эмбеддингов по train-части датасета
RAG_INDEX_DIR = DATA_DIR / "rag_index"
RAG_EMBED_MODEL = "BAAI/bge-m3"

LANGS: tuple[str, ...] = ("en", "ru", "kk")
LANG_NAMES = {"en": "английский", "ru": "русский", "kk": "казахский"}
SEED = 42
VAL_RATIO = 0.1
