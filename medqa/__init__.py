"""medqa — медицинский QA на трёх языках (en / ru / kk) по датасету MedQuAD.

Две модели в одном проекте:
  * Saluki 27B (GGUF через llama.cpp) — только ответы, без обучения, с RAG по датасету;
  * medbot — Qwen3-4B, дообученный LoRA-адаптером на MedQuAD.

Запуск: python -m medqa --help
"""

__version__ = "1.0.0"
