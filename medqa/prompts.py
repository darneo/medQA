"""Системные подсказки и сборка диалога для medbot.

SYSTEM менять нельзя без переобучения: адаптер medbot обучен именно на них.
"""

from __future__ import annotations

SYSTEM = {
    "en": ("You are a careful medical assistant. Answer in English with general information. "
           "Encourage consulting a qualified clinician for personal diagnosis or treatment."),
    "ru": ("Ты осторожный медицинский ассистент. Отвечай по-русски и давай общую информацию. "
           "При личных симптомах или выборе лечения советуй обратиться к врачу."),
    "kk": ("Сен мұқият медициналық көмекшісің. Қазақ тілінде жалпы ақпарат бер. "
           "Жеке белгілер немесе ем таңдау үшін дәрігерге жүгінуге кеңес бер."),
}


def build_messages(question: str, lang: str, *, answer: str | None = None) -> list[dict]:
    """Диалог для чат-шаблона: system, вопрос, [ответ]."""
    if lang not in SYSTEM:
        raise ValueError(f"Неизвестный язык: {lang}")
    messages = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
