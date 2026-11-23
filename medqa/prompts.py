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


# Для Saluki: просьба отвечать кратко и инструкция к справке RAG.
CONCISE = {
    "en": "Be concise: 3-5 sentences.",
    "ru": "Отвечай кратко: 3–5 предложений.",
    "kk": "Қысқа жауап бер: 3–5 сөйлем.",
}

RAG_INSTRUCTION = {
    "en": ("Below is reference information from a medical database. Use it if it is relevant to the question; "
           "if it is not relevant, answer from your own knowledge. Do not mention the reference itself."),
    "ru": ("Ниже — справочная информация из медицинской базы. Используй её, если она относится к вопросу; "
           "если не относится — отвечай из своих знаний. Не упоминай саму справку."),
    "kk": ("Төменде медициналық базадан алынған анықтамалық ақпарат берілген. Егер ол сұраққа қатысты болса, "
           "соны пайдалан; қатысты болмаса, өз біліміңнен жауап бер. Анықтаманың өзін атама."),
}


def build_messages(question: str, lang: str, *, answer: str | None = None, concise: bool = False,
                   context: str | None = None) -> list[dict]:
    """Диалог для чат-шаблона: system (+ краткость, + справка RAG), вопрос, [ответ].

    medbot вызывает только build_messages(question, lang[, answer]) — его подсказка не меняется.
    """
    if lang not in SYSTEM:
        raise ValueError(f"Неизвестный язык: {lang}")
    system = SYSTEM[lang]
    if concise:
        system += " " + CONCISE[lang]
    if context:
        system += "\n\n" + RAG_INSTRUCTION[lang] + "\n\n" + context
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
