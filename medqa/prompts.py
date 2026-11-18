"""Системные подсказки и сборка диалога. Одни и те же тексты для Saluki и medbot.

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

CONCISE = {
    "en": "Be concise: 3-5 sentences, no long headings.",
    "ru": "Отвечай кратко: 3–5 предложений, без длинных заголовков.",
    "kk": "Қысқа жауап бер: 3–5 сөйлем, ұзақ тақырыпшаларсыз.",
}

RAG_INSTRUCTION = {
    "en": "Use the reference information below to answer. If it is not relevant, answer from your own knowledge.",
    "ru": "Используй справочную информацию ниже для ответа. Если она не подходит, отвечай из своих знаний.",
    "kk": ("Жауап беру үшін төмендегі анықтамалық ақпаратты пайдалан. "
           "Егер ол сәйкес келмесе, өз біліміңнен жауап бер."),
}


def build_messages(question: str, lang: str, *, answer: str | None = None, concise: bool = False,
                   context: str | None = None) -> list[dict]:
    """Диалог для чат-шаблона: system (+ справка RAG, + просьба краткости), вопрос, [ответ]."""
    if lang not in SYSTEM:
        raise ValueError(f"Неизвестный язык: {lang}")
    system = SYSTEM[lang]
    if context:
        system += " " + RAG_INSTRUCTION[lang] + "\n\n" + context
    if concise:
        system += " " + CONCISE[lang]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
