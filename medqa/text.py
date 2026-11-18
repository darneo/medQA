"""Работа с текстом ответов: токены, язык, очистка, безопасность."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
KK_LETTERS = set("әғқңөұүһі")

# Номер 911 работает в США. В Казахстане и России скорая — 103, единый номер — 112.
EMERGENCY_NUMBERS = {"ru": "103 или 112", "kk": "103 немесе 112"}
_911_RE = re.compile(r"(?<!\d)911(?!\d)")


def tokens(text: str) -> list[str]:
    """Слова в нижнем регистре; работает для латиницы, кириллицы и казахских букв."""
    return TOKEN_RE.findall(str(text).lower())


def strip_think(text: str) -> str:
    """Убирает блоки рассуждений Qwen3 (<think>...</think>), в том числе пустые и незакрытые в начале."""
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # начало блока могло остаться в промпте
        text = text.split("</think>", 1)[1]
    return text.strip()


def localize_emergency(text: str, lang: str) -> str:
    """Заменяет американский 911 на местные номера экстренной помощи для ru / kk."""
    number = EMERGENCY_NUMBERS.get(lang)
    return _911_RE.sub(number, text) if number else text


def detect_language(text: str) -> str:
    """Быстрое определение языка по буквам: казахские буквы -> kk, кириллица -> ru, иначе en."""
    lower = str(text).lower()
    if any(ch in KK_LETTERS for ch in lower):
        return "kk"
    if any("а" <= ch <= "я" or ch == "ё" for ch in lower):
        return "ru"
    return "en"


EMERGENCY_HINTS = ("103", "112", "911", "скор", "экстрен", "врач", "медицин", "дәрігер", "жедел",
                   "emergency", "doctor", "ambulance")


def advises_help(text: str) -> bool:
    """Советует ли ответ обратиться за помощью (врач, скорая, экстренный номер)."""
    lower = str(text).lower()
    return any(hint in lower for hint in EMERGENCY_HINTS)
