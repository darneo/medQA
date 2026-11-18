from medqa.data import col, split_train_val, to_examples
from medqa.prompts import SYSTEM, build_messages
from medqa.text import advises_help, detect_language, localize_emergency, strip_think, tokens


def test_dataset_rows_are_stable(dataset):
    # На этом числе держатся номера row во всех сохранённых ответах и метриках.
    assert len(dataset) == 14368


def test_split_matches_medbot_training(dataset):
    train, val = split_train_val(dataset)
    assert (len(train), len(val)) == (12931, 1437)  # как в отчёте medbot
    assert not set(train["question"]) & set(val["question"])  # переводы не утекают между train и validation
    again_train, again_val = split_train_val(dataset)
    assert again_val["question"].tolist() == val["question"].tolist()


def test_examples_cover_three_languages(dataset):
    examples = to_examples(dataset.head(2))
    assert [e["lang"] for e in examples] == ["en", "ru", "kk", "en", "ru", "kk"]
    assert examples[4]["question"] == dataset.at[1, col("question", "ru")]


def test_build_messages():
    msgs = build_messages("Q?", "kk", answer="A", concise=True, context="CTX")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"].startswith(SYSTEM["kk"]) and "CTX" in msgs[0]["content"]


def test_strip_think():
    assert strip_think("<think>\n\n</think>\n\nОтвет.") == "Ответ."
    assert strip_think("<think>долго думаю</think> Ответ") == "Ответ"
    assert strip_think("остаток мыслей</think>\nОтвет") == "Ответ"
    assert strip_think("Просто ответ") == "Просто ответ"


def test_localize_emergency():
    assert localize_emergency("Позвоните 911.", "ru") == "Позвоните 103 или 112."
    assert localize_emergency("911-ге қоңырау шалыңыз", "kk") == "103 немесе 112-ге қоңырау шалыңыз"
    assert localize_emergency("Call 911.", "en") == "Call 911."
    assert localize_emergency("в 1911 году", "ru") == "в 1911 году"


def test_detect_language_and_help():
    assert detect_language("Диабет дегеніміз не?") == "kk"
    assert detect_language("Что такое астма?") == "ru"
    assert detect_language("What is asthma?") == "en"
    # старый тест писал «NO» на такой безопасный ответ
    assert advises_help("Немедленно позвоните 911.")
    assert not advises_help("Пейте больше воды.")
    assert tokens("Қант — энергия!") == ["қант", "энергия"]
