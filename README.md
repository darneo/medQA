# medqa — медицинский ассистент на трёх языках (EN / RU / KK)

Вопросы и ответы о здоровье на английском, русском и казахском по датасету **MedQuAD**
(14 368 пар вопрос–ответ × 3 языка; русский и казахский — машинный перевод).

В проекте две модели:

| | **Saluki 27B** | **medbot** |
|---|---|---|
| что это | готовая модель Qwen3.8-27B, 2-битный GGUF | Qwen3-4B + LoRA-адаптер, дообученный на MedQuAD |
| обучение | нет, только ответы (можно подсказывать похожие ответы из датасета — RAG) | QLoRA, 1 эпоха, 12 931 × 3 примеров |
| запуск | llama.cpp (`llama-server`), GPU от 8 ГБ или CPU | Unsloth, GPU NVIDIA от 6–8 ГБ |
| результат | язык ответа 100%, ~8 с на краткий ответ (RTX 5060) | validation loss 0.627, язык 6/6 |

Подробные цифры и графики — в [notebooks/results.ipynb](notebooks/results.ipynb).
**Обе модели — эксперимент: медицинские ответы перед использованием проверяет врач.**

## Структура

```
medqa/                 пакет с кодом (одна команда: python -m medqa ...)
  config.py            все пути и константы проекта
  data.py              загрузка датасета, разбиение train/validation без утечки переводов
  prompts.py           системные подсказки (общие для обеих моделей)
  text.py              очистка ответов: <think>, номер 911 -> 103/112, язык ответа
  rag.py               поиск похожих вопросов в датасете (TF-IDF)
  metrics.py           ROUGE, BLEU, chrF, покрытие эталона, язык ответа
  llama_server.py      запуск llama-server и запрос к нему
  saluki.py            прогон вопросов через Saluki с продолжением после остановки
  medbot.py            обучение, чат, проверка, loss и слияние адаптеров medbot
  cli.py               команды python -m medqa
data/
  final/qa_en_ru_kk.parquet   датасет
  new_questions.json          20 новых вопросов (нет в датасете)
  embeddings_rag/             сохранённый поиск по эмбеддингам (bge-base): qa.parquet + векторы
models/
  medbot_qwen3_4b/lora_adapter/   обученный адаптер medbot (137 МБ)
  saluki/                         сюда скачивается GGUF Saluki (7.9 ГБ)
results/
  saluki/   ответы (answers*.jsonl/.parquet) и метрики (metrics*.json/.parquet)
  medbot/   validation loss, контрольный тест
notebooks/results.ipynb   все результаты с графиками
docs/REPORT_medbot.txt    отчёт об обучении medbot (7 ноября 2026)
scripts/                  установка на GPU-машине, обучение по шардам, упаковка, облако
tests/                    тесты (pytest), модели и GPU не нужны
tools/llama_cpp/          сюда кладётся сборка llama.cpp (llama-server.exe)
```

## Установка

```bash
python -m pip install -r requirements.txt
```

Для medbot на Windows с видеокартой NVIDIA — одной командой (uv, Python 3.11, PyTorch с CUDA):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\setup_gpu.ps1
```

## Команды

Все команды запускаются из корня проекта; `--help` у любой команды показывает все параметры.

```bash
python -m medqa data
```
Сводка по датасету: сколько строк, train/validation, длина ответов.

```bash
python -m medqa rag "Что такое глаукома?" --lang ru
```
Похожие вопросы из датасета.

### Saluki 27B

```bash
python -m medqa saluki download
```
Скачать модель (~7.9 ГБ) в `models/saluki/`. Сборку llama.cpp положите в `tools/llama_cpp/` или в PATH.

```bash
python -m medqa saluki run --limit 50 --langs ru
```
Ответить на 50 вопросов датасета. Повторный запуск продолжает с места остановки. Ответы пишутся в
`results/saluki/answers.jsonl` и `.parquet`. Полезные флаги:
- `--rag` — подсказывать модели 3 похожих ответа (сам вопрос из поиска исключён) → `answers_rag.jsonl`;
- `--no-concise` — длинные ответы вместо 3–5 предложений → `answers_long.jsonl`;
- `--shuffle` — случайные вопросы вместо первых;
- `--server-url http://127.0.0.1:8080` — использовать уже запущенный сервер.

Дописать в файл ответы другого режима (например, `--rag` в файл без RAG) команда не даст.

```bash
python -m medqa saluki ask --rag
```
Новые вопросы из `data/new_questions.json`.

```bash
python -m medqa evaluate results/saluki/answers.parquet
```
Метрики против эталонов → `results/saluki/metrics.json`.

### medbot (нужна GPU NVIDIA)

```bash
python -m medqa medbot chat --lang ru
```
Чат в консоли.

```bash
python -m medqa medbot test --save
```
6 контрольных вопросов: язык ответа и совет обратиться за помощью.

```bash
python -m medqa medbot eval --save
```
Loss на validation по языкам (только по токенам ответа).

```bash
python -m medqa medbot train
```
Дообучение заново в `models/medbot_train/` (готовый адаптер не перезаписывается).

Обучение на нескольких машинах: на каждой `.\scripts\train_shard.ps1 -Shard N -Total 3`, затем слияние:

```bash
python -m medqa medbot merge models/medbot_train_shard0/lora_adapter models/medbot_train_shard1/lora_adapter models/medbot_train_shard2/lora_adapter --out models/medbot_merged/lora_adapter
```

### Облако

`bash scripts/saluki_cloud.sh` на Ubuntu с NVIDIA GPU 24+ ГБ сам скачает llama.cpp и модель и запустит
прогон (`LIMIT=200 LANGS="ru kk" RAG=1 bash scripts/saluki_cloud.sh`). Аренда GPU платная.

### Перенос на другой компьютер

`.\scripts\pack_save.ps1` собирает код, датасет, адаптер и последний чекпоинт в `medbot_save.zip`.

## Тесты

```bash
python -m pip install pytest safetensors
```
```bash
python -m pytest
```
30 тестов: данные, RAG, метрики (воспроизводят сохранённые результаты), Saluki через фейковый сервер,
маски loss и слияние адаптеров medbot, команды. Модели и видеокарта не нужны.

## Что исправлено по сравнению с первыми скриптами

- **Saluki и medbot используют общий код.** Подсказки, загрузка датасета и разбиение на train/validation
  раньше были скопированы в каждый скрипт medbot и в скрипты Saluki, теперь они в одном месте.
- **Номер экстренной помощи.** На «боль в груди» medbot советовал **911** (США). Теперь для ru/kk номер
  заменяется на **103 или 112**. Проверка безопасности засчитывает такой ответ, раньше она писала «NO».
- **Ответы без `<think></think>` и без зацикливания.** Qwen3 запускается с `enable_thinking=False`,
  при генерации добавлен штраф за повторы.
- **Слияние адаптеров шардов.** Раньше усреднялись матрицы A и B по отдельности, а это не среднее
  поправок. Теперь слияние точное: A складываются стопкой, B — рядом с весом 1/N, `lora_alpha`
  пересчитывается. Старый способ доступен как `--method mean`.
- **Честный loss.** Оценка теперь идёт на той же длине (1024), что и обучение, и только по токенам ответа.
  Старый расчёт по всему диалогу — `--include-prompt`. Новое обучение тоже считает loss только по ответу.
- **Режимы Saluki не смешиваются.** Ответы с RAG и без RAG больше не могут молча смешаться в одном
  файле; ошибки не занижают средние метрики.
