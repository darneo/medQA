# medqa — медицинский ассистент на трёх языках (EN / RU / KK)

Вопросы и ответы о здоровье по датасету **MedQuAD**
(14 368 пар вопрос–ответ × 3 языка; русский и казахский — машинный перевод). Две модели:

- **medbot** — Qwen3-4B, дообученный методом **QLoRA** (модель: [dARYNel/qwen3-4b-medQA-kklang](https://huggingface.co/dARYNel/qwen3-4b-medQA-kklang));
- **Saluki 27B** — готовая модель без обучения (GGUF, 2 бита) через llama.cpp, с **RAG** по датасету или без.

## medbot

| | |
|---|---|
| базовая модель | `unsloth/Qwen3-4B-unsloth-bnb-4bit` (4 бита) |
| метод | QLoRA, r=16, alpha=16, attention + MLP, 1 эпоха |
| данные | 12 931 × 3 примера для обучения, 1 437 × 3 для проверки (переводы не пересекаются) |
| запуск | Unsloth, GPU NVIDIA от 6–8 ГБ |
| результат | validation loss 0.627, язык ответа 6/6 |

Подробности обучения — [docs/REPORT_medbot.txt](docs/REPORT_medbot.txt).
**Модель — эксперимент: медицинские ответы перед использованием проверяет врач.**

## Структура

```
medqa/                 пакет с кодом (одна команда: python -m medqa ...)
  config.py            пути и константы
  data.py              загрузка датасета, разбиение train/validation без утечки переводов
  prompts.py           системные подсказки (на них обучен адаптер)
  text.py              очистка ответов: <think>, номер 911 -> 103/112, язык ответа
  medbot.py            обучение, чат, проверка, loss и слияние адаптеров
  saluki.py            Saluki 27B: запуск llama-server и запросы к нему
  rag.py               RAG: индекс эмбеддингов bge-m3 по train-части и поиск похожих вопросов
  evaluation.py        метрики сравнения: язык, ROUGE-L, chrF, обрывы, повторы, совет к врачу
  cli.py               команды python -m medqa
data/
  final/qa_en_ru_kk.parquet   датасет en / ru / kk
  new_questions.json          20 новых вопросов на трёх языках (нет в датасете)
  rag_index/                  индекс RAG (собирается командой, в git не хранится)
models/
  medbot_qwen3_4b/lora_adapter/   конфиг и токенизатор адаптера
  saluki/                         сюда скачивается GGUF Saluki (7.9 ГБ)
results/
  medbot/                 сравнение medbot с Qwen3-4B, контрольный тест
  saluki/                 сравнение Saluki без RAG и с RAG
notebooks/
  compare_base_vs_medbot.ipynb   medbot против обычной Qwen3-4B
  compare_saluki_rag.ipynb       Saluki без RAG против Saluki с RAG
docs/REPORT_medbot.txt    отчёт об обучении (7 ноября 2026)
scripts/                  установка на GPU, обучение по шардам, слияние адаптера, упаковка
tools/llama_cpp/          сюда кладётся сборка llama.cpp (llama-server.exe)
```

Веса адаптера `adapter_model.safetensors` (~132 МБ) больше лимита GitHub и в репозитории не хранятся —
их нужно положить в `models/medbot_qwen3_4b/lora_adapter/`.

## Установка

Windows с видеокартой NVIDIA — одной командой (uv, Python 3.11, PyTorch с CUDA):

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

### Что дало обучение

[notebooks/compare_base_vs_medbot.ipynb](notebooks/compare_base_vs_medbot.ipynb) задаёт одни и те же вопросы
обычной Qwen3-4B и medbot (validation, новые и срочные вопросы), сравнивает язык ответа, ROUGE-L, chrF,
обрывы и повторы и сохраняет слепую таблицу `results/medbot/doctor_sheet.csv` для оценки врачом.
Нужна GPU и файл весов адаптера.

## Saluki 27B и RAG

Saluki — готовая модель (основа Qwen3.8-27B, лицензия **CC BY-NC 4.0** — только некоммерческое использование),
запускается через llama.cpp. На 8 ГБ видеопамяти часть слоёв считается в ОЗУ: ~15 токенов/с на RTX 5060.

1. Скачать модель (~7.9 ГБ):
   ```bash
   python -m medqa saluki download
   ```
2. Положить сборку llama.cpp с CUDA в `tools/llama_cpp/` — архивы `llama-bXXXX-bin-win-cuda-13.x-x64.zip`
   и `cudart-llama-bin-win-cuda-13.x-x64.zip` со страницы
   [релизов llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (для RTX 50xx нужна CUDA 13).
3. Собрать индекс RAG (один раз, ~4 минуты на GPU; модель эмбеддингов bge-m3 скачается сама):
   ```bash
   python -m medqa rag build
   ```
4. Чат:
   ```bash
   python -m medqa saluki chat --lang ru          # только модель
   python -m medqa saluki chat --lang kk --rag    # модель + справка из датасета
   ```

Другие команды: `python -m medqa saluki serve` (сервер с OpenAI API на :8080),
`python -m medqa rag search "Что такое астма?" --lang ru` (что найдёт RAG).

**Как устроен RAG** (`medqa/rag.py`): каждая пара «вопрос + начало ответа» из **train** превращается
моделью bge-m3 в вектор смысла; для вопроса ищутся 3 пары того же языка с наибольшим сходством, их ответы
вставляются в системную подсказку. Validation в индекс не входит, поэтому при проверке модель не видит эталон.

**Сравнение:** [notebooks/compare_saluki_rag.ipynb](notebooks/compare_saluki_rag.ipynb) — одни и те же
126 вопросов Saluki без RAG и с RAG: метрики, качество поиска, срочные вопросы, примеры, слепая таблица для врача.

### Перенос на другой компьютер

`.\scripts\pack_save.ps1` собирает код, датасет, адаптер и последний чекпоинт в `medbot_save.zip`.

## Что исправлено по сравнению с первыми скриптами

- **Номер экстренной помощи.** На «боль в груди» medbot советовал **911** (США). Теперь для ru/kk номер
  заменяется на **103 или 112**.
- **Ответы без `<think></think>` и без зацикливания.** Qwen3 запускается с `enable_thinking=False`,
  при генерации добавлен штраф за повторы.
- **Слияние адаптеров шардов.** Раньше усреднялись матрицы A и B по отдельности, а это не среднее
  поправок. Теперь слияние точное: A складываются стопкой, B — рядом с весом 1/N, `lora_alpha`
  пересчитывается. Старый способ доступен как `--method mean`.
- **Честный loss.** Оценка идёт на той же длине (1024), что и обучение, и только по токенам ответа.
  Старый расчёт по всему диалогу — `--include-prompt`. Новое обучение тоже считает loss только по ответу.
