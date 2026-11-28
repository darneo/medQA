# medqa — медицинский ассистент на трёх языках (EN / RU / KK)

Вопросы и ответы о здоровье по датасету **MedQuAD**
(14 368 пар вопрос–ответ × 3 языка; русский и казахский — машинный перевод). Три модели:

- **medbot** — Qwen3-4B, дообученный методом **QLoRA** (модель: [dARYNel/qwen3-4b-medQA-kklang](https://huggingface.co/dARYNel/qwen3-4b-medQA-kklang));
- **Saluki 27B** — готовая модель без обучения (GGUF, 2 бита) через llama.cpp, с **RAG** по датасету или без;
- **Hopfield-QA** — современная сеть Хопфилда поверх bge-m3: не генерирует, а вспоминает готовый ответ из датасета.

## medbot

| | |
|---|---|
| базовая модель | `unsloth/Qwen3-4B-unsloth-bnb-4bit` (4 бита) |
| метод | QLoRA, r=16, alpha=16, attention + MLP, 1 эпоха |
| данные | весь датасет: 14 368 × 3 примера (сначала 12 931 × 3, затем дообучение на оставшихся 1 437 × 3) |
| запуск | Unsloth, GPU NVIDIA от 6–8 ГБ |
| результат | loss: 0.627 на отложенных строках до дообучения; после дообучения на них train loss 0.72 |
| Hugging Face | объединённая 16-битная модель + адаптер в `lora_adapter/` |

Подробности обучения — [docs/REPORT_medbot.txt](docs/REPORT_medbot.txt).
**Модель — эксперимент: медицинские ответы перед использованием проверяет врач.**

## Структура

```
medqa/                 пакет с кодом (одна команда: python -m medqa ...)
  config.py            пути и константы
  data.py              загрузка датасета (validation-выборки нет: всё идёт в обучение и поиск)
  prompts.py           системные подсказки (на них обучен адаптер)
  text.py              очистка ответов: <think>, номер 911 -> 103/112, язык ответа
  medbot.py            обучение, чат, проверка, loss и слияние адаптеров
  saluki.py            Saluki 27B: запуск llama-server и запросы к нему
  rag.py               RAG: индекс эмбеддингов bge-m3 по всему датасету и поиск похожих вопросов
  hopfield.py          Hopfield-QA: современная сеть Хопфилда — память ответов, обучение, проверка
  evaluation.py        метрики сравнения: язык, ROUGE-L, chrF, обрывы, повторы, совет к врачу
  cli.py               команды python -m medqa
data/
  final/qa_en_ru_kk.parquet   датасет en / ru / kk
  new_questions.json          20 новых вопросов на трёх языках (нет в датасете)
  rag_index/                  индекс RAG (собирается командой, в git не хранится)
  hopfield_cache/             память Хопфилда (собирается командой, в git не хранится)
models/
  medbot_qwen3_4b/lora_adapter/   конфиг и токенизатор адаптера
  hopfield/                       обученная сеть Хопфилда (W_Q, W_K, β; ~4 МБ) и отчёт обучения
  saluki/                         сюда скачивается GGUF Saluki (7.9 ГБ)
results/
  medbot/                 сравнение medbot с Qwen3-4B, контрольный тест
  saluki/                 сравнение Saluki без RAG и с RAG
  hopfield/               проверка Hopfield-QA по всему датасету
  all_models/             все модели на одних и тех же случайных вопросах датасета
notebooks/
  compare_base_vs_medbot.ipynb   medbot против обычной Qwen3-4B
  compare_saluki_rag.ipynb       Saluki без RAG против Saluki с RAG
docs/REPORT_medbot.txt    отчёт об обучении (7 ноября 2026)
harness/tasks/medquad/   задачи MedQuAD en / ru / kk для lm-evaluation-harness
scripts/                  установка на GPU, обучение по шардам, слияние адаптера, упаковка, run_harness.py
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
Сводка по датасету: сколько строк, длина ответов.

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
Loss на датасете по языкам (только по токенам ответа); `--rows N` — первые N строк, `--unseen-only` — только
строки, которых не было в первом обучении.

```bash
python -m medqa medbot train
```
Обучение нового адаптера на всём датасете в `models/medbot_train/` (готовый адаптер не перезаписывается).

Дообучить готовый адаптер на строках, которых он ещё не видел (так получена текущая версия):

```bash
python -m medqa medbot train --init-adapter models/medbot_qwen3_4b/lora_adapter --rows unseen --output models/medbot_train_full --lr 1e-4
.venv\Scripts\python.exe scripts\merge_adapter.py --adapter models/medbot_train_full/lora_adapter --out models/medbot_merged_full_16bit
```

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

**Как устроен RAG** (`medqa/rag.py`): каждая пара «вопрос + начало ответа» из **всего датасета** (43 104 пары)
превращается моделью bge-m3 в вектор смысла; для вопроса ищутся 3 пары того же языка с наибольшим сходством,
их ответы вставляются в системную подсказку. На вопросы из датасета RAG находит и сам эталонный ответ.

**Сравнение:** [notebooks/compare_saluki_rag.ipynb](notebooks/compare_saluki_rag.ipynb) — одни и те же
126 вопросов Saluki без RAG и с RAG: метрики, качество поиска, срочные вопросы, примеры, слепая таблица для врача.

## Hopfield-QA (сеть Хопфилда)

Третья модель — **ассоциативная память**: не генерирует текст, а «вспоминает» готовый ответ из датасета.
Основа — современная сеть Хопфилда (Ramsauer et al., 2020, «Hopfield Networks is All You Need»).

- **Память:** все пары датасета на трёх языках (43 104) → векторы bge-m3 (образы K).
- **Динамика:** вопрос → вектор ξ; шаг обновления ξ ← норм(Kᵀ·softmax(β·Kξ)) притягивает запрос к аттрактору.
  Итоговое p = softmax(β·Kξ) — вероятность каждого ответа; max p — **уверенность** (ниже порога — отказ).
- **Тип вопроса.** В датасете колонка `qtype` пустая («Unknown»), поэтому тип (симптомы, лечение, причины,
  наследование, …) определяется по шаблону английского вопроса (`medqa.data.question_type`) и общий для трёх языков.
- **Память — пары «вопрос + ответ»:** образ = нормированная сумма векторов вопроса и ответа, поэтому тип вопроса
  виден сети (раньше образом был только ответ, и сеть путала «симптомы» с «лечением»).
- **Обучение:** поверх bge-m3 обучаются проекции W_Q, W_K и β (старт — единичные матрицы) на всех вопросах
  датасета. Правильный ответ — та же болезнь **и** тот же тип вопроса. В половине примеров свой ответ скрыт
  (учит находить болезнь для новых формулировок), в другой половине — в памяти. 5 эпох, ~1 мин на GPU.

```bash
python -m medqa hopfield build                          # память: эмбеддинги (~5 минут на GPU)
python -m medqa hopfield train                          # обучение → models/hopfield/
python -m medqa hopfield ask "Что такое астма?" --lang ru
python -m medqa hopfield eval                           # проверка по всему датасету → results/hopfield/
python -m medqa saluki chat --lang ru --rag hopfield    # Хопфилд как поиск для RAG
```

**Проверка** (`hopfield eval`) — все 43 104 вопроса датасета в двух режимах. Отдельной validation нет, поэтому
обученная сеть видела эти вопросы при обучении.

| вариант | exact@1 | type@1 | focus@1, свой скрыт | focus@3, свой скрыт |
|---|---|---|---|---|
| RAG (косинус по «вопрос + 600 символов ответа») | 0.881 | 0.922 | 0.607 | 0.749 |
| косинус по ответам | 0.608 | 0.647 | 0.608 | 0.755 |
| косинус по парам «вопрос + ответ» | **0.957** | 0.989 | 0.657 | 0.790 |
| Хопфилд без обучения (β = 50, 1 шаг) | 0.599 | 0.638 | 0.575 | 0.666 |
| **обученный Хопфилд** (пары, болезнь + тип) | **0.957** | **0.990** | **0.820** | **0.836** |

- **exact@1** — на вопрос из датасета вернулся ровно его эталонный ответ; **type@1** — ответ про ту же болезнь
  и того же типа (свой ответ в памяти); **focus@1** — ответ про ту же болезнь, когда свой ответ скрыт
  (максимум 0.843: у остальных болезнь встречается один раз).
- При скрытом своём ответе тот же тип находится редко (type@1 ≈ 0.10): только у ~10 % вопросов в датасете
  есть второй ответ той же болезни и того же типа — сеть в этом не виновата.
- Предыдущая сеть (образ = только ответ, цель «та же болезнь» при всегда скрытом своём ответе) давала
  exact@1 = 0.20: она возвращала ответ про ту же болезнь, но на другой тип вопроса.
- Сравнить с реальными ответами вручную: `python -m medqa hopfield export` →
  `results/hopfield/hopfield_vs_reference.csv` (свой ответ скрыт; `--show-own` — не скрывать).

## Все модели на одних вопросах

`scripts/test_all_models.py` задаёт всем моделям одни и те же 24 случайных вопроса датасета (8 строк × 3 языка)
и сохраняет `results/all_models/answers_side_by_side.csv` — вопрос, эталон и ответы всех моделей в одной строке.

| модель | язык верный | ROUGE-L | chrF (en / ru / kk) | обрезан |
|---|---|---|---|---|
| Хопфилд (свой ответ в памяти) | 100 % | 1.00 | 100 (эталон найден во всех 24) | 0 % |
| Хопфилд (свой ответ скрыт) | 100 % | 0.12 | 26.2 (27 / 28 / 24) | 0 % |
| medbot старый (90 % строк) | 100 % | 0.24 | **36.9** (46 / 37 / 28) | 21 % |
| medbot новый (весь датасет) | 100 % | 0.23 | 35.9 (43 / 35 / 29) | 29 % |
| Saluki 27B + RAG (весь датасет) | 100 % | 0.21 | 31.1 (30 / 30 / **34**) | 0 % |

24 вопроса — мало для выводов: разница между medbot старым и новым в пределах шума (большинство этих строк
старый адаптер тоже видел). RAG нашёл эталон для всех 24 вопросов, но Saluki пересказывает его кратко,
поэтому chrF ниже; зато на казахском Saluki + RAG лучше медботов.

## lm-evaluation-harness

Оценка через стандартный фреймворк [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(EleutherAI): те же метрики считаются одинаково для всех моделей, а результаты можно сравнивать с цифрами из статей.
Установка: `uv pip install --python .venv\Scripts\python.exe "lm-eval[hf]"` (уже есть в `requirements-gpu.txt`).

```powershell
.venv\Scripts\python.exe scripts\run_harness.py --models base medbot_old medbot_full --limit 100
.venv\Scripts\python.exe scripts\run_harness.py --models base medbot_full --suite bench --limit 200
```

- **`--suite medquad`** — свои задачи `harness/tasks/medquad/` (`medquad_en`, `medquad_ru`, `medquad_kk`):
  вопрос датасета → ответ модели с той же системной подсказкой, на которой обучен medbot, без `<think>`;
  метрики ROUGE-L, chrF, язык верный, повторы, совет к врачу (код из `medqa/evaluation.py`). Строки перемешаны
  с seed 42, поэтому `--limit N` даёт всем моделям одни и те же N строк. Все они были в обучении medbot.
- **`--suite bench`** — готовые бенчмарки на английском с выбором ответа (точность `acc`): MedQA-USMLE (4 варианта),
  MedMCQA, PubMedQA, медицинские разделы MMLU. Модели их не видели — это проверка того, не испортило ли
  дообучение общие медицинские знания по сравнению с базовой Qwen3-4B.

Результаты (сводка `results_*.json` и все ответы `samples_*.jsonl`) — в `results/harness/<модель>/`.
Задачи можно запускать и напрямую: `python -m lm_eval run --include_path harness/tasks --tasks medquad ...`.

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
