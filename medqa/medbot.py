"""medbot: Qwen3-4B + LoRA-адаптер, дообученный на MedQuAD (en/ru/kk).

Обучение, генерация, чат, быстрая проверка, оценка потерь и слияние адаптеров.
unsloth и torch импортируются внутри функций: остальной проект работает без них.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .config import LANGS, MEDBOT_BASE_MODEL, MODELS_DIR, SEED
from .data import load_dataset, split_train_val, to_examples
from .prompts import build_messages
from .text import advises_help, detect_language, localize_emergency, strip_think

# ---------------------------------------------------------------- модель и генерация


def load_model(name: str | Path, max_seq_length: int = 1024, for_inference: bool = True):
    """Загружает базовую модель или адаптер (папку lora_adapter) в 4 бита через Unsloth."""
    looks_local = isinstance(name, Path) or Path(name).is_absolute() or str(name).startswith((".", "models"))
    if looks_local and not Path(name).exists():
        raise FileNotFoundError(f"Адаптер не найден: {name}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Нужна видеокарта NVIDIA с CUDA: в 4 бита модель на CPU не запускается.")
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(model_name=str(name), max_seq_length=max_seq_length,
                                                 load_in_4bit=True, full_finetuning=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if for_inference:
        FastModel.for_inference(model)
    return model, tokenizer


def chat_text(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    """Текст по чат-шаблону. enable_thinking=False: Qwen3 отвечает сразу, без блока рассуждений."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False)


def generate(model, tokenizer, question: str, lang: str, *, max_new_tokens: int = 512, temperature: float = 0.7,
             top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
    """Ответ модели на один вопрос: без <think>, с местными номерами экстренной помощи.

    repetition_penalty гасит зацикливание (раньше фрагменты ответа повторялись по 3–4 раза).
    temperature=0 — детерминированный ответ.
    """
    import torch

    prompt = chat_text(tokenizer, build_messages(question, lang), add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    sample = temperature > 0
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sample,
                             temperature=temperature if sample else None, top_p=top_p if sample else None,
                             repetition_penalty=repetition_penalty,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return localize_emergency(strip_think(text), lang)


SMOKE_QUESTIONS = [
    ("en", "What are the symptoms of type 2 diabetes?"),
    ("en", "How is high blood pressure treated?"),
    ("ru", "Что такое астма и какие у неё симптомы?"),
    ("ru", "У меня сильная боль в груди, что мне делать?"),
    ("kk", "Диабет дегеніміз не?"),
    ("kk", "Жоғары қан қысымын қалай емдейді?"),
]
URGENT_MARKERS = ("боль в груди", "chest pain", "кеуде")


def smoke_test(model, tokenizer, questions=SMOKE_QUESTIONS, **gen_kwargs) -> dict:
    """Быстрая проверка: язык ответа и совет обратиться за помощью на срочных вопросах."""
    results = []
    for lang, question in questions:
        answer = generate(model, tokenizer, question, lang, **gen_kwargs)
        urgent = any(m in question.lower() for m in URGENT_MARKERS)
        results.append({"lang": lang, "question": question, "answer": answer,
                        "detected": detect_language(answer), "language_ok": detect_language(answer) == lang,
                        "urgent": urgent, "advises_help": advises_help(answer) if urgent else None})
    return {"results": results, "language_ok": sum(r["language_ok"] for r in results), "total": len(results)}


# ---------------------------------------------------------------- оценка потерь


def _answer_only_labels(tokenizer, question: str, answer: str, lang: str, max_len: int, include_prompt: bool):
    """input_ids и labels: в labels токены подсказки и вопроса заменены на -100 (не учитываются в loss)."""
    prompt = chat_text(tokenizer, build_messages(question, lang), add_generation_prompt=True)
    full = chat_text(tokenizer, build_messages(question, lang, answer=answer), add_generation_prompt=False)
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    labels = list(full_ids)
    if not include_prompt:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        n = 0  # длина общего префикса (на случай разной токенизации стыка)
        while n < min(len(prompt_ids), len(full_ids)) and prompt_ids[n] == full_ids[n]:
            n += 1
        labels[:n] = [-100] * n
    return full_ids, labels


def evaluate_loss(model, tokenizer, examples: list[dict], max_seq_length: int = 1024,
                  include_prompt: bool = False) -> dict:
    """Средний loss на токен по каждому языку и в целом.

    По умолчанию loss считается только по токенам ответа — так видно, насколько
    хорошо модель предсказывает ответы, а не повторяет вопрос. include_prompt=True
    воспроизводит старый расчёт (loss по всему диалогу).
    """
    import torch

    model.eval()
    sums = {lang: [0.0, 0, 0] for lang in LANGS}  # loss·токены, токены, примеры
    for ex in examples:
        ids, labels = _answer_only_labels(tokenizer, ex["question"], ex["answer"], ex["lang"], max_seq_length,
                                          include_prompt)
        n_tokens = sum(1 for x in labels[1:] if x != -100)  # loss считается со сдвигом на 1
        if n_tokens == 0:
            continue
        with torch.no_grad():
            loss = model(input_ids=torch.tensor([ids], device=model.device),
                         labels=torch.tensor([labels], device=model.device)).loss.item()
        acc = sums[ex["lang"]]
        acc[0] += loss * n_tokens
        acc[1] += n_tokens
        acc[2] += 1
    result = {lang: {"loss": round(s / t, 4), "samples": n, "tokens": t}
              for lang, (s, t, n) in sums.items() if t}
    total_tokens = sum(r["tokens"] for r in result.values())
    result["overall"] = round(sum(r["loss"] * r["tokens"] for r in result.values()) / total_tokens, 4)
    result["mode"] = "диалог целиком" if include_prompt else "только ответ"
    return result


def validation_examples(val_rows: int | None = None) -> list[dict]:
    """Примеры validation (те же 1437 строк × 3 языка, что не участвовали в обучении)."""
    _, val = split_train_val(load_dataset())
    if val_rows:
        val = val.head(val_rows)
    return to_examples(val)


# ---------------------------------------------------------------- обучение


@dataclass
class TrainOptions:
    output: Path = MODELS_DIR / "medbot_train"
    base_model: str = MEDBOT_BASE_MODEL
    shard: int = 0
    shards: int = 1
    max_train_rows: int = 0
    epochs: float = 1.0
    lr: float = 2e-4
    max_seq_length: int = 1024
    batch_size: int = 2
    grad_accum: int = 4
    lora_r: int = 16
    save_steps: int = 500
    responses_only: bool = True


def _sft_kwargs(SFTConfig, SFTTrainer, max_seq_length: int, tokenizer) -> tuple[dict, dict]:
    """Совместимость версий TRL: max_seq_length -> max_length, tokenizer= -> processing_class=."""
    seq_key = "max_length" if "max_length" in inspect.signature(SFTConfig).parameters else "max_seq_length"
    tok_key = ("processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
               else "tokenizer")
    return {seq_key: max_seq_length}, {tok_key: tokenizer}


def train(opts: TrainOptions) -> Path:
    """QLoRA-дообучение Qwen3-4B. Чекпоинты каждые save_steps шагов, повторный запуск продолжает с последнего."""
    import random

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    if not 0 <= opts.shard < opts.shards:
        raise ValueError("--shard должен быть от 0 до --shards - 1")
    out = Path(opts.output)
    if opts.shards > 1:
        out = out.with_name(f"{out.name}_shard{opts.shard}")
    adapter_dir, ckpt_dir = out / "lora_adapter", out / "checkpoints"
    if adapter_dir.exists() and not ckpt_dir.exists():
        raise FileExistsError(f"В {out} уже есть готовый адаптер. Укажите другую папку --output.")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения нужна видеокарта NVIDIA с CUDA.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM: {props.total_memory / 1024**3:.1f} ГБ", flush=True)

    train_df, val_df = split_train_val(load_dataset(verbose=True))
    if opts.max_train_rows and len(train_df) > opts.max_train_rows:
        train_df = train_df.head(opts.max_train_rows)
    if opts.shards > 1:
        train_df = train_df.iloc[opts.shard::opts.shards].reset_index(drop=True)
        print(f"Шард {opts.shard} из {opts.shards}", flush=True)
    print(f"Train: {len(train_df)} строк × 3 языка; validation: {len(val_df)} строк × 3 языка", flush=True)

    model, tokenizer = load_model(opts.base_model, opts.max_seq_length, for_inference=False)

    def texts(frame):
        return Dataset.from_list([
            {"text": chat_text(tokenizer, build_messages(e["question"], e["lang"], answer=e["answer"]), False)}
            for e in to_examples(frame)])

    train_ds, val_ds = texts(train_df), texts(val_df)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=opts.lora_r, lora_alpha=opts.lora_r, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED, use_rslora=False)
    model.print_trainable_parameters()

    seq_kw, tok_kw = _sft_kwargs(SFTConfig, SFTTrainer, opts.max_seq_length, tokenizer)
    args = SFTConfig(output_dir=str(ckpt_dir), dataset_text_field="text", per_device_train_batch_size=opts.batch_size,
                     per_device_eval_batch_size=opts.batch_size, gradient_accumulation_steps=opts.grad_accum,
                     num_train_epochs=opts.epochs, learning_rate=opts.lr, warmup_ratio=0.03, logging_steps=10,
                     eval_strategy="epoch", save_strategy="steps", save_steps=opts.save_steps, save_total_limit=2,
                     optim="adamw_8bit", gradient_checkpointing=True, seed=SEED, report_to="none",
                     dataset_num_proc=1, **seq_kw)
    trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=args, **tok_kw)
    if opts.responses_only:
        # loss только по ответу ассистента: повторять системную подсказку и вопрос модели не нужно
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                          response_part="<|im_start|>assistant\n")
        print("Loss считается только по ответам ассистента.", flush=True)

    resume = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if resume:
        print("Продолжаю с чекпоинта:", resume, flush=True)
    print(trainer.train(resume_from_checkpoint=resume).metrics, flush=True)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = trainer.evaluate()
    (out / "validation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Адаптер сохранён:", adapter_dir.resolve(), flush=True)
    return adapter_dir


# ---------------------------------------------------------------- слияние адаптеров


def merge_adapters(adapter_dirs: list[Path | str], out: Path | str, method: str = "exact") -> Path:
    """Сливает LoRA-адаптеры шардов в один.

    exact — точное усреднение поправок: ΔW = (1/N)·Σ Bᵢ·Aᵢ. Матрицы A складываются
            стопкой, B — рядом с весом 1/N; ранг становится N·r, а lora_alpha
            пересчитывается так, чтобы масштаб alpha/r не изменился.
    mean  — старый способ: среднее A и среднее B по отдельности. Это НЕ среднее
            поправок (mean(B)·mean(A) ≠ mean(B·A)), оставлено для сравнения.
    """
    import torch
    from safetensors.torch import load_file, save_file

    dirs = [Path(d) for d in adapter_dirs]
    if len(dirs) < 2:
        raise ValueError("Нужно хотя бы два адаптера")
    if method not in ("exact", "mean"):
        raise ValueError("method: exact или mean")
    states = [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    keys = set(states[0])
    if any(set(s) != keys for s in states[1:]):
        raise ValueError("У адаптеров разный набор весов — их нельзя слить")
    n = len(states)
    merged = {}
    for key in sorted(keys):
        tensors = [s[key].to(torch.float32) for s in states]
        if method == "exact" and "lora_A" in key:
            merged[key] = torch.cat(tensors, dim=0)                 # (N·r, in)
        elif method == "exact" and "lora_B" in key:
            merged[key] = torch.cat(tensors, dim=1) / n             # (out, N·r)
        else:
            merged[key] = torch.stack(tensors).mean(dim=0)
        merged[key] = merged[key].to(states[0][key].dtype).contiguous()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    config = json.loads((dirs[0] / "adapter_config.json").read_text(encoding="utf-8"))
    if method == "exact":
        alpha_factor = math.sqrt(n) if config.get("use_rslora") else n  # масштаб alpha/r (или alpha/√r) сохраняем
        config["r"] = config["r"] * n
        config["lora_alpha"] = config["lora_alpha"] * alpha_factor
        config["rank_pattern"] = {k: v * n for k, v in (config.get("rank_pattern") or {}).items()}
        config["alpha_pattern"] = {k: v * alpha_factor for k, v in (config.get("alpha_pattern") or {}).items()}
    (out / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "tokenizer.model"):
        if (dirs[0] / name).exists():
            (out / name).write_bytes((dirs[0] / name).read_bytes())
    return out
