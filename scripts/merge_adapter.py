"""Слияние LoRA-адаптера medbot с базовой Qwen3-4B в одну 16-битную модель (на CPU, видеокарта не нужна).

    .venv\\Scripts\\python.exe scripts\\merge_adapter.py
    .venv\\Scripts\\python.exe scripts\\merge_adapter.py --adapter models\\medbot_qwen3_4b\\lora_adapter --out models\\medbot_merged_16bit

Адаптер обучали поверх 4-битной базы, но сливать в 4 битах нельзя (потеря точности),
поэтому берётся исходная 16-битная Qwen3-4B: W_new = W + B·A · (alpha / r).
Перед сохранением проверяется, что слитая модель даёт те же выходы, что база + адаптер.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from medqa.prompts import build_messages  # noqa: E402

CHECK = [("ru", "Что такое астма?"), ("kk", "Диабет дегеніміз не?"), ("en", "Is familial cold autoinflammatory syndrome inherited?")]


def logits_for(model, tokenizer, prompts):
    out = []
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out.append(model(**ids).logits[0, -1].float())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Слить LoRA-адаптер medbot с Qwen3-4B (16 бит)")
    p.add_argument("--base", default="unsloth/Qwen3-4B", help="16-битная базовая модель")
    p.add_argument("--adapter", default=str(ROOT / "models" / "medbot_qwen3_4b" / "lora_adapter"))
    p.add_argument("--out", default=str(ROOT / "models" / "medbot_merged_16bit"))
    args = p.parse_args()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    prompts = [tokenizer.apply_chat_template(build_messages(q, lang), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for lang, q in CHECK]

    print(f"1/4 Загрузка базы {args.base} (bf16, CPU) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base_logits = logits_for(base, tokenizer, prompts)

    print("2/4 Наложение адаптера ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    adapter_logits = logits_for(model, tokenizer, prompts)

    print("3/4 Слияние W + B·A·alpha/r ...", flush=True)
    merged = model.merge_and_unload()
    merged_logits = logits_for(merged, tokenizer, prompts)

    print("\nПроверка на последнем токене подсказки (макс. |разница| логитов):")
    ok = True
    for (lang, q), b, a, m in zip(CHECK, base_logits, adapter_logits, merged_logits):
        d_merge = (a - m).abs().max().item()
        d_adapter = (a - b).abs().max().item()
        same_top = a.argmax().item() == m.argmax().item()
        print(f"  [{lang}] слитая vs база+адаптер: {d_merge:.3f} (тот же следующий токен: {same_top}) | "
              f"адаптер меняет базу на: {d_adapter:.2f}")
        ok &= same_top and d_merge < 0.5 * max(d_adapter, 1e-6)
    if not ok:
        print("ОШИБКА: слитая модель заметно отличается от база+адаптер — не сохраняю.")
        return 1

    print(f"\n4/4 Сохранение в {args.out} ...", flush=True)
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.out)
    size = sum(f.stat().st_size for f in Path(args.out).glob("*.safetensors")) / 1024**3
    print(f"Готово: {size:.1f} ГБ за {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
