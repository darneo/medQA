"""Командная строка: python -m medqa <команда>. Подробности: python -m medqa <команда> --help."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import (DATASET, HOPFIELD_DIR, LANG_NAMES, LANGS, MEDBOT_ADAPTER, MEDBOT_BASE_MODEL, MEDBOT_RESULTS,
                     MODELS_DIR, RAG_EMBED_MODEL)


_ARGPARSE_RU = {
    "usage: ": "использование: ",
    "positional arguments": "аргументы",
    "options": "параметры",
    "show this help message and exit": "показать эту справку и выйти",
    "show program's version number and exit": "показать версию и выйти",
    "the following arguments are required: %s": "не хватает обязательных аргументов: %s",
    "unrecognized arguments: %s": "неизвестные аргументы: %s",
    "%(prog)s: error: %(message)s\n": "%(prog)s: ошибка: %(message)s\n",
    "argument %(argument_name)s: %(message)s": "аргумент %(argument_name)s: %(message)s",
    "invalid choice: %(value)r (choose from %(choices)s)": "недопустимое значение: %(value)r (варианты: %(choices)s)",
    "invalid %(type)s value: %(value)r": "неверное значение (%(type)s): %(value)r",
    "expected one argument": "нужно одно значение",
}


def _russian_argparse() -> None:
    """Переводит служебные надписи argparse (заголовки справки и ошибки) на русский."""
    if getattr(argparse, "_medqa_ru", False):
        return
    original = argparse._
    argparse._ = lambda text: _ARGPARSE_RU.get(text, original(text))
    argparse._medqa_ru = True


def _utf8_console() -> None:
    # Консоль Windows (cp1251) не печатает казахские буквы — без этого print падает.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


# ---------------------------------------------------------------- data


def cmd_data(args) -> None:
    from .data import col, load_dataset

    df = load_dataset(args.data, verbose=True)
    print(f"Всего: {len(df)} строк × 3 языка (validation-выборки нет — всё идёт в обучение и поиск)")
    print("\nДлина ответов, символов:")
    for lang in LANGS:
        s = df[col("answer", lang)].str.len()
        print(f"  {lang} ({LANG_NAMES[lang]}): медиана {int(s.median())}, 90% ≤ {int(s.quantile(0.9))}, макс {int(s.max())}")
    if "category" in df:
        print("\nИсточники:")
        for name, n in df["category"].value_counts().items():
            print(f"  {name}: {n}")


# ---------------------------------------------------------------- medbot


def cmd_medbot_train(args) -> None:
    from .medbot import TrainOptions, train

    train(TrainOptions(output=Path(args.output), base_model=args.base_model, shard=args.shard, shards=args.shards,
                       max_train_rows=args.max_train_rows, rows=args.rows,
                       init_adapter=Path(args.init_adapter) if args.init_adapter else None, epochs=args.epochs, lr=args.lr,
                       max_seq_length=args.max_seq_length, batch_size=args.batch_size, grad_accum=args.grad_accum,
                       save_steps=args.save_steps, responses_only=not args.full_text_loss))


def _gen_kwargs(args) -> dict:
    return {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty}


def cmd_medbot_chat(args) -> None:
    from .medbot import generate, load_model

    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    print(f"Модель загружена. Язык: {LANG_NAMES[args.lang]}. Пустая строка или «выход» — завершить.")
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit", "выход"):
            break
        print("\nАссистент:", generate(model, tokenizer, question, args.lang, **_gen_kwargs(args)))


def cmd_medbot_test(args) -> None:
    from .medbot import load_model, smoke_test

    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    report = smoke_test(model, tokenizer, **_gen_kwargs(args))
    for r in report["results"]:
        extra = f" | совет обратиться за помощью: {'да' if r['advises_help'] else 'НЕТ'}" if r["urgent"] else ""
        mark = "OK" if r["language_ok"] else "НЕ ТОТ ЯЗЫК"
        print(f"\n===== [{r['lang']}] {r['question']}\nязык: ожидался {r['lang']}, получен {r['detected']} -> {mark}"
              f"{extra}\n{r['answer']}")
    print(f"\nИТОГ: {report['language_ok']}/{report['total']} ответов на нужном языке")
    if args.save:
        MEDBOT_RESULTS.mkdir(parents=True, exist_ok=True)
        path = MEDBOT_RESULTS / "smoke_test.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Сохранено:", path)


def cmd_medbot_eval(args) -> None:
    from .medbot import dataset_examples, evaluate_loss, load_model

    model, tokenizer = load_model(args.adapter, args.max_seq_length)
    result = evaluate_loss(model, tokenizer, dataset_examples(args.rows, args.unseen_only), args.max_seq_length,
                           include_prompt=args.include_prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save:
        MEDBOT_RESULTS.mkdir(parents=True, exist_ok=True)
        name = "dataset_loss_full_dialog.json" if args.include_prompt else "dataset_loss_answer_only.json"
        (MEDBOT_RESULTS / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Сохранено:", MEDBOT_RESULTS / name)


def cmd_medbot_merge(args) -> None:
    from .medbot import merge_adapters

    out = merge_adapters(args.adapters, args.out, args.method)
    print(f"Слито адаптеров: {len(args.adapters)} ({args.method}) -> {out.resolve()}")


# ---------------------------------------------------------------- saluki / rag


def cmd_saluki_download(args) -> None:
    from .saluki import download

    print("Скачивание GGUF Saluki (~7.9 ГБ) в models/saluki/ ...", flush=True)
    print("Готово:", download())


def _server_config(args):
    from .saluki import ServerConfig

    return ServerConfig(port=args.port, ctx=args.ctx, parallel=args.parallel, gpu_layers=args.gpu_layers)


def cmd_saluki_serve(args) -> None:
    import time

    from .saluki import SalukiServer

    with SalukiServer(_server_config(args)) as server:
        print(f"Сервер работает: {server.url} (OpenAI API: {server.url}/v1). Ctrl+C — остановить.")
        try:
            while True:
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nОстановка ...")


def cmd_saluki_chat(args) -> None:
    from .rag import RAGIndex, format_context
    from .saluki import SalukiServer, ask

    rag = None
    if args.rag == "hopfield":
        from .hopfield import HopfieldQA

        rag = HopfieldQA()
    elif args.rag:
        rag = RAGIndex()
    with SalukiServer(_server_config(args)) as server:
        print(f"Saluki{f' + RAG ({args.rag})' if rag else ''}, язык: {LANG_NAMES[args.lang]}. "
              "Пустая строка или «выход» — завершить.")
        while True:
            try:
                question = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in ("exit", "quit", "выход"):
                break
            context = None
            if rag:
                hits = rag.search(question, args.lang, args.k)
                context = format_context(hits)
                for h in hits:
                    print(f"  [RAG {h['score']:.2f}] {h['question']}")
            print("\nSaluki:", ask(question, args.lang, url=server.url, context=context,
                                   max_tokens=args.max_tokens)["answer"])


def cmd_rag_build(args) -> None:
    from .rag import RAGIndex

    RAGIndex.build(model_name=args.model, device=args.device)


def cmd_rag_search(args) -> None:
    from .rag import RAGIndex

    for h in RAGIndex().search(args.question, args.lang, args.k):
        print(f"[похожесть {h['score']:.3f}, строка {h['row']}] {h['question']}\n  {h['answer'][:300]}\n")


# ---------------------------------------------------------------- hopfield


def cmd_hopfield_build(args) -> None:
    from .hopfield import build_cache

    build_cache(model_name=args.model, device=args.device)


def cmd_hopfield_train(args) -> None:
    from .hopfield import TrainOptions, train

    train(TrainOptions(out=args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, steps=args.steps,
                       beta=args.beta, memory=args.memory, hidden_share=args.hidden_share, device=args.device))


def cmd_hopfield_ask(args) -> None:
    from .hopfield import HopfieldQA

    qa = HopfieldQA(model_dir=None if args.untrained else args.model_dir, steps=args.steps, beta=args.beta)
    kind = "обученная" if qa.trained else "без обучения"
    print(f"Hopfield-QA ({kind}, шагов {qa.steps}), язык: {LANG_NAMES[args.lang]}\n")
    hits = qa.search(args.question, args.lang, args.k)
    best = hits[0]
    if best["score"] < args.min_confidence:
        print(f"Не уверена (p = {best['score']:.3f} < {args.min_confidence}) — ответа нет.\n")
    else:
        print(f"Ответ (уверенность {best['score']:.3f}, вспомнен вопрос: «{best['question']}»):\n{best['answer']}\n")
    if args.k > 1:
        print("Другие кандидаты:")
        for h in hits[1:]:
            print(f"  p = {h['score']:.3f}  {h['question']}")


def cmd_hopfield_eval(args) -> None:
    from .hopfield import evaluate

    summary = evaluate(rows=args.rows, chrf=not args.no_chrf)
    names = [n for n in summary if not n.startswith("_")]
    for mode, title in (("hidden", "свой ответ скрыт"), ("recall", "свой ответ в памяти")):
        keys = list(summary[names[0]][mode]["all"])
        print(f"\nВесь датасет, {title} (доли 0–1, chrF 0–100):\n{'метрика':<12}" + "".join(f"{n:>18}" for n in names))
        for k in keys:
            print(f"{k:<12}" + "".join(f"{summary[n][mode]['all'].get(k, float('nan')):>18.4f}" for n in names))
    for k, v in summary["_note"].items():
        print(f"{k}: {v}")
    print("\nРезультаты: results/hopfield/eval_summary.json, eval_dataset.jsonl")


def cmd_hopfield_export(args) -> None:
    from .hopfield import export_csv

    path = export_csv(args.out, args.lang, args.rows, None if args.untrained else args.model_dir,
                      hide_own=not args.show_own)
    print("Сохранено:", path)


# ---------------------------------------------------------------- парсер


def _add_medbot_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--adapter", default=str(MEDBOT_ADAPTER), help="папка lora_adapter (или базовая модель)")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7, help="0 — детерминированный ответ")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="штраф за повторы (1.0 — без штрафа)")


def build_parser() -> argparse.ArgumentParser:
    _russian_argparse()
    parser = argparse.ArgumentParser(prog="python -m medqa",
                                     description="Медицинский QA на en/ru/kk: medbot (Qwen3-4B + QLoRA) и "
                                                 "Saluki 27B (+ RAG).")
    parser.add_argument("--version", action="version", version=f"medqa {__version__}")
    parser.add_argument("--data", default=str(DATASET), help="датасет parquet (по умолчанию data/final/...)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="команда")

    p = sub.add_parser("data", help="сводка по датасету")
    p.set_defaults(func=cmd_data)

    m = sub.add_parser("medbot", help="medbot: Qwen3-4B + LoRA (нужна GPU NVIDIA)")
    ms = m.add_subparsers(dest="medbot_command", required=True, metavar="действие")

    p = ms.add_parser("train", help="дообучить Qwen3-4B (QLoRA), с чекпоинтами и продолжением")
    p.add_argument("--output", default=str(MODELS_DIR / "medbot_train"),
                   help="папка результата (существующий адаптер не перезаписывается)")
    p.add_argument("--base-model", default=MEDBOT_BASE_MODEL)
    p.add_argument("--shard", type=int, default=0, help="номер шарда (для обучения на нескольких машинах)")
    p.add_argument("--shards", type=int, default=1, help="всего шардов")
    p.add_argument("--max-train-rows", type=int, default=0, help="ограничить число строк (0 — все)")
    p.add_argument("--rows", choices=["all", "unseen"], default="all",
                   help="all — весь датасет; unseen — строки, которых не было в первом обучении medbot")
    p.add_argument("--init-adapter", help="дообучить этот адаптер (lora_adapter), а не начинать новый")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--full-text-loss", action="store_true",
                   help="считать loss по всему диалогу, как в первом обучении (по умолчанию только по ответу)")
    p.set_defaults(func=cmd_medbot_train)

    p = ms.add_parser("chat", help="чат с моделью в консоли")
    p.add_argument("--lang", default="ru", choices=LANGS)
    _add_medbot_gen_args(p)
    p.set_defaults(func=cmd_medbot_chat)

    p = ms.add_parser("test", help="6 контрольных вопросов: язык ответа и безопасность")
    p.add_argument("--save", action="store_true", help=f"сохранить в {MEDBOT_RESULTS.name}/smoke_test.json")
    _add_medbot_gen_args(p)
    p.set_defaults(func=cmd_medbot_test)

    p = ms.add_parser("eval", help="loss на датасете по языкам")
    p.add_argument("--adapter", default=str(MEDBOT_ADAPTER))
    p.add_argument("--max-seq-length", type=int, default=1024, help="как при обучении, иначе loss несравним")
    p.add_argument("--rows", type=int, help="взять только N строк датасета (быстрее)")
    p.add_argument("--unseen-only", action="store_true", help="только строки, которых не было в первом обучении")
    p.add_argument("--include-prompt", action="store_true",
                   help="считать loss по всему диалогу, как в старом отчёте (по умолчанию только по ответу)")
    p.add_argument("--save", action="store_true", help="сохранить результат в results/medbot/")
    p.set_defaults(func=cmd_medbot_eval)

    p = ms.add_parser("merge", help="слить адаптеры шардов в один")
    p.add_argument("adapters", nargs="+", help="папки lora_adapter шардов")
    p.add_argument("--out", required=True, help="папка для слитого адаптера")
    p.add_argument("--method", choices=["exact", "mean"], default="exact",
                   help="exact — точное среднее поправок (по умолчанию); mean — старый приблизительный способ")
    p.set_defaults(func=cmd_medbot_merge)

    # saluki
    s = sub.add_parser("saluki", help="Saluki 27B через llama.cpp (готовая модель, без обучения)")
    ss = s.add_subparsers(dest="saluki_command", required=True, metavar="действие")
    p = ss.add_parser("download", help="скачать GGUF Saluki (~7.9 ГБ) в models/saluki/")
    p.set_defaults(func=cmd_saluki_download)
    for name, func, help_text in (("serve", cmd_saluki_serve, "запустить llama-server с Saluki"),
                                  ("chat", cmd_saluki_chat, "чат с Saluki в консоли (можно с --rag)")):
        p = ss.add_parser(name, help=help_text)
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--ctx", type=int, default=8192, help="контекст на все слоты, токенов")
        p.add_argument("--parallel", type=int, default=2, help="одновременных запросов")
        p.add_argument("--gpu-layers", type=int, help="слоёв на видеокарте (по умолчанию подбирается сам)")
        p.set_defaults(func=func)
        if name == "chat":
            p.add_argument("--lang", default="ru", choices=LANGS)
            p.add_argument("--rag", nargs="?", const="cosine", choices=["cosine", "hopfield"],
                           help="подсказывать похожие ответы из датасета: cosine (по умолчанию) или hopfield")
            p.add_argument("--k", type=int, default=3, help="сколько пар давать при --rag")
            p.add_argument("--max-tokens", type=int, default=512)

    # rag
    r = sub.add_parser("rag", help="RAG: индекс эмбеддингов по датасету и поиск похожих вопросов")
    rs = r.add_subparsers(dest="rag_command", required=True, metavar="действие")
    p = rs.add_parser("build", help="собрать индекс по всему датасету в data/rag_index/")
    p.add_argument("--model", default=RAG_EMBED_MODEL, help="модель эмбеддингов")
    p.add_argument("--device", help="cuda или cpu (по умолчанию cuda, если есть)")
    p.set_defaults(func=cmd_rag_build)
    p = rs.add_parser("search", help="найти похожие вопросы")
    p.add_argument("question")
    p.add_argument("--lang", default="ru", choices=LANGS)
    p.add_argument("--k", type=int, default=3)
    p.set_defaults(func=cmd_rag_search)

    # hopfield
    h = sub.add_parser("hopfield", help="Hopfield-QA: сеть Хопфилда вспоминает готовый ответ из датасета")
    hs = h.add_subparsers(dest="hopfield_command", required=True, metavar="действие")
    p = hs.add_parser("build", help="эмбеддинги памяти (все ответы) и вопросов в data/hopfield_cache/")
    p.add_argument("--model", default=RAG_EMBED_MODEL, help="модель эмбеддингов")
    p.add_argument("--device", help="cuda или cpu (по умолчанию cuda, если есть)")
    p.set_defaults(func=cmd_hopfield_build)
    p = hs.add_parser("train", help="обучить проекции W_Q, W_K и β → models/hopfield/")
    p.add_argument("--out", default=str(HOPFIELD_DIR))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=1, help="шагов обновления Хопфилда")
    p.add_argument("--beta", type=float, default=50.0, help="начальная β (обучается)")
    p.add_argument("--memory", choices=["qa", "answer"], default="qa",
                   help="образ памяти: qa — «вопрос + ответ» (по умолчанию), answer — только ответ")
    p.add_argument("--hidden-share", type=float, default=0.5,
                   help="доля примеров со скрытым своим ответом (0 — всегда в памяти)")
    p.add_argument("--device", help="cuda или cpu")
    p.set_defaults(func=cmd_hopfield_train)
    p = hs.add_parser("ask", help="задать вопрос: ответ из памяти и уверенность")
    p.add_argument("question")
    p.add_argument("--lang", default="ru", choices=LANGS)
    p.add_argument("--k", type=int, default=3, help="сколько кандидатов показать")
    p.add_argument("--model-dir", default=str(HOPFIELD_DIR))
    p.add_argument("--untrained", action="store_true", help="сеть без обучения (W = I)")
    p.add_argument("--steps", type=int, help="шагов обновления (по умолчанию — как при обучении)")
    p.add_argument("--beta", type=float, help="β вместо обученной")
    p.add_argument("--min-confidence", type=float, default=0.0, help="ниже — отказ от ответа")
    p.set_defaults(func=cmd_hopfield_ask)
    p = hs.add_parser("eval", help="проверка по датасету: RAG vs косинус vs Хопфилд vs обученный Хопфилд")
    p.add_argument("--rows", type=int, help="взять только N строк датасета")
    p.add_argument("--no-chrf", action="store_true", help="не считать chrF (быстрее)")
    p.set_defaults(func=cmd_hopfield_eval)
    p = hs.add_parser("export", help="CSV: вопросы датасета, реальный ответ и ответ Hopfield рядом")
    p.add_argument("--out", help="путь CSV (по умолчанию results/hopfield/hopfield_vs_reference.csv)")
    p.add_argument("--lang", nargs="+", choices=LANGS, help="только эти языки (по умолчанию все три)")
    p.add_argument("--rows", type=int, help="взять только N строк датасета")
    p.add_argument("--show-own", action="store_true", help="не скрывать свой ответ (по умолчанию скрыт)")
    p.add_argument("--model-dir", default=str(HOPFIELD_DIR))
    p.add_argument("--untrained", action="store_true", help="сеть без обучения (W = I)")
    p.set_defaults(func=cmd_hopfield_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nПрервано. Повторный запуск продолжит с того же места.")
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0
