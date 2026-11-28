"""Summary of all lm-evaluation-harness runs -> results/harness/SUMMARY.md (English, for the defense).

    .venv\\Scripts\\python.exe scripts\\harness_report.py
    .venv\\Scripts\\python.exe scripts\\harness_report.py --no-semantic   # skip bge-m3 (faster)

Generative tasks (medquad*): harness metrics + semantic similarity (cosine of bge-m3 embeddings of answer and
reference, 0–1), which does not penalise a correct answer phrased differently.
Held-out test: paired comparison of each model against `base` on the same questions with a 95 % bootstrap CI,
and a breakdown by whether the disease (focus) appeared in training with another question type.
Multiple-choice tasks: accuracy ± standard error reported by the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "harness"
CACHE = OUT / "semantic_cache.json"
MODEL_ORDER = ["base", "base_norep4", "medbot_old", "medbot_full", "medbot_full_norep4", "base_len768", "medbot_full_len768", "saluki", "lingsante"]
# medbot_full в harness = «medbot новый» в README и test_all_models.py (адаптер models/medbot_train_full)
DISPLAY = {"medbot_full": "medbot_new"}
VARIANTS = {"medquad": "no RAG", "medquad_rag": "RAG (own answer may be retrieved)",
            "medquad_raghidden": "RAG, own answer hidden", "medquad_paraphrase": "paraphrased question"}
GEN_METRICS = {"chrf": "chrF", "rougeL": "ROUGE-L", "semantic": "semantic (bge-m3)", "lang_ok": "right language",
               "repeats": "repetition", "advises_doctor": "advises doctor", "answer_words": "answer words"}
BOOT = 2000


def set_dir(path: Path) -> None:
    global OUT, CACHE
    OUT, CACHE = Path(path), Path(path) / "semantic_cache.json"


def _load_utils():
    spec = importlib.util.spec_from_file_location("medquad_utils", ROOT / "harness/tasks/medquad/utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latest_runs() -> dict[tuple[str, str, int], dict]:
    """(model, task, n-shot) -> metrics and samples path of the latest run."""
    runs: dict[tuple[str, str, int], dict] = {}
    for path in sorted(OUT.glob("*/*/results_*.json")):  # file names contain the timestamp
        model = path.parent.parent.name
        d = json.loads(path.read_text(encoding="utf-8"))
        stamp = path.stem.removeprefix("results_")
        for task, res in d["results"].items():
            if task not in d.get("n-shot", {}):
                continue  # group row
            samples = path.parent / f"samples_{task}_{stamp}.jsonl"
            runs[(model, task, int(d["n-shot"][task]))] = {
                "res": res, "n": d["n-samples"][task]["effective"], "samples": samples if samples.exists() else None}
    return runs


def semantic_scores(pairs: list[tuple[str, str]]) -> list[float]:
    """Cosine of bge-m3 embeddings (answer, reference); cached in semantic_cache.json."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    keys = [hashlib.sha1(f"{a}\x00{r}".encode()).hexdigest() for a, r in pairs]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
        model = SentenceTransformer("BAAI/bge-m3", device="cpu")  # GPU is busy with the harness
        model.max_seq_length = 512
        texts = [t for i in todo for t in pairs[i]]
        emb = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        for j, i in enumerate(todo):
            cache[keys[i]] = float(emb[2 * j] @ emb[2 * j + 1])
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[k] for k in keys]


def per_sample(runs: dict, semantic: bool) -> pd.DataFrame:
    """One row per (model, task, question): sentence chrF, ROUGE-L, semantic similarity."""
    import sacrebleu

    utils = _load_utils()
    rows = []
    for (model, task, shot), r in runs.items():
        if not task.startswith("medquad") or not r["samples"] or "_dev_" in task:
            continue  # dev-прогоны (подбор настроек) — в scripts/decoding_sweep.py
        variant, lang = task.rsplit("_", 1)
        samples = [json.loads(line) for line in r["samples"].open(encoding="utf-8")]
        answers = [utils.clean(s["resps"][0][0], s["doc"]["lang"]) for s in samples]
        refs = [s["doc"]["ref"] for s in samples]
        sem = semantic_scores(list(zip(answers, refs))) if semantic else [np.nan] * len(samples)
        for s, a, ref, sm in zip(samples, answers, refs, sem):
            rows.append({"model": model, "split": "test" if ("_heldout" in variant or "paraphrase" in variant) else "all",
                         "variant": variant.replace("_heldout", ""), "lang": lang, "row": s["doc"].get("row", s["doc_id"]),
                         "seen_focus": s["doc"].get("seen_focus"), "chrf_s": sacrebleu.sentence_chrf(a, [ref]).score,
                         "rougeL": s["rougeL"], "semantic": sm, "answer": a, "ref": ref, "q": s["doc"]["q"]})
    return pd.DataFrame(rows)


def generative_table(runs: dict, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, shot), r in runs.items():
        if not task.startswith("medquad") or "_dev_" in task:
            continue
        variant, lang = task.rsplit("_", 1)
        split = "test" if ("_heldout" in variant or "paraphrase" in variant) else "all"
        variant = variant.replace("_heldout", "")
        row = {"split": split, "model": model, "variant": VARIANTS[variant], "lang": lang, "n": r["n"]}
        for m in GEN_METRICS:
            if f"{m},none" in r["res"]:
                row[m] = r["res"][f"{m},none"]
        if not samples.empty:
            sel = samples[(samples.model == model) & (samples.split == split) & (samples.variant == variant)
                          & (samples.lang == lang)]
            row["semantic"] = sel["semantic"].mean() if len(sel) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_s"] = df["split"].map({"test": 0, "all": 1})
    df["_v"] = df["variant"].map({v: i for i, v in enumerate(VARIANTS.values())})
    df["_m"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["_l"] = df["lang"].map({"en": 0, "ru": 1, "kk": 2})
    cols = ["split", "model", "variant", "lang", "n"] + [m for m in GEN_METRICS if m in df]
    return df.sort_values(["_s", "_v", "_m", "_l"])[cols]


def paired_ci(samples: pd.DataFrame) -> pd.DataFrame:
    """Held-out test: (system − base, no RAG) on the same questions, mean and 95 % bootstrap CI."""
    test = samples[samples.split == "test"]
    base = test[(test.model == "base") & (test.variant == "medquad")]
    if base.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    rows = []
    for (model, variant), sys_df in test.groupby(["model", "variant"]):
        if (model == "base" and variant == "medquad") or variant == "medquad_paraphrase":
            continue
        for lang in ["en", "ru", "kk", "all"]:
            a = sys_df if lang == "all" else sys_df[sys_df.lang == lang]
            b = base if lang == "all" else base[base.lang == lang]
            m = a.merge(b, on=["lang", "row"], suffixes=("", "_b"))
            if m.empty:
                continue
            row = {"system": f"{model} + {VARIANTS[variant]}" if variant != "medquad" else model,
                   "lang": lang, "n": len(m)}
            for metric, name in (("chrf_s", "Δ chrF"), ("rougeL", "Δ ROUGE-L"), ("semantic", "Δ semantic")):
                d = (m[metric] - m[f"{metric}_b"]).to_numpy()
                if np.isnan(d).all():
                    continue
                boot = rng.choice(d, size=(BOOT, len(d))).mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                fmt = "{:+.1f} [{:+.1f}, {:+.1f}]" if metric == "chrf_s" else "{:+.3f} [{:+.3f}, {:+.3f}]"
                row[name] = fmt.format(d.mean(), lo, hi)
            rows.append(row)
    return pd.DataFrame(rows)


def paraphrase_drop(samples: pd.DataFrame) -> pd.DataFrame:
    """Same rows, original vs paraphrased question, per model: does the score hold when the wording changes?"""
    test = samples[samples.split == "test"]
    para = test[test.variant == "medquad_paraphrase"]
    rows = []
    for model, p in para.groupby("model", sort=False):
        orig = test[(test.model == model) & (test.variant == "medquad")]
        m = p.merge(orig, on=["lang", "row"], suffixes=("_p", "_o"))
        if m.empty:
            continue
        for lang in ["en", "ru", "kk", "all"]:
            g = m if lang == "all" else m[m.lang == lang]
            rows.append({"model": model, "lang": lang, "n": len(g),
                         "chrF original → paraphrase": f"{g.chrf_s_o.mean():.1f} → {g.chrf_s_p.mean():.1f}",
                         "semantic original → paraphrase": f"{g.semantic_o.mean():.3f} → {g.semantic_p.mean():.3f}"})
    return pd.DataFrame(rows)


def seen_breakdown(samples: pd.DataFrame) -> pd.DataFrame:
    test = samples[(samples.split == "test") & samples.seen_focus.notna()]
    if test.empty:
        return test
    g = test.groupby(["model", "variant", "seen_focus"]).agg(n=("row", "size"), chrf=("chrf_s", "mean"),
                                                              rougeL=("rougeL", "mean"), semantic=("semantic", "mean"))
    g = g.reset_index()
    g["variant"] = g["variant"].map(VARIANTS)
    g["seen_focus"] = g["seen_focus"].map({True: "disease seen in train", False: "unseen disease"})
    return g.rename(columns={"chrf": "mean sentence chrF"})


def mc_table(runs: dict) -> pd.DataFrame:
    rows = {}
    for (model, task, shot), r in runs.items():
        if task.startswith("medquad") or "acc,none" not in r["res"]:
            continue
        key = (task, shot)
        rows.setdefault(key, {"task": task, "shot": shot, "n": r["n"]})
        rows[key][model] = f"{100 * r['res']['acc,none']:.1f} ± {100 * r['res']['acc_stderr,none']:.1f}"
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    models = [m for m in MODEL_ORDER if m in df] + [m for m in df if m not in MODEL_ORDER + ["task", "shot", "n"]]
    return df[["task", "shot", "n"] + models].sort_values(["shot", "task"])


def to_markdown(df: pd.DataFrame) -> str:
    fmt = df.copy()
    for c in fmt.columns:
        if c in ("lang_ok", "repeats", "advises_doctor"):
            fmt[c] = (100 * fmt[c]).map("{:.0f} %".format)
        elif c in ("chrf", "answer_words", "mean sentence chrF"):
            fmt[c] = fmt[c].map("{:.1f}".format)
        elif c in ("rougeL", "semantic"):
            fmt[c] = fmt[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    fmt = fmt.rename(columns=GEN_METRICS)
    lines = ["| " + " | ".join(map(str, fmt.columns)) + " |", "|" + "---|" * len(fmt.columns)]
    lines += ["| " + " | ".join("—" if (isinstance(v, float) and pd.isna(v)) else str(v) for v in r) + " |"
              for r in fmt.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--dir", type=Path, default=OUT, help="runs directory (default results/harness)")
    a = p.parse_args()
    set_dir(a.dir)
    runs = latest_runs()
    samples = per_sample(runs, not a.no_semantic)
    gen, mc = generative_table(runs, samples), mc_table(runs)
    text = ["# lm-evaluation-harness results", "",
            "Generated by `scripts/harness_report.py` from `results/harness/*/*/results_*.json` (latest run of each "
            "task). Split `test` = held-out rows medbot_old never saw (see `docs/DATASET_CARD.md`); split `all` = "
            "random rows of the whole dataset, all of them seen in training by medbot_full (memorisation check).",
            "", "Decoding: greedy, max 512 new tokens, repetition penalty 1.1, Qwen3 chat template, thinking off.", ""]
    if not gen.empty:
        text += ["## MedQuAD — open-ended answers (EN / RU / KK)", "",
                 "chrF: corpus chrF (sacrebleu, char 6-grams, β=2). ROUGE-L: word-level LCS F1, mean over questions. "
                 "semantic: mean cosine of bge-m3 embeddings. All three measure similarity to the dataset reference, "
                 "**not** medical correctness.", "", to_markdown(gen), ""]
    ci = paired_ci(samples) if not samples.empty else pd.DataFrame()
    if not ci.empty:
        text += ["## Held-out test: difference vs base (same questions)", "",
                 f"Mean of per-question differences (sentence chrF, ROUGE-L, semantic), 95 % bootstrap CI "
                 f"({BOOT} resamples). A CI that excludes 0 is a significant difference.", "", to_markdown(ci), ""]
    pd_ = paraphrase_drop(samples) if not samples.empty else pd.DataFrame()
    if not pd_.empty:
        text += ["## Robustness to rewording", "",
                 "20 held-out questions × 3 languages rewritten by hand in different words (`data/paraphrased_questions.json`; "
                 "treatment / prevention asked as 'what is the best ...'). Same reference answer. A model that only "
                 "memorised the wording drops sharply; mean sentence chrF and semantic similarity.", "", to_markdown(pd_), ""]
    sb = seen_breakdown(samples[samples.variant != "medquad_paraphrase"]) if not samples.empty else pd.DataFrame()
    if not sb.empty:
        text += ["## Held-out test by topic overlap", "",
                 "Test rows whose disease appeared in training with another question type vs unseen diseases.", "",
                 to_markdown(sb), ""]
    if not mc.empty:
        text += ["## Multiple-choice benchmarks — accuracy, % (± standard error)", "",
                 "Log-likelihood of each option letter, no chat template; shot = solved examples in the prompt.", "",
                 to_markdown(mc), ""]
    text = [line.replace("medbot_full", DISPLAY["medbot_full"]) for line in text]
    (OUT / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    if not samples.empty:
        samples.drop(columns=["answer", "ref", "q"]).to_csv(OUT / "per_question_metrics.csv", index=False)
        samples.to_json(OUT / "per_question_answers.jsonl", orient="records", lines=True, force_ascii=False)
    print("\n".join(text))


if __name__ == "__main__":
    main()
