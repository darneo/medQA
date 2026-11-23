"""Hopfield-QA: современная сеть Хопфилда как ассоциативная память вопрос → ответ.

Модель НЕ генерирует текст, а «вспоминает»: возвращает готовый ответ из датасета.

Как это работает (Ramsauer et al., 2020, «Hopfield Networks is All You Need»):

1. **Память** (`build_cache`, один раз). Каждый ответ ВСЕГО датасета на каждом из трёх языков превращается
   моделью bge-m3 в вектор — это сохранённые образы K (N × 1024). Отдельно кодируются все вопросы
   (для обучения). Отдельной validation-выборки нет.
2. **Динамика.** Вопрос → вектор ξ. Шаг обновления сети:  ξ ← норм(Kᵀ · softmax(β · K ξ)).
   За несколько шагов запрос притягивается к аттрактору — ближайшему образу или «центру темы».
   Итог: p = softmax(β · K ξ) — вероятность каждого сохранённого ответа; argmax p — ответ модели,
   max p — её уверенность. Поиск идёт только среди ответов того же языка.
3. **Обучение** (`train`). Поверх bge-m3 обучаются проекции W_Q (для вопроса), W_K (для образов) и β:
   вопрос должен притягиваться к ответам про ту же болезнь, при этом свой ответ скрыт — так сеть учится
   обобщать, а не запоминать. Обучение идёт на всех вопросах датасета. W_Q и W_K стартуют с единичной
   матрицы — то есть с необученной сети.

    python -m medqa hopfield build                    # эмбеддинги памяти (≈ 5–10 мин на GPU)
    python -m medqa hopfield train                    # обучить проекции → models/hopfield/
    python -m medqa hopfield ask "Что такое астма?" --lang ru
    python -m medqa hopfield eval                     # проверка по всему датасету → results/hopfield/
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HOPFIELD_CACHE_DIR, HOPFIELD_DIR, LANGS, RAG_EMBED_MODEL, SEED
from .data import col, load_dataset, question_type

LANG_ID = {lang: i for i, lang in enumerate(LANGS)}


# ---------------------------------------------------------------- память (эмбеддинги)


def _encoder(model_name: str, device: str | None):
    from .rag import _encoder as rag_encoder

    return rag_encoder(model_name, device)


def _encode(model, texts: list[str], batch_size: int, desc: str) -> np.ndarray:
    print(f"  {desc}: {len(texts)} текстов", flush=True)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float16)


def _expand(frame: pd.DataFrame) -> pd.DataFrame:
    """Строка датасета → по записи на язык (row — номер строки датасета)."""
    return pd.DataFrame([{"row": i, "lang": lang, "question": r[col("question", lang)],
                          "answer": r[col("answer", lang)], "focus": r.get("focus"), "qtype": r.get("qtype")}
                         for i, r in frame.iterrows() for lang in LANGS])


def build_cache(cache_dir: Path | str = HOPFIELD_CACHE_DIR, model_name: str = RAG_EMBED_MODEL,
                device: str | None = None, batch_size: int = 32) -> Path:
    """Эмбеддинги всех ответов (память) и всех вопросов (обучение и проверка) датасета."""
    tr = _expand(load_dataset())
    t0 = time.time()
    model = _encoder(model_name, device)
    print(f"Память Хопфилда: модель {model_name}, {model.device}", flush=True)
    mem = _encode(model, tr["answer"].tolist(), batch_size, "ответы (образы памяти)")
    questions = _encode(model, tr["question"].tolist(), batch_size * 4, "вопросы")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in ("train.parquet", "validation.parquet", "q_train.npy", "q_val.npy"):   # старый формат с validation
        (cache_dir / old).unlink(missing_ok=True)
    tr.to_parquet(cache_dir / "passages.parquet", index=False)
    np.save(cache_dir / "memory.npy", mem)
    np.save(cache_dir / "questions.npy", questions)
    meta = {"model": model_name, "memory": len(tr), "rows": int(tr["row"].nunique()), "dim": int(mem.shape[1]),
            "built_seconds": round(time.time() - t0)}
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено в {cache_dir} за {meta['built_seconds']} с", flush=True)
    return cache_dir


def normalize_focus(focus: pd.Series) -> pd.Series:
    """«Freeman-Sheldon syndrome» и «Freeman Sheldon syndrome» — одна болезнь: регистр и знаки не важны."""
    return focus.str.lower().str.replace(r"[^\w]+", " ", regex=True).str.strip()


def load_cache(cache_dir: Path | str = HOPFIELD_CACHE_DIR) -> dict:
    cache_dir = Path(cache_dir)
    if not (cache_dir / "passages.parquet").exists():
        raise FileNotFoundError(f"Нет памяти Хопфилда (по всему датасету) в {cache_dir} — соберите: "
                                "python -m medqa hopfield build")
    passages = pd.read_parquet(cache_dir / "passages.parquet")
    passages["focus"] = normalize_focus(passages["focus"])
    # тип вопроса (симптомы, лечение, …) — по английской формулировке строки, общий для трёх языков
    en = passages[passages["lang"] == "en"]
    passages["qtype"] = passages["row"].map(dict(zip(en["row"], en["question"].map(question_type))))
    answers, questions = np.load(cache_dir / "memory.npy"), np.load(cache_dir / "questions.npy")
    pair = answers.astype(np.float32) + questions.astype(np.float32)
    pair /= np.linalg.norm(pair, axis=1, keepdims=True)
    return {"meta": json.loads((cache_dir / "meta.json").read_text(encoding="utf-8")),
            "passages": passages,
            "memory": answers,                        # образ = ответ
            "memory_qa": pair.astype(np.float16),     # образ = пара «вопрос + ответ» (как в RAG)
            "questions": questions}


def _memory(cache: dict, kind: str) -> np.ndarray:
    if kind not in ("answer", "qa"):
        raise ValueError("memory: answer или qa")
    return cache["memory_qa"] if kind == "qa" else cache["memory"]


# ---------------------------------------------------------------- сеть


def _torch():
    import torch

    return torch


class HopfieldNet:
    """Современная сеть Хопфилда с обучаемыми проекциями W_Q, W_K и температурой β.

    Без обучения (W = I) это bge-m3 + динамика Хопфилда; steps=0 — обычный косинусный поиск.
    """

    def __init__(self, dim: int, beta: float = 20.0, device: str = "cpu"):
        torch = _torch()
        self.device = device
        self.w_q = torch.eye(dim, device=device)
        self.w_k = torch.eye(dim, device=device)
        self.log_beta = torch.tensor(float(np.log(beta)), device=device)

    @property
    def beta(self) -> float:
        return float(self.log_beta.detach().exp())

    def parameters(self):
        return [self.w_q, self.w_k, self.log_beta]

    def keys(self, memory):
        torch = _torch()
        return torch.nn.functional.normalize(memory @ self.w_k, dim=-1)

    def retrieve(self, queries, keys, mask=None, steps: int = 1, beta: float | None = None):
        """Логиты β·Kξ после `steps` шагов обновления. mask (B × N, bool) — разрешённые образы (свой язык)."""
        torch = _torch()
        b = self.log_beta.exp() if beta is None else torch.tensor(beta, device=keys.device)
        xi = torch.nn.functional.normalize(queries @ self.w_q, dim=-1)
        for step in range(steps + 1):
            logits = b * (xi @ keys.T)
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf"))
            if step == steps:
                return logits
            xi = torch.nn.functional.normalize(torch.softmax(logits, dim=-1) @ keys, dim=-1)

    def save(self, out_dir: Path | str, meta: dict) -> Path:
        torch = _torch()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"w_q": self.w_q.detach().half().cpu(), "w_k": self.w_k.detach().half().cpu(),
                    "log_beta": self.log_beta.detach().cpu()}, out_dir / "hopfield.pt")
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, model_dir: Path | str = HOPFIELD_DIR, device: str = "cpu") -> tuple["HopfieldNet", dict]:
        torch = _torch()
        model_dir = Path(model_dir)
        if not (model_dir / "hopfield.pt").exists():
            raise FileNotFoundError(f"Нет обученной сети в {model_dir} — обучите: python -m medqa hopfield train")
        state = torch.load(model_dir / "hopfield.pt", map_location=device)
        net = cls(state["w_q"].shape[0], device=device)
        net.w_q, net.w_k = state["w_q"].float().to(device), state["w_k"].float().to(device)
        net.log_beta = state["log_beta"].float().to(device)
        return net, json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- обучение


@dataclass
class TrainOptions:
    out: str = str(HOPFIELD_DIR)
    epochs: int = 5
    batch_size: int = 512
    lr: float = 3e-4
    steps: int = 1               # шагов обновления Хопфилда при обучении и по умолчанию при ответе
    beta: float = 50.0           # начальная β (дальше обучается); при меньшей запрос «размывается»
    identity_decay: float = 1e-3  # штраф за уход W от единичной матрицы (против переобучения)
    memory: str = "qa"           # образ памяти: qa — «вопрос + ответ» (тип вопроса виден), answer — только ответ
    hidden_share: float = 0.5    # доля примеров, где свой ответ скрыт (учит обобщать на новые формулировки)
    device: str | None = None


def _ids(values: pd.Series) -> np.ndarray:
    """Номер болезни / типа вопроса у каждого образа; пусто — -1 (такие не бывают «правильными»)."""
    return pd.factorize(values)[0]


def train(opts: TrainOptions) -> dict:
    """Обучить W_Q, W_K, β на всех вопросах датасета.

    Правильный ответ — про ту же болезнь (focus) И того же типа (симптомы, лечение, …) на том же языке;
    loss = −log Σ p(правильных). В доле hidden_share примеров свой ответ скрыт: тогда правильны другие ответы
    той же болезни и типа, а если таких нет — любые про ту же болезнь. В остальных свой ответ в памяти
    (так сеть учится возвращать ровно ответ на заданный вопрос).
    Метрики после эпох считаются по тем же вопросам — отдельной validation нет.
    """
    torch = _torch()
    device = opts.device or ("cuda" if torch.cuda.device_count() else "cpu")
    cache = load_cache()
    tr = cache["passages"]
    memory = torch.tensor(_memory(cache, opts.memory), dtype=torch.float32, device=device)
    queries = torch.tensor(cache["questions"], dtype=torch.float32, device=device)
    mem_lang = torch.tensor(tr["lang"].map(LANG_ID).to_numpy(), device=device)
    focus_np = _ids(tr["focus"])
    focus = torch.tensor(focus_np, device=device)
    qtype = torch.tensor(_ids(tr["qtype"]), device=device)
    # можно ли скрыть свой ответ: есть другой ответ про ту же болезнь на том же языке
    group = pd.Series(list(zip(focus_np, tr["lang"])))
    can_hide = torch.tensor((group.map(group.value_counts()) > 1).to_numpy() & (focus_np >= 0), device=device)

    all_idx = torch.tensor(np.flatnonzero(focus_np >= 0), device=device)
    hide_idx = all_idx[can_hide[all_idx]]
    all_ids = torch.arange(len(tr), device=device)

    def masks(idx, hidden):
        """allowed — тот же язык (и не свой ответ, если hidden); positive — та же болезнь и тип,
        а при скрытом своём ответе без таких — просто та же болезнь."""
        allowed = mem_lang[None, :] == mem_lang[idx][:, None]
        allowed &= ~(hidden[:, None] & (all_ids[None, :] == idx[:, None]))
        same_focus = allowed & (focus[None, :] == focus[idx][:, None])
        same_type = same_focus & (qtype[None, :] == qtype[idx][:, None])
        positive = torch.where(same_type.any(dim=1, keepdim=True), same_type, same_focus)
        return allowed, positive

    net = HopfieldNet(memory.shape[1], beta=opts.beta, device=device)
    eye = torch.eye(memory.shape[1], device=device)
    for p in net.parameters():
        p.requires_grad_(True)
    optim = torch.optim.AdamW(net.parameters(), lr=opts.lr, weight_decay=0.0)

    def report_metrics(steps):
        """exact@1 и type@1 со своим ответом в памяти; focus@1 и type@1 при скрытом своём ответе."""
        out = {}
        with torch.no_grad():
            keys = net.keys(memory)
            for mode, idx_all in (("own", all_idx), ("hidden", hide_idx)):
                sums = {"exact@1": 0.0, "type@1": 0.0, "focus@1": 0.0}
                for s in range(0, len(idx_all), 1024):
                    idx = idx_all[s:s + 1024]
                    hidden = torch.full((len(idx),), mode == "hidden", device=device)
                    allowed, _ = masks(idx, hidden)
                    top = net.retrieve(queries[idx], keys, allowed, steps=steps).argmax(dim=-1)
                    same_f = focus[top] == focus[idx]
                    sums["exact@1"] += float((top == idx).sum())
                    sums["focus@1"] += float(same_f.sum())
                    sums["type@1"] += float((same_f & (qtype[top] == qtype[idx])).sum())
                keep = ("exact@1", "type@1") if mode == "own" else ("focus@1", "type@1")
                out.update({f"{mode} {k}": round(sums[k] / len(idx_all), 4) for k in keep})
        return out

    report = {"options": asdict(opts), "device": device,
              "objective": "та же болезнь и тип вопроса; свой ответ скрыт в доле hidden_share примеров",
              "train_queries": len(all_idx), "memory": len(tr),
              "baseline_cosine": report_metrics(0), "baseline_hopfield": report_metrics(opts.steps), "epochs": []}
    print(f"Обучение на {device}: {len(all_idx)} вопросов, память {len(tr)} образов ({opts.memory}), "
          f"свой ответ скрыт в {opts.hidden_share:.0%} примеров", flush=True)
    print(f"  до обучения, косинус:  {report['baseline_cosine']}", flush=True)
    print(f"  до обучения, Хопфилд:  {report['baseline_hopfield']}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    t0 = time.time()
    for epoch in range(1, opts.epochs + 1):
        perm = all_idx[torch.randperm(len(all_idx), generator=gen).to(device)]
        total, batches = 0.0, 0
        for s in range(0, len(perm), opts.batch_size):
            idx = perm[s:s + opts.batch_size]
            coin = torch.rand(len(idx), generator=gen).to(device) < opts.hidden_share
            hidden = coin & can_hide[idx]
            allowed, positive = masks(idx, hidden)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                keys = net.keys(memory)
                logits = net.retrieve(queries[idx], keys, allowed, steps=opts.steps)
            logits = logits.float()
            loss = (torch.logsumexp(logits, dim=-1)
                    - torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=-1)).mean()
            loss = loss + opts.identity_decay * (((net.w_q - eye) ** 2).sum() + ((net.w_k - eye) ** 2).sum())
            optim.zero_grad()
            loss.backward()
            optim.step()
            total, batches = total + float(loss.detach()), batches + 1
        fit = report_metrics(opts.steps)
        report["epochs"].append({"epoch": epoch, "loss": round(total / batches, 4), "beta": round(net.beta, 2), **fit})
        print(f"  эпоха {epoch}: loss {total / batches:.4f}, β {net.beta:.1f}, {fit}", flush=True)
    for p in net.parameters():
        p.requires_grad_(False)
    report["final"] = report["epochs"][-1]
    report["trained_steps0"] = report_metrics(0)
    report["train_seconds"] = round(time.time() - t0)
    net.save(opts.out, {"steps": opts.steps, "beta": round(net.beta, 3), "memory": opts.memory,
                        "embed_model": cache["meta"]["model"], "dim": cache["meta"]["dim"],
                        "objective": report["objective"], "data": "весь датасет", "final": report["final"],
                        "trained": time.strftime("%Y-%m-%d %H:%M")})
    (Path(opts.out) / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(f"Готово: {report['final']}. Сохранено в {opts.out}", flush=True)
    return report


# ---------------------------------------------------------------- модель для ответов


class HopfieldQA:
    """Отвечает готовым ответом из памяти. Подходит и как поиск для RAG (те же поля, что у RAGIndex.search)."""

    def __init__(self, model_dir: Path | str | None = HOPFIELD_DIR, cache_dir: Path | str = HOPFIELD_CACHE_DIR,
                 device: str | None = None, steps: int | None = None, beta: float | None = None,
                 memory: str | None = None):
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.device_count() else "cpu")
        cache = load_cache(cache_dir)
        self.meta = cache["meta"]
        self.passages = cache["passages"]
        if model_dir is None:                     # необученная сеть: W = I; β=50 — при меньшей запрос «размывается»
            self.net = HopfieldNet(cache["memory"].shape[1], beta=50.0, device=self.device)
            info, self.trained = {"steps": 1, "memory": "answer"}, False
        else:
            self.net, info = HopfieldNet.load(model_dir, self.device)
            self.trained = True
        # чем обучена сеть (старые сети — только ответы); memory= переопределяет
        self.memory_kind = memory or info.get("memory", "answer")
        memory = torch.tensor(_memory(cache, self.memory_kind), dtype=torch.float32, device=self.device)
        self.steps = info.get("steps", 1) if steps is None else steps
        self.beta = beta
        self.keys = self.net.keys(memory)
        self.mem_lang = torch.tensor(self.passages["lang"].map(LANG_ID).to_numpy(), device=self.device)
        self._model = None

    @property
    def model(self):
        if self._model is None:   # bge-m3 грузится при первом текстовом вопросе
            self._model = _encoder(self.meta["model"], self.device)
        return self._model

    def search_vectors(self, vectors: np.ndarray, langs: list[str], k: int = 3,
                       hide: np.ndarray | None = None) -> list[list[dict]]:
        """Для готовых векторов вопросов — k ответов с вероятностями p (max p — уверенность).

        hide — по одному индексу образа памяти на вопрос, который нельзя возвращать: так при проверке
        по датасету скрывается собственный ответ вопроса."""
        torch = _torch()
        q = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=self.device)
        lang_ids = torch.tensor([LANG_ID[lang] for lang in langs], device=self.device)
        hide_t = None if hide is None else torch.tensor(np.asarray(hide), device=self.device)
        all_ids = torch.arange(len(self.passages), device=self.device)
        results = []
        with torch.no_grad():
            for s in range(0, len(q), 512):
                mask = self.mem_lang[None, :] == lang_ids[s:s + 512, None]
                if hide_t is not None:
                    mask &= all_ids[None, :] != hide_t[s:s + 512, None]
                logits = self.net.retrieve(q[s:s + 512], self.keys, mask, steps=self.steps, beta=self.beta)
                probs = torch.softmax(logits, dim=-1)
                top_p, top_i = torch.topk(probs, k, dim=-1)
                for pv, iv in zip(top_p.cpu().numpy(), top_i.cpu().numpy()):
                    rows = self.passages.iloc[iv]
                    results.append([{"score": float(p), "question": r.question, "answer": r.answer,
                                     "row": int(r.row), "focus": r.focus, "qtype": r.qtype}
                                    for p, r in zip(pv, rows.itertuples())])
        return results

    def search_many(self, queries: list[tuple[str, str]], k: int = 3) -> list[list[dict]]:
        vectors = self.model.encode([q for q, _ in queries], normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=32)
        return self.search_vectors(vectors, [lang for _, lang in queries], k)

    def search(self, question: str, lang: str, k: int = 3) -> list[dict]:
        return self.search_many([(question, lang)], k)[0]

    def answer(self, question: str, lang: str, min_confidence: float = 0.0) -> dict:
        """Ответ модели: лучший образ памяти; при уверенности ниже порога — отказ (answer=None)."""
        best = self.search(question, lang, k=1)[0]
        return {"answer": best["answer"] if best["score"] >= min_confidence else None,
                "confidence": best["score"], "matched_question": best["question"], "row": best["row"]}


# ---------------------------------------------------------------- проверка по датасету


def _rag_hits(qs: pd.DataFrame, vecs: np.ndarray, focus_of: dict, type_of: dict, hide_own: bool, k: int = 3) -> list[list[dict]] | None:
    """Для сравнения: RAG (косинус по парам «вопрос + начало ответа», data/rag_index/)."""
    from .rag import RAGIndex

    try:
        rag = RAGIndex()
    except FileNotFoundError:
        return None
    if "row" not in rag.passages:
        raise FileNotFoundError("Индекс RAG старый (только train) — пересоберите: python -m medqa rag build")
    torch = _torch()
    device = "cuda" if torch.cuda.device_count() else "cpu"
    rag_rows = rag.passages["row"].to_numpy()
    questions, answers = rag.passages["question"].tolist(), rag.passages["answer"].tolist()
    q_langs, q_rows = qs["lang"].to_numpy(), qs["row"].to_numpy()
    out: list = [None] * len(qs)
    with torch.no_grad():   # пакетами на видеокарте: скалярные произведения + top-k внутри языка
        for lang in LANGS:
            idx = rag._by_lang[lang]
            emb = torch.tensor(rag.embeddings[idx], device=device)
            rows_t = torch.tensor(rag_rows[idx], device=device)
            q_idx = np.flatnonzero(q_langs == lang)
            for s in range(0, len(q_idx), 1024):
                part = q_idx[s:s + 1024]
                scores = torch.tensor(vecs[part].astype(np.float32), device=device) @ emb.T
                if hide_own:
                    own = torch.tensor(q_rows[part], device=device)
                    scores = scores.masked_fill(rows_t[None, :] == own[:, None], float("-inf"))
                top_s, top_i = torch.topk(scores, k, dim=-1)
                for j, (sv, iv) in zip(part, zip(top_s.cpu().numpy(), top_i.cpu().numpy())):
                    out[j] = [{"score": float(sc), "row": int(rag_rows[idx[t]]),
                               "focus": focus_of.get((int(rag_rows[idx[t]]), lang)),
                               "qtype": type_of.get(int(rag_rows[idx[t]])),
                               "question": questions[idx[t]], "answer": answers[idx[t]]} for sc, t in zip(sv, iv)]
    return out


def _subset(cache: dict, rows: int | None, langs: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Вопросы для проверки: все (или первые rows строк датасета, только языки langs). Индекс = номер образа."""
    p = cache["passages"]
    keep = p["lang"].isin(langs or LANGS) & ((p["row"] < rows) if rows else True)
    return p[keep], cache["questions"][keep.to_numpy()]


def evaluate(rows: int | None = None, out_dir: Path | str | None = None, chrf: bool = True) -> dict:
    """Проверка по всему датасету (validation нет) в двух режимах:

    - hidden — свой ответ вопроса скрыт: можно вспомнить только ответ про ту же болезнь на другой вопрос
      (focus@1, focus@3; chrF к эталону). Честная оценка обобщения на перефразированные вопросы;
    - recall — свой ответ в памяти: exact@1 — вспомнен ровно ответ на этот вопрос.
    Варианты: rag_index — RAG (пары «вопрос + ответ»); cosine — косинус по ответам; hopfield — необученная
    сеть (β=50, 1 шаг); hopfield_trained — обученная.
    """
    from .config import HOPFIELD_RESULTS

    cache = load_cache()
    passages = cache["passages"]
    qs, vecs = _subset(cache, rows)
    own = qs.index.to_numpy()
    focus_of = dict(zip(zip(passages["row"], passages["lang"]), passages["focus"]))
    type_of = dict(zip(passages["row"], passages["qtype"]))
    groups = pd.Series(list(zip(passages["focus"], passages["lang"])))
    has_other = (groups.map(groups.value_counts()) > 1).to_numpy()[own]
    variants = {"rag_index": None, "cosine": dict(model_dir=None, steps=0),
                "cosine_qa": dict(model_dir=None, steps=0, memory="qa"), "hopfield": dict(model_dir=None, steps=1)}
    if (Path(HOPFIELD_DIR) / "hopfield.pt").exists():
        variants["hopfield_trained"] = dict(model_dir=HOPFIELD_DIR)
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None

    records, summary = [], {}
    for name, kw in variants.items():
        qa = None if kw is None else HopfieldQA(**kw)
        for mode in ("hidden", "recall"):
            hide = own if mode == "hidden" else None
            hits = (_rag_hits(qs, vecs, focus_of, type_of, mode == "hidden") if qa is None
                    else qa.search_vectors(vecs, qs["lang"].tolist(), k=3, hide=hide))
            if hits is None:
                break
            for (_, v), h in zip(qs.iterrows(), hits):
                top = h[0]
                rec = {"variant": name, "mode": mode, "lang": v.lang, "row": int(v.row), "question": v.question,
                       "answer": top["answer"][:500], "matched_question": top["question"],
                       "confidence": round(top["score"], 4), "exact@1": top["row"] == v.row,
                       "focus@1": top["focus"] == v.focus, "focus@3": any(x["focus"] == v.focus for x in h),
                       "type@1": top["focus"] == v.focus and top["qtype"] == v.qtype}
                if chrf and sacrebleu is not None and mode == "hidden":
                    rec["chrF"] = round(sacrebleu.sentence_chrf(top["answer"][:3000], [v.answer[:3000]]).score, 2)
                records.append(rec)
        print(f"  {name}: готово ({len(qs)} вопросов × 2 режима)", flush=True)

    df = pd.DataFrame(records)
    for name, g in df.groupby("variant", sort=False):
        hid, rec = g[g["mode"] == "hidden"], g[g["mode"] == "recall"]
        hm = [m for m in ("focus@1", "type@1", "focus@3", "chrF", "confidence") if m in hid and hid[m].notna().any()]
        rm = ["exact@1", "type@1", "confidence"]
        summary[name] = {
            "hidden": {"all": hid[hm].mean().round(4).to_dict(),
                       **{lang: gl[hm].mean().round(4).to_dict() for lang, gl in hid.groupby("lang")}},
            "recall": {"all": rec[rm].mean().round(4).to_dict(),
                       **{lang: gl[rm].mean().round(4).to_dict() for lang, gl in rec.groupby("lang")}},
        }
        if name != "rag_index":   # у rag_index уверенность — косинус, не вероятность
            for thr in (0.3, 0.5):
                sure = hid[hid["confidence"] >= thr]
                summary[name]["hidden"][f"уверенность≥{thr}"] = {
                    "доля вопросов": round(len(sure) / len(hid), 4),
                    "focus@1": round(float(sure["focus@1"].mean()), 4) if len(sure) else None}
    summary["_note"] = {"вопросов (строк × языков)": len(qs),
                        "focus@1 максимум при скрытом своём ответе": round(float(has_other.mean()), 4),
                        "внимание": "обученная сеть видела эти вопросы при обучении — отдельной validation нет"}
    out_dir = Path(out_dir) if out_dir else HOPFIELD_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_validation.jsonl").unlink(missing_ok=True)   # старый формат
    df.to_json(out_dir / "eval_dataset.jsonl", orient="records", lines=True, force_ascii=False)
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_csv(path: Path | str | None = None, langs: list[str] | None = None, rows: int | None = None,
               model_dir: Path | str | None = HOPFIELD_DIR, hide_own: bool = True) -> Path:
    """CSV для сравнения с реальными ответами: вопрос датасета, эталон и ответ Hopfield-QA бок о бок.

    hide_own=True — свой ответ скрыт (иначе сеть почти всегда вернёт сам эталон).
    Кодировка utf-8-sig (как doctor_sheet.csv) — Excel открывает кириллицу и казахские буквы без настройки.
    """
    from .config import HOPFIELD_RESULTS

    cache = load_cache()
    qs, vecs = _subset(cache, rows, langs)
    qa = HopfieldQA(model_dir=model_dir)
    hits = qa.search_vectors(vecs, qs["lang"].tolist(), k=1, hide=qs.index.to_numpy() if hide_own else None)
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None

    out = []
    for (_, v), (top,) in zip(qs.iterrows(), hits):
        out.append({
            "id": f"row{int(v.row)}_{v.lang}", "язык": v.lang, "вопрос": v.question,
            "реальный ответ (эталон)": v.answer, "ответ Hopfield": top["answer"],
            "вспомненный вопрос": top["question"], "уверенность": round(top["score"], 3),
            "болезнь (эталон)": v.focus, "болезнь (ответ Hopfield)": top["focus"],
            "болезнь совпала": "да" if top["focus"] == v.focus else "нет",
            "chrF": round(sacrebleu.sentence_chrf(top["answer"][:3000], [v.answer[:3000]]).score, 1)
            if sacrebleu is not None else None,
        })
    path = Path(path) if path else HOPFIELD_RESULTS / "hopfield_vs_reference.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False, encoding="utf-8-sig")
    return path
