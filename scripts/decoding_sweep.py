"""Decoding settings chosen on the DEV split -> results/harness/decoding_sweep.{csv,md}.

    .venv\\Scripts\\python.exe scripts\\decoding_sweep.py

Runs: results/harness/medbot_old_dev_<tag>/ (scripts/run_harness.py --suite dev --tag _dev_<tag>), medbot_old on the
dev rows (held-out rows 150-249, never used for testing, never seen by medbot_old in training).
Selection rule (fixed before looking at the numbers): highest mean sentence chrF over all dev answers; ties broken
by semantic similarity. Only the winner is then run on the test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "harness" / "tasks" / "medquad"))
import utils  # noqa: E402
from harness_report import semantic_scores, set_dir  # noqa: E402
from medqa.evaluation import repeated_share  # noqa: E402
from medqa.text import tokens  # noqa: E402

OUT = ROOT / "results" / "harness"
SETTINGS = {"rp110": "default: repetition_penalty 1.1, max 512", "rp100": "repetition_penalty 1.0",
            "rp105": "repetition_penalty 1.05", "rp120": "repetition_penalty 1.2", "rp130": "repetition_penalty 1.3",
            "nr8": "no_repeat_ngram_size 8", "len768": "max 768 new tokens"}


def main() -> None:
    set_dir(OUT)
    rows = []
    for tag, desc in SETTINGS.items():
        files = sorted((OUT / f"medbot_old_dev_{tag}").glob("*/samples_medquad_dev_*.jsonl"))
        if not files:
            continue
        recs = []
        for f in files:
            for line in f.open(encoding="utf-8"):
                s = json.loads(line)
                a = utils.clean(s["resps"][0][0], s["doc"]["lang"])
                recs.append((s["doc"]["lang"], a, s["doc"]["ref"], s["rougeL"]))
        sem = semantic_scores([(a, r) for _, a, r, _ in recs])
        df = pd.DataFrame(recs, columns=["lang", "answer", "ref", "rougeL"]).assign(semantic=sem)
        df["chrf"] = [sacrebleu.sentence_chrf(a, [r]).score for a, r in zip(df.answer, df.ref)]
        df["rep"] = [repeated_share(tokens(a)) for a in df.answer]
        row = {"tag": tag, "setting": desc, "n": len(df), "chrF": df.chrf.mean(), "ROUGE-L": df.rougeL.mean(),
               "semantic": df.semantic.mean(), "repetition": df.rep.mean(), "loops>20%": (df.rep > 0.2).mean()}
        for lang, g in df.groupby("lang"):
            row[f"chrF {lang}"] = g.chrf.mean()
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["chrF", "semantic"], ascending=False)
    res.round(4).to_csv(OUT / "decoding_sweep.csv", index=False)
    cols = ["setting", "n", "chrF", "chrF en", "chrF ru", "chrF kk", "ROUGE-L", "semantic", "repetition", "loops>20%"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for d in res.to_dict("records"):
        vals = [d["setting"], str(d["n"])]
        vals += [f"{d.get(c, np.nan):.1f}" for c in ("chrF", "chrF en", "chrF ru", "chrF kk")]
        vals += [f"{d['ROUGE-L']:.3f}", f"{d['semantic']:.3f}", f"{d['repetition']:.1%}", f"{d['loops>20%']:.1%}"]
        lines.append("| " + " | ".join(vals) + " |")
    text = "\n".join(lines) + f"\n\nWinner on dev: **{res.iloc[0].setting}** (`{res.iloc[0].tag}`).\n"
    (OUT / "decoding_sweep.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
