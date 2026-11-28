"""Training curves for the defense (English labels): docs/figures/*.png.

    .venv\\Scripts\\python.exe scripts\\plot_training.py

Only logged values are plotted. The step-by-step log of the first medbot training (medbot_old, 4850 steps)
was not saved; its plot shows only the checkpoints recorded in docs/REPORT_medbot.txt, marked as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e8833a", "#2f9e6b", "#8a8f98"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def medbot_old() -> None:
    """Stage 1 (12 931 rows x 3 languages, held-out 10 %): only recorded checkpoints are available."""
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, loss = [0, 300, 4850], [2.17, 0.90, 0.58]
    ax.plot(steps, loss, "o--", color=BLUE, label="train loss (recorded checkpoints)")
    ax.axhline(0.6273, color=ORANGE, ls="-", label="validation loss after 1 epoch = 0.627 (held-out 10 %)")
    for s, v in zip(steps, loss):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(6, 6), fontsize=9)
    style(ax, "medbot_old (QLoRA, stage 1): loss", "training step (batch 8)", "cross-entropy loss")
    ax.text(0.99, 0.55, "full step log not saved —\nonly values from REPORT_medbot.txt", transform=ax.transAxes,
            ha="right", fontsize=8, color=GREY)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "medbot_old_loss.png", dpi=160)


def medbot_full() -> None:
    """Stage 2: continued training of the stage-1 adapter on the 10 % held-out rows (logged every 10 steps)."""
    r = json.loads((ROOT / "models/medbot_train_full/train_report.json").read_text(encoding="utf-8"))
    log = r["loss_log"]
    steps = [x["step"] for x in log]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(steps, [x["loss"] for x in log], color=BLUE, label="train loss")
    a1.plot(steps, [1 - x["mean_token_accuracy"] for x in log], color=GREEN, alpha=0.8,
            label="1 − token accuracy")
    style(a1, "medbot_full (stage 2) training", "step (batch 8)", "value")
    a1.legend(frameon=False, fontsize=9)
    a2.plot(steps, [x["learning_rate"] for x in log], color=ORANGE)
    style(a2, "learning rate (warmup + linear decay)", "step", "learning rate")
    fig.text(0.01, 0.01, f"{r['examples']} examples, {r['epochs']:.0f} epoch, peak LR {r['lr']}, "
             f"final train loss {r['train_metrics']['train_loss']:.3f}, "
             f"{r['train_metrics']['train_runtime'] / 60:.0f} min on RTX 5060", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "medbot_full_training.png", dpi=160)


def hopfield() -> None:
    r = json.loads((ROOT / "models/hopfield/train_report.json").read_text(encoding="utf-8"))
    ep = r["epochs"]
    x = [e["epoch"] for e in ep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(x, [e["loss"] for e in ep], "o-", color=BLUE)
    style(a1, "Hopfield-QA: training loss", "epoch", "loss")
    a2.plot(x, [e["hidden focus@1"] for e in ep], "o-", color=GREEN, label="trained: same disease @1 (own answer hidden)")
    a2.axhline(r["baseline_cosine"]["hidden focus@1"], color=GREY, ls="--",
               label=f"baseline cosine bge-m3 = {r['baseline_cosine']['hidden focus@1']:.3f}")
    a2.plot(x, [e["own exact@1"] for e in ep], "s-", color=ORANGE, label="own answer exact @1")
    a2.set_ylim(0.7, 1.0)
    style(a2, "Hopfield-QA: retrieval accuracy @1", "epoch", "accuracy")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hopfield_training.png", dpi=160)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    medbot_old()
    medbot_full()
    hopfield()
    print("saved:", *sorted(p.name for p in FIG.glob("*.png")))
