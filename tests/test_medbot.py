"""medbot без видеокарты: маски loss и математика слияния адаптеров."""

import json

import pytest

from medqa.medbot import _answer_only_labels, merge_adapters


class QwenLikeTokenizer:
    """Мини-токенизатор: шаблон как у Qwen3, один символ = один токен."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        text = "".join(f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "")
                       + m["content"] + "<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, **_):
        ids = [ord(c) for c in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def test_loss_counts_only_answer_tokens():
    ids, labels = _answer_only_labels(QwenLikeTokenizer(), "Вопрос?", "Ответ", "ru", 10_000, include_prompt=False)
    kept = "".join(chr(t) for t, lab in zip(ids, labels) if lab != -100)
    assert kept == "Ответ<|im_end|>\n"  # подсказка и вопрос не учитываются


def test_loss_can_include_prompt_like_old_report():
    ids, labels = _answer_only_labels(QwenLikeTokenizer(), "Вопрос?", "Ответ", "ru", 10_000, include_prompt=True)
    assert labels == ids


def test_truncated_answer_leaves_nothing_to_score():
    ids, labels = _answer_only_labels(QwenLikeTokenizer(), "Вопрос?" * 50, "Ответ", "ru", 100, include_prompt=False)
    assert all(lab == -100 for lab in labels)


def _fake_adapter(path, torch, seed, r=4, alpha=4, d_in=6, d_out=5):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    path.mkdir(parents=True)
    save_file({"layer.lora_A.weight": torch.randn(r, d_in, generator=g),
               "layer.lora_B.weight": torch.randn(d_out, r, generator=g)}, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "use_rslora": False}))
    (path / "tokenizer.json").write_text("{}")
    return path


def _delta(state, config):
    return state["layer.lora_B.weight"] @ state["layer.lora_A.weight"] * (config["lora_alpha"] / config["r"])


def test_exact_merge_is_mean_of_updates(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    dirs = [_fake_adapter(tmp_path / f"s{i}", torch, seed=i) for i in range(3)]
    expected = sum(_delta(load_file(str(d / "adapter_model.safetensors")),
                          json.loads((d / "adapter_config.json").read_text())) for d in dirs) / 3

    out = merge_adapters(dirs, tmp_path / "exact", "exact")
    config = json.loads((out / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == (12, 12)
    assert torch.allclose(_delta(load_file(str(out / "adapter_model.safetensors")), config), expected, atol=1e-5)
    assert (out / "tokenizer.json").exists()

    old = merge_adapters(dirs, tmp_path / "mean", "mean")  # старый способ — заметно другой результат
    old_config = json.loads((old / "adapter_config.json").read_text())
    assert not torch.allclose(_delta(load_file(str(old / "adapter_model.safetensors")), old_config), expected,
                              atol=1e-2)
