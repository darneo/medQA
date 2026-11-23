"""Saluki 27B через llama.cpp: запуск llama-server и запросы к нему.

Saluki — готовая модель без дообучения (GGUF, 2 бита, 7.9 ГБ, основа Qwen3.8-27B).
llama-server сам делит слои между видеокартой и оперативной памятью, если модель
целиком не помещается в видеопамять (на 8 ГБ часть слоёв уходит в ОЗУ).

    with SalukiServer() as server:                       # запустить сервер (или взять уже запущенный)
        print(ask("Что такое астма?", "ru", url=server.url)["answer"])
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import LLAMA_CPP_DIR, ROOT, SALUKI_MODEL
from .prompts import build_messages
from .text import localize_emergency, strip_think

LOG_PATH = ROOT / "logs" / "llama_server.log"


def find_llama_server() -> Path:
    """llama-server из tools/llama_cpp/ (любая вложенная папка) или из PATH."""
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    if LLAMA_CPP_DIR.exists():
        found = sorted(LLAMA_CPP_DIR.rglob(exe))
        if found:
            return found[0]
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError("Не найден llama-server: распакуйте сборку llama.cpp в tools/llama_cpp/ или добавьте в PATH")


@dataclass
class ServerConfig:
    """Параметры llama-server."""
    model: Path = SALUKI_MODEL
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 8192                 # весь контекст; на один параллельный запрос приходится ctx / parallel
    parallel: int = 2               # сколько запросов сервер считает одновременно
    kv_cache: str = "q8_0"          # сжатие KV-кэша: вдвое меньше памяти, чем f16
    gpu_layers: int | None = None   # None — llama-server сам подбирает, сколько слоёв влезет в видеопамять
    extra: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self, exe: Path) -> list[str]:
        cmd = [str(exe), "-m", str(self.model), "--host", self.host, "--port", str(self.port),
               "-c", str(self.ctx), "-np", str(self.parallel), "--jinja",
               "-fa", "on", "-ctk", self.kv_cache, "-ctv", self.kv_cache, "--no-webui",
               "--reasoning-budget", "0"]   # без рассуждений: модель отвечает сразу
        if self.gpu_layers is not None:
            cmd += ["-ngl", str(self.gpu_layers)]
        return cmd + list(self.extra)


def is_ready(url: str, timeout: float = 2) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


class SalukiServer:
    """Запускает llama-server и останавливает его при выходе из `with`.

    Если по адресу уже работает сервер, новый не запускается и при выходе ничего не останавливается.
    Лог сервера — logs/llama_server.log.
    """

    def __init__(self, config: ServerConfig | None = None, startup_timeout: float = 900, log_path: Path = LOG_PATH):
        self.config = config or ServerConfig()
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path)
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return self.config.url

    def start(self) -> "SalukiServer":
        if is_ready(self.url):
            print("Сервер уже работает:", self.url)
            return self
        if not Path(self.config.model).is_file():
            raise FileNotFoundError(f"Нет модели {self.config.model} — скачайте: python -m medqa saluki download")
        cmd = self.config.command(find_llama_server())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        print(f"Запуск llama-server (лог: {self.log_path.relative_to(ROOT)}) ...", flush=True)
        t0 = time.time()
        while not is_ready(self.url):
            if self.proc.poll() is not None:
                log.close()
                tail = "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                raise RuntimeError(f"llama-server завершился с кодом {self.proc.returncode}:\n{tail}")
            if time.time() - t0 > self.startup_timeout:
                self.stop()
                raise TimeoutError(f"llama-server не запустился за {self.startup_timeout:.0f} с")
            time.sleep(2)
        print(f"Сервер готов за {time.time() - t0:.0f} с: {self.url}", flush=True)
        return self

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self) -> "SalukiServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def chat(messages: list[dict], *, url: str, max_tokens: int = 384, temperature: float = 0.0, top_p: float = 0.9,
         repeat_penalty: float = 1.1, timeout: float = 900, session=None) -> dict:
    """Один запрос к /v1/chat/completions. Рассуждения (<think>) выключены и вырезаются."""
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
               "repeat_penalty": repeat_penalty, "cache_prompt": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    r = (session or requests).post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice, usage = data["choices"][0], data.get("usage", {})
    return {"answer": strip_think(choice["message"].get("content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def ask(question: str, lang: str, *, url: str, context: str | None = None, concise: bool = True, **kwargs) -> dict:
    """Вопрос Saluki на языке lang; context — справка RAG (или None). Номер 911 заменяется на 103/112."""
    result = chat(build_messages(question, lang, concise=concise, context=context), url=url, **kwargs)
    result["answer"] = localize_emergency(result["answer"], lang)
    return result


def download(repo: str | None = None, filename: str | None = None) -> Path:
    """Скачать GGUF Saluki (~7.9 ГБ) в models/saluki/."""
    from huggingface_hub import hf_hub_download

    from .config import SALUKI_FILE, SALUKI_REPO

    SALUKI_MODEL.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo or SALUKI_REPO, filename=filename or SALUKI_FILE,
                                local_dir=str(SALUKI_MODEL.parent)))
