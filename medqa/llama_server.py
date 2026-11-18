"""llama-server (llama.cpp): найти, запустить, дождаться готовности, остановить, задать вопрос."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import LLAMA_CPP_DIR, LOGS_DIR, SALUKI_FILE, SALUKI_MODEL, SALUKI_SIZE


def find_llama_server(explicit: str | Path | None = None) -> Path:
    """Ищет llama-server: явный путь -> tools/llama_cpp/ -> PATH."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"llama-server не найден: {path}")
    candidates = [LLAMA_CPP_DIR / "llama-server.exe", LLAMA_CPP_DIR / "llama-server"]
    candidates += sorted(LLAMA_CPP_DIR.glob("**/llama-server.exe")) + sorted(LLAMA_CPP_DIR.glob("**/llama-server"))
    for path in candidates:
        if path.is_file():
            return path
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(f"llama-server не найден. Положите сборку llama.cpp в {LLAMA_CPP_DIR} "
                            "или добавьте её в PATH.")


def find_model(explicit: str | Path | None = None) -> Path:
    """Путь к GGUF. Размер проверяется только у стандартного файла Saluki."""
    path = Path(explicit).expanduser().resolve() if explicit else SALUKI_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"Модель не найдена: {path}\nСкачайте: python -m medqa saluki download")
    if path.name == SALUKI_FILE and path.stat().st_size != SALUKI_SIZE:
        raise RuntimeError(f"Размер {path.name} {path.stat().st_size} байт, ожидался {SALUKI_SIZE}: "
                           "скачивание не завершено — повторите download.")
    return path


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return "<лог пуст>"


def health(base_url: str, timeout: float = 2) -> int | None:
    """HTTP-код /health: 200 — готов, 503 — грузит модель, None — недоступен."""
    import requests

    try:
        return requests.get(f"{base_url}/health", timeout=timeout).status_code
    except requests.RequestException:
        return None


def wait_ready(base_url: str, proc: subprocess.Popen | None = None, timeout_s: float = 900,
               log_path: Path | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server завершился с кодом {proc.returncode}.\n"
                               + (_tail(log_path) if log_path else ""))
        if health(base_url) == 200:
            return
        time.sleep(2)
    raise TimeoutError(f"llama-server не стал готов за {timeout_s:.0f} с.\n" + (_tail(log_path) if log_path else ""))


@dataclass
class ServerOptions:
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 4096
    fit: bool = True            # --fit on: llama.cpp сам делит слои между GPU и RAM
    ngl: int | None = None      # число слоёв на GPU (по умолчанию авто)
    model: str | None = None
    llama_server: str | None = None
    startup_timeout: float = 900
    log_name: str = "llama-server.log"
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def start_server(opts: ServerOptions) -> subprocess.Popen:
    server, model = find_llama_server(opts.llama_server), find_model(opts.model)
    cmd = [str(server), "-m", str(model), "--host", opts.host, "--port", str(opts.port),
           "--jinja", "-fa", "on", "-c", str(opts.ctx)]
    if opts.fit:
        cmd += ["--fit", "on"]
    if opts.ngl is not None:
        cmd += ["-ngl", str(opts.ngl)]
    cmd += opts.extra_args
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / opts.log_name
    print("Запуск:", " ".join(cmd))
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(server.parent), stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(f"Загрузка модели (лог: {log_path}) ...")
    try:
        wait_ready(opts.base_url, proc, opts.startup_timeout, log_path)
    except Exception:
        stop_server(proc)
        raise
    print(f"Сервер готов: {opts.base_url}")
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def ensure_server(opts: ServerOptions | None = None) -> subprocess.Popen | None:
    """Подключается к уже работающему серверу (или ждёт, пока он загрузится), иначе запускает свой.

    Возвращает процесс, если сервер запустили мы, иначе None — тогда его и не надо останавливать.
    """
    opts = opts or ServerOptions()
    status = health(opts.base_url)
    if status == 200:
        print(f"Использую уже запущенный сервер: {opts.base_url}")
        return None
    if status == 503:
        print("llama-server уже загружает модель — жду ...")
        wait_ready(opts.base_url, None, opts.startup_timeout)
        return None
    return start_server(opts)


def chat(session, base_url: str, messages: list[dict], *, max_tokens: int = 384, temperature: float = 0.2,
         top_p: float = 0.9, thinking: bool = False, timeout_s: float = 900) -> dict:
    """Один запрос к /v1/chat/completions. Возвращает answer, finish_reason и число токенов."""
    payload = {"model": "saluki", "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p, "stream": False}
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    url = f"{base_url}/v1/chat/completions"
    resp = session.post(url, json=payload, timeout=timeout_s)
    if resp.status_code == 400 and "chat_template_kwargs" in resp.text:  # старые сборки llama.cpp
        payload.pop("chat_template_kwargs", None)
        resp = session.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip() or (message.get("reasoning_content") or "").strip()
    usage = data.get("usage") or {}
    return {"answer": text, "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
