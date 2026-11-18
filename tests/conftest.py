"""Общие фикстуры: датасет и фейковый llama-server (настоящая модель для тестов не нужна)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from medqa.data import load_dataset


class FakeLlamaHandler(BaseHTTPRequestHandler):
    """Отвечает как llama-server: /health и /v1/chat/completions.

    Ответ содержит пустой <think>-блок (как у Qwen3), а на вопрос про боль в груди — номер 911,
    чтобы проверять очистку и замену номера.
    """

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = req["messages"][0]["content"], req["messages"][1]["content"]
        text = "Позвоните 911." if "боль в груди" in user else f"FAKE: {user} | rag={'Ответ:' in system}"
        self._send(200, {"choices": [{"message": {"content": "<think>\n\n</think>\n\n" + text},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": len(system) // 4, "completion_tokens": 12}})


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()
