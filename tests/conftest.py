"""Fixtures mínimas para testes de interface sem uma tela conectada."""

import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QImage

from openframe_ocr.app import create_application
from openframe_ocr.main_window import MainWindow


@pytest.fixture(scope="session")
def application():
    return create_application([])


@pytest.fixture
def wait_for(application):
    def wait(condition, timeout=3):
        if condition():
            return
        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(5)
        poll.timeout.connect(lambda: loop.quit() if condition() else None)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(int(timeout * 1000))
        loop.exec()
        poll.stop()
        deadline.stop()
        assert condition(), "A condição esperada não ocorreu dentro do prazo."

    return wait


@pytest.fixture
def window(application, wait_for):
    main_window = MainWindow()
    yield main_window
    main_window.close()
    wait_for(lambda: main_window._camera is None and main_window._ocr_worker is None
             and main_window._quality_worker is None and not main_window._llm.busy)


@pytest.fixture
def llm_server():
    """Servidor HTTP simulado, vinculado exclusivamente a 127.0.0.1."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json
    from threading import Event, Thread
    from types import SimpleNamespace

    state = SimpleNamespace(requests=[], started=Event(), gates={}, disconnect=False)
    state.responses = {
        "/v1/models": (200, {"data": [{"id": "modelo-sintetico"}]}, {}),
        "/v1/chat/completions": (200, {"choices": [{"message": {
            "content": "Documento sintético sobre uma solicitação de material em São Paulo."
        }, "finish_reason": "stop"}]}, {}),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append((self.command, self.path, json.loads(raw) if raw else None))
            state.started.set()
            gate = state.gates.get(self.path)
            if gate is not None and not gate.wait(5):
                return
            if state.disconnect:
                self.connection.close()
                return
            status, body, headers = state.responses[self.path]
            encoded = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state.port = server.server_port
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield state
    for gate in state.gates.values():
        gate.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


@pytest.fixture
def portuguese_text_image(application):
    from PIL import Image, ImageDraw, ImageFont

    font_candidates = [
        str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans.ttf",
    ]
    font = None
    for candidate in font_candidates:
        try:
            font = ImageFont.truetype(candidate, 48)
            break
        except OSError:
            continue
    if font is None:
        pytest.skip("Não há fonte disponível para gerar o texto português sintético.")
    image = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "Documento de teste",
        "Informações em português",
        "Ação, coração e São Paulo",
        "Café, açúcar e atenção",
        "Total: 123,45",
    ]
    for index, line in enumerate(lines):
        draw.text((70, 55 + index * 100), line, fill="black", font=font)
    return QImage(image.tobytes(), 1400, 620, 1400 * 3, QImage.Format.Format_RGB888).copy()
