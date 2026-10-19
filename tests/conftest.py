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
    wait_for(lambda: main_window._camera is None and main_window._ocr_worker is None)


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
