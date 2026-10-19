"""Verificações do escopo inicial do aplicativo."""

import subprocess
import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton


def test_imports_do_not_create_application():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from PySide6.QtWidgets import QApplication; "
            "import openframe_ocr; "
            "import openframe_ocr.app; "
            "import openframe_ocr.main_window; "
            "import openframe_ocr.__main__; "
            "assert QApplication.instance() is None",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_window_can_be_created_without_event_loop(application, window):
    assert QApplication.instance() is application
    assert isinstance(window, QMainWindow)
    assert window.centralWidget() is not None
    assert not window.isVisible()


def test_window_title(window):
    assert window.windowTitle() == "OpenFrame OCR"


def test_initial_portuguese_labels(window):
    labels = {label.text() for label in window.findChildren(QLabel)}
    assert labels == {
        "OpenFrame OCR",
        "O aplicativo permitirá extrair texto de documentos e imagens localmente.",
        "Nenhum documento carregado",
        "Carregamento de imagens e webcam ainda não disponíveis.",
    }


def test_unavailable_actions_are_visible_and_disabled(window, application):
    window.show()
    application.processEvents()
    buttons = window.findChildren(QPushButton)
    assert {button.text() for button in buttons} == {
        "Carregar imagens",
        "Abrir webcam",
    }
    assert len(buttons) == 2
    assert all(button.isVisible() and not button.isEnabled() for button in buttons)
