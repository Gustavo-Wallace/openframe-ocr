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
            "import openframe_ocr.image_loader; "
            "import openframe_ocr.camera; "
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
        "Documentos",
        "Pré-visualização",
    }


def test_initial_actions_have_correct_state(window, application):
    window.show()
    application.processEvents()
    buttons = window.findChildren(QPushButton)
    assert {button.text() for button in buttons if button.isVisible()} == {
        "Carregar imagens",
        "Abrir webcam",
        "Remover",
        "Limpar lista",
    }
    assert len(buttons) == 6
    assert window.import_button.isEnabled()
    assert window.webcam_button.isEnabled()
    assert not window.capture_button.isVisible()
    assert not window.capture_button.isEnabled()
    assert not window.close_camera_button.isVisible()
    assert not window.remove_button.isEnabled()
    assert not window.clear_button.isEnabled()
