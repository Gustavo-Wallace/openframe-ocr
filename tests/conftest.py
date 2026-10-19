"""Fixtures mínimas para testes de interface sem uma tela conectada."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

from openframe_ocr.app import create_application
from openframe_ocr.main_window import MainWindow


@pytest.fixture(scope="session")
def application():
    return create_application([])


@pytest.fixture
def window(application):
    main_window = MainWindow()
    yield main_window
    main_window.close()
