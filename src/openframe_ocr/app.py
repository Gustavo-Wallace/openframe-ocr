"""Inicialização do Qt e execução do aplicativo."""

import sys

from PySide6.QtCore import QLibraryInfo, QTranslator
from PySide6.QtWidgets import QApplication

from openframe_ocr.main_window import MainWindow


def create_application(argv: list[str] | None = None) -> QApplication:
    """Inicializa o Qt sem abrir janelas ou executar o loop de eventos."""
    application = QApplication(sys.argv if argv is None else argv)
    application.setApplicationName("OpenFrame OCR")
    translator = QTranslator(application)
    if translator.load("qtbase_pt_BR", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        application.installTranslator(translator)
    return application


def main() -> int:
    """Exibe a janela principal e executa o loop de eventos do Qt."""
    application = create_application()
    window = MainWindow()
    window.show()
    return application.exec()
