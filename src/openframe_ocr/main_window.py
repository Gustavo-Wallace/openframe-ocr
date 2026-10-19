"""Janela inicial do OpenFrame OCR."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Apresenta o fluxo inicial, ainda sem carregamento de documentos."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenFrame OCR")
        self.resize(900, 600)
        self.setMinimumSize(640, 420)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        heading = QLabel("OpenFrame OCR")
        heading_font = heading.font()
        heading_font.setPointSize(heading_font.pointSize() + 8)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        description = QLabel(
            "O aplicativo permitirá extrair texto de documentos e imagens localmente."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        for text in ("Carregar imagens", "Abrir webcam"):
            button = QPushButton(text)
            button.setMinimumHeight(36)
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)

        placeholder = QFrame()
        placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_layout.setContentsMargins(24, 24, 24, 24)
        placeholder_layout.setSpacing(12)
        placeholder_layout.addStretch()

        empty_message = QLabel("Nenhum documento carregado")
        empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_message.setWordWrap(True)
        placeholder_layout.addWidget(empty_message)

        availability = QLabel(
            "Carregamento de imagens e webcam ainda não disponíveis."
        )
        availability.setAlignment(Qt.AlignmentFlag.AlignCenter)
        availability.setWordWrap(True)
        placeholder_layout.addWidget(availability)
        placeholder_layout.addStretch()
        layout.addWidget(placeholder, stretch=1)
