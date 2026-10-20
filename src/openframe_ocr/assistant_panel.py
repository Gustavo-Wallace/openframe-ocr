"""Seção opcional de resumos locais, integrada à rolagem da transcrição."""

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy,
    QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from openframe_ocr.document import Document
from openframe_ocr.local_llm import LocalLLMClient


class _WrappedLabel(QLabel):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setMinimumHeight(max(0, self.heightForWidth(event.size().width())))


class AssistantPanel(QWidget):
    expanded = Signal()

    def __init__(self, client: LocalLLMClient, parent=None):
        super().__init__(parent)
        self.client = client
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        title = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setAccessibleName("Mostrar ou ocultar assistente de IA")
        self.toggle.setToolTip("Mostrar ou ocultar assistente de IA")
        self.toggle.toggled.connect(self._toggle)
        title.addWidget(self.toggle)
        title.addWidget(QLabel("Assistente de IA"), stretch=1)
        layout.addLayout(title)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        connection = QHBoxLayout()
        port_label = QLabel("Porta local:")
        connection.addWidget(port_label)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(client.port)
        self.port.setAccessibleName("Porta do servidor local")
        port_label.setBuddy(self.port)
        self.port.valueChanged.connect(client.set_port)
        connection.addWidget(self.port)
        self.connect_button = QPushButton("Conectar")
        self.connect_button.clicked.connect(lambda: client.connect_server(self.port.value()))
        connection.addWidget(self.connect_button)
        connection.addStretch()
        body.addLayout(connection)
        self.connection_status = self._label(client.status, body)
        self.model_row = QWidget()
        models = QHBoxLayout(self.model_row)
        models.setContentsMargins(0, 0, 0, 0)
        models.addWidget(QLabel("Modelo:"))
        self.model = QComboBox()
        self.model.setMinimumWidth(0)
        self.model.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.model.setAccessibleName("Modelo local para resumo")
        models.addWidget(self.model, stretch=1)
        body.addWidget(self.model_row)
        self.summarize_button = QPushButton("Resumir documento")
        body.addWidget(self.summarize_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.status = self._label("Selecione um documento para resumir.", body)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setAccessibleName("Resumo do documento")
        self.summary.setPlaceholderText("O resumo aparecerá aqui.")
        self.summary.setMinimumHeight(90)
        self.summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body.addWidget(self.summary)
        self._label("Resumo gerado por IA local. Revise as informações.", body)
        self._label("Somente a transcrição atual é enviada ao servidor local.", body)
        layout.addWidget(self.body)
        self.body.hide()
        self.model_row.hide()
        self.summarize_button.setEnabled(False)

    def _label(self, text, layout):
        label = _WrappedLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(label)
        return label

    def _toggle(self, checked):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.body.setVisible(checked)
        if checked:
            self.expanded.emit()

    def update_state(self, document: Document | None, *, available=True):
        client = self.client
        self.port.setEnabled(available and not client.busy)
        self.connect_button.setEnabled(available and not client.busy)
        self.model.setEnabled(available and not client.busy)
        current_models = tuple(self.model.itemData(index) for index in range(self.model.count()))
        if current_models != client.models:
            with QSignalBlocker(self.model):
                self.model.clear()
                for model in client.models:
                    self.model.addItem(model, model)
        self.model_row.setVisible(len(client.models) > 1)
        self.connection_status.setText(client.status)
        self.summarize_button.setEnabled(
            available and document is not None and bool(document.text.strip())
            and bool(client.models) and not client.busy
        )
        if document is None:
            message = "Selecione um documento para resumir."
        elif document.summary_status == "running":
            message = "Gerando resumo desta transcrição..."
            if document.summary_stale:
                message += " O resumo anterior está desatualizado."
        elif document.summary_stale:
            message = "Resumo desatualizado. O texto foi alterado; gere um novo resumo."
            if document.summary_error:
                message += " " + document.summary_error
        elif document.summary_error:
            message = document.summary_error
            if document.summary:
                message += " Resumo anterior preservado."
        elif document.summary:
            message = "Resumo concluído. Revise as informações."
        elif not document.text.strip():
            message = "Extraia ou edite a transcrição antes de resumir."
        else:
            message = "Nenhum resumo gerado para este documento."
        self.status.setText(message)
        summary = document.summary if document else ""
        if self.summary.toPlainText() != summary:
            self.summary.setPlainText(summary)
