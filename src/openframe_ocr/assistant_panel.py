"""Seção opcional de resumos locais, integrada à rolagem da transcrição."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from openframe_ocr.document import Document
from openframe_ocr.local_llm import LocalLLMClient
from openframe_ocr.llm_runtime import LLMRuntime, RuntimeState


class _WrappedLabel(QLabel):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setMinimumHeight(max(0, self.heightForWidth(event.size().width())))


class AssistantPanel(QWidget):
    expanded = Signal()

    def __init__(self, client: LocalLLMClient, runtime: LLMRuntime, parent=None):
        super().__init__(parent)
        self.client = client
        self.runtime = runtime
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
        self.runtime_status = self._label(runtime.status, body)
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
        self._label("Somente a transcrição atual é usada no resumo local.", body)
        layout.addWidget(self.body)
        self.body.hide()
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
        self.runtime_status.setText(self.runtime.status)
        self.summarize_button.setEnabled(
            available and document is not None and bool(document.text.strip())
            and self.runtime.can_request and not client.busy
        )
        if document is None:
            message = "Selecione um documento para resumir."
        elif document.summary_status == "running":
            message = (
                "Preparando IA para resumir esta transcrição..."
                if self.runtime.state == RuntimeState.STARTING else "Gerando resumo desta transcrição..."
            )
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
