"""Apresentação compacta de resultados de qualidade por documento."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QToolButton,
    QVBoxLayout, QWidget,
)

from openframe_ocr.document import Document


class _WrappedLabel(QLabel):
    """Impede que o layout comprima linhas após reduzir a largura da janela."""

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        required = max(0, self.heightForWidth(event.size().width()))
        if self.minimumHeight() != required:
            self.setMinimumHeight(required)


class QualityPanel(QWidget):
    """Resultados opcionais, sem iniciar processamento ou alterar o documento."""

    expanded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document: Document | None = None
        self._shown_result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._title = QWidget()
        title_layout = QHBoxLayout(self._title)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setAccessibleName("Mostrar ou ocultar qualidade da imagem")
        self.toggle.setToolTip("Mostrar ou ocultar qualidade da imagem")
        self.toggle.toggled.connect(self._toggle)
        title_layout.addWidget(self.toggle)
        title = _WrappedLabel("Qualidade da imagem")
        title.setWordWrap(True)
        title_layout.addWidget(title, stretch=1)
        self._actions.addWidget(self._title, stretch=1)
        self.analyze_button = QPushButton("Analisar qualidade")
        self._actions.addWidget(self.analyze_button)
        layout.addLayout(self._actions)
        self.status = _WrappedLabel("Selecione um documento para analisar a qualidade.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.body = QWidget()
        self._details = QVBoxLayout(self.body)
        self._details.setContentsMargins(0, 0, 0, 0)
        self._details.setSpacing(6)
        layout.addWidget(self.body)
        self.body.hide()
        self.analyze_button.setEnabled(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        required = self._title.sizeHint().width() + self.analyze_button.sizeHint().width() + 6
        self._actions.setDirection(
            QBoxLayout.Direction.TopToBottom if self.width() < required
            else QBoxLayout.Direction.LeftToRight
        )

    def _toggle(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.body.setVisible(checked)
        if checked:
            self.expanded.emit()

    def set_document(self, document: Document | None, *, busy=False, available=True) -> None:
        changed = self.document is not document
        self.document = document
        result = document.quality_result if document else None
        if changed or result is not self._shown_result:
            self._render_result(result)
            self._shown_result = result
            self.toggle.setChecked(result is not None)
        self.analyze_button.setEnabled(document is not None and not busy and available)
        if not available:
            message = "Finalizando análise..."
        elif document is None:
            message = "Selecione um documento para analisar a qualidade."
        elif document.quality_status == "running":
            message = "Analisando qualidade..."
        elif document.quality_status == "error":
            message = "Falha na análise. " + document.quality_error
            if result is not None:
                message += " Resultado anterior preservado."
        elif result is not None:
            message = "Análise concluída. Leitura visual estimada: " + result.metrics[-1].status + "."
        else:
            message = "Ainda não analisada."
        if busy and document is not None and document.quality_status != "running":
            message += " Análise de outro documento em andamento."
        self.status.setText(message)

    def _label(self, text, layout):
        label = _WrappedLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(label)
        return label

    def _render_result(self, result) -> None:
        while self._details.count():
            widget = self._details.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        if result is None:
            return
        self._label(
            "Indicadores heurísticos em três níveis: Ruim, Atenção e Boa. "
            "Não representam a precisão do OCR.",
            self._details,
        )
        if result.recommendations:
            self._label("Sugestões:\n" + "\n".join("• " + text for text in result.recommendations), self._details)
        for metric in result.metrics:
            row = QWidget()
            row.setObjectName(metric.key)
            layout = QVBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)
            summary = QHBoxLayout()
            title = _WrappedLabel(metric.label + ": " + metric.status)
            title.setWordWrap(True)
            summary.addWidget(title, stretch=3)
            if metric.score is not None:
                bar = QProgressBar()
                bar.setRange(0, 3)
                bar.setValue({0: 1, 50: 2, 100: 3}[metric.score])
                bar.setFormat(metric.status)
                bar.setAccessibleName(metric.label + ": " + metric.status)
                bar.setAccessibleDescription("Indicador qualitativo em três níveis: Ruim, Atenção e Boa.")
                color = {0: "#e5aaa5", 50: "#ecc975", 100: "#8dcc9e"}[metric.score]
                title.setStyleSheet("color: " + {0: "#9c2727", 50: "#805b00", 100: "#225c33"}[metric.score])
                bar.setStyleSheet(
                    "QProgressBar { border: 1px solid #b8b8b8; border-radius: 3px; "
                    "background: #ededed; color: #111111; text-align: center; } "
                    "QProgressBar::chunk { background-color: " + color + "; border-radius: 2px; }"
                )
                summary.addWidget(bar, stretch=2)
            layout.addLayout(summary)
            self._label(metric.explanation, layout)
            self._details.addWidget(row)
