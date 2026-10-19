"""Edição de campos por documento, sem sincronização automática com a transcrição."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from openframe_ocr.document import Document
from openframe_ocr.fields import StructuredField, extract_fields


class _FieldRow(QWidget):
    """Distribui os controles em duas linhas quando há pouca largura."""

    def __init__(self, label, value, copy, remove):
        super().__init__()
        self._controls = (label, value, copy, remove)
        self._stacked = None
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(6)
        self._layout.setVerticalSpacing(4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._arrange()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._arrange()

    def _arrange(self) -> None:
        stacked = self.width() < 480
        if self._stacked == stacked:
            return
        self._stacked = stacked
        for control in self._controls:
            self._layout.removeWidget(control)
        label, value, copy, remove = self._controls
        if stacked:
            self._layout.addWidget(label, 0, 0, 1, 3)
            for column, control in enumerate((value, copy, remove)):
                self._layout.addWidget(control, 1, column)
        else:
            for column, control in enumerate(self._controls):
                self._layout.addWidget(control, 0, column)
        for column, stretch in enumerate((1, 0, 0, 0) if stacked else (2, 3, 0, 0)):
            self._layout.setColumnStretch(column, stretch)
        self.updateGeometry()


class FieldsPanel(QWidget):
    """Apresenta apenas o documento selecionado; o modelo conserva os ajustes."""

    expanded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document: Document | None = None
        self._available = True
        self._running = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._title = QWidget()
        title_layout = QHBoxLayout(self._title)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setAccessibleName("Mostrar ou ocultar campos identificados")
        self.toggle.setToolTip("Mostrar ou ocultar campos identificados")
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.toggled.connect(self._toggle)
        title_layout.addWidget(self.toggle)
        title = QLabel("Campos identificados")
        title.setWordWrap(True)
        title_layout.addWidget(title, stretch=1)
        self._actions.addWidget(self._title, stretch=1)
        self.identify_button = QPushButton("Identificar campos")
        self.identify_button.clicked.connect(self._identify)
        self._actions.addWidget(self.identify_button)
        layout.addLayout(self._actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(6)
        body_layout.addLayout(self.rows)
        self.add_button = QPushButton("Adicionar campo")
        self.add_button.clicked.connect(self._add)
        body_layout.addWidget(self.add_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.body)
        self.body.hide()
        self.update_state()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        required = self._title.sizeHint().width() + self.identify_button.sizeHint().width() + 6
        self._actions.setDirection(
            QBoxLayout.Direction.TopToBottom if self.width() < required
            else QBoxLayout.Direction.LeftToRight
        )

    def set_document(self, document: Document | None) -> None:
        if self.document is not document:
            self.document = document
            self._rebuild_rows()
            self.toggle.setChecked(bool(document and document.fields))
        self.update_state()

    def update_state(self, *, available: bool | None = None, running: bool | None = None) -> None:
        if available is not None:
            self._available = available
        if running is not None:
            self._running = running
        document = self.document
        self.identify_button.setEnabled(
            document is not None and bool(document.text.strip())
            and self._available and not self._running
        )
        self.body.setEnabled(document is not None and self._available)
        if document is None:
            message = "Selecione um documento para identificar campos."
        elif document.fields_stale:
            message = "Texto alterado. Os campos podem estar desatualizados."
        elif document.fields:
            message = "Revise os campos. Os valores não foram verificados."
        else:
            message = "Nenhum campo identificado."
        self.status.setText(message)

    def _toggle(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if checked:
            self.expanded.emit()

    def _confirm_replace(self) -> bool:
        message = QMessageBox(self)
        message.setWindowTitle("Substituir campos editados")
        message.setIcon(QMessageBox.Icon.Question)
        message.setText("Identificar os campos novamente?")
        message.setInformativeText(
            "Os ajustes e campos adicionados ou removidos manualmente serão substituídos."
        )
        replace = message.addButton("Substituir campos", QMessageBox.ButtonRole.DestructiveRole)
        cancel = message.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(cancel)
        message.setEscapeButton(cancel)
        message.exec()
        return message.clickedButton() is replace

    def _identify(self) -> None:
        document = self.document
        if not self.identify_button.isEnabled() or document is None:
            return
        if document.fields_modified and not self._confirm_replace():
            return
        if self.document is not document or not self.identify_button.isEnabled():
            return
        document.fields = extract_fields(document.text)
        document.fields_source_text = document.text
        document.fields_modified = False
        self._rebuild_rows()
        self.toggle.setChecked(True)
        self.update_state()

    def _add(self) -> None:
        document = self.document
        if document is None or not self._available:
            return
        if document.fields_source_text is None:
            document.fields_source_text = document.text
        document.fields.append(StructuredField("", ""))
        document.fields_modified = True
        self._rebuild_rows()
        self.toggle.setChecked(True)
        self.rows.itemAt(self.rows.count() - 1).widget().findChild(QLineEdit).setFocus()
        self.update_state()

    def _remove(self, document: Document, field: StructuredField) -> None:
        if self.document is not document or not self._available:
            return
        document.fields = [item for item in document.fields if item.id != field.id]
        document.fields_modified = True
        self._rebuild_rows()
        self.update_state()

    def _rebuild_rows(self) -> None:
        while self.rows.count():
            widget = self.rows.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        document = self.document
        if document is None:
            return
        for index, field in enumerate(document.fields, start=1):
            self.rows.addWidget(self._make_row(document, field, index))

    def _make_row(self, document: Document, field: StructuredField, index: int) -> QWidget:
        label = QLineEdit(field.label)
        value = QLineEdit(field.value)
        for editor, caption in [(label, "Rótulo"), (value, "Valor")]:
            editor.setPlaceholderText(caption)
            editor.setAccessibleName(f"{caption} do campo {index}")
            editor.setMinimumWidth(0)
            editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        copy = QPushButton("Copiar")
        copy.setAccessibleName(f"Copiar valor do campo {index}")
        copy.setEnabled(bool(field.value))
        remove = QPushButton("Remover")
        remove.setAccessibleName(f"Remover campo {index}")

        def is_current():
            return self.document is document and any(item.id == field.id for item in document.fields)

        def edited():
            if not is_current():
                return
            field.label, field.value = label.text(), value.text()
            document.fields_modified = True
            copy.setEnabled(bool(field.value))
            self.update_state()

        def copy_value():
            if is_current() and value.text():
                QApplication.clipboard().setText(value.text())

        label.textChanged.connect(edited)
        value.textChanged.connect(edited)
        copy.clicked.connect(copy_value)
        remove.clicked.connect(lambda: self._remove(document, field))
        row = _FieldRow(label, value, copy, remove)
        row.setObjectName(field.id)
        return row
