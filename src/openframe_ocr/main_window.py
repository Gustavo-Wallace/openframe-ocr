"""Janela principal com importação e pré-visualização de imagens locais."""

from collections.abc import Iterable
import os
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from openframe_ocr.image_loader import load_preview


class ImagePreview(QLabel):
    """Ajusta uma prévia limitada à área disponível, sem distorcer a imagem."""

    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setContentsMargins(12, 12, 12, 12)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.clear_image()

    def set_image(self, image: QImage) -> None:
        self._source = QPixmap.fromImage(image)
        self._fit_image()

    def clear_image(self, message: str = "Nenhum documento carregado") -> None:
        self._source = QPixmap()
        self.setText(message)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._source.isNull():
            self._fit_image()

    def _fit_image(self) -> None:
        self.setPixmap(
            self._source.scaled(
                self.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    """Mantém os caminhos dos documentos e uma única prévia selecionada."""

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
        self.import_button = QPushButton("Carregar imagens")
        self.import_button.setMinimumHeight(36)
        self.import_button.clicked.connect(self._choose_images)
        actions.addWidget(self.import_button)
        self.webcam_button = QPushButton("Abrir webcam")
        self.webcam_button.setMinimumHeight(36)
        self.webcam_button.setEnabled(False)
        actions.addWidget(self.webcam_button)
        actions.addStretch()
        layout.addLayout(actions)

        documents_panel = QWidget()
        documents_layout = QVBoxLayout(documents_panel)
        documents_layout.setContentsMargins(0, 0, 0, 0)
        documents_layout.addWidget(QLabel("Documentos"))
        self.document_list = QListWidget()
        self.document_list.setMinimumWidth(160)
        self.document_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.document_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.document_list.currentItemChanged.connect(self._show_selected)
        documents_layout.addWidget(self.document_list, stretch=1)

        removal_actions = QHBoxLayout()
        self.remove_button = QPushButton("Remover")
        self.remove_button.clicked.connect(self._remove_selected)
        removal_actions.addWidget(self.remove_button)
        self.clear_button = QPushButton("Limpar lista")
        self.clear_button.clicked.connect(self._clear_documents)
        removal_actions.addWidget(self.clear_button)
        documents_layout.addLayout(removal_actions)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("Pré-visualização"))
        self.preview = ImagePreview()
        preview_layout.addWidget(self.preview, stretch=1)

        workspace = QHBoxLayout()
        workspace.setSpacing(16)
        workspace.addWidget(documents_panel, stretch=1)
        workspace.addWidget(preview_panel, stretch=3)
        layout.addLayout(workspace, stretch=1)
        self._update_removal_controls()

    @property
    def document_paths(self) -> tuple[Path, ...]:
        """Caminhos originais disponíveis para o processamento posterior."""
        return tuple(
            Path(self.document_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.document_list.count())
        )

    @property
    def current_document_path(self) -> Path | None:
        item = self.document_list.currentItem()
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _choose_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Carregar imagens",
            "",
            "Imagens PNG e JPEG (*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG)",
        )
        if paths:
            self.add_images(paths)

    def add_images(self, paths: Iterable[str | Path]) -> None:
        """Acrescenta arquivos válidos sem perder documentos ou repetir caminhos."""
        known_paths = {os.path.normcase(str(path)) for path in self.document_paths}
        errors = []
        for filename in paths:
            try:
                path = Path(filename).resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                errors.append(f"{Path(filename).name}: arquivo indisponível para leitura.")
                continue
            key = os.path.normcase(str(path))
            if key in known_paths:
                continue
            try:
                # Valida a decodificação real e descarta a prévia dos itens inativos.
                load_preview(path)
            except ValueError as error:
                errors.append(f"{path.name}: {error}")
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.document_list.addItem(item)
            known_paths.add(key)

        if self.document_list.currentRow() < 0 and self.document_list.count():
            self.document_list.setCurrentRow(0)
        self._update_removal_controls()
        if errors:
            self._report_errors(errors)

    def _show_selected(self) -> None:
        self._update_removal_controls()
        path = self.current_document_path
        if path is None:
            self.preview.clear_image()
            return
        self.preview.clear_image()
        try:
            self.preview.set_image(load_preview(path))
        except ValueError as error:
            self.preview.clear_image("Não foi possível carregar esta imagem.")
            self._report_errors([f"{path.name}: {error}"])

    def _remove_selected(self) -> None:
        row = self.document_list.currentRow()
        if row < 0:
            return
        with QSignalBlocker(self.document_list):
            self.document_list.takeItem(row)
            if self.document_list.count():
                self.document_list.setCurrentRow(min(row, self.document_list.count() - 1))
        self._show_selected()

    def _clear_documents(self) -> None:
        with QSignalBlocker(self.document_list):
            self.document_list.clear()
        self._show_selected()

    def _update_removal_controls(self) -> None:
        self.remove_button.setEnabled(self.document_list.currentRow() >= 0)
        self.clear_button.setEnabled(self.document_list.count() > 0)

    def _report_errors(self, errors: list[str]) -> None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Imagens não carregadas")
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setText("Não foi possível carregar:\n\n" + "\n".join(errors))
        message.addButton("Fechar", QMessageBox.ButtonRole.AcceptRole)
        message.exec()
