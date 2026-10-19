"""Janela principal com importação e pré-visualização de imagens locais."""

from collections.abc import Iterable
import os
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from openframe_ocr.camera import CameraWorker
from openframe_ocr.document import Document
from openframe_ocr.fields_panel import FieldsPanel
from openframe_ocr.image_loader import MAX_PREVIEW_EDGE, load_preview
from openframe_ocr.ocr import OCREngine, OCRWorker


MAX_CAPTURE_MEMORY_BYTES = 128 * 1024 * 1024


class ImagePreview(QLabel):
    """Ajusta uma prévia limitada à área disponível, sem distorcer a imagem."""

    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setContentsMargins(12, 12, 12, 12)
        self.setMinimumSize(160, 60)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.clear_image()

    def set_image(self, image: QImage) -> None:
        if max(image.width(), image.height()) > MAX_PREVIEW_EDGE:
            image = image.scaled(
                MAX_PREVIEW_EDGE,
                MAX_PREVIEW_EDGE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
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
    """Apresenta documentos importados, capturas em memória e a prévia da webcam."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenFrame OCR")
        self.resize(900, 600)
        self.setMinimumSize(640, 420)
        self._camera: CameraWorker | None = None
        self._camera_stopping = False
        self._camera_error = ""
        self._latest_frame: QImage | None = None
        self._capture_count = 0
        self._closing = False
        self._ocr_engine: OCREngine | None = None
        self._ocr_worker: OCRWorker | None = None
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._update_camera_preview)

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
            "Extraia texto de documentos e imagens localmente."
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
        self.webcam_button.clicked.connect(self._start_camera)
        actions.addWidget(self.webcam_button)
        self.capture_button = QPushButton("Capturar imagem")
        self.capture_button.setMinimumHeight(36)
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self._capture_image)
        self.capture_button.hide()
        actions.addWidget(self.capture_button)
        self.close_camera_button = QPushButton("Fechar webcam")
        self.close_camera_button.setMinimumHeight(36)
        self.close_camera_button.clicked.connect(self._stop_camera)
        self.close_camera_button.hide()
        actions.addWidget(self.close_camera_button)
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
        self.preview.setMinimumHeight(150)
        preview_layout.addWidget(self.preview, stretch=1)

        text_panel = QWidget()
        text_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_layout = QVBoxLayout(text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(6)
        text_actions = QHBoxLayout()
        self.extract_button = QPushButton("Extrair texto")
        self.extract_button.clicked.connect(self._start_ocr)
        text_actions.addWidget(self.extract_button)
        self.copy_button = QPushButton("Copiar texto")
        self.copy_button.clicked.connect(self._copy_text)
        text_actions.addWidget(self.copy_button)
        text_actions.addStretch()
        text_layout.addLayout(text_actions)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("O texto extraído aparecerá aqui.")
        self.text_edit.setMinimumHeight(60)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.text_edit.textChanged.connect(self._text_edited)
        text_layout.addWidget(self.text_edit, stretch=1)
        self.ocr_status = QLabel()
        self.ocr_status.setWordWrap(True)
        text_layout.addWidget(self.ocr_status)
        self.fields_panel = FieldsPanel()
        self.fields_panel.expanded.connect(self._expand_fields_space)
        text_layout.addWidget(self.fields_panel)
        self.text_scroll = QScrollArea()
        self.text_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setWidget(text_panel)
        self.text_scroll.setMinimumHeight(100)

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.content_splitter = content_splitter
        content_splitter.setChildrenCollapsible(False)
        content_splitter.addWidget(preview_panel)
        content_splitter.addWidget(self.text_scroll)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([240, 180])

        workspace = QHBoxLayout()
        workspace.setSpacing(16)
        workspace.addWidget(documents_panel, stretch=1)
        workspace.addWidget(content_splitter, stretch=3)
        layout.addLayout(workspace, stretch=1)
        self._update_removal_controls()
        self._show_document_text()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        extra_height = max(0, event.size().height() - self.minimumHeight())
        self.preview.setMinimumHeight(min(150, 60 + extra_height // 2))
        if self.fields_panel.toggle.isChecked():
            QTimer.singleShot(0, self._expand_fields_space)

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(
            self.document_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.document_list.count())
        )

    @property
    def document_paths(self) -> tuple[Path, ...]:
        """Caminhos originais disponíveis para o processamento posterior."""
        return tuple(document.path for document in self.documents if document.path is not None)

    @property
    def current_document(self) -> Document | None:
        item = self.document_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    @property
    def current_document_path(self) -> Path | None:
        document = self.current_document
        return document.path if document else None

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
            item.setData(Qt.ItemDataRole.UserRole, Document(path.name, path=path))
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
        self._show_document_text()
        if self._camera is not None and not self._camera_stopping:
            return
        document = self.current_document
        if document is None:
            self.preview.clear_image()
            return
        self.preview.clear_image()
        try:
            image = document.image if document.image is not None else load_preview(document.path)
            self.preview.set_image(image)
        except ValueError as error:
            self.preview.clear_image("Não foi possível carregar esta imagem.")
            self._report_errors([f"{document.name}: {error}"])

    def _show_document_text(self) -> None:
        document = self.current_document
        with QSignalBlocker(self.text_edit):
            self.text_edit.setPlainText(document.text if document else "")
        self.fields_panel.set_document(document)
        self._update_ocr_controls()

    def _expand_fields_space(self) -> None:
        sizes = self.content_splitter.sizes()
        total = sum(sizes)
        preview_minimum = self.content_splitter.widget(0).minimumSizeHint().height()
        text_size = max(0, min(max(sizes[1], 310), total - preview_minimum))
        self.content_splitter.setSizes([total - text_size, text_size])

    def _update_ocr_controls(self) -> None:
        document = self.current_document
        worker = self._ocr_worker
        self.extract_button.setEnabled(
            document is not None
            and worker is None
            and self._camera is None
            and not self._closing
        )
        self.copy_button.setEnabled(
            document is not None and bool(document.text.strip()) and not self._closing
        )
        self.text_edit.setEnabled(document is not None and not self._closing)
        self.text_edit.setReadOnly(
            worker is not None and document is not None and worker.document_id == document.id
        )
        if self._closing:
            message = "Finalizando tarefas..."
        elif worker is not None:
            message = (
                "Extraindo texto..."
                if document is not None and worker.document_id == document.id
                else "Extraindo texto de outro documento..."
            )
        elif document is None:
            message = "Selecione um documento para extrair texto."
        elif document.ocr_status == "error":
            message = document.ocr_error
        elif document.is_edited:
            message = "Texto editado."
        elif document.ocr_status == "completed":
            message = "Texto extraído. Revise o resultado."
        elif document.ocr_status == "empty":
            message = "Nenhum texto detectado."
        else:
            message = "Texto ainda não extraído."
        self.ocr_status.setText(message)
        self.fields_panel.update_state(
            available=not self._closing,
            running=worker is not None and document is not None and worker.document_id == document.id,
        )

    def _text_edited(self) -> None:
        document = self.current_document
        if document is not None:
            document.text = self.text_edit.toPlainText()
        self._update_ocr_controls()

    def _copy_text(self) -> None:
        if self.current_document is not None and self.text_edit.toPlainText().strip():
            QApplication.clipboard().setText(self.text_edit.toPlainText())
            self.ocr_status.setText("Texto copiado.")

    def _confirm_replace_text(self) -> bool:
        message = QMessageBox(self)
        message.setWindowTitle("Substituir texto editado")
        message.setIcon(QMessageBox.Icon.Question)
        message.setText("Substituir o texto editado por uma nova extração?")
        message.setInformativeText(
            "As alterações serão substituídas somente se o OCR for concluído."
        )
        replace = message.addButton("Substituir texto", QMessageBox.ButtonRole.DestructiveRole)
        cancel = message.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(cancel)
        message.setEscapeButton(cancel)
        message.exec()
        return message.clickedButton() is replace

    def _start_ocr(self) -> None:
        document = self.current_document
        if (
            document is None
            or self._ocr_worker is not None
            or self._camera is not None
            or self._closing
        ):
            return
        if document.is_edited and not self._confirm_replace_text():
            return
        if self.current_document is not document:
            return
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        document.ocr_status = "running"
        document.ocr_error = ""
        self._ocr_worker = OCRWorker(
            document.id, document.path, document.image, self._ocr_engine, self
        )
        self._ocr_worker.finished.connect(self._ocr_finished, Qt.ConnectionType.QueuedConnection)
        self._update_ocr_controls()
        self._ocr_worker.start()

    @Slot()
    def _ocr_finished(self) -> None:
        worker = self._ocr_worker
        if worker is None or self.sender() is not worker:
            return
        self._ocr_worker = None
        document = next(
            (document for document in self.documents if document.id == worker.document_id), None
        )
        if document is not None:
            if worker.result is not None:
                document.recognized_text = worker.result.text
                document.text = worker.result.text
                document.ocr_status = worker.result.status
            elif worker.error:
                document.ocr_status = "error"
                document.ocr_error = worker.error
            else:
                document.ocr_status = (
                    "idle" if document.recognized_text is None
                    else "completed" if document.recognized_text else "empty"
                )
        worker.deleteLater()
        if self._closing:
            self.close()
            return
        if document is not None and self.current_document is document:
            self._show_document_text()
        else:
            self._update_ocr_controls()

    def _start_camera(self) -> None:
        if self._camera is not None or self._closing:
            return
        self._camera_error = ""
        self._camera_stopping = False
        self._latest_frame = None
        self._camera = CameraWorker(self)
        self._camera.failed.connect(self._camera_failed, Qt.ConnectionType.QueuedConnection)
        self._camera.finished.connect(self._camera_finished, Qt.ConnectionType.QueuedConnection)
        self.webcam_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.capture_button.show()
        self.close_camera_button.setEnabled(True)
        self.close_camera_button.show()
        self.preview.clear_image("Abrindo webcam...")
        self._camera.start()
        self._frame_timer.start()
        self._update_ocr_controls()

    def _update_camera_preview(self) -> None:
        if self._camera is None or self._camera_stopping:
            return
        image = self._camera.take_frame()
        if image is not None:
            self._latest_frame = image
            self.preview.set_image(image)
            self.capture_button.setEnabled(True)

    def _capture_image(self) -> None:
        if self._camera is None or self._camera_stopping or self._latest_frame is None:
            return
        image = self._camera.take_frame()
        if image is None:
            image = self._latest_frame
        used_bytes = sum(
            document.image.sizeInBytes()
            for document in self.documents
            if document.image is not None
        )
        if used_bytes + image.sizeInBytes() > MAX_CAPTURE_MEMORY_BYTES:
            self._report_camera_error(
                "O limite de memória para capturas foi atingido. "
                "Remova capturas da lista antes de continuar."
            )
            return
        self._capture_count += 1
        name = f"Captura {self._capture_count}"
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, Document(name, image=image))
        item.setToolTip("Captura da webcam mantida apenas nesta sessão.")
        self.document_list.addItem(item)
        self.document_list.setCurrentItem(item)
        self._stop_camera()

    def _stop_camera(self) -> None:
        if self._camera is None or self._camera_stopping:
            return
        self._camera_stopping = True
        self._frame_timer.stop()
        self._camera.stop()
        self._latest_frame = None
        self.capture_button.setEnabled(False)
        self.close_camera_button.setEnabled(False)
        if self._closing:
            self.preview.clear_image("Fechando webcam...")
        else:
            self._show_selected()

    @Slot(str)
    def _camera_failed(self, message: str) -> None:
        if self.sender() is not self._camera or self._camera_stopping:
            return
        self._camera_error = message
        self._stop_camera()

    @Slot()
    def _camera_finished(self) -> None:
        worker = self._camera
        if worker is None or self.sender() is not worker:
            return
        restore_preview = not self._camera_stopping
        self._frame_timer.stop()
        self._camera = None
        self._latest_frame = None
        self._camera_stopping = False
        self.capture_button.setEnabled(False)
        self.capture_button.hide()
        self.close_camera_button.hide()
        self.webcam_button.setEnabled(True)
        self._update_ocr_controls()
        worker.deleteLater()
        if self._closing:
            self.close()
            return
        if restore_preview:
            self._show_selected()
        if self._camera_error:
            message = self._camera_error
            self._camera_error = ""
            self._report_camera_error(message)

    def _report_camera_error(self, text: str) -> None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Webcam indisponível")
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setText(text)
        message.addButton("Fechar", QMessageBox.ButtonRole.AcceptRole)
        message.exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._camera is None and self._ocr_worker is None:
            self._ocr_engine = None
            event.accept()
            return
        # As operações nativas terminam nas respectivas threads. A interface
        # continua processando eventos até ambas liberarem seus recursos.
        event.ignore()
        self._closing = True
        self._stop_camera()
        if self._ocr_worker is not None:
            self._ocr_worker.requestInterruption()
        self._update_ocr_controls()

    def _remove_selected(self) -> None:
        row = self.document_list.currentRow()
        if row < 0:
            return
        if (
            self._ocr_worker is not None
            and self.current_document.id == self._ocr_worker.document_id
        ):
            self._ocr_worker.requestInterruption()
        with QSignalBlocker(self.document_list):
            self.document_list.takeItem(row)
            if self.document_list.count():
                self.document_list.setCurrentRow(min(row, self.document_list.count() - 1))
        self._show_selected()

    def _clear_documents(self) -> None:
        if self._ocr_worker is not None:
            self._ocr_worker.requestInterruption()
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
