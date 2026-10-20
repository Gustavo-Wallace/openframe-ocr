"""Adapta a imagem original e executa a análise fora da thread da interface."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, QThread, Qt
from PySide6.QtGui import QImage

from openframe_ocr.image_loader import MAX_ALLOCATION_MIB, MAX_IMAGE_PIXELS, load_image
from openframe_ocr.image_quality import QualityError, QualityResult, WORKING_EDGE, analyze_pixels


def analyze_source(path: Path | None, image: QImage | None) -> QualityResult:
    """Lê o original orientado; reduz antes da conversão de cor e das medições."""
    try:
        source = image if image is not None else load_image(path)
        if (
            source.isNull() or source.width() * source.height() > MAX_IMAGE_PIXELS
            or source.sizeInBytes() > MAX_ALLOCATION_MIB * 1024 * 1024
        ):
            raise ValueError
        source_size = (source.width(), source.height())
        if max(source_size) > WORKING_EDGE:
            size = source.size().scaled(
                WORKING_EDGE, WORKING_EDGE, Qt.AspectRatioMode.KeepAspectRatio,
            ).expandedTo(QSize(1, 1))
            source = source.scaled(
                size, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        rgba = source.convertToFormat(QImage.Format.Format_RGBA8888)
        if rgba.isNull():
            raise ValueError
        pixels = np.frombuffer(rgba.constBits(), dtype=np.uint8).reshape(
            rgba.height(), rgba.bytesPerLine()
        )[:, :rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4)
        # RGBA8888 desfaz formatos premultiplicados do Qt. Transparência é
        # composta sobre branco, como uma página, sem tocar no QImage original.
        alpha = pixels[:, :, 3:4].astype(np.float32) / 255
        rgb = np.rint(pixels[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        result = analyze_pixels(rgb)
        return replace(result, measurements=replace(result.measurements, source_size=source_size))
    except QualityError:
        raise
    except Exception as error:
        raise QualityError("Não foi possível analisar a imagem original. Verifique se ela está disponível.") from error


class QualityWorker(QThread):
    """Um pedido por vez, sem fila e com descarte seguro após interrupção."""

    def __init__(self, document_id, path, image, parent=None):
        super().__init__(parent)
        self.document_id = document_id
        self.path = path
        self.image = image
        self.result: QualityResult | None = None
        self.error = ""

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            result = analyze_source(self.path, self.image)
            if not self.isInterruptionRequested():
                self.result = result
        except Exception as error:
            if not self.isInterruptionRequested():
                self.error = (
                    str(error) if isinstance(error, QualityError)
                    else "Não foi possível analisar a qualidade da imagem. Tente novamente."
                )
        finally:
            self.image = None
