"""Aquisição da webcam fora da interface, com apenas um quadro pendente."""

from threading import Event, Lock

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from openframe_ocr.image_loader import MAX_IMAGE_PIXELS


class CameraError(Exception):
    """Falha da câmera com uma mensagem apropriada para a interface."""


def frame_to_image(frame) -> QImage:
    """Obtém uma imagem própria do Qt a partir de um quadro BGR do OpenCV."""
    if (
        frame is None
        or len(frame.shape) != 3
        or frame.shape[2] != 3
        or frame.dtype.name != "uint8"
    ):
        raise CameraError("A webcam enviou uma imagem inválida.")
    height, width, _ = frame.shape
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise CameraError("A imagem da webcam é vazia ou excede o limite de tamanho.")
    if not frame.flags.c_contiguous:
        frame = frame.copy()

    # BGR888 evita uma conversão intermediária para RGB. A cópia desvincula os
    # pixels do buffer reutilizável do OpenCV antes de entregá-los à interface.
    image = QImage(
        frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
    ).copy()
    if image.isNull():
        raise CameraError("Não há memória suficiente para a imagem da webcam.")
    return image


class CameraWorker(QThread):
    """Abre, lê e libera VideoCapture exclusivamente na thread de aquisição."""

    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._stop_requested = Event()
        self._frame_lock = Lock()
        self._latest_frame: QImage | None = None

    def stop(self) -> None:
        """Solicita parada sem esperar uma chamada nativa na thread da interface."""
        self._stop_requested.set()

    def take_frame(self) -> QImage | None:
        """Consome o último quadro; quadros anteriores não formam uma fila."""
        with self._frame_lock:
            image = self._latest_frame
            self._latest_frame = None
        return image

    def run(self) -> None:
        capture = None
        try:
            if self._stop_requested.is_set():
                return
            capture = cv2.VideoCapture(0)
            if self._stop_requested.is_set():
                return
            if not capture.isOpened():
                raise CameraError(
                    "Não foi possível abrir a webcam. Verifique a conexão, "
                    "as permissões e se ela está em uso por outro aplicativo."
                )
            while not self._stop_requested.is_set():
                success, frame = capture.read()
                if self._stop_requested.is_set():
                    break
                if not success:
                    raise CameraError(
                        "A webcam parou de enviar imagens. Verifique a conexão "
                        "e tente abri-la novamente."
                    )
                image = frame_to_image(frame)
                with self._frame_lock:
                    self._latest_frame = image
                if self._stop_requested.wait(1 / 30):
                    break
        except Exception as error:
            if not self._stop_requested.is_set():
                message = (
                    str(error)
                    if isinstance(error, CameraError)
                    else "Não foi possível acessar a webcam. Verifique a conexão e as permissões."
                )
                self.failed.emit(message)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    if not self._stop_requested.is_set():
                        self.failed.emit("Não foi possível encerrar o acesso à webcam.")
            with self._frame_lock:
                self._latest_frame = None
