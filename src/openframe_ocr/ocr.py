"""OCR local em CPU, usando somente os modelos incluídos no RapidOCR."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage

from openframe_ocr.image_loader import MAX_ALLOCATION_MIB, MAX_IMAGE_PIXELS, load_image


logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Falha com mensagem apropriada para a interface, sem dados do documento."""


@dataclass(frozen=True)
class OCRResult:
    text: str
    status: str


def normalize_result(output) -> OCRResult:
    """Preserva o texto e a ordem das linhas retornadas pela API atual."""
    lines = output.txts
    if lines is None:
        return OCRResult("", "empty")
    if not isinstance(lines, (tuple, list)) or not all(isinstance(line, str) for line in lines):
        raise OCRError("Resultado de OCR incompatível. Verifique a instalação.")
    text = "\n".join(lines)
    return OCRResult(text, "completed") if text.strip() else OCRResult("", "empty")


def _create_engine():
    # Imports e sessões são caros e ficam restritos ao primeiro pedido no worker.
    import onnxruntime as ort
    from omegaconf import flag_override
    import rapidocr
    from rapidocr import RapidOCR

    ort.disable_telemetry_events()
    models_dir = Path(rapidocr.__file__).resolve().parent / "models"
    models = {
        "Det": models_dir / "PP-OCRv6_det_small.onnx",
        "Cls": models_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "Rec": models_dir / "PP-OCRv6_rec_small.onnx",
    }
    if not all(path.is_file() for path in models.values()):
        raise OCRError("Modelos locais de OCR ausentes. Reinstale as dependências.")

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    sessions = {
        section: ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for section, path in models.items()
    }
    characters = sessions["Rec"].get_modelmeta().custom_metadata_map.get("character", "")
    if not characters or not set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ").issubset(characters):
        # A ausência do dicionário não pode acionar o download automático da biblioteca.
        raise OCRError("O modelo local de OCR não oferece os caracteres necessários ao português.")

    engine = RapidOCR(
        params={
            "Global.log_level": "critical",
            "Global.model_root_dir": str(models_dir),
            "Rec.lang_type": "pt",
            **{f"{section}.model_path": str(path) for section, path in models.items()},
        }
    )
    # A API aceita sessões ONNX fornecidas pelo chamador. O DictConfig precisa
    # permitir esses objetos; passá-los diretamente em params é rejeitado na 3.10.
    # Assim não há resolução de URLs, downloads nem sessões de modelo duplicadas.
    with flag_override(engine.cfg, "allow_objects", True):
        for section, session in sessions.items():
            engine.cfg[section].session = session
    return engine


class OCREngine:
    """Reutiliza uma única instância do motor, com pedidos sequenciais."""

    def __init__(self) -> None:
        self._engine = None

    def recognize(self, path: Path | None, image: QImage | None) -> OCRResult:
        try:
            source = image if image is not None else load_image(path)
            if (
                source.isNull()
                or source.width() * source.height() > MAX_IMAGE_PIXELS
                or source.sizeInBytes() > MAX_ALLOCATION_MIB * 1024 * 1024
            ):
                raise ValueError
            bgr = source.convertToFormat(QImage.Format.Format_BGR888)
            pixels = np.frombuffer(bgr.constBits(), dtype=np.uint8).reshape(
                bgr.height(), bgr.bytesPerLine()
            )
            pixels = np.ascontiguousarray(
                pixels[:, : bgr.width() * 3].reshape(bgr.height(), bgr.width(), 3)
            )
        except Exception as error:
            logger.error("Falha ao ler a imagem para OCR (%s).", type(error).__name__)
            raise OCRError("Não foi possível ler a imagem original para extrair texto.") from error

        if self._engine is None:
            try:
                self._engine = _create_engine()
            except OCRError:
                raise
            except (ImportError, OSError) as error:
                logger.error("Dependência de OCR indisponível (%s).", type(error).__name__)
                raise OCRError("Dependências de OCR indisponíveis. Verifique a instalação.") from error
            except Exception as error:
                logger.error("Falha ao iniciar OCR (%s).", type(error).__name__)
                raise OCRError("Não foi possível iniciar o OCR. Verifique a instalação.") from error
        try:
            return normalize_result(self._engine(pixels))
        except OCRError:
            raise
        except Exception as error:
            logger.error("Falha durante o reconhecimento (%s).", type(error).__name__)
            raise OCRError("Não foi possível extrair o texto. Tente novamente.") from error


class OCRWorker(QThread):
    """Executa um único pedido, identificado pelo documento, sem fila de tarefas."""

    def __init__(
        self,
        document_id: str,
        path: Path | None,
        image: QImage | None,
        engine: OCREngine,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.document_id = document_id
        self.path = path
        self.image = image
        self.engine = engine
        self.result: OCRResult | None = None
        self.error = ""

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            result = self.engine.recognize(self.path, self.image)
            if not self.isInterruptionRequested():
                self.result = result
        except Exception as error:
            if not self.isInterruptionRequested():
                self.error = (
                    str(error)
                    if isinstance(error, OCRError)
                    else "Não foi possível extrair o texto. Tente novamente."
                )
                if not isinstance(error, OCRError):
                    logger.error("Falha inesperada no worker de OCR (%s).", type(error).__name__)
        finally:
            self.image = None
