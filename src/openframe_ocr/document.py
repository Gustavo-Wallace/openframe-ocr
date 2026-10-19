"""Documentos e texto editável mantidos apenas durante a sessão."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from PySide6.QtGui import QImage


@dataclass
class Document:
    """Imagem original e estado de OCR independentes da posição na lista."""

    name: str
    path: Path | None = None
    image: QImage | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    recognized_text: str | None = None
    ocr_status: str = "idle"
    ocr_error: str = ""

    @property
    def is_edited(self) -> bool:
        return self.text != (self.recognized_text or "")
