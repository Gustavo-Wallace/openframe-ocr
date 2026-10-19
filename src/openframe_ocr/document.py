"""Documentos e texto editável mantidos apenas durante a sessão."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from PySide6.QtGui import QImage

from openframe_ocr.fields import StructuredField


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
    fields: list[StructuredField] = field(default_factory=list)
    fields_source_text: str | None = None
    fields_modified: bool = False

    @property
    def is_edited(self) -> bool:
        return self.text != (self.recognized_text or "")

    @property
    def fields_stale(self) -> bool:
        return self.fields_source_text is not None and self.fields_source_text != self.text
