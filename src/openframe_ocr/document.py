"""Documentos e texto editável mantidos apenas durante a sessão."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from PySide6.QtGui import QImage

from openframe_ocr.fields import StructuredField
from openframe_ocr.image_quality import QualityResult


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
    quality_result: QualityResult | None = None
    quality_status: str = "idle"
    quality_error: str = ""
    text_revision: int = 0
    summary: str = ""
    summary_source_revision: int | None = None
    summary_status: str = "idle"
    summary_error: str = ""

    def set_text(self, text: str) -> None:
        if self.text != text:
            self.text = text
            self.text_revision += 1

    @property
    def summary_stale(self) -> bool:
        return bool(self.summary) and self.summary_source_revision != self.text_revision

    @property
    def is_edited(self) -> bool:
        return self.text != (self.recognized_text or "")

    @property
    def fields_stale(self) -> bool:
        return self.fields_source_text is not None and self.fields_source_text != self.text
