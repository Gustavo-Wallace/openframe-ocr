"""Leitura local de PNG/JPEG com limites para a pré-visualização."""

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageReader


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_ALLOCATION_MIB = 128
MAX_PREVIEW_EDGE = 1600


def load_preview(path: Path) -> QImage:
    """Valida o conteúdo e retorna uma prévia, preservando o arquivo original."""
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Formato não suportado. Use PNG ou JPEG.")

    try:
        if not path.is_file():
            raise OSError
        file_size = path.stat().st_size
    except OSError as error:
        raise ValueError("O arquivo não está disponível para leitura.") from error

    if file_size > MAX_FILE_BYTES:
        raise ValueError("O arquivo excede o limite de 64 MiB.")

    # Alguns formatos são decodificados por inteiro antes da redução da prévia.
    allocation_limit = QImageReader.allocationLimit()
    if allocation_limit <= 0 or allocation_limit > MAX_ALLOCATION_MIB:
        QImageReader.setAllocationLimit(MAX_ALLOCATION_MIB)

    reader = QImageReader(str(path))
    reader.setDecideFormatFromContent(True)
    reader.setAutoTransform(True)
    if not reader.canRead():
        raise ValueError("Imagem inválida, corrompida ou sem permissão de leitura.")
    if bytes(reader.format()).lower() not in {b"png", b"jpg", b"jpeg"}:
        raise ValueError("Formato não suportado. Use PNG ou JPEG.")

    size = reader.size()
    if not size.isValid() or size.isEmpty():
        raise ValueError("Não foi possível identificar as dimensões da imagem.")
    if size.width() * size.height() > MAX_IMAGE_PIXELS:
        raise ValueError("A imagem excede o limite de 32 milhões de pixels.")

    if max(size.width(), size.height()) > MAX_PREVIEW_EDGE:
        preview_size = size.scaled(
            QSize(MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE),
            Qt.AspectRatioMode.KeepAspectRatio,
        ).expandedTo(QSize(1, 1))
        reader.setScaledSize(preview_size)

    try:
        image = reader.read()
    except MemoryError as error:
        raise ValueError("Não há memória suficiente para carregar a imagem.") from error
    if image.isNull():
        raise ValueError(
            "Não foi possível ler a imagem. O arquivo pode estar corrompido "
            "ou exceder o limite de memória."
        )
    return image
