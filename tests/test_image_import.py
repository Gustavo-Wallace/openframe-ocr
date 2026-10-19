"""Importação e prévias com imagens sintéticas, sem documentos pessoais."""

from pathlib import Path
import struct
import zlib

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QImageReader, QImageWriter
from PySide6.QtWidgets import QFileDialog, QMessageBox

from openframe_ocr.image_loader import load_preview


@pytest.fixture
def make_image(tmp_path, application):
    def create(name="imagem.png", size=(240, 360), color="red", image_format="PNG"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(*size, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        assert image.save(str(path), image_format)
        return path

    return create


@pytest.fixture(autouse=True)
def messages(monkeypatch):
    captured = []

    def record(message):
        captured.append(
            {
                "title": message.windowTitle(),
                "text": message.text(),
                "format": message.textFormat(),
                "buttons": [button.text() for button in message.buttons()],
            }
        )
        return 0

    monkeypatch.setattr(QMessageBox, "exec", record)
    return captured


@pytest.fixture
def jpeg_support():
    readable = {bytes(value) for value in QImageReader.supportedImageFormats()}
    writable = {bytes(value) for value in QImageWriter.supportedImageFormats()}
    if b"jpeg" not in readable or b"jpeg" not in writable:
        pytest.skip("O ambiente Qt não oferece leitura e escrita JPEG.")


def preview_color(window):
    image = window.preview.pixmap().toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


def assert_empty(window):
    assert window.document_paths == ()
    assert window.current_document_path is None
    assert window.document_list.count() == 0
    assert window.preview.text() == "Nenhum documento carregado"
    assert window.preview.pixmap().isNull()
    assert not window.remove_button.isEnabled()
    assert not window.clear_button.isEnabled()
    assert window.import_button.isEnabled()
    assert window.webcam_button.isEnabled()


def test_import_one_png_through_file_dialog(window, make_image, monkeypatch, messages):
    path = make_image()
    dialog_calls = []

    def choose(*args):
        dialog_calls.append(args)
        return [str(path)], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", choose)
    window.import_button.click()

    assert dialog_calls[0][0] is window
    assert dialog_calls[0][1] == "Carregar imagens"
    assert "*.png" in dialog_calls[0][3]
    assert "*.jpg" in dialog_calls[0][3]
    assert "*.jpeg" in dialog_calls[0][3]
    assert window.document_paths == (path.resolve(),)
    assert window.current_document_path == path.resolve()
    assert window.document_list.item(0).text() == path.name
    assert preview_color(window) == QColor("red")
    assert window.remove_button.isEnabled()
    assert window.clear_button.isEnabled()
    assert window.webcam_button.isEnabled()
    assert messages == []


def test_import_multiple_pngs(window, make_image):
    paths = [make_image("primeira.png"), make_image("segunda.PNG", color="blue")]
    window.add_images(paths)

    assert window.document_paths == tuple(path.resolve() for path in paths)
    assert window.current_document_path == paths[0].resolve()
    assert window.document_list.count() == 2


@pytest.mark.parametrize("extension", ["jpg", "jpeg"])
def test_import_jpeg(window, make_image, jpeg_support, extension, messages):
    path = make_image(f"imagem.{extension}", image_format="JPEG")
    window.add_images([path])

    assert window.current_document_path == path.resolve()
    assert not window.preview.pixmap().isNull()
    assert messages == []


def test_switch_documents(window, make_image):
    first = make_image("primeira.png", color="red")
    second = make_image("segunda.png", color="blue")
    window.add_images([first, second])

    window.document_list.setCurrentRow(1)
    assert window.current_document_path == second.resolve()
    assert window.document_list.currentItem().isSelected()
    assert preview_color(window) == QColor("blue")

    window.document_list.setCurrentRow(0)
    assert window.current_document_path == first.resolve()
    assert preview_color(window) == QColor("red")


def test_additional_import_preserves_documents_and_selection(window, make_image):
    first = make_image("primeira.png")
    second = make_image("segunda.png", color="blue")
    third = make_image("terceira.png", color="green")
    window.add_images([first, second])
    window.document_list.setCurrentRow(1)
    window.add_images([third])

    assert window.document_paths == tuple(path.resolve() for path in (first, second, third))
    assert window.current_document_path == second.resolve()
    assert preview_color(window) == QColor("blue")


def test_duplicate_paths_are_ignored(window, make_image, monkeypatch, messages):
    path = make_image()
    monkeypatch.chdir(path.parent)
    window.add_images([path, str(path), Path(path.name)])
    window.add_images([path.resolve()])

    assert window.document_paths == (path.resolve(),)
    assert messages == []


def test_same_filename_in_different_directories_is_not_a_duplicate(window, make_image):
    first = make_image("pasta_a/imagem.png")
    second = make_image("pasta_b/imagem.png", color="blue")
    window.add_images([first, second])

    assert window.document_list.count() == 2
    assert window.document_list.item(0).toolTip() == str(first.resolve())
    assert window.document_list.item(1).toolTip() == str(second.resolve())
    window.document_list.setCurrentRow(1)
    assert preview_color(window) == QColor("blue")


@pytest.mark.parametrize("already_loaded", [False, True])
def test_cancel_dialog_preserves_state(
    window, make_image, monkeypatch, already_loaded, messages
):
    if already_loaded:
        window.add_images([make_image()])
    paths_before = window.document_paths
    selected_before = window.current_document_path
    pixmap_before = window.preview.pixmap().cacheKey()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args: ([], ""))
    window.import_button.click()

    assert window.document_paths == paths_before
    assert window.current_document_path == selected_before
    assert window.preview.pixmap().cacheKey() == pixmap_before
    assert messages == []
    if not already_loaded:
        assert_empty(window)


@pytest.mark.parametrize(
    "kind", ["text", "truncated", "disguised", "unsupported_extension", "missing", "directory"]
)
def test_reject_invalid_images(window, make_image, tmp_path, kind, messages):
    path = tmp_path / "invalida.png"
    if kind == "text":
        path.write_bytes(b"isto nao e uma imagem")
    elif kind == "truncated":
        path = make_image("invalida.png")
        path.write_bytes(path.read_bytes()[:60])
        assert QImageReader(str(path)).canRead()
    elif kind == "disguised":
        path = make_image("invalida.png", image_format="BMP")
    elif kind == "unsupported_extension":
        path = make_image("invalida.gif")
    elif kind == "directory":
        path.mkdir()

    window.add_images([path])

    assert_empty(window)
    assert len(messages) == 1
    assert messages[0]["title"] == "Imagens não carregadas"
    assert path.name in messages[0]["text"]
    assert messages[0]["format"] == Qt.TextFormat.PlainText
    assert messages[0]["buttons"] == ["Fechar"]


def test_mixed_selection_imports_valid_images_and_reports_rejections(
    window, make_image, tmp_path, messages
):
    previous = make_image("anterior.png")
    valid = make_image("valida.png", color="blue")
    invalid = tmp_path / "corrompida.png"
    invalid.write_bytes(b"arquivo corrompido")
    unsupported = make_image("nao_suportada.bmp", image_format="BMP")
    window.add_images([previous])
    window.add_images([invalid, valid, unsupported])

    assert window.document_paths == (previous.resolve(), valid.resolve())
    assert window.current_document_path == previous.resolve()
    assert len(messages) == 1
    assert invalid.name in messages[0]["text"]
    assert unsupported.name in messages[0]["text"]
    assert "Use PNG ou JPEG" in messages[0]["text"]


def test_remove_selected_and_last_document(window, make_image):
    paths = [make_image(f"imagem_{index}.png") for index in range(3)]
    window.add_images(paths)
    window.document_list.setCurrentRow(1)
    window.remove_button.click()

    assert window.document_paths == (paths[0].resolve(), paths[2].resolve())
    assert window.current_document_path == paths[2].resolve()
    window.remove_button.click()
    assert window.current_document_path == paths[0].resolve()
    window.remove_button.click()
    assert_empty(window)


def test_clear_list_and_reimport(window, make_image):
    paths = [make_image("primeira.png"), make_image("segunda.png")]
    window.add_images(paths)
    window.clear_button.click()
    assert_empty(window)

    window.add_images([paths[0]])
    assert window.document_paths == (paths[0].resolve(),)


def test_original_files_are_unchanged_and_no_copies_are_created(window, make_image):
    path = make_image()
    before = path.read_bytes()
    files_before = set(path.parent.iterdir())
    window.add_images([path])
    window.remove_button.click()
    window.add_images([path])
    window.clear_button.click()

    assert path.read_bytes() == before
    assert set(path.parent.iterdir()) == files_before


@pytest.mark.parametrize("size", [(300, 900), (1000, 250)])
def test_preview_fits_and_preserves_aspect_ratio_on_resize(
    window, application, make_image, size
):
    window.show()
    window.add_images([make_image(size=size)])
    displayed_sizes = []
    for width, height in [(640, 420), (900, 600)]:
        window.resize(width, height)
        application.processEvents()
        assert (window.width(), window.height()) == (width, height)
        pixmap = window.preview.pixmap()
        available = window.preview.contentsRect().size()
        assert 0 < pixmap.width() <= available.width()
        assert 0 < pixmap.height() <= available.height()
        assert abs(pixmap.width() * size[1] - pixmap.height() * size[0]) <= max(size)
        displayed_sizes.append(pixmap.size())
    assert displayed_sizes[0] != displayed_sizes[1]


def test_preview_is_bounded_without_modifying_source(make_image):
    path = make_image(size=(2400, 1200))
    source = path.read_bytes()
    image = load_preview(path)

    assert (image.width(), image.height()) == (1600, 800)
    assert path.read_bytes() == source


def test_excessive_dimensions_are_rejected_before_decoding(window, make_image, messages):
    path = make_image()
    data = bytearray(path.read_bytes())
    data[16:24] = struct.pack(">II", 10000, 10000)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    path.write_bytes(data)

    window.add_images([path])

    assert_empty(window)
    assert "32 milhões de pixels" in messages[0]["text"]


def test_jpeg_exif_orientation_is_applied(make_image, jpeg_support):
    path = make_image("orientada.jpg", size=(120, 80), image_format="JPEG")
    # EXIF sintético: orientação 6 (rotação de 90 graus), sem outros metadados.
    exif = (
        b"Exif\x00\x00II"
        + struct.pack("<HIH", 42, 8, 1)
        + struct.pack("<HHI", 0x0112, 3, 1)
        + struct.pack("<H", 6)
        + b"\x00\x00"
        + struct.pack("<I", 0)
    )
    data = path.read_bytes()
    path.write_bytes(data[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + data[2:])

    image = load_preview(path)

    assert (image.width(), image.height()) == (80, 120)


def test_file_deleted_after_import_does_not_leave_stale_preview(
    window, make_image, messages
):
    first = make_image("primeira.png", color="red")
    second = make_image("segunda.png", color="blue")
    window.add_images([first, second])
    second.unlink()
    window.document_list.setCurrentRow(1)

    assert window.current_document_path == second.resolve()
    assert window.preview.pixmap().isNull()
    assert window.preview.text() == "Não foi possível carregar esta imagem."
    assert window.remove_button.isEnabled()
    assert second.name in messages[0]["text"]
    window.document_list.setCurrentRow(0)
    assert preview_color(window) == QColor("red")
