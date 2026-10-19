"""Testes determinísticos de OCR: motor simulado e threads Qt reais."""

from threading import Event, get_ident
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

from openframe_ocr import main_window, ocr
from openframe_ocr.document import Document
from openframe_ocr.ocr import OCREngine, OCRError, OCRResult, normalize_result


def make_image(width=120, height=80):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    return image


def add_capture(window, image=None):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem

    document = Document("Captura sintética", image=image if image is not None else make_image())
    item = QListWidgetItem(document.name)
    item.setData(Qt.ItemDataRole.UserRole, document)
    window.document_list.addItem(item)
    window.document_list.setCurrentItem(item)
    return document


def add_import(window, tmp_path, name="sintetica.png", width=120, height=80):
    path = tmp_path / name
    assert make_image(width, height).save(str(path), "PNG")
    window.add_images([path])
    return window.documents[-1]


@pytest.fixture
def rig(window, monkeypatch, wait_for):
    state = SimpleNamespace(
        window=window,
        started=Event(),
        gate=Event(),
        calls=[],
        constructions=[],
        output=OCRResult("Ação e informação\nCafé com açúcar", "completed"),
        error=None,
        confirm=False,
        confirmations=[],
    )
    state.gate.set()

    class FakeEngine:
        def __init__(self):
            state.constructions.append(self)

        def recognize(self, path, image):
            state.calls.append((path, image, get_ident()))
            state.started.set()
            assert state.gate.wait(3), "O OCR simulado não foi liberado pelo teste."
            if state.error:
                raise state.error
            return state.output

    def choose_confirmation(message):
        state.confirmations.append((message.windowTitle(), message.defaultButton().text()))
        caption = "Substituir texto" if state.confirm else "Cancelar"
        next(button for button in message.buttons() if button.text() == caption).click()
        return 0

    monkeypatch.setattr(main_window, "OCREngine", FakeEngine)
    monkeypatch.setattr(QMessageBox, "exec", choose_confirmation)
    window.show()
    yield state
    state.gate.set()
    window.close()
    wait_for(lambda: window._ocr_worker is None)


@pytest.mark.parametrize(
    ("lines", "expected", "status"),
    [
        (("Ação, coração e São Paulo", "Café com açúcar"), "Ação, coração e São Paulo\nCafé com açúcar", "completed"),
        (("  primeira linha  ", "", "segunda\nterceira"), "  primeira linha  \n\nsegunda\nterceira", "completed"),
        (None, "", "empty"),
        ((), "", "empty"),
        ((" ", "\n"), "", "empty"),
    ],
)
def test_result_normalization_preserves_unicode_and_lines(lines, expected, status):
    assert normalize_result(SimpleNamespace(txts=lines)) == OCRResult(expected, status)


@pytest.mark.parametrize("lines", ["texto isolado", (42,), {"text": "incompatível"}])
def test_incompatible_result_is_reported(lines):
    with pytest.raises(OCRError, match="incompatível"):
        normalize_result(SimpleNamespace(txts=lines))


def test_engine_uses_original_import_and_reuses_model(monkeypatch, tmp_path):
    path = tmp_path / "grande.png"
    assert make_image(2400, 1800).save(str(path), "PNG")
    original = path.read_bytes()
    inputs = []
    initialized = []

    def recognize(pixels):
        inputs.append(pixels.shape)
        return SimpleNamespace(txts=("Informação preservada",))

    def create():
        initialized.append(True)
        return recognize

    monkeypatch.setattr(ocr, "_create_engine", create)
    engine = OCREngine()
    assert initialized == []
    assert engine.recognize(path, None).text == "Informação preservada"
    assert engine.recognize(path, None).status == "completed"
    assert initialized == [True]
    assert inputs == [(1800, 2400, 3), (1800, 2400, 3)]
    assert path.read_bytes() == original


def test_engine_reads_in_memory_image_with_correct_colors(monkeypatch):
    image = make_image(121, 80)
    image.fill(QColor("red"))
    inputs = []

    def recognize(pixels):
        inputs.append((pixels.shape, tuple(pixels[0, 0]), pixels.flags.c_contiguous))
        return SimpleNamespace(txts=("Captura de teste",))

    monkeypatch.setattr(ocr, "_create_engine", lambda: recognize)
    result = OCREngine().recognize(None, image)

    assert result.text == "Captura de teste"
    assert inputs == [((80, 121, 3), (0, 0, 255), True)]


@pytest.mark.parametrize("failure", [ImportError, RuntimeError])
def test_initialization_failure_is_localized_without_logging_contents(monkeypatch, caplog, failure):
    def fail():
        raise failure("CONTEUDO_FICTICIO_QUE_NAO_DEVE_SER_REGISTRADO")

    monkeypatch.setattr(ocr, "_create_engine", fail)
    with pytest.raises(OCRError, match="instalação"):
        OCREngine().recognize(None, make_image())
    assert "CONTEUDO_FICTICIO" not in caplog.text
    assert failure.__name__ in caplog.text


def test_inference_failure_is_localized_without_logging_contents(monkeypatch, caplog):
    def fail(pixels):
        raise RuntimeError("CONTEUDO_FICTICIO_QUE_NAO_DEVE_SER_REGISTRADO")

    monkeypatch.setattr(ocr, "_create_engine", lambda: fail)
    with pytest.raises(OCRError, match="Não foi possível extrair"):
        OCREngine().recognize(None, make_image())
    assert "CONTEUDO_FICTICIO" not in caplog.text


def test_unreadable_original_does_not_initialize_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "_create_engine", lambda: pytest.fail("O motor não deve iniciar."))
    with pytest.raises(OCRError, match="imagem original"):
        OCREngine().recognize(tmp_path / "ausente.png", None)


@pytest.mark.parametrize("source", ["import", "capture"])
def test_ocr_success_runs_in_worker_and_updates_correct_document(rig, wait_for, tmp_path, source):
    window = rig.window
    document = add_import(window, tmp_path) if source == "import" else add_capture(window)
    assert document.recognized_text is None
    assert rig.constructions == []
    gui_thread = get_ident()
    rig.gate.clear()
    window.extract_button.click()
    wait_for(rig.started.is_set)

    assert window.ocr_status.text() == "Extraindo texto..."
    assert not window.extract_button.isEnabled()
    assert window.text_edit.isReadOnly()
    rig.gate.set()
    wait_for(lambda: window._ocr_worker is None)

    assert document.text == rig.output.text
    assert document.recognized_text == rig.output.text
    assert document.ocr_status == "completed"
    assert window.text_edit.toPlainText() == rig.output.text
    assert "Revise" in window.ocr_status.text()
    assert not window.text_edit.isReadOnly()
    assert window.extract_button.isEnabled()
    assert window.copy_button.isEnabled()
    assert rig.calls[0][2] != gui_thread
    assert rig.calls[0][0] == document.path
    assert rig.calls[0][1] is document.image


def test_empty_output_has_no_invented_text(rig, wait_for):
    window = rig.window
    document = add_capture(window)
    rig.output = OCRResult("", "empty")
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)

    assert document.recognized_text == ""
    assert window.text_edit.toPlainText() == ""
    assert window.ocr_status.text() == "Nenhum texto detectado."
    assert not window.copy_button.isEnabled()


def test_switching_documents_preserves_results_and_edits(rig, wait_for, tmp_path):
    window = rig.window
    first = add_import(window, tmp_path, "primeira.png")
    second = add_import(window, tmp_path, "segunda.png")
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    window.text_edit.setPlainText("Texto corrigido: ação e café.")
    assert first.is_edited
    window.document_list.setCurrentRow(1)
    assert window.text_edit.toPlainText() == ""
    assert second.recognized_text is None
    rig.output = OCRResult("Segundo documento\nSão Paulo", "completed")
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    window.document_list.setCurrentRow(0)

    assert window.text_edit.toPlainText() == "Texto corrigido: ação e café."
    assert first.recognized_text != first.text
    window.document_list.setCurrentRow(1)
    assert window.text_edit.toPlainText() == "Segundo documento\nSão Paulo"
    assert len(rig.constructions) == 1


def test_copy_uses_current_edited_unicode_text(rig, wait_for):
    window = rig.window
    document = add_capture(window)
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    edited = "Informação corrigida\nAção, coração e atenção."
    window.text_edit.setPlainText(edited)
    window.copy_button.click()

    assert document.text == edited
    assert document.is_edited
    assert QApplication.clipboard().text() == edited
    assert window.ocr_status.text() == "Texto copiado."
    window.text_edit.clear()
    assert not window.copy_button.isEnabled()


def test_editor_accepts_keyboard_corrections_and_has_portuguese_menu(rig):
    window = rig.window
    document = add_capture(window)
    window.text_edit.setFocus()
    QTest.keyClicks(window.text_edit, "Texto corrigido")
    QTest.keyClick(window.text_edit, Qt.Key.Key_Return)
    window.text_edit.insertPlainText("Informação em português.")

    assert document.text == "Texto corrigido\nInformação em português."
    menu = window.text_edit.createStandardContextMenu()
    captions = [action.text().replace("&", "").split("\t")[0] for action in menu.actions()]
    assert "Copiar" in captions
    assert "Colar" in captions
    menu.deleteLater()


@pytest.mark.parametrize("accept", [False, True])
def test_reocr_requires_explicit_confirmation_to_replace_edits(rig, wait_for, accept):
    window = rig.window
    document = add_capture(window)
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    window.text_edit.setPlainText("Correção manual: informação.")
    rig.output = OCRResult("Nova extração", "completed")
    rig.confirm = accept
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)

    assert rig.confirmations == [("Substituir texto editado", "Cancelar")]
    assert document.text == ("Nova extração" if accept else "Correção manual: informação.")
    assert len(rig.calls) == (2 if accept else 1)


def test_failure_preserves_existing_result_and_edited_text(rig, wait_for):
    window = rig.window
    document = add_capture(window)
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    original = document.recognized_text
    window.text_edit.setPlainText("Texto manual preservado")
    rig.confirm = True
    rig.error = OCRError("Não foi possível extrair o texto. Tente novamente.")
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)

    assert document.text == "Texto manual preservado"
    assert document.recognized_text == original
    assert document.ocr_status == "error"
    assert "Não foi possível" in window.ocr_status.text()
    assert window.extract_button.isEnabled()
    assert window.copy_button.isEnabled()


def test_manual_text_before_first_ocr_also_requires_confirmation(rig):
    window = rig.window
    document = add_capture(window)
    window.text_edit.setPlainText("Texto inserido antes do OCR")
    window.extract_button.click()

    assert len(rig.confirmations) == 1
    assert document.recognized_text is None
    assert rig.calls == []
    assert window._ocr_engine is None


def test_switching_while_processing_does_not_apply_result_to_current_row(rig, wait_for, tmp_path):
    window = rig.window
    first = add_import(window, tmp_path, "primeira.png")
    second = add_import(window, tmp_path, "segunda.png")
    rig.gate.clear()
    window.extract_button.click()
    wait_for(rig.started.is_set)
    window.document_list.setCurrentRow(1)
    window.text_edit.setPlainText("Anotação do segundo documento")
    rig.gate.set()
    wait_for(lambda: window._ocr_worker is None)

    assert first.text == rig.output.text
    assert second.text == "Anotação do segundo documento"
    assert second.recognized_text is None
    assert window.text_edit.toPlainText() == second.text
    window.document_list.setCurrentRow(0)
    assert window.text_edit.toPlainText() == first.text


@pytest.mark.parametrize("clear_all", [False, True])
def test_removed_document_cannot_be_resurrected_by_old_job(rig, wait_for, tmp_path, clear_all):
    window = rig.window
    removed = add_import(window, tmp_path)
    rig.gate.clear()
    window.extract_button.click()
    wait_for(rig.started.is_set)
    if clear_all:
        window.clear_button.click()
    else:
        window.remove_button.click()
    replacement = add_import(window, tmp_path)
    assert replacement.id != removed.id
    rig.gate.set()
    wait_for(lambda: window._ocr_worker is None)

    assert window.documents == (replacement,)
    assert replacement.recognized_text is None
    assert replacement.text == ""
    assert window.text_edit.toPlainText() == ""
    assert window.extract_button.isEnabled()


def test_prevent_duplicate_or_queued_ocr_requests_and_keep_event_loop_alive(rig, wait_for):
    window = rig.window
    add_capture(window)
    rig.gate.clear()
    window.extract_button.click()
    wait_for(rig.started.is_set)
    worker = window._ocr_worker
    window._start_ocr()
    window.extract_button.click()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))

    assert window._ocr_worker is worker
    assert len(rig.calls) == 1
    assert len(rig.constructions) == 1
    rig.gate.set()
    wait_for(lambda: window._ocr_worker is None)


def test_remove_and_clear_drop_document_text_state(rig, wait_for):
    window = rig.window
    add_capture(window)
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    window.remove_button.click()

    assert window.documents == ()
    assert window.text_edit.toPlainText() == ""
    assert not window.copy_button.isEnabled()
    assert not window.extract_button.isEnabled()
    fresh = add_capture(window)
    assert fresh.text == "" and fresh.recognized_text is None
    window.text_edit.setPlainText("Texto temporário sintético")
    window.clear_button.click()
    assert window.text_edit.toPlainText() == ""
    assert not window.copy_button.isEnabled()


def test_closing_window_waits_asynchronously_for_ocr(rig, wait_for):
    window = rig.window
    add_capture(window)
    rig.gate.clear()
    window.extract_button.click()
    wait_for(rig.started.is_set)
    worker = window._ocr_worker
    assert not window.close()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))
    assert worker.isRunning()
    assert window.isVisible()
    rig.gate.set()
    wait_for(lambda: window._ocr_worker is None and not window.isVisible())


def test_ocr_editor_and_controls_fit_both_window_sizes(rig, wait_for, application):
    window = rig.window
    add_capture(window)
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    for width, height in [(900, 600), (640, 420)]:
        window.resize(width, height)
        application.processEvents()
        assert (window.width(), window.height()) == (width, height)
        assert window.text_edit.isVisible()
        assert window.text_edit.viewport().height() > 20
        assert window.extract_button.isVisible()
        assert window.copy_button.isVisible()
        assert not window.text_edit.isReadOnly()
