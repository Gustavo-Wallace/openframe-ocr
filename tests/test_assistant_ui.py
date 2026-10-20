"""Resumos por documento e interface Qt usando apenas um servidor HTTP simulado."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QListWidgetItem

from openframe_ocr import camera
from openframe_ocr.document import Document
from openframe_ocr.local_llm import MAX_INPUT_CHARACTERS
from openframe_ocr.main_window import MainWindow
from openframe_ocr.ocr import OCRResult


TEXT = "Solicitação de material para o laboratório.\nProtocolo: TESTE-006\nMunicípio: São Paulo"


@pytest.fixture
def rig(window, llm_server, wait_for):
    window.show()

    def add(text=TEXT, name="Documento sintético"):
        image = QImage(640, 480, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        document = Document(name, image=image)
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, document)
        window.document_list.addItem(item)
        window.document_list.setCurrentItem(item)
        window.text_edit.setPlainText(text)
        document.recognized_text = text
        return document

    state = SimpleNamespace(window=window, server=llm_server, panel=window.assistant_panel, add=add)
    yield state
    window.close()
    for gate in llm_server.gates.values():
        gate.set()


def connect(rig, wait_for):
    panel = rig.panel
    panel.toggle.setChecked(True)
    panel.port.setValue(rig.server.port)
    panel.connect_button.click()
    wait_for(lambda: not rig.window._llm.busy)
    assert rig.window._llm.models


def summarize(rig, wait_for):
    rig.panel.summarize_button.click()
    wait_for(lambda: not rig.window._llm.busy)


def set_response(rig, content):
    rig.server.responses["/v1/chat/completions"] = (200, {"choices": [{"message": {
        "content": content}, "finish_reason": "stop"}]}, {})


def test_startup_and_expansion_do_not_contact_any_server(application, monkeypatch):
    def forbidden(*args):
        pytest.fail("A interface não deve fazer requisições sem ação explícita.")

    monkeypatch.setattr(QNetworkAccessManager, "get", forbidden)
    monkeypatch.setattr(QNetworkAccessManager, "post", forbidden)
    window = MainWindow()
    window.show()
    panel = window.assistant_panel
    assert not panel.toggle.isChecked() and not panel.body.isVisible()
    panel.toggle.click()
    panel.port.setValue(8081)
    application.processEvents()
    assert panel.body.isVisible() and not panel.summarize_button.isEnabled()
    assert window.import_button.isEnabled() and window.webcam_button.isEnabled()
    assert window._ocr_engine is None and not window._llm.busy
    window.close()


def test_current_edited_text_is_only_document_payload_and_existing_data_is_preserved(rig, wait_for):
    document = rig.add()
    window = rig.window
    window.fields_panel.identify_button.click()
    editors = window.fields_panel.rows.itemAt(0).widget().findChildren(QLineEdit)
    editors[1].setText("CAMPO-EDITADO-SEPARADO")
    window.text_edit.setPlainText(TEXT + "\nCorreção: entrega em 22 de outubro.")
    window.quality_panel.analyze_button.click()
    wait_for(lambda: window._quality_worker is None)
    fields = [(field.id, field.label, field.value) for field in document.fields]
    quality = document.quality_result
    image_bytes = bytes(document.image.constBits())
    QApplication.clipboard().setText("Marcador sintético")
    assert rig.server.requests == []
    connect(rig, wait_for)
    summarize(rig, wait_for)
    assert document.summary and document.summary_status == "completed"
    assert not document.summary_stale
    assert document.summary_source_revision == document.text_revision
    assert rig.panel.summary.toPlainText() == document.summary
    assert rig.panel.summary.isReadOnly()
    outgoing = rig.server.requests[-1][2]["messages"][1]["content"]
    assert window.text_edit.toPlainText() in outgoing
    assert "CAMPO-EDITADO-SEPARADO" not in outgoing
    assert document.text == window.text_edit.toPlainText()
    assert [(field.id, field.label, field.value) for field in document.fields] == fields
    assert document.fields_modified and document.quality_result is quality
    assert bytes(document.image.constBits()) == image_bytes
    assert QApplication.clipboard().text() == "Marcador sintético"


def test_disabled_without_document_empty_text_or_while_generating(rig, wait_for):
    connect(rig, wait_for)
    assert not rig.panel.summarize_button.isEnabled()
    rig.add(" \n")
    assert not rig.panel.summarize_button.isEnabled()
    rig.window.text_edit.setPlainText(TEXT)
    assert rig.panel.summarize_button.isEnabled()
    rig.server.gates["/v1/chat/completions"] = Event()
    rig.panel.summarize_button.click()
    wait_for(lambda: len(rig.server.requests) == 2)
    assert not rig.panel.summarize_button.isEnabled()
    assert not rig.panel.port.isEnabled() and not rig.panel.connect_button.isEnabled()
    rig.window._start_summary()
    assert len(rig.server.requests) == 2


def test_multiple_models_have_explicit_selection_and_port_change_requires_reconnect(rig, wait_for):
    rig.add()
    rig.server.responses["/v1/models"] = (200, {"data": [{"id": "modelo-a"}, {"id": "modelo-b"}]}, {})
    connect(rig, wait_for)
    assert rig.panel.model_row.isVisible()
    rig.panel.model.setCurrentIndex(1)
    summarize(rig, wait_for)
    assert rig.server.requests[-1][2]["model"] == "modelo-b"
    rig.panel.port.setValue(8081 if rig.server.port != 8081 else 8082)
    assert not rig.panel.summarize_button.isEnabled()
    assert len(rig.server.requests) == 2


def test_switching_restores_independent_summaries_and_collapse_preserves_them(rig, wait_for):
    first = rig.add(name="Primeiro")
    connect(rig, wait_for)
    set_response(rig, "Resumo do primeiro documento.")
    summarize(rig, wait_for)
    rig.panel.toggle.click()
    assert not rig.panel.body.isVisible()
    second = rig.add("Ata sintética de reunião.", "Segundo")
    assert not rig.panel.toggle.isChecked() and rig.panel.summary.toPlainText() == ""
    rig.panel.toggle.click()
    set_response(rig, "Resumo do segundo documento.")
    summarize(rig, wait_for)
    rig.window.document_list.setCurrentRow(0)
    assert rig.panel.summary.toPlainText() == first.summary == "Resumo do primeiro documento."
    rig.window.document_list.setCurrentRow(1)
    assert rig.panel.summary.toPlainText() == second.summary == "Resumo do segundo documento."
    assert len(rig.server.requests) == 3


def test_selected_document_can_change_during_generation_without_mixing_results(rig, wait_for):
    first = rig.add(name="Primeiro")
    connect(rig, wait_for)
    gate = rig.server.gates["/v1/chat/completions"] = Event()
    rig.panel.summarize_button.click()
    wait_for(lambda: len(rig.server.requests) == 2)
    second = rig.add("Outro documento sintético.", "Segundo")
    rig.window._start_summary()
    assert rig.panel.summary.toPlainText() == ""
    gate.set()
    wait_for(lambda: not rig.window._llm.busy)
    assert first.summary and second.summary == ""
    assert rig.panel.summary.toPlainText() == ""
    rig.window.document_list.setCurrentRow(0)
    assert rig.panel.summary.toPlainText() == first.summary
    assert len(rig.server.requests) == 2


@pytest.mark.parametrize("restore_source", [False, True])
def test_editing_during_generation_discards_response_even_if_text_is_restored(rig, wait_for, restore_source):
    document = rig.add()
    connect(rig, wait_for)
    set_response(rig, "Resumo anterior.")
    summarize(rig, wait_for)
    gate = rig.server.gates["/v1/chat/completions"] = Event()
    set_response(rig, "NÃO APLICAR RESULTADO OBSOLETO")
    rig.panel.summarize_button.click()
    wait_for(lambda: len(rig.server.requests) == 3)
    rig.window.text_edit.setPlainText(TEXT + "\nTexto corrigido.")
    if restore_source:
        rig.window.text_edit.setPlainText(TEXT)
    assert document.summary_stale
    assert "desatualizado" in rig.panel.status.text()
    gate.set()
    wait_for(lambda: not rig.window._llm.busy)
    assert document.summary == "Resumo anterior." and document.summary_stale
    assert "Resultado descartado" in rig.panel.status.text()
    assert "NÃO APLICAR" not in rig.panel.summary.toPlainText()
    rig.server.gates.clear()
    set_response(rig, "Resumo atualizado.")
    summarize(rig, wait_for)
    assert not document.summary_stale and document.summary == "Resumo atualizado."


@pytest.mark.parametrize("clear", [False, True])
def test_removal_abandons_generation_and_does_not_reuse_document_names(rig, wait_for, clear):
    removed = rig.add()
    connect(rig, wait_for)
    gate = rig.server.gates["/v1/chat/completions"] = Event()
    rig.panel.summarize_button.click()
    wait_for(lambda: len(rig.server.requests) == 2)
    (rig.window.clear_button if clear else rig.window.remove_button).click()
    replacement = rig.add()
    assert replacement.id != removed.id
    gate.set()
    wait_for(lambda: not rig.window._llm.busy)
    assert rig.window._summary_request is None
    assert removed not in rig.window.documents and replacement.summary == ""
    assert rig.panel.summary.toPlainText() == ""


def test_clearing_completed_summaries_resets_selected_document(rig, wait_for):
    rig.add()
    connect(rig, wait_for)
    summarize(rig, wait_for)
    rig.window.clear_button.click()
    assert rig.window.documents == () and rig.panel.summary.toPlainText() == ""
    assert not rig.panel.summarize_button.isEnabled()


def test_errors_preserve_previous_summary_and_oversized_text_never_leaves_application(rig, wait_for, caplog):
    document = rig.add()
    connect(rig, wait_for)
    summarize(rig, wait_for)
    original = document.summary
    rig.server.responses["/v1/chat/completions"] = (500, {"error": "NÃO EXIBIR"}, {})
    summarize(rig, wait_for)
    assert document.summary == original and "Resumo anterior preservado" in rig.panel.status.text()
    rig.window.text_edit.setPlainText("á" * (MAX_INPUT_CHARACTERS + 1))
    count = len(rig.server.requests)
    summarize(rig, wait_for)
    assert len(rig.server.requests) == count
    assert "4.000 caracteres" in rig.panel.status.text() and "nenhum trecho foi enviado" in rig.panel.status.text()
    assert document.summary == original and document.summary_stale
    assert caplog.text == ""


def test_unavailable_server_keeps_core_controls_usable(rig, wait_for):
    rig.add()
    rig.server.disconnect = True
    rig.panel.toggle.setChecked(True)
    rig.panel.port.setValue(rig.server.port)
    rig.panel.connect_button.click()
    wait_for(lambda: not rig.window._llm.busy)
    assert "IA local indisponível" in rig.panel.connection_status.text()
    assert not rig.panel.summarize_button.isEnabled()
    for control in [rig.window.import_button, rig.window.webcam_button, rig.window.extract_button,
                    rig.window.fields_panel.identify_button, rig.window.quality_panel.analyze_button]:
        assert control.isEnabled()


@pytest.mark.parametrize("operation", ["models", "summary"])
def test_close_abandons_pending_request_and_gui_remains_responsive(rig, wait_for, operation):
    rig.add()
    path = "/v1/models" if operation == "models" else "/v1/chat/completions"
    if operation == "summary":
        connect(rig, wait_for)
    rig.server.gates[path] = Event()
    if operation == "models":
        rig.panel.port.setValue(rig.server.port)
        rig.panel.connect_button.click()
    else:
        rig.panel.summarize_button.click()
    wait_for(lambda: any(request[1] == path for request in rig.server.requests))
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))
    assert rig.window._llm.busy
    assert rig.window.close()
    assert not rig.window._llm.busy and not rig.window._llm._deadline.isActive()
    assert rig.window._summary_request is None and not rig.window.isVisible()


def test_summary_is_selectable_plain_text_and_all_controls_are_scroll_accessible(rig, wait_for, application):
    rig.add()
    connect(rig, wait_for)
    raw = "<b>Resumo sintético</b>\nSolicitação de material para o laboratório.\nRevise os detalhes."
    set_response(rig, raw)
    summarize(rig, wait_for)
    assert rig.panel.summary.toPlainText() == raw
    rig.panel.summary.selectAll()
    assert rig.panel.summary.textCursor().selectedText()
    window = rig.window
    for size in [(900, 600), (640, 420), (900, 600)]:
        window.resize(*size)
        application.processEvents()
        wait_for(lambda: window.preview.height() >= (150 if size[1] == 600 else 60))
        assert (window.width(), window.height()) == size
        assert window.text_scroll.horizontalScrollBar().maximum() == 0
        assert rig.panel.body.rect().contains(rig.panel.summary.geometry())
        for label in rig.panel.body.findChildren(QLabel):
            if label.parentWidget() is rig.panel.body:
                assert not label.geometry().intersects(rig.panel.summary.geometry())
        for label in rig.panel.findChildren(QLabel):
            if label.isVisible():
                wait_for(lambda: label.height() >= label.heightForWidth(label.width()))
        for control in [window.text_edit, window.extract_button, window.fields_panel.identify_button,
                        window.quality_panel.analyze_button, rig.panel.toggle, rig.panel.port,
                        rig.panel.connect_button, rig.panel.summarize_button, rig.panel.summary]:
            control.setFocus()

            def reveal():
                window.text_scroll.ensureWidgetVisible(control, 0, 4)
                center = control.mapTo(window.text_scroll.widget(), control.rect().center())
                window.text_scroll.ensureVisible(center.x(), center.y(), 0, 4)
                viewport = window.text_scroll.viewport()
                return viewport.rect().contains(control.mapTo(viewport, control.rect().center()))

            wait_for(reveal)
        pixmap = window.preview.pixmap()
        assert abs(pixmap.width() * 480 - pixmap.height() * 640) <= 640
        assert rig.panel.summary.viewport().height() > 40
    rig.panel.toggle.click()
    assert not rig.panel.body.isVisible()
    rig.panel.toggle.click()
    assert rig.panel.summary.toPlainText() == raw
    assert len(rig.server.requests) == 2


def test_ocr_webcam_fields_and_quality_do_not_trigger_llm_requests(rig, wait_for, monkeypatch):
    document = rig.add()
    connect(rig, wait_for)
    summarize(rig, wait_for)
    initial_count = len(rig.server.requests)
    original_summary = document.summary
    window = rig.window
    window._ocr_engine = SimpleNamespace(recognize=lambda path, image: OCRResult(TEXT + "\nTotal: 20", "completed"))
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    assert document.summary_stale and document.summary == original_summary
    window.fields_panel.identify_button.click()
    assert any(field.label == "Total" and field.value == "20" for field in document.fields)
    window.quality_panel.analyze_button.click()
    wait_for(lambda: window._quality_worker is None)
    assert document.quality_result is not None
    capture = SimpleNamespace(isOpened=lambda: True,
                              read=lambda: (True, np.full((480, 640, 3), 235, np.uint8)),
                              release=lambda: None)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: capture)
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    wait_for(lambda: window._camera is None)
    assert window.current_document is not document and window.current_document.summary == ""
    assert len(rig.server.requests) == initial_count
