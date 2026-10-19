"""Campos na interface real com OCR simulado e dados sintéticos."""

from threading import Event
from types import SimpleNamespace

import pytest
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QListWidgetItem, QMessageBox, QPushButton

from openframe_ocr.document import Document
from openframe_ocr import camera
from openframe_ocr.ocr import OCRResult


SYNTHETIC_TEXT = "Nome: Ana Exemplo\nCPF: 000.000.000-00\nDepartamento: Tecnologia"


@pytest.fixture
def rig(window, monkeypatch, tmp_path, wait_for):
    state = SimpleNamespace(
        window=window, output=SYNTHETIC_TEXT, calls=[], confirmations=[],
        accept=False, gate=Event(), started=Event(), error=None,
    )
    state.gate.set()

    class FakeEngine:
        def recognize(self, path, image):
            state.started.set()
            state.calls.append((path, image))
            assert state.gate.wait(3)
            if state.error:
                raise state.error
            return OCRResult(state.output, "completed" if state.output else "empty")

    window._ocr_engine = FakeEngine()

    def add_document(source="capture", name="Documento sintético"):
        image = QImage(240, 160, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        if source == "import":
            path = tmp_path / f"{name}.png"
            assert image.save(str(path), "PNG")
            window.add_images([path])
            window.document_list.setCurrentRow(window.document_list.count() - 1)
        else:
            document = Document(name, image=image)
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, document)
            window.document_list.addItem(item)
            window.document_list.setCurrentItem(item)
        return window.current_document

    def confirm(message):
        state.confirmations.append((message.windowTitle(), message.defaultButton().text()))
        caption = (
            "Substituir campos" if message.windowTitle() == "Substituir campos editados"
            else "Substituir texto"
        ) if state.accept else "Cancelar"
        next(button for button in message.buttons() if button.text() == caption).click()
        return 0

    state.add = add_document
    monkeypatch.setattr(QMessageBox, "exec", confirm)
    window.show()
    yield state
    state.gate.set()
    window.close()
    wait_for(lambda: window._ocr_worker is None)


def recognize(rig, wait_for):
    rig.window.extract_button.click()
    wait_for(lambda: rig.window._ocr_worker is None)


def row(panel, index=0):
    widget = panel.rows.itemAt(index).widget()
    label, value = widget.findChildren(QLineEdit)
    buttons = {button.text(): button for button in widget.findChildren(QPushButton)}
    return label, value, buttons


@pytest.mark.parametrize("source", ["import", "capture"])
def test_fields_use_current_corrected_transcription_for_both_sources(rig, wait_for, source, caplog):
    document = rig.add(source)
    image = document.image
    panel = rig.window.fields_panel
    assert not panel.identify_button.isEnabled()
    recognize(rig, wait_for)
    assert panel.identify_button.isEnabled()
    assert document.fields == []
    QApplication.clipboard().setText("marcador de teste")
    rig.window.text_edit.setPlainText("Nome: Bia Exemplo\nProtocolo: TESTE-123")
    panel.identify_button.click()

    assert [(field.label, field.value) for field in document.fields] == [
        ("Nome", "Bia Exemplo"), ("Protocolo", "TESTE-123")
    ]
    assert document.fields_source_text == document.text
    assert not document.fields_stale
    assert not document.fields_modified
    assert document.recognized_text == SYNTHETIC_TEXT
    assert document.image is image
    assert len(rig.calls) == 1
    assert "não foram verificados" in panel.status.text()
    assert QApplication.clipboard().text() == "marcador de teste"
    assert caplog.text == ""


@pytest.mark.parametrize("source", ["import", "capture"])
def test_next_line_protocol_is_identified_without_manual_addition(rig, wait_for, source):
    document = rig.add(source)
    rig.output = "Protocolo\nMANUAL-001\nDepartamento: Tecnologia"
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    assert [(field.label, field.value) for field in document.fields] == [
        ("Protocolo", "MANUAL-001"), ("Departamento", "Tecnologia")
    ]
    assert panel.rows.count() == 2
    assert not document.fields_modified
    assert document.fields_source_text == document.text


def test_actual_capture_flow_and_fields_share_original_in_memory_image(rig, wait_for, monkeypatch):
    frame = np.full((80, 120, 3), 255, dtype=np.uint8)
    released = []
    capture = SimpleNamespace(
        isOpened=lambda: True, read=lambda: (True, frame),
        release=lambda: released.append(True),
    )
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: capture)
    window = rig.window
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    wait_for(lambda: window._camera is None)
    document = window.current_document
    original_image = document.image
    assert document.path is None
    assert (original_image.width(), original_image.height()) == (120, 80)
    assert document.fields == []
    recognize(rig, wait_for)
    window.fields_panel.identify_button.click()
    assert document.fields[0].value == "Ana Exemplo"
    assert document.image is original_image
    assert rig.calls == [(None, original_image)]
    assert released == [True]


def test_duplicate_labels_edit_and_copy_independently_without_rewriting_text(rig, wait_for):
    document = rig.add()
    rig.output = "Responsável: Ana Exemplo\nResponsável: Bia Exemplo"
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    original_ids = [field.id for field in document.fields]
    first_label, first_value, first_buttons = row(panel)
    second_label, second_value, second_buttons = row(panel, 1)
    QApplication.clipboard().setText("marcador de teste")
    first_label.setText("Responsável corrigido")
    first_value.setText("João Exemplo — ação")

    assert QApplication.clipboard().text() == "marcador de teste"
    assert document.fields[0].label == "Responsável corrigido"
    assert document.fields[0].value == "João Exemplo — ação"
    assert document.fields[1].value == "Bia Exemplo"
    assert document.fields_modified
    assert document.text == rig.output == rig.window.text_edit.toPlainText()
    assert [field.id for field in document.fields] == original_ids
    first_buttons["Copiar"].click()
    assert QApplication.clipboard().text() == "João Exemplo — ação"
    second_buttons["Copiar"].click()
    assert QApplication.clipboard().text() == "Bia Exemplo"
    first_value.clear()
    assert document.fields[0].value == ""
    assert not first_buttons["Copiar"].isEnabled()
    assert second_label.text() == "Responsável"
    assert second_value.text() == "Bia Exemplo"


def test_manual_blank_fields_are_editable_keyboard_accessible_and_removable(rig):
    document = rig.add()
    panel = rig.window.fields_panel
    panel.toggle.click()
    panel.add_button.click()
    assert [(field.label, field.value) for field in document.fields] == [("", "")]
    label, value, buttons = row(panel)
    QTest.keyClicks(label, "Protocolo")
    QTest.keyClick(label, Qt.Key.Key_Tab)
    QTest.keyClicks(value, "MANUAL-001")
    assert [(field.label, field.value) for field in document.fields] == [("Protocolo", "MANUAL-001")]
    buttons["Copiar"].click()
    assert QApplication.clipboard().text() == "MANUAL-001"
    assert document.text == ""
    assert not panel.identify_button.isEnabled()
    buttons["Remover"].click()
    assert document.fields == []
    assert panel.rows.count() == 0
    assert panel.status.text() == "Nenhum campo identificado."


def test_switching_documents_preserves_fields_and_manual_changes(rig, wait_for):
    first = rig.add("import", "Primeiro")
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[1].setText("Valor corrigido")
    first_ids = [field.id for field in first.fields]
    second = rig.add("capture", "Segundo")
    assert panel.rows.count() == 0
    panel.toggle.click()
    panel.add_button.click()
    row(panel)[0].setText("Campo manual")
    row(panel)[1].setText("Somente segundo")
    rig.window.document_list.setCurrentRow(0)
    assert row(panel)[1].text() == "Valor corrigido"
    assert [field.id for field in first.fields] == first_ids
    assert first.fields_modified
    rig.window.document_list.setCurrentRow(1)
    assert row(panel)[1].text() == "Somente segundo"
    assert len(second.fields) == 1
    assert second.fields_modified


@pytest.mark.parametrize("clear_all", [False, True])
def test_removing_documents_drops_fields_and_new_documents_start_empty(rig, wait_for, clear_all):
    rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    assert panel.rows.count() == 3
    (rig.window.clear_button if clear_all else rig.window.remove_button).click()
    assert rig.window.documents == ()
    assert panel.document is None
    assert panel.rows.count() == 0
    assert not panel.identify_button.isEnabled()
    document = rig.add()
    assert document.fields == []
    assert document.fields_source_text is None
    assert not document.fields_modified


def test_text_changes_mark_fields_stale_without_automatic_parsing_or_edit_loss(rig, wait_for):
    document = rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[1].setText("Correção manual")
    ids = [field.id for field in document.fields]
    rig.window.text_edit.setPlainText("Protocolo: NOVO-123")
    assert document.fields_stale
    assert "desatualizados" in panel.status.text()
    assert [field.id for field in document.fields] == ids
    assert row(panel)[1].text() == "Correção manual"
    rig.window.copy_button.click()
    assert QApplication.clipboard().text() == "Protocolo: NOVO-123"
    rig.window.text_edit.setPlainText(SYNTHETIC_TEXT)
    assert not document.fields_stale
    assert document.fields_modified


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
@pytest.mark.parametrize("accept", [False, True])
def test_refresh_requires_confirmation_for_all_manual_changes(rig, wait_for, change, accept):
    document = rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    if change == "edit":
        row(panel)[1].setText("Campo corrigido")
    elif change == "add":
        panel.add_button.click()
    else:
        row(panel)[2]["Remover"].click()
    before = [(field.id, field.label, field.value) for field in document.fields]
    rig.accept = accept
    panel.identify_button.click()
    assert rig.confirmations == [("Substituir campos editados", "Cancelar")]
    if accept:
        assert [(field.label, field.value) for field in document.fields] == [
            ("Nome", "Ana Exemplo"), ("CPF", "000.000.000-00"), ("Departamento", "Tecnologia")
        ]
        assert not document.fields_modified
    else:
        assert [(field.id, field.label, field.value) for field in document.fields] == before
        assert document.fields_modified


def test_refresh_of_untouched_fields_uses_new_text_without_confirmation(rig, wait_for):
    document = rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    rig.window.text_edit.setPlainText("Protocolo: NOVO-123")
    assert document.fields_stale
    panel.identify_button.click()
    assert [(field.label, field.value) for field in document.fields] == [("Protocolo", "NOVO-123")]
    assert not document.fields_stale
    assert rig.confirmations == []


@pytest.mark.parametrize("output", ["Nome: Outra Pessoa Exemplo", ""])
def test_new_ocr_preserves_manual_fields_and_exposes_stale_state(rig, wait_for, output):
    document = rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[1].setText("Correção preservada")
    rig.output = output
    recognize(rig, wait_for)
    assert document.text == output
    assert document.fields[0].value == "Correção preservada"
    assert document.fields_stale
    assert document.fields_modified
    assert "desatualizados" in panel.status.text()
    assert panel.identify_button.isEnabled() == bool(output)


def test_failed_ocr_preserves_fields_and_their_source_text(rig, wait_for):
    document = rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[1].setText("Correção preservada")
    fields_before = [(field.id, field.label, field.value) for field in document.fields]
    rig.error = RuntimeError("Falha sintética")
    recognize(rig, wait_for)
    assert document.ocr_status == "error"
    assert document.text == document.fields_source_text == SYNTHETIC_TEXT
    assert not document.fields_stale
    assert [(field.id, field.label, field.value) for field in document.fields] == fields_before


def test_ocr_finishes_on_inactive_document_without_touching_selected_fields(rig, wait_for):
    first = rig.add("capture", "Primeiro")
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    second = rig.add("capture", "Segundo")
    panel.toggle.click()
    panel.add_button.click()
    row(panel)[1].setText("Somente segundo")
    rig.window.document_list.setCurrentRow(0)
    rig.gate.clear()
    rig.started.clear()
    rig.output = "Protocolo: NOVA-EXTRAÇÃO"
    rig.window.extract_button.click()
    wait_for(rig.started.is_set)
    rig.window.document_list.setCurrentRow(1)
    rig.gate.set()
    wait_for(lambda: rig.window._ocr_worker is None)
    assert first.fields_stale
    assert first.fields[0].value == "Ana Exemplo"
    assert second.fields[0].value == "Somente segundo"
    assert row(panel)[1].text() == "Somente segundo"
    rig.window.document_list.setCurrentRow(0)
    assert "desatualizados" in panel.status.text()


def test_refresh_cannot_silently_restore_all_manually_deleted_fields(rig, wait_for):
    document = rig.add()
    rig.output = "Protocolo: TESTE-123"
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[2]["Remover"].click()
    assert document.fields == []
    panel.identify_button.click()
    assert rig.confirmations == [("Substituir campos editados", "Cancelar")]
    assert document.fields == []


def test_empty_or_unrecognized_transcription_has_concise_empty_state(rig, wait_for):
    document = rig.add()
    panel = rig.window.fields_panel
    assert not panel.identify_button.isEnabled()
    rig.output = "Informações sem campos explícitos.\n000.000.000-00"
    recognize(rig, wait_for)
    panel.identify_button.click()
    assert document.fields == []
    assert panel.status.text() == "Nenhum campo identificado."
    rig.window.text_edit.setPlainText("  \n")
    assert not panel.identify_button.isEnabled()


def test_identification_unavailable_during_same_document_ocr(rig, wait_for):
    rig.add()
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    rig.gate.clear()
    rig.started.clear()
    rig.window.extract_button.click()
    wait_for(rig.started.is_set)
    assert not panel.identify_button.isEnabled()
    rig.gate.set()
    wait_for(lambda: rig.window._ocr_worker is None)
    assert panel.identify_button.isEnabled()


def test_modal_refresh_cannot_replace_a_different_selected_document(rig, wait_for, monkeypatch):
    first = rig.add("capture", "Primeiro")
    recognize(rig, wait_for)
    panel = rig.window.fields_panel
    panel.identify_button.click()
    row(panel)[1].setText("Primeiro corrigido")
    second = rig.add("capture", "Segundo")
    rig.window.document_list.setCurrentRow(0)

    def change_selection(message):
        rig.window.document_list.setCurrentRow(1)
        next(button for button in message.buttons() if button.text() == "Substituir campos").click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", change_selection)
    panel.identify_button.click()
    assert first.fields[0].value == "Primeiro corrigido"
    assert second.fields == []
    assert panel.document is second


def test_many_fields_scroll_at_both_sizes_and_full_text_editor_remains_usable(rig, wait_for, application):
    rig.add()
    rig.output = "\n".join(f"Campo {index}: Valor sintético {index}" for index in range(12))
    recognize(rig, wait_for)
    window = rig.window
    panel = window.fields_panel
    panel.identify_button.click()
    panel.add_button.click()
    for width, height in [(900, 600), (640, 420), (900, 600)]:
        window.resize(width, height)
        application.processEvents()
        wait_for(lambda: window.text_scroll.verticalScrollBar().maximum() > 0)
        minimum_preview = 150 if height == 600 else 60
        wait_for(lambda: window.preview.height() >= minimum_preview)
        assert (window.width(), window.height()) == (width, height)
        assert window.text_edit.viewport().height() > 20
        assert not window.text_edit.isReadOnly()
        assert window.preview.parentWidget().rect().contains(window.preview.geometry())
        pixmap = window.preview.pixmap()
        assert abs(pixmap.width() * 160 - pixmap.height() * 240) <= 240
        assert window.text_scroll.horizontalScrollBar().maximum() == 0
        for index in [0, 12]:
            label, value, buttons = row(panel, index)
            value.setFocus()

            def reveal_row():
                window.text_scroll.ensureWidgetVisible(panel.rows.itemAt(index).widget(), 0, 4)
                viewport = window.text_scroll.viewport()
                return all(
                    viewport.rect().contains(widget.mapTo(viewport, widget.rect().center()))
                    for widget in [label, value, *buttons.values()]
                )

            wait_for(reveal_row)
            viewport_rect = window.text_scroll.viewport().rect()
            for widget in [label, value, *buttons.values()]:
                assert widget.width() >= 30
                assert viewport_rect.contains(widget.mapTo(window.text_scroll.viewport(), widget.rect().center()))
            if width == 640:
                assert label.width() > value.width()
                assert label.y() < value.y()
        assert window.text_scroll.verticalScrollBar().maximum() > 0
        for control in [
            window.extract_button, window.copy_button, window.text_edit,
            panel.identify_button, panel.add_button,
        ]:
            control.setFocus()

            def reveal_control():
                window.text_scroll.ensureWidgetVisible(control, 0, 4)
                viewport = window.text_scroll.viewport()
                return viewport.rect().contains(control.mapTo(viewport, control.rect().center()))

            wait_for(reveal_control)
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))


def test_expanded_fields_keep_practical_preview_height_and_editor_accessible(rig, wait_for, application):
    document = rig.add()
    rig.output = "Protocolo\nMANUAL-001\nNome: Ana Exemplo\nDepartamento: Tecnologia"
    recognize(rig, wait_for)
    window = rig.window
    panel = window.fields_panel
    panel.identify_button.click()
    application.processEvents()
    wait_for(lambda: window.preview.height() >= 150)
    window.content_splitter.setSizes([1, 1000])
    application.processEvents()
    assert window.preview.height() >= 150
    assert window.preview.parentWidget().rect().contains(window.preview.geometry())
    panel.toggle.click()
    panel.toggle.click()
    application.processEvents()
    assert window.preview.height() >= 150

    original_fields = [(field.id, field.label, field.value) for field in document.fields]
    for width, height in [(640, 420), (900, 600)]:
        window.resize(width, height)
        application.processEvents()
        window.text_edit.setFocus()

        def reveal_editor():
            window.text_scroll.ensureWidgetVisible(window.text_edit, 0, 4)
            viewport = window.text_scroll.viewport()
            return viewport.rect().contains(
                window.text_edit.mapTo(viewport, window.text_edit.rect().center())
            )

        wait_for(reveal_editor)
        assert (window.width(), window.height()) == (width, height)
        assert window.text_edit.isEnabled() and not window.text_edit.isReadOnly()
        assert window.text_edit.viewport().height() > 20
        window.text_edit.insertPlainText("Correção manual. ")
        assert document.text == window.text_edit.toPlainText()
        assert [(field.id, field.label, field.value) for field in document.fields] == original_fields
        assert document.fields_stale
