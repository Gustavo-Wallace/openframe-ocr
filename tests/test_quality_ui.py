"""Estado por documento e workers reais do Qt, com imagens e falhas sintéticas."""

from threading import Event, get_ident
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QListWidgetItem, QProgressBar, QMessageBox, QWidget

from openframe_ocr import camera, quality_worker
from openframe_ocr.document import Document
from openframe_ocr.image_quality import QualityError, analyze_pixels
from openframe_ocr.ocr import OCRResult


@pytest.fixture
def pixels():
    image = np.full((480, 640), 235, np.uint8)
    for index in range(7):
        cv2.putText(image, f"DOCUMENTO TESTE {index}", (30, 60 + index * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 35, 2, cv2.LINE_AA)
    return image


@pytest.fixture
def rig(window, monkeypatch, wait_for, tmp_path, pixels):
    state = SimpleNamespace(
        window=window, calls=[], started=Event(), gate=Event(),
        result=analyze_pixels(pixels), error=None,
    )
    state.gate.set()

    def analyze(path, image):
        state.calls.append((path, image, get_ident()))
        state.started.set()
        assert state.gate.wait(3), "Análise sintética não liberada."
        if state.error:
            raise state.error
        return state.result

    def add(source="capture", name="Documento sintético"):
        image = QImage(pixels.data, 640, 480, pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
        if source == "import":
            path = tmp_path / (name + ".png")
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

    def unexpected_dialog(message):
        pytest.fail("A análise de qualidade não deve abrir diálogos.")

    state.add = add
    monkeypatch.setattr(quality_worker, "analyze_source", analyze)
    monkeypatch.setattr(QMessageBox, "exec", unexpected_dialog)
    window.show()
    yield state
    state.gate.set()
    window.close()
    wait_for(lambda: window._quality_worker is None)


def analyze(rig, wait_for):
    rig.window.quality_panel.analyze_button.click()
    wait_for(lambda: rig.window._quality_worker is None)


@pytest.mark.parametrize("source", ["import", "capture"])
def test_explicit_analysis_runs_outside_gui_thread_for_each_source(rig, wait_for, source):
    document = rig.add(source)
    panel = rig.window.quality_panel
    assert document.quality_result is None and document.quality_status == "idle"
    assert panel.status.text() == "Ainda não analisada."
    assert not panel.toggle.isChecked()
    assert rig.calls == []
    rig.gate.clear()
    panel.analyze_button.click()
    wait_for(rig.started.is_set)
    assert document.quality_status == "running"
    assert panel.status.text() == "Analisando qualidade..."
    assert not panel.analyze_button.isEnabled()
    rig.gate.set()
    wait_for(lambda: rig.window._quality_worker is None)
    assert document.quality_status == "completed"
    assert document.quality_result is rig.result
    assert document.recognized_text is None and rig.window._ocr_engine is None
    assert panel.status.text().startswith("Análise concluída.")
    assert panel.toggle.isChecked()
    assert rig.calls[0][:2] == (document.path, document.image)
    assert rig.calls[0][2] != get_ident()
    assert panel.analyze_button.isEnabled()


def test_real_quality_worker_returns_actual_measurements(window, pixels, tmp_path, wait_for, monkeypatch):
    original = quality_worker.analyze_source
    thread_ids = []

    def record(path, image):
        thread_ids.append(get_ident())
        return original(path, image)

    monkeypatch.setattr(quality_worker, "analyze_source", record)
    image = QImage(pixels.data, 640, 480, pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
    path = tmp_path / "documento.png"
    assert image.save(str(path), "PNG")
    window.add_images([path])
    window.quality_panel.analyze_button.click()
    wait_for(lambda: window._quality_worker is None)
    result = window.current_document.quality_result
    assert result.measurements.source_size == (640, 480)
    assert result.measurements.percentile_98 > result.measurements.percentile_2
    assert thread_ids and thread_ids[0] != get_ident()


def test_import_capture_and_ocr_do_not_trigger_quality_automatically(rig, wait_for, monkeypatch):
    window = rig.window
    rig.add("import")
    capture = SimpleNamespace(isOpened=lambda: True,
                              read=lambda: (True, np.full((480, 640, 3), 235, np.uint8)),
                              release=lambda: None)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: capture)
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    assert rig.calls == []
    window.capture_button.click()
    wait_for(lambda: window._camera is None)
    assert window.current_document.path is None
    assert rig.calls == []
    window._ocr_engine = SimpleNamespace(recognize=lambda path, image: OCRResult("Protocolo: TESTE-001", "completed"))
    window.extract_button.click()
    wait_for(lambda: window._ocr_worker is None)
    window.fields_panel.identify_button.click()
    assert rig.calls == []
    assert window.current_document.fields[0].value == "TESTE-001"
    analyze(rig, wait_for)
    assert len(rig.calls) == 1


def test_switching_restores_results_and_collapse_preserves_data(rig, wait_for):
    first = rig.add("import", "Primeiro")
    analyze(rig, wait_for)
    panel = rig.window.quality_panel
    result = first.quality_result
    panel.toggle.click()
    assert not panel.body.isVisible()
    rig.window.text_edit.setPlainText("Anotação sintética")
    assert not panel.toggle.isChecked()
    assert first.quality_result is result
    second = rig.add("capture", "Segundo")
    assert second.quality_result is None
    assert not panel.toggle.isChecked()
    rig.window.document_list.setCurrentRow(0)
    assert panel.document is first and first.quality_result is result
    assert panel.toggle.isChecked()
    assert len(rig.calls) == 1


def test_switching_while_running_and_duplicate_requests_cannot_queue_jobs(rig, wait_for):
    first = rig.add("capture", "Primeiro")
    rig.gate.clear()
    rig.window.quality_panel.analyze_button.click()
    wait_for(rig.started.is_set)
    worker = rig.window._quality_worker
    rig.window._start_quality()
    second = rig.add("capture", "Segundo")
    rig.window._start_quality()
    assert rig.window._quality_worker is worker
    assert not rig.window.quality_panel.analyze_button.isEnabled()
    assert "outro documento" in rig.window.quality_panel.status.text()
    rig.gate.set()
    wait_for(lambda: rig.window._quality_worker is None)
    assert len(rig.calls) == 1
    assert first.quality_result is rig.result
    assert second.quality_result is None and second.quality_status == "idle"
    assert rig.window.quality_panel.document is second
    assert rig.window.quality_panel.analyze_button.isEnabled()


def test_reanalysis_failure_and_retry_preserve_previous_result(rig, wait_for, pixels, caplog):
    document = rig.add()
    analyze(rig, wait_for)
    original = document.quality_result
    rig.error = RuntimeError("CONTEUDO_NAO_REGISTRAR")
    analyze(rig, wait_for)
    assert document.quality_result is original
    assert document.quality_status == "error"
    assert "Resultado anterior preservado" in rig.window.quality_panel.status.text()
    assert "CONTEUDO" not in document.quality_error
    assert caplog.text == ""
    rig.error = None
    rig.result = analyze_pixels((pixels * 0.2).astype(np.uint8))
    analyze(rig, wait_for)
    assert len(rig.calls) == 3
    assert document.quality_result is rig.result and rig.result is not original
    assert document.quality_status == "completed"
    assert document.quality_error == ""


def test_failure_does_not_corrupt_other_documents(rig, wait_for):
    first = rig.add("capture", "Primeiro")
    analyze(rig, wait_for)
    first_result = first.quality_result
    second = rig.add("capture", "Segundo")
    rig.error = QualityError("Não foi possível analisar a imagem original.")
    analyze(rig, wait_for)
    assert second.quality_status == "error" and second.quality_result is None
    assert first.quality_result is first_result and first.quality_status == "completed"
    assert rig.window.quality_panel.status.text().startswith("Falha na análise.")


@pytest.mark.parametrize("clear_all", [False, True])
def test_removal_during_analysis_discards_stale_results_and_releases_image(rig, wait_for, clear_all):
    removed = rig.add("capture", "Mesmo nome")
    rig.gate.clear()
    rig.window.quality_panel.analyze_button.click()
    wait_for(rig.started.is_set)
    worker = rig.window._quality_worker
    (rig.window.clear_button if clear_all else rig.window.remove_button).click()
    replacement = rig.add("capture", "Mesmo nome")
    assert replacement.id != removed.id
    rig.gate.set()
    wait_for(lambda: rig.window._quality_worker is None)
    assert rig.window.documents == (replacement,)
    assert replacement.quality_result is None and replacement.quality_status == "idle"
    assert worker.image is None
    assert rig.window.quality_panel.document is replacement


def test_clear_completed_documents_resets_analysis_state(rig, wait_for):
    rig.add()
    analyze(rig, wait_for)
    rig.window.clear_button.click()
    assert rig.window.documents == ()
    assert rig.window.quality_panel.document is None
    assert not rig.window.quality_panel.analyze_button.isEnabled()
    fresh = rig.add()
    assert fresh.quality_result is None
    assert fresh.quality_status == "idle"


@pytest.mark.parametrize("edited", [False, True])
def test_analysis_preserves_original_image_text_fields_and_clipboard(rig, wait_for, edited):
    document = rig.add()
    window = rig.window
    window.text_edit.setPlainText("Protocolo: TESTE-001\nNome: Ana Exemplo")
    window.fields_panel.identify_button.click()
    if edited:
        from PySide6.QtWidgets import QLineEdit
        window.fields_panel.rows.itemAt(0).widget().findChildren(QLineEdit)[1].setText("TESTE-CORRIGIDO")
    fields = [(field.id, field.label, field.value) for field in document.fields]
    source_text = document.fields_source_text
    before = bytes(document.image.constBits())
    image = document.image
    QApplication.clipboard().setText("marcador sintético")
    analyze(rig, wait_for)
    assert document.text == window.text_edit.toPlainText() == source_text
    assert document.fields_source_text == source_text
    assert [(field.id, field.label, field.value) for field in document.fields] == fields
    assert document.fields_modified == edited and not document.fields_stale
    assert document.image is image and bytes(document.image.constBits()) == before
    assert QApplication.clipboard().text() == "marcador sintético"


@pytest.mark.parametrize("value", [128, 255])
def test_inconclusive_results_have_no_fabricated_score_bars(rig, wait_for, value):
    document = rig.add()
    rig.result = analyze_pixels(np.full((640, 640), value, np.uint8))
    analyze(rig, wait_for)
    panel = rig.window.quality_panel
    assert document.quality_status == "completed"
    assert "Inconclusiva" in panel.status.text()
    for metric in rig.result.metrics:
        row = panel.body.findChild(QWidget, metric.key)
        if metric.score is None:
            assert row.findChildren(QProgressBar) == []
            texts = [label.text() for label in row.findChildren(QLabel)]
            assert metric.label + ": Inconclusiva" in texts
            assert metric.explanation in texts
        else:
            assert row.findChild(QProgressBar).text() == metric.status


@pytest.mark.parametrize("factor, status, level", [(1, "Boa", 3), (0.3, "Atenção", 2), (0.12, "Ruim", 1)])
def test_actual_analysis_displays_three_qualitative_levels(rig, wait_for, pixels, factor, status, level):
    rig.add()
    rig.result = analyze_pixels((pixels * factor).astype(np.uint8))
    analyze(rig, wait_for)
    panel = rig.window.quality_panel
    contrast = panel.body.findChild(QWidget, "contrast")
    bar = contrast.findChild(QProgressBar)
    assert bar.text() == status and bar.value() == level
    assert (bar.minimum(), bar.maximum()) == (0, 3)
    assert bar.accessibleName() == "Contraste: " + status
    assert "três níveis" in bar.accessibleDescription()
    assert "Contraste: " + status in [label.text() for label in contrast.findChildren(QLabel)]
    labels = [label.text() for label in panel.body.findChildren(QLabel)]
    assert any("três níveis: Ruim, Atenção e Boa" in text for text in labels)
    assert not any("0 a 100" in text or "/100" in text for text in labels)
    assert all(item.text() in ("Ruim", "Atenção", "Boa")
               for item in panel.body.findChildren(QProgressBar))


def test_score_bars_and_all_controls_remain_accessible_at_both_sizes(rig, wait_for, application):
    rig.add()
    window = rig.window
    window.text_edit.setPlainText("Protocolo: TESTE-001")
    window.fields_panel.identify_button.click()
    analyze(rig, wait_for)
    panel = window.quality_panel
    bars = panel.body.findChildren(QProgressBar)
    assert len(bars) == 5
    assert [bar.value() for bar in bars] == [{0: 1, 50: 2, 100: 3}[metric.score] for metric in rig.result.metrics]
    assert [bar.text() for bar in bars] == [metric.status for metric in rig.result.metrics]
    for width, height in [(900, 600), (640, 420), (900, 600)]:
        window.resize(width, height)
        application.processEvents()
        wait_for(lambda: window.preview.height() >= (150 if height == 600 else 60))
        assert (window.width(), window.height()) == (width, height)
        assert window.preview.parentWidget().rect().contains(window.preview.geometry())
        assert window.text_scroll.horizontalScrollBar().maximum() == 0
        for control in [window.text_edit, window.extract_button, window.fields_panel.identify_button,
                        panel.analyze_button, panel.toggle, *bars]:
            control.setFocus()

            def reveal():
                window.text_scroll.ensureWidgetVisible(control, 0, 4)
                viewport = window.text_scroll.viewport()
                return viewport.rect().contains(control.mapTo(viewport, control.rect().center()))

            wait_for(reveal)
            assert control.isVisible()
        assert window.text_edit.viewport().height() > 20
        assert not window.text_edit.isReadOnly()
        pixmap = window.preview.pixmap()
        assert abs(pixmap.width() * 480 - pixmap.height() * 640) <= 640
    cached = window.current_document.quality_result
    panel.toggle.click()
    assert not panel.body.isVisible()
    panel.toggle.click()
    assert window.current_document.quality_result is cached
    assert len(rig.calls) == 1


def test_wrapped_recommendations_and_explanations_are_not_clipped(rig, wait_for, pixels, application):
    rig.add()
    blurred = cv2.GaussianBlur(pixels, (0, 0), 4)
    rig.result = analyze_pixels((blurred * np.linspace(0.45, 1, 640)).astype(np.uint8))
    assert rig.result.recommendations
    analyze(rig, wait_for)
    panel = rig.window.quality_panel
    labels = panel.findChildren(QLabel)
    for size in [(900, 600), (640, 420), (900, 600)]:
        rig.window.resize(*size)
        application.processEvents()
        for label in labels:
            wait_for(lambda: label.height() >= label.heightForWidth(label.width()))
        assert rig.window.text_scroll.horizontalScrollBar().maximum() == 0


def test_close_waits_asynchronously_for_quality_worker(rig, wait_for):
    rig.add()
    window = rig.window
    rig.gate.clear()
    window.quality_panel.analyze_button.click()
    wait_for(rig.started.is_set)
    assert not window.close()
    assert window._quality_worker.isInterruptionRequested()
    assert not window.quality_panel.analyze_button.isEnabled()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))
    assert window.isVisible()
    rig.gate.set()
    wait_for(lambda: window._quality_worker is None and not window.isVisible())


def test_close_waits_for_camera_ocr_and_quality_without_blocking_gui(rig, wait_for, monkeypatch):
    window = rig.window
    rig.add()
    ocr_gate, ocr_started = Event(), Event()

    def recognize(path, image):
        ocr_started.set()
        assert ocr_gate.wait(3)
        return OCRResult("Texto sintético", "completed")

    capture = SimpleNamespace(isOpened=lambda: True,
                              read=lambda: (True, np.full((480, 640, 3), 235, np.uint8)),
                              release=lambda: None)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda index: capture)
    window._ocr_engine = SimpleNamespace(recognize=recognize)
    try:
        window.extract_button.click()
        wait_for(ocr_started.is_set)
        rig.gate.clear()
        window.quality_panel.analyze_button.click()
        wait_for(rig.started.is_set)
        window.webcam_button.click()
        wait_for(lambda: window.capture_button.isEnabled())
        assert not window.close()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        wait_for(lambda: bool(heartbeat))
        ocr_gate.set()
        wait_for(lambda: window._ocr_worker is None and window._camera is None)
        assert window.isVisible() and window._quality_worker is not None
        rig.gate.set()
        wait_for(lambda: window._quality_worker is None and not window.isVisible())
    finally:
        ocr_gate.set()
        rig.gate.set()
