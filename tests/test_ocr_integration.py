"""Inferência real em CPU com os modelos locais e texto português sintético."""

from pathlib import Path
import socket
from time import perf_counter

from openframe_ocr.ocr import OCREngine, OCRWorker


def test_real_ocr_portuguese_offline_reuses_cpu_models(
    portuguese_text_image, tmp_path, monkeypatch, wait_for
):
    import onnxruntime as ort
    import rapidocr
    from rapidocr.utils.download_file import DownloadFile
    import requests

    def deny_network(*args, **kwargs):
        raise AssertionError("O OCR tentou acessar a rede ou baixar um arquivo.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(requests.sessions.Session, "request", deny_network)
    monkeypatch.setattr(DownloadFile, "run", deny_network)
    telemetry_disabled = []
    disable_telemetry = ort.disable_telemetry_events

    def disable():
        telemetry_disabled.append(True)
        disable_telemetry()

    monkeypatch.setattr(ort, "disable_telemetry_events", disable)
    models_dir = Path(rapidocr.__file__).resolve().parent / "models"
    assets_before = {path.name: path.stat().st_mtime_ns for path in models_dir.iterdir()}
    path = tmp_path / "portugues-sintetico.png"
    assert portuguese_text_image.save(str(path), "PNG")
    source_before = path.read_bytes()
    engine = OCREngine()

    started = perf_counter()
    first_worker = OCRWorker("sintetico", path, None, engine)
    first_worker.start()
    wait_for(first_worker.isFinished, timeout=20)
    assert first_worker.wait(1000)
    assert not first_worker.error, first_worker.error
    first = first_worker.result
    first_seconds = perf_counter() - started
    instance = engine._engine
    started = perf_counter()
    second_worker = OCRWorker("sintetico", None, portuguese_text_image, engine)
    second_worker.start()
    wait_for(second_worker.isFinished, timeout=20)
    assert second_worker.wait(1000)
    assert not second_worker.error, second_worker.error
    second = second_worker.result
    second_seconds = perf_counter() - started

    for result in (first, second):
        assert result.status == "completed"
        text = result.text.casefold()
        assert "documento" in text and "teste" in text
        assert any(word in text for word in ("café", "ação", "coração", "açúcar", "atenção"))
        assert len(result.text.splitlines()) >= 3
    assert engine._engine is instance
    assert telemetry_disabled == [True]
    for section in ("Det", "Cls", "Rec"):
        assert instance.cfg[section].session.get_providers() == ["CPUExecutionProvider"]
    characters = instance.cfg.Rec.session.get_modelmeta().custom_metadata_map["character"]
    assert set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ").issubset(characters)
    assert assets_before == {path.name: path.stat().st_mtime_ns for path in models_dir.iterdir()}
    assert path.read_bytes() == source_before
    assert set(tmp_path.iterdir()) == {path}
    print(f"\nOCR REAL: primeira execução={first_seconds:.3f}s; segunda={second_seconds:.3f}s")
    print(f"OCR REAL (conteúdo inteiramente sintético):\n{first.text}")
