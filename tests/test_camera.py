"""Câmera simulada com o worker Qt real, sem acessar dispositivos físicos."""

import gc
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from openframe_ocr import camera, main_window
from openframe_ocr.camera import CameraError, CameraWorker, frame_to_image


class FakeCapture:
    def __init__(self, *, opened=True, size=(160, 120), color=(0, 0, 255)):
        self.opened = opened
        self.frame = np.empty((size[1], size[0], 3), dtype=np.uint8)
        self.frame[:] = color
        self.read_gate = Event()
        self.read_gate.set()
        self.read_blocked = Event()
        self.disconnected = Event()
        self.released = Event()
        self.release_count = 0
        self.read_count = 0
        self.read_threads = []
        self.release_thread = None
        self.read_error = None

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_threads.append(get_ident())
        if not self.read_gate.is_set():
            self.read_blocked.set()
        if not self.read_gate.wait(3):
            raise RuntimeError("A leitura simulada não foi liberada pelo teste.")
        if self.read_error:
            raise self.read_error
        if self.disconnected.is_set():
            return False, None
        self.read_count += 1
        return True, self.frame

    def release(self):
        self.release_count += 1
        self.release_thread = get_ident()
        self.released.set()


@pytest.fixture
def wait_for(application):
    def wait(condition, timeout=3):
        deadline = monotonic() + timeout
        while not condition() and monotonic() < deadline:
            application.processEvents()
            QTest.qWait(5)
        application.processEvents()
        assert condition(), "A condição esperada não ocorreu dentro do prazo."

    return wait


@pytest.fixture
def rig(window, application, monkeypatch, wait_for):
    state = SimpleNamespace(
        window=window,
        next_capture=FakeCapture(),
        captures=[],
        indices=[],
        init_threads=[],
        init_gate=Event(),
        init_started=Event(),
        init_error=None,
        messages=[],
    )
    state.init_gate.set()

    def open_camera(index):
        state.indices.append(index)
        state.init_threads.append(get_ident())
        state.init_started.set()
        if not state.init_gate.wait(3):
            raise RuntimeError("A inicialização simulada não foi liberada pelo teste.")
        if state.init_error:
            raise state.init_error
        capture = state.next_capture
        assert all(previous.released.is_set() for previous in state.captures)
        state.captures.append(capture)
        return capture

    def record_message(message):
        state.messages.append((message.windowTitle(), message.text()))
        return 0

    monkeypatch.setattr(camera.cv2, "VideoCapture", open_camera)
    monkeypatch.setattr(QMessageBox, "exec", record_message)
    window.show()
    application.processEvents()
    yield state
    state.init_gate.set()
    state.next_capture.read_gate.set()
    for capture in state.captures:
        capture.read_gate.set()
    window.close()
    wait_for(lambda: window._camera is None)
    assert all(capture.release_count == 1 for capture in state.captures)


def make_import(tmp_path, name="importada.png", color="blue"):
    path = tmp_path / name
    image = QImage(120, 180, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path), "PNG")
    return path


def displayed_color(window):
    image = window.preview.pixmap().toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


def test_frame_bgr_conversion_owns_pixels():
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    image = frame_to_image(frame)
    frame[:] = 0

    assert image.pixelColor(0, 0) == QColor("red")
    assert image.pixelColor(1, 0) == QColor("blue")


def test_non_contiguous_frame_conversion():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    image = frame_to_image(frame[:, ::2])

    assert (image.width(), image.height()) == (4, 4)
    assert image.pixelColor(0, 0) == QColor("lime")


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.broadcast_to(np.zeros(3, dtype=np.uint8), (1, 32_000_001, 3)),
    ],
)
def test_invalid_or_excessive_frames_are_rejected(frame):
    with pytest.raises(CameraError):
        frame_to_image(frame)


def test_camera_activates_and_widgets_receive_frames_on_gui_thread(
    rig, wait_for, monkeypatch
):
    window = rig.window
    gui_thread = get_ident()
    display_threads = []
    original_set_image = window.preview.set_image

    def display(image):
        display_threads.append(get_ident())
        original_set_image(image)

    monkeypatch.setattr(window.preview, "set_image", display)
    window.webcam_button.click()
    assert not window.webcam_button.isEnabled()
    assert window.capture_button.isVisible()
    assert not window.capture_button.isEnabled()
    assert window.close_camera_button.isEnabled()
    wait_for(lambda: window.capture_button.isEnabled())

    assert rig.indices == [0]
    assert all(thread != gui_thread for thread in rig.init_threads)
    assert all(thread != gui_thread for thread in rig.next_capture.read_threads)
    assert display_threads and set(display_threads) == {gui_thread}
    assert displayed_color(window) == QColor("red")
    assert rig.messages == []


@pytest.mark.parametrize("failure", ["unavailable", "permission", "initialization"])
def test_initialization_failure_restores_import_and_controls(
    rig, wait_for, tmp_path, failure
):
    if failure == "unavailable":
        rig.next_capture.opened = False
    elif failure == "permission":
        rig.init_error = PermissionError("permission denied")
    else:
        rig.init_error = RuntimeError("initialization failed")
    window = rig.window
    window.webcam_button.click()
    wait_for(lambda: window._camera is None)

    assert len(rig.messages) == 1
    assert rig.messages[0][0] == "Webcam indisponível"
    assert "permissões" in rig.messages[0][1]
    assert window.webcam_button.isEnabled()
    assert window.import_button.isEnabled()
    assert not window.capture_button.isVisible()
    assert not window._frame_timer.isActive()
    path = make_import(tmp_path)
    window.add_images([path])
    assert window.current_document_path == path.resolve()
    assert displayed_color(window) == QColor("blue")


def test_capture_is_in_memory_full_size_and_selected(rig, wait_for, tmp_path, monkeypatch):
    window = rig.window
    monkeypatch.chdir(tmp_path)
    files_before = set(tmp_path.iterdir())
    rig.next_capture = FakeCapture(size=(2000, 1200))
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    wait_for(lambda: window._camera is None)

    document = window.current_document
    assert document.name == "Captura 1"
    assert document.path is None
    assert (document.image.width(), document.image.height()) == (2000, 1200)
    assert document.image.pixelColor(0, 0) == QColor("red")
    assert window.current_document_path is None
    assert window.document_paths == ()
    assert window.document_list.currentItem().isSelected()
    assert displayed_color(window) == QColor("red")
    assert window.remove_button.isEnabled()
    assert window.clear_button.isEnabled()
    assert window.webcam_button.isEnabled()
    assert window._latest_frame is None
    assert set(tmp_path.iterdir()) == files_before


def test_imports_and_captures_share_selection_and_removal(rig, wait_for, tmp_path):
    window = rig.window
    path = make_import(tmp_path)
    original = path.read_bytes()
    window.add_images([path])
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    wait_for(lambda: window._camera is None)
    assert window.document_paths == (path.resolve(),)
    assert len(window.documents) == 2
    captured_reference = weakref.ref(window.current_document)

    window.document_list.setCurrentRow(0)
    assert window.current_document_path == path.resolve()
    assert displayed_color(window) == QColor("blue")
    window.document_list.setCurrentRow(1)
    assert window.current_document.name == "Captura 1"
    assert displayed_color(window) == QColor("red")
    window.remove_button.click()
    gc.collect()

    assert captured_reference() is None
    assert window.current_document_path == path.resolve()
    assert displayed_color(window) == QColor("blue")
    assert path.read_bytes() == original


@pytest.mark.parametrize("with_document", [False, True])
def test_close_without_capture_restores_previous_preview(rig, wait_for, tmp_path, with_document):
    window = rig.window
    path = make_import(tmp_path) if with_document else None
    if path:
        window.add_images([path])
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.close_camera_button.click()
    wait_for(lambda: window._camera is None)

    assert window.current_document_path == (path.resolve() if path else None)
    assert len(window.documents) == int(with_document)
    assert not window._frame_timer.isActive()
    assert not window.capture_button.isVisible()
    if path:
        assert displayed_color(window) == QColor("blue")
    else:
        assert window.preview.text() == "Nenhum documento carregado"
    assert rig.next_capture.release_thread == rig.init_threads[0]


def test_capture_before_first_frame_is_ignored(rig, wait_for):
    rig.next_capture.read_gate.clear()
    window = rig.window
    window.webcam_button.click()
    wait_for(rig.next_capture.read_blocked.is_set)
    assert not window.capture_button.isEnabled()
    window._capture_image()
    assert window.documents == ()
    window.close_camera_button.click()
    rig.next_capture.read_gate.set()
    wait_for(lambda: window._camera is None)
    assert rig.messages == []


@pytest.mark.parametrize("failure", ["disconnect", "invalid", "exception"])
def test_frame_failure_releases_camera_and_restores_document(
    rig, wait_for, tmp_path, failure
):
    window = rig.window
    path = make_import(tmp_path)
    window.add_images([path])
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    if failure == "disconnect":
        rig.next_capture.disconnected.set()
    elif failure == "invalid":
        rig.next_capture.frame = None
    else:
        rig.next_capture.read_error = RuntimeError("device failure")
    wait_for(lambda: window._camera is None)

    assert rig.next_capture.released.is_set()
    assert len(rig.messages) == 1
    assert window.current_document_path == path.resolve()
    assert displayed_color(window) == QColor("blue")
    assert window.webcam_button.isEnabled()
    assert window.import_button.isEnabled()
    assert not window.capture_button.isEnabled()


def test_rapid_stop_during_initialization_is_responsive_and_prevents_overlap(
    rig, wait_for, application
):
    window = rig.window
    rig.init_gate.clear()
    window.webcam_button.click()
    wait_for(rig.init_started.is_set)
    worker = window._camera
    window.close_camera_button.click()
    window._start_camera()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))

    assert window._camera is worker
    assert not window.webcam_button.isEnabled()
    assert rig.indices == [0]
    rig.init_gate.set()
    wait_for(lambda: window._camera is None)
    assert rig.next_capture.read_count == 0
    assert rig.next_capture.release_count == 1

    for _ in range(2):
        rig.next_capture = FakeCapture()
        window.webcam_button.click()
        wait_for(lambda: window.capture_button.isEnabled())
        window.close_camera_button.click()
        window._start_camera()
        wait_for(lambda: window._camera is None)
    assert len(rig.captures) == 3
    assert all(capture.release_count == 1 for capture in rig.captures)


def test_close_window_during_blocked_read_keeps_event_loop_alive(rig, wait_for):
    window = rig.window
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    rig.next_capture.read_gate.clear()
    wait_for(rig.next_capture.read_blocked.is_set)
    worker = window._camera
    assert not window.close()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))

    assert worker.isRunning()
    assert window.isVisible()
    rig.next_capture.read_gate.set()
    wait_for(lambda: window._camera is None and not window.isVisible())
    assert rig.next_capture.release_count == 1


def test_worker_retains_only_latest_frame_without_a_gui_queue(rig, wait_for):
    worker = CameraWorker()
    try:
        worker.start()
        wait_for(lambda: rig.next_capture.read_count >= 3)
        rig.next_capture.read_gate.clear()
        wait_for(rig.next_capture.read_blocked.is_set)
        image = worker.take_frame()
        assert image is not None
        assert image.pixelColor(0, 0) == QColor("red")
        assert worker.take_frame() is None
    finally:
        worker.stop()
        rig.next_capture.read_gate.set()
        assert worker.wait(3000)
    assert worker.take_frame() is None
    assert rig.next_capture.release_count == 1


def test_capture_budget_is_released_when_documents_are_removed(rig, wait_for, monkeypatch):
    window = rig.window
    monkeypatch.setattr(main_window, "MAX_CAPTURE_MEMORY_BYTES", rig.next_capture.frame.nbytes)
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    wait_for(lambda: window._camera is None)

    rig.next_capture = FakeCapture()
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    window.capture_button.click()
    assert len(window.documents) == 1
    assert "limite de memória" in rig.messages[0][1]
    assert window._camera is not None
    window.remove_button.click()
    window.capture_button.click()
    wait_for(lambda: window._camera is None)
    assert len(window.documents) == 1
    assert window.current_document.name == "Captura 2"
    window.clear_button.click()
    assert window.documents == ()
    assert not window.remove_button.isEnabled()
    assert not window.clear_button.isEnabled()


def test_import_and_selection_do_not_replace_active_camera_preview(rig, wait_for, tmp_path):
    window = rig.window
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    path = make_import(tmp_path)
    window.add_images([path])

    assert window.current_document_path == path.resolve()
    assert window.import_button.isEnabled()
    assert displayed_color(window) == QColor("red")
    window.close_camera_button.click()
    wait_for(lambda: window._camera is None)
    assert displayed_color(window) == QColor("blue")


def test_live_preview_preserves_aspect_ratio_at_both_window_sizes(rig, wait_for, application):
    window = rig.window
    window.webcam_button.click()
    wait_for(lambda: window.capture_button.isEnabled())
    for width, height in [(640, 420), (900, 600)]:
        window.resize(width, height)
        application.processEvents()
        assert (window.width(), window.height()) == (width, height)
        pixmap = window.preview.pixmap()
        available = window.preview.contentsRect().size()
        assert 0 < pixmap.width() <= available.width()
        assert 0 < pixmap.height() <= available.height()
        assert abs(pixmap.width() * 120 - pixmap.height() * 160) <= 160
