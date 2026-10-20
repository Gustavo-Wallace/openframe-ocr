"""Comparações de qualidade em imagens sintéticas; sem OCR ou dados pessoais."""

import struct

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from openframe_ocr.image_quality import QualityError, WORKING_EDGE, analyze_pixels
from openframe_ocr.quality_worker import analyze_source


@pytest.fixture
def text_pixels():
    image = np.full((768, 1024), 235, dtype=np.uint8)
    for index in range(10):
        cv2.putText(image, f"DOCUMENTO SINTETICO TESTE {index:03d}", (45, 65 + index * 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, 35, 3, cv2.LINE_AA)
    return image


def metric(result, key):
    return next(item for item in result.metrics if item.key == key)


def test_clear_document_has_consistent_indicators_without_generic_recommendations(text_pixels):
    result = analyze_pixels(text_pixels)
    assert metric(result, "sharpness").score == 100
    assert metric(result, "contrast").score == 100
    assert metric(result, "lighting").score == 100
    assert metric(result, "uniformity").score == 100
    assert metric(result, "readability").score == 100
    assert result.recommendations == ()


def test_gaussian_blur_reduces_sharpness_and_visual_estimate(text_pixels):
    clear = analyze_pixels(text_pixels)
    blurred = analyze_pixels(cv2.GaussianBlur(text_pixels, (0, 0), 4))
    assert blurred.measurements.normalized_laplacian_variance < clear.measurements.normalized_laplacian_variance / 5
    assert metric(blurred, "sharpness").score < metric(clear, "sharpness").score
    assert metric(blurred, "readability").score < metric(clear, "readability").score
    assert any("desfocada" in text for text in blurred.recommendations)


def test_dark_image_reports_insufficient_light(text_pixels):
    clear = analyze_pixels(text_pixels)
    dark = analyze_pixels((text_pixels * 0.2).astype(np.uint8))
    assert dark.measurements.mean_luminance < clear.measurements.mean_luminance
    assert metric(dark, "lighting").score < metric(clear, "lighting").score
    assert any("mais iluminado" in text for text in dark.recommendations)
    assert not any("Reduza a luz" in text for text in dark.recommendations)


def test_washed_out_image_is_not_confused_with_normal_white_paper(text_pixels):
    washed = analyze_pixels(np.clip(text_pixels * 0.15 + 220, 0, 255).astype(np.uint8))
    white_paper = np.where(text_pixels >= 230, 255, text_pixels).astype(np.uint8)
    normal = analyze_pixels(white_paper)
    assert metric(washed, "lighting").score < metric(normal, "lighting").score
    assert metric(normal, "lighting").score == 100
    assert washed.measurements.bright_fraction > 0.6
    assert any("reflexos" in text for text in washed.recommendations)
    assert not any("mais iluminado" in text for text in washed.recommendations)


def test_low_contrast_uses_percentiles_not_isolated_extremes(text_pixels):
    low = (120 + text_pixels * 0.12).astype(np.uint8)
    low[0, 0], low[-1, -1] = 0, 255
    result = analyze_pixels(low)
    clear = analyze_pixels(text_pixels)
    assert metric(result, "contrast").score < metric(clear, "contrast").score
    assert result.measurements.percentile_98 - result.measurements.percentile_2 < 40
    assert any("destacar o texto" in text for text in result.recommendations)


def test_regional_background_detects_gradient_without_penalizing_printed_lines(text_pixels):
    gradient = (text_pixels * np.linspace(0.35, 1, text_pixels.shape[1])).astype(np.uint8)
    clear = analyze_pixels(text_pixels)
    uneven = analyze_pixels(gradient)
    assert clear.measurements.background_spread < 5
    assert uneven.measurements.background_spread > clear.measurements.background_spread + 55
    exposure = metric(uneven, "lighting")
    uniformity = metric(uneven, "uniformity")
    assert exposure.label == "Exposição geral" and exposure.status == "Boa"
    assert uniformity.label == "Uniformidade da luz" and uniformity.status == "Ruim"
    assert "global" in exposure.explanation and "sombras locais" in exposure.explanation
    assert "regional" in uniformity.explanation and "exposição geral" in uniformity.explanation
    assert any("sombras" in text for text in uneven.recommendations)


@pytest.mark.parametrize("value", [0, 128, 255])
def test_textureless_images_have_no_sharpness_or_readability_score(value):
    result = analyze_pixels(np.full((640, 640), value, dtype=np.uint8))
    assert metric(result, "sharpness").score is None
    assert metric(result, "contrast").score is None
    assert metric(result, "readability").score is None
    if value == 255:
        assert metric(result, "lighting").score is None
        assert result.recommendations == ()


def test_noise_and_smooth_gradient_do_not_receive_verified_readability():
    noise = np.random.default_rng(45).integers(0, 256, (640, 640), dtype=np.uint8)
    gradient = np.tile(np.linspace(40, 240, 640, dtype=np.uint8), (640, 1))
    for image in (noise, gradient):
        assert metric(analyze_pixels(image), "readability").score is None


@pytest.mark.parametrize("shape", [(1, 1), (3, 9), (40, 40), (240, 320)])
def test_small_images_are_safe_and_inconclusive(shape):
    result = analyze_pixels(np.full(shape, 128, dtype=np.uint8))
    assert result.measurements.working_size == shape[::-1]
    assert metric(result, "sharpness").score is None
    assert metric(result, "readability").score is None


def test_large_images_are_bounded_and_different_resolutions_are_comparable(text_pixels):
    enlarged = cv2.resize(text_pixels, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    baseline = analyze_pixels(text_pixels)
    result = analyze_pixels(enlarged)
    assert max(result.measurements.working_size) <= WORKING_EDGE
    assert result.measurements.source_size == (3072, 2304)
    assert result.metrics == baseline.metrics
    assert result.measurements.normalized_laplacian_variance == pytest.approx(
        baseline.measurements.normalized_laplacian_variance, rel=0.05)


def test_gray_rgb_results_are_deterministic_and_do_not_change_source(text_pixels):
    before = text_pixels.copy()
    first = analyze_pixels(text_pixels)
    assert first == analyze_pixels(text_pixels)
    assert first == analyze_pixels(cv2.cvtColor(text_pixels, cv2.COLOR_GRAY2RGB))
    assert np.array_equal(text_pixels, before)
    for item in first.metrics:
        assert item.score in (None, 0, 50, 100)


@pytest.mark.parametrize("pixels", [None, [], np.zeros((0, 3), np.uint8),
                                  np.zeros((2, 2), np.float32), np.zeros((2, 2, 2), np.uint8)])
def test_invalid_array_data_has_a_safe_error(pixels):
    with pytest.raises(QualityError, match="inválidos"):
        analyze_pixels(pixels)


def test_real_source_uses_original_not_preview_and_leaves_file_unchanged(tmp_path):
    image = QImage(2400, 1800, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    path = tmp_path / "original.png"
    assert image.save(str(path), "PNG")
    original = path.read_bytes()
    result = analyze_source(path, None)
    assert result.measurements.source_size == (2400, 1800)
    assert max(result.measurements.working_size) <= WORKING_EDGE
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("image_format", [QImage.Format.Format_Grayscale8,
                                         QImage.Format.Format_RGB888,
                                         QImage.Format.Format_BGR888])
def test_qimage_color_and_stride_formats_are_converted(image_format):
    image = QImage(641, 480, image_format)
    image.fill(QColor(120, 120, 120))
    before = bytes(image.constBits())
    result = analyze_source(None, image)
    assert result.measurements.mean_luminance == pytest.approx(120)
    assert bytes(image.constBits()) == before


def test_rgb_luminance_uses_correct_channel_order():
    values = []
    for color in ("red", "green", "blue"):
        image = QImage(640, 480, QImage.Format.Format_BGR888)
        image.fill(QColor(color))
        values.append(analyze_source(None, image).measurements.mean_luminance)
    assert values[0] == pytest.approx(76, abs=1)
    assert values[1] == pytest.approx(75, abs=1)  # Qt 'green' = RGB(0,128,0).
    assert values[2] == pytest.approx(29, abs=1)


@pytest.mark.parametrize("premultiplied", [False, True])
def test_transparency_is_composited_on_white_without_changing_image(premultiplied):
    image_format = (QImage.Format.Format_ARGB32_Premultiplied if premultiplied
                    else QImage.Format.Format_RGBA8888)
    image = QImage(640, 480, image_format)
    image.fill(QColor(0, 0, 0, 0))
    before = bytes(image.constBits())
    result = analyze_source(None, image)
    assert result.measurements.mean_luminance == 255
    assert metric(result, "readability").score is None
    image.fill(QColor(0, 0, 0, 128))
    partial = analyze_source(None, image)
    assert partial.measurements.mean_luminance == pytest.approx(127, abs=1)
    image.fill(QColor(0, 0, 0, 0))
    assert bytes(image.constBits()) == before


def test_imported_jpeg_orientation_is_respected(tmp_path):
    path = tmp_path / "orientada.jpg"
    image = QImage(240, 120, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    assert image.save(str(path), "JPEG")
    exif = (b"Exif\x00\x00II" + struct.pack("<HIH", 42, 8, 1)
            + struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<H", 6) + b"\x00\x00"
            + struct.pack("<I", 0))
    data = path.read_bytes()
    path.write_bytes(data[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + data[2:])
    original = path.read_bytes()
    result = analyze_source(path, None)
    assert result.measurements.source_size == (120, 240)
    assert path.read_bytes() == original


def test_extreme_aspect_source_remains_bounded_and_inconclusive():
    image = QImage(6400, 1, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    result = analyze_source(None, image)
    assert result.measurements.source_size == (6400, 1)
    assert result.measurements.working_size == (640, 1)
    assert metric(result, "readability").score is None


@pytest.mark.parametrize("source", [None, "missing", "corrupt"])
def test_invalid_or_unavailable_source_has_no_document_data_in_error(tmp_path, source, caplog):
    path = tmp_path / "NAO_REGISTRAR.png" if source else None
    if source == "corrupt":
        path.write_bytes(b"not an image")
    with pytest.raises(QualityError) as captured:
        analyze_source(path, None)
    assert "NAO_REGISTRAR" not in str(captured.value)
    assert caplog.text == ""
