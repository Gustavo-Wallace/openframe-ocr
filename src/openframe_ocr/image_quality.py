"""Medições locais e interpretações conservadoras; não executa OCR."""

from dataclasses import dataclass

import cv2
import numpy as np


WORKING_EDGE = 640


class QualityError(Exception):
    """Mensagem segura para a interface, sem conteúdo ou caminhos da imagem."""


@dataclass(frozen=True)
class QualityMeasurements:
    source_size: tuple[int, int]
    working_size: tuple[int, int]
    laplacian_variance: float
    normalized_laplacian_variance: float
    mean_luminance: float
    percentile_2: float
    percentile_98: float
    dark_fraction: float
    bright_fraction: float
    edge_fraction: float
    background_spread: float | None


@dataclass(frozen=True)
class QualityMetric:
    key: str
    label: str
    score: int | None
    status: str
    explanation: str
    recommendation: str = ""


@dataclass(frozen=True)
class QualityResult:
    measurements: QualityMeasurements
    metrics: tuple[QualityMetric, ...]

    @property
    def recommendations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            metric.recommendation for metric in self.metrics if metric.recommendation
        ))


def _metric(key, label, score, explanation, recommendation="") -> QualityMetric:
    # Três faixas discretas evitam sugerir uma precisão que a heurística não tem.
    status = {None: "Inconclusiva", 0: "Ruim", 50: "Atenção", 100: "Boa"}[score]
    return QualityMetric(key, label, score, status, explanation, recommendation)


def analyze_pixels(pixels: np.ndarray) -> QualityResult:
    """Aceita uint8 em cinza ou RGB; nunca altera a matriz recebida."""
    if (
        not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
        or pixels.ndim not in (2, 3) or pixels.size == 0
        or (pixels.ndim == 3 and pixels.shape[2] != 3)
    ):
        raise QualityError("Os dados da imagem são inválidos para análise.")
    height, width = pixels.shape[:2]
    scale = min(1.0, WORKING_EDGE / max(width, height))
    if scale < 1:
        pixels = cv2.resize(
            pixels, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
    work_height, work_width = gray.shape
    low, high = (float(value) for value in np.percentile(gray, (2, 98)))
    span = high - low
    mean = float(gray.mean())
    dark = float(np.mean(gray <= 20))
    bright = float(np.mean(gray >= 250))
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Compara bordas numa escala de contraste comum, sem alterar o original.
    # Isso reduz a confusão entre pouca luz/contraste e desfoque. Não elimina
    # a influência de ruído, tamanho dos caracteres ou conteúdo da imagem.
    normalized = np.clip((gray.astype(np.float32) - low) * (255 / max(span, 1)), 0, 255)
    normalized_laplacian = float(cv2.Laplacian(normalized, cv2.CV_32F).var())
    gradients = (
        np.abs(cv2.Sobel(normalized, cv2.CV_32F, 1, 0))
        + np.abs(cv2.Sobel(normalized, cv2.CV_32F, 0, 1))
    )
    edges = float(np.mean(gradients > 8))
    enough_size = min(work_width, work_height) >= 64
    comparable_resolution = max(work_width, work_height) == WORKING_EDGE
    texture = span >= 5 and 0.003 <= edges <= 0.65

    background_spread = None
    if enough_size and dark < 0.98 and bright < 0.98:
        # O percentil 90 de cada região aproxima o fundo claro da página,
        # reduzindo a influência de linhas de texto sobre a iluminação.
        # Gráficos extensos, páginas coloridas e fundos escuros podem interferir.
        backgrounds = [
            float(np.percentile(cell, 90))
            for rows in np.array_split(gray, 4, axis=0)
            for cell in np.array_split(rows, 4, axis=1)
        ]
        background_spread = float(np.percentile(backgrounds, 90) - np.percentile(backgrounds, 10))

    if not enough_size or not comparable_resolution:
        sharpness = _metric("sharpness", "Nitidez", None,
                            "Resolução insuficiente para comparar a nitidez nesta escala.")
    elif not texture:
        sharpness = _metric("sharpness", "Nitidez", None,
                            "Poucos detalhes ou textura densa demais para estimar desfoque.")
    else:
        score = 0 if normalized_laplacian < 30 else 50 if normalized_laplacian < 120 else 100
        sharpness = _metric(
            "sharpness", "Nitidez", score,
            "Intensidade das bordas em uma escala comum de tamanho e contraste.",
            "A imagem parece desfocada. Mantenha a câmera estável e ajuste a distância."
            if score < 100 else "",
        )

    if mean < 60 or (dark > 0.60 and high < 100):
        lighting = _metric("lighting", "Exposição geral", 0,
                           "A imagem está predominantemente escura.",
                           "Procure um ambiente mais iluminado para registrar os detalhes.")
    elif span < 5 and bright > 0.98:
        lighting = _metric("lighting", "Exposição geral", None,
                           "Uma área branca uniforme não permite distinguir página vazia de excesso de luz.")
    elif mean >= 240 and low >= 180 and bright > 0.60:
        lighting = _metric("lighting", "Exposição geral", 0,
                           "Possível excesso de luz: áreas muito claras com detalhes reduzidos.",
                           "Reduza a luz direta ou os reflexos sobre o documento.")
    elif dark > 0.35:
        lighting = _metric("lighting", "Exposição geral", 50,
                           "Há uma proporção elevada de regiões escuras.",
                           "Distribua melhor a luz para reduzir as áreas escuras.")
    else:
        lighting = _metric("lighting", "Exposição geral", 100,
                           "Não foram detectados sinais fortes de exposição global inadequada; "
                           "ainda pode haver sombras locais.")

    if span < 5:
        contrast = _metric("contrast", "Contraste", None,
                           "Imagem quase uniforme; o contraste do conteúdo é inconclusivo.")
    else:
        score = 0 if span < 40 else 50 if span < 90 else 100
        contrast = _metric("contrast", "Contraste", score,
                           "Diferença entre os percentis 2 e 98 da luminância.",
                           "Use iluminação uniforme para destacar o texto do fundo."
                           if score < 100 else "")

    if background_spread is None:
        uniformity = _metric("uniformity", "Uniformidade da luz", None,
                             "Não há regiões suficientes com fundo mensurável.")
    else:
        score = 100 if background_spread <= 25 else 50 if background_spread <= 55 else 0
        uniformity = _metric("uniformity", "Uniformidade da luz", score,
                             "Variação regional do fundo claro; avalia a distribuição da luz, "
                             "separadamente da exposição geral.",
                             "Há diferenças de iluminação. Evite sombras sobre o documento."
                             if score < 100 else "")

    conditions = (sharpness, lighting, contrast, uniformity)
    # Regra de gargalo, sem média ponderada: a pior condição limita a estimativa.
    # Uma condição inconclusiva impede a estimativa. Não identifica caracteres
    # nem mede acurácia de OCR, autenticidade ou aceitação de um documento.
    readability_score = (
        min(metric.score for metric in conditions)
        if all(metric.score is not None for metric in conditions) else None
    )
    readability = _metric(
        "readability", "Leitura visual estimada", readability_score,
        "Estimativa limitada pelo pior indicador; não confirma a leitura do texto."
        if readability_score is not None else
        "Os indicadores disponíveis não sustentam uma estimativa de leitura.",
    )
    return QualityResult(
        QualityMeasurements(
            (width, height), (work_width, work_height), laplacian, normalized_laplacian,
            mean, low, high, dark, bright, edges, background_spread,
        ),
        (*conditions, readability),
    )
