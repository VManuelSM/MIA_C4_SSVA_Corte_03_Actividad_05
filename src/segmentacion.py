"""Etapas de segmentación: umbralización, morfología y selección de objeto.

Tras el preprocesamiento, la imagen llega como un mapa escalar donde la hoja
y el fondo ocupan rangos de valor distintos. Este módulo cierra el pipeline
hasta la **máscara binaria final**:

4.  **Umbralización de Otsu** (Otsu, 1979): elige automáticamente el umbral
    que minimiza la varianza intraclase del histograma —equivalente a
    maximizar la separación entre las dos poblaciones de píxeles—. Es la
    elección natural cuando el histograma del canal discriminante es
    bimodal, como aquí: un lomo de fondo acromático y un lomo de hoja
    cromática.
5.  **Limpieza morfológica**: apertura para borrar islas sueltas del fondo
    texturizado, cierre para soldar las grietas que las nervaduras abren
    dentro de la hoja, y **relleno de huecos** para tapar los agujeros
    interiores que queden. Sin SciPy en el entorno, el relleno se hace con
    inundación (``cv2.floodFill``) desde el fondo, la técnica estándar en
    OpenCV.
6.  **Selección de la componente conexa mayor**: de todos los blobs que
    sobreviven, se conserva únicamente el más grande —la hoja—, lo que
    elimina de un golpe cualquier residuo del concreto.
7.  **Verificación** por superposición del contorno (Suzuki & Abe, 1985)
    sobre la imagen original, para juzgar a ojo la calidad del recorte.

Por convención del pipeline, la máscara final usa **hoja = blanco (255),
fondo = negro (0)**.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- Etapa 4: umbralización de Otsu -----------------------------------------


def umbral_otsu(canal: np.ndarray) -> tuple[np.ndarray, float]:
    """Binariza un canal 8 bits con Otsu. Devuelve (binaria, umbral).

    El objeto (hoja) queda en 255 y el fondo en 0, asumiendo que el canal
    discriminante da valores *altos* a la hoja (así se construyeron los tres
    canales de ``preprocesamiento``).
    """
    umbral, binaria = cv2.threshold(
        canal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binaria, umbral


# --- Etapa 5: limpieza morfológica ------------------------------------------


def _elemento(radio: int) -> np.ndarray:
    """Elemento estructurante elíptico de diámetro ``2·radio + 1``."""
    lado = 2 * radio + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))


def apertura(binaria: np.ndarray, radio: int = 2) -> np.ndarray:
    """Apertura = erosión seguida de dilatación: borra islas < elemento."""
    return cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _elemento(radio))


def cierre(binaria: np.ndarray, radio: int = 3) -> np.ndarray:
    """Cierre = dilatación seguida de erosión: suelda grietas finas."""
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _elemento(radio))


def rellenar_huecos(binaria: np.ndarray) -> np.ndarray:
    """Rellena los huecos interiores de los objetos blancos.

    Técnica de inundación: se inunda el fondo desde la esquina (0,0); todo lo
    que la inundación *no* alcanza son huecos encerrados dentro del objeto. Se
    invierte esa inundación y se une (OR) con la máscara original. Equivale al
    ``binary_fill_holes`` de SciPy, pero sólo con OpenCV.
    """
    h, w = binaria.shape
    inundado = binaria.copy()
    mascara = np.zeros((h + 2, w + 2), np.uint8)  # floodFill requiere +2 de borde
    cv2.floodFill(inundado, mascara, (0, 0), 255)
    huecos = cv2.bitwise_not(inundado)  # lo no alcanzado = huecos + objeto
    return cv2.bitwise_or(binaria, huecos)


def limpieza_morfologica(
    binaria: np.ndarray, r_apertura: int = 2, r_cierre: int = 3
) -> np.ndarray:
    """Apertura → cierre → relleno de huecos, en ese orden."""
    m = apertura(binaria, r_apertura)
    m = cierre(m, r_cierre)
    m = rellenar_huecos(m)
    return m


# --- Etapa 6: componente conexa mayor ---------------------------------------


def componente_mayor(binaria: np.ndarray) -> np.ndarray:
    """Conserva sólo la componente conexa blanca de mayor área.

    Usa ``cv2.connectedComponentsWithStats`` (etiquetado de clase). La
    etiqueta 0 es el fondo; de las demás se elige la de mayor área en píxeles
    y se devuelve como máscara binaria 0/255.
    """
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:  # sólo fondo
        return np.zeros_like(binaria)
    areas = stats[1:, cv2.CC_STAT_AREA]
    mayor = 1 + int(np.argmax(areas))
    return np.where(etiquetas == mayor, 255, 0).astype(np.uint8)


# --- Etapa 7: verificación por contorno -------------------------------------


def contorno_sobre_original(
    img_bgr: np.ndarray, mascara: np.ndarray, grosor: int = 2,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Dibuja el contorno de la máscara sobre la imagen original (BGR).

    El contorno se extrae con ``cv2.findContours``, que implementa el
    seguimiento de bordes de Suzuki y Abe (1985). El color por defecto es rojo
    (BGR) para que resalte sobre el verde de la hoja.
    """
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    lienzo = img_bgr.copy()
    cv2.drawContours(lienzo, contornos, -1, color, grosor)
    return lienzo


def superposicion_translucida(
    img_bgr: np.ndarray, mascara: np.ndarray, alpha: float = 0.45,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Tiñe de rojo translúcido la región segmentada, para ver fugas y huecos."""
    capa = img_bgr.copy()
    capa[mascara > 0] = color
    return cv2.addWeighted(capa, alpha, img_bgr, 1 - alpha, 0)


# --- Métrica auxiliar --------------------------------------------------------


def fraccion_objeto(mascara: np.ndarray) -> float:
    """Fracción del área total ocupada por el objeto (hoja)."""
    return float((mascara > 0).sum()) / mascara.size
