"""Etapas de preprocesamiento del pipeline de segmentación.

La segmentación de estas dos imágenes es, en el fondo, un problema de
**separar lo cromático de lo acromático**: cada JPEG de 256×256 muestra
una hoja de planta —verde en PVC1, verde con clorosis amarilla en PVC2—
sobre un fondo de concreto gris y texturizado. El concreto es *gris*: sus
tres canales R, G y B valen casi lo mismo, de modo que su **saturación**
es baja y su **exceso de verde** es cercano a cero. La hoja, en cambio, es
*croma pura*: domina el verde (y el amarillo en las zonas cloróticas), con
saturación alta. Esa es la palanca que explota todo el pipeline.

Este módulo reúne las etapas que *preparan* la imagen antes de umbralizar:

1.  **Cambio de espacio de color** a escala de grises, la primera etapa
    clásica del curso. Se incluye por completitud del pipeline y como
    control negativo: la hoja oscura y el concreto gris tienen luminancias
    que se solapan, así que el gris —por sí solo— **no** separa bien, y eso
    justifica pasar a un canal cromático.
2.  **Suavizado paso bajo** (gaussiano y mediana). Se suaviza *antes* de
    umbralizar porque la textura del concreto y las nervaduras de la hoja
    meten ruido de alta frecuencia que, sin filtrar, fragmentaría la
    binaria en cientos de islas. El gaussiano difumina de forma isótropa;
    la mediana —no lineal— borra el moteado sal y pimienta de la arenilla
    del concreto sin correr los bordes de la hoja.
3.  **Transformadas cromáticas discriminantes**: el canal de saturación de
    HSV, el canal ``a*`` de CIE-Lab y el índice de exceso de verde
    ExG = 2G − R − B (Woebbecke et al., 1995). Cada uno colapsa los tres
    canales de color en un solo mapa escalar donde la hoja y el fondo caen
    en rangos separables por un umbral.

Todo se apoya en OpenCV. La convención de OpenCV es BGR al leer y H∈[0,180]
en HSV de 8 bits; el código lo tiene en cuenta en cada conversión.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# --- Etapa 0: carga ----------------------------------------------------------


def cargar_color(ruta: str | Path) -> np.ndarray:
    """Lee la imagen en color tal como la entrega OpenCV (BGR, uint8)."""
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return img


# --- Etapa 1: escala de grises ----------------------------------------------


def a_grises(img_bgr: np.ndarray) -> np.ndarray:
    """Escala de grises por luminancia Y = 0.299R + 0.587G + 0.114B.

    Es la primera etapa del preprocesamiento clásico. Aquí cumple además un
    papel diagnóstico: muestra que la intensidad por sí sola no basta para
    separar una hoja oscura de un concreto de brillo medio.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# --- Etapa 2: suavizado paso bajo -------------------------------------------


def suavizado_gaussiano(img: np.ndarray, k: int = 5, sigma: float = 0) -> np.ndarray:
    """Filtro paso bajo gaussiano de ventana ``k``×``k`` (σ automática).

    Difumina de manera isótropa: atenúa el ruido de alta frecuencia a costa
    de ablandar ligeramente los bordes. ``sigma=0`` deja que OpenCV derive σ
    del tamaño de ventana, como en el curso.
    """
    return cv2.GaussianBlur(img, (k, k), sigma)


def suavizado_mediana(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Filtro de mediana de ventana ``k``×``k`` (no lineal).

    Sustituye cada píxel por la mediana de su vecindad. Es el remedio clásico
    contra el ruido *sal y pimienta* —aquí, el moteado de la arenilla del
    concreto— porque un valor atípico no arrastra la mediana como sí arrastra
    a la media, y preserva mejor el borde recto de la hoja.
    """
    return cv2.medianBlur(img, k)


# --- Etapa 3: transformadas cromáticas discriminantes -----------------------


def canal_saturacion(img_bgr: np.ndarray) -> np.ndarray:
    """Canal S de HSV, escalado a 8 bits [0, 255].

    La saturación mide cuán "pura" es la coloración de un píxel,
    independientemente de su brillo. El concreto gris es casi acromático
    (S ≈ 0) tanto en sus zonas claras como en sus sombras; la hoja —verde o
    amarilla— es cromática (S alta). Es el separador más general porque no
    distingue *qué* color, sólo *cuánto* color, de modo que sirve igual para
    la hoja verde de PVC1 que para la hoja clorótica de PVC2.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def canal_a_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Canal ``a*`` de CIE-Lab invertido (verde → valores altos), uint8.

    En Lab, ``a*`` es el eje verde–rojo: negativo hacia el verde, positivo
    hacia el rojo, neutro (128 en la codificación de 8 bits de OpenCV) para
    los grises. Para que la hoja verde quede *clara* sobre fondo oscuro se
    devuelve 255 − a*, de modo que el verde intenso se vuelve el valor más
    alto. Discrimina muy bien el verde, pero es ciego a la clorosis amarilla,
    que en ``a*`` cae cerca del neutro.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    return cv2.bitwise_not(a)  # 255 - a*: el verde (a* bajo) queda alto


def exceso_de_verde(img_bgr: np.ndarray) -> np.ndarray:
    """Índice de exceso de verde ExG = 2G − R − B, normalizado a [0, 255].

    Propuesto por Woebbecke et al. (1995) para separar vegetación de suelo en
    fotografía agrícola, es el discriminante vegetal por excelencia: maximiza
    la respuesta del verde y anula la del gris (donde R≈G≈B ⇒ ExG≈0). Sobre la
    hoja clorótica es menos tajante, porque el amarillo tiene R y G altos y
    reduce el margen 2G − R.
    """
    bgr = img_bgr.astype(np.float32)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    exg = 2.0 * g - r - b
    return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canales_discriminantes(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Devuelve los tres mapas escalares candidatos, ya en 8 bits.

    Claves: ``"HSV-S"``, ``"Lab-a*"`` y ``"ExG"``. Se entregan juntos para
    poder compararlos lado a lado y elegir el mejor por imagen.
    """
    return {
        "HSV-S": canal_saturacion(img_bgr),
        "Lab-a*": canal_a_lab(img_bgr),
        "ExG": exceso_de_verde(img_bgr),
    }
