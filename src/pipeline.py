"""Pipeline de segmentación parametrizable, imagen por imagen.

Encadena las etapas de ``preprocesamiento`` y ``segmentacion`` en una única
función reproducible y devuelve **todas las etapas intermedias**, de modo que
el cuaderno y las figuras del reporte puedan mostrar el proceso completo —de
la original a la máscara binaria— sin recomputar nada.

Los parámetros se agrupan en un ``ParametrosPipeline`` para que PVC1 y PVC2
puedan afinarse por separado (distinto canal discriminante, distinta ventana
de suavizado, distintos radios morfológicos) dejando constancia explícita de
cada elección.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import preprocesamiento as pre
from . import segmentacion as seg


@dataclass
class ParametrosPipeline:
    """Hiperparámetros de una corrida del pipeline sobre una imagen."""

    canal: str = "HSV-S"          # discriminante: "HSV-S" | "Lab-a*" | "ExG"
    suavizado: str = "mediana"    # "mediana" | "gaussiano"
    k_suavizado: int = 5          # ventana del suavizado (impar)
    r_apertura: int = 2           # radio del elemento de apertura
    r_cierre: int = 3             # radio del elemento de cierre


@dataclass
class ResultadoPipeline:
    """Todas las etapas de una corrida, para visualización y verificación."""

    nombre: str
    parametros: ParametrosPipeline
    original: np.ndarray
    grises: np.ndarray
    suavizada: np.ndarray            # el canal discriminante tras suavizar
    canales: dict = field(default_factory=dict)  # los tres candidatos crudos
    canal_elegido: np.ndarray = None
    umbral: float = 0.0
    binaria_cruda: np.ndarray = None
    limpia: np.ndarray = None
    final: np.ndarray = None         # máscara binaria final (hoja=255)
    contorno: np.ndarray = None
    overlay: np.ndarray = None


def ejecutar(nombre: str, img_bgr: np.ndarray,
             params: ParametrosPipeline) -> ResultadoPipeline:
    """Corre el pipeline completo y devuelve todas las etapas.

    Secuencia: grises (diagnóstico) → canal discriminante → suavizado paso
    bajo del canal → Otsu → limpieza morfológica → componente mayor →
    verificación (contorno y superposición).
    """
    grises = pre.a_grises(img_bgr)
    canales = pre.canales_discriminantes(img_bgr)
    canal = canales[params.canal]

    # El suavizado se aplica sobre el canal discriminante, justo antes de
    # umbralizar: ahí es donde la textura del concreto haría daño.
    if params.suavizado == "mediana":
        suavizada = pre.suavizado_mediana(canal, params.k_suavizado)
    else:
        suavizada = pre.suavizado_gaussiano(canal, params.k_suavizado)

    binaria_cruda, umbral = seg.umbral_otsu(suavizada)
    limpia = seg.limpieza_morfologica(binaria_cruda, params.r_apertura, params.r_cierre)
    final = seg.componente_mayor(limpia)
    contorno = seg.contorno_sobre_original(img_bgr, final)
    overlay = seg.superposicion_translucida(img_bgr, final)

    return ResultadoPipeline(
        nombre=nombre, parametros=params, original=img_bgr, grises=grises,
        suavizada=suavizada, canales=canales, canal_elegido=canal,
        umbral=umbral, binaria_cruda=binaria_cruda, limpia=limpia,
        final=final, contorno=contorno, overlay=overlay,
    )
