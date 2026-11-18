"""Figuras del reporte: comparativas de canales, montajes y binarias.

Criterio editorial único: imágenes sin ejes, cada panel con su título corto,
fondo blanco y una sola tinta de acento. Las figuras se guardan en ``salidas``
con nombre ``figNN_<slug>.png`` y luego se copian al vault con el prefijo de
la materia.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CARPETA_SALIDAS = Path(__file__).resolve().parent.parent / "salidas"

TINTA = "#262626"
TINTA_SUAVE = "#6B6B6B"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlecolor": TINTA,
    "text.color": TINTA,
    "font.size": 10,
})


def _mostrar(ax, img: np.ndarray, titulo: str, cmap: str | None = None) -> None:
    if img.ndim == 3:  # BGR → RGB
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=255)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(TINTA_SUAVE)


def guardar(fig, nombre: str) -> Path:
    """Guarda la figura en ``salidas`` a 150 dpi y devuelve la ruta."""
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    ruta = CARPETA_SALIDAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def fig_entradas(img1: np.ndarray, img2: np.ndarray, nombre: str) -> Path:
    """Las dos imágenes de entrada, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    _mostrar(axes[0], img1, "PVC1.JPG — hoja verde (tipo chile)")
    _mostrar(axes[1], img2, "PVC2.JPG — hoja con clorosis (tipo tomate)")
    fig.tight_layout()
    return guardar(fig, nombre)


def fig_suavizado(canal: np.ndarray, gauss: np.ndarray, mediana: np.ndarray,
                  etiqueta_canal: str, nombre: str) -> Path:
    """Canal discriminante crudo frente a su versión gaussiana y mediana."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    _mostrar(axes[0], canal, f"{etiqueta_canal} (sin suavizar)")
    _mostrar(axes[1], gauss, "Gaussiano 5×5")
    _mostrar(axes[2], mediana, "Mediana 5×5")
    fig.tight_layout()
    return guardar(fig, nombre)


def fig_canales(canales: dict, umbrales: dict, elegido: str, nombre: str) -> Path:
    """Los tres candidatos (fila 1) y su binaria de Otsu (fila 2).

    ``umbrales`` mapea cada clave de canal a su binaria de Otsu ya calculada.
    El candidato elegido se marca en el título.
    """
    claves = ["HSV-S", "Lab-a*", "ExG"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    for j, c in enumerate(claves):
        marca = "  ✓ elegido" if c == elegido else ""
        _mostrar(axes[0, j], canales[c], f"{c}{marca}")
        _mostrar(axes[1, j], umbrales[c], f"Otsu de {c}")
    fig.tight_layout()
    return guardar(fig, nombre)


def fig_montaje(res, nombre: str) -> Path:
    """Montaje de 8 etapas: de la original a la verificación por contorno."""
    p = res.parametros
    paneles = [
        (res.original, "(a) Original"),
        (res.grises, "(b) Escala de grises"),
        (res.canal_elegido, f"(c) Canal {p.canal}"),
        (res.suavizada, f"(d) Suavizado ({p.suavizado} {p.k_suavizado}×{p.k_suavizado})"),
        (res.binaria_cruda, f"(e) Otsu (umbral={res.umbral:.0f})"),
        (res.limpia, "(f) Morfología + relleno"),
        (res.final, "(g) Componente mayor (final)"),
        (res.contorno, "(h) Contorno sobre original"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for ax, (img, tit) in zip(axes.ravel(), paneles):
        _mostrar(ax, img, tit)
    fig.tight_layout()
    return guardar(fig, nombre)


def fig_binaria_final(res, nombre: str) -> Path:
    """Original, máscara binaria final y superposición translúcida."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    _mostrar(axes[0], res.original, "Original")
    _mostrar(axes[1], res.final, "Binaria final (hoja = blanco)")
    _mostrar(axes[2], res.overlay, "Verificación (superposición)")
    fig.tight_layout()
    return guardar(fig, nombre)
