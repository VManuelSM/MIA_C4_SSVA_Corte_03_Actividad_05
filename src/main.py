"""Reproduce la práctica completa: ejecuta el pipeline de las dos hojas,
guarda las binarias finales en ``salidas/`` y genera todas las figuras del
reporte.

Uso:
    python -m src.main          (desde la raíz del repositorio)

Decisión de diseño documentada en el reporte:
    - PVC1 (hoja verde oscura y uniforme) → canal **ExG** (2G−R−B). El índice
      de vegetación cubre toda la hoja; la saturación (HSV-S) deja fuera la
      zona baja en sombra, donde el verde oscuro tiene saturación baja.
    - PVC2 (hoja con clorosis) → canal **HSV-S**. La saturación responde a
      *cualquier* croma —verde y amarillo— y recupera las zonas cloróticas
      que el ExG, ciego al amarillo, dejaría fuera.
El resto del pipeline (suavizado de mediana 5×5, apertura r=2, cierre r=3,
relleno de huecos y componente mayor) es común a ambas imágenes.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from . import preprocesamiento as pre
from . import segmentacion as seg
from . import visualizacion as vis
from .pipeline import ParametrosPipeline, ejecutar

RAIZ = Path(__file__).resolve().parent.parent
IMAGENES = RAIZ / "imagenes"
SALIDAS = RAIZ / "salidas"

CONFIG = {
    "PVC1": ParametrosPipeline(canal="ExG", suavizado="mediana", k_suavizado=5,
                               r_apertura=2, r_cierre=3),
    "PVC2": ParametrosPipeline(canal="HSV-S", suavizado="mediana", k_suavizado=5,
                               r_apertura=2, r_cierre=3),
}


def main() -> None:
    SALIDAS.mkdir(exist_ok=True)
    imgs = {n: pre.cargar_color(IMAGENES / f"{n}.JPG") for n in CONFIG}

    # fig01: las dos entradas
    vis.fig_entradas(imgs["PVC1"], imgs["PVC2"], "fig01_entradas.png")

    resultados = {}
    # Figuras por hoja: suavizado, comparación de canales, montaje y binaria.
    plan = {
        "PVC1": dict(suav="fig02_pvc1_suavizado.png",
                     canales="fig03_pvc1_canales.png",
                     montaje="fig04_pvc1_pipeline.png",
                     binaria="fig05_pvc1_binaria.png"),
        "PVC2": dict(suav="fig06_pvc2_suavizado.png",
                     canales="fig07_pvc2_canales.png",
                     montaje="fig08_pvc2_pipeline.png",
                     binaria="fig09_pvc2_binaria.png"),
    }

    for nombre, params in CONFIG.items():
        img = imgs[nombre]
        res = ejecutar(nombre, img, params)
        resultados[nombre] = res

        # Figura de suavizado: canal elegido, crudo vs gaussiano vs mediana.
        gauss = pre.suavizado_gaussiano(res.canal_elegido, params.k_suavizado)
        medi = pre.suavizado_mediana(res.canal_elegido, params.k_suavizado)
        vis.fig_suavizado(res.canal_elegido, gauss, medi, params.canal,
                          plan[nombre]["suav"])

        # Figura de canales: los tres candidatos y su Otsu.
        otsus = {}
        for k, c in res.canales.items():
            m = pre.suavizado_mediana(c, params.k_suavizado)
            b, _ = seg.umbral_otsu(m)
            otsus[k] = b
        vis.fig_canales(res.canales, otsus, params.canal, plan[nombre]["canales"])

        # Montaje de 8 etapas y figura de binaria final + verificación.
        vis.fig_montaje(res, plan[nombre]["montaje"])
        vis.fig_binaria_final(res, plan[nombre]["binaria"])

        # Binaria final como archivo independiente.
        cv2.imwrite(str(SALIDAS / f"{nombre}_binaria.png"), res.final)
        print(f"{nombre}: canal={params.canal}  umbral Otsu={res.umbral:.0f}  "
              f"área hoja={seg.fraccion_objeto(res.final)*100:.1f}%")

    print("Figuras y binarias escritas en", SALIDAS)


if __name__ == "__main__":
    main()
