# MIA_C4_SSVA_Corte_03_Actividad_05

**Segmentación de imágenes a binaria por preprocesamiento**

| | |
|---|---|
| **Alumno** | Víctor Manuel Santos Martínez |
| **Matrícula** | 253220020 |
| **Materia** | Sistemas de Visión Artificial |
| **Profesor** | Dr. Ernesto García Amaro |
| **Actividad** | Actividad 5 — Segmentación de imágenes, Corte 3 |
| **Universidad** | Universidad Politécnica Metropolitana de Hidalgo |
| **Fecha de entrega** | 17 de noviembre de 2026 |

---

## Descripción de la actividad

Se segmentan dos imágenes de 256×256 (`PVC1.JPG` y `PVC2.JPG`), cada una con
**una hoja de planta en primer plano sobre un fondo de concreto gris**, hasta
obtener de cada una una **imagen binaria** producto del preprocesamiento. De
cada hoja se describe el pipeline que lleva a una segmentación óptima.

El problema de fondo es separar lo **cromático** (la hoja, verde o amarilla)
de lo **acromático** (el concreto gris). El pipeline combina las etapas
clásicas del curso con un canal de color discriminante elegido por imagen:

1. Escala de grises (diagnóstico: la intensidad por sí sola no separa).
2. Transformada cromática discriminante: **HSV-S**, **Lab-a\*** o
   **ExG = 2G − R − B**; se comparan las tres y se elige la mejor por imagen.
3. Suavizado paso bajo (gaussiano y mediana; se adopta la mediana 5×5).
4. Umbralización automática de **Otsu**.
5. Limpieza morfológica: apertura, cierre y relleno de huecos por inundación.
6. Selección de la **componente conexa mayor** → máscara binaria final.
7. Verificación por superposición del contorno sobre la original.

**Decisión por imagen:** PVC1 (hoja verde oscura y uniforme) se segmenta mejor
con **ExG**; PVC2 (hoja con clorosis amarilla) con **HSV-S**, que responde a
cualquier croma. El reporte justifica cada elección.

## Estructura del repositorio

```
├── segmentacion.ipynb        # Cuaderno ejecutado con todo el experimento
├── src/
│   ├── preprocesamiento.py   # Grises, suavizado y canales discriminantes
│   ├── segmentacion.py       # Otsu, morfología, relleno, componente mayor
│   ├── pipeline.py           # Pipeline parametrizable por imagen
│   ├── visualizacion.py      # Figuras del reporte
│   └── main.py               # Reproduce todo: binarias + figuras
├── imagenes/                 # PVC1.JPG, PVC2.JPG (entradas)
├── salidas/                  # Figuras y binarias finales (PVCx_binaria.png)
└── requirements.txt
```

## Reproducir

```bash
pip install -r requirements.txt
python -m src.main
# o, para el cuaderno:
jupyter nbconvert --to notebook --execute --inplace segmentacion.ipynb
```
