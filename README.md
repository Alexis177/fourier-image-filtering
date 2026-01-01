# Fourier Image Filtering

Proyecto en Python que aplica la Transformada de Fourier 2D para el filtrado de imágenes
mediante filtros pasa-bajas y pasa-altas.

## Requisitos
- Python 3.10 o 3.11

## Instalación
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt


Colocar una imagen (.png o .jpg) en la carpeta:

imagenes/input/


Verificar la ruta de la imagen en main.py:

IMG_PATH = "imagenes/input/edificio.png"


Ejecutar el programa:

python main.py


El programa mostrará los resultados en pantalla y guardará las imágenes generadas en
imagenes/output/.