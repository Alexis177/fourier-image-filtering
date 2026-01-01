import os                      # Para manejar rutas y crear carpetas
import cv2                     # OpenCV para leer/guardar imágenes
import numpy as np             # Arreglos, log, abs, etc.
import matplotlib.pyplot as plt # Para visualizar resultados

from filters import fft2_image, ifft2_image, low_pass_filter, high_pass_filter
from metrics import mse

# =========================
# CONFIGURACIÓN
# =========================
IMG_PATH = "imagenes/input/edificio.png"  # Ruta de la imagen de entrada
OUT_DIR = "imagenes/output"              # Carpeta donde guardaremos resultados
CUTOFF = 40                              # Radio del filtro (ajustable)

# Crea la carpeta de salida si no existe (no falla si ya existe)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) CARGA + GRIS + NORMALIZACIÓN
# =========================
# Lee la imagen en escala de grises directamente (2D)
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# Si la ruta es incorrecta o el archivo no existe, img será None
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {IMG_PATH}")

# Convierte a float32 y normaliza a rango [0,1] para operar de forma estable
img = img.astype(np.float32) / 255.0

# =========================
# 2) FFT2 (centrada)
# =========================
# Obtiene el espectro complejo centrado
fshift = fft2_image(img)

# Magnitud del espectro para visualizar (log para comprimir rango dinámico)
spectrum_before = np.log(1 + np.abs(fshift))

# =========================
# 3) MÁSCARAS (AL MENOS 2)
# =========================
# Máscara pasa-bajas y pasa-altas con el mismo cutoff
lpf = low_pass_filter(img.shape, CUTOFF)
hpf = high_pass_filter(img.shape, CUTOFF)

# =========================
# 4) FILTRADO + RECONSTRUCCIÓN (IFFT2)
# =========================
# Aplica filtrado en el dominio de frecuencia: multiplicación punto a punto
f_lpf = fshift * lpf
f_hpf = fshift * hpf

# Reconstruye las imágenes filtradas regresando al dominio espacial
img_lpf = ifft2_image(f_lpf)
img_hpf = ifft2_image(f_hpf)

# Espectros después del filtrado (también en escala log)
spectrum_lpf = np.log(1 + np.abs(f_lpf))
spectrum_hpf = np.log(1 + np.abs(f_hpf))

# =========================
# 5) MÉTRICAS (MSE)
# =========================
# Calcula el error cuadrático medio entre original y reconstruidas
mse_lpf = mse(img, img_lpf)
mse_hpf = mse(img, img_hpf)

# Imprime resultados para evidencia y para el reporte
print(f"Cutoff = {CUTOFF}")
print(f"MSE Pasa-bajas: {mse_lpf:.6f}")
print(f"MSE Pasa-altas: {mse_hpf:.6f}")

# =========================
# 6) GUARDAR RESULTADOS (PARA REPORTE)
# =========================
def save_gray(path: str, arr01: np.ndarray) -> None:
    """
    Guarda una imagen float en [0,1] como PNG en [0,255].
    """
    arr01 = np.clip(arr01, 0, 1)              # Asegura rango válido
    arr255 = (arr01 * 255).astype(np.uint8)   # Convierte a uint8
    cv2.imwrite(path, arr255)                 # Guarda en disco

# Guarda imágenes principales
save_gray(os.path.join(OUT_DIR, "original.png"), img)
save_gray(os.path.join(OUT_DIR, f"lpf_cutoff_{CUTOFF}.png"), img_lpf)
save_gray(os.path.join(OUT_DIR, f"hpf_cutoff_{CUTOFF}.png"), img_hpf)

def save_spectrum(path: str, spec: np.ndarray) -> None:
    """
    Normaliza un espectro para guardarlo como imagen visible.
    """
    spec_norm = spec - spec.min()         # Lleva mínimo a 0
    if spec_norm.max() != 0:
        spec_norm = spec_norm / spec_norm.max()  # Escala a [0,1]
    save_gray(path, spec_norm)            # Reutiliza función de guardado

# Guarda espectros (antes y después)
save_spectrum(os.path.join(OUT_DIR, "spectrum_before.png"), spectrum_before)
save_spectrum(os.path.join(OUT_DIR, f"spectrum_lpf_{CUTOFF}.png"), spectrum_lpf)
save_spectrum(os.path.join(OUT_DIR, f"spectrum_hpf_{CUTOFF}.png"), spectrum_hpf)

# =========================
# 7) VISUALIZACIÓN (MÍNIMA Y COMPLETA)
# =========================
plt.figure(figsize=(12, 8))  # Tamaño de la ventana

# Original
plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

# LPF
plt.subplot(2, 3, 2)
plt.title(f"Pasa-bajas (cutoff={CUTOFF})")
plt.imshow(img_lpf, cmap="gray")
plt.axis("off")

# HPF
plt.subplot(2, 3, 3)
plt.title(f"Pasa-altas (cutoff={CUTOFF})")
plt.imshow(img_hpf, cmap="gray")
plt.axis("off")

# Espectro antes
plt.subplot(2, 3, 4)
plt.title("Espectro (antes, log)")
plt.imshow(spectrum_before, cmap="gray")
plt.axis("off")

# Espectro LPF
plt.subplot(2, 3, 5)
plt.title("Espectro (LPF, log)")
plt.imshow(spectrum_lpf, cmap="gray")
plt.axis("off")

# Espectro HPF
plt.subplot(2, 3, 6)
plt.title("Espectro (HPF, log)")
plt.imshow(spectrum_hpf, cmap="gray")
plt.axis("off")

plt.tight_layout()  # Ajusta espacios entre subplots
plt.show()          # Muestra ventana
