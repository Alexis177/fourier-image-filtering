import numpy as np  # Librería numérica para arreglos y FFT

def fft2_image(img: np.ndarray) -> np.ndarray:
    """
    Calcula la Transformada de Fourier 2D (FFT2) de una imagen y centra el espectro.
    - Entrada: img (2D) en dominio espacial (grises, float).
    - Salida: espectro complejo centrado (0 Hz en el centro).
    """
    f = np.fft.fft2(img)          # FFT 2D: pasa de dominio espacial a dominio frecuencia
    return np.fft.fftshift(f)     # Mueve (0,0) al centro para visualizar/filtrar más fácil

def ifft2_image(fshift: np.ndarray) -> np.ndarray:
    """
    Reconstruye la imagen aplicando IFFT2 desde un espectro centrado.
    - Entrada: fshift (espectro complejo centrado).
    - Salida: imagen reconstruida en dominio espacial (magnitud).
    """
    f_ishift = np.fft.ifftshift(fshift)  # Regresa el cero de frecuencia a la esquina (formato original)
    img_back = np.fft.ifft2(f_ishift)    # IFFT 2D: vuelve al dominio espacial (resultado complejo)
    return np.abs(img_back)              # Magnitud para obtener imagen real/visible

def low_pass_filter(shape, cutoff: float) -> np.ndarray:
    """
    Construye una máscara pasa-bajas IDEAL circular.
    - shape: (filas, columnas) de la imagen.
    - cutoff: radio del círculo (frecuencias permitidas).
    - Salida: mask 2D con 1 dentro del círculo y 0 fuera.
    """
    rows, cols = shape                                # Dimensiones
    mask = np.zeros((rows, cols), dtype=np.float32)    # Inicializa máscara en ceros
    crow, ccol = rows // 2, cols // 2                  # Centro del espectro (por fftshift)

    # Recorremos cada punto del plano de frecuencias (u,v)
    for i in range(rows):
        for j in range(cols):
            # Distancia euclidiana al centro (frecuencia 0)
            dist = ((i - crow) ** 2 + (j - ccol) ** 2) ** 0.5
            # Si está dentro del radio, se deja pasar (1)
            if dist <= cutoff:
                mask[i, j] = 1.0

    return mask  # Regresa máscara pasa-bajas

def high_pass_filter(shape, cutoff: float) -> np.ndarray:
    """
    Construye una máscara pasa-altas IDEAL circular como complemento del pasa-bajas.
    - HPF = 1 - LPF
    """
    return 1.0 - low_pass_filter(shape, cutoff)
