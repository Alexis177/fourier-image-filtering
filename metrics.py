import numpy as np  # Para operar arreglos y calcular promedio

def mse(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Mean Squared Error (Error Cuadrático Medio).
    MSE = promedio( (original - filtrada)^2 )
    - Entre más pequeño, más parecidas.
    """
    # Diferencia pixel a pixel
    diff = (original - filtered)
    # Error cuadrático medio
    return float(np.mean(diff ** 2))
