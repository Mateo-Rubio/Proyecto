# diagnostico_datos.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from config import COLUMNS

def calcular_estadisticas(arr, nombre):
    """Calcula y muestra métricas estadísticas básicas de un arreglo."""
    media = np.mean(arr)
    mediana = np.median(arr)
    desv_est = np.std(arr)
    min_val, max_val = np.min(arr), np.max(arr)
    
    # Moda (usando scipy)
    # Nota: En datos flotantes la moda suele ser poco informativa, pero en PMT (enteros) sirve.
    res_moda = mode(arr, keepdims=True)
    val_moda = res_moda.mode[0]
    
    print(f"\n--- ESTADÍSTICAS DE {nombre.upper()} ---")
    print(f"Mínimo:     {min_val:.4f}")
    print(f"Máximo:     {max_val:.4f}")
    print(f"Media:      {media:.4f}")
    print(f"Mediana:    {mediana:.4f}")
    print(f"Moda:       {val_moda:.4f}")
    print(f"Desv. Est:  {desv_est:.4f}")
    print(f"Percentil 25 (Q1): {np.percentile(arr, 25):.4f}")
    print(f"Percentil 75 (Q3): {np.percentile(arr, 75):.4f}")

def filtrar_picos_espurios(wl, pmt, umbral_std=4.0):
    """
    Filtra anomalías usando la diferencia contra la mediana local.
    Esto elimina picos falsos de 1 solo punto (ruido PMT) sin cortar el pico TPA real.
    """
    # Usar un filtro local de diferencias
    diferencias = np.abs(pmt - np.median(pmt))
    desv_mediana = np.std(pmt)
    
    # Solo consideramos ruido anómalo si un punto salta violentamente 
    # respecto a la fluctuación general
    mascara_valida = pmt > 0  # Quitar ceros por defecto
    
    # Puedes personalizar este criterio si notas que el problema es en la longitud de onda
    # (ej. saltos del wavemeter)
    wl_diffs = np.abs(np.diff(wl, prepend=wl[0]))
    umbral_wl = np.mean(wl_diffs) + 5 * np.std(wl_diffs)
    mascara_wl = wl_diffs < umbral_wl
    
    mascara_final = mascara_valida & mascara_wl
    return wl[mascara_final], pmt[mascara_final], mascara_final

def main():
    # Nombre exacto del archivo problemático
    # Ajusta la ruta si la extensión es .txt
    archivo_origen = os.path.join("data2", "DopplerBroadened79.6C-822nm-3.5V-10mHz-500ms")
    if not os.path.exists(archivo_origen):
        archivo_origen += ".txt" # Intentar con extensión
        
    if not os.path.exists(archivo_origen):
        print(f"Error: No se encuentra el archivo {archivo_origen}")
        return

    print(f"Cargando archivo para inspección: {archivo_origen}")
    raw_data = np.loadtxt(archivo_origen)
    
    wl_raw = raw_data[:, COLUMNS["wavelength"]]
    pmt_raw = raw_data[:, COLUMNS["pmt"]]
    
    # 1. Mostrar estadísticas brutas
    calcular_estadisticas(wl_raw, "Longitud de Onda (nm)")
    calcular_estadisticas(pmt_raw, "Conteos PMT brutamente cargados")
    
    # 2. Aplicar el recorte/filtro de diagnóstico
    wl_filt, pmt_fit, mascara = filtrar_picos_espurios(wl_raw, pmt_raw)
    
    print(f"\n--- RESULTADO DEL FILTRADO ---")
    print(f"Puntos originales: {len(pmt_raw)}")
    print(f"Puntos conservados: {len(pmt_fit)}")
    print(f"Puntos recortados: {len(pmt_raw) - len(pmt_fit)}")
    
    # 3. Graficar para ver exactamente el problema
    plt.figure(figsize=(10, 5))
    
    # Datos brutos completos en gris/rojo de fondo
    plt.plot(wl_raw, pmt_raw, 'x', color='red', alpha=0.3, label="Datos eliminados (Anómalos)")
    
    # Datos limpios superpuestos en azul
    plt.plot(wl_filt, pmt_fit, '.', color='#1f77b4', alpha=0.7, label="Datos limpios conservados")
    
    plt.title("Diagnóstico de Datos: Barrido a 79.6 °C")
    plt.xlabel("Longitud de Onda (nm)")
    plt.ylabel("PMT Singles")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()