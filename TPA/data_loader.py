# data_loader.py
import numpy as np
import glob
import re
import os
from config import COLUMNS, FILE_PATTERN

def extract_temperature(filename):
    """
    Extrae la temperatura como número flotante del nombre del archivo.
    Ejemplo: 'DopplerBroadened79.6C...' -> 79.6
    """
    match = re.search(r"Broadened(\d+\.\d+|\d+)C", os.path.basename(filename))
    return float(match.group(1)) if match else None

def load_spectra_data(data_dir="data2"):
    """
    Carga los archivos de espectroscopía aplicando un filtro de limpieza 
    física para descartar caídas de señal del ondámetro y ceros del PMT.
    """
    search_path = os.path.join(data_dir, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"Advertencia: No se encontraron archivos con el patrón {search_path}")
        return {}

    dataset = {}
    
    for filepath in files:
        temp = extract_temperature(filepath)
        if temp is None:
            continue
            
        # 1. Cargar datos brutos
        raw_data = np.loadtxt(filepath)
        wavelength = raw_data[:, COLUMNS["wavelength"]]
        pmt = raw_data[:, COLUMNS["pmt"]]
        
        # 2. FILTRO RIGUROSO DE LIMPIEZA
        # - Descartar PMT en cero
        # - Descartar errores de lectura del wavemeter (ej. códigos -3.0 o saltos fuera de rango)
        mask_valid = (pmt > 0) & (wavelength > 800.0) & (wavelength < 830.0)
        
        wavelength = wavelength[mask_valid]
        pmt = pmt[mask_valid]
        
        # Si el archivo entero era ruido o falló por completo, lo saltamos
        if len(pmt) == 0:
            print(f"Advertencia: El archivo para {temp}°C se descartó por falta de datos válidos.")
            continue
        
        # 3. Redondear ligeramente para agrupar mediciones en la misma longitud de onda
        rounded_wl = np.round(wavelength, decimals=4)
        
        # 4. Promediar puntos que caen en el mismo valor de longitud de onda
        unique_wl, inverse_indices = np.unique(rounded_wl, return_inverse=True)
        sum_pmt = np.bincount(inverse_indices, weights=pmt)
        counts_per_wl = np.bincount(inverse_indices)
        avg_pmt = sum_pmt / counts_per_wl
        
        dataset[temp] = {
            "wavelength": unique_wl,
            "pmt": avg_pmt
        }
        
    return dataset