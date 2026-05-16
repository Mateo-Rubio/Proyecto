# data_loader.py
import numpy as np
import glob
import re
import os
from config import COLUMNS, FILE_PATTERN

def extract_temperature(filename):
    match = re.search(r"Broadened(\d+\.\d+|\d+)C", os.path.basename(filename))
    return float(match.group(1)) if match else None

def load_spectra_data(data_dir="data2"):
    search_path = os.path.join(data_dir, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"Advertencia: No se encontraron archivos con el patron {search_path}")
        return {}

    dataset = {}
    
    for filepath in files:
        temp = extract_temperature(filepath)
        if temp is None:
            continue
            
        raw_data = np.loadtxt(filepath)
        wavelength = raw_data[:, COLUMNS["wavelength"]]
        pmt = raw_data[:, COLUMNS["pmt"]]
        
        mask_valid = (wavelength > 800.0) & (wavelength < 822.47) & (pmt > 0)
        
        wavelength = wavelength[mask_valid]
        pmt = pmt[mask_valid]
        
        if len(pmt) == 0:
            print(f"Advertencia: El archivo para {temp}C se descarto por falta de datos validos.")
            continue
        
        rounded_wl = np.round(wavelength, decimals=5)
        
        unique_wl, inverse_indices = np.unique(rounded_wl, return_inverse=True)
        
        # Calcular promedio
        sum_pmt = np.bincount(inverse_indices, weights=pmt)
        counts_per_wl = np.bincount(inverse_indices)
        avg_pmt = sum_pmt / counts_per_wl
        
        # Calcular desviacion estandar poblacional para cada grupo
        sum_pmt_sq = np.bincount(inverse_indices, weights=pmt**2)
        variance = (sum_pmt_sq / counts_per_wl) - avg_pmt**2
        # Prevenir varianzas negativas por precision de punto flotante
        variance[variance < 0] = 0
        std_pmt = np.sqrt(variance)
        
        dataset[temp] = {
            "wavelength": unique_wl,
            "pmt": avg_pmt,
            "std_pmt": std_pmt
        }
        
    return dataset