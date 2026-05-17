import numpy as np
import glob
import re
import os
from config import COLUMNS
import scipy.constants as const

C_LIGHT = const.c # Velocidad de la luz en m/s
print(C_LIGHT)

def load_deltas_file(data_dir="data3"):
    """
    Lee el archivo deltas.txt y retorna una lista con el Delta calculado en Hz.
    La línea 1 corresponde a Delta1, la línea 2 a Delta2, etc.
    """
    deltas_path = os.path.join(data_dir, "deltas.txt")
    if not os.path.exists(deltas_path):
        print(f"Advertencia: No se encontro el archivo {deltas_path}")
        return []
        
    true_deltas_hz = []
    lam_0 = 852.3565 * 1e-9 # Transición en metros
    
    with open(deltas_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("."):
                # Convertimos el formato .XXXX a un float sumado a 852 nm
                lam_laser_nm = 852.0 + float(line)
                lam_laser_m = lam_laser_nm * 1e-9
                
                # Cálculo de la desintonía correcta en Hz (Diferencia de frecuencias)
                # Delta_Hz = nu_0 - nu_laser = c/lam_0 - c/lam_laser
                delta_hz = (C_LIGHT / lam_0) - (C_LIGHT / lam_laser_m)
                true_deltas_hz.append(delta_hz)
                
    return true_deltas_hz

def extract_delta_index(filename):
    match = re.search(r"Delta(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None

def load_spectra_data_delta(data_dir="data3"):
    true_deltas_hz = load_deltas_file(data_dir)
    print(true_deltas_hz)
    search_path = os.path.join(data_dir, "*Delta*")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Advertencia: No se encontraron archivos con el patrón {search_path}")
        return {}

    dataset = {}
    
    for filepath in files:
        idx = extract_delta_index(filepath)
        # Verificamos que el índice extraído tenga correspondencia en el txt
        if idx is None or idx < 1 or idx > len(true_deltas_hz):
            continue
            
        # Asignamos el delta calculado (idx - 1 porque las listas en python inician en 0)
        true_delta = true_deltas_hz[idx - 1]
            
        raw_data = np.loadtxt(filepath)
        wavelength = raw_data[:, COLUMNS["wavelength"]]
        pmt = raw_data[:, COLUMNS["pmt"]]
        
        # Nueva máscara ajustada a la región de interés
        mask_valid = (wavelength > 822.4675) & (wavelength < 822.47) & (pmt > 0)
        
        wavelength = wavelength[mask_valid]
        pmt = pmt[mask_valid]
        
        # Conteo total de datos en el barrido tras aplicar el rango
        num_datos = len(pmt)
        
        if num_datos == 0:
            print(f"Advertencia: El archivo para Delta{idx} se descartó por falta de datos válidos.")
            continue
        
        rounded_wl = np.round(wavelength, decimals=5)
        unique_wl, inverse_indices = np.unique(rounded_wl, return_inverse=True)
        
        # Calcular promedio
        sum_pmt = np.bincount(inverse_indices, weights=pmt)
        counts_per_wl = np.bincount(inverse_indices)
        avg_pmt = sum_pmt / counts_per_wl
        
        # Calcular desviación estándar poblacional
        sum_pmt_sq = np.bincount(inverse_indices, weights=pmt**2)
        variance = (sum_pmt_sq / counts_per_wl) - avg_pmt**2
        variance[variance < 0] = 0
        std_pmt = np.sqrt(variance)
        
        dataset[idx] = {
            "true_delta_hz": true_delta,
            "wavelength": unique_wl,
            "pmt": avg_pmt,
            "std_pmt": std_pmt,
            "num_datos": num_datos  # Retornamos el número de datos para la tabla
        }
        
    return dataset