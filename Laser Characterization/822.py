import pandas as pd
import numpy as np 
import glob
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
import os

# Set formatting for matplotlib
mpl.rcParams['text.usetex'] = False

dataFrames_datos = {}
N = 20 # Número de datos tomados ajustados en el logsetup

# --- CORTAR PUNTOS AL FINAL ---
CORTAR_AL_FINAL = 13 # Cambia este número para borrar los últimos N puntos de la medición
# ------------------------------

posiciones = []

# Construct the absolute path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
# 1. Update the folder and file search pattern to 822nm
search_path = os.path.join(script_dir, "data2","822nm", "z*mm-822nm.xlsx")

archivos = glob.glob(search_path)

for archivo in archivos:
    nombre_archivo = os.path.basename(archivo)
    
    # 2. Update regex to look for "822nm"
    match = re.search(r"z(\d+(\.\d+)?)mm-822nm\.xlsx", nombre_archivo)
    if match:
        posicion = float(match.group(1))
        posiciones.append(posicion)

        # Read the file, skipping the 21 rows of BeamMaster metadata
        df = pd.read_excel(archivo, skiprows=21)
        
        # Drop the first row (index 0) which contains the text units like '(micron)'
        df = df.drop(0).reset_index(drop=True)
        
        # Clean the comma decimals so Python can do math on them
        cols_to_convert = ["W Width I", "V Width I"]
        for col in cols_to_convert:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
        # Store only the first N valid numerical data rows
        dataFrames_datos[posicion] = df.head(N)

# Sort the positions so the x-axis arrays and linspace are ordered correctly
posiciones.sort()

# --- CUTTING LOGIC ---
if CORTAR_AL_FINAL > 0:
    lista_posiciones = posiciones[:-CORTAR_AL_FINAL] # Slices off the last N points
    print(f"Ignorando los últimos {CORTAR_AL_FINAL} puntos. Puntos restantes: {len(lista_posiciones)}")
else:
    lista_posiciones = posiciones
# ---------------------

# Pre-allocate arrays
W_width_I = np.zeros_like(lista_posiciones, dtype=float)
V_width_I = np.zeros_like(lista_posiciones, dtype=float)
std_W_width_I = np.zeros_like(lista_posiciones, dtype=float)
std_V_width_I = np.zeros_like(lista_posiciones, dtype=float)

# Process mean and standard deviation
for i, posicion in enumerate(lista_posiciones):
    W_data = dataFrames_datos[posicion]["W Width I"].to_numpy() / 2 * 1e-6
    V_data = dataFrames_datos[posicion]["V Width I"].to_numpy() / 2 * 1e-6
    
    W_width_I[i] = W_data.mean()
    V_width_I[i] = V_data.mean()
    std_W_width_I[i] = W_data.std()
    std_V_width_I[i] = V_data.std()

# Convert variables to SI units / processing units
z = np.array(lista_posiciones) * 1e-3

# Gaussian Beam Formula
def w(z, w_0, z_0, z_R):
    return w_0 * np.sqrt(1 + ((z - z_0) / z_R)**2)

# Fit curves to data
params1, cov1 = curve_fit(w, z, W_width_I)
params2, cov2 = curve_fit(w, z, V_width_I)

# Setup z_linspace using min/max of sorted array
if len(lista_posiciones) > 0:
    z_linspace = np.linspace(min(lista_posiciones), max(lista_posiciones), 200) * 1e-3
    w_lins_W = w(z_linspace, *params1)
    w_lins_V = w(z_linspace, *params2)
else:
    print("No data extracted. Check your file names and paths.")

# Plotting the Final Graph
fig, ax = plt.subplots(figsize=(10, 5))

# Plot Error Bars for W and V directions
plt.errorbar(z * 1e3, W_width_I * 1e6, yerr=std_W_width_I * 1e6, fmt="*", label="Direction $W$", capsize=3)
plt.errorbar(z * 1e3, V_width_I * 1e6, yerr=std_V_width_I * 1e6, fmt="*", label="Direction $V$", capsize=3)

# Plot Fitted Curves
if len(lista_posiciones) > 0:
    plt.plot(z_linspace * 1e3, w_lins_W * 1e6, "--", label="Fit on $W$ Direction data")
    plt.plot(z_linspace * 1e3, w_lins_V * 1e6, "--", label="Fit on $V$ Direction data")

# Graph labels and styling
plt.xlabel(r"Position in propagation axis $z$ [mm]")
plt.ylabel(r"Beam waist $w(z)$ [$\mu$m]")
# 3. Update title to 822nm
plt.title("Beam Characterization - 822nm Laser")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 4. Update the savefile to 822nm
plt.savefig("822nm_beam_waist_fit.png")

# --- IMPRESIÓN DE PARÁMETROS AJUSTADOS ---
print("\n" + "="*40)
print(" PARÁMETROS AJUSTADOS DEL LÁSER")
print("="*40)

# Dirección W (params1)
print("\n--- Dirección W (Eje Horizontal) ---")
print(f"Cintura del haz (w_0):   {params1[0]*1e6:.2f} µm")
print(f"Posición focal (z_0):    {params1[1]*1e3:.2f} mm")
print(f"Rango de Rayleigh (z_R): {params1[2]*1e3:.2f} mm")

# Dirección V (params2)
print("\n--- Dirección V (Eje Vertical) ---")
print(f"Cintura del haz (w_0):   {params2[0]*1e6:.2f} µm")
print(f"Posición focal (z_0):    {params2[1]*1e3:.2f} mm")
print(f"Rango de Rayleigh (z_R): {params2[2]*1e3:.2f} mm")
print("="*40)


# --- CÁLCULO DE INTENSIDAD ---
print("\n" + "="*40)
print(" CÁLCULO DE INTENSIDAD EN EL FOCO")
print("="*40)

# Pedir al usuario la potencia en milivatios (mW)
potencia_input = input("Potencia del láser(mW): ")

try:
    # Convertir el texto ingresado a un número flotante
    potencia_mW = float(potencia_input)
    
    # Calcular el área en el foco (z_0) usando los parámetros ajustados
    # params[0] es la cintura del haz (w_0) en metros
    w_0_W_metros = params1[0]
    w_0_V_metros = params2[0]
    
    # Área en m^2 = pi * w_W * w_V
    area_m2 = np.pi * w_0_W_metros * w_0_V_metros
    
    # Convertir el Área a cm^2 (1 m^2 = 10,000 cm^2)
    area_cm2 = area_m2 * 1e4
    
    # Calcular Intensidad Pico (I_0 = 2P / Area)
    intensidad_pico_mW_cm2 = (2 * potencia_mW) / area_cm2
    intensidad_pico_W_cm2 = intensidad_pico_mW_cm2 / 1000
    
    print(f"\nResultados para {potencia_mW} mW:")
    print(f"Área focal (1/e²): {area_cm2:.2e} cm²")
    print(f"Intensidad Pico (I_0): {intensidad_pico_mW_cm2:.2f} mW/cm²")
    print(f"Intensidad Pico (I_0): {intensidad_pico_W_cm2:.2f} W/cm²")

except ValueError:
    print("\nError: Por favor, ejecute el script nuevamente e ingrese un número válido para la potencia.")