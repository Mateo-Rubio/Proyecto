# analysis.py
import numpy as np
from scipy.optimize import curve_fit

# --- Constantes Físicas ---
C_LIGHT = 299792458.0              # m/s
KB = 1.380649e-23                  # J/K
M_CS = 132.905e-3 / 6.02214076e23  # Masa de un átomo de Cesio-133 en kg

def gaussian_tpa_freq(nu, a, b, nu_L_center, gamma_D):
    """
    Función teórica: I = a + b * exp(-((2*(nu - nu_L_center)) / Gamma_D)^2)
    nu: Frecuencia del láser (THz)
    nu_L_center: Frecuencia de resonancia del láser (~364.5029 THz)
    gamma_D: Parámetro de ensanchamiento Doppler (Gamma_D)
    """
    return a + b * np.exp(-((2 * (nu - nu_L_center)) / gamma_D)**2)

def calcular_densidad_cesio_steck(T_celsius):
    """
    Calcula la densidad atómica del vapor de Cesio (átomos/cm^3) 
    para una temperatura dada en grados Celsius.
    Utiliza el modelo empírico de presión de vapor de Daniel A. Steck.
    """
    T = T_celsius + 273.15  # Convertir a Kelvin absoluto
    
    # Evaluar fase basada en el Melting Point (28.44 °C)
    if T_celsius < 28.44:
        log10_P = -219.48200 + (1088.676 / T) - (0.08336185 * T) + (94.88752 * np.log10(T))
    else:
        log10_P = 8.22127 - (4006.048 / T) - (0.00060194 * T) - (0.19623 * np.log10(T))
        
    P_torr = 10**log10_P
    P_pa = P_torr * 133.322368
    
    N_m3 = P_pa / (KB * T)  
    N_cm3 = N_m3 * 1e-6     
    
    return N_cm3

def calculate_temperature(gamma_D, nu_L_center):
    """
    Despeje riguroso de la Ecuación B.6.
    """
    T_kelvin = ((M_CS * C_LIGHT**2) / (8*KB)) * (gamma_D / nu_L_center)**2
    return T_kelvin

def fit_all_spectra(dataset):
    fit_results = {}
    
    for temp_meas, data in dataset.items():
        lam_nm = data["wavelength"]
        pmt_raw = data["pmt"]
        
        # 1. Transformación Rigurosa a Frecuencia (THz)
        nu_thz = C_LIGHT / (lam_nm * 1e3) 
        
        sort_idx = np.argsort(nu_thz)
        nu_thz = nu_thz[sort_idx]
        pmt = pmt_raw[sort_idx]
        
        dataset[temp_meas]["frequency_thz"] = nu_thz
        dataset[temp_meas]["pmt_sorted"] = pmt
        
        # 2. Estimaciones Iniciales
        a_guess = np.median(pmt)
        b_guess = np.max(pmt) - a_guess
        nu_center_guess = C_LIGHT / (822.4689 * 1e3) 
        gamma_D_guess = 0.0005 
        
        # 3. Límites (Bounds)
        bounds_min = [0, 0, 364.4, 0]
        bounds_max = [np.inf, np.inf, 364.6, 0.005]
        
        try:
            popt, _ = curve_fit(
                gaussian_tpa_freq, nu_thz, pmt, 
                p0=[a_guess, b_guess, nu_center_guess, gamma_D_guess],
                bounds=(bounds_min, bounds_max),
                maxfev=10000
            )
            
            a_fit, b_fit, nu_center_fit, gamma_D_fit = popt
            
            t_fit_k = calculate_temperature(gamma_D_fit, nu_center_fit)
            t_fit_c = t_fit_k - 273.15
            
            # --- NUEVO: Cálculo del área bajo la curva ---
            # Integral analítica de la Gaussiana parametrizada con Gamma_D
            area_fit = b_fit * (np.sqrt(np.pi) / 2.0) * gamma_D_fit
            
            fit_results[temp_meas] = {
                "popt": popt,           
                "T_fit_C": t_fit_c,     
                "T_fit_K": t_fit_k,
                "Area": area_fit        # Guardamos el área
            }
            
        except RuntimeError:
            print(f"Advertencia: El algoritmo no convergió para {temp_meas}°C")
            
    return fit_results