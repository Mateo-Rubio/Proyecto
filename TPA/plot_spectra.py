# plot_spectra.py
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_spectra_data
from analysis import fit_all_spectra, gaussian_tpa_freq, calcular_densidad_cesio_steck

def main():
    # 1. Cargar datos
    dataset = load_spectra_data(data_dir="data")
    if not dataset:
        return

    # 2. Correr el módulo de ajuste (internamente mapea a THz y calcula Área)
    fit_results = fit_all_spectra(dataset)
    if not fit_results:
        print("No se generaron resultados de ajuste.")
        return

    # ==========================================
    # GRÁFICA 1: Espectros en Frecuencia
    # ==========================================
    plt.figure(figsize=(10, 5))
    
    for temp in sorted(dataset.keys()):
        nu = dataset[temp]["frequency_thz"]
        pmt = dataset[temp]["pmt_sorted"]
        
        # Puntos experimentales
        p = plt.plot(nu, pmt, marker='.', linestyle='', alpha=0.4, label=f"Datos {temp}°C")
        
        # Curva continua del ajuste
        if temp in fit_results:
            popt = fit_results[temp]["popt"]
            nu_smooth = np.linspace(nu.min(), nu.max(), 500)
            plt.plot(nu_smooth, gaussian_tpa_freq(nu_smooth, *popt), color=p[0].get_color(), linewidth=2)

    plt.title("Ajuste Gaussiano Doppler-Broadened TPA")
    plt.xlabel(r"Frecuencia del Láser $\nu$ (THz)")
    plt.ylabel("Conteos PMT (Promedio)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # ==========================================
    # GRÁFICA 2: Temperatura Medida vs Ajustada
    # ==========================================
    measured_temps = []
    fit_temps = []

    for temp in sorted(fit_results.keys()):
        measured_temps.append(temp)
        fit_temps.append(fit_results[temp]["T_fit_C"])

    plt.figure(figsize=(7, 5))
    plt.plot(measured_temps, fit_temps, 'ro-', linewidth=2, markersize=8, label="Temp. Ajuste")
    
    t_min, t_max = min(measured_temps), max(measured_temps)
    plt.plot([t_min, t_max], [t_min, t_max], 'k--', label="Ideal ($T_{ajuste} = T_{medida}$)")

    plt.title("Comparación: Termocupla vs Espectroscopía (Dominio THz)")
    plt.xlabel("Temperatura Medida por Termocupla (°C)")
    plt.ylabel("Temperatura Extraída del Ajuste (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # ==========================================
    # GRÁFICA 3: Densidad (Steck) vs Área Experimental
    # ==========================================
    experiment_areas = [fit_results[t]["Area"] for t in measured_temps]
    teorica_puntos = [calcular_densidad_cesio_steck(t) for t in measured_temps]

    # Factor de escala anclado al primer punto (el más frío)
    factor_escala = teorica_puntos[0] / experiment_areas[0]
    areas_escaladas = [area * factor_escala for area in experiment_areas]

    # Curva teórica continua para el fondo
    T_smooth = np.linspace(min(measured_temps) - 5, max(measured_temps) + 5, 100)
    densidad_smooth = [calcular_densidad_cesio_steck(t) for t in T_smooth]

    plt.figure(figsize=(9, 6))
    
    # Línea teórica de Steck
    plt.plot(T_smooth, densidad_smooth, 'k--', linewidth=2, label="Densidad Teórica (Steck)")
    
    # Puntos experimentales (Áreas escaladas)
    plt.plot(measured_temps, areas_escaladas, 'ro', markersize=9, 
             markeredgecolor='black', label="Densidad Exp. (Basada en Área TPA)")

    plt.title("Densidad Atómica del Cesio: Teoría vs Espectroscopía TPA")
    plt.xlabel("Temperatura Medida (°C)")
    plt.ylabel(r"Densidad Atómica ($\text{átomos} / \text{cm}^3$)")
    plt.yscale('log') # Eje Y logarítmico
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Mostrar las 3 gráficas simultáneamente
    plt.show()

    # Resumen en consola
    print("\n--- RESUMEN DE DENSIDADES ---")
    for i, t in enumerate(measured_temps):
        error = abs(areas_escaladas[i] - teorica_puntos[i]) / teorica_puntos[i] * 100
        print(f"Temp: {t}°C | Teoría: {teorica_puntos[i]:.2e} | Exp: {areas_escaladas[i]:.2e} | Error: {error:.1f}%")

if __name__ == "__main__":
    main()