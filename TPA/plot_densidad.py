# plot_grafica3_densidades.py
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_spectra_data
from analysis import fit_all_spectra, calcular_densidad_cesio_steck

def main():
    dataset = load_spectra_data()
    if not dataset:
        return

    fit_results = fit_all_spectra(dataset)
    if not fit_results:
        print("No se generaron resultados de ajuste.")
        return

    measured_temps = sorted(fit_results.keys())
    experiment_areas = [fit_results[t]["Area"] for t in measured_temps]
    teorica_puntos = [calcular_densidad_cesio_steck(t) for t in measured_temps]

    factor_escala = teorica_puntos[0] / experiment_areas[0]
    areas_escaladas = [area * factor_escala for area in experiment_areas]

    T_smooth = np.linspace(min(measured_temps) - 5, max(measured_temps) + 5, 100)
    densidad_smooth = [calcular_densidad_cesio_steck(t) for t in T_smooth]

    plt.figure(figsize=(9, 6))
    plt.plot(T_smooth, densidad_smooth, 'k--', linewidth=2, label="Densidad Teórica (Steck)")
    plt.plot(measured_temps, areas_escaladas, 'ro', markersize=9, 
             markeredgecolor='black', label="Densidad Exp. (Basada en Área TPA)")

    plt.title("Densidad Atómica del Cesio: Teoría vs Espectroscopía TPA")
    plt.xlabel("Temperatura Medida (°C)")
    plt.ylabel(r"Densidad Atómica ($\text{átomos} / \text{cm}^3$)")
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("densidad_teorica_vs_exp.png", dpi=300)
    plt.show()

    print("\n--- RESUMEN DE DENSIDADES ---")
    for i, t in enumerate(measured_temps):
        error = abs(areas_escaladas[i] - teorica_puntos[i]) / teorica_puntos[i] * 100
        print(f"Temp: {t}°C | Teoría: {teorica_puntos[i]:.2e} | Exp: {areas_escaladas[i]:.2e} | Error: {error:.1f}%")

if __name__ == "__main__":
    main()