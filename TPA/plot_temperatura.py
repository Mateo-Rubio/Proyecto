# plot_grafica2_temperaturas.py
import matplotlib.pyplot as plt
from data_loader import load_spectra_data
from analysis import fit_all_spectra

def main():
    dataset = load_spectra_data()
    if not dataset:
        return

    fit_results = fit_all_spectra(dataset)
    if not fit_results:
        print("No se generaron resultados de ajuste.")
        return

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
    plt.savefig("temp_ajuste_vs_medida.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()