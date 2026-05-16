# plot_grafica1_espectros.py
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_spectra_data
from analysis import fit_all_spectra, gaussian_tpa_freq, C_LIGHT

def main():
    dataset = load_spectra_data()
    if not dataset:
        return

    fit_results = fit_all_spectra(dataset)
    if not fit_results:
        print("No se generaron resultados de ajuste.")
        return

    plt.figure(figsize=(10, 5))
    
    for temp in sorted(dataset.keys()):
        lam = dataset[temp]["wavelength"]
        pmt = dataset[temp]["pmt"]
        std_pmt = dataset[temp]["std_pmt"]
        
        # Graficar datos con barras de error
        p = plt.errorbar(lam, pmt, yerr=std_pmt, marker='.', linestyle='', alpha=0.4, label=f"{temp}C", capsize=2)
        
        if temp in fit_results:
            popt = fit_results[temp]["popt"]
            nu_min = C_LIGHT / (lam.max() * 1e3)
            nu_max = C_LIGHT / (lam.min() * 1e3)
            nu_smooth = np.linspace(nu_min, nu_max, 500)
            
            lam_smooth = C_LIGHT / (nu_smooth * 1e3)
            
            plt.plot(lam_smooth, gaussian_tpa_freq(nu_smooth, *popt), color=p[0].get_color(), linewidth=2)

    plt.title("Espectro de Fluorescencia TPA Cesio",fontsize=16)
    plt.xlabel(r"Longitud de Onda del Laser $\lambda$ (nm)", fontsize=16)
    plt.ylabel("Conteos PMT (Promedio)", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("./figures/espectros_longitud_onda.png", dpi=300)

if __name__ == "__main__":
    main()