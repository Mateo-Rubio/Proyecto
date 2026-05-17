import matplotlib.pyplot as plt
import numpy as np
from data_loader_delta import load_spectra_data_delta
from analysis import fit_all_spectra, gaussian_tpa_freq, C_LIGHT

# ==============================================================================
# VARIABLES GLOBALES PARA CONTROL DE ESTILOS
# ==============================================================================
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 15

def format_hz(hz):
    """Formatea la frecuencia en GHz o MHz para mejor lectura"""
    if abs(hz) >= 1e9:
        return f"{hz/1e9:.2f} GHz"
    else:
        return f"{hz/1e6:.2f} MHz"

def main():
    dataset = load_spectra_data_delta()
    if not dataset:
        return

    fit_results = fit_all_spectra(dataset)
    if not fit_results:
        print("No se generaron resultados de ajuste.")
        return

    plt.figure(figsize=(10, 6))
    
    table_rows = []
    
    for idx in sorted(dataset.keys()):
        lam = dataset[idx]["wavelength"]
        pmt = dataset[idx]["pmt"]
        std_pmt = dataset[idx]["std_pmt"]
        true_delta_hz = dataset[idx]["true_delta_hz"]
        num_datos = dataset[idx]["num_datos"]
        
        # Etiqueta para la gráfica
        delta_str = format_hz(true_delta_hz)
        label_str = r"$\Delta = $" + delta_str
        
        # Graficar datos
        p = plt.errorbar(lam, pmt, yerr=std_pmt, marker='.', linestyle='', alpha=0.4, label=label_str, capsize=2)
        
        if idx in fit_results:
            popt = fit_results[idx]["popt"]
            
            # --- CORRECCIÓN DEL DESEMPAQUETADO ---
            # La función en analysis.py es: (nu, a, b, nu_L_center, gamma_D)
            # a = offset, b = altura, nu_L_center = posición, gamma_D = desviación (ancho Doppler)
            offset = popt[0]
            altura = popt[1]
            posicion = popt[2]
            desviacion = popt[3]
            
            nu_min = C_LIGHT / (lam.max() * 1e3)
            nu_max = C_LIGHT / (lam.min() * 1e3)
            nu_smooth = np.linspace(nu_min, nu_max, 500)
            
            lam_smooth = C_LIGHT / (nu_smooth * 1e3)
            
            plt.plot(lam_smooth, gaussian_tpa_freq(nu_smooth, *popt), color=p[0].get_color(), linewidth=2)
            
            # Guardamos la fila para la tabla LaTeX usando los valores correctos
            table_rows.append(f"{delta_str} & {desviacion:.3e} & {posicion:.3e} & {altura:.2f} & {num_datos} \\\\")

    plt.title("Espectro de Fluorescencia TPA Cesio vs Desintonía", fontsize=TITLE_SIZE)
    plt.xlabel(r"Longitud de Onda del Láser $\lambda$ (nm)", fontsize=LABEL_SIZE)
    plt.ylabel("Conteos PMT (Promedio)", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE, loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("./figures/espectros_delta.png", dpi=300)
    
    # Generación de la tabla LaTeX
    tabla_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Parámetros del ajuste gaussiano y datos del barrido para cada desintonía experimentada.}}
\\label{{tab:ajuste_deltas}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\hline
Desintonía $\\Delta$ & Desv. Estándar $\\sigma$ & Posición del Pico $\\mu$ & Altura del Pico & N$^\\circ$ Datos \\\\
\\hline
{"\n".join(table_rows)}
\\hline
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open("tabla_deltas.tex", "w", encoding="utf-8") as f:
        f.write(tabla_latex)

    print("Procesamiento completado.")
    print("Gráfica guardada como: ./figures/espectros_delta.png")
    print("Tabla LaTeX guardada como: tabla_deltas.tex")

if __name__ == "__main__":
    main()