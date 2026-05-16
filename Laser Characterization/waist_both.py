import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# VARIABLES GLOBALES PARA CONTROL DE ESTILOS
# ==============================================================================
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 20

def w_beam(z, w_0, z_0, z_R):
    return w_0 * np.sqrt(1 + ((z - z_0) / z_R)**2)

def procesar_laser(carpeta_laser, cortar_al_final=2, n_puntos=20):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, "data4", carpeta_laser, "z*mm.xlsx")
    archivos = glob.glob(search_path)
    
    posiciones = []
    dataframes = {}
    
    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo)
        match = re.search(r"z(\d+(\.\d+)?)mm\.xlsx", nombre_archivo)
        if match:
            posicion = float(match.group(1))
            if(posicion > 14):
                posiciones.append(posicion)
                
                df = pd.read_excel(archivo, skiprows=21)
                df = df.drop(0).reset_index(drop=True)
                
                # Se añaden Pos.X y Pos.Y a la limpieza de formato
                for col in ["W Width I", "V Width I", "Pos.X", "Pos.Y"]:
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                    
                dataframes[posicion] = df.head(n_puntos)
            
    posiciones.sort()
    
    if cortar_al_final > 0 and len(posiciones) > cortar_al_final:
        lista_posiciones = posiciones[:-cortar_al_final]
    else:
        lista_posiciones = posiciones
        
    n_pos = len(lista_posiciones)
    w_mean = np.zeros(n_pos)
    v_mean = np.zeros(n_pos)
    w_std = np.zeros(n_pos)
    v_std = np.zeros(n_pos)
    
    x_mean = np.zeros(n_pos)
    y_mean = np.zeros(n_pos)
    x_std = np.zeros(n_pos)
    y_std = np.zeros(n_pos)
    
    for i, pos in enumerate(lista_posiciones):
        w_data = dataframes[pos]["W Width I"].to_numpy() / 2.0 * 1e-6
        v_data = dataframes[pos]["V Width I"].to_numpy() / 2.0 * 1e-6
        
        # Extracción de centroides (asumimos vienen en micras, los pasamos a SI)
        x_data = dataframes[pos]["Pos.X"].to_numpy() * 1e-6
        y_data = dataframes[pos]["Pos.Y"].to_numpy() * 1e-6
        
        w_mean[i] = w_data.mean()
        v_mean[i] = v_data.mean()
        w_std[i] = w_data.std()
        v_std[i] = v_data.std()
        
        x_mean[i] = x_data.mean()
        y_mean[i] = y_data.mean()
        x_std[i] = x_data.std()
        y_std[i] = y_data.std()
        
    z = np.array(lista_posiciones) * 1e-3
    
    limites = ([0.0, -np.inf, 1e-5], [np.inf, np.inf, np.inf])
    
    p0_w = [w_mean.min(), z[np.argmin(w_mean)], 0.01]
    p0_v = [v_mean.min(), z[np.argmin(v_mean)], 0.01]
    
    params_w, cov_w = curve_fit(w_beam, z, w_mean, p0=p0_w, bounds=limites)
    params_v, cov_v = curve_fit(w_beam, z, v_mean, p0=p0_v, bounds=limites)
    
    err_w = np.sqrt(np.diag(cov_w))
    err_v = np.sqrt(np.diag(cov_v))
    
    z_lins = np.linspace(z.min(), z.max(), 200) if n_pos > 0 else np.array([])
    w_fit = w_beam(z_lins, *params_w) if n_pos > 0 else np.array([])
    v_fit = w_beam(z_lins, *params_v) if n_pos > 0 else np.array([])
    
    return {
        "z_exp": z,
        "w_exp": w_mean,
        "v_exp": v_mean,
        "w_std": w_std,
        "v_std": v_std,
        "x_exp": x_mean,
        "y_exp": y_mean,
        "x_std": x_std,
        "y_std": y_std,
        "params_w": params_w,
        "err_w": err_w,
        "params_v": params_v,
        "err_v": err_v,
        "z_lins": z_lins,
        "w_fit": w_fit,
        "v_fit": v_fit
    }

res_852 = procesar_laser("852nm", cortar_al_final=2)
res_822 = procesar_laser("822nm", cortar_al_final=2)

# ==============================================================================
# 1. GRÁFICAS DE ANCHO (CINTURAS W y V)
# ==============================================================================
fig, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(10, 12))

# Configuración y ploteo ax_v
ax_v.errorbar(res_852["z_exp"]*1e3, res_852["v_exp"]*1e6, yerr=res_852["v_std"]*1e6, fmt="o", color="#E6550A", label="852 nm Exp", capsize=3)
ax_v.plot(res_852["z_lins"]*1e3, res_852["v_fit"]*1e6, "--", color="#E6550A", label="852 nm Fit")
ax_v.errorbar(res_822["z_exp"]*1e3, res_822["v_exp"]*1e6, yerr=res_822["v_std"]*1e6, fmt="s", color="#005A70", label="822 nm Exp", capsize=3)
ax_v.plot(res_822["z_lins"]*1e3, res_822["v_fit"]*1e6, "-.", color="#005A70", label="822 nm Fit")

ax_v.set_xlabel("Posicion en el eje z [mm]", fontsize=LABEL_SIZE)
ax_v.set_ylabel("Cintura vertical v(z) [$\\mu$m]", fontsize=LABEL_SIZE)
ax_v.set_title("Ancho Vertical de los Haces (Direccion V)", fontsize=TITLE_SIZE)
ax_v.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax_v.grid(True, linestyle=":", alpha=0.6)
ax_v.legend(fontsize=LEGEND_SIZE)

# Configuración y ploteo ax_w
ax_w.errorbar(res_852["z_exp"]*1e3, res_852["w_exp"]*1e6, yerr=res_852["w_std"]*1e6, fmt="o", color="#E6550A", label="852 nm Exp", capsize=3)
ax_w.plot(res_852["z_lins"]*1e3, res_852["w_fit"]*1e6, "--", color="#E6550A", label="852 nm Fit")
ax_w.errorbar(res_822["z_exp"]*1e3, res_822["w_exp"]*1e6, yerr=res_822["w_std"]*1e6, fmt="s", color="#005A70", label="822 nm Exp", capsize=3)
ax_w.plot(res_822["z_lins"]*1e3, res_822["w_fit"]*1e6, "-.", color="#005A70", label="822 nm Fit")

ax_w.set_xlabel("Posicion en el eje z [mm]", fontsize=LABEL_SIZE)
ax_w.set_ylabel("Cintura horizontal w(z) [$\\mu$m]", fontsize=LABEL_SIZE)
ax_w.set_title("Ancho Horizontal de los Haces (Direccion W)", fontsize=TITLE_SIZE)
ax_w.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax_w.grid(True, linestyle=":", alpha=0.6)
ax_w.legend(fontsize=LEGEND_SIZE)

plt.tight_layout()
plt.savefig("comparacion_haces_852_822.png", dpi=300)
plt.close()

# ==============================================================================
# 2. GRÁFICAS DE EVOLUCIÓN DE CENTROIDES (POSICIONES X y Y)
# ==============================================================================
fig2, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Gráfica Centroide X ----
ax_x.errorbar(res_852["z_exp"]*1e3, res_852["x_exp"]*1e6, yerr=res_852["x_std"]*1e6, fmt="o-", color="#E6550A", label="852 nm Pos X", capsize=3)
ax_x.errorbar(res_822["z_exp"]*1e3, res_822["x_exp"]*1e6, yerr=res_822["x_std"]*1e6, fmt="s-", color="#005A70", label="822 nm Pos X", capsize=3)

# Líneas de cintura vertical (Waist)
ax_x.axvline(res_852["params_v"][1]*1e3, color="#E6550A", linestyle=":", label="Waist V 852 nm")
ax_x.axvline(res_822["params_v"][1]*1e3, color="#005A70", linestyle=":", label="Waist V 822 nm")

ax_x.set_xlabel("Posición en el eje z [mm]", fontsize=LABEL_SIZE)
ax_x.set_ylabel("Posición X [$\\mu$m]", fontsize=LABEL_SIZE)
ax_x.set_title("Evolución del Centroide Horizontal (Pos.X)", fontsize=TITLE_SIZE)
ax_x.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax_x.grid(True, linestyle=":", alpha=0.6)
ax_x.legend(fontsize=LEGEND_SIZE)

# ---- Gráfica Centroide Y ----
ax_y.errorbar(res_852["z_exp"]*1e3, res_852["y_exp"]*1e6, yerr=res_852["y_std"]*1e6, fmt="o-", color="#E6550A", label="852 nm Pos Y", capsize=3)
ax_y.errorbar(res_822["z_exp"]*1e3, res_822["y_exp"]*1e6, yerr=res_822["y_std"]*1e6, fmt="s-", color="#005A70", label="822 nm Pos Y", capsize=3)

# Líneas de cintura vertical (Waist)
ax_y.axvline(res_852["params_v"][1]*1e3, color="#E6550A", linestyle=":", label="Waist V 852 nm")
ax_y.axvline(res_822["params_v"][1]*1e3, color="#005A70", linestyle=":", label="Waist V 822 nm")

ax_y.set_xlabel("Posición en el eje z [mm]", fontsize=LABEL_SIZE)
ax_y.set_ylabel("Posición Y [$\\mu$m]", fontsize=LABEL_SIZE)
ax_y.set_title("Evolución del Centroide Vertical (Pos.Y)", fontsize=TITLE_SIZE)
ax_y.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax_y.grid(True, linestyle=":", alpha=0.6)
ax_y.legend(fontsize=LEGEND_SIZE)

plt.tight_layout()
plt.savefig("evolucion_centroides_852_822.png", dpi=300)
plt.close()

# ==============================================================================
# 3. TABLAS LATEX
# ==============================================================================
def formato_parametro(val, err, factor):
    return f"{val*factor:.2f} \\pm {err*factor:.2f}"

# Tabla Horizontal
tabla_latex_w = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Caracteristicas espaciales de los laseres incidentes en el eje horizontal ($W$).}}
\\label{{tab:caracteristicas_haces_w}}
\\begin{{tabular}}{{lcc}}
\\hline
\\hline
Parámetro & Láser 822 nm & Láser 852 nm \\\\
\\hline
Cintura del haz ($w_0$) [$\\mu$m] & ${formato_parametro(res_822['params_w'][0], res_822['err_w'][0], 1e6)}$ & ${formato_parametro(res_852['params_w'][0], res_852['err_w'][0], 1e6)}$ \\\\
Posición focal ($z_0$) [mm] & ${formato_parametro(res_822['params_w'][1], res_822['err_w'][1], 1e3)}$ & ${formato_parametro(res_852['params_w'][1], res_852['err_w'][1], 1e3)}$ \\\\
Rango de Rayleigh ($z_R$) [mm] & ${formato_parametro(res_822['params_w'][2], res_822['err_w'][2], 1e3)}$ & ${formato_parametro(res_852['params_w'][2], res_852['err_w'][2], 1e3)}$ \\\\
\\hline
\\hline
\\end{{tabular}}
\\end{{table}}
"""

# Tabla Vertical
tabla_latex_v = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Caracteristicas espaciales de los laseres incidentes en el eje vertical ($V$).}}
\\label{{tab:caracteristicas_haces_v}}
\\begin{{tabular}}{{lcc}}
\\hline
\\hline
Parámetro & Láser 822 nm & Láser 852 nm \\\\
\\hline
Cintura del haz ($w_0$) [$\\mu$m] & ${formato_parametro(res_822['params_v'][0], res_822['err_v'][0], 1e6)}$ & ${formato_parametro(res_852['params_v'][0], res_852['err_v'][0], 1e6)}$ \\\\
Posición focal ($z_0$) [mm] & ${formato_parametro(res_822['params_v'][1], res_822['err_v'][1], 1e3)}$ & ${formato_parametro(res_852['params_v'][1], res_852['err_v'][1], 1e3)}$ \\\\
Rango de Rayleigh ($z_R$) [mm] & ${formato_parametro(res_822['params_v'][2], res_822['err_v'][2], 1e3)}$ & ${formato_parametro(res_852['params_v'][2], res_852['err_v'][2], 1e3)}$ \\\\
\\hline
\\hline
\\end{{tabular}}
\\end{{table}}
"""

with open("caracteristicas_haces.tex", "w", encoding="utf-8") as f:
    f.write(tabla_latex_w)
    f.write("\n")
    f.write(tabla_latex_v)

print("Procesamiento completado.")
print("Grafica de anchos guardada como: comparacion_haces_852_822.png")
print("Grafica de centroides guardada como: evolucion_centroides_852_822.png")
print("Tabla LaTeX guardada como: caracteristicas_haces.tex")