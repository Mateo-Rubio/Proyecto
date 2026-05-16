import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. IMPORTACIÓN DE CONSTANTES FÍSICAS
# ==============================================================================
try:
    from physics_constants import *
except ImportError:
    import scipy.constants as const
    print("Advertencia: No se encontró 'physics_constants.py'. Usando fallback interno.")
    h_bar = const.hbar
    c = const.c
    au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
    
    # Tasas de decaimiento del Cesio
    Gamma_ba = 2 * np.pi * 5.2227e6
    Gamma_ca = 2 * np.pi * 1.0e6
    Gamma_cb = 2 * np.pi * 3.71e6
    gamma_c  = 2 * np.pi * 2.2e6
    
    # Momentos dipolares
    mu_ba = 3.4707 * au_to_Cm
    mu_cb = 0.7394 * au_to_Cm
    
    # Campos y longitudes de onda
    E_field = 86.8
    E0 = np.array([E_field, E_field, E_field])
    lmb = np.array([852.35e-9, 822.48e-9, 794.39e-9])
    k = 2 * np.pi / lmb 
    om = k * c
    
    # Cálculo de la desintonía de un fotón base
    omega_ba = 2 * np.pi * c / 852.35e-9
    omega_laser1 = 2 * np.pi * c / 822.48e-9
    Delta1 = omega_ba - omega_laser1
    factor_Delta1 = Delta1 / Gamma_ba

# ==============================================================================
# 2. FUNCIONES DEL MODELO DE MATRIZ DE DENSIDAD
# ==============================================================================
def calc_density_matrix(D1_factor, D2_factor, d3_array, t, z):
    D1 = D1_factor * Gamma_ba
    D2 = D2_factor * Gamma_ba
    D3 = d3_array * Gamma_ba
    
    # Propagación espaciotemporal
    E_z = E0 * np.exp(1j * k * z)
    e_t = np.exp(-1j * om * t)
    
    # Denominadores resonantes
    D_A = (D1 - 1j * Gamma_ba)
    D_B = (2 * D1 - D3 - 1j * Gamma_ba)
    D_C = (D3 - 1j * Gamma_ba)
    D_D = (D2 - D1 - 1j * Gamma_cb)
    D_E = (D2 + D3 - 2 * D1 - 1j * Gamma_cb)
    D_F = (D2 - D3 - 1j * Gamma_cb)
    D_G = (2 * D1 - D3 - 1j * Gamma_cb)
    D_pre_2 = (D2 - 1j * Gamma_ca)
    
    # Coherencia de Tercer Orden (FWM principal)
    prefix_3 = -(np.abs(mu_ba)**2 * mu_cb) / (D_pre_2 * h_bar**3)
    t1_3 = (np.abs(E_z[0])**2 * E_z[0] * e_t[0]) / (D_A * D_D)
    t2_3 = (E_z[0]**2 * np.conj(E_z[1]) * e_t[2]) / (D_A * D_E)
    t3_3 = (E_z[0]**2 * np.conj(E_z[2]) * e_t[1]) / (D_A * D_F)
    t4_3 = (E_z[1] * E_z[2] * np.conj(E_z[0]) * e_t[0]) / (D_B * D_D)
    t5_3 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_B * D_E)
    t6_3 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_B * D_F)
    t7_3 = (E_z[2] * E_z[1] * np.conj(E_z[0]) * e_t[0]) / (D_C * D_D)
    t8_3 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_C * D_E)
    t9_3 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_C * D_F)
    rho_cb_3 = prefix_3 * (t1_3 + t2_3 + t3_3 + t4_3 + t5_3 + t6_3 + t7_3 + t8_3 + t9_3)
    
    # Población del Nivel Superior (Cuarto Orden)
    prefix_4 = -(2 * np.abs(mu_ba)**2 * np.abs(mu_cb)**2) / (h_bar**4 * gamma_c)
    b1 = (np.abs(E_z[0])**4) / (D_A * D_D)
    b2 = (E_z[0]**2 * np.conj(E_z[1]) * np.conj(E_z[2])) * (1/(D_A * D_E) + 1/(D_A * D_F))
    b3 = (np.conj(E_z[0])**2 * E_z[1] * E_z[2]) * (1/(D_B * D_D) + 1/(D_C * D_D))
    b4 = (np.abs(E_z[1])**2 * np.abs(E_z[2])**2) * (1/(D_B * D_E) + 1/(D_B * D_F) + 1/(D_C * D_E) + 1/(D_C * D_F))
    rho_cc_4 = prefix_4 * np.imag((1/D_pre_2) * (b1 + b2 + b3 + b4))
    
    shape_enforcer = np.ones_like(d3_array, dtype=np.complex128)
    return rho_cb_3 * shape_enforcer, np.real(rho_cc_4 * shape_enforcer)

# ==============================================================================
# 3. CONFIGURACIÓN ESTILÍSTICA Y GRÁFICOS HORIZONTALES (1x4)
# ==============================================================================
SCALE_FACTOR = 1.3  # Escala global para aumentar el tamaño de todos los elementos

base_title  = 14 * SCALE_FACTOR
base_label  = 14 * SCALE_FACTOR
base_ticks  = 12 * SCALE_FACTOR
base_legend = 12 * SCALE_FACTOR
base_box    = 12 * SCALE_FACTOR

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11 * SCALE_FACTOR,
    'axes.labelsize': base_label,
    'axes.titlesize': base_title,
    'xtick.labelsize': base_ticks,
    'ytick.labelsize': base_ticks
})

# Disposición en una sola fila horizontal
fig, axs = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
plt.subplots_adjust(wspace=0.38)

# Escenarios con descripciones simplificadas centradas en Delta_2
scenarios = [
    {
        "d1": factor_Delta1, "d2": 0.0,
        "t": 0.0, "z": 0.0,
        "grid": (0.0, 40.0),
        "title": "Resonancia de Dos Fotones",
        "desc": r"$\Delta_2 = 0$"
    },
    {
        "d1": factor_Delta1, "d2": 0.0,
        "t": 1.5e-15, "z": 1.2e-6,
        "grid": (0.0, 40.0),
        "title": "Modulación Espaciotemporal",
        "desc": r"$\Delta_2 = 0$" + "\n" + r"$t=1.5\,\text{fs}$" + "\n" + r"$z=1.2\,\mu\text{m}$"
    },
    {
        "d1": factor_Delta1 * 1.5, "d2": 0.0,
        "t": 0.0, "z": 0.0,
        "grid": (40.0, 60.0),
        "title": "Bombeo Altamente Desintonizado",
        "desc": r"$\Delta_2 = 0$"
    },
]

c_real = "#292327"       
c_imag = "#B2C6D5"       
c_pop  = "#292327"       

for idx, sc in enumerate(scenarios):
    ax1 = axs[idx]
    
    d1_val = sc["d1"]
    d2_val = sc["d2"]
    t_val  = sc["t"]
    z_val  = sc["z"]
    
    x_center, x_width = sc["grid"]
    x_arr = np.linspace(x_center - x_width/2, x_center + x_width/2, 2000)
    
    rho_cb, rho_cc = calc_density_matrix(d1_val, d2_val, x_arr, t_val, z_val)
    
    # EJE IZQUIERDO: Coherencia FWM
    l1 = ax1.plot(x_arr, np.real(rho_cb), color=c_real, lw=2.5, label=r"$\text{Re}[\rho_{cb}^{(3)}]$")
    l2 = ax1.plot(x_arr, np.imag(rho_cb), color=c_imag, lw=2.5, linestyle="--", label=r"$\text{Im}[\rho_{cb}^{(3)}]$")
    
    ax1.set_title(sc["title"], fontweight="bold", fontsize=base_title, pad=16)
    ax1.set_xlabel(r"Desintonía normalizada ($\Delta_3 / \Gamma_{ba}$)", fontsize=base_label, labelpad=8)
    ax1.set_ylabel(r"$\rho_{cb}^{(3)}$", color=c_real, fontweight="bold", fontsize=base_label)
    ax1.tick_params(axis='both', which='major', labelsize=base_ticks)
    ax1.tick_params(axis='y', labelcolor=c_real)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(base_ticks)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # EJE DERECHO: Población superior
    ax2 = ax1.twinx()
    l3 = ax2.plot(x_arr, rho_cc, color=c_pop, lw=2.0, linestyle="-.", alpha=0.85, label=r"$\rho_{cc}^{(4)}$")
    
    ax2.set_ylabel(r"$\rho_{cc}^{(4)}$", color=c_pop, fontweight="bold", fontsize=base_label)
    ax2.tick_params(axis='y', which='major', labelsize=base_ticks, labelcolor=c_pop)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(base_ticks)
    
    # Caja descriptiva depurada
    ax1.text(0.6, 0.88, sc["desc"], 
            transform=ax1.transAxes, 
            fontsize=base_box, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=c_real, lw=1.5, alpha=0.95))
    
    # Leyenda unificada únicamente en el primer panel
    if idx == 0:
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=base_legend)

plt.savefig("fixed_fwm_scenarios_horizontal.png", format="png", dpi=300, bbox_inches="tight")
print("¡Gráfico horizontal generado exitosamente!")
plt.show()