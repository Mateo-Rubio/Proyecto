import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from physics_constants import CesiumFWMSystem

# 1. Instanciar el sistema físico global para tu celda de 71.8 mm
L_cell = 0.0718  # Metros
fwm_sys = CesiumFWMSystem()

# 2. Definir el eje z espacial (500 puntos a lo largo de la celda)
z_eval = np.linspace(0, L_cell, 500)

# Factores de conversión de Amplitud a Intensidad (W/cm^2)
c = 299792458.0
eps0 = 8.85418782e-12

def A_to_I(A): 
    return (0.5 * c * eps0 * A**2) / 1e4

# =========================================================================
# 3. Inicialización de la Figura con 3 Paneles Independientes
# =========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
plt.subplots_adjust(bottom=0.25, left=0.12, right=0.95, hspace=0.2)

# Panel 1: Haces de inyección (Bombeo y Semilla)
line_I1, = ax1.plot([], [], 'k', lw=2, label=r'Pump $S_1$ (822nm)')
line_I3, = ax1.plot([], [], 'g--', lw=1.5, label=r'Seed $S_3$ (852nm)')
ax1.set_ylabel(r'$I_{\text{inc}}$ ($W/cm^2$)')
ax1.set_title(f'Evolución Espacial Interactiva en Celda de Cesio ({L_cell*1e3:.1f} mm)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: EXCLUSIVO para la Onda Generada (Permite evaluar señales débiles)
line_I2, = ax2.plot([], [], 'r', lw=2.5, label=r'Generated $S_2$ (794nm)')
ax2.set_ylabel(r'$I_{\text{gen}}$ ($W/cm^2$)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Fase Colectiva theta
line_theta, = ax3.plot([], [], 'b', lw=2, label=r'Fase Colectiva $\theta$')
ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label=r'Límite Destructivo ($\pm\pi$)')
ax3.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Posición en la celda z (mm)')
ax3.set_ylabel(r'Fase $\theta / \pi$')
ax3.set_ylim(-1.1, 1.1)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# =========================================================================
# 4. Controles Deslizantes (Sliders) para Detunings Normalizados
# =========================================================================
ax_d2 = plt.axes([0.15, 0.12, 0.75, 0.03])
ax_d3 = plt.axes([0.15, 0.05, 0.75, 0.03])

init_d2_norm = 0.0
init_d3_norm = fwm_sys.Delta1 / fwm_sys.Gamma_ba

slider_d2 = Slider(ax_d2, r'$\Delta_2 / \Gamma_{ba}$', -50.0, 50.0, valinit=init_d2_norm)
slider_d3 = Slider(ax_d3, r'$\Delta_3 / \Gamma_{ba}$', -100.0, 100.0, valinit=init_d3_norm)

# =========================================================================
# 5. Función de Actualización con Escalas Independientes
# =========================================================================
def update(val):
    fwm_sys.Delta2 = slider_d2.val * fwm_sys.Gamma_ba
    fwm_sys.Delta3 = slider_d3.val * fwm_sys.Gamma_ba
    
    # Integración BDF estable usando las variables internas escaladas
    E_scale = 1e5
    y0_scaled = [fwm_sys.A1_0 / E_scale, 1e-12 / E_scale, fwm_sys.A3_0 / E_scale, 0.0]
    
    sol = solve_ivp(fwm_sys.coupled_polar_svea, (0.0, L_cell), y0_scaled, 
                    method='BDF', t_eval=z_eval, rtol=1e-5, atol=1e-8)
    
    if sol.success:
        A1_z = sol.y[0] * E_scale
        A2_z = sol.y[1] * E_scale
        A3_z = sol.y[2] * E_scale
        theta_z = sol.y[3]
        
        I1 = A_to_I(A1_z)
        I2 = A_to_I(A2_z)
        I3 = A_to_I(A3_z)
        
        theta_wrapped = (theta_z + np.pi) % (2.0 * np.pi) - np.pi
        
        z_mm = z_eval * 1e3
        
        # Actualizar datos en cada uno de los 3 paneles
        line_I1.set_data(z_mm, I1)
        line_I3.set_data(z_mm, I3)
        line_I2.set_data(z_mm, I2)
        line_theta.set_data(z_mm, theta_wrapped / np.pi)
        
        # Auto-escalado independiente para inyección (ax1)
        ax1.set_xlim(0, L_cell * 1e3)
        max_inc = max(np.max(I1), np.max(I3))
        ax1.set_ylim(0, max_inc * 1.05 if max_inc > 0 else 1.0)
        
        # Auto-escalado independiente y preciso para la onda generada (ax2)
        # Permite ver perfectamente crecimientos débiles sin importar la escala de I1
        ax2.set_xlim(0, L_cell * 1e3)
        max_gen = np.max(I2)
        ax2.set_ylim(0, max_gen * 1.05 if max_gen > 1e-15 else 1e-12)
        
        ax3.set_xlim(0, L_cell * 1e3)
        
        fig.canvas.draw_idle()
    else:
        print(f"Advertencia del Solver: {sol.message}")

slider_d2.on_changed(update)
slider_d3.on_changed(update)

update(None)

plt.show()