import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from physics_constants import CesiumFWMSystem

# 1. Configurar el sistema con tu celda de 71.8 mm
L_cell = 0.0718  # Metros
fwm_sys = CesiumFWMSystem()

# 2. Definir el eje z para la evaluación (500 puntos a lo largo de la celda)
z_eval = np.linspace(0, L_cell, 500)

# 3. Resolver la propagación completa con pasos controlados (max_step=100 um)
sol = solve_ivp(fwm_sys.coupled_polar_svea, (0, L_cell), fwm_sys.y0_polar, 
                t_eval=z_eval, rtol=1e-6, atol=1e-9, max_step=1e-4)

# Verificamos si la integración fue exitosa
if not sol.success:
    print(f"Advertencia del integrador: {sol.message}")

A1_z, A2_z, A3_z, theta_z = sol.y

# Convertir amplitudes a Intensidad (W/cm^2) para la gráfica
c = 299792458
eps0 = 8.85418782e-12
def A_to_I(A): return (0.5 * c * eps0 * A**2) / 1e4

I1_plot = A_to_I(A1_z)
I2_plot = A_to_I(A2_z)
I3_plot = A_to_I(A3_z)

# 4. Generar la Gráfica de Evolución Espacial
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Intensidades
ax1.plot(z_eval * 1e3, I1_plot, 'k', lw=2, label=r'Pump $S_1$ (822nm)')
ax1.plot(z_eval * 1e3, I3_plot, 'g', lw=2, label=r'Seed $S_3$ (852nm)')
ax1.plot(z_eval * 1e3, I2_plot, 'r', lw=2, label=r'Generated $S_2$ (794nm)')
ax1.set_ylabel(r'Intensity ($W/cm^2$)')
ax1.set_title(f'Propagación en Celda de Cesio ({L_cell*1e3} mm)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Fase Colectiva (theta)
theta_wrapped = (theta_z + np.pi) % (2.0 * np.pi) - np.pi
ax2.plot(z_eval * 1e3, theta_wrapped / np.pi, 'b', lw=2)

# CORRECCIÓN DE SINTAXIS: Se agregó la 'r' antes de la cadena
ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label=r'Destructive Limit ($\pi$)')
ax2.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)

ax2.set_xlabel('Posición en la celda z (mm)')
ax2.set_ylabel(r'Fase Colectiva $\theta / \pi$')
ax2.set_ylim(-1.1, 1.1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()