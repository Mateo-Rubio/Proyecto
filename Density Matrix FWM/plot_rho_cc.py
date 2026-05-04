import numpy as np
import matplotlib.pyplot as plt
from physics_constants import *

# 1. Time and Space
z = 0 
# (Time 't' is not needed here because the diagonal population elements 
# are stationary in the steady state, the exponential terms cancel out!)

# Complex field components
E = E0 * np.exp(1j * k * z)

# 2. X-Axis Detuning Array (Delta_3)
start_plot = min(0, 2 * factor_Delta1) - 50
end_plot = max(0, 2 * factor_Delta1) + 50

factors_Delta3 = np.linspace(start_plot, end_plot, 5000) 
Delta3 = factors_Delta3 * Gamma_ba

# 3. Calculate rho_cc (Fourth Order)
# Real Prefix multiplier
prefix = -(2 * np.abs(mu_ba)**2 * np.abs(mu_cb)**2) / (h_bar**4 * gamma_c)

# Pre-factor Denominator
D_pre = (Delta2 - 1j * Gamma_ca)

# Common Denominators inside the brackets
D_A = (Delta1 - 1j * Gamma_ba)
D_B = (2 * Delta1 - Delta3 - 1j * Gamma_ba)
D_C = (Delta3 - 1j * Gamma_ba)

D_D = (Delta2 - Delta1 - 1j * Gamma_cb)
D_E = (Delta2 + Delta3 - 2 * Delta1 - 1j * Gamma_cb)
D_F = (Delta2 - Delta3 - 1j * Gamma_cb)

# The 4 main brackets of the equation
term1 = (np.abs(E[0])**4) / (D_A * D_D)

term2 = (E[0]**2 * np.conj(E[1]) * np.conj(E[2])) * ( (1 / (D_A * D_E)) + (1 / (D_A * D_F)) )

term3 = (np.conj(E[0])**2 * E[1] * E[2]) * ( (1 / (D_B * D_D)) + (1 / (D_C * D_D)) )

term4 = (np.abs(E[1])**2 * np.abs(E[2])**2) * ( 
    (1 / (D_B * D_E)) + 
    (1 / (D_B * D_F)) + 
    (1 / (D_C * D_E)) + 
    (1 / (D_C * D_F)) 
)

# Combine, apply the 1/D_pre factor, and take the Imaginary part
bracket_sum = term1 + term2 + term3 + term4
complex_core = (1 / D_pre) * bracket_sum

# The final population is purely real!
rho_cc_4th = prefix * np.imag(complex_core)

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# We only plot one line because populations are strictly real numbers
ax.plot(factors_Delta3, rho_cc_4th, label=r'$\rho_{cc}^{(4)}$ (Population)', color='purple', linewidth=2)

ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')
ax.set_ylabel(r'Fractional Population')
ax.set_title(r'Población del Nivel Superior $\rho_{cc}^{(4)}$ (Cuarto Orden)')

# Add vertical lines to show where the resonances occur
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 0$') 
ax.axvline(2 * factor_Delta1, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 2\Delta_1$') 

ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()