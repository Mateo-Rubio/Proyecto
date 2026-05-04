import numpy as np
import matplotlib.pyplot as plt
from physics_constants import *

# 1. Time and Space
t = 0 
z = 0 

# Complex field components
E = E0 * np.exp(1j * k * z)
e = np.exp(-1j * om * t)

# 2. X-Axis Detuning Array (Delta_3)
start_plot = min(0, 2 * factor_Delta1) - 50
end_plot = max(0, 2 * factor_Delta1) + 50

factors_Delta3 = np.linspace(start_plot, end_plot, 5000) 
Delta3 = factors_Delta3 * Gamma_ba

# 3. Calculate rho_ba (Third Order)
# Prefix multiplier
prefix = (np.abs(mu_cb)**2 * mu_ba) / ((Delta2 - 1j * Gamma_ca) * h_bar**3)

# Common Denominators
D_A = (Delta1 - 1j * Gamma_ba)
D_B = (2 * Delta1 - Delta3 - 1j * Gamma_ba)
D_C = (Delta3 - 1j * Gamma_ba)

# The final term specifically asks for Gamma_cb instead of Gamma_ba
D_G = (2 * Delta1 - Delta3 - 1j * Gamma_cb)

# The 9 Terms inside the bracket
term1 = (np.abs(E[0])**2 * E[0] * e[0]) / (D_A**2)
term2 = (E[0]**2 * np.conj(E[1]) * e[2]) / (D_A * D_C)
term3 = (E[0]**2 * np.conj(E[2]) * e[1]) / (D_A * D_B)

term4 = (E[1] * E[2] * np.conj(E[0]) * e[0]) / (D_B * D_A)
term5 = (np.abs(E[1])**2 * E[2] * e[2]) / (D_B * D_C)
term6 = (np.abs(E[2])**2 * E[1] * e[1]) / (D_B**2)

term7 = (E[2] * E[1] * np.conj(E[0]) * e[0]) / (D_C * D_A)
term8 = (np.abs(E[1])**2 * E[2] * e[2]) / (D_C**2)
term9 = (np.abs(E[2])**2 * E[1] * e[1]) / (D_C * D_G)

# Summing it all up
rho_ba_3rd = prefix * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# We use different colors here to distinguish it from the 1st order plot
ax.plot(factors_Delta3, np.real(rho_ba_3rd), label=r'Real($\rho_{ba}^{(3)}$)', color='crimson')
ax.plot(factors_Delta3, np.imag(rho_ba_3rd), label=r'Imag($\rho_{ba}^{(3)}$)', color='navy', linestyle='--')

ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')
ax.set_ylabel(r'Amplitude (3rd Order Correction)')
ax.set_title(r'Coherencia $\rho_{ba}^{(3)}$ (Third Order Nonlinearity on Lower Leg)')

# Add vertical lines to show where the resonances occur
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 0$') 
ax.axvline(2 * factor_Delta1, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 2\Delta_1$') 

ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()