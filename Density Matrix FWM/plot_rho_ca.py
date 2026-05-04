import numpy as np
import matplotlib.pyplot as plt
from physics_constants import *

# 1. Time and Space
t = 0 
z = 0 

# Complex field components
E = E0 * np.exp(1j * k * z)

# 2. X-Axis Detuning Array (Delta_3)
start_plot = min(0, 2 * factor_Delta1) - 50
end_plot = max(0, 2 * factor_Delta1) + 50

factors_Delta3 = np.linspace(start_plot, end_plot, 5000) 
Delta3 = factors_Delta3 * Gamma_ba

# 3. Calculate rho_ca (Second Order)
# Denominator constants
den_term2 = (Delta2 - 1j * Gamma_ca)

# Term 1: E1^2 / ((Delta1 - i*Gamma_ba)*(Delta2 - i*Gamma_ca))
num1 = (E[0]**2) * np.exp(-1j * 2 * om[0] * t)
den1 = (Delta1 - 1j * Gamma_ba) * den_term2
term1 = num1 / den1

# Term 2: E2*E3 / ((2*Delta1 - Delta3 - i*Gamma_ba)*(Delta2 - i*Gamma_ca))
num2 = (E[1] * E[2]) * np.exp(-1j * (om[1] + om[2]) * t)
den2 = (2 * Delta1 - Delta3 - 1j * Gamma_ba) * den_term2
term2 = num2 / den2

# Term 3: E2*E3 / ((Delta3 - i*Gamma_ba)*(Delta2 - i*Gamma_ca))
num3 = (E[1] * E[2]) * np.exp(-1j * (om[1] + om[2]) * t)
den3 = (Delta3 - 1j * Gamma_ba) * den_term2
term3 = num3 / den3

# Total density matrix element
rho_ca = ((mu_cb * mu_ba) / (h_bar**2)) * (term1 + term2 + term3)

# 4. Plotting
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(factors_Delta3, np.real(rho_ca), label=r'Real($\rho_{ca}^{(2)}$)', color='green')
ax.plot(factors_Delta3, np.imag(rho_ca), label=r'Imag($\rho_{ca}^{(2)}$)', color='purple', linestyle='--')

ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')
ax.set_ylabel(r'Amplitude')
ax.set_title(r'Coherencia $\rho_{ca}^{(2)}$ (Second Order Two-Photon Resonance)')

ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 0$') 
ax.axvline(2 * factor_Delta1, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 2\Delta_1$') 

ax.legend()
ax.grid(True, alpha=0.3)

plt.show()