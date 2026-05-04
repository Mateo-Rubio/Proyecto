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
# Widening the plot to ensure we see the complex multi-resonance structure
start_plot = min(0, 2 * factor_Delta1) - 50
end_plot = max(0, 2 * factor_Delta1) + 50

factors_Delta3 = np.linspace(start_plot, end_plot, 5000) 
Delta3 = factors_Delta3 * Gamma_ba

# 3. Calculate rho_cb (Third Order)
# Prefix multiplier
prefix = -(np.abs(mu_ba)**2 * mu_cb) / ((Delta2 - 1j * Gamma_ca) * h_bar**3)

# Common Denominators (to keep the code clean and prevent typos)
D_A = (Delta1 - 1j * Gamma_ba)
D_B = (2 * Delta1 - Delta3 - 1j * Gamma_ba)
D_C = (Delta3 - 1j * Gamma_ba)

D_D = (Delta2 - Delta1 - 1j * Gamma_cb)
D_E = (Delta2 + Delta3 - 2 * Delta1 - 1j * Gamma_cb)
D_F = (Delta2 - Delta3 - 1j * Gamma_cb)

# The 9 Terms inside the bracket
term1 = (np.abs(E[0])**2 * E[0] * e[0]) / (D_A * D_D)
term2 = (E[0]**2 * np.conj(E[1]) * e[2]) / (D_A * D_E)
term3 = (E[0]**2 * np.conj(E[2]) * e[1]) / (D_A * D_F)

term4 = (E[1] * E[2] * np.conj(E[0]) * e[0]) / (D_B * D_D)
term5 = (np.abs(E[1])**2 * E[2] * e[2]) / (D_B * D_E)
term6 = (np.abs(E[2])**2 * E[1] * e[1]) / (D_B * D_F)

term7 = (E[2] * E[1] * np.conj(E[0]) * e[0]) / (D_C * D_D)
term8 = (np.abs(E[1])**2 * E[2] * e[2]) / (D_C * D_E)
term9 = (np.abs(E[2])**2 * E[1] * e[1]) / (D_C * D_F)

# Summing it all up
rho_cb = prefix * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(factors_Delta3, np.real(rho_cb), label=r'Real($\rho_{cb}^{(3)}$)', color='darkorange')
ax.plot(factors_Delta3, np.imag(rho_cb), label=r'Imag($\rho_{cb}^{(3)}$)', color='teal', linestyle='--')

ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')
ax.set_ylabel(r'Amplitude (3rd Order Nonlinearity)')
ax.set_title(r'Coherencia $\rho_{cb}^{(3)}$ (Third Order FWM Signal Source)')

# Add vertical lines to show where the complex resonances occur
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 0$') 
ax.axvline(2 * factor_Delta1, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 2\Delta_1$') 

# Note: Because Delta2 = 0 in our physics_constants, some of the D_E and D_F resonances 
# will perfectly overlap with the D_B and D_C resonances!

ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()