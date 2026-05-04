import numpy as np
import matplotlib.pyplot as plt
from physics_constants import *

# 1. Time and Space (Isolating the steady-state envelope)
t = 0 
z = 0 

# Complex field components
E = E0 * np.exp(1j * k * z)
e = np.exp(-1j * om * t)

# 2. X-Axis Detuning Array (Delta_3)
# Expand the view to encompass both 0 and 2*Delta1 resonance spikes
start_plot = min(0, 2 * factor_Delta1) - 50
end_plot = max(0, 2 * factor_Delta1) + 50

factors_Delta3 = np.linspace(start_plot, end_plot, 5000) 
Delta3 = factors_Delta3 * Gamma_ba

# 3. Calculate rho_ba (First Order)
term1 = (E[0] * e[0]) / (Delta1 - 1j * Gamma_ba)
term2 = (E[1] * e[1]) / (2 * Delta1 - Delta3 - 1j * Gamma_ba)
term3 = (E[2] * e[2]) / (Delta3 - 1j * Gamma_ba)

rho_ba = (mu_ba / h_bar) * (term1 + term2 + term3)

# 4. Plotting
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(factors_Delta3, np.real(rho_ba), label=r'Real($\rho_{ba}^{(1)}$)', color='blue')
ax.plot(factors_Delta3, np.imag(rho_ba), label=r'Imag($\rho_{ba}^{(1)}$)', color='red', linestyle='--')

ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')
ax.set_ylabel(r'Amplitude')
ax.set_title(r'Coherencia $\rho_{ba}^{(1)}$ (First Order)')

ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 0$') 
ax.axvline(2 * factor_Delta1, color='gray', linestyle=':', alpha=0.5, label=r'Resonance: $\Delta_3 = 2\Delta_1$') 

ax.legend()
ax.grid(True, alpha=0.3)

plt.show()