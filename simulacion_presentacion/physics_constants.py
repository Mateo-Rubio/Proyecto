import numpy as np
import scipy.constants as const

# ==========================================
# 1. FUNDAMENTAL CONSTANTS
# ==========================================
h_bar = const.hbar  # 1.0545718e-34 J*s
c = const.c         # 299792458 m/s
au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
# ==========================================
# 2. CESIUM ATOMIC PARAMETERS (a=6S_1/2, b=6P_3/2, c=8S_1/2)
# ==========================================
# Gamma Decay Rates (Angular Frequency)
Gamma_ba = 2 * np.pi * 5.2227e6  # ~3.28e7 rad/s (Natural decay of 6P_3/2)
Gamma_ca = 2 * np.pi * 1.0e6     # Approximation for 8S_1/2 two-photon coherence decay
Gamma_cb = 2 * np.pi * 3.71e6  # ~2.33e7 rad/s

mu_ba_au = 3.4707  # Value for 6S_1/2 -> 6P_3/2
mu_cb_au = 0.7394  # Value for 6P_3/2 -> 8S_1/2

mu_ba = mu_ba_au * au_to_Cm
mu_cb = mu_cb_au * au_to_Cm

# gamma_c: Population decay rate (inverse lifetime) of the 8S_1/2 state
gamma_c = 2 * np.pi * 2.2e6  # ~1.38e7 rad/s

# ==========================================
# 3. LASER FIELDS AND WAVEVECTORS
# ==========================================
# Assuming a weak laser intensity of ~1 mW/cm^2
E_field = 86.8  # V/m
E0 = np.array([E_field, E_field, E_field])

# Wavelengths: [Laser 1, Laser 2, Laser 3]
lmb = np.array([852.35e-9, 822.48e-9, 794.39e-9])
k = 2 * np.pi / lmb 
om = k * c

# ==========================================
# 4. DETUNINGS
# ==========================================
omega_ba = 2 * np.pi * c / 852.35e-9
omega_laser1 = 2 * np.pi * c / 822.48e-9

# One-photon detuning (Delta_1)
Delta1 = omega_ba - omega_laser1
factor_Delta1 = Delta1 / Gamma_ba
print(factor_Delta1)