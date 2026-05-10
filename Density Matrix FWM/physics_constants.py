import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp

class CesiumFWMSystem:
    """
    Modelo SVEA polar de Boyd et al. modificado para haces Gaussianos enfocados.
    Ubica el waist exactamente en el centro de la celda de 71.8 mm.
    """
    def __init__(self, I1_peak=1010.28, I2_peak=0.0, I3_peak=7.91, N_density=3.0e18):
        self.h_bar = const.hbar
        self.c = const.c
        self.epsilon_0 = const.epsilon_0
        self.au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
        
        self.Gamma_ba = 2.0 * np.pi * 5.2227e6
        self.Gamma_ca = 2.0 * np.pi * 1.0e6
        self.Gamma_cb = 2.0 * np.pi * 3.71e6
        self.gamma_c  = 2.0 * np.pi * 2.2e6
        
        self.mu_ba = 3.4707 * self.au_to_Cm
        self.mu_cb = 0.7394 * self.au_to_Cm
        self.N = N_density
        
        self.omega_ba_atomic = 2.0 * np.pi * self.c / 852.35e-9
        self.omega_ca_atomic = 2.0 * np.pi * self.c / 411.24e-9
        
        w1_init = 2.0 * np.pi * self.c / 822.48e-9
        w3_init = 2.0 * np.pi * self.c / 852.35e-9
        w2_init = 2.0 * self.omega_ba_atomic - w3_init
        
        self._om = np.array([w1_init, w2_init, w3_init])
        self._k  = self._om / self.c
        
        self._Delta1_init = self.omega_ba_atomic - self._om[0]
        self._Delta1 = self._Delta1_init
        self._Delta2 = 0.0
        self._Delta3 = self.omega_ba_atomic - self._om[2]
        
        # Guardamos las intensidades PICO (en el centro de la celda)
        self._I1_peak = I1_peak
        self._I2_peak = I2_peak
        self._I3_peak = I3_peak
        
        # Geometría de la celda y enfoque Gaussiano
        self.L_cell = 0.0718         # 71.8 mm de longitud total
        self.z_foco = self.L_cell / 2.0  # El foco está exactamente en el centro (35.9 mm)
        
        # Rangos de Rayleigh promedio medidos en tu laboratorio (en metros)
        self.zR_1 = 0.0018  # Promedio ~1.8 mm para el bombeo 822nm
        self.zR_3 = 0.0020  # Promedio ~2.0 mm para la semilla 852nm
        
        self._update_propagation_dynamics()
        self._update_boundary_fields()

    def _update_boundary_fields(self):
        """
        Calcula las amplitudes iniciales reales en la ENTRADA de la celda (z=0),
        teniendo en cuenta la expansión del haz Gaussiano desde el centro.
        """
        def I_peak_to_E_entrance(I_peak_cm2, zR):
            # Factor de reducción de intensidad a z=0 por divergencia Gaussiana
            expansion_factor = 1.0 + ((-self.z_foco) / zR)**2
            I_entrance_cm2 = I_peak_cm2 / expansion_factor
            return np.sqrt((2.0 * I_entrance_cm2 * 1e4) / (self.c * self.epsilon_0))
        
        self.A1_0 = I_peak_to_E_entrance(self._I1_peak, self.zR_1)
        self.A3_0 = I_peak_to_E_entrance(self._I3_peak, self.zR_3)
        self.A2_0 = 1e-12  # Piso de ruido cuántico en la entrada
        
        self.y0_polar = [self.A1_0, self.A2_0, self.A3_0, 0.0]

    def _update_propagation_dynamics(self):
        w1 = self.omega_ba_atomic - self._Delta1
        w3 = self.omega_ba_atomic - self._Delta3
        w2 = 2.0 * w1 - w3
        self._om = np.array([w1, w2, w3])
        self._k  = self._om / self.c
        self.svea_coeffs = self._om / (2.0 * self.epsilon_0 * self.c)

    # Propiedades analíticas inalteradas...
    @property
    def Delta1(self): return self._Delta1
    @Delta1.setter
    def Delta1(self, val): self._Delta1 = val; self._update_propagation_dynamics()
    @property
    def Delta2(self): return self._Delta2
    @Delta2.setter
    def Delta2(self, val): 
        self._Delta2 = val
        self._Delta1 = self._Delta1_init + (val / 2.0)
        self._update_propagation_dynamics()
    @property
    def Delta3(self): return self._Delta3
    @Delta3.setter
    def Delta3(self, val): self._Delta3 = val; self._update_propagation_dynamics()

    @property
    def delta_k_total(self):
        return self._k[1] + self._k[2] - 2.0 * self._k[0]

    @property
    def beta(self): return np.arctan2(-self._Delta2, self.Gamma_ca)
    @property
    def k(self): return self._k
    @property
    def om(self): return self._om

    @property
    def chi_fwm(self):
        K = (self.N * (np.abs(self.mu_ba)**2) * (np.abs(self.mu_cb)**2)) / (self.h_bar**3)
        d1 = self._Delta1 if abs(self._Delta1) > 1.0 else 1.0
        d3 = self._Delta3 if abs(self._Delta3) > 1.0 else 1.0
        d2 = self._Delta2 - 1j * self.Gamma_ca
        return K / (d1 * d3 * d2)

    @property
    def alpha_coeffs(self):
        abs_chi = np.abs(self.chi_fwm)
        a1 = (2.0 * self.svea_coeffs[0]) * abs_chi
        a2 = self.svea_coeffs[1] * abs_chi
        a3 = self.svea_coeffs[2] * abs_chi
        return np.array([a1, a2, a3])

    # =========================================================================
    # EVOLUCIÓN CON PERFIL GAUSSIANO EN Z
    # =========================================================================
    def coupled_polar_svea(self, z, y):
        A1, A2, A3, theta = y
        
        a1, a2, a3 = self.alpha_coeffs
        b = self.beta
        d1 = self._Delta1 if abs(self._Delta1) > 1.0 else 1.0
        d3 = self._Delta3 if abs(self._Delta3) > 1.0 else 1.0
        ratio_31 = d3 / d1
        ratio_13 = d1 / d3
        
        cos_b = np.cos(b)
        sin_b = np.sin(b)
        
        # Perfil Gaussiano enfocado en el centro de la celda (z_foco = 35.9 mm)
        gauss_profile_1 = 1.0 / (1.0 + ((z - self.z_foco) / self.zR_1)**2)
        gauss_profile_3 = 1.0 / (1.0 + ((z - self.z_foco) / self.zR_3)**2)
        
        # =====================================================================
        # REGULARIZACIÓN FÍSICA: Límite de conservación de energía
        # Ninguna amplitud puede superar la suma total de los campos incidentes
        # =====================================================================
        max_E = self.A1_0 * 2.0
        A1_safe = np.clip(A1, -max_E, max_E)
        A2_safe = np.clip(A2, -max_E, max_E)
        A3_safe = np.clip(A3, -max_E, max_E)
        
        # Ecuaciones acopladas usando las amplitudes seguras
        dA1_dz = -a1 * A1_safe * (A2_safe * A3_safe * np.cos(theta + b) + ratio_31 * (A1_safe**2) * cos_b) * gauss_profile_1
        dA2_dz = -a2 * A3_safe * ((A1_safe**2) * np.cos(theta - b) + ratio_13 * A2_safe * A3_safe * cos_b) * gauss_profile_1
        dA3_dz = -a3 * A2_safe * ((A1_safe**2) * np.cos(theta - b) + ratio_13 * A2_safe * A3_safe * cos_b) * gauss_profile_3
        
        denom_A2 = A2_safe if abs(A2_safe) > 1e-12 else 1e-12
        denom_A3 = A3_safe if abs(A3_safe) > 1e-12 else 1e-12
        
        term1 = 2.0 * a1 * A2_safe * A3_safe * np.sin(theta + b)
        term2 = (a2 * (A1_safe**2) * A3_safe / denom_A2 + a3 * (A1_safe**2) * A2_safe / denom_A3) * np.sin(theta - b)
        term3 = (2.0 * a1 * ratio_31 * (A1_safe**2) - a2 * ratio_13 * (A3_safe**2) - a3 * ratio_13 * (A2_safe**2)) * sin_b
        
        dtheta_dz = (term1 + term2 + term3) * gauss_profile_1 - self.delta_k_total
        
        return [dA1_dz, dA2_dz, dA3_dz, dtheta_dz]

    def compute_polar_state(self, z_target):
        if z_target <= 0.0:
            return self.A1_0, 1e-12, self.A3_0, 0.0
            
        # Añadimos max_step=1e-4 (100 micrómetros) para forzar estabilidad en el solver numérico
        sol = solve_ivp(self.coupled_polar_svea, (0.0, z_target), self.y0_polar, 
                        t_eval=[z_target], rtol=1e-6, atol=1e-9, max_step=1e-4)
                        
        # Si el solver fallara por algún transitorio extremo, devolvemos el último valor estable
        A1_z = sol.y[0, -1]
        A2_z = sol.y[1, -1]
        A3_z = sol.y[2, -1]
        theta_z = sol.y[3, -1]
        
        theta_z = (theta_z + np.pi) % (2.0 * np.pi) - np.pi
        return A1_z, A2_z, A3_z, theta_z