import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp

class CesiumFWMSystem:
    """
    Modelo SVEA polar de Boyd et al. modificado en Aproximación de Onda Plana.
    Asume que la intensidad pico se mantiene constante en toda la celda (71.8 mm),
    eliminando la divergencia geométrica para optimizar la estabilidad computacional.
    """
    def __init__(self, I1_peak=1010.28, I2_peak=0.0, I3_peak=7.91, N_density=3.0e18):
        self.h_bar = const.hbar
        self.c = const.c
        self.epsilon_0 = const.epsilon_0
        self.au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
        
        # Tasas de decaimiento y relajación (Gammas)
        self.Gamma_ba = 2.0 * np.pi * 5.2227e6
        self.Gamma_ca = 2.0 * np.pi * 1.0e6
        self.Gamma_cb = 2.0 * np.pi * 3.71e6
        self.gamma_c  = 2.0 * np.pi * 2.2e6
        
        # Momentos dipolares de transición del Cesio
        self.mu_ba = 3.4707 * self.au_to_Cm
        self.mu_cb = 0.7394 * self.au_to_Cm
        self.N = N_density
        
        # Frecuencias atómicas de resonancia
        self.omega_ba_atomic = 2.0 * np.pi * self.c / 852.35e-9
        self.omega_ca_atomic = 2.0 * np.pi * self.c / 411.24e-9
        
        # Frecuencias angulares iniciales de los láseres
        w1_init = 2.0 * np.pi * self.c / 822.48e-9
        w3_init = 2.0 * np.pi * self.c / 852.35e-9
        
        self._om = np.array([w1_init, 0.0, w3_init])
        self._Delta1_init = self.omega_ba_atomic - w1_init
        self._Delta1 = self._Delta1_init
        self._Delta2 = 0.0
        self._Delta3 = self.omega_ba_atomic - w3_init
        
        self._I1_peak = I1_peak
        self._I3_peak = I3_peak
        self.L_cell = 0.0718  # Longitud de la celda: 71.8 mm
        
        self._update_propagation_dynamics()
        self._update_boundary_fields()

    def _update_boundary_fields(self):
        """
        Amplitudes de campo eléctrico iniciales (z=0) basadas directamente 
        en la intensidad pico constante (Onda Plana).
        """
        def I_to_E(I_cm2):
            return np.sqrt((2.0 * I_cm2 * 1e4) / (self.c * self.epsilon_0))
        
        self.A1_0 = I_to_E(self._I1_peak)
        self.A3_0 = I_to_E(self._I3_peak)
        self.A2_0 = 1e-12  # Piso de ruido cuántico inicial para inicializar FWM
        
    def _update_propagation_dynamics(self):
        """
        Recalcula vectores de onda y coeficientes SVEA ante cambios en detunings.
        """
        w1 = self.omega_ba_atomic - self._Delta1
        w3 = self.omega_ba_atomic - self._Delta3
        w2 = 2.0 * w1 - w3
        self._om = np.array([w1, w2, w3])
        self._k  = self._om / self.c
        self.svea_coeffs = self._om / (2.0 * self.epsilon_0 * self.c)

    # =========================================================================
    # INTERFAZ DE PROPIEDADES (Expone variables internas de forma segura)
    # =========================================================================
    @property
    def Delta1(self): return self._Delta1
    
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
    def Delta3(self, val): 
        self._Delta3 = val
        self._update_propagation_dynamics()

    # ---> SOLUCIÓN: Exponemos explícitamente k y om para el script exterior <---
    @property
    def k(self): return self._k

    @property
    def om(self): return self._om
    # -------------------------------------------------------------------------

    @property
    def delta_k_total(self):
        return self._k[1] + self._k[2] - 2.0 * self._k[0]
        
    @property
    def beta(self): 
        return np.arctan2(-self._Delta2, self.Gamma_ca)

    @property
    def chi_fwm(self):
        """Susceptibilidad no lineal efectiva de tercer orden chi^(3)."""
        K = (self.N * (np.abs(self.mu_ba)**2) * (np.abs(self.mu_cb)**2)) / (self.h_bar**3)
        d1 = self._Delta1 if abs(self._Delta1) > 1.0 else 1.0
        d3 = self._Delta3 if abs(self._Delta3) > 1.0 else 1.0
        d2 = self._Delta2 - 1j * self.Gamma_ca
        return K / (d1 * d3 * d2)

    @property
    def alpha_coeffs(self):
        """Coeficientes de acoplamiento SVEA."""
        abs_chi = np.abs(self.chi_fwm)
        return np.array([
            (2.0 * self.svea_coeffs[0]) * abs_chi,
            self.svea_coeffs[1] * abs_chi,
            self.svea_coeffs[2] * abs_chi
        ])

    def coupled_polar_svea(self, z, y_scaled):
        """
        Sistema de ecuaciones diferenciales acopladas SVEA en onda plana.
        """
        E_scale = 1e5
        max_y = (self.A1_0 * 10.0) / E_scale
        
        # Estabilización numérica primaria
        A1_s = np.clip(y_scaled[0], -max_y, max_y)
        A2_s = np.clip(y_scaled[1], -max_y, max_y)
        A3_s = np.clip(y_scaled[2], -max_y, max_y)
        theta = y_scaled[3]
        
        A1 = A1_s * E_scale
        A2 = A2_s * E_scale
        A3 = A3_s * E_scale
        
        a1, a2, a3 = self.alpha_coeffs
        b = self.beta
        
        d1 = self._Delta1 if abs(self._Delta1) > 1.0 else 1.0
        d3 = self._Delta3 if abs(self._Delta3) > 1.0 else 1.0
        ratio_31, ratio_13 = d3 / d1, d1 / d3
        
        cos_b, sin_b = np.cos(b), np.sin(b)
        
        # Evolución espacial de amplitudes (Onda Plana)
        dA1_dz = - a1 * A1 * (A2 * A3 * np.cos(theta + b) + ratio_31 * (A1**2) * cos_b)
        dA2_dz = - a2 * A3 * ((A1**2) * np.cos(theta - b) + ratio_13 * A2 * A3 * cos_b)
        dA3_dz = - a3 * A2 * ((A1**2) * np.cos(theta - b) + ratio_13 * A2 * A3 * cos_b)
        
        # Regularización para evitar singularidades de fase a intensidades nulas
        denom_A2 = np.sqrt(A2**2 + (E_scale * 1e-4)**2)
        denom_A3 = np.sqrt(A3**2 + (E_scale * 1e-4)**2)
        
        term1 = 2.0 * a1 * A2 * A3 * np.sin(theta + b)
        term2 = (a2 * (A1**2) * A3 / denom_A2 + a3 * (A1**2) * A2 / denom_A3) * np.sin(theta - b)
        term3 = (2.0 * a1 * ratio_31 * (A1**2) - a2 * ratio_13 * (A3**2) - a3 * ratio_13 * (A2**2)) * sin_b
        
        dtheta_dz = term1 + term2 + term3 - self.delta_k_total
        
        return [dA1_dz / E_scale, dA2_dz / E_scale, dA3_dz / E_scale, dtheta_dz]

    def compute_polar_state(self, z_target):
        """
        Integra el sistema desde la entrada hasta una posición z_target.
        """
        if z_target <= 0.0:
            return self.A1_0, 1e-12, self.A3_0, 0.0
            
        E_scale = 1e5
        y0_scaled = [self.A1_0 / E_scale, 1e-12 / E_scale, self.A3_0 / E_scale, 0.0]
        
        sol = solve_ivp(self.coupled_polar_svea, (0.0, z_target), y0_scaled, 
                        method='BDF', t_eval=[z_target], rtol=1e-4, atol=1e-7)
                        
        return (sol.y[0, -1] * E_scale, sol.y[1, -1] * E_scale, 
                sol.y[2, -1] * E_scale, sol.y[3, -1])