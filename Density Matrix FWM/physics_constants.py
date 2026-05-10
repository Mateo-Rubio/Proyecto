import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp

class CesiumFWMSystem:
    """
    Modelo unificado para Mezclado de Cuatro Ondas en Cesio.
    Resuelve la propagación espacial utilizando estrictamente el formalismo 
    de amplitudes reales acopladas (A_j) y fase relativa (theta) de Boyd et al.
    """
    def __init__(self, I1_Wcm2=1010.28, I2_Wcm2=0.0, I3_Wcm2=7.91, N_density=3.0e18):
        self.h_bar = const.hbar
        self.c = const.c
        self.epsilon_0 = const.epsilon_0
        self.au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
        
        # Tasas de decaimiento (rad/s)
        self.Gamma_ba = 2.0 * np.pi * 5.2227e6
        self.Gamma_ca = 2.0 * np.pi * 1.0e6
        self.Gamma_cb = 2.0 * np.pi * 3.71e6
        self.gamma_c  = 2.0 * np.pi * 2.2e6
        
        self.mu_ba = 3.4707 * self.au_to_Cm
        self.mu_cb = 0.7394 * self.au_to_Cm
        self.N = N_density
        
        # Frecuencias: [omega_1 (Pump), omega_2 (Gen), omega_3 (Seed)]
        self.lmb = np.array([822.48e-9, 794.39e-9, 852.35e-9])
        self.k = 2.0 * np.pi / self.lmb
        self.om = self.k * self.c
        
        self._I1 = I1_Wcm2
        self._I2 = I2_Wcm2
        self._I3 = I3_Wcm2
        
        omega_ba = 2.0 * np.pi * self.c / 852.35e-9
        omega_laser1 = 2.0 * np.pi * self.c / 822.48e-9
        
        self._Delta1_init = omega_ba - omega_laser1
        self._Delta1 = self._Delta1_init
        self._Delta2 = 0.0
        self._Delta3 = self._Delta1
        
        self.svea_coeffs = self.om / (2.0 * self.epsilon_0 * self.c)
        self._update_boundary_fields()

    def _update_boundary_fields(self):
        """Calcula las amplitudes de campo reales en la frontera z=0."""
        def I_to_E(I_cm2):
            return np.sqrt((2.0 * I_cm2 * 1e4) / (self.c * self.epsilon_0))
        
        self.A1_0 = I_to_E(self._I1)
        # Piso de ruido cuántico para evitar divergencias de fase en el vacío inicial
        self.A2_0 = I_to_E(self._I2) if self._I2 > 0 else 1e-12
        self.A3_0 = I_to_E(self._I3)
        
        # Vector de estado polar inicial: [A1, A2, A3, theta]
        # Asumimos que entran en fase colectiva inicial theta(0) = 0
        self.y0_polar = [self.A1_0, self.A2_0, self.A3_0, 0.0]

    # =========================================================================
    # PROPIEDADES POLARES ANALÍTICAS Y SUSCEPTIBILIDADES
    # =========================================================================
    @property
    def beta(self):
        return np.arctan2(-self._Delta2, self.Gamma_ca)

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
    # GETTERS Y SETTERS CON CANDADO FÍSICO
    # =========================================================================
    @property
    def I1(self): return self._I1
    @I1.setter
    def I1(self, val): self._I1 = val; self._update_boundary_fields()

    @property
    def I3(self): return self._I3
    @I3.setter
    def I3(self, val): self._I3 = val; self._update_boundary_fields()

    @property
    def Delta1(self): return self._Delta1
    @Delta1.setter
    def Delta1(self, val): self._Delta1 = val

    @property
    def Delta2(self): return self._Delta2
    @Delta2.setter
    def Delta2(self, val): 
        self._Delta2 = val
        self._Delta1 = self._Delta1_init + (val / 2.0)

    @property
    def Delta3(self): return self._Delta3
    @Delta3.setter
    def Delta3(self, val): self._Delta3 = val

    # =========================================================================
    # SISTEMA DIFERENCIAL POLAR PURO DE BOYD ET AL. (Ecs. dA1, dA2, dA3, dtheta)
    # =========================================================================
    def coupled_polar_svea(self, z, y, delta_k):
        A1, A2, A3, theta = y
        
        # Extraer parámetros dinámicos
        a1, a2, a3 = self.alpha_coeffs
        b = self.beta
        d1 = self._Delta1 if abs(self._Delta1) > 1.0 else 1.0
        d3 = self._Delta3 if abs(self._Delta3) > 1.0 else 1.0
        ratio_31 = d3 / d1
        ratio_13 = d1 / d3
        
        cos_b = np.cos(b)
        sin_b = np.sin(b)
        
        # Ecuaciones de Amplitud Reales (dA1/dz, dA2/dz, dA3/dz)
        dA1_dz = -a1 * A1 * (A2 * A3 * np.cos(theta + b) + ratio_31 * (A1**2) * cos_b)
        dA2_dz = -a2 * A3 * ((A1**2) * np.cos(theta - b) + ratio_13 * A2 * A3 * cos_b)
        dA3_dz = -a3 * A2 * ((A1**2) * np.cos(theta - b) + ratio_13 * A2 * A3 * cos_b)
        
        # Ecuación de Fase Relativa (dtheta/dz) con protección contra división por cero
        denom_A2 = A2 if A2 > 1e-15 else 1e-15
        denom_A3 = A3 if A3 > 1e-15 else 1e-15
        
        term1 = 2.0 * a1 * A2 * A3 * np.sin(theta + b)
        term2 = (a2 * (A1**2) * A3 / denom_A2 + a3 * (A1**2) * A2 / denom_A3) * np.sin(theta - b)
        term3 = (2.0 * a1 * ratio_31*(A1**2) - a2 * ratio_13 * (A3**2) - a3 * ratio_13 * (A2**2)) * sin_b
        
        dtheta_dz = term1 + term2 + term3 - delta_k
        
        return [dA1_dz, dA2_dz, dA3_dz, dtheta_dz]

    def compute_polar_state(self, z_target, delta_k):
        """
        Integra las ecuaciones polares de Boyd desde z=0 hasta z_target.
        Devuelve las amplitudes reales A1, A2, A3 y la fase colectiva theta.
        """
        if z_target <= 0.0:
            return self.A1_0, self.A2_0, self.A3_0, 0.0
            
        sol = solve_ivp(self.coupled_polar_svea, (0.0, z_target), self.y0_polar, 
                        args=(delta_k,), t_eval=[z_target], rtol=1e-8, atol=1e-8)
                        
        A1_z = sol.y[0, -1]
        A2_z = sol.y[1, -1]
        A3_z = sol.y[2, -1]
        theta_z = sol.y[3, -1]
        
        # Envoltura de fase estricta a [-pi, pi]
        theta_z = (theta_z + np.pi) % (2.0 * np.pi) - np.pi
        
        return A1_z, A2_z, A3_z, theta_z