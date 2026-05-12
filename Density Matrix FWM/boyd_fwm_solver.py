import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class BoydFWMSolver:
    """
    Solución al sistema de ecuaciones acopladas SVEA en aproximación de onda plana
    basado estrictamente en la formulación de Robert Boyd et al. (Phys. Rev. A 35, 1648).
    """
    def __init__(self, I1_peak=1010.28, I3_peak=7.91, N_density=3.0e18):
        # 1. Constantes Fundamentales (SI base)
        self.h_bar = const.hbar
        self.c = const.c
        self.epsilon_0 = const.epsilon_0
        self.au_to_Cm = const.e * const.physical_constants['Bohr radius'][0]
        
        # Tasas de decaimiento y relajación (Gammas en rad/s)
        self.Gamma_ba = 2.0 * np.pi * 5.2227e6
        self.Gamma_ca = 2.0 * np.pi * 1.0e6
        self.Gamma_cb = 2.0 * np.pi * 3.71e6
        self.gamma_c  = 2.0 * np.pi * 2.2e6
        
        # Momentos dipolares de transición del Cesio
        self.mu_ba = 3.4707 * self.au_to_Cm
        self.mu_cb = 0.7394 * self.au_to_Cm
        self.N = N_density
        
        # Frecuencias atómicas de resonancia (Cesio D line y transiciones superiores)
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
        Calcula las amplitudes de campo reales en la frontera z=0.
        Para un mapeo numérico limpio en el integrador, los campos se representan
        en el sistema electrostático estándar (CGS) escalados internamente.
        """
        # I (W/cm^2) a Campo Eléctrico Real en el vacío
        def I_to_E(I_cm2):
            return np.sqrt((2.0 * I_cm2 * 1e4) / (self.c * self.epsilon_0))
        
        self.A1_0 = I_to_E(self._I1_peak)
        self.A3_0 = I_to_E(self._I3_peak)
        # Inyectamos una semilla cuántica ultra-baja para permitir el arranque de A2
        self.A2_0 = 1e-12 

    def _update_propagation_dynamics(self):
        """
        Recalcula frecuencias, vectores de onda y susceptibilidades efectivas
        basado en la conservación de la energía del FWM: w2 = 2*w1 - w3.
        """                 
        w1 = self.omega_ba_atomic - self._Delta1
        w3 = self.omega_ba_atomic - self._Delta3
        w2 = 2.0 * w1 - w3
        self._om = np.array([w1, w2, w3])
        self._k  = self._om / self.c

    # =========================================================================
    # PROPIEDADES ÓPTICAS (Expone la física con nomenclatura estricta de Boyd)
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

    @property
    def delta_k(self):
        """Desacople del vector de onda en el vacío (Ecuación 14)."""
        return self._k[1] + self._k[2] - 2.0 * self._k[0]

    @property
    def beta(self):
        """Ángulo de fase de la susceptibilidad no lineal (Ecuación tan(beta))."""
        return np.arctan2(-self._Delta2, self.Gamma_ca)

    @property
    def chi_fwm(self):
        """
        Susceptibilidad no lineal resonante de tercer orden chi^(3).
        Saturamos denominadores cercanos a cero para evitar inestabilidades en resonancia pura.
        """
        K = (self.N * (np.abs(self.mu_ba)**2) * (np.abs(self.mu_cb)**2)) / (self.h_bar**3)
        # Suavizado para evitar singularidades
        d1 = self._Delta1 if abs(self._Delta1) > 1e3 else 1e3
        d3 = self._Delta3 if abs(self._Delta3) > 1e3 else 1e3
        d2 = self._Delta2 - 1j * self.Gamma_ca
        return K / (d1 * d3 * d2)

    @property
    def alpha_coeffs(self):
        """
        Coeficientes de acoplamiento espaciales alpha_j definidos en
        las Ecuaciones (eq:alpha1) y (eq:alphaj) de Boyd.
        """
        abs_chi = np.abs(self.chi_fwm)
        # Factor CGS a SI equivalente para el sistema SVEA en m^-1
        svea_base = self._om / (2.0 * self.epsilon_0 * self.c)
        return np.array([
            (2.0 * svea_base[0]) * abs_chi,
            svea_base[1] * abs_chi,
            svea_base[2] * abs_chi
        ])

    # =========================================================================
    # SISTEMA DIFERENCIAL DE BOYD (Ecuaciones dA1, dA2, dA3, dtheta)
    # =========================================================================
    def coupled_boyd_system(self, z, y_scaled):
        """
        Evaluación de derivadas espaciales d/dz estrictamente implementando 
        las ecuaciones acopladas (eq:dA1), (eq:dA2), (eq:dA3) y (eq:dtheta).
        """
        E_scale = 1e5  # Factor de acondicionamiento numérico
        
        # Bloqueo de seguridad para evitar desbordamientos
        max_limit = (self.A1_0 * 10.0) / E_scale
        A1_s = np.clip(y_scaled[0], -max_limit, max_limit)
        A2_s = np.clip(y_scaled[1], -max_limit, max_limit)
        A3_s = np.clip(y_scaled[2], -max_limit, max_limit)
        theta = y_scaled[3]
        
        # Desescala a amplitudes reales del sistema
        A1 = A1_s * E_scale
        A2 = A2_s * E_scale
        A3 = A3_s * E_scale
        
        a1, a2, a3 = self.alpha_coeffs
        b = self.beta
        dk = self.delta_k
        
        d1 = self._Delta1 if abs(self._Delta1) > 1e3 else 1e3
        d3 = self._Delta3 if abs(self._Delta3) > 1e3 else 1e3
        r_31 = d3 / d1
        r_13 = d1 / d3
        
        cos_b = np.cos(b)
        sin_b = np.sin(b)
        
        # Ecuaciones de Amplitud de Boyd (eq:dA1, eq:dA2, eq:dA3)
        dA1_dz = -a1 * A1 * (A2 * A3 * np.cos(theta + b) + r_31 * (A1**2) * cos_b)
        dA2_dz = -a2 * A3 * ((A1**2) * np.cos(theta - b) + r_13 * A2 * A3 * cos_b)
        dA3_dz = -a3 * A2 * ((A1**2) * np.cos(theta - b) + r_13 * A2 * A3 * cos_b)
        
        # Regularización de denominadores para la fase theta en campos nulos
        eps_f = E_scale * 1e-5
        denom_A2 = np.sqrt(A2**2 + eps_f**2)
        denom_A3 = np.sqrt(A3**2 + eps_f**2)
        
        # Ecuación de Fase de Boyd (eq:dtheta)
        t1 = 2.0 * a1 * A2 * A3 * np.sin(theta + b)
        t2 = (a2 * (A1**2) * A3 / denom_A2 + a3 * (A1**2) * A2 / denom_A3) * np.sin(theta - b)
        t3 = (2.0 * a1 * r_31 * (A1**2) - a2 * r_13 * (A3**2) - a3 * r_13 * (A2**2)) * sin_b
        
        dtheta_dz = t1 + t2 + t3 - dk
        
        return [dA1_dz / E_scale, dA2_dz / E_scale, dA3_dz / E_scale, dtheta_dz]

    def compute_field_evolution(self, z_points=200):
        """Integra espacialmente el sistema a lo largo de la celda."""
        E_scale = 1e5
        y0_scaled = [self.A1_0 / E_scale, self.A2_0 / E_scale, self.A3_0 / E_scale, 0.0]
        z_span = np.linspace(0.0, self.L_cell, z_points)
        
        # Usamos el método Radau/BDF adaptado a sistemas fuertemente acoplados
        sol = solve_ivp(self.coupled_boyd_system, (0.0, self.L_cell), y0_scaled,
                        method='BDF', t_eval=z_span, rtol=1e-6, atol=1e-9)
        
        A1_z = sol.y[0] * E_scale
        A2_z = sol.y[1] * E_scale
        A3_z = sol.y[2] * E_scale
        theta_z = sol.y[3]
        
        return z_span, A1_z, A2_z, A3_z, theta_z

    def calculate_rho_cc_4(self, A1, A2, A3, theta):
        """
        Cálculo riguroso de la población del estado superior rho_cc^(4)
        implementando la Ecuación analítica de Boyd (eq:rhocc_theta).
        """
        # Extraemos la parte imaginaria de la susceptibilidad
        im_chi = np.imag(self.chi_fwm)
        
        d1 = self._Delta1 if abs(self._Delta1) > 1e3 else 1e3
        d3 = self._Delta3 if abs(self._Delta3) > 1e3 else 1e3
        r_31 = d3 / d1
        r_13 = d1 / d3
        
        # Coeficiente global de la ecuación
        pref = (2.0 * im_chi) / (self.N * self.h_bar * self.gamma_c) * r_31
        
        # Evaluamos el corchete coherente de Boyd
        term_bombeo = A1**4
        term_fwm = (r_13**2) * (A2**2) * (A3**2)
        term_interf = 2.0 * r_13 * A2 * A3 * (A1**2) * np.cos(theta)
        
        rho_cc = pref * (term_bombeo + term_fwm + term_interf)
        return np.abs(rho_cc)  # Retornamos magnitud física poblacional