import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import warnings

# Importar la clase física blindada
from physics_constants import CesiumFWMSystem

def main():
    print("--- Generando Cortes Marginales de FWM en el Pico de Máxima Intensidad ---")
    
    # 1. Configuración del sistema y geometría
    L_cell = 0.0718  # 71.8 mm
    fwm_sys = CesiumFWMSystem()
    
    c = 299792458.0
    eps0 = 8.85418782e-12
    def A_to_I(A): 
        return (0.5 * c * eps0 * A**2) / 1e4

    E_scale = 1e5
    y0_scaled = [fwm_sys.A1_0 / E_scale, 1e-12 / E_scale, fwm_sys.A3_0 / E_scale, 0.0]

    # Función auxiliar limpia y numéricamente blindada
    def evaluate_fwm(d2, d3):
        fwm_sys.Delta2 = d2 * fwm_sys.Gamma_ba
        fwm_sys.Delta3 = d3 * fwm_sys.Gamma_ba
        try:
            # Silenciamos localmente los RuntimeWarning de SciPy (Jacobiano interno)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sol = solve_ivp(
                    fwm_sys.coupled_polar_svea, 
                    (0.0, L_cell), 
                    y0_scaled, 
                    method='BDF',
                    t_eval=[L_cell], 
                    max_step=5e-4,       # <--- SOLUCIÓN: Evita saltos gigantes y desbordamientos
                    rtol=1e-4, 
                    atol=1e-7
                )
            if sol.success:
                A2_out = sol.y[1, -1] * E_scale
                return A_to_I(A2_out)
        except Exception:
            pass
        return 0.0

    # 2. Definir los límites de los ejes
    d2_min, d2_max = -30.0, 30.0
    d3_min, d3_max = -80.0, 20.0

    start_time = time.time()

    # =========================================================================
    # FASE 1: Búsqueda Robusta del Punto de Máxima Ganancia (Malla Gruesa)
    # =========================================================================
    coarse_pts = 10  # 10x10 = 100 puntos de evaluación rápida
    print(f"\nFase 1: Buscando el pico global en malla gruesa de {coarse_pts}x{coarse_pts}...")
    
    d2_coarse = np.linspace(d2_min, d2_max, coarse_pts)
    d3_coarse = np.linspace(d3_min, d3_max, coarse_pts)
    
    max_I = -1.0
    d2_opt, d3_opt = 0.0, 0.0
    
    pts_eval = 0
    for d2 in d2_coarse:
        for d3 in d3_coarse:
            pts_eval += 1
            I_val = evaluate_fwm(d2, d3)
            if I_val > max_I:
                max_I = I_val
                d2_opt, d3_opt = d2, d3
            print(f" Explorando: {pts_eval}/{coarse_pts**2} | Máx encontrado: {max_I:.2e} W/cm²", end='\r')
            
    print(f"\n-> Pico global localizado en d2={d2_opt:.2f}, d3={d3_opt:.2f} (I={max_I:.2e} W/cm²)")

    # =========================================================================
    # FASE 2: Cálculo de las Marginales Finas en el Punto Óptimo
    # =========================================================================
    fine_pts = 100  # Alta resolución unidimensional
    print(f"\nFase 2: Extrayendo marginales de alta resolución ({fine_pts} puntos c/u)...")
    
    d2_fine = np.linspace(d2_min, d2_max, fine_pts)
    d3_fine = np.linspace(d3_min, d3_max, fine_pts)
    
    marginal_d2 = np.zeros(fine_pts)
    marginal_d3 = np.zeros(fine_pts)
    
    # A. Marginal respecto a Delta_2 (fijando Delta_3 óptimo)
    for i, d2 in enumerate(d2_fine):
        marginal_d2[i] = evaluate_fwm(d2, d3_opt)
        print(f" Calculando Marginal D2: {i+1}/{fine_pts}", end='\r')
    print()
        
    # B. Marginal respecto a Delta_3 (fijando Delta_2 óptimo)
    for i, d3 in enumerate(d3_fine):
        marginal_d3[i] = evaluate_fwm(d2_opt, d3)
        print(f" Calculando Marginal D3: {i+1}/{fine_pts}", end='\r')
    print()

    print(f"\n¡Cálculo total finalizado en {time.time() - start_time:.2f} segundos!")

    # =========================================================================
    # 3. Visualización de los Cortes Marginales
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica Izquierda: Resonancia de 2 fotones
    ax1.plot(d2_fine, marginal_d2, 'b-', lw=2, label=r'Marginal $\Delta_2$')
    ax1.axvline(d2_opt, color='r', linestyle='--', alpha=0.7, label=f'Pico óptimo ({d2_opt:.1f})')
    ax1.set_xlabel(r'$\Delta_2 / \Gamma_{ba}$ (Resonancia 2-fotones)', fontsize=12)
    ax1.set_ylabel(r'Intensidad Generada $I_{\text{gen}}$ ($W/cm^2$)', fontsize=12)
    ax1.set_title(r'Corte Marginal fijando $\Delta_3 =$' + f'{d3_opt:.1f}', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Gráfica Derecha: Sintonización de la semilla
    ax2.plot(d3_fine, marginal_d3, 'g-', lw=2, label=r'Marginal $\Delta_3$')
    ax2.axvline(d3_opt, color='r', linestyle='--', alpha=0.7, label=f'Pico óptimo ({d3_opt:.1f})')
    ax2.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$ (Sintonización Semilla)', fontsize=12)
    ax2.set_ylabel(r'Intensidad Generada $I_{\text{gen}}$ ($W/cm^2$)', fontsize=12)
    ax2.set_title(r'Corte Marginal fijando $\Delta_2 =$' + f'{d2_opt:.1f}', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.suptitle(f'Perfiles Marginales de FWM en Máxima Intensidad a la Salida ($L = {L_cell*1e3:.1f}$ mm)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()