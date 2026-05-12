import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import time

# Importamos la clase física unificada
from boyd_fwm_solver import BoydFWMSolver

def main():
    print("--- Exploración Espectral de Ganancia FWM (Fórmula de Boyd) ---")
    
    # Instanciamos el sistema físico con los valores pico de tu laboratorio
    solver = BoydFWMSolver()
    
    # Función auxiliar para convertir Amplitud (CGS interno) a Intensidad SI (W/cm^2)
    def A_to_I(A):
        return (0.5 * const.c * const.epsilon_0 * A**2) / 1e4

    # 1. Definición de los rangos asimétricos de Boyd
    # Delta_2 acotado estrictamente a la resonancia estrecha (+/- 3 Gamma_ba)
    d2_span = np.linspace(-3.0, 3.0, 120)
    
    # Delta_3 abarca una región de barrido ancho para ver el desacople de fase (+/- 40 Gamma_ba)
    d3_span = np.linspace(-40.0, 40.0, 120)
    
    # Variables para almacenar los perfiles marginales de intensidad generada a la salida (z=L)
    I_gen_d2 = np.zeros_like(d2_span)
    I_gen_d3 = np.zeros_like(d3_span)
    
    # Anclamos los detunings cruzados en valores típicos donde exista coincidencia paramétrica
    d3_fixed = -10.0  # Mantenemos la semilla desintonizada hacia el rojo
    d2_fixed = 0.5    # Mantenemos el sistema cerca del pico de dos fotones
    
    start_time = time.time()
    
    # =========================================================================
    # BARRIDO 1: Perfil de Resonancia de Dos Fotones (Delta_2)
    # =========================================================================
    print(f"\nCalculando marginal fina para Delta_2 (fijando Delta_3 = {d3_fixed} Gamma_ba)...")
    solver.Delta3 = d3_fixed * solver.Gamma_ba
    
    for i, d2_factor in enumerate(d2_span):
        solver.Delta2 = d2_factor * solver.Gamma_ba
        try:
            # Integramos espacialmente la celda
            _, _, A2_z, _, _ = solver.compute_field_evolution(z_points=50)
            I_gen_d2[i] = A_to_I(A2_z[-1])
        except Exception:
            I_gen_d2[i] = 0.0
            
        print(f" Progreso D2: {i+1}/{len(d2_span)}", end='\r')
    print()

    # =========================================================================
    # BARRIDO 2: Perfil de Sintonización de Semilla (Delta_3)
    # =========================================================================
    print(f"Calculando marginal fina para Delta_3 (fijando Delta_2 = {d2_fixed} Gamma_ba)...")
    solver.Delta2 = d2_fixed * solver.Gamma_ba
    
    for i, d3_factor in enumerate(d3_span):
        solver.Delta3 = d3_factor * solver.Gamma_ba
        try:
            _, _, A2_z, _, _ = solver.compute_field_evolution(z_points=50)
            I_gen_d3[i] = A_to_I(A2_z[-1])
        except Exception:
            I_gen_d3[i] = 0.0
            
        print(f" Progreso D3: {i+1}/{len(d3_span)}", end='\r')
    print()
    
    print(f"\n¡Cálculo finalizado exitosamente en {time.time() - start_time:.2f} segundos!")

    # =========================================================================
    # VISUALIZACIÓN DE LOS RESULTADOS
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Gráfica 1: Sensibilidad extrema a la resonancia de 2 fotones
    ax1.plot(d2_span, I_gen_d2, 'b-', lw=2.5, label=r'Intensidad de Salida ($I_2$)')
    ax1.axvline(d2_fixed, color='gray', linestyle=':', label=f'Anclaje D3 ({d2_fixed})')
    ax1.set_xlabel(r'Desintonía de Dos Fotones $\Delta_2 / \Gamma_{ba}$', fontsize=12)
    ax1.set_ylabel(r'Intensidad FWM Generada $I_2$ ($\text{W/cm}^2$)', fontsize=12)
    ax1.set_title(r'Corte Espectral en Resonancia Estrecha ($\Delta_3 = -10\Gamma_{ba}$)', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Gráfica 2: Tolerancia espectral de la onda semilla
    ax2.plot(d3_span, I_gen_d3, 'g-', lw=2.5, label=r'Intensidad de Salida ($I_2$)')
    ax2.axvline(d3_fixed, color='gray', linestyle=':', label=f'Anclaje D2 ({d3_fixed})')
    ax2.set_xlabel(r'Desintonía de la Onda Semilla $\Delta_3 / \Gamma_{ba}$', fontsize=12)
    ax2.set_ylabel(r'Intensidad FWM Generada $I_2$ ($\text{W/cm}^2$)', fontsize=12)
    ax2.set_title(r'Corte Espectral de Sintonización Ancha ($\Delta_2 = 0.5\Gamma_{ba}$)', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.suptitle('Respuesta Paramétrica del FWM a la Salida de la Celda (Boyd et al.)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()