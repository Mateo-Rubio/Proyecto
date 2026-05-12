import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Importar la clase física blindada
from physics_constants import CesiumFWMSystem

def main():
    print("--- Generando Mapa de Superficie 3D de Mezclado de Cuatro Ondas ---")
    
    # 1. Configuración del sistema y geometría
    L_cell = 0.0718  # 71.8 mm
    fwm_sys = CesiumFWMSystem()
    
    c = 299792458.0
    eps0 = 8.85418782e-12
    def A_to_I(A): 
        return (0.5 * c * eps0 * A**2) / 1e4

    # 2. Definir los límites de los ejes (Desintonías normalizadas por Gamma_ba)
    # Ajusta estos rangos según la región que desees explorar
    d2_min, d2_max = -30.0, 30.0
    d3_min, d3_max = -80.0, 20.0
    grid_points = 35  # Resolución de la malla (35x35 = 1225 integraciones)

    d2_arr = np.linspace(d2_min, d2_max, grid_points)
    d3_arr = np.linspace(d3_min, d3_max, grid_points)
    
    # Crear la malla bidimensional (Meshgrid)
    D2_mesh, D3_mesh = np.meshgrid(d2_arr, d3_arr)
    I_gen_surface = np.zeros_like(D2_mesh, dtype=np.float64)
    
    E_scale = 1e5
    y0_scaled = [fwm_sys.A1_0 / E_scale, 1e-12 / E_scale, fwm_sys.A3_0 / E_scale, 0.0]

    print(f"Calculando malla de {grid_points}x{grid_points} ({grid_points**2} puntos)...")
    start_time = time.time()

    # 3. Bucle de cálculo sobre la cuadrícula paramétrica
    total_points = grid_points * grid_points
    current_point = 0
    
    for i in range(grid_points):
        for j in range(grid_points):
            current_point += 1
            
            # Actualizar detunings en la clase física
            fwm_sys.Delta2 = D2_mesh[i, j] * fwm_sys.Gamma_ba
            fwm_sys.Delta3 = D3_mesh[i, j] * fwm_sys.Gamma_ba
            
            # Integrar solo hasta la salida (z = L_cell) para máxima velocidad
            sol = solve_ivp(fwm_sys.coupled_polar_svea, (0.0, L_cell), y0_scaled, 
                            method='BDF', t_eval=[L_cell], rtol=1e-4, atol=1e-7)
            
            if sol.success:
                # Extraer la amplitud generada final y convertirla a intensidad
                A2_out = sol.y[1, -1] * E_scale
                I_gen_surface[i, j] = A_to_I(A2_out)
            else:
                I_gen_surface[i, j] = 0.0
                
        # Imprimir progreso por filas
        elapsed = time.time() - start_time
        print(f"Progreso: {current_point}/{total_points} completado ({current_point/total_points*100:.1f}%) - Tiempo: {elapsed:.1f}s", end='\r')
        
    print(f"\n¡Cálculo finalizado en {time.time() - start_time:.2f} segundos!")

    # =========================================================================
    # 4. Visualización Profesional de la Superficie 3D
    # =========================================================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar superficie con mapa de color 'turbo' o 'inferno' para resaltar intensidades
    surf = ax.plot_surface(D2_mesh, D3_mesh, I_gen_surface, 
                           cmap='turbo', edgecolor='none', alpha=0.9, 
                           antialiased=False)
    
    # Configuración de etiquetas y visuales
    ax.set_xlabel(r'$\Delta_2 / \Gamma_{ba}$ (Resonancia 2-fotones)', labelpad=15)
    ax.set_ylabel(r'$\Delta_3 / \Gamma_{ba}$ (Sintonización Semilla)', labelpad=15)
    ax.set_zlabel(r'Intensidad Generada $I_{\text{gen}}$ ($W/cm^2$)', labelpad=15)
    ax.set_title(f'Superficie de Ganancia FWM a la Salida ($L = {L_cell*1e3:.1f}$ mm)', fontsize=14, pad=20)
    
    # Añadir barra de color (Colorbar)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label(r'$I_{\text{gen}}$ ($W/cm^2$)')
    
    # Ajustar el ángulo de vista inicial (Elevación, Azimut)
    ax.view_init(elev=30, azim=-125)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()