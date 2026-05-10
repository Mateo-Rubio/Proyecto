import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1. Importar la clase física orientada a objetos (con propagación polar SVEA)
from physics_constants import CesiumFWMSystem

# Instanciar el sistema físico global inicializado con datos de laboratorio
fwm_sys = CesiumFWMSystem()

# 2. Rutina microscópica acoplada de forma nativa a las variables polares A_j y theta
def calc_density_matrix(D2_factor, d3_array, t, z, delta_k):
    # Delegar el candado físico: asignar Delta2 arrastra automáticamente a Delta1 en la clase
    fwm_sys.Delta2 = D2_factor * fwm_sys.Gamma_ba
    fwm_sys.Delta3 = fwm_sys.Delta1  # Anclar la desintonía central al valor actual
    
    # =========================================================================
    # EVOLUCIÓN MACROSCÓPICA POLAR PURA (Teoría de Boyd et al.)
    # Extraemos las amplitudes reales A_j y la fase colectiva autónoma theta
    # =========================================================================
    A1, A2, A3, theta = fwm_sys.compute_polar_state(z, delta_k)
    
    # Mapeo espectral a lo largo del eje X de la gráfica
    D1 = fwm_sys.Delta1
    D2 = fwm_sys.Delta2
    D3 = d3_array * fwm_sys.Gamma_ba
    
    Gamma_ba = fwm_sys.Gamma_ba
    Gamma_cb = fwm_sys.Gamma_cb
    Gamma_ca = fwm_sys.Gamma_ca
    
    # Denominadores característicos para las coherencias
    D_A = (D1 - 1j * Gamma_ba)
    D_B = (2.0 * D1 - D3 - 1j * Gamma_ba)
    D_C = (D3 - 1j * Gamma_ba)
    D_D = (D2 - D1 - 1j * Gamma_cb)
    D_E = (D2 + D3 - 2.0 * D1 - 1j * Gamma_cb)
    D_F = (D2 - D3 - 1j * Gamma_cb)
    D_G = (2.0 * D1 - D3 - 1j * Gamma_cb)
    D_pre_2 = (D2 - 1j * Gamma_ca)
    
    mu_ba = fwm_sys.mu_ba
    mu_cb = fwm_sys.mu_cb
    h_bar = fwm_sys.h_bar
    
    # Reconstrucción de los campos asumiendo la fase relativa en la onda generada (E2)
    # Esto permite evaluar las coherencias de orden inferior de forma estándar
    E_z = np.array([A1, A2 * np.exp(1j * theta), A3]) * np.exp(1j * fwm_sys.k * z)
    e_t = np.exp(-1j * fwm_sys.om * t)
    
    # 1er Orden
    rho_ba_1 = (mu_ba/h_bar) * ( (E_z[0]*e_t[0])/D_A + (E_z[1]*e_t[1])/D_B + (E_z[2]*e_t[2])/D_C )
    
    # 2do Orden
    prefix_2 = (mu_cb * mu_ba) / (h_bar**2)
    t1_2 = (E_z[0]**2 * np.exp(-1j * 2.0 * fwm_sys.om[0] * t)) / (D_A * D_pre_2)
    t2_2 = (E_z[1] * E_z[2] * np.exp(-1j * (fwm_sys.om[1] + fwm_sys.om[2]) * t)) / (D_B * D_pre_2)
    t3_2 = (E_z[1] * E_z[2] * np.exp(-1j * (fwm_sys.om[1] + fwm_sys.om[2]) * t)) / (D_C * D_pre_2)
    rho_ca_2 = prefix_2 * (t1_2 + t2_2 + t3_2)
    
    # 3er Orden Signal
    prefix_3 = -(np.abs(mu_ba)**2 * mu_cb) / (D_pre_2 * h_bar**3)
    t1_3 = (np.abs(E_z[0])**2 * E_z[0] * e_t[0]) / (D_A * D_D)
    t2_3 = (E_z[0]**2 * np.conj(E_z[1]) * e_t[2]) / (D_A * D_E)
    t3_3 = (E_z[0]**2 * np.conj(E_z[2]) * e_t[1]) / (D_A * D_F)
    t4_3 = (E_z[1] * E_z[2] * np.conj(E_z[0]) * e_t[0]) / (D_B * D_D)
    t5_3 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_B * D_E)
    t6_3 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_B * D_F)
    t7_3 = (E_z[2] * E_z[1] * np.conj(E_z[0]) * e_t[0]) / (D_C * D_D)
    t8_3 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_C * D_E)
    t9_3 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_C * D_F)
    rho_cb_3 = prefix_3 * (t1_3 + t2_3 + t3_3 + t4_3 + t5_3 + t6_3 + t7_3 + t8_3 + t9_3)

    # 3er Orden Corrección
    prefix_ba_3 = (np.abs(mu_cb)**2 * mu_ba) / (D_pre_2 * h_bar**3)
    ba3_1 = (np.abs(E_z[0])**2 * E_z[0] * e_t[0]) / (D_A**2)
    ba3_2 = (E_z[0]**2 * np.conj(E_z[1]) * e_t[2]) / (D_A * D_C)
    ba3_3 = (E_z[0]**2 * np.conj(E_z[2]) * e_t[1]) / (D_A * D_B)
    ba3_4 = (E_z[1] * E_z[2] * np.conj(E_z[0]) * e_t[0]) / (D_B * D_A)
    ba3_5 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_B * D_C)
    ba3_6 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_B**2)
    ba3_7 = (E_z[2] * E_z[1] * np.conj(E_z[0]) * e_t[0]) / (D_C * D_A)
    ba3_8 = (np.abs(E_z[1])**2 * E_z[2] * e_t[2]) / (D_C**2)
    ba3_9 = (np.abs(E_z[2])**2 * E_z[1] * e_t[1]) / (D_C * D_G)
    rho_ba_3 = prefix_ba_3 * (ba3_1 + ba3_2 + ba3_3 + ba3_4 + ba3_5 + ba3_6 + ba3_7 + ba3_8 + ba3_9)
    
    # =========================================================================
    # POBLACIÓN ANALÍTICA POLAR (Teoría de Boyd et al. - Ec. 16)
    # Demuestra analíticamente la supresión destructiva cuando theta -> pi
    # =========================================================================
    im_chi = np.imag(fwm_sys.chi_fwm)
    d1_safe = D1 if abs(D1) > 1.0 else 1.0
    ratio = D3 / d1_safe
    
    coeff_4 = (2.0 * im_chi) / (fwm_sys.N * h_bar * fwm_sys.gamma_c)
    
    # Trinomio acoplado a la fase colectiva theta
    term_A = A1**4
    term_B = ( (1.0 / ratio)**2 ) * (A2**2) * (A3**2)
    term_C = 2.0 * (1.0 / ratio) * A2 * A3 * (A1**2) * np.cos(theta)
    
    rho_cc_4_polar = coeff_4 * ratio * (term_A + term_B + term_C)
    
    shape_enforcer = np.ones_like(d3_array, dtype=np.float64)
    
    return rho_ba_1 * shape_enforcer, rho_ca_2 * shape_enforcer, rho_cb_3 * shape_enforcer, rho_ba_3 * shape_enforcer, rho_cc_4_polar * shape_enforcer

# 3. Inicialización de la Cuadrícula Gráfica
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(bottom=0.45, left=0.05, right=0.95, hspace=0.65, wspace=0.3)
axs[1, 2].axis('off')

# Estado inicial para los controles
init_D2_factor  = 0.0
init_view_width = 20.0          
init_t  = 0.0
init_z  = 0.0
init_dk = 0.0  

view_state = {'center': 0.0}

def get_x_array(center, width):
    return np.linspace(center - width/2, center + width/2, 2000)

current_x = get_x_array(view_state['center'], init_view_width)
r1, r2, r3, rba3, r4 = calc_density_matrix(init_D2_factor, current_x, init_t, init_z, init_dk)

# Trazado inicial de las curvas atómicas
line_r1_real, = axs[0,0].plot(current_x, np.real(r1), 'b', label="Real")
line_r1_imag, = axs[0,0].plot(current_x, np.imag(r1), 'r--', label="Imag")
axs[0,0].set_title(r'1st Order: $\rho_{ba}^{(1)}$')

line_r2_real, = axs[0,1].plot(current_x, np.real(r2), 'g', label="Real")
line_r2_imag, = axs[0,1].plot(current_x, np.imag(r2), 'purple', linestyle='--', label="Imag")
axs[0,1].set_title(r'2nd Order: $\rho_{ca}^{(2)}$')

line_r3_real, = axs[0,2].plot(current_x, np.real(r3), 'darkorange', label="Real")
line_r3_imag, = axs[0,2].plot(current_x, np.imag(r3), 'teal', linestyle='--', label="Imag")
axs[0,2].set_title(r'3rd Order Signal: $\rho_{cb}^{(3)}$')

line_rba3_real, = axs[1,0].plot(current_x, np.real(rba3), 'crimson', label="Real")
line_rba3_imag, = axs[1,0].plot(current_x, np.imag(rba3), 'navy', linestyle='--', label="Imag")
axs[1,0].set_title(r'3rd Order Correction: $\rho_{ba}^{(3)}$')

line_r4, = axs[1,1].plot(current_x, r4, 'k', label="Population")
axs[1,1].set_title(r'4th Order: $\rho_{cc}^{(4)}$ (Boyd Polar)')

for ax in axs.flat:
    if ax.has_data():
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')

# =========================================================
# 5. Controles y Botones de Interfaz
# =========================================================
ax_d2     = plt.axes([0.10, 0.32, 0.35, 0.02])
ax_width  = plt.axes([0.55, 0.32, 0.35, 0.02])
ax_t      = plt.axes([0.10, 0.25, 0.35, 0.02])
ax_z      = plt.axes([0.55, 0.25, 0.35, 0.02])
ax_dk     = plt.axes([0.10, 0.18, 0.35, 0.02])  

ax_btn_0  = plt.axes([0.35, 0.08, 0.12, 0.05])
ax_btn_2d = plt.axes([0.55, 0.08, 0.12, 0.05])

slider_d2    = Slider(ax_d2, r'822nm Tuning $\Delta_2$', -20.0, 20.0, valinit=init_D2_factor)
slider_width = Slider(ax_width, 'View Width', 1.0, 1000.0, valinit=init_view_width)
slider_t     = Slider(ax_t, r'Time $t$ (s)', 0.0, 4e-15, valinit=init_t, valfmt='%e')
slider_z     = Slider(ax_z, r'Position $z$ (m)', 0.0, 0.1, valinit=init_z, valfmt='%.4f')
slider_dk    = Slider(ax_dk, r'Phase Mismatch $\Delta k$', 0.0, 50.0, valinit=init_dk)

btn_zero = Button(ax_btn_0, 'Teleport to 0')
btn_2d1  = Button(ax_btn_2d, r'Teleport to $2\Delta_1$')

def jump_to_zero(event):
    view_state['center'] = 0.0
    update(None)

def jump_to_2d1(event):
    # Accedemos de forma directa a Delta1 desde la clase física sincronizada
    view_state['center'] = 2.0 * (fwm_sys.Delta1 / fwm_sys.Gamma_ba)
    update(None)

btn_zero.on_clicked(jump_to_zero)
btn_2d1.on_clicked(jump_to_2d1)

# =========================================================
# 6. Lógica de Actualización Gráfica Ultralimpia
# =========================================================
def update(val):
    d2 = slider_d2.val
    width  = slider_width.val
    t_val  = slider_t.val
    z_val  = slider_z.val
    dk_val = slider_dk.val
    
    new_x = get_x_array(view_state['center'], width)
    
    # Delegación absoluta: la clase orquesta internamente el arrastre Delta2 -> Delta1
    r1, r2, r3, rba3, r4 = calc_density_matrix(d2, new_x, t_val, z_val, dk_val)
    
    line_r1_real.set_data(new_x, np.real(r1)); line_r1_imag.set_data(new_x, np.imag(r1))
    line_r2_real.set_data(new_x, np.real(r2)); line_r2_imag.set_data(new_x, np.imag(r2))
    line_r3_real.set_data(new_x, np.real(r3)); line_r3_imag.set_data(new_x, np.imag(r3))
    line_rba3_real.set_data(new_x, np.real(rba3)); line_rba3_imag.set_data(new_x, np.imag(rba3))
    line_r4.set_data(new_x, r4)
    
    for ax in axs.flat:
        if ax.has_data():
            ax.set_xlim(new_x[0], new_x[-1])
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        
    fig.canvas.draw_idle()

slider_d2.on_changed(update)
slider_width.on_changed(update)
slider_t.on_changed(update)
slider_z.on_changed(update)
slider_dk.on_changed(update)

plt.show()