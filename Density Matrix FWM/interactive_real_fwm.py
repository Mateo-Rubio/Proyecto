import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1. Import your exact physical constants
from physics_constants import *

# 2. Define the Mathematical Functions using Real SI Units + t and z
def calc_density_matrix(D1_factor, D2_factor, d3_array, t, z):
    # Convert factors back to real angular frequencies
    D1 = D1_factor * Gamma_ba
    D2 = D2_factor * Gamma_ba
    D3 = d3_array * Gamma_ba
    
    # Calculate Complex Fields dynamically based on t and z
    E_z = E0 * np.exp(1j * k * z)
    e_t = np.exp(-1j * om * t)
    
    # Common Denominators
    D_A = (D1 - 1j * Gamma_ba)
    D_B = (2 * D1 - D3 - 1j * Gamma_ba)
    D_C = (D3 - 1j * Gamma_ba)
    D_D = (D2 - D1 - 1j * Gamma_cb)
    D_E = (D2 + D3 - 2 * D1 - 1j * Gamma_cb)
    D_F = (D2 - D3 - 1j * Gamma_cb)
    D_G = (2 * D1 - D3 - 1j * Gamma_cb)
    D_pre_2 = (D2 - 1j * Gamma_ca)
    
    # 1st Order (rho_ba)
    rho_ba_1 = (mu_ba/h_bar) * ( (E_z[0]*e_t[0])/D_A + (E_z[1]*e_t[1])/D_B + (E_z[2]*e_t[2])/D_C )
    
    # 2nd Order (rho_ca)
    prefix_2 = (mu_cb * mu_ba) / (h_bar**2)
    t1_2 = (E_z[0]**2 * np.exp(-1j * 2 * om[0] * t)) / (D_A * D_pre_2)
    t2_2 = (E_z[1] * E_z[2] * np.exp(-1j * (om[1] + om[2]) * t)) / (D_B * D_pre_2)
    t3_2 = (E_z[1] * E_z[2] * np.exp(-1j * (om[1] + om[2]) * t)) / (D_C * D_pre_2)
    rho_ca_2 = prefix_2 * (t1_2 + t2_2 + t3_2)
    
    # 3rd Order (rho_cb)
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
    
    # 4th Order (rho_cc Population)
    prefix_4 = -(2 * np.abs(mu_ba)**2 * np.abs(mu_cb)**2) / (h_bar**4 * gamma_c)
    b1 = (np.abs(E_z[0])**4) / (D_A * D_D)
    b2 = (E_z[0]**2 * np.conj(E_z[1]) * np.conj(E_z[2])) * (1/(D_A * D_E) + 1/(D_A * D_F))
    b3 = (np.conj(E_z[0])**2 * E_z[1] * E_z[2]) * (1/(D_B * D_D) + 1/(D_C * D_D))
    b4 = (np.abs(E_z[1])**2 * np.abs(E_z[2])**2) * (1/(D_B * D_E) + 1/(D_B * D_F) + 1/(D_C * D_E) + 1/(D_C * D_F))
    rho_cc_4 = prefix_4 * np.imag((1/D_pre_2) * (b1 + b2 + b3 + b4))
    
    # --- BULLETPROOF SHAPE ENFORCER ---
    # This guarantees that even if the math collapses to a scalar, Matplotlib receives 2000 points
    shape_enforcer = np.ones_like(d3_array, dtype=np.complex128)
    
    return rho_ba_1 * shape_enforcer, rho_ca_2 * shape_enforcer, rho_cb_3 * shape_enforcer, rho_ba_3 * shape_enforcer, np.real(rho_cc_4 * shape_enforcer)


# 3. Setup the Matplotlib Figure and Grid
# Increased height from 9 to 10 for more safe space
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Massive 40% bottom margin ensures plots never touch the UI
# INCREASED hspace to 0.55 to prevent x-axis labels of row 1 overlapping with titles of row 2!
plt.subplots_adjust(bottom=0.40, left=0.05, right=0.95, hspace=0.65, wspace=0.3)
axs[1, 2].axis('off')

# Initial State Setup
init_D1_factor = factor_Delta1  
init_D2_factor = 0.0
init_view_width = 20.0          
init_t = 0.0
init_z = 0.0

view_state = {'center': 0.0}

def get_x_array(center, width):
    return np.linspace(center - width/2, center + width/2, 2000)

current_x = get_x_array(view_state['center'], init_view_width)
r1, r2, r3, rba3, r4 = calc_density_matrix(init_D1_factor, init_D2_factor, current_x, init_t, init_z)

# 4. Plotting Lines 
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
axs[1,1].set_title(r'4th Order: $\rho_{cc}^{(4)}$ (Real)')

for ax in axs.flat:
    if ax.has_data():
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'$\Delta_3 / \Gamma_{ba}$')

# =========================================================
# 5. Create Sliders and Buttons (Bulletproof Layout)
# =========================================================
# The highest UI element is at y=0.25, leaving a massive 15% gap below the subplots.
# The horizontal widths are constrained to 0.25 so the floating text never overlaps.

# ROW 1
ax_d2     = plt.axes([0.20, 0.25, 0.25, 0.02])  # Left Column
ax_width  = plt.axes([0.65, 0.25, 0.25, 0.02])  # Right Column

# ROW 2
ax_t      = plt.axes([0.20, 0.15, 0.25, 0.02])  # Left Column
ax_z      = plt.axes([0.65, 0.15, 0.25, 0.02])  # Right Column

# ROW 3 (Buttons cleanly spaced at the bottom)
ax_btn_0  = plt.axes([0.35, 0.05, 0.12, 0.05])
ax_btn_2d = plt.axes([0.55, 0.05, 0.12, 0.05])

# Initialize the Widgets
slider_d2 = Slider(ax_d2, r'822nm Tuning $\Delta_2$', -20.0, 20.0, valinit=init_D2_factor)
slider_width  = Slider(ax_width, 'View Width', 1.0, 1000.0, valinit=init_view_width)
slider_t = Slider(ax_t, r'Time $t$ (s)', 0.0, 4e-15, valinit=init_t, valfmt='%e')
slider_z = Slider(ax_z, r'Position $z$ (m)', 0.0, 2e-6, valinit=init_z, valfmt='%e')

btn_zero = Button(ax_btn_0, 'Teleport to 0')
btn_2d1  = Button(ax_btn_2d, r'Teleport to $2\Delta_1$')

# Button Logic
def jump_to_zero(event):
    view_state['center'] = 0.0
    update(None)

def jump_to_2d1(event):
    current_d1 = init_D1_factor + (slider_d2.val / 2.0)
    view_state['center'] = 2 * current_d1
    update(None)

btn_zero.on_clicked(jump_to_zero)
btn_2d1.on_clicked(jump_to_2d1)

# =========================================================
# 6. Update Function
# =========================================================
def update(val):
    d2 = slider_d2.val
    d1 = init_D1_factor + (d2 / 2.0)
    
    width = slider_width.val
    t_val = slider_t.val
    z_val = slider_z.val
    
    new_x = get_x_array(view_state['center'], width)
    
    r1, r2, r3, rba3, r4 = calc_density_matrix(d1, d2, new_x, t_val, z_val)
    
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

plt.show()