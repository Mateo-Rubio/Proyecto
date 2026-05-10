import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# 1. POSTER STYLING & CONFIGURATION (Matches baposter settings)
# =====================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

# Define the poster's dark green/light green theme
COLOR_PUMP = '#222222'       # Solid dark for driving field
COLOR_SEED = '#008800'       # Highlight green for amplified seed
COLOR_GEN = '#004400'        # Deep green for generated wave
COLOR_LOSS = '#AA0000'       # Warning red for TPA suppression

# =====================================================================
# 2. PHYSICAL PARAMETERS & COUPLED DIFFERENTIAL EQUATIONS
# =====================================================================
# Susceptibilities (normalized units)
chi_fwm = 1.0 + 0.0j
chi_s_tpa = 0.1j       # Purely imaginary on resonance (absorption loss)
chi_d_tpa = 0.2j       # Cross-TPA penalty term

def coupled_svea(z, y, delta_k):
    """
    Solves dE_j/dz = i * P(omega_j) for complex fields.
    y = [E1_real, E1_imag, E2_real, E2_imag, E3_real, E3_imag]
    """
    E1 = y[0] + 1j*y[1]
    E2 = y[2] + 1j*y[3]
    E3 = y[4] + 1j*y[5]
    
    # Phase matching exponential terms
    exp_neg = np.exp(-1j * delta_k * z)
    exp_pos = np.exp(1j * delta_k * z)
    
    # Macroscopic Polarizations (Equations P1, P2, P3)
    P1 = 2 * chi_fwm * E2 * E3 * np.conj(E1) * exp_neg + 2 * chi_s_tpa * (np.abs(E1)**2) * E1
    P2 = chi_fwm * (E1**2) * np.conj(E3) * exp_pos + chi_d_tpa * (np.abs(E3)**2) * E2
    P3 = chi_fwm * (E1**2) * np.conj(E2) * exp_pos + chi_d_tpa * (np.abs(E2)**2) * E3
    
    # Maxwell SVEA derivatives: dE/dz \propto i * P
    dE1_dz = 1j * P1
    dE2_dz = 1j * P2
    dE3_dz = 1j * P3
    
    return [dE1_dz.real, dE1_dz.imag, dE2_dz.real, dE2_dz.imag, dE3_dz.real, dE3_dz.imag]

# Spatial grid (z-axis from entrance window to exit window)
z_span = (0, 10.0)
z_eval = np.linspace(0, 10.0, 500)

# Initial Boundary Conditions: E1=10, E2=0 (Vacuum), E3=1 (Seeded)
y0 = [10.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# =====================================================================
# 3. GENERATE FIGURES FOR POSTER BOXES
# =====================================================================

# ---------------------------------------------------------------------
# FIGURE 1: Phase Matching Beating (For Box 3: 'phase_matching.pdf')
# ---------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4.5))

# Solve with perfect matching (Dk = 0) vs Mismatch (Dk = 3.0)
sol_match = solve_ivp(coupled_svea, z_span, y0, args=(0.0,), t_eval=z_eval)
sol_mismatch = solve_ivp(coupled_svea, z_span, y0, args=(3.0,), t_eval=z_eval)

# Calculate Intensity S_j \propto |E_j|^2
I2_match = sol_match.y[2]**2 + sol_match.y[3]**2
I2_mismatch = sol_mismatch.y[2]**2 + sol_mismatch.y[3]**2

ax1.plot(z_eval, I2_match, color=COLOR_GEN, lw=2.5, label=r'$\Delta k = 0$ (Phase Matched)')
ax1.plot(z_eval, I2_mismatch, color=COLOR_LOSS, lw=2, linestyle='--', label=r'$\Delta k \neq 0$ (Maker Fringes)')

ax1.set_xlabel('Propagation Distance $z$')
ax1.set_ylabel('Generated Intensity $S_2(z)$')
ax1.set_title('Parametric Growth vs. Wave-Vector Mismatch')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.6)
fig1.tight_layout()
fig1.savefig('phase_matching.pdf')

# ---------------------------------------------------------------------
# FIGURE 2: Complete Spatial Dynamics (For Box 4: 'python_console.pdf')
# ---------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4.5))

I1_match = sol_match.y[0]**2 + sol_match.y[1]**2
I3_match = sol_match.y[4]**2 + sol_match.y[5]**2

ax2.plot(z_eval, I1_match, color=COLOR_PUMP, lw=2, label=r'Pump $S_1$ (Depletion)')
ax2.plot(z_eval, I3_match, color=COLOR_SEED, lw=2, label=r'Seed $S_3$ (Amplification)')
ax2.plot(z_eval, I2_match, color=COLOR_GEN, lw=2, label=r'Generated $S_2$ (Growth)')

ax2.set_xlabel('Propagation Distance $z$')
ax2.set_ylabel('Optical Intensity $S_j(z)$')
ax2.set_title('Macroscopic Multi-Wave Mixing Dynamics')
ax2.legend(loc='center right')
ax2.grid(True, linestyle=':', alpha=0.6)
fig2.tight_layout()
fig2.savefig('python_console.pdf')

# ---------------------------------------------------------------------
# FIGURE 3: Quantum Interference / Suppression (For Box 5: 'theoretical_suppression.pdf')
# ---------------------------------------------------------------------
# The interference drive relies on the phase evolution E1^2 vs -E2*E3
fig3, ax3 = plt.subplots(figsize=(6, 4.5))

E1_cplx = sol_match.y[0] + 1j*sol_match.y[1]
E2_cplx = sol_match.y[2] + 1j*sol_match.y[3]
E3_cplx = sol_match.y[4] + 1j*sol_match.y[5]

# Upper state excitation probability \propto |E1^2 + E2*E3|^2
drive_pump = E1_cplx**2
drive_fwm = E2_cplx * E3_cplx
excitation_rate = np.abs(drive_pump + drive_fwm)**2
# Normalize to start at 1.0
excitation_rate = excitation_rate / excitation_rate[0]

ax3.plot(z_eval, excitation_rate, color=COLOR_LOSS, lw=2.5, label=r'Upper State Excitation $\rho_{cc}^{(4)}$')

ax3.set_xlabel('Propagation Distance $z$')
ax3.set_ylabel('Normalized Excitation Rate')
ax3.set_title('ASE Suppression via Destructive Interference')

# Add an arrow pointing to the transparency trap
ax3.annotate('Perfect Transparency\nAchieved (EIT Trap)', xy=(8.0, 0.05), xytext=(4.0, 0.4),
             arrowprops=dict(facecolor=COLOR_PUMP, shrink=0.08, width=1.5, headwidth=6),
             fontsize=11, fontweight='bold', color=COLOR_GEN)

ax3.grid(True, linestyle=':', alpha=0.6)
fig3.tight_layout()
fig3.savefig('theoretical_suppression.pdf')

print("Success: Generated phase_matching.pdf, python_console.pdf, and theoretical_suppression.pdf")