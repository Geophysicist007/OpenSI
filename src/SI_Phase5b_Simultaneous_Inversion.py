#!/usr/bin/env python3
"""
OpenSI — Phase 5b: Simultaneous Pre-Stack Inversion

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase5b_Simultaneous_Inversion.py
"""


# ======================================================================
# # OpenSI — Phase 5b: Simultaneous Pre-Stack Inversion
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Invert near/mid/far angle stacks simultaneously for IP, IS, and density.
# **Reference:** Hampson et al. (2005), Fatti et al. (1994), Aki & Richards (2002)
#
# ---
#
# ## The Simultaneous Inversion Equation
#
# For each angle trace $\theta$:
#
# $$s_\theta = \tilde{c}_1 W_\theta D L_P + \tilde{c}_2 W_\theta D \Delta L_S + c_3 W_\theta D \Delta L_D$$
#
# Stacking all $N$ angle traces into one system:
#
# $$\begin{pmatrix} s_{\theta_1} \\ s_{\theta_2} \\ s_{\theta_3} \end{pmatrix} =
# \begin{pmatrix} \tilde{c}_{1,1} W_1 D & \tilde{c}_{2,1} W_1 D & c_{3,1} W_1 D \\
# \tilde{c}_{1,2} W_2 D & \tilde{c}_{2,2} W_2 D & c_{3,2} W_2 D \\
# \tilde{c}_{1,3} W_3 D & \tilde{c}_{2,3} W_3 D & c_{3,3} W_3 D \end{pmatrix}
# \begin{pmatrix} L_P \\ \Delta L_S \\ \Delta L_D \end{pmatrix}$$
#
# We solve for $L_P$, $\Delta L_S$, $\Delta L_D$ using conjugate gradients, then recover:
# - $I_P = \exp(L_P)$
# - $I_S = \exp(k \cdot L_P + k_c + \Delta L_S)$
# - $\rho = \exp(m \cdot L_P + m_c + \Delta L_D)$
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Phase 5a outputs ---
ANGLE_TRACES_CSV = 'Phase5a_angle_traces.csv'
SIM_PARAMS_CSV   = 'Phase5a_sim_params.csv'
WAV_NEAR_CSV     = 'Phase5a_wavelet_near.csv'
WAV_MID_CSV      = 'Phase5a_wavelet_mid.csv'
WAV_FAR_CSV      = 'Phase5a_wavelet_far.csv'

# --- Phase 1 outputs ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'

# --- Inversion parameters ---
T_WIN_START      = 2.60
T_WIN_END        = 3.26
N_ITERATIONS     = 30
PREWHITENING_LP  = 0.01      # Prewhitening for LP (well resolved)
PREWHITENING_LS  = 0.05      # Prewhitening for ΔLS (less resolved → more regularization)
PREWHITENING_LD  = 0.10      # Prewhitening for ΔLD (least resolved → most regularization)

# --- Output ---
OUTPUT_PREFIX    = 'Phase5b'

# --- Angle definitions (must match Phase 5a) ---
ANGLES = {
    'near': {'central': 10.0},
    'mid':  {'central': 25.0},
    'far':  {'central': 37.0},
}

# %%  [Cell 2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 11, 'figure.dpi': 120,
})

# --- Load all data ---
df_ai = pd.read_csv(AI_FULLBAND_CSV)
df_lfm = pd.read_csv(LFM_CSV)
df_angle = pd.read_csv(ANGLE_TRACES_CSV)

params = pd.read_csv(SIM_PARAMS_CSV, index_col=0)
k = float(params.loc['k', 'value'])
k_c = float(params.loc['k_c', 'value'])
m = float(params.loc['m', 'value'])
m_c = float(params.loc['m_c', 'value'])

twt_s = df_ai['TWT_s'].values
twt_ms = df_ai['TWT_ms'].values
dt_s = np.round(np.median(np.diff(twt_s)), 6)
fs = 1.0 / dt_s

# Truth
ai_true = df_ai['AI'].values
si_true = df_ai['SI'].values
rhob_true = df_ai['RHOB'].values
ln_ai_true = df_ai['LN_AI'].values
ln_si_true = df_ai['LN_SI'].values
ln_rhob_true = df_ai['LN_RHOB'].values

# LFM
ln_ai_lfm = df_lfm['LN_AI_LFM'].values
ln_si_lfm = df_lfm['LN_SI_LFM'].values
ln_rhob_lfm = df_lfm['LN_RHOB_LFM'].values

# Angle stacks
angle_names = ['near', 'mid', 'far']
seismic_angles = {}
fatti_coeffs = {}
wavelets = {}

for name in angle_names:
    seismic_angles[name] = df_angle[f'Seismic_{name}'].values
    fatti_coeffs[name] = {
        'c1_tilde': float(params.loc[f'c1_tilde_{name}', 'value']),
        'c2_tilde': float(params.loc[f'c2_tilde_{name}', 'value']),
        'c3': float(params.loc[f'c3_{name}', 'value']),
        'theta': float(params.loc[f'theta_{name}', 'value']),
    }

for name, path in [('near', WAV_NEAR_CSV), ('mid', WAV_MID_CSV), ('far', WAV_FAR_CSV)]:
    wdf = pd.read_csv(path)
    wavelets[name] = wdf['amplitude'].values

print(f'Data loaded: {len(twt_s)} samples, dt={dt_s*1000:.1f} ms')
print(f'Background: k={k:.4f}, m={m:.4f}')
for name in angle_names:
    fc = fatti_coeffs[name]
    print(f'  {name}: θ={fc["theta"]:.0f}°, c̃₁={fc["c1_tilde"]:.4f}, '
          f'c̃₂={fc["c2_tilde"]:.4f}, c₃={fc["c3"]:.4f}')

# %%  [Cell 3]
# --- Window extraction ---
win_mask = (twt_s >= T_WIN_START) & (twt_s <= T_WIN_END)
idx_s = np.argmax(win_mask)
idx_e = len(win_mask) - np.argmax(win_mask[::-1])
n = idx_e - idx_s
twt_w = twt_ms[idx_s:idx_e]

# Window the data
seis_w = {name: seismic_angles[name][idx_s:idx_e].copy() for name in angle_names}

ln_ai_true_w = ln_ai_true[idx_s:idx_e].copy()
ln_si_true_w = ln_si_true[idx_s:idx_e].copy()
ln_rhob_true_w = ln_rhob_true[idx_s:idx_e].copy()
ln_ai_lfm_w = ln_ai_lfm[idx_s:idx_e].copy()
ln_si_lfm_w = ln_si_lfm[idx_s:idx_e].copy()
ln_rhob_lfm_w = ln_rhob_lfm[idx_s:idx_e].copy()

# Fill NaNs
for arr in [ln_ai_true_w, ln_si_true_w, ln_rhob_true_w,
            ln_ai_lfm_w, ln_si_lfm_w, ln_rhob_lfm_w]:
    nans = np.isnan(arr)
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

for name in angle_names:
    seis_w[name][np.isnan(seis_w[name])] = 0.0

# True ΔLS, ΔLD for validation
delta_LS_true_w = ln_si_true_w - (k * ln_ai_true_w + k_c)
delta_LD_true_w = ln_rhob_true_w - (m * ln_ai_true_w + m_c)

print(f'Window: {n} samples ({T_WIN_START*1000:.0f}–{T_WIN_END*1000:.0f} ms)')


# ======================================================================
# ---
# ## 1. Build the Simultaneous Forward Operator
#
# The joint system matrix $\mathbf{G}$ has dimensions $(N_{angles} \times n) \times (3n)$
# where the three blocks of unknowns are $[L_P, \Delta L_S, \Delta L_D]$.
# ======================================================================

# %%  [Cell 4]
def build_derivative_matrix(n):
    """First-difference matrix D (n×n)."""
    D = np.zeros((n, n))
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D

def build_wavelet_matrix(wav, n_trace):
    """Wavelet convolution matrix W (n_trace × n_trace)."""
    n_wav = len(wav)
    half = n_wav // 2
    W = np.zeros((n_trace, n_trace))
    for j in range(n_trace):
        for k_idx in range(n_wav):
            row = j + k_idx - half
            if 0 <= row < n_trace:
                W[row, j] = wav[k_idx]
    return W

# Build D (shared)
D = build_derivative_matrix(n)

# Build W for each angle
W_mats = {}
WD_mats = {}
for name in angle_names:
    W_mats[name] = build_wavelet_matrix(wavelets[name], n)
    WD_mats[name] = W_mats[name] @ D

# Build the joint system matrix G (3n_angles*n rows × 3n cols)
# Rows: [s_near; s_mid; s_far]
# Cols: [LP; ΔLS; ΔLD]
n_angles = len(angle_names)
G = np.zeros((n_angles * n, 3 * n))

for i, name in enumerate(angle_names):
    fc = fatti_coeffs[name]
    row_start = i * n
    row_end = (i + 1) * n
    
    # Column block 0: c̃₁ × W_θ × D × LP
    G[row_start:row_end, 0:n] = fc['c1_tilde'] * WD_mats[name]
    
    # Column block 1: c̃₂ × W_θ × D × ΔLS
    G[row_start:row_end, n:2*n] = fc['c2_tilde'] * WD_mats[name]
    
    # Column block 2: c₃ × W_θ × D × ΔLD
    G[row_start:row_end, 2*n:3*n] = fc['c3'] * WD_mats[name]

# Stack seismic traces into one vector
s_joint = np.concatenate([seis_w[name] for name in angle_names])

print(f'Joint system: G is {G.shape}, s is {s_joint.shape}')
print(f'Unknowns: 3 × {n} = {3*n} (LP, ΔLS, ΔLD)')

# Verify forward model
x_true = np.concatenate([ln_ai_true_w, 
                          delta_LS_true_w, 
                          delta_LD_true_w])
s_verify = G @ x_true
for i, name in enumerate(angle_names):
    cc_v = np.corrcoef(s_joint[i*n:(i+1)*n], s_verify[i*n:(i+1)*n])[0, 1]
    print(f'  {name} forward verify: CC = {cc_v:.4f}')


# ======================================================================
# ---
# ## 2. Simultaneous Inversion via Conjugate Gradients
#
# We solve $\mathbf{G}\mathbf{x} \approx \mathbf{s}$ where $\mathbf{x} = [L_P, \Delta L_S, \Delta L_D]^T$.
#
# Key advantage of simultaneous inversion (Hampson et al., 2005):
# - The three unknowns are **independent** (LP, deviations ΔLS, ΔLD)
# - Background trends are **built in** — we only solve for anomalies
# - We can apply **different regularization** to each variable:
# LP is well-resolved, ΔLS less so, ΔLD least of all
# ======================================================================

# %%  [Cell 5]
class SimultaneousInversion:
    """
    Simultaneous Pre-Stack Inversion (Hampson et al., 2005).
    
    Solves the joint Fatti system for LP, ΔLS, ΔLD
    using conjugate gradients with variable-specific regularization.
    """
    
    def __init__(self, G, s_joint, x0, n_samples,
                 lam_P=0.01, lam_S=0.05, lam_D=0.10):
        """
        Parameters
        ----------
        G : (N_angles*n, 3n) array
            Joint forward operator.
        s_joint : (N_angles*n,) array
            Stacked seismic traces.
        x0 : (3n,) array
            Initial guess [LP0, ΔLS0, ΔLD0].
        n_samples : int
            Number of time samples per trace.
        lam_P, lam_S, lam_D : float
            Prewhitening for each variable block.
        """
        self.G = G
        self.GT = G.T
        self.s = s_joint
        self.x0 = x0.copy()
        self.n = n_samples
        
        # Build block-diagonal regularization matrix
        self.lam = np.concatenate([
            lam_P * np.ones(n_samples),
            lam_S * np.ones(n_samples),
            lam_D * np.ones(n_samples),
        ])
        
    def objective(self, x):
        """J = ||s - Gx||² + Σ λᵢ(xᵢ - x0ᵢ)²"""
        residual = self.s - self.G @ x
        J_data = np.sum(residual**2)
        J_reg = np.sum(self.lam * (x - self.x0)**2)
        return J_data + J_reg, J_data, J_reg
    
    def gradient(self, x):
        """∇J = -G^T(s - Gx) + Λ(x - x0)"""
        residual = self.s - self.G @ x
        g = -self.GT @ residual + self.lam * (x - self.x0)
        return g
    
    def solve(self, n_iter=30, verbose=True):
        """Solve via Polak-Ribière conjugate gradients."""
        x = self.x0.copy()
        
        history = {'J': [], 'J_data': [], 'J_reg': [], 'cc_angles': []}
        
        g = self.gradient(x)
        d = -g.copy()
        
        for it in range(n_iter):
            # Line search
            Gd = self.G @ d
            num = np.dot(g, d)
            den = np.dot(Gd, Gd) + np.dot(self.lam * d, d)
            if abs(den) < 1e-30:
                break
            alpha = -num / den
            
            x_new = x + alpha * d
            g_new = self.gradient(x_new)
            
            # Polak-Ribière
            beta = max(0, np.dot(g_new, g_new - g) / (np.dot(g, g) + 1e-30))
            d = -g_new + beta * d
            
            x = x_new
            g = g_new
            
            # Record
            J, Jd, Jr = self.objective(x)
            history['J'].append(J)
            history['J_data'].append(Jd)
            history['J_reg'].append(Jr)
            
            # Per-angle synthetic CC
            s_synth = self.G @ x
            ccs = []
            for i in range(len(s_synth) // self.n):
                cc = np.corrcoef(self.s[i*self.n:(i+1)*self.n],
                                 s_synth[i*self.n:(i+1)*self.n])[0, 1]
                ccs.append(cc)
            history['cc_angles'].append(ccs)
            
            if verbose and (it < 5 or it % 5 == 0 or it == n_iter - 1):
                cc_str = ', '.join([f'{c:.3f}' for c in ccs])
                print(f'  Iter {it:3d}: J={J:.4f} Jd={Jd:.4f} Jr={Jr:.4f} '
                      f'CC=[{cc_str}]')
        
        # Final synthetic
        history['synth_final'] = self.G @ x
        
        return x, history

# %%  [Cell 6]
# --- Set up initial guess ---
# LP = LFM of ln(IP)
# ΔLS = 0 (assume no AVO anomaly initially)
# ΔLD = 0
x0 = np.concatenate([ln_ai_lfm_w, np.zeros(n), np.zeros(n)])

print('='*70)
print('SIMULTANEOUS PRE-STACK INVERSION')
print(f'  Iterations: {N_ITERATIONS}')
print(f'  Prewhitening: LP={PREWHITENING_LP}, ΔLS={PREWHITENING_LS}, ΔLD={PREWHITENING_LD}')
print(f'  Angles: {[fatti_coeffs[n]["theta"] for n in angle_names]}°')
print('='*70)

inv = SimultaneousInversion(
    G, s_joint, x0, n,
    lam_P=PREWHITENING_LP,
    lam_S=PREWHITENING_LS,
    lam_D=PREWHITENING_LD,
)

x_result, history = inv.solve(N_ITERATIONS, verbose=True)

# Extract results
LP_inv = x_result[0:n]
dLS_inv = x_result[n:2*n]
dLD_inv = x_result[2*n:3*n]

# Recover physical properties
IP_inv = np.exp(LP_inv)
IS_inv = np.exp(k * LP_inv + k_c + dLS_inv)
Rho_inv = np.exp(m * LP_inv + m_c + dLD_inv)
VpVs_inv = IP_inv / IS_inv

# Truth for comparison
IP_true = np.exp(ln_ai_true_w)
IS_true = np.exp(ln_si_true_w)
Rho_true = np.exp(ln_rhob_true_w)
VpVs_true = IP_true / IS_true

# Metrics
valid = (np.isfinite(IP_true) & np.isfinite(IP_inv) &
         np.isfinite(IS_true) & np.isfinite(IS_inv) &
         np.isfinite(Rho_true) & np.isfinite(Rho_inv))

cc_ip = np.corrcoef(IP_true[valid], IP_inv[valid])[0, 1]
cc_is = np.corrcoef(IS_true[valid], IS_inv[valid])[0, 1]
cc_rho = np.corrcoef(Rho_true[valid], Rho_inv[valid])[0, 1]
cc_vpvs = np.corrcoef(VpVs_true[valid], VpVs_inv[valid])[0, 1]

print(f'\nResults vs well truth:')
print(f'  IP:    CC = {cc_ip:.4f}')
print(f'  IS:    CC = {cc_is:.4f}')
print(f'  ρ:     CC = {cc_rho:.4f}')
print(f'  VP/VS: CC = {cc_vpvs:.4f}')


# ======================================================================
# ---
# ## 3. Convergence Diagnostics
# ======================================================================

# %%  [Cell 7]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].semilogy(history['J'], 'k-o', ms=3)
axes[0].set_title('Total Objective', fontweight='bold')
axes[0].set_xlabel('Iteration')

axes[1].semilogy(history['J_data'], 'b-o', ms=3, label='Data misfit')
axes[1].semilogy(history['J_reg'], 'r-s', ms=3, label='Regularization')
axes[1].set_title('Data vs Regularization', fontweight='bold')
axes[1].set_xlabel('Iteration')
axes[1].legend(fontsize=8)

cc_arr = np.array(history['cc_angles'])
for i, (name, color) in enumerate([('Near','blue'),('Mid','green'),('Far','red')]):
    axes[2].plot(cc_arr[:, i], f'{color[0].lower()}-o', ms=3, label=name)
axes[2].set_title('Synth-to-Seismic CC per Angle', fontweight='bold')
axes[2].set_xlabel('Iteration')
axes[2].set_ylim(0, 1.05)
axes[2].legend(fontsize=8)

fig.suptitle('Simultaneous Inversion — Convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 4. Inversion Results — IP, IS, Density, VP/VS
# ======================================================================

# %%  [Cell 8]
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.3], wspace=0.1)

tracks = [
    ('P-Impedance', IP_true, IP_inv, np.exp(ln_ai_lfm_w), cc_ip, 'red'),
    ('S-Impedance', IS_true, IS_inv, np.exp(ln_si_lfm_w), cc_is, 'blue'),
    ('Density', Rho_true, Rho_inv, np.exp(ln_rhob_lfm_w), cc_rho, 'green'),
    ('VP/VS', VpVs_true, VpVs_inv, None, cc_vpvs, 'purple'),
    ('ΔLS Anomaly', delta_LS_true_w, dLS_inv, None, 
     np.corrcoef(delta_LS_true_w[valid], dLS_inv[valid])[0,1], 'darkorange'),
]

for i, (title, truth, inv_result, lfm_data, cc, color) in enumerate(tracks):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(truth, twt_w, 'k-', lw=0.5, label='Well truth')
    ax.plot(inv_result, twt_w, '-', color=color, lw=1.2, label='Inverted')
    if lfm_data is not None:
        ax.plot(lfm_data, twt_w, '--', color='gray', lw=0.8, alpha=0.5, label='LFM')
    ax.set_title(f'{title}\nCC={cc:.3f}', fontweight='bold', color=color)
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    else:
        ax.set_yticklabels([])
    ax.legend(fontsize=7, loc='lower left')

# Info panel
ax = fig.add_subplot(gs[0, 5])
ax.axis('off')
ax.text(0.1, 0.5,
    f'Parameters\n'
    f'───────────\n'
    f'Iters: {N_ITERATIONS}\n'
    f'λ_P: {PREWHITENING_LP}\n'
    f'λ_S: {PREWHITENING_LS}\n'
    f'λ_D: {PREWHITENING_LD}\n'
    f'k: {k:.3f}\n'
    f'm: {m:.3f}\n'
    f'\nResults\n'
    f'───────────\n'
    f'IP:  {cc_ip:.3f}\n'
    f'IS:  {cc_is:.3f}\n'
    f'ρ:   {cc_rho:.3f}\n'
    f'VP/VS:{cc_vpvs:.3f}',
    transform=ax.transAxes, fontsize=10, va='center', fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('OpenSI Phase 5b — Simultaneous Pre-Stack Inversion\n'
             f'Well 15/9-F-11 T2 | Near({ANGLES["near"]["central"]:.0f}°) + '
             f'Mid({ANGLES["mid"]["central"]:.0f}°) + Far({ANGLES["far"]["central"]:.0f}°)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_Results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ======================================================================
# ---
# ## 5. LMR (Lambda-Mu-Rho) Transform & Cross-Plots
#
# From Goodway et al. (1997):
# - $\mu\rho = I_S^2$ (rigidity × density)
# - $\lambda\rho = I_P^2 - 2 I_S^2$ (incompressibility × density)
#
# These are the key fluid/lithology discriminators.
# ======================================================================

# %%  [Cell 9]
# LMR from inversion
mu_rho_inv = IS_inv**2
lambda_rho_inv = IP_inv**2 - 2 * IS_inv**2

# LMR from truth
mu_rho_true = IS_true**2
lambda_rho_true = IP_true**2 - 2 * IS_true**2

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Cross-plot 1: λρ vs µρ
ax = axes[0]
ax.scatter(mu_rho_true[valid], lambda_rho_true[valid], c='black', s=8, alpha=0.4, label='Well truth')
ax.scatter(mu_rho_inv[valid], lambda_rho_inv[valid], c='red', s=8, alpha=0.4, label='Inverted')
ax.set_xlabel('µρ (rigidity × density)')
ax.set_ylabel('λρ (incompressibility × density)')
ax.set_title('λρ vs µρ — Fluid Discrimination', fontweight='bold')
ax.legend(fontsize=9)

# Cross-plot 2: IP vs VP/VS
ax = axes[1]
ax.scatter(IP_true[valid], VpVs_true[valid], c='black', s=8, alpha=0.4, label='Well truth')
ax.scatter(IP_inv[valid], VpVs_inv[valid], c='blue', s=8, alpha=0.4, label='Inverted')
ax.set_xlabel('P-Impedance')
ax.set_ylabel('VP/VS')
ax.set_title('IP vs VP/VS — Lithology + Fluid', fontweight='bold')
ax.legend(fontsize=9)

# Cross-plot 3: IP vs IS
ax = axes[2]
ax.scatter(IP_true[valid], IS_true[valid], c='black', s=8, alpha=0.4, label='Well truth')
ax.scatter(IP_inv[valid], IS_inv[valid], c='green', s=8, alpha=0.4, label='Inverted')
# Add background trend line
ip_range = np.linspace(np.nanmin(IP_true), np.nanmax(IP_true), 50)
is_trend = np.exp(k * np.log(ip_range) + k_c)
ax.plot(ip_range, is_trend, 'r--', lw=2, label=f'Trend (k={k:.2f})')
ax.set_xlabel('P-Impedance')
ax.set_ylabel('S-Impedance')
ax.set_title('IP vs IS', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('LMR & Interpretation Cross-Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 6. Synthetic Trace Matching (Per Angle)
# ======================================================================

# %%  [Cell 10]
s_synth = history['synth_final']
colors = {'near': 'blue', 'mid': 'green', 'far': 'red'}

fig, axes = plt.subplots(1, 6, figsize=(20, 10), sharey=True)

for i, name in enumerate(angle_names):
    s_real = seis_w[name]
    s_inv = s_synth[i*n:(i+1)*n]
    cc = np.corrcoef(s_real, s_inv)[0, 1]
    resid = s_real - s_inv
    
    # Trace overlay
    ax = axes[i*2]
    ax.plot(s_real, twt_w, 'k-', lw=0.5, label='Seismic')
    ax.plot(s_inv, twt_w, '-', color=colors[name], lw=0.5, alpha=0.8, label='Synthetic')
    ax.set_title(f'{name.upper()}\nCC={cc:.3f}', fontweight='bold', color=colors[name])
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    ax.legend(fontsize=7)
    
    # Residual
    ax = axes[i*2+1]
    ax.plot(resid, twt_w, '-', color=colors[name], lw=0.4)
    ax.fill_betweenx(twt_w, 0, resid, where=resid>0, alpha=0.2, color='red')
    ax.fill_betweenx(twt_w, 0, resid, where=resid<0, alpha=0.2, color='blue')
    rms_r = np.sqrt(np.mean(resid**2))
    ax.set_title(f'Residual\nRMS={rms_r:.5f}', fontweight='bold', fontsize=10)

fig.suptitle('Synthetic Trace Matching — All Angles', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 7. Export Results
# ======================================================================

# %%  [Cell 11]
df_out = pd.DataFrame({
    'TWT_s': twt_s[idx_s:idx_e],
    'TWT_ms': twt_w,
    'IP_Truth': IP_true,
    'IS_Truth': IS_true,
    'Rho_Truth': Rho_true,
    'VpVs_Truth': VpVs_true,
    'IP_Inv': IP_inv,
    'IS_Inv': IS_inv,
    'Rho_Inv': Rho_inv,
    'VpVs_Inv': VpVs_inv,
    'LP_Inv': LP_inv,
    'DeltaLS_Inv': dLS_inv,
    'DeltaLD_Inv': dLD_inv,
    'MuRho_Inv': mu_rho_inv,
    'LambdaRho_Inv': lambda_rho_inv,
    'MuRho_Truth': mu_rho_true,
    'LambdaRho_Truth': lambda_rho_true,
})
out_csv = f'{OUTPUT_PREFIX}_inversion_results.csv'
df_out.to_csv(out_csv, index=False, float_format='%.6f')

print(f'Saved: {out_csv}')
print(f'\n{"="*60}')
print(f'Phase 5b COMPLETE — Simultaneous Pre-Stack Inversion')
print(f'{"="*60}')
print(f'  IP:    CC = {cc_ip:.4f}')
print(f'  IS:    CC = {cc_is:.4f}')
print(f'  ρ:     CC = {cc_rho:.4f}')
print(f'  VP/VS: CC = {cc_vpvs:.4f}')
print(f'\nOutputs:')
print(f'  {out_csv}')
print(f'  {OUTPUT_PREFIX}_Results.png')
print(f'\nReady for Phase 6 (Interpretation Dashboard)')

