#!/usr/bin/env python3
"""
OpenSI — Phase 3: Model-Based (Constrained) Post-Stack Inversion

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase3_Model_Based_Inversion.py
"""


# ======================================================================
# # OpenSI — Phase 3: Model-Based (Constrained) Post-Stack Inversion
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** The industry-standard iterative inversion that uses the wavelet explicitly.
# **Reference:** Hampson & Russell (1991), Russell (1988), Oldenburg et al. (1983)
#
# ---
#
# ## Why Model-Based Inversion?
#
# Phase 2 showed that recursive and colored inversion barely beat the LFM baseline.
# This is because they **ignore the wavelet** — they treat the seismic as raw reflectivity,
# which it is not. Model-based inversion:
#
# 1. **Knows the wavelet** and explicitly removes its effect
# 2. **Starts from the LFM** as an initial guess and perturbs it
# 3. **Iteratively minimizes** the misfit between synthetic (W × RC_model) and real seismic
# 4. **Constrains** how far the result can deviate from the initial model
#
# ### The Forward Model
#
# $$\mathbf{s} = \frac{1}{2} \mathbf{W} \mathbf{D} \mathbf{L}_P$$
#
# where:
# - $\mathbf{s}$ = seismic trace vector
# - $\mathbf{W}$ = wavelet convolution matrix
# - $\mathbf{D}$ = first-difference (derivative) matrix
# - $\mathbf{L}_P = \ln(I_P)$ = log-impedance vector (the unknown)
#
# ### The Objective Function
#
# **Stochastic (soft constraint):**
# $$J = (1-w) \cdot \frac{\|\mathbf{s} - \mathbf{G}\mathbf{L}\|^2}{\|\mathbf{s}\|^2} + w \cdot \frac{\|\mathbf{L}_0 - \mathbf{L}\|^2}{\|\mathbf{L}_0\|^2}$$
#
# **Constrained (hard bounds):**
# $$\min_\mathbf{L} \|\mathbf{s} - \mathbf{G}\mathbf{L}\|^2 \quad \text{subject to} \quad L_{0i} - \delta \leq L_i \leq L_{0i} + \delta$$
#
# where $\mathbf{G} = \frac{1}{2}\mathbf{W}\mathbf{D}$ is the combined forward operator.
# ======================================================================


# ======================================================================
# ---
# ## User Parameters
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Input files (from Phase 1 & 2) ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'
PHASE2_CSV       = 'Phase2_inversion_results.csv'
WAVELET_FILE     = 'final_wavelet.csv'

# --- Analysis window (TWT in seconds) ---
T_WIN_START      = 2.60
T_WIN_END        = 3.26

# --- Inversion engine parameters ---
N_ITERATIONS     = 20         # Conjugate gradient iterations (STRATA default: 10–20)
PREWHITENING     = 0.01       # Prewhitening factor λ (stabilization, typical: 0.01–0.05)

# --- Stochastic (soft) constraint ---
MODEL_WEIGHT     = 0.3        # w: 0=pure seismic, 1=pure model. Typical: 0.2–0.5

# --- Constrained (hard bounds) ---
MAX_IMPEDANCE_CHANGE_PCT = 50.0  # Max deviation from initial model (% of avg impedance)

# --- Noise on synthetic for testing ---
ADD_NOISE        = True
SNR_DB           = 20

# --- Output ---
OUTPUT_PREFIX    = 'Phase3'


# ======================================================================
# ---
# ## 0. Imports & Load Data
# ======================================================================

# %%  [Cell 2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, fftconvolve, welch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 11, 'figure.dpi': 120,
})

# Load data
df_ai = pd.read_csv(AI_FULLBAND_CSV)
df_lfm = pd.read_csv(LFM_CSV)

twt_s = df_ai['TWT_s'].values
twt_ms = df_ai['TWT_ms'].values
dt_s = np.round(np.median(np.diff(twt_s)), 6)
fs = 1.0 / dt_s

ai_full = df_ai['AI'].values       # Truth
rc_p = df_ai['RC_P'].values        # True reflectivity
ln_ai_full = df_ai['LN_AI'].values # True ln(AI)
ln_ai_lfm = df_lfm['LN_AI_LFM'].values  # Initial model (LFM in log space)
ai_lfm = df_lfm['AI_LFM'].values

# Load Phase 2 results for comparison
try:
    df_p2 = pd.read_csv(PHASE2_CSV)
    ai_rec = df_p2['AI_Recursive'].values
    ai_ci = df_p2['AI_Colored'].values
    has_p2 = True
except:
    has_p2 = False

# Load wavelet
wav_df = pd.read_csv(WAVELET_FILE)
if len(wav_df.columns) == 1:
    wavelet = wav_df.iloc[:, 0].values
elif len(wav_df.columns) >= 2:
    wavelet = wav_df.iloc[:, 1].values
wavelet = wavelet / np.max(np.abs(wavelet))

print(f'Data: {len(twt_s)} samples, dt={dt_s*1000:.1f} ms')
print(f'Wavelet: {len(wavelet)} samples')

# %%  [Cell 3]
# --- Define analysis window ---
win_mask = (twt_s >= T_WIN_START) & (twt_s <= T_WIN_END)
idx_start = np.argmax(win_mask)
idx_end = len(win_mask) - np.argmax(win_mask[::-1])
n_win = idx_end - idx_start

print(f'Window: {T_WIN_START*1000:.0f}–{T_WIN_END*1000:.0f} ms, '
      f'{n_win} samples (idx {idx_start}:{idx_end})')

# Extract windowed data
twt_w = twt_ms[idx_start:idx_end]
rc_w = rc_p[idx_start:idx_end].copy()
rc_w[np.isnan(rc_w)] = 0.0
ln_ai_true_w = ln_ai_full[idx_start:idx_end].copy()
ln_ai_lfm_w = ln_ai_lfm[idx_start:idx_end].copy()
ai_true_w = ai_full[idx_start:idx_end].copy()

# Fill any NaNs in LFM with interpolation
for arr in [ln_ai_lfm_w, ln_ai_true_w]:
    nans = np.isnan(arr)
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), 
                              np.flatnonzero(~nans), arr[~nans])


# ======================================================================
# ---
# ## 1. Generate Synthetic Seismic (Input to Inversion)
# ======================================================================

# %%  [Cell 4]
def make_synthetic(rc, wavelet, snr_db=None, dt_sec=0.002):
    """Create synthetic seismic trace = W * rc + noise."""
    rc_c = rc.copy()
    rc_c[np.isnan(rc_c)] = 0.0
    synth = fftconvolve(rc_c, wavelet, mode='same')
    if snr_db is not None:
        sig_pow = np.mean(synth**2)
        noise_pow = sig_pow / (10**(snr_db/10))
        noise = np.random.randn(len(synth)) * np.sqrt(noise_pow)
        # Bandlimit noise
        nyq = 1/(2*dt_sec)
        sos = butter(4, [5/nyq, min(80/nyq, 0.99)], btype='band', output='sos')
        noise = sosfiltfilt(sos, noise)
        noise *= np.sqrt(noise_pow / np.mean(noise**2))
        synth += noise
    return synth

seismic_w = make_synthetic(rc_w, wavelet, 
                           snr_db=SNR_DB if ADD_NOISE else None, 
                           dt_sec=dt_s)
print(f'Seismic trace: {len(seismic_w)} samples, RMS = {np.sqrt(np.mean(seismic_w**2)):.6f}')


# ======================================================================
# ---
# ## 2. Build Forward Operator Matrices
#
# The forward model $\mathbf{s} = \frac{1}{2}\mathbf{W}\mathbf{D}\mathbf{L}$ requires two matrices:
#
# ### Derivative Matrix D
# Converts log-impedance to reflectivity: $r_i \approx \frac{1}{2}(L_{i+1} - L_i)$
#
# $$\mathbf{D} = \begin{pmatrix} -1 & 1 & 0 & \cdots \\ 0 & -1 & 1 & \cdots \\ \vdots & & \ddots & \ddots \end{pmatrix}$$
#
# ### Wavelet Convolution Matrix W
# Each column is the wavelet shifted by the appropriate time:
#
# $$\mathbf{W}_{ij} = w_{i-j}$$
#
# ### Combined: G = ½ W D
# ======================================================================

# %%  [Cell 5]
def build_derivative_matrix(n):
    """
    Build the (n-1) × n first-difference matrix D.
    D[i,:] has -1 at position i and +1 at position i+1.
    So D @ L gives the differences L[i+1] - L[i].
    """
    D = np.zeros((n, n))
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    # Last row: zero (no derivative for last sample)
    # This makes D square (n×n) for convenience
    return D


def build_wavelet_matrix(wavelet, n_trace):
    """
    Build the n_trace × n_trace wavelet convolution matrix W.
    Each column j contains the wavelet centered at row j.
    """
    n_wav = len(wavelet)
    half = n_wav // 2
    W = np.zeros((n_trace, n_trace))
    for j in range(n_trace):
        for k in range(n_wav):
            row = j + k - half
            if 0 <= row < n_trace:
                W[row, j] = wavelet[k]
    return W


# Build matrices
n = n_win
D = build_derivative_matrix(n)
W = build_wavelet_matrix(wavelet, n)
G = 0.5 * W @ D  # Combined forward operator

print(f'D: {D.shape}, W: {W.shape}, G: {G.shape}')

# Verify: G @ ln_ai_true should approximate seismic
synth_verify = G @ ln_ai_true_w
seismic_clean = make_synthetic(rc_w, wavelet, snr_db=None)
cc_verify = np.corrcoef(synth_verify, seismic_clean)[0, 1]
print(f'Forward model verification: CC(G@L_true, W*rc) = {cc_verify:.4f}')


# ======================================================================
# ---
# ## 3. Conjugate Gradient Inversion Engine
#
# The conjugate gradient (CG) method iteratively solves:
#
# $$\mathbf{G}^T\mathbf{G}\mathbf{L} + \lambda\mathbf{I}\mathbf{L} = \mathbf{G}^T\mathbf{s}$$
#
# Starting from the initial model $\mathbf{L}_0$, it searches along conjugate directions
# to minimize the objective function. At each iteration:
#
# 1. Compute residual: $\mathbf{r} = \mathbf{s} - \mathbf{G}\mathbf{L}$
# 2. Compute gradient: $\mathbf{g} = -\mathbf{G}^T\mathbf{r} + \lambda(\mathbf{L} - \mathbf{L}_0)$
# 3. Update search direction (CG formula)
# 4. Line search for optimal step size
# 5. Apply constraints (hard bounds or soft penalty)
# ======================================================================

# %%  [Cell 6]
class ModelBasedInversion:
    """
    Model-Based Post-Stack Seismic Inversion.
    
    Implements the STRATA-style constrained conjugate gradient algorithm.
    Solves for ln(impedance) by minimizing the misfit between
    observed seismic and forward-modeled synthetic.
    """
    
    def __init__(self, G, seismic, ln_ai_initial, 
                 prewhitening=0.01, model_weight=0.0,
                 max_change_pct=None):
        """
        Parameters
        ----------
        G : ndarray (n, n)
            Forward operator = 0.5 * W @ D
        seismic : ndarray (n,)
            Observed seismic trace.
        ln_ai_initial : ndarray (n,)
            Initial model: ln(AI) from LFM.
        prewhitening : float
            Regularization parameter λ.
        model_weight : float
            Soft constraint weight w (0=seismic only, 1=model only).
        max_change_pct : float or None
            Hard constraint: max % deviation from initial model.
            If None, no hard constraints (stochastic mode only).
        """
        self.G = G
        self.GT = G.T
        self.GTG = G.T @ G
        self.s = seismic.copy()
        self.L0 = ln_ai_initial.copy()
        self.n = len(seismic)
        self.lam = prewhitening
        self.w = model_weight
        self.max_change = None
        
        if max_change_pct is not None:
            # max_change in ln(impedance) units
            avg_ln = np.mean(self.L0)
            self.max_change = avg_ln * (max_change_pct / 100.0)
        
        # Normalize seismic and model for balanced objective function
        self.s_norm = np.sqrt(np.mean(self.s**2))
        self.L_norm = np.sqrt(np.mean(self.L0**2))
        
        # Pre-compute GT @ s
        self.GTs = self.GT @ self.s
        
    def objective(self, L):
        """Compute the objective function J."""
        residual = self.s - self.G @ L
        J_seis = np.sum(residual**2) / (self.s_norm**2 * self.n + 1e-30)
        J_model = np.sum((L - self.L0)**2) / (self.L_norm**2 * self.n + 1e-30)
        J = (1 - self.w) * J_seis + self.w * J_model
        return J, J_seis, J_model
    
    def gradient(self, L):
        """Compute the gradient of J with respect to L."""
        residual = self.s - self.G @ L
        # Seismic misfit gradient
        g_seis = -self.GT @ residual / (self.s_norm**2 * self.n + 1e-30)
        # Model constraint gradient
        g_model = (L - self.L0) / (self.L_norm**2 * self.n + 1e-30)
        # Regularization gradient
        g_reg = self.lam * (L - self.L0)
        
        g = (1 - self.w) * g_seis + self.w * g_model + g_reg
        return g
    
    def apply_hard_constraints(self, L):
        """Clip L to be within max_change of initial model."""
        if self.max_change is not None:
            L = np.clip(L, self.L0 - self.max_change, 
                        self.L0 + self.max_change)
        return L
    
    def solve(self, n_iter=10, verbose=True):
        """
        Solve using Conjugate Gradient with optional hard constraints.
        
        Returns
        -------
        L_final : ndarray
            Final ln(impedance) solution.
        history : dict
            Convergence history.
        """
        L = self.L0.copy()
        
        history = {
            'J': [], 'J_seis': [], 'J_model': [],
            'cc': [], 'rms_err': [],
            'L_snapshots': [],
        }
        
        # Initial state
        J, Js, Jm = self.objective(L)
        ai_current = np.exp(L)
        ai_truth = np.exp(self.L0)  # We'll update CC against truth externally
        
        # CG initialization
        g = self.gradient(L)
        d = -g.copy()  # Search direction
        
        for it in range(n_iter):
            # --- Line search: find optimal step α ---
            # For quadratic objective: α = (g·d) / (d·A·d) where A = GTG + regularization
            Gd = self.G @ d
            
            numerator = np.dot(g, d)
            denom_seis = (1 - self.w) * np.dot(Gd, Gd) / (self.s_norm**2 * self.n + 1e-30)
            denom_model = self.w * np.dot(d, d) / (self.L_norm**2 * self.n + 1e-30)
            denom_reg = self.lam * np.dot(d, d)
            denominator = denom_seis + denom_model + denom_reg
            
            if abs(denominator) < 1e-30:
                if verbose:
                    print(f'  Iter {it}: denominator near zero, stopping.')
                break
            
            alpha = -numerator / denominator
            
            # --- Update L ---
            L_new = L + alpha * d
            
            # Apply hard constraints
            L_new = self.apply_hard_constraints(L_new)
            
            # --- Conjugate gradient direction update ---
            g_new = self.gradient(L_new)
            
            # Polak-Ribière formula
            beta_num = np.dot(g_new, g_new - g)
            beta_den = np.dot(g, g)
            beta = max(0, beta_num / (beta_den + 1e-30))  # Reset if negative
            
            d = -g_new + beta * d
            
            L = L_new
            g = g_new
            
            # --- Record history ---
            J, Js, Jm = self.objective(L)
            synth_current = self.G @ L
            cc_synth = np.corrcoef(self.s, synth_current)[0, 1]
            
            history['J'].append(J)
            history['J_seis'].append(Js)
            history['J_model'].append(Jm)
            history['cc'].append(cc_synth)
            
            # Save snapshots at key iterations
            if it in [0, 1, 2, 4, 9, n_iter-1] or it == n_iter//2:
                history['L_snapshots'].append((it, L.copy()))
            
            if verbose and (it < 5 or it % 5 == 0 or it == n_iter - 1):
                print(f'  Iter {it:3d}: J={J:.6f}  J_seis={Js:.6f}  '
                      f'J_model={Jm:.6f}  CC(synth)={cc_synth:.4f}  α={alpha:.4e}')
        
        # Final synthetic
        history['synth_final'] = self.G @ L
        history['residual'] = self.s - history['synth_final']
        
        return L, history


# ======================================================================
# ---
# ## 4. Run Stochastic (Soft Constraint) Inversion
# ======================================================================

# %%  [Cell 7]
print('='*60)
print('STOCHASTIC (Soft Constraint) Model-Based Inversion')
print(f'  Model weight: {MODEL_WEIGHT}')
print(f'  Prewhitening: {PREWHITENING}')
print(f'  Iterations: {N_ITERATIONS}')
print('='*60)

inv_stoch = ModelBasedInversion(
    G, seismic_w, ln_ai_lfm_w,
    prewhitening=PREWHITENING,
    model_weight=MODEL_WEIGHT,
    max_change_pct=None  # No hard bounds
)

ln_ai_stoch, hist_stoch = inv_stoch.solve(N_ITERATIONS, verbose=True)
ai_stoch = np.exp(ln_ai_stoch)

# Metrics vs truth
valid_w = np.isfinite(ai_true_w) & np.isfinite(ai_stoch)
cc_stoch = np.corrcoef(ai_true_w[valid_w], ai_stoch[valid_w])[0, 1]
rms_stoch = np.sqrt(np.mean((ai_true_w[valid_w] - ai_stoch[valid_w])**2))
nrms_stoch = rms_stoch / np.std(ai_true_w[valid_w])

print(f'\nStochastic result vs truth:')
print(f'  CC = {cc_stoch:.4f}, NRMS = {nrms_stoch:.3f}')


# ======================================================================
# ---
# ## 5. Run Constrained (Hard Bounds) Inversion
# ======================================================================

# %%  [Cell 8]
print('='*60)
print('CONSTRAINED (Hard Bounds) Model-Based Inversion')
print(f'  Max impedance change: {MAX_IMPEDANCE_CHANGE_PCT}%')
print(f'  Prewhitening: {PREWHITENING}')
print(f'  Iterations: {N_ITERATIONS}')
print('='*60)

inv_constr = ModelBasedInversion(
    G, seismic_w, ln_ai_lfm_w,
    prewhitening=PREWHITENING,
    model_weight=0.0,  # No soft constraint — hard bounds instead
    max_change_pct=MAX_IMPEDANCE_CHANGE_PCT
)

ln_ai_constr, hist_constr = inv_constr.solve(N_ITERATIONS, verbose=True)
ai_constr = np.exp(ln_ai_constr)

# Metrics
valid_c = np.isfinite(ai_true_w) & np.isfinite(ai_constr)
cc_constr = np.corrcoef(ai_true_w[valid_c], ai_constr[valid_c])[0, 1]
rms_constr = np.sqrt(np.mean((ai_true_w[valid_c] - ai_constr[valid_c])**2))
nrms_constr = rms_constr / np.std(ai_true_w[valid_c])

print(f'\nConstrained result vs truth:')
print(f'  CC = {cc_constr:.4f}, NRMS = {nrms_constr:.3f}')


# ======================================================================
# ---
# ## 6. Convergence Diagnostics
# ======================================================================

# %%  [Cell 9]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Total objective
axes[0].semilogy(hist_stoch['J'], 'r-o', ms=3, label='Stochastic')
axes[0].semilogy(hist_constr['J'], 'b-s', ms=3, label='Constrained')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Objective J')
axes[0].set_title('Total Objective Function', fontweight='bold')
axes[0].legend()

# Seismic misfit
axes[1].semilogy(hist_stoch['J_seis'], 'r-o', ms=3, label='Stochastic')
axes[1].semilogy(hist_constr['J_seis'], 'b-s', ms=3, label='Constrained')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Seismic Misfit')
axes[1].set_title('Seismic Misfit J_seis', fontweight='bold')
axes[1].legend()

# Synthetic CC
axes[2].plot(hist_stoch['cc'], 'r-o', ms=3, label='Stochastic')
axes[2].plot(hist_constr['cc'], 'b-s', ms=3, label='Constrained')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('CC (synth vs seismic)')
axes[2].set_title('Synthetic-to-Seismic CC', fontweight='bold')
axes[2].legend()
axes[2].set_ylim(0, 1.05)

fig.suptitle('Convergence Diagnostics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 7. Iteration Snapshots — Watching the Model Evolve
#
# This shows how the impedance profile evolves from the smooth LFM initial guess
# toward the final detailed result — exactly like STRATA's iteration display.
# ======================================================================

# %%  [Cell 10]
# Use constrained result for snapshots (typically the preferred method)
snapshots = hist_constr['L_snapshots']

fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(18, 10), sharey=True)

# Initial model
axes[0].plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.4)
axes[0].plot(np.exp(ln_ai_lfm_w), twt_w, 'b-', lw=1.5)
axes[0].set_title('Initial\n(LFM)', fontweight='bold', color='blue')
axes[0].set_ylabel('TWT (ms)')
axes[0].invert_yaxis()

for i, (it_num, L_snap) in enumerate(snapshots):
    ax = axes[i + 1]
    ai_snap = np.exp(L_snap)
    cc_snap = np.corrcoef(ai_true_w[valid_w], ai_snap[valid_w])[0, 1]
    ax.plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.4)
    ax.plot(ai_snap, twt_w, 'r-', lw=0.8)
    ax.set_title(f'Iter {it_num}\nCC={cc_snap:.3f}', fontweight='bold', fontsize=10)
    ax.set_yticklabels([])

fig.suptitle('Constrained Inversion — Iteration Snapshots (black = well truth)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 8. Synthetic Trace Matching
#
# The key QC: does the synthetic from the inverted model match the input seismic?
# ======================================================================

# %%  [Cell 11]
fig, axes = plt.subplots(1, 4, figsize=(16, 10), sharey=True)

for idx, (label, hist, color) in enumerate([
    ('Stochastic', hist_stoch, 'red'),
    ('Constrained', hist_constr, 'blue'),
]):
    # Seismic vs synthetic
    ax = axes[idx * 2]
    ax.plot(seismic_w, twt_w, 'k-', lw=0.5, label='Seismic')
    ax.plot(hist['synth_final'], twt_w, '-', color=color, lw=0.8, 
            alpha=0.8, label='Synthetic')
    cc_s = np.corrcoef(seismic_w, hist['synth_final'])[0, 1]
    ax.set_title(f'{label}\nSynth CC = {cc_s:.3f}', fontweight='bold')
    if idx == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    ax.legend(fontsize=8)
    
    # Residual
    ax = axes[idx * 2 + 1]
    ax.plot(hist['residual'], twt_w, '-', color=color, lw=0.5)
    ax.fill_betweenx(twt_w, 0, hist['residual'], 
                      where=hist['residual'] > 0, alpha=0.2, color='red')
    ax.fill_betweenx(twt_w, 0, hist['residual'], 
                      where=hist['residual'] < 0, alpha=0.2, color='blue')
    res_rms = np.sqrt(np.mean(hist['residual']**2))
    ax.set_title(f'Residual\nRMS = {res_rms:.5f}', fontweight='bold')

fig.suptitle('Synthetic Trace Matching', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 9. Model Weight Sensitivity (Stochastic)
#
# The model weight controls the seismic-vs-model trade-off.
# Low weight → more seismic detail but potential instability.
# High weight → smoother, closer to LFM, but less seismic information.
# ======================================================================

# %%  [Cell 12]
weights_test = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
weight_ccs = []

fig, axes = plt.subplots(1, len(weights_test), figsize=(20, 9), sharey=True)

for i, w in enumerate(weights_test):
    inv_test = ModelBasedInversion(G, seismic_w, ln_ai_lfm_w,
                                   prewhitening=PREWHITENING,
                                   model_weight=w, max_change_pct=None)
    L_test, _ = inv_test.solve(N_ITERATIONS, verbose=False)
    ai_test = np.exp(L_test)
    cc_test = np.corrcoef(ai_true_w[valid_w], ai_test[valid_w])[0, 1]
    weight_ccs.append(cc_test)
    
    axes[i].plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.4)
    axes[i].plot(ai_test, twt_w, 'r-', lw=0.8)
    is_chosen = abs(w - MODEL_WEIGHT) < 0.01
    axes[i].set_title(f'w={w}\nCC={cc_test:.3f}', 
                       fontweight='bold', fontsize=10,
                       color='red' if is_chosen else 'black')
    if i == 0:
        axes[i].set_ylabel('TWT (ms)')
        axes[i].invert_yaxis()
    else:
        axes[i].set_yticklabels([])

fig.suptitle('Stochastic Inversion — Model Weight Sensitivity', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

best_w_idx = np.argmax(weight_ccs)
print(f'Best weight: w={weights_test[best_w_idx]} → CC={weight_ccs[best_w_idx]:.4f}')


# ======================================================================
# ---
# ## 10. Grand Comparison — All Methods
#
# STRATA-style summary comparing LFM, Recursive, Colored, Stochastic, and Constrained.
# ======================================================================

# %%  [Cell 13]
# Collect all results
methods = [
    ('Seismic', seismic_w, 'k', 'trace'),
    ('Well AI\n(Truth)', ai_true_w, 'k', 'impedance'),
    ('LFM', np.exp(ln_ai_lfm_w), 'blue', 'impedance'),
    ('Stochastic\nModel-Based', ai_stoch, 'red', 'impedance'),
    ('Constrained\nModel-Based', ai_constr, 'darkgreen', 'impedance'),
]

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 5, height_ratios=[2.5, 1], hspace=0.4, wspace=0.12)

cc_lfm_w = np.corrcoef(ai_true_w[valid_w], np.exp(ln_ai_lfm_w)[valid_w])[0, 1]
all_ccs = [None, None, cc_lfm_w, cc_stoch, cc_constr]
all_nrms = [None, None, None, nrms_stoch, nrms_constr]

for i, (label, data, color, dtype) in enumerate(methods):
    ax = fig.add_subplot(gs[0, i])
    if dtype == 'trace':
        ax.plot(data, twt_w, 'k-', lw=0.4)
        ax.fill_betweenx(twt_w, 0, data, where=data>0, alpha=0.25, color='red')
        ax.fill_betweenx(twt_w, 0, data, where=data<0, alpha=0.25, color='blue')
    else:
        ax.plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.3)
        ax.plot(data, twt_w, '-', color=color, lw=1.2 if i > 1 else 0.6)
    
    title = label
    if all_ccs[i] is not None:
        title += f'\nCC={all_ccs[i]:.3f}'
    ax.set_title(title, fontweight='bold', color=color if i > 1 else 'black', fontsize=11)
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    else:
        ax.set_yticklabels([])

# --- Bottom row: cross-plot, spectra, metrics table ---

# Cross-plot
ax = fig.add_subplot(gs[1, 0:2])
for label, data, color, marker in [
    ('Stochastic', ai_stoch, 'red', '.'),
    ('Constrained', ai_constr, 'darkgreen', '.'),
]:
    ax.scatter(ai_true_w[valid_w], data[valid_w], c=color, s=3, alpha=0.3, label=label)
lims = [np.nanmin(ai_true_w)*0.9, np.nanmax(ai_true_w)*1.05]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Well AI (truth)')
ax.set_ylabel('Inverted AI')
ax.set_title('Cross-plot', fontweight='bold')
ax.legend(fontsize=8)

# Spectra
ax = fig.add_subplot(gs[1, 2:4])
for label, data, color, ls in [
    ('Well AI', ai_true_w, 'black', '-'),
    ('LFM', np.exp(ln_ai_lfm_w), 'blue', '--'),
    ('Stochastic', ai_stoch, 'red', '-'),
    ('Constrained', ai_constr, 'darkgreen', '-'),
]:
    d = data.copy()
    d[np.isnan(d)] = np.nanmean(d)
    fw, aw = welch(d - np.mean(d), fs=fs, nperseg=min(len(d), 256))
    ax.plot(fw, np.sqrt(aw), color=color, ls=ls, lw=1.5 if label == 'Well AI' else 1,
            label=label)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.set_title('Amplitude Spectra', fontweight='bold')
ax.set_xlim(0, 100)
ax.legend(fontsize=8)

# Metrics table
ax = fig.add_subplot(gs[1, 4])
ax.axis('off')
table_data = [
    ['Method', 'CC', 'NRMS'],
    ['LFM only', f'{cc_lfm_w:.3f}', '—'],
    ['Stochastic', f'{cc_stoch:.3f}', f'{nrms_stoch:.3f}'],
    ['Constrained', f'{cc_constr:.3f}', f'{nrms_constr:.3f}'],
]
# Add Phase 2 if available
if has_p2:
    ai_rec_w = ai_rec[idx_start:idx_end]
    ai_ci_w = ai_ci[idx_start:idx_end]
    v2 = valid_w & np.isfinite(ai_rec_w) & np.isfinite(ai_ci_w)
    cc_rec_w = np.corrcoef(ai_true_w[v2], ai_rec_w[v2])[0, 1]
    cc_ci_w = np.corrcoef(ai_true_w[v2], ai_ci_w[v2])[0, 1]
    table_data.insert(2, ['Recursive', f'{cc_rec_w:.3f}', '—'])
    table_data.insert(3, ['Colored', f'{cc_ci_w:.3f}', '—'])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for j in range(3):
    table[0, j].set_text_props(fontweight='bold')
    table[0, j].set_facecolor('#d4e6f1')
# Highlight best
all_table_ccs = [float(row[1]) for row in table_data[1:] if row[1] != '—']
best_val = max(all_table_ccs)
for r in range(1, len(table_data)):
    try:
        if float(table_data[r][1]) == best_val:
            for j in range(3):
                table[r, j].set_facecolor('#d5f5e3')
    except:
        pass
ax.set_title('All Methods', fontweight='bold')

fig.suptitle(
    'OpenSI Phase 3 — Model-Based Inversion Summary\n'
    f'Well 15/9-F-11 T2 | {N_ITERATIONS} iterations | '
    f'λ={PREWHITENING} | SNR={SNR_DB} dB',
    fontsize=14, fontweight='bold', y=1.02
)
plt.savefig(f'{OUTPUT_PREFIX}_Summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 11. Export Results
# ======================================================================

# %%  [Cell 14]
# Save windowed results
df_out = pd.DataFrame({
    'TWT_s': twt_s[idx_start:idx_end],
    'TWT_ms': twt_w,
    'Seismic': seismic_w,
    'AI_Truth': ai_true_w,
    'AI_LFM': np.exp(ln_ai_lfm_w),
    'AI_Stochastic': ai_stoch,
    'AI_Constrained': ai_constr,
    'LN_AI_Stochastic': ln_ai_stoch,
    'LN_AI_Constrained': ln_ai_constr,
    'Synth_Stochastic': hist_stoch['synth_final'],
    'Synth_Constrained': hist_constr['synth_final'],
    'Residual_Stochastic': hist_stoch['residual'],
    'Residual_Constrained': hist_constr['residual'],
})
out_csv = f'{OUTPUT_PREFIX}_inversion_results.csv'
df_out.to_csv(out_csv, index=False, float_format='%.6f')
print(f'Saved: {out_csv}')

# Summary
print(f'\n{"="*60}')
print(f'Phase 3 COMPLETE — Model-Based Inversion')
print(f'{"="*60}')
print(f'  Stochastic:  CC = {cc_stoch:.4f}  NRMS = {nrms_stoch:.3f}')
print(f'  Constrained: CC = {cc_constr:.4f}  NRMS = {nrms_constr:.3f}')
if has_p2:
    print(f'  (Phase 2 baselines: Recursive CC={cc_rec_w:.3f}, Colored CC={cc_ci_w:.3f})')
print(f'\nOutputs:')
print(f'  {out_csv}')
print(f'  {OUTPUT_PREFIX}_Summary.png')
print(f'\nReady for Phase 4 (Sparse Spike Inversion)')

