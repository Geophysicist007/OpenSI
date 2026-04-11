#!/usr/bin/env python3
"""
OpenSI — Phase 4: Sparse Spike Inversion

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase4_Sparse_Spike_Inversion.py
"""


# ======================================================================
# # OpenSI — Phase 4: Sparse Spike Inversion
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Model the earth with the minimum number of reflection coefficients.
# **Reference:** Oldenburg et al. (1983), Chi et al. (1984), Sacchi & Ulrych (1995)
#
# ---
#
# ## Concept
#
# Sparse spike inversion differs fundamentally from model-based inversion:
#
# - **Model-based** (Phase 3): starts from an initial model and perturbs every sample
# - **Sparse spike**: finds the **fewest reflectors** that explain the seismic data
#
# The reflectivity is modeled as a few large spikes on a quiet background. This produces
# **blocky impedance** — each spike defines a layer boundary, and impedance is constant
# within each layer.
#
# ### Two Methods
#
# **1. Linear Programming Sparse Spike (LPSS)**
# Minimizes the L1-norm (sum of absolute values) of reflectivity:
# $$\min \sum |r_i| \quad \text{subject to} \quad \|\mathbf{s} - \mathbf{W}\mathbf{r}\|_2 < \epsilon$$
# L1 minimization naturally produces sparse solutions (most $r_i = 0$).
#
# **2. Maximum Likelihood Sparse Spike (MLSS)**
# Iteratively adds spikes one at a time at the position that best reduces the residual,
# until the residual is small enough or spike amplitudes fall below a threshold.
#
# Both methods are best for areas with **few major reflectors** and less geological knowledge.
# ======================================================================


# ======================================================================
# ---
# ## User Parameters
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Input files ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'
PHASE2_CSV       = 'Phase2_inversion_results.csv'
PHASE3_CSV       = 'Phase3_inversion_results.csv'
WAVELET_FILE     = 'final_wavelet.csv'

# --- Analysis window ---
T_WIN_START      = 2.60
T_WIN_END        = 3.26

# --- LPSS parameters ---
LPSS_SPARSENESS  = 0.3       # 0 to 1. Higher = sparser. 0.2-0.5 typical
LPSS_MAX_ITER    = 1000      # FISTA iterations

# --- MLSS parameters ---
MLSS_THRESHOLD   = 0.10      # Spike detection threshold (fraction of avg spike amplitude)
MLSS_MAX_SPIKES  = 80        # Maximum number of spikes to add

# --- LFM merge ---
LFM_MERGE_HZ     = 10.0

# --- Noise ---
ADD_NOISE        = True
SNR_DB           = 20

# --- Output ---
OUTPUT_PREFIX    = 'Phase4'


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

# Load Phase 1 data
df_ai = pd.read_csv(AI_FULLBAND_CSV)
df_lfm = pd.read_csv(LFM_CSV)

twt_s = df_ai['TWT_s'].values
twt_ms = df_ai['TWT_ms'].values
dt_s = np.round(np.median(np.diff(twt_s)), 6)
fs = 1.0 / dt_s

ai_full = df_ai['AI'].values
rc_p = df_ai['RC_P'].values
ln_ai_full = df_ai['LN_AI'].values
ai_lfm = df_lfm['AI_LFM'].values
ln_ai_lfm = df_lfm['LN_AI_LFM'].values

# Load wavelet
wav_df = pd.read_csv(WAVELET_FILE)
wavelet = wav_df.iloc[:, 1].values if len(wav_df.columns) >= 2 else wav_df.iloc[:, 0].values
wavelet = wavelet / np.max(np.abs(wavelet))

# Load prior results for comparison
prior_results = {}
for label, path, col in [
    ('Colored', PHASE2_CSV, 'AI_Colored'),
    ('Stochastic', PHASE3_CSV, 'AI_Stochastic'),
    ('Constrained', PHASE3_CSV, 'AI_Constrained'),
]:
    try:
        prior_results[label] = pd.read_csv(path)[col].values
    except:
        pass

print(f'Data: {len(twt_s)} samples, dt={dt_s*1000:.1f} ms')
print(f'Wavelet: {len(wavelet)} samples')

# %%  [Cell 3]
# --- Window extraction ---
win_mask = (twt_s >= T_WIN_START) & (twt_s <= T_WIN_END)
idx_s = np.argmax(win_mask)
idx_e = len(win_mask) - np.argmax(win_mask[::-1])
n_win = idx_e - idx_s

twt_w = twt_ms[idx_s:idx_e]
rc_w = rc_p[idx_s:idx_e].copy()
rc_w[np.isnan(rc_w)] = 0.0
ai_true_w = ai_full[idx_s:idx_e].copy()
ln_ai_lfm_w = ln_ai_lfm[idx_s:idx_e].copy()
ai_lfm_w = ai_lfm[idx_s:idx_e].copy()

# Fill NaNs
for arr in [ln_ai_lfm_w, ai_true_w, ai_lfm_w]:
    nans = np.isnan(arr)
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

# Generate synthetic seismic
def make_synthetic(rc, wav, snr_db=None):
    rc_c = rc.copy(); rc_c[np.isnan(rc_c)] = 0.0
    s = fftconvolve(rc_c, wav, mode='same')
    if snr_db is not None:
        sp = np.mean(s**2)
        np_ = sp / (10**(snr_db/10))
        n = np.random.randn(len(s)) * np.sqrt(np_)
        nyq = 1/(2*dt_s)
        sos = butter(4, [5/nyq, min(80/nyq, 0.99)], btype='band', output='sos')
        n = sosfiltfilt(sos, n)
        n *= np.sqrt(np_ / np.mean(n**2))
        s += n
    return s

seismic_w = make_synthetic(rc_w, wavelet, SNR_DB if ADD_NOISE else None)
print(f'Window: {n_win} samples, seismic RMS = {np.sqrt(np.mean(seismic_w**2)):.6f}')


# ======================================================================
# ---
# ## 1. Build Wavelet Convolution Matrix
#
# Both sparse spike methods need the wavelet as a matrix $\mathbf{W}$ such that
# $\mathbf{s} = \mathbf{W} \mathbf{r}$.
# ======================================================================

# %%  [Cell 4]
def build_wavelet_matrix(wav, n_trace):
    """Build convolution matrix W such that s = W @ r."""
    n_wav = len(wav)
    half = n_wav // 2
    W = np.zeros((n_trace, n_trace))
    for j in range(n_trace):
        for k in range(n_wav):
            row = j + k - half
            if 0 <= row < n_trace:
                W[row, j] = wav[k]
    return W

W = build_wavelet_matrix(wavelet, n_win)
print(f'Wavelet matrix: {W.shape}')

# Verify
synth_check = W @ rc_w
synth_clean = make_synthetic(rc_w, wavelet, snr_db=None)
cc_check = np.corrcoef(synth_check, synth_clean)[0, 1]
print(f'W@r verification: CC = {cc_check:.4f}')


# ======================================================================
# ---
# ## 2. Linear Programming Sparse Spike (LPSS)
#
# ### Algorithm (Oldenburg et al., 1983)
#
# We solve:
# $$\min_\mathbf{r} \sum_i |r_i| \quad \text{s.t.} \quad \mathbf{W}\mathbf{r} \approx \mathbf{s}$$
#
# The L1 norm naturally promotes sparsity — most $r_i$ go to zero.
#
# To handle the absolute value, we split $r_i = r_i^+ - r_i^-$ where both $\geq 0$:
# $$\min \sum (r_i^+ + r_i^-) \quad \text{s.t.} \quad \mathbf{W}(\mathbf{r}^+ - \mathbf{r}^-) = \mathbf{s}, \quad r_i^+, r_i^- \geq 0$$
#
# We process in overlapping windows for stability (Oldenburg et al., 1983).
# ======================================================================

# %%  [Cell 5]
def lpss_inversion(seismic, W_mat, sparseness=1.0, max_iter=1000):
    """
    Linear Programming Sparse Spike Inversion via FISTA.
    
    Solves Basis Pursuit Denoising:
        min  0.5 * ||s - W r||^2  +  lambda * ||r||_1
    
    using Fast Iterative Shrinkage-Thresholding (FISTA).
    The L1 penalty naturally produces sparse reflectivity.
    
    Parameters
    ----------
    seismic : array (n,)
        Input seismic trace.
    W_mat : array (n, n)
        Wavelet convolution matrix.
    sparseness : float
        0 to 1. Controls lambda: higher = sparser (fewer spikes).
    max_iter : int
        Maximum FISTA iterations.
    
    Returns
    -------
    rc : array (n,)
        Sparse reflectivity series.
    info : dict
        Diagnostic information.
    """
    n = len(seismic)
    s = seismic.copy()
    
    # Pre-compute matrices
    WTW = W_mat.T @ W_mat
    WTs = W_mat.T @ s
    
    # Step size: 1 / spectral radius of WTW
    L_lip = np.linalg.norm(WTW, ord=2)
    step = 1.0 / L_lip
    
    # Lambda: scale sparseness to data magnitude
    # At sparseness=1, lambda = max(|W^T s|) * step (kills all but largest)
    # At sparseness=0, lambda = 0 (pure least-squares)
    lam = sparseness * np.max(np.abs(WTs)) * step
    
    # FISTA (with Nesterov acceleration)
    r = np.zeros(n)
    r_prev = np.zeros(n)
    t_k = 1.0
    
    converged_at = max_iter
    for k in range(max_iter):
        # Momentum term
        y = r + ((t_k - 1) / (t_k + 2)) * (r - r_prev)
        
        # Gradient step
        grad = WTW @ y - WTs
        z = y - step * grad
        
        # Soft thresholding (proximal operator of L1)
        r_new = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
        
        # Check convergence
        if k > 10 and np.linalg.norm(r_new - r) < 1e-7 * (np.linalg.norm(r_new) + 1e-30):
            converged_at = k
            break
        
        # FISTA update
        t_k_new = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
        r_prev = r.copy()
        r = r_new
        t_k = t_k_new
    
    rc = r
    synth = W_mat @ rc
    residual = s - synth
    
    # Count meaningful spikes (> 1% of max amplitude)
    rc_max = np.max(np.abs(rc)) if np.max(np.abs(rc)) > 0 else 1
    n_nonzero = np.sum(np.abs(rc) > 0.01 * rc_max)
    
    info = {
        'synth': synth,
        'residual': residual,
        'n_spikes': n_nonzero,
        'converged_at': converged_at,
        'lambda': lam,
    }
    
    return rc, info


# --- Run LPSS on full trace ---
print('Running LPSS (FISTA) inversion...')
rc_lpss, lpss_info = lpss_inversion(seismic_w, W, LPSS_SPARSENESS, max_iter=LPSS_MAX_ITER)

synth_lpss = lpss_info['synth']
cc_synth_lpss = np.corrcoef(seismic_w, synth_lpss)[0, 1]
n_spikes_lpss = lpss_info['n_spikes']

print(f'  Converged at iter: {lpss_info["converged_at"]}')
print(f'  Lambda: {lpss_info["lambda"]:.6f}')
print(f'  Spikes (>1% of max): {n_spikes_lpss}')
print(f'  Synth CC: {cc_synth_lpss:.4f}')
print(f'  RC range: {rc_lpss.min():.4f} to {rc_lpss.max():.4f}')


# ======================================================================
# ---
# ## 3. Maximum Likelihood Sparse Spike (MLSS)
#
# ### Algorithm (Chi et al., 1984)
#
# 1. Start with zero reflectivity and the seismic trace as residual
# 2. Cross-correlate residual with wavelet → find the position of maximum correlation
# 3. Add a spike at that position with optimal amplitude
# 4. Update residual: $\mathbf{r}_{new} = \mathbf{s} - \mathbf{W} \mathbf{r}$
# 5. Repeat until spike amplitude falls below threshold × average amplitude
# ======================================================================

# %%  [Cell 6]
def mlss_inversion(seismic, wav, threshold=0.1, max_spikes=100):
    """
    Maximum Likelihood Sparse Spike Inversion.
    
    Iteratively adds spikes using the single-most-likely-addition algorithm.
    
    Parameters
    ----------
    seismic : array (n,)
        Input seismic trace.
    wav : array (m,)
        Seismic wavelet.
    threshold : float
        Stop when new spike amplitude < threshold * average spike amplitude.
    max_spikes : int
        Maximum number of spikes to add.
    
    Returns
    -------
    rc : array (n,)
        Sparse reflectivity.
    info : dict
        Diagnostic information.
    """
    n = len(seismic)
    m = len(wav)
    half = m // 2
    
    rc = np.zeros(n)
    residual = seismic.copy()
    
    # Wavelet autocorrelation at zero lag (for amplitude calculation)
    wav_energy = np.sum(wav**2)
    
    spike_history = []  # (position, amplitude, residual_energy)
    
    for i_spike in range(max_spikes):
        # Cross-correlate residual with wavelet to find best spike position
        xcorr = fftconvolve(residual, wav[::-1], mode='same')
        
        # Find position of maximum absolute cross-correlation
        pos = np.argmax(np.abs(xcorr))
        
        # Optimal spike amplitude at this position
        # amplitude = xcorr[pos] / wav_energy
        amplitude = xcorr[pos] / wav_energy
        
        # Check threshold
        if len(spike_history) > 0:
            avg_amp = np.mean([abs(sh[1]) for sh in spike_history])
            if abs(amplitude) < threshold * avg_amp:
                break
        
        # Add spike
        rc[pos] += amplitude
        
        # Update residual: subtract the wavelet centered at pos
        for k in range(m):
            row = pos + k - half
            if 0 <= row < n:
                residual[row] -= amplitude * wav[k]
        
        res_energy = np.sqrt(np.mean(residual**2))
        spike_history.append((pos, amplitude, res_energy))
    
    synth = seismic - residual
    n_final = len(spike_history)
    
    info = {
        'synth': synth,
        'residual': residual,
        'n_spikes': n_final,
        'spike_history': spike_history,
        'spike_positions': [sh[0] for sh in spike_history],
        'spike_amplitudes': [sh[1] for sh in spike_history],
        'residual_history': [sh[2] for sh in spike_history],
    }
    
    return rc, info


# --- Run MLSS ---
print('Running MLSS inversion...')
rc_mlss, mlss_info = mlss_inversion(seismic_w, wavelet, MLSS_THRESHOLD, MLSS_MAX_SPIKES)

synth_mlss = mlss_info['synth']
cc_synth_mlss = np.corrcoef(seismic_w, synth_mlss)[0, 1]

print(f'  Spikes found: {mlss_info["n_spikes"]}')
print(f'  Synth CC: {cc_synth_mlss:.4f}')
print(f'  Final residual RMS: {mlss_info["residual_history"][-1]:.6f}')


# ======================================================================
# ---
# ## 4. Convert Reflectivity → Impedance
#
# For both methods:
# 1. Integrate sparse reflectivity to get relative (blocky) impedance
# 2. Merge with LFM to get absolute impedance
# ======================================================================

# %%  [Cell 7]
def rc_to_impedance(rc, ai_lfm, dt_sec, merge_hz, filter_order=8):
    """
    Convert reflectivity to absolute impedance via recursion + LFM merge.
    """
    # Recursive integration: I_j = I_{j-1} * (1+r)/(1-r)
    rc_clipped = np.clip(rc, -0.5, 0.5)
    n = len(rc)
    ai_rel = np.ones(n)
    for j in range(1, n):
        ai_rel[j] = ai_rel[j-1] * (1 + rc_clipped[j]) / (1 - rc_clipped[j])
    
    # Merge with LFM in log domain
    ln_rel = np.log(ai_rel)
    ln_lfm = np.log(ai_lfm)
    
    nyq = 1 / (2 * dt_sec)
    sos_hp = butter(filter_order, merge_hz, btype='high', fs=1/dt_sec, output='sos')
    sos_lp = butter(filter_order, merge_hz, btype='low', fs=1/dt_sec, output='sos')
    
    ln_rel_hp = sosfiltfilt(sos_hp, ln_rel)
    ln_lfm_lp = sosfiltfilt(sos_lp, ln_lfm)
    
    ln_merged = ln_lfm_lp + ln_rel_hp
    ai_abs = np.exp(ln_merged)
    
    return ai_abs, ai_rel


# LPSS impedance
ai_lpss, ai_lpss_rel = rc_to_impedance(rc_lpss, ai_lfm_w, dt_s, LFM_MERGE_HZ)

# MLSS impedance
ai_mlss, ai_mlss_rel = rc_to_impedance(rc_mlss, ai_lfm_w, dt_s, LFM_MERGE_HZ)

# Metrics
valid = np.isfinite(ai_true_w) & np.isfinite(ai_lpss) & np.isfinite(ai_mlss)

cc_lpss = np.corrcoef(ai_true_w[valid], ai_lpss[valid])[0, 1]
cc_mlss = np.corrcoef(ai_true_w[valid], ai_mlss[valid])[0, 1]
nrms_lpss = np.sqrt(np.mean((ai_true_w[valid]-ai_lpss[valid])**2)) / np.std(ai_true_w[valid])
nrms_mlss = np.sqrt(np.mean((ai_true_w[valid]-ai_mlss[valid])**2)) / np.std(ai_true_w[valid])

print(f'LPSS:  CC = {cc_lpss:.4f}, NRMS = {nrms_lpss:.3f}, Spikes = {n_spikes_lpss}')
print(f'MLSS:  CC = {cc_mlss:.4f}, NRMS = {nrms_mlss:.3f}, Spikes = {mlss_info["n_spikes"]}')


# ======================================================================
# ---
# ## 5. Sparse Spike Diagnostics
# ======================================================================

# %%  [Cell 8]
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 4, height_ratios=[2, 1], hspace=0.35, wspace=0.2)

# --- Row 1: Reflectivity and Impedance ---

# LPSS reflectivity
ax = fig.add_subplot(gs[0, 0])
ax.plot(rc_w, twt_w, 'k-', lw=0.3, alpha=0.3, label='True RC')
ax.stem(rc_lpss, twt_w, linefmt='r-', markerfmt='r.', basefmt='', label='LPSS')
ax.set_title(f'LPSS Reflectivity\n{n_spikes_lpss} spikes', fontweight='bold', color='red')
ax.set_ylabel('TWT (ms)')
ax.set_xlabel('RC')
ax.invert_yaxis()
ax.legend(fontsize=8)

# LPSS impedance
ax = fig.add_subplot(gs[0, 1], sharey=fig.axes[0])
ax.plot(ai_true_w, twt_w, 'k-', lw=0.4, alpha=0.4, label='Truth')
ax.plot(ai_lpss, twt_w, 'r-', lw=1.2, label='LPSS')
ax.set_title(f'LPSS Impedance\nCC = {cc_lpss:.3f}', fontweight='bold', color='red')
ax.set_xlabel('AI')
ax.legend(fontsize=8)
ax.set_yticklabels([])

# MLSS reflectivity
ax = fig.add_subplot(gs[0, 2], sharey=fig.axes[0])
ax.plot(rc_w, twt_w, 'k-', lw=0.3, alpha=0.3, label='True RC')
ax.stem(rc_mlss, twt_w, linefmt='b-', markerfmt='b.', basefmt='', label='MLSS')
ax.set_title(f'MLSS Reflectivity\n{mlss_info["n_spikes"]} spikes', 
             fontweight='bold', color='blue')
ax.set_xlabel('RC')
ax.legend(fontsize=8)
ax.set_yticklabels([])

# MLSS impedance
ax = fig.add_subplot(gs[0, 3], sharey=fig.axes[0])
ax.plot(ai_true_w, twt_w, 'k-', lw=0.4, alpha=0.4, label='Truth')
ax.plot(ai_mlss, twt_w, 'b-', lw=1.2, label='MLSS')
ax.set_title(f'MLSS Impedance\nCC = {cc_mlss:.3f}', fontweight='bold', color='blue')
ax.set_xlabel('AI')
ax.legend(fontsize=8)
ax.set_yticklabels([])

# --- Row 2: MLSS convergence + synthetic matching ---

# MLSS residual energy vs spike number
ax = fig.add_subplot(gs[1, 0])
ax.plot(mlss_info['residual_history'], 'b-o', ms=3)
ax.set_xlabel('Spike number')
ax.set_ylabel('Residual RMS')
ax.set_title('MLSS Convergence', fontweight='bold')

# Spike amplitude distribution
ax = fig.add_subplot(gs[1, 1])
amps = np.array(mlss_info['spike_amplitudes'])
ax.bar(range(len(amps)), np.abs(amps), color='steelblue', alpha=0.7)
ax.axhline(np.mean(np.abs(amps)) * MLSS_THRESHOLD, color='red', ls='--', 
           label=f'Threshold ({MLSS_THRESHOLD}×avg)')
ax.set_xlabel('Spike index (by addition order)')
ax.set_ylabel('|Amplitude|')
ax.set_title('MLSS Spike Amplitudes', fontweight='bold')
ax.legend(fontsize=8)

# Synthetic matching: LPSS
ax = fig.add_subplot(gs[1, 2])
ax.plot(seismic_w, twt_w, 'k-', lw=0.5, label='Seismic')
ax.plot(synth_lpss, twt_w, 'r-', lw=0.5, alpha=0.7, label='LPSS synth')
ax.set_title(f'LPSS Synth Match\nCC={cc_synth_lpss:.3f}', fontweight='bold')
ax.set_ylabel('TWT (ms)')
ax.invert_yaxis()
ax.legend(fontsize=8)

# Synthetic matching: MLSS
ax = fig.add_subplot(gs[1, 3])
ax.plot(seismic_w, twt_w, 'k-', lw=0.5, label='Seismic')
ax.plot(synth_mlss, twt_w, 'b-', lw=0.5, alpha=0.7, label='MLSS synth')
ax.set_title(f'MLSS Synth Match\nCC={cc_synth_mlss:.3f}', fontweight='bold')
ax.invert_yaxis()
ax.legend(fontsize=8)

fig.suptitle('Sparse Spike Inversion — LPSS vs MLSS Diagnostics',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 6. MLSS Sparseness Sensitivity
# ======================================================================

# %%  [Cell 9]
thresholds_test = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]

fig, axes = plt.subplots(1, len(thresholds_test), figsize=(20, 9), sharey=True)

for i, thr in enumerate(thresholds_test):
    rc_t, info_t = mlss_inversion(seismic_w, wavelet, thr, MLSS_MAX_SPIKES)
    ai_t, _ = rc_to_impedance(rc_t, ai_lfm_w, dt_s, LFM_MERGE_HZ)
    v = valid & np.isfinite(ai_t)
    cc_t = np.corrcoef(ai_true_w[v], ai_t[v])[0, 1]
    
    axes[i].plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.3)
    axes[i].plot(ai_t, twt_w, 'b-', lw=0.8)
    is_chosen = abs(thr - MLSS_THRESHOLD) < 0.001
    axes[i].set_title(f'thr={thr}\n{info_t["n_spikes"]} spikes\nCC={cc_t:.3f}',
                       fontsize=10, fontweight='bold',
                       color='blue' if is_chosen else 'black')
    if i == 0:
        axes[i].set_ylabel('TWT (ms)')
        axes[i].invert_yaxis()
    else:
        axes[i].set_yticklabels([])

fig.suptitle('MLSS — Spike Threshold Sensitivity (fewer spikes = more blocky)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 7. Grand Comparison — All Methods (Phases 1–4)
# ======================================================================

# %%  [Cell 10]
# Collect all methods
all_methods = [
    ('Seismic', seismic_w, 'k', 'trace'),
    ('Well AI\n(Truth)', ai_true_w, 'k', 'imp'),
    ('LFM', ai_lfm_w, 'blue', 'imp'),
    ('LPSS', ai_lpss, 'red', 'imp'),
    ('MLSS', ai_mlss, 'darkorange', 'imp'),
]

# Add Phase 3 if available
if 'Constrained' in prior_results:
    # Phase 3 results are windowed differently, need to align
    try:
        df_p3 = pd.read_csv(PHASE3_CSV)
        p3_twt = df_p3['TWT_ms'].values
        # Find overlap
        p3_mask = (p3_twt >= twt_w[0]) & (p3_twt <= twt_w[-1])
        if p3_mask.sum() > 0:
            ai_p3_aligned = np.interp(twt_w, p3_twt, df_p3['AI_Constrained'].values)
            all_methods.insert(3, ('Model-Based\n(Phase 3)', ai_p3_aligned, 'darkgreen', 'imp'))
    except:
        pass

n_methods = len(all_methods)
fig = plt.figure(figsize=(4*n_methods, 14))
gs = GridSpec(2, n_methods, height_ratios=[2.5, 1], hspace=0.35, wspace=0.1)

all_ccs = []
for i, (label, data, color, dtype) in enumerate(all_methods):
    ax = fig.add_subplot(gs[0, i])
    if dtype == 'trace':
        ax.plot(data, twt_w, 'k-', lw=0.4)
        ax.fill_betweenx(twt_w, 0, data, where=data>0, alpha=0.25, color='red')
        ax.fill_betweenx(twt_w, 0, data, where=data<0, alpha=0.25, color='blue')
        all_ccs.append(None)
    else:
        ax.plot(ai_true_w, twt_w, 'k-', lw=0.3, alpha=0.3)
        ax.plot(data, twt_w, '-', color=color, lw=1.2 if 'Truth' not in label else 0.5)
        if 'Truth' not in label:
            v = valid & np.isfinite(data)
            cc = np.corrcoef(ai_true_w[v], data[v])[0, 1]
            all_ccs.append(cc)
            label += f'\nCC={cc:.3f}'
        else:
            all_ccs.append(None)
    
    ax.set_title(label, fontweight='bold', color=color if i > 1 else 'black', fontsize=10)
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    else:
        ax.set_yticklabels([])

# Bottom: cross-plot + spectra + table
ax = fig.add_subplot(gs[1, 0:n_methods//2])
for label, data, color, dtype in all_methods[2:]:
    if dtype == 'imp' and 'Truth' not in label:
        v = valid & np.isfinite(data)
        clean_label = label.split('\n')[0]
        ax.scatter(ai_true_w[v], data[v], c=color, s=2, alpha=0.3, label=clean_label)
lims = [np.nanmin(ai_true_w)*0.9, np.nanmax(ai_true_w)*1.05]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Well AI')
ax.set_ylabel('Inverted AI')
ax.set_title('Cross-plot', fontweight='bold')
ax.legend(fontsize=7, ncol=2)

ax = fig.add_subplot(gs[1, n_methods//2:])
for label, data, color, dtype in all_methods[1:]:
    if dtype == 'imp':
        d = data.copy()
        d[np.isnan(d)] = np.nanmean(d)
        fw, aw = welch(d - np.mean(d), fs=fs, nperseg=min(len(d), 256))
        clean_label = label.split('\n')[0]
        ax.plot(fw, np.sqrt(aw), color=color, lw=1.5 if 'Truth' in label else 1,
                label=clean_label)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.set_title('Spectra', fontweight='bold')
ax.set_xlim(0, 100)
ax.legend(fontsize=7, ncol=2)

fig.suptitle(
    'OpenSI Phase 4 — Sparse Spike Inversion + All Methods Comparison\n'
    f'Well 15/9-F-11 T2 | SNR={SNR_DB} dB',
    fontsize=14, fontweight='bold', y=1.02
)
plt.savefig(f'{OUTPUT_PREFIX}_Summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 8. Export Results
# ======================================================================

# %%  [Cell 11]
df_out = pd.DataFrame({
    'TWT_s': twt_s[idx_s:idx_e],
    'TWT_ms': twt_w,
    'Seismic': seismic_w,
    'AI_Truth': ai_true_w,
    'AI_LFM': ai_lfm_w,
    'RC_LPSS': rc_lpss,
    'RC_MLSS': rc_mlss,
    'AI_LPSS': ai_lpss,
    'AI_MLSS': ai_mlss,
    'Synth_LPSS': synth_lpss,
    'Synth_MLSS': synth_mlss,
})
out_csv = f'{OUTPUT_PREFIX}_inversion_results.csv'
df_out.to_csv(out_csv, index=False, float_format='%.6f')
print(f'Saved: {out_csv}')

print(f'\n{"="*60}')
print(f'Phase 4 COMPLETE — Sparse Spike Inversion')
print(f'{"="*60}')
print(f'  LPSS:  CC = {cc_lpss:.4f}  NRMS = {nrms_lpss:.3f}  Spikes = {n_spikes_lpss}')
print(f'  MLSS:  CC = {cc_mlss:.4f}  NRMS = {nrms_mlss:.3f}  Spikes = {mlss_info["n_spikes"]}')
print(f'\nOutputs:')
print(f'  {out_csv}')
print(f'  {OUTPUT_PREFIX}_Summary.png')
print(f'\nReady for Phase 5 (Simultaneous Pre-Stack Inversion)')

