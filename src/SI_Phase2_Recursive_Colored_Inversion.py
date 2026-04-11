#!/usr/bin/env python3
"""
OpenSI — Phase 2: Recursive (Bandlimited) & Colored Inversion

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase2_Recursive_Colored_Inversion.py
"""


# ======================================================================
# # OpenSI — Phase 2: Recursive (Bandlimited) & Colored Inversion
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Implement the two simplest post-stack inversion methods as baselines.
# **Reference:** Lindseth (1979), Lancaster & Whitcombe (2000), Hampson & Russell (1991)
#
# ---
#
# ## Method 1: Recursive (Bandlimited) Inversion
#
# The classical trace-based approach (Lindseth, 1979):
# 1. Treat the seismic trace as an approximation of reflectivity
# 2. Recursively convert reflectivity → impedance: $I_{j} = I_{j-1} \cdot \frac{1 + r_j}{1 - r_j}$
# 3. Add the low-frequency model to restore the missing trend
#
# **Limitation:** Ignores the wavelet — assumes seismic ≈ zero-phase reflectivity.
#
# ## Method 2: Colored Inversion
#
# Lancaster & Whitcombe (2000) spectral shaping approach:
# 1. Fit a straight line through the log(AI) amplitude spectrum on a log-log scale
# 2. Derive an operator that shapes the seismic spectrum to match the impedance spectrum
# 3. Apply with −90° phase rotation
#
# **Advantage:** Fast, robust, no wavelet or detailed model needed.
#
# ---
#
# ### Important Note: Synthetic Seismic
#
# Since we are working at a single well location and may not have the actual post-stack
# seismic trace extracted at this well, we **generate a synthetic seismic trace** by convolving
# our reflectivity with the extracted wavelet. This is standard practice for validating
# inversion algorithms at the well — commercial tools (STRATA, Petrel) do exactly this
# in their "Inversion Analysis" mode. When actual seismic is available, simply replace the
# synthetic with the real trace.
# ======================================================================


# ======================================================================
# ---
# ## User Parameters
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Phase 1 outputs ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'
BG_PARAMS_CSV    = 'background_trend_params.csv'
WAVELET_FILE     = 'final_wavelet.csv'

# --- Analysis window (TWT in seconds) ---
T_WIN_START      = 2.60       # s TWT
T_WIN_END        = 3.40       # s TWT

# --- Recursive inversion parameters ---
REC_SCALE_METHOD = 'rms'      # 'rms' or 'max' — how to scale trace to RC amplitudes
REC_TARGET_RC    = 0.05       # Target RMS reflectivity (typical: 0.03–0.08)
LFM_MERGE_HZ     = 10.0      # Crossover frequency for LFM merge (should match Phase 1)

# --- Colored inversion parameters ---
CI_MIN_FREQ      = 5.0        # Min frequency for operator calculation (Hz)
CI_MAX_FREQ      = 80.0       # Max frequency for operator calculation (Hz)
CI_THRESHOLD     = 0.20       # Spectrum threshold (fraction of max) below which operator=1
CI_OPERATOR_LEN  = 101        # Operator length in samples (odd number)
CI_TAPER_LEN     = 15         # Cosine taper length (samples) on operator edges

# --- Noise for realistic test (optional) ---
ADD_NOISE        = True       # Add random noise to synthetic for realism
SNR_DB           = 20         # Signal-to-noise ratio in dB

# --- Output ---
OUTPUT_PREFIX    = 'Phase2'


# ======================================================================
# ---
# ## 0. Imports & Load Phase 1 Data
# ======================================================================

# %%  [Cell 2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, welch, fftconvolve
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 11, 'figure.dpi': 120,
})

# Load Phase 1 outputs
df_ai = pd.read_csv(AI_FULLBAND_CSV)
df_lfm = pd.read_csv(LFM_CSV)

twt_s = df_ai['TWT_s'].values
twt_ms = df_ai['TWT_ms'].values
dt_s = np.round(np.median(np.diff(twt_s)), 6)
fs = 1.0 / dt_s

ai_full = df_ai['AI'].values
si_full = df_ai['SI'].values
rc_p = df_ai['RC_P'].values
ln_ai = df_ai['LN_AI'].values

ai_lfm = df_lfm['AI_LFM'].values
ln_ai_lfm = df_lfm['LN_AI_LFM'].values

print(f'Loaded {len(twt_s)} samples, dt = {dt_s*1000:.1f} ms, fs = {fs:.0f} Hz')
print(f'TWT range: {twt_s[0]*1000:.1f} – {twt_s[-1]*1000:.1f} ms')

# %%  [Cell 3]
# Load wavelet
wav_df = pd.read_csv(WAVELET_FILE)
print(f'Wavelet file columns: {list(wav_df.columns)}')

# Flexible column detection
cols = [c.upper() for c in wav_df.columns]
if len(wav_df.columns) == 1:
    wavelet = wav_df.iloc[:, 0].values
    wav_time = (np.arange(len(wavelet)) - len(wavelet)//2) * dt_s * 1000
elif len(wav_df.columns) == 2:
    wav_time = wav_df.iloc[:, 0].values
    wavelet = wav_df.iloc[:, 1].values
else:
    # Try to find amplitude column
    amp_col = [c for c in wav_df.columns if any(k in c.upper() for k in ['AMP', 'WAV', 'VAL'])]
    time_col = [c for c in wav_df.columns if any(k in c.upper() for k in ['TIME', 'T', 'MS'])]
    wavelet = wav_df[amp_col[0]].values if amp_col else wav_df.iloc[:, -1].values
    wav_time = wav_df[time_col[0]].values if time_col else np.arange(len(wavelet))

# Normalize wavelet
wavelet = wavelet / np.max(np.abs(wavelet))

print(f'Wavelet: {len(wavelet)} samples')
print(f'Peak amplitude at sample {np.argmax(np.abs(wavelet))}')


# ======================================================================
# ---
# ## 1. Generate Synthetic Seismic Trace
#
# Forward model: $s = W * r$ (convolution of wavelet with reflectivity).
#
# This synthetic acts as our "seismic input" for inversion testing.
# Optionally, we add band-limited Gaussian noise to simulate realistic conditions.
# ======================================================================

# %%  [Cell 4]
def convolve_wavelet(rc, wavelet):
    """Convolve reflectivity with wavelet, preserving length and alignment."""
    rc_clean = rc.copy()
    rc_clean[np.isnan(rc_clean)] = 0.0
    synth = fftconvolve(rc_clean, wavelet, mode='same')
    return synth

def add_bandlimited_noise(signal, snr_db, dt_sec, fmin=5, fmax=80):
    """Add band-limited Gaussian noise at specified SNR."""
    sig_power = np.mean(signal[np.isfinite(signal)]**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    # Bandpass filter the noise
    nyq = 1.0 / (2.0 * dt_sec)
    sos = butter(4, [fmin/nyq, fmax/nyq], btype='band', output='sos')
    noise = sosfiltfilt(sos, noise)
    # Re-scale to target SNR
    noise = noise * np.sqrt(noise_power / np.mean(noise**2))
    return signal + noise, noise

# Generate synthetic seismic
synthetic_clean = convolve_wavelet(rc_p, wavelet)

if ADD_NOISE:
    synthetic, noise = add_bandlimited_noise(synthetic_clean, SNR_DB, dt_s)
    actual_snr = 10 * np.log10(np.mean(synthetic_clean**2) / np.mean(noise**2))
    print(f'Added noise: target SNR = {SNR_DB} dB, actual SNR = {actual_snr:.1f} dB')
else:
    synthetic = synthetic_clean.copy()
    noise = np.zeros_like(synthetic)

print(f'Synthetic trace: {len(synthetic)} samples')
print(f'RMS amplitude: {np.sqrt(np.mean(synthetic**2)):.6f}')

# %%  [Cell 5]
# Quick view: synthetic vs reflectivity
win = (twt_s >= T_WIN_START) & (twt_s <= T_WIN_END)
t_win = twt_ms[win]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), sharey=True)

ax1.plot(rc_p[win], t_win, 'k-', lw=0.4)
ax1.fill_betweenx(t_win, 0, rc_p[win], where=rc_p[win]>0, alpha=0.3, color='red')
ax1.fill_betweenx(t_win, 0, rc_p[win], where=rc_p[win]<0, alpha=0.3, color='blue')
ax1.set_title('Reflectivity (RC)', fontweight='bold')
ax1.set_xlabel('RC')
ax1.set_ylabel('TWT (ms)')
ax1.invert_yaxis()

ax2.plot(synthetic_clean[win], t_win, 'b-', lw=0.5, alpha=0.5, label='Clean')
ax2.plot(synthetic[win], t_win, 'k-', lw=0.5, label='With noise')
ax2.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]>0, alpha=0.2, color='red')
ax2.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]<0, alpha=0.2, color='blue')
ax2.set_title('Synthetic Seismic', fontweight='bold')
ax2.set_xlabel('Amplitude')
ax2.legend(fontsize=8)

ax3.plot(ai_full[win], t_win, 'k-', lw=0.5, label='AI (truth)')
ax3.plot(ai_lfm[win], t_win, 'r-', lw=2, label='LFM')
ax3.set_title('Target: P-Impedance', fontweight='bold')
ax3.set_xlabel('AI')
ax3.legend(fontsize=8)

fig.suptitle('Inversion Inputs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 2. Recursive (Bandlimited) Inversion
#
# ### Algorithm
#
# 1. **Scale** the seismic trace to reflectivity range (±0.1)
# 2. **Integrate** reflectivity to impedance using the recursive formula:
# $$I_j = I_{j-1} \cdot \frac{1 + r_j}{1 - r_j}$$
# 3. **Merge** with LFM: low-pass filter the LFM, high-pass the inverted trace at the crossover frequency, add them.
#
# ### Why it works (and its limitations)
#
# The recursion is the exact inverse of the reflectivity equation. But because seismic is
# bandlimited and not true reflectivity, the result is *relative* impedance — it captures
# impedance changes within the seismic band but lacks the absolute trend. The LFM merge
# restores the trend.
# ======================================================================

# %%  [Cell 6]
def recursive_inversion(trace, dt_sec, scale_method='rms', target_rc=0.05):
    """
    Recursive (Bandlimited) Inversion.
    
    Scales the seismic trace to reflectivity range, then applies
    the recursive impedance formula.
    
    Parameters
    ----------
    trace : array
        Input seismic trace (assumed approximately zero-phase).
    dt_sec : float
        Sample interval in seconds.
    scale_method : str
        'rms' — scale so RMS matches target_rc
        'max' — scale so max amplitude matches target_rc
    target_rc : float
        Target reflectivity amplitude for scaling.
    
    Returns
    -------
    ai_relative : array
        Relative (bandlimited) acoustic impedance.
    rc_scaled : array
        The scaled reflectivity used in recursion.
    scale_factor : float
        The applied scaling factor.
    """
    tr = trace.copy()
    tr[np.isnan(tr)] = 0.0
    
    # Scale to reflectivity range
    if scale_method == 'rms':
        rms = np.sqrt(np.mean(tr**2))
        scale_factor = target_rc / rms if rms > 0 else 1.0
    elif scale_method == 'max':
        mx = np.max(np.abs(tr))
        scale_factor = target_rc / mx if mx > 0 else 1.0
    else:
        raise ValueError(f'Unknown scale method: {scale_method}')
    
    rc_scaled = tr * scale_factor
    
    # Clip to avoid instability (|r| must be < 1)
    rc_scaled = np.clip(rc_scaled, -0.5, 0.5)
    
    # Recursive integration: I_j = I_{j-1} * (1+r)/(1-r)
    # Start with I_0 = 1 (relative impedance)
    n = len(rc_scaled)
    ai_relative = np.ones(n)
    for j in range(1, n):
        ai_relative[j] = ai_relative[j-1] * (1 + rc_scaled[j]) / (1 - rc_scaled[j])
    
    return ai_relative, rc_scaled, scale_factor


def merge_lfm(ai_relative, ai_lfm, dt_sec, crossover_hz, filter_order=8):
    """
    Merge bandlimited inversion result with LFM.
    
    The LFM provides frequencies below crossover_hz.
    The inverted trace provides frequencies above crossover_hz.
    
    We work in log-impedance space for proper merging (additive in log domain).
    """
    nyq = 1.0 / (2.0 * dt_sec)
    
    # Work in log domain
    ln_inv = np.log(ai_relative)
    ln_lfm = np.log(ai_lfm)
    
    # Handle NaNs
    nan_mask = np.isnan(ln_inv) | np.isnan(ln_lfm)
    ln_inv_clean = ln_inv.copy()
    ln_lfm_clean = ln_lfm.copy()
    if nan_mask.any():
        good = ~nan_mask
        ln_inv_clean[nan_mask] = np.interp(np.flatnonzero(nan_mask), 
                                            np.flatnonzero(good), ln_inv[good])
        ln_lfm_clean[nan_mask] = np.interp(np.flatnonzero(nan_mask), 
                                            np.flatnonzero(good), ln_lfm[good])
    
    # High-pass the inverted trace
    sos_hp = butter(filter_order, crossover_hz, btype='high', fs=1/dt_sec, output='sos')
    ln_inv_hp = sosfiltfilt(sos_hp, ln_inv_clean)
    
    # Low-pass the LFM (should already be low-freq, but ensure clean separation)
    sos_lp = butter(filter_order, crossover_hz, btype='low', fs=1/dt_sec, output='sos')
    ln_lfm_lp = sosfiltfilt(sos_lp, ln_lfm_clean)
    
    # Merge: LFM (low freq) + Inverted (high freq)
    ln_merged = ln_lfm_lp + ln_inv_hp
    
    # Back to linear impedance
    ai_merged = np.exp(ln_merged)
    ai_merged[nan_mask] = np.nan
    
    return ai_merged, ln_merged


# --- Run recursive inversion ---
ai_rec_rel, rc_scaled, scale_fac = recursive_inversion(
    synthetic, dt_s, REC_SCALE_METHOD, REC_TARGET_RC
)
print(f'Scale factor: {scale_fac:.6f}')
print(f'Scaled RC range: {rc_scaled.min():.4f} to {rc_scaled.max():.4f}')

# --- Merge with LFM ---
ai_rec_abs, ln_ai_rec = merge_lfm(ai_rec_rel, ai_lfm, dt_s, LFM_MERGE_HZ)

# --- Compute correlation in analysis window ---
valid_win = win & np.isfinite(ai_full) & np.isfinite(ai_rec_abs)
cc_rec = np.corrcoef(ai_full[valid_win], ai_rec_abs[valid_win])[0, 1]
rms_err_rec = np.sqrt(np.mean((ai_full[valid_win] - ai_rec_abs[valid_win])**2))
nrms_rec = rms_err_rec / np.std(ai_full[valid_win])

print(f'\nRecursive Inversion Results (window {T_WIN_START*1000:.0f}–{T_WIN_END*1000:.0f} ms):')
print(f'  CC  = {cc_rec:.4f}')
print(f'  RMS error = {rms_err_rec:.0f}')
print(f'  NRMS = {nrms_rec:.3f}')

# %%  [Cell 7]
# --- QC Plot: Recursive Inversion ---
fig, axes = plt.subplots(1, 4, figsize=(16, 10), sharey=True)

# Seismic input
axes[0].plot(synthetic[win], t_win, 'k-', lw=0.4)
axes[0].fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]>0, 
                       alpha=0.3, color='red')
axes[0].fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]<0, 
                       alpha=0.3, color='blue')
axes[0].set_title('Seismic Input', fontweight='bold')
axes[0].set_xlabel('Amplitude')
axes[0].set_ylabel('TWT (ms)')
axes[0].invert_yaxis()

# Relative impedance
axes[1].plot(ai_rec_rel[win], t_win, 'orange', lw=0.8)
axes[1].axvline(1.0, color='gray', ls='--', lw=0.5)
axes[1].set_title('Relative AI\n(before LFM merge)', fontweight='bold')
axes[1].set_xlabel('Relative AI')

# Absolute impedance: inverted vs truth
axes[2].plot(ai_full[win], t_win, 'k-', lw=0.5, label='Well AI (truth)')
axes[2].plot(ai_rec_abs[win], t_win, 'r-', lw=1.2, label='Recursive inv.')
axes[2].plot(ai_lfm[win], t_win, 'b--', lw=1, alpha=0.5, label='LFM only')
axes[2].set_title(f'Absolute AI\nCC = {cc_rec:.3f}', fontweight='bold')
axes[2].set_xlabel('P-Impedance')
axes[2].legend(fontsize=8)

# Error
err = ai_full - ai_rec_abs
axes[3].plot(err[win], t_win, 'r-', lw=0.5)
axes[3].axvline(0, color='gray', ls='--', lw=0.5)
axes[3].fill_betweenx(t_win, 0, err[win], where=err[win]>0, alpha=0.2, color='red')
axes[3].fill_betweenx(t_win, 0, err[win], where=err[win]<0, alpha=0.2, color='blue')
axes[3].set_title(f'Error (Truth − Inv)\nNRMS = {nrms_rec:.3f}', fontweight='bold')
axes[3].set_xlabel('AI error')

fig.suptitle('Recursive (Bandlimited) Inversion — Well 15/9-F-11 T2', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 3. Colored Inversion (Lancaster & Whitcombe, 2000)
#
# ### Algorithm
#
# 1. Compute the **amplitude spectrum of ln(AI)** from well data. Fit a straight line on a log-log scale.
# 2. Compute the **amplitude spectrum of the seismic trace**. Smooth it.
# 3. The **colored operator spectrum** = impedance spectrum / seismic spectrum
# 4. Apply **−90° phase rotation** (the derivative relationship between impedance and reflectivity)
# 5. Convolve operator with seismic → relative AI
#
# The result is relative impedance (zero-mean). To get absolute AI, we add the LFM.
# ======================================================================

# %%  [Cell 8]
def colored_inversion(trace, ln_impedance, dt_sec, 
                       fmin=5, fmax=80, threshold=0.2,
                       operator_len=101, taper_len=15):
    """
    Colored Inversion (Lancaster & Whitcombe, 2000).
    
    Parameters
    ----------
    trace : array
        Input seismic trace.
    ln_impedance : array
        ln(AI) from well log (for target spectrum).
    dt_sec : float
        Sample interval in seconds.
    fmin, fmax : float
        Frequency range for operator computation.
    threshold : float
        Below this fraction of max seismic spectrum, operator = 1.
    operator_len : int
        Length of the inversion operator (samples, odd).
    taper_len : int
        Cosine taper length on operator edges.
    
    Returns
    -------
    ai_colored : array
        Colored inversion result (relative impedance).
    operator : array
        The colored inversion operator.
    diag : dict
        Diagnostic info (spectra, fit, etc.).
    """
    fs = 1.0 / dt_sec
    n = len(trace)
    
    # --- Step 1: Impedance spectrum (from well) ---
    ln_imp = ln_impedance.copy()
    ln_imp[np.isnan(ln_imp)] = np.nanmean(ln_imp)
    
    nfft = max(2**int(np.ceil(np.log2(n))), 1024)
    freq = np.fft.rfftfreq(nfft, dt_sec)
    
    imp_fft = np.abs(np.fft.rfft(ln_imp - np.mean(ln_imp), n=nfft))
    
    # Fit straight line on log-log scale (within frequency range)
    f_mask = (freq >= fmin) & (freq <= fmax) & (freq > 0) & (imp_fft > 0)
    log_f = np.log10(freq[f_mask])
    log_a = np.log10(imp_fft[f_mask])
    
    # Robust fit (linear regression in log-log space)
    coeffs = np.polyfit(log_f, log_a, 1)
    imp_spectrum_fit = 10**(coeffs[0] * np.log10(freq + 1e-10) + coeffs[1])
    imp_spectrum_fit[0] = 0  # DC = 0 for relative result
    
    # --- Step 2: Seismic spectrum ---
    tr_clean = trace.copy()
    tr_clean[np.isnan(tr_clean)] = 0.0
    seis_fft = np.abs(np.fft.rfft(tr_clean, n=nfft))
    
    # Smooth the seismic spectrum
    from scipy.ndimage import uniform_filter1d
    seis_smooth = uniform_filter1d(seis_fft, size=max(5, nfft//128))
    
    # --- Step 3: Operator spectrum ---
    seis_max = np.max(seis_smooth)
    operator_spectrum = np.ones_like(freq)
    
    valid_f = (freq >= fmin) & (freq <= fmax) & (seis_smooth > threshold * seis_max)
    operator_spectrum[valid_f] = imp_spectrum_fit[valid_f] / seis_smooth[valid_f]
    
    # Outside the valid range, operator = 0 (don't pass those frequencies)
    operator_spectrum[freq < fmin] = 0
    operator_spectrum[freq > fmax] = 0
    
    # --- Step 4: Add −90° phase rotation ---
    # −90° phase = multiply by -j in frequency domain
    phase_90 = np.exp(-1j * np.pi/2)  # = -j
    operator_fft = operator_spectrum * phase_90
    
    # --- Step 5: Convert to time domain operator ---
    op_full = np.fft.irfft(operator_fft, n=nfft)
    
    # Extract centered operator of desired length
    half = operator_len // 2
    op_time = np.roll(op_full, half)[:operator_len]
    
    # Apply cosine taper
    taper = np.ones(operator_len)
    if taper_len > 0:
        cos_taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
        taper[:taper_len] = cos_taper
        taper[-taper_len:] = cos_taper[::-1]
    op_time *= taper
    
    # --- Step 6: Apply operator to seismic ---
    ai_colored = fftconvolve(tr_clean, op_time, mode='same')
    
    diag = {
        'freq': freq,
        'imp_spectrum': imp_fft,
        'imp_fit': imp_spectrum_fit,
        'seis_spectrum': seis_smooth,
        'operator_spectrum': operator_spectrum,
        'fit_slope': coeffs[0],
        'fit_intercept': coeffs[1],
    }
    
    return ai_colored, op_time, diag


# Run colored inversion
ai_ci_rel, ci_operator, ci_diag = colored_inversion(
    synthetic, ln_ai, dt_s,
    fmin=CI_MIN_FREQ, fmax=CI_MAX_FREQ,
    threshold=CI_THRESHOLD,
    operator_len=CI_OPERATOR_LEN,
    taper_len=CI_TAPER_LEN
)

print(f'Colored inversion operator: {len(ci_operator)} samples')
print(f'Spectrum fit: slope = {ci_diag["fit_slope"]:.2f}, '
      f'intercept = {ci_diag["fit_intercept"]:.2f}')

# %%  [Cell 9]
# --- Convert colored inversion to absolute impedance ---
# Colored inversion gives relative AI. Scale and add LFM.

# Method: match the RMS of the colored result to the RMS of the 
# bandlimited (high-freq) component of the well AI
ai_hfm = df_lfm['AI_HFM'].values  # from Phase 1

# Scale colored result
ci_rms = np.sqrt(np.nanmean(ai_ci_rel[win]**2))
hfm_rms = np.sqrt(np.nanmean(ai_hfm[win]**2))
ci_scale = hfm_rms / ci_rms if ci_rms > 0 else 1.0

ai_ci_scaled = ai_ci_rel * ci_scale

# Add LFM
ai_ci_abs = ai_lfm + ai_ci_scaled

# Metrics
valid_ci = win & np.isfinite(ai_full) & np.isfinite(ai_ci_abs)
cc_ci = np.corrcoef(ai_full[valid_ci], ai_ci_abs[valid_ci])[0, 1]
rms_err_ci = np.sqrt(np.mean((ai_full[valid_ci] - ai_ci_abs[valid_ci])**2))
nrms_ci = rms_err_ci / np.std(ai_full[valid_ci])

print(f'CI scale factor: {ci_scale:.4f}')
print(f'\nColored Inversion Results:')
print(f'  CC  = {cc_ci:.4f}')
print(f'  RMS error = {rms_err_ci:.0f}')
print(f'  NRMS = {nrms_ci:.3f}')

# %%  [Cell 10]
# --- Colored Inversion Diagnostic Plots ---
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, height_ratios=[1, 1.5], hspace=0.35, wspace=0.3)

# Row 1: Operator diagnostics

# Spectrum comparison (log-log)
ax1 = fig.add_subplot(gs[0, 0])
f = ci_diag['freq']
f_pos = f > 0
ax1.loglog(f[f_pos], ci_diag['imp_spectrum'][f_pos], 'b-', lw=0.5, alpha=0.5, label='ln(AI) spectrum')
ax1.loglog(f[f_pos], ci_diag['imp_fit'][f_pos], 'r-', lw=2, label='Linear fit')
ax1.loglog(f[f_pos], ci_diag['seis_spectrum'][f_pos], 'k-', lw=1, label='Seismic spectrum')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Log-Log Spectra', fontweight='bold')
ax1.set_xlim(1, 120)
ax1.legend(fontsize=8)

# Operator spectrum
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(f, ci_diag['operator_spectrum'], 'g-', lw=1.5)
ax2.axvline(CI_MIN_FREQ, color='gray', ls='--', lw=0.5)
ax2.axvline(CI_MAX_FREQ, color='gray', ls='--', lw=0.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Operator amplitude')
ax2.set_title('CI Operator Spectrum', fontweight='bold')
ax2.set_xlim(0, 100)

# Operator in time domain
ax3 = fig.add_subplot(gs[0, 2])
op_t = (np.arange(len(ci_operator)) - len(ci_operator)//2) * dt_s * 1000
ax3.plot(op_t, ci_operator, 'g-', lw=1.5)
ax3.axhline(0, color='gray', lw=0.5)
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Amplitude')
ax3.set_title('CI Operator (time domain)', fontweight='bold')

# Row 2: Inversion result
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(synthetic[win], t_win, 'k-', lw=0.4)
ax4.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]>0, alpha=0.2, color='red')
ax4.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]<0, alpha=0.2, color='blue')
ax4.set_title('Seismic Input', fontweight='bold')
ax4.set_ylabel('TWT (ms)')
ax4.invert_yaxis()

ax5 = fig.add_subplot(gs[1, 1], sharey=ax4)
ax5.plot(ai_ci_rel[win], t_win, 'g-', lw=0.8)
ax5.axvline(0, color='gray', ls='--', lw=0.5)
ax5.set_title('Relative AI (colored)', fontweight='bold')
ax5.set_xlabel('Relative AI')

ax6 = fig.add_subplot(gs[1, 2], sharey=ax4)
ax6.plot(ai_full[win], t_win, 'k-', lw=0.5, label='Well AI')
ax6.plot(ai_ci_abs[win], t_win, 'g-', lw=1.2, label='CI result')
ax6.plot(ai_lfm[win], t_win, 'b--', lw=1, alpha=0.4, label='LFM')
ax6.set_title(f'Absolute AI — CC = {cc_ci:.3f}', fontweight='bold')
ax6.set_xlabel('P-Impedance')
ax6.legend(fontsize=8)

fig.suptitle('Colored Inversion — Diagnostics & Result', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 4. Method Comparison
#
# Side-by-side comparison of Recursive vs. Colored inversion against the well truth.
# This is the panel that should look like STRATA's inversion analysis display.
# ======================================================================

# %%  [Cell 11]
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 5, height_ratios=[2.5, 1], hspace=0.35, wspace=0.12)

# --- Row 1: Time-domain comparison ---

# Track 1: Seismic
ax = fig.add_subplot(gs[0, 0])
ax.plot(synthetic[win], t_win, 'k-', lw=0.4)
ax.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]>0, alpha=0.25, color='red')
ax.fill_betweenx(t_win, 0, synthetic[win], where=synthetic[win]<0, alpha=0.25, color='blue')
ax.set_title('Seismic\nInput', fontweight='bold')
ax.set_ylabel('TWT (ms)')
ax.invert_yaxis()

# Track 2: Well AI (truth)
ax = fig.add_subplot(gs[0, 1], sharey=fig.axes[0])
ax.plot(ai_full[win], t_win, 'k-', lw=0.6)
ax.fill_betweenx(t_win, ai_full[win].min(), ai_full[win], alpha=0.1, color='gold')
ax.set_title('Well AI\n(Truth)', fontweight='bold')
ax.set_yticklabels([])

# Track 3: LFM only
ax = fig.add_subplot(gs[0, 2], sharey=fig.axes[0])
ax.plot(ai_full[win], t_win, 'k-', lw=0.3, alpha=0.3)
ax.plot(ai_lfm[win], t_win, 'b-', lw=2, label='LFM')
cc_lfm = np.corrcoef(ai_full[valid_win], ai_lfm[valid_win])[0, 1]
ax.set_title(f'LFM Only\nCC={cc_lfm:.3f}', fontweight='bold', color='blue')
ax.set_yticklabels([])

# Track 4: Recursive inversion
ax = fig.add_subplot(gs[0, 3], sharey=fig.axes[0])
ax.plot(ai_full[win], t_win, 'k-', lw=0.3, alpha=0.3)
ax.plot(ai_rec_abs[win], t_win, 'r-', lw=1.2, label='Recursive')
ax.set_title(f'Recursive\nCC={cc_rec:.3f}', fontweight='bold', color='red')
ax.set_yticklabels([])

# Track 5: Colored inversion
ax = fig.add_subplot(gs[0, 4], sharey=fig.axes[0])
ax.plot(ai_full[win], t_win, 'k-', lw=0.3, alpha=0.3)
ax.plot(ai_ci_abs[win], t_win, 'g-', lw=1.2, label='Colored')
ax.set_title(f'Colored\nCC={cc_ci:.3f}', fontweight='bold', color='green')
ax.set_yticklabels([])

# --- Row 2: Cross-plots and spectra ---

# Cross-plot: Recursive
ax = fig.add_subplot(gs[1, 0:2])
ax.scatter(ai_full[valid_win], ai_rec_abs[valid_win], c='red', s=2, alpha=0.3, label='Recursive')
ax.scatter(ai_full[valid_ci], ai_ci_abs[valid_ci], c='green', s=2, alpha=0.3, label='Colored')
lims = [np.nanmin(ai_full[valid_win])*0.95, np.nanmax(ai_full[valid_win])*1.05]
ax.plot(lims, lims, 'k--', lw=1, label='1:1 line')
ax.set_xlabel('Well AI (truth)')
ax.set_ylabel('Inverted AI')
ax.set_title('Cross-plot: Inverted vs Truth', fontweight='bold')
ax.legend(fontsize=8)
ax.set_aspect('equal')

# Spectra comparison
ax = fig.add_subplot(gs[1, 2:4])
for label, data, color in [
    ('Well AI', ai_full, 'black'),
    ('Recursive', ai_rec_abs, 'red'),
    ('Colored', ai_ci_abs, 'green'),
    ('LFM', ai_lfm, 'blue'),
]:
    d = data[win].copy()
    d[np.isnan(d)] = np.nanmean(d)
    f_w, a_w = welch(d - np.mean(d), fs=fs, nperseg=min(len(d), 256))
    ax.plot(f_w, np.sqrt(a_w), color=color, lw=1.5 if label=='Well AI' else 1, 
            ls='-' if label!='LFM' else '--', label=label)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.set_title('Amplitude Spectra Comparison', fontweight='bold')
ax.set_xlim(0, 100)
ax.legend(fontsize=8)

# Metrics table
ax = fig.add_subplot(gs[1, 4])
ax.axis('off')
table_data = [
    ['Method', 'CC', 'NRMS'],
    ['LFM only', f'{cc_lfm:.3f}', '—'],
    ['Recursive', f'{cc_rec:.3f}', f'{nrms_rec:.3f}'],
    ['Colored', f'{cc_ci:.3f}', f'{nrms_ci:.3f}'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
# Bold header
for j in range(3):
    table[0, j].set_text_props(fontweight='bold')
    table[0, j].set_facecolor('#d4e6f1')
# Highlight best CC
best_row = 2 if cc_rec > cc_ci else 3
for j in range(3):
    table[best_row, j].set_facecolor('#d5f5e3')
ax.set_title('Metrics', fontweight='bold')

fig.suptitle(
    'OpenSI Phase 2 — Recursive & Colored Inversion Comparison\n'
    f'Well 15/9-F-11 T2 | Window {T_WIN_START*1000:.0f}–{T_WIN_END*1000:.0f} ms | '
    f'SNR = {SNR_DB} dB' if ADD_NOISE else '',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_Comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ======================================================================
# ---
# ## 5. Sensitivity: Scaling Parameter (Recursive) and Cutoff (Colored)
#
# These are the main tuning knobs. Let's see how sensitive the results are.
# ======================================================================

# %%  [Cell 12]
# --- Recursive: sensitivity to target RC ---
rc_targets = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
rec_ccs = []

fig, axes = plt.subplots(1, len(rc_targets), figsize=(18, 8), sharey=True)

for i, trc in enumerate(rc_targets):
    ai_r, _, _ = recursive_inversion(synthetic, dt_s, 'rms', trc)
    ai_r_abs, _ = merge_lfm(ai_r, ai_lfm, dt_s, LFM_MERGE_HZ)
    v = win & np.isfinite(ai_full) & np.isfinite(ai_r_abs)
    cc = np.corrcoef(ai_full[v], ai_r_abs[v])[0, 1]
    rec_ccs.append(cc)
    
    axes[i].plot(ai_full[win], t_win, 'k-', lw=0.3, alpha=0.3)
    axes[i].plot(ai_r_abs[win], t_win, 'r-', lw=0.8)
    title_color = 'red' if trc == REC_TARGET_RC else 'black'
    axes[i].set_title(f'RC={trc}\nCC={cc:.3f}', fontweight='bold', 
                       color=title_color, fontsize=10)
    if i == 0:
        axes[i].set_ylabel('TWT (ms)')
        axes[i].invert_yaxis()
    else:
        axes[i].set_yticklabels([])

fig.suptitle('Recursive Inversion — Target RC Sensitivity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Print best
best_idx = np.argmax(rec_ccs)
print(f'Best target RC = {rc_targets[best_idx]} → CC = {rec_ccs[best_idx]:.4f}')


# ======================================================================
# ---
# ## 6. Export Results
# ======================================================================

# %%  [Cell 13]
# Save inversion results
df_results = pd.DataFrame({
    'TWT_s': twt_s,
    'TWT_ms': twt_ms,
    'Seismic_Synthetic': synthetic,
    'AI_Well_Truth': ai_full,
    'AI_LFM': ai_lfm,
    'AI_Recursive': ai_rec_abs,
    'AI_Colored': ai_ci_abs,
    'AI_Recursive_Relative': ai_rec_rel,
    'AI_Colored_Relative': ai_ci_rel,
})
out_csv = f'{OUTPUT_PREFIX}_inversion_results.csv'
df_results.to_csv(out_csv, index=False, float_format='%.6f')
print(f'Saved: {out_csv}')

# Save CI operator
op_df = pd.DataFrame({
    'Time_ms': (np.arange(len(ci_operator)) - len(ci_operator)//2) * dt_s * 1000,
    'Amplitude': ci_operator,
})
op_df.to_csv(f'{OUTPUT_PREFIX}_CI_operator.csv', index=False, float_format='%.8f')
print(f'Saved: {OUTPUT_PREFIX}_CI_operator.csv')

# Summary
print(f'\n{"="*60}')
print(f'Phase 2 COMPLETE — Recursive & Colored Inversion')
print(f'{"="*60}')
print(f'  Recursive Inversion:  CC = {cc_rec:.4f}  NRMS = {nrms_rec:.3f}')
print(f'  Colored Inversion:    CC = {cc_ci:.4f}  NRMS = {nrms_ci:.3f}')
print(f'  LFM baseline:         CC = {cc_lfm:.4f}')
print(f'\nOutputs:')
print(f'  {out_csv}')
print(f'  {OUTPUT_PREFIX}_CI_operator.csv')
print(f'  {OUTPUT_PREFIX}_Comparison.png')
print(f'\nReady for Phase 3 (Model-Based Inversion)')

