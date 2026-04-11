#!/usr/bin/env python3
"""
OpenSI — Phase 1: Low-Frequency Model (LFM) Builder

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase1_LFM_Builder.py
"""


# ======================================================================
# # OpenSI — Phase 1: Low-Frequency Model (LFM) Builder
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Build the low-frequency impedance model that underpins every seismic inversion method.
# **Reference:** Lindseth (1979), Chopra & Castagna (2014), Hampson & Russell (1991)
#
# ---
#
# ## Why the LFM Matters
#
# Seismic data is bandlimited — typically 8–80 Hz. The **low frequencies (0–8 Hz)** that define the
# absolute impedance trend are missing from the seismic but present in well logs. Without restoring
# them, any inversion result will be *relative* impedance (oscillating around zero) rather than
# *absolute* impedance (actual rock property values).
#
# The LFM provides:
# 1. The **DC component and trend** — converting relative to absolute impedance
# 2. The **low-frequency structure** — regional velocity increase with depth, formation boundaries
# 3. A **constraint** for model-based inversion — the starting model that gets perturbed
#
# In commercial software (STRATA, Petrel), this is built by:
# 1. Computing impedance logs at well locations
# 2. Low-pass filtering these logs below the seismic bandwidth
# 3. Interpolating between wells guided by horizons
#
# For our single-well Volve demonstration, we build a 1-D LFM and prepare the architecture
# for multi-well extension.
# ======================================================================


# ======================================================================
# ---
# ## User Parameters
#
# All tunable parameters are collected here. Adjust these before running the notebook.
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS — Edit this cell only
# ============================================================

# --- File paths (from prior AVO workflow) ---
LAS_FILE        = 'WLC_PETRO_COMPUTED_INPUT_1.LAS'   # Well log data
TD_FILE         = 'td_calibrated_15_9_F11_T2.csv'    # Calibrated time-depth table
WAVELET_FILE    = 'final_wavelet.csv'                # Extracted wavelet (for bandwidth ref)

# --- Log curve names in the LAS file ---
DEPTH_CURVE     = 'DEPTH'    # Depth curve (MD in metres)
VP_CURVE        = 'VP'       # P-wave velocity (m/s) — or 'VELP', 'DT' (slowness)
VS_CURVE        = 'VS'       # S-wave velocity (m/s) — or 'VELS', 'DTS'
RHOB_CURVE      = 'RHOB'     # Bulk density (g/cc)
IS_SLOWNESS     = False      # Set True if VP_CURVE/VS_CURVE are DT/DTS in µs/ft

# --- LFM filter parameters ---
LFM_CUTOFF_HZ   = 10.0      # Low-pass cutoff frequency (Hz). Typical: 6–12 Hz
FILTER_ORDER     = 8         # Butterworth filter order (higher = sharper cutoff)
SEISMIC_DT_MS    = 2.0       # Seismic sample rate (ms) for resampling

# --- Analysis window (TWT in seconds) ---
T_WIN_START      = 2.5       # Start of analysis window (s TWT)
T_WIN_END        = 3.5       # End of analysis window (s TWT)

# --- Gardner's equation fallback (if RHOB missing) ---
GARDNER_A        = 0.31      # Gardner coefficient (SI: km/s, g/cc)
GARDNER_B        = 0.25      # Gardner exponent

# --- Output ---
OUTPUT_LFM_CSV   = 'lfm_model_15_9_F11_T2.csv'
OUTPUT_AI_CSV    = 'ai_fullband_15_9_F11_T2.csv'


# ======================================================================
# ---
# ## 0. Imports
# ======================================================================

# %%  [Cell 2]
import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, welch
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 120,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})

print('Imports OK')


# ======================================================================
# ---
# ## 1. Load Well Log Data
#
# We need VP (or DT), RHOB, and optionally VS (or DTS) to compute:
# - **P-impedance:** AI = VP × RHOB
# - **S-impedance:** SI = VS × RHOB (if shear data available)
#
# If density is missing, we fall back to Gardner's equation: ρ = a × VP^b
# ======================================================================

# %%  [Cell 3]
# Load LAS
las = lasio.read(LAS_FILE)
print(f'Well: {las.well.WELL.value if hasattr(las.well, "WELL") else "Unknown"}')
print(f'Curves: {[c.mnemonic for c in las.curves]}')
print(f'Depth range: {las.index[0]:.1f} – {las.index[-1]:.1f} m')
print(f'Samples: {len(las.index)}')

# Build working dataframe
df = las.df().reset_index()
df.rename(columns={df.columns[0]: 'DEPTH'}, inplace=True)

# --- Identify available curves (flexible matching) ---
available = {c.upper() for c in df.columns}

def find_curve(preferred, alternatives, available_set):
    """Find first available curve name from a priority list."""
    for name in [preferred] + alternatives:
        if name.upper() in available_set:
            return name.upper()
    return None

vp_col = find_curve(VP_CURVE, ['VELP', 'VP', 'DT', 'DTC', 'DTCO'], available)
vs_col = find_curve(VS_CURVE, ['VELS', 'VS', 'DTS', 'DTSM'], available)
rho_col = find_curve(RHOB_CURVE, ['RHOB', 'RHOZ', 'DEN', 'DENSITY'], available)

print(f'\nUsing: VP={vp_col}, VS={vs_col}, RHOB={rho_col}')
if vp_col is None:
    raise ValueError('No P-wave velocity or slowness curve found!')

# %%  [Cell 4]
# --- Extract and convert curves ---
depth = df['DEPTH'].values.copy()

# P-wave velocity
if IS_SLOWNESS or vp_col in ('DT', 'DTC', 'DTCO'):
    dt_raw = df[vp_col].values.copy()
    # DT in µs/ft → VP in m/s:  VP = 1e6 / (DT * 3.28084)
    vp = 1e6 / (dt_raw * 3.28084)
    print(f'Converted {vp_col} (µs/ft) → VP (m/s)')
else:
    vp = df[vp_col].values.copy()

# S-wave velocity
has_vs = vs_col is not None
if has_vs:
    if IS_SLOWNESS or vs_col in ('DTS', 'DTSM'):
        dts_raw = df[vs_col].values.copy()
        vs = 1e6 / (dts_raw * 3.28084)
        print(f'Converted {vs_col} (µs/ft) → VS (m/s)')
    else:
        vs = df[vs_col].values.copy()
else:
    vs = None
    print('No S-wave data — SI model will use Castagna mudrock line')

# Density
if rho_col is not None:
    rhob = df[rho_col].values.copy()
    print(f'Density from {rho_col}')
else:
    # Gardner fallback
    vp_kms = vp / 1000.0
    rhob = GARDNER_A * (vp_kms ** GARDNER_B)
    print(f'Density estimated via Gardner: ρ = {GARDNER_A} × VP^{GARDNER_B}')

# Quick stats
valid = np.isfinite(vp) & np.isfinite(rhob) & (vp > 0) & (rhob > 0)
print(f'\nValid samples: {valid.sum()} / {len(valid)}')
print(f'VP range: {np.nanmin(vp[valid]):.0f} – {np.nanmax(vp[valid]):.0f} m/s')
print(f'RHOB range: {np.nanmin(rhob[valid]):.3f} – {np.nanmax(rhob[valid]):.3f} g/cc')


# ======================================================================
# ---
# ## 2. Compute Acoustic & Shear Impedance
#
# $$AI = V_P \times \rho \quad \text{(P-impedance, units: m/s × g/cc = kPa·s/m)}$$
# $$SI = V_S \times \rho \quad \text{(S-impedance)}$$
#
# These are the *full-bandwidth* impedance logs — they contain both the low-frequency
# trend we want for the LFM and the high-frequency detail that comes from the well but
# not from seismic.
# ======================================================================

# %%  [Cell 5]
# Compute impedances in depth domain
ai_depth = vp * rhob          # P-impedance
ai_depth[~valid] = np.nan

if has_vs:
    valid_s = valid & np.isfinite(vs) & (vs > 0)
    si_depth = vs * rhob
    si_depth[~valid_s] = np.nan
else:
    # Castagna mudrock line: VS = 0.8621*VP - 1172.4 (m/s)
    vs_est = 0.8621 * vp - 1172.4
    vs_est[vs_est < 100] = np.nan
    si_depth = vs_est * rhob
    si_depth[~valid] = np.nan
    vs = vs_est
    print('SI computed via Castagna mudrock line')

# VP/VS ratio
vpvs_depth = np.full_like(vp, np.nan)
mask_vpvs = valid & np.isfinite(vs) & (vs > 0)
vpvs_depth[mask_vpvs] = vp[mask_vpvs] / vs[mask_vpvs]

print(f'AI range: {np.nanmin(ai_depth):.0f} – {np.nanmax(ai_depth):.0f}')
print(f'SI range: {np.nanmin(si_depth):.0f} – {np.nanmax(si_depth):.0f}')
print(f'VP/VS range: {np.nanmin(vpvs_depth):.2f} – {np.nanmax(vpvs_depth):.2f}')


# ======================================================================
# ---
# ## 3. Depth-to-Time Conversion
#
# The LFM must be in the **time domain** (TWT) to merge with seismic data.
# We use the calibrated T-D relationship from the prior AVO workflow,
# applying PCHIP interpolation for smooth conversion.
# ======================================================================

# %%  [Cell 6]
# Load calibrated time-depth table
td = pd.read_csv(TD_FILE)
print(f'T-D table columns: {list(td.columns)}')
print(td.head())

# Identify columns (flexible)
td_cols = [c.upper() for c in td.columns]
depth_td_col = td.columns[[i for i, c in enumerate(td_cols) 
                            if any(k in c for k in ['DEPTH', 'MD', 'Z'])][0]]
twt_td_col = td.columns[[i for i, c in enumerate(td_cols) 
                          if any(k in c for k in ['TWT', 'TIME', 'T_'])][0]]

td_depth = td[depth_td_col].values
td_twt = td[twt_td_col].values

# Ensure TWT is in seconds (detect if in ms)
if np.max(td_twt) > 100:  # likely milliseconds
    td_twt = td_twt / 1000.0
    print('Converted TWT from ms to seconds')

print(f'\nT-D range: {td_depth[0]:.1f}–{td_depth[-1]:.1f} m  →  '
      f'{td_twt[0]:.4f}–{td_twt[-1]:.4f} s TWT')

# %%  [Cell 7]
# PCHIP interpolation: depth → TWT
depth_to_twt = PchipInterpolator(td_depth, td_twt)

# Restrict logs to the T-D table depth range
z_min, z_max = td_depth.min(), td_depth.max()
in_range = (depth >= z_min) & (depth <= z_max) & valid

depth_valid = depth[in_range]
twt_log = depth_to_twt(depth_valid)  # TWT in seconds

ai_valid = ai_depth[in_range]
si_valid = si_depth[in_range]
vp_valid = vp[in_range]
vs_valid = vs[in_range]
rhob_valid = rhob[in_range]
vpvs_valid = vpvs_depth[in_range]

print(f'Logs in T-D range: {in_range.sum()} samples')
print(f'TWT range: {twt_log[0]:.4f} – {twt_log[-1]:.4f} s')


# ======================================================================
# ---
# ## 4. Resample to Uniform Time Grid
#
# Logs in depth are unevenly sampled in time (because velocity varies with depth).
# We resample to a uniform time grid matching the seismic sample rate.
# This is essential for frequency-domain filtering.
# ======================================================================

# %%  [Cell 8]
# Create uniform TWT grid
dt_s = SEISMIC_DT_MS / 1000.0  # sample interval in seconds
twt_uniform = np.arange(
    np.ceil(twt_log[0] / dt_s) * dt_s,
    np.floor(twt_log[-1] / dt_s) * dt_s + dt_s/2,
    dt_s
)

# PCHIP interpolation for each curve: TWT(irregular) → TWT(uniform)
def resample_to_time(twt_irreg, values, twt_grid):
    """Resample a log from irregular TWT to uniform TWT grid via PCHIP."""
    mask = np.isfinite(values)
    if mask.sum() < 4:
        return np.full_like(twt_grid, np.nan)
    interp = PchipInterpolator(twt_irreg[mask], values[mask], extrapolate=False)
    return interp(twt_grid)

ai_twt = resample_to_time(twt_log, ai_valid, twt_uniform)
si_twt = resample_to_time(twt_log, si_valid, twt_uniform)
vp_twt = resample_to_time(twt_log, vp_valid, twt_uniform)
vs_twt = resample_to_time(twt_log, vs_valid, twt_uniform)
rhob_twt = resample_to_time(twt_log, rhob_valid, twt_uniform)
vpvs_twt = resample_to_time(twt_log, vpvs_valid, twt_uniform)

print(f'Uniform TWT grid: {len(twt_uniform)} samples at {SEISMIC_DT_MS} ms')
print(f'Time range: {twt_uniform[0]*1000:.1f} – {twt_uniform[-1]*1000:.1f} ms TWT')
print(f'Nyquist frequency: {1/(2*dt_s):.1f} Hz')


# ======================================================================
# ---
# ## 5. Compute Reflectivity Series
#
# The reflection coefficient at interface $i$ is:
#
# $$r_i = \frac{I_{i+1} - I_i}{I_{i+1} + I_i}$$
#
# This reflectivity series is what gets convolved with the wavelet in forward modeling.
# We compute it here for later use in Phases 2–5.
# ======================================================================

# %%  [Cell 9]
def compute_rc(impedance):
    """Compute reflection coefficient series from impedance log."""
    rc = np.zeros_like(impedance)
    rc[1:] = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1])
    rc[np.isnan(rc)] = 0.0
    return rc

rc_p = compute_rc(ai_twt)
rc_s = compute_rc(si_twt)

print(f'P-reflectivity: max |rc| = {np.nanmax(np.abs(rc_p)):.4f}')
print(f'S-reflectivity: max |rc| = {np.nanmax(np.abs(rc_s)):.4f}')


# ======================================================================
# ---
# ## 6. Build the Low-Frequency Model
#
# ### Theory
#
# The LFM is obtained by **low-pass filtering** the full-bandwidth impedance log.
# We use a zero-phase Butterworth filter to avoid any time shift.
#
# The cutoff frequency should be at or below the lowest usable seismic frequency.
# For typical offshore data: 6–12 Hz. The seismic band is typically
# typically 10–80 Hz, so 10 Hz is the standard default.
#
# ### Filter Design
#
# - **Type:** Butterworth low-pass (maximally flat passband)
# - **Implementation:** `sosfiltfilt` (zero-phase, second-order sections for numerical stability)
# - **Order:** 8 (sharp cutoff matching commercial tools)
#
# We also compute the **complementary high-pass** (full − low = bandlimited portion)
# to visualize exactly what the seismic sees vs. what only the well provides.
# ======================================================================

# %%  [Cell 10]
def build_lfm(impedance, dt_sec, cutoff_hz, order=8):
    """
    Build Low-Frequency Model by zero-phase Butterworth low-pass filtering.
    
    Parameters
    ----------
    impedance : array
        Full-bandwidth impedance in uniform time samples.
    dt_sec : float
        Sample interval in seconds.
    cutoff_hz : float
        Low-pass cutoff frequency in Hz.
    order : int
        Butterworth filter order.
    
    Returns
    -------
    lfm : array
        Low-frequency model (same length as input).
    hfm : array
        High-frequency (bandlimited) residual = input - lfm.
    """
    fs = 1.0 / dt_sec  # sampling frequency
    nyq = fs / 2.0
    
    if cutoff_hz >= nyq:
        raise ValueError(f'Cutoff {cutoff_hz} Hz >= Nyquist {nyq} Hz')
    
    # Handle NaNs: interpolate, filter, then restore NaN positions
    nan_mask = np.isnan(impedance)
    imp_filled = impedance.copy()
    if nan_mask.any():
        good = ~nan_mask
        imp_filled[nan_mask] = np.interp(
            np.flatnonzero(nan_mask), np.flatnonzero(good), impedance[good]
        )
    
    # Design Butterworth filter (SOS for stability)
    sos = butter(order, cutoff_hz, btype='low', fs=fs, output='sos')
    
    # Zero-phase filtering
    lfm = sosfiltfilt(sos, imp_filled)
    hfm = imp_filled - lfm
    
    # Restore NaNs
    lfm[nan_mask] = np.nan
    hfm[nan_mask] = np.nan
    
    return lfm, hfm

# Build LFM for P-impedance
ai_lfm, ai_hfm = build_lfm(ai_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)

# Build LFM for S-impedance
si_lfm, si_hfm = build_lfm(si_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)

# Build LFM for density
rhob_lfm, rhob_hfm = build_lfm(rhob_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)

print(f'LFM built with {LFM_CUTOFF_HZ} Hz cutoff, order {FILTER_ORDER}')
print(f'AI full-band mean: {np.nanmean(ai_twt):.0f}')
print(f'AI LFM mean:       {np.nanmean(ai_lfm):.0f}')
print(f'AI residual RMS:   {np.sqrt(np.nanmean(ai_hfm**2)):.0f}')


# ======================================================================
# ---
# ## 7. Cutoff Frequency Sensitivity Analysis
#
# The cutoff frequency is the single most important parameter in LFM construction.
# Too low → you lose real low-frequency structure. Too high → the model contains
# frequencies that compete with the seismic band, causing artifacts in inversion.
#
# We test multiple cutoffs to see the trade-off.
# ======================================================================

# %%  [Cell 11]
cutoffs_test = [4, 6, 8, 10, 12, 15]
twt_ms = twt_uniform * 1000

fig, axes = plt.subplots(1, len(cutoffs_test) + 1, figsize=(18, 10), sharey=True)

# Full bandwidth AI
axes[0].plot(ai_twt, twt_ms, 'k-', lw=0.5, label='Full bandwidth')
axes[0].set_xlabel('AI')
axes[0].set_ylabel('TWT (ms)')
axes[0].set_title('Full BW')
axes[0].invert_yaxis()
axes[0].legend(fontsize=8)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cutoffs_test)))

for i, fc in enumerate(cutoffs_test):
    lfm_i, _ = build_lfm(ai_twt, dt_s, fc, FILTER_ORDER)
    ax = axes[i + 1]
    ax.plot(ai_twt, twt_ms, 'k-', lw=0.3, alpha=0.4)
    ax.plot(lfm_i, twt_ms, '-', color=colors[i], lw=1.5)
    ax.set_xlabel('AI')
    ax.set_title(f'{fc} Hz', color=colors[i], fontweight='bold')
    # Highlight chosen cutoff
    if fc == LFM_CUTOFF_HZ:
        ax.set_title(f'► {fc} Hz ◄', color='red', fontweight='bold')
        ax.spines['left'].set_color('red')
        ax.spines['left'].set_linewidth(2)

fig.suptitle('LFM Cutoff Frequency Sensitivity — P-Impedance', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 8. Frequency Domain QC
#
# This is the most important QC panel. We compare the amplitude spectra of:
# 1. **Full-bandwidth AI** (what the well sees)
# 2. **LFM** (what we add below seismic band)
# 3. **Bandlimited residual** (what should overlap with seismic)
#
# The LFM should dominate below the cutoff and contribute nothing above it.
# ======================================================================

# %%  [Cell 12]
def compute_spectrum(signal, dt_sec, nperseg=None):
    """Compute smoothed amplitude spectrum using Welch's method."""
    clean = signal[np.isfinite(signal)]
    if len(clean) < 32:
        return np.array([]), np.array([])
    if nperseg is None:
        nperseg = min(len(clean), 256)
    freqs, psd = welch(clean, fs=1.0/dt_sec, nperseg=nperseg, 
                       noverlap=nperseg//2, window='hann')
    amp = np.sqrt(psd)  # amplitude spectrum
    return freqs, amp

f_full, a_full = compute_spectrum(ai_twt, dt_s)
f_lfm, a_lfm = compute_spectrum(ai_lfm, dt_s)
f_hfm, a_hfm = compute_spectrum(ai_hfm, dt_s)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot(f_full, a_full / a_full.max(), 'k-', lw=1.5, label='Full bandwidth AI')
ax1.plot(f_lfm, a_lfm / a_full.max(), 'r-', lw=2, label=f'LFM (< {LFM_CUTOFF_HZ} Hz)')
ax1.plot(f_hfm, a_hfm / a_full.max(), 'b-', lw=1, label='Bandlimited residual')
ax1.axvline(LFM_CUTOFF_HZ, color='red', ls='--', alpha=0.7, label=f'Cutoff = {LFM_CUTOFF_HZ} Hz')
ax1.axvspan(8, 80, alpha=0.08, color='blue', label='Typical seismic band (8–80 Hz)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Normalized Amplitude')
ax1.set_title('Amplitude Spectrum — Linear Scale')
ax1.set_xlim(0, 120)
ax1.legend(fontsize=9)

# Log scale (dB)
eps = 1e-10
ax2.plot(f_full, 20*np.log10(a_full+eps), 'k-', lw=1.5, label='Full bandwidth AI')
ax2.plot(f_lfm, 20*np.log10(a_lfm+eps), 'r-', lw=2, label=f'LFM')
ax2.plot(f_hfm, 20*np.log10(a_hfm+eps), 'b-', lw=1, label='Bandlimited residual')
ax2.axvline(LFM_CUTOFF_HZ, color='red', ls='--', alpha=0.7)
ax2.axvspan(8, 80, alpha=0.08, color='blue')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude (dB)')
ax2.set_title('Amplitude Spectrum — dB Scale')
ax2.set_xlim(0, 120)
ax2.legend(fontsize=9)

fig.suptitle('Frequency Domain QC: LFM vs. Full-Bandwidth AI', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 9. Time Domain QC — Multi-Property Composite Display
#
# This is the panel that should look like a STRATA or Petrel well-section display.
# Five tracks showing the full-bandwidth log (black) and LFM (red) for each property.
# ======================================================================

# %%  [Cell 13]
# Restrict to analysis window
win = (twt_uniform >= T_WIN_START) & (twt_uniform <= T_WIN_END)
twt_win = twt_uniform[win] * 1000  # ms

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.3], wspace=0.08)

tracks = [
    ('VP (m/s)', vp_twt, None, 'k'),
    ('RHOB (g/cc)', rhob_twt, rhob_lfm, 'k'),
    ('AI (P-Imp)', ai_twt, ai_lfm, 'k'),
    ('SI (S-Imp)', si_twt, si_lfm, 'k'),
    ('VP/VS', vpvs_twt, None, 'k'),
]

for i, (title, full, lfm, col) in enumerate(tracks):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(full[win], twt_win, 'k-', lw=0.6, label='Full BW')
    if lfm is not None:
        ax.plot(lfm[win], twt_win, 'r-', lw=2.0, label='LFM')
        # Shade the difference
        ax.fill_betweenx(twt_win, lfm[win], full[win], 
                         alpha=0.15, color='blue', label='Seismic band')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    if i == 0:
        ax.set_ylabel('TWT (ms)', fontsize=12)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    if lfm is not None:
        ax.legend(fontsize=7, loc='lower right')

# Empty track for legend space
ax_legend = fig.add_subplot(gs[0, 5])
ax_legend.axis('off')
ax_legend.text(0.1, 0.5, 
    f'LFM Parameters\n'
    f'─────────────\n'
    f'Cutoff: {LFM_CUTOFF_HZ} Hz\n'
    f'Order: {FILTER_ORDER}\n'
    f'Sample: {SEISMIC_DT_MS} ms\n'
    f'Window:\n'
    f'  {T_WIN_START*1000:.0f}–\n'
    f'  {T_WIN_END*1000:.0f} ms',
    transform=ax_legend.transAxes, fontsize=10, verticalalignment='center',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('OpenSI Phase 1 — Low-Frequency Model QC\n'
             f'Well 15/9-F-11 T2 | Butterworth LP {LFM_CUTOFF_HZ} Hz',
             fontsize=14, fontweight='bold', y=1.02)
plt.show()


# ======================================================================
# ---
# ## 10. Log(Impedance) Domain — For Model-Based Inversion
#
# Model-based inversion works in **log(impedance)** space (Hampson & Russell, 1991; Hampson et al., 2005):
#
# $$r_i \approx \frac{1}{2} \Delta \ln(I) = \frac{1}{2}[\ln(I_{i+1}) - \ln(I_i)]$$
#
# The forward model is: $s = \frac{1}{2} W D L_P$
#
# where $L_P = \ln(I_P)$, $D$ is the derivative matrix, and $W$ is the wavelet matrix.
#
# So we also need the LFM in log-impedance space.
# ======================================================================

# %%  [Cell 14]
# Log-impedance
ln_ai_twt = np.log(ai_twt)
ln_si_twt = np.log(si_twt)
ln_rhob_twt = np.log(rhob_twt)

# LFM in log-impedance space
ln_ai_lfm, ln_ai_hfm = build_lfm(ln_ai_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)
ln_si_lfm, ln_si_hfm = build_lfm(ln_si_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)
ln_rhob_lfm, ln_rhob_hfm = build_lfm(ln_rhob_twt, dt_s, LFM_CUTOFF_HZ, FILTER_ORDER)

# Quick QC: plot ln(AI) full vs LFM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

ax1.plot(ln_ai_twt[win], twt_win, 'k-', lw=0.6, label='ln(AI) full')
ax1.plot(ln_ai_lfm[win], twt_win, 'r-', lw=2, label='ln(AI) LFM')
ax1.set_xlabel('ln(AI)')
ax1.set_ylabel('TWT (ms)')
ax1.set_title('ln(P-Impedance)')
ax1.invert_yaxis()
ax1.legend(fontsize=9)

ax2.plot(ln_si_twt[win], twt_win, 'k-', lw=0.6, label='ln(SI) full')
ax2.plot(ln_si_lfm[win], twt_win, 'b-', lw=2, label='ln(SI) LFM')
ax2.set_xlabel('ln(SI)')
ax2.set_title('ln(S-Impedance)')
ax2.legend(fontsize=9)

fig.suptitle('Log-Impedance Domain LFM (for Model-Based Inversion)', fontsize=13)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 11. Background Trend Estimation (for Simultaneous Inversion)
#
# Simultaneous pre-stack inversion (Phase 5) requires **background trend coefficients**
# relating ln(IS) and ln(ρ) to ln(IP) (Hampson et al., 2005):
#
# $$\ln(I_S) = k \cdot \ln(I_P) + k_c + \Delta L_S$$
# $$\ln(\rho) = m \cdot \ln(I_P) + m_c + \Delta L_D$$
#
# We compute k, k_c, m, m_c here from the well data and save them for Phase 5.
# ======================================================================

# %%  [Cell 15]
# Use only valid (non-NaN) samples
mask_bg = (np.isfinite(ln_ai_twt) & np.isfinite(ln_si_twt) & 
           np.isfinite(ln_rhob_twt))

lnIP = ln_ai_twt[mask_bg]
lnIS = ln_si_twt[mask_bg]
lnRho = ln_rhob_twt[mask_bg]

# Linear regression: ln(IS) = k * ln(IP) + k_c
k, k_c = np.polyfit(lnIP, lnIS, 1)
lnIS_pred = k * lnIP + k_c
delta_LS = lnIS - lnIS_pred

# Linear regression: ln(ρ) = m * ln(IP) + m_c
m, m_c = np.polyfit(lnIP, lnRho, 1)
lnRho_pred = m * lnIP + m_c
delta_LD = lnRho - lnRho_pred

print(f'Background trend coefficients:')
print(f'  ln(IS) = {k:.4f} × ln(IP) + ({k_c:.4f})    R² = {np.corrcoef(lnIP, lnIS)[0,1]**2:.4f}')
print(f'  ln(ρ)  = {m:.4f} × ln(IP) + ({m_c:.4f})    R² = {np.corrcoef(lnIP, lnRho)[0,1]**2:.4f}')

# Cross-plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(lnIP, lnIS, c='steelblue', s=2, alpha=0.3, rasterized=True)
x_fit = np.array([lnIP.min(), lnIP.max()])
ax1.plot(x_fit, k*x_fit + k_c, 'r-', lw=2, label=f'k={k:.3f}, k_c={k_c:.3f}')
ax1.set_xlabel('ln(IP)')
ax1.set_ylabel('ln(IS)')
ax1.set_title('ln(IS) vs ln(IP) — Background Trend')
ax1.legend()

ax2.scatter(lnIP, lnRho, c='forestgreen', s=2, alpha=0.3, rasterized=True)
ax2.plot(x_fit, m*x_fit + m_c, 'r-', lw=2, label=f'm={m:.3f}, m_c={m_c:.3f}')
ax2.set_xlabel('ln(IP)')
ax2.set_ylabel('ln(ρ)')
ax2.set_title('ln(ρ) vs ln(IP) — Generalized Gardner')
ax2.legend()

fig.suptitle('Background Trend Cross-Plots (for Simultaneous Inversion)', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 12. Export All Outputs
#
# Save everything downstream phases need:
# - Full-bandwidth impedance logs in TWT
# - LFM (both linear and log domain)
# - Reflectivity series
# - Background trend coefficients
# - Uniform TWT grid
# ======================================================================

# %%  [Cell 16]
# Full-bandwidth impedance output
df_ai = pd.DataFrame({
    'TWT_s': twt_uniform,
    'TWT_ms': twt_uniform * 1000,
    'VP': vp_twt,
    'VS': vs_twt,
    'RHOB': rhob_twt,
    'AI': ai_twt,
    'SI': si_twt,
    'VPVS': vpvs_twt,
    'RC_P': rc_p,
    'RC_S': rc_s,
    'LN_AI': ln_ai_twt,
    'LN_SI': ln_si_twt,
    'LN_RHOB': ln_rhob_twt,
})
df_ai.to_csv(OUTPUT_AI_CSV, index=False, float_format='%.6f')
print(f'Saved: {OUTPUT_AI_CSV} ({len(df_ai)} samples)')

# LFM output
df_lfm = pd.DataFrame({
    'TWT_s': twt_uniform,
    'TWT_ms': twt_uniform * 1000,
    'AI_LFM': ai_lfm,
    'SI_LFM': si_lfm,
    'RHOB_LFM': rhob_lfm,
    'LN_AI_LFM': ln_ai_lfm,
    'LN_SI_LFM': ln_si_lfm,
    'LN_RHOB_LFM': ln_rhob_lfm,
    'AI_HFM': ai_hfm,   # bandlimited residual
    'SI_HFM': si_hfm,
})
df_lfm.to_csv(OUTPUT_LFM_CSV, index=False, float_format='%.6f')
print(f'Saved: {OUTPUT_LFM_CSV} ({len(df_lfm)} samples)')

# Background trend coefficients
bg_params = {
    'k': k, 'k_c': k_c,
    'm': m, 'm_c': m_c,
    'LFM_CUTOFF_HZ': LFM_CUTOFF_HZ,
    'FILTER_ORDER': FILTER_ORDER,
    'SEISMIC_DT_MS': SEISMIC_DT_MS,
}
pd.Series(bg_params).to_csv('background_trend_params.csv', header=['value'])
print(f'Saved: background_trend_params.csv')
print(f'\nBackground trend: k={k:.4f}, k_c={k_c:.4f}, m={m:.4f}, m_c={m_c:.4f}')


# ======================================================================
# ---
# ## 13. Summary Report
#
# Final composite figure suitable for reports and presentations.
# ======================================================================

# %%  [Cell 17]
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 4, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

# --- Row 1: Time-domain displays ---

# Track 1: AI
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ai_twt[win], twt_win, 'k-', lw=0.5, label='Full BW')
ax1.plot(ai_lfm[win], twt_win, 'r-', lw=2, label=f'LFM ({LFM_CUTOFF_HZ} Hz)')
ax1.fill_betweenx(twt_win, ai_lfm[win], ai_twt[win], alpha=0.12, color='blue')
ax1.set_ylabel('TWT (ms)')
ax1.set_xlabel('P-Impedance')
ax1.set_title('AI (P-Impedance)', fontweight='bold')
ax1.invert_yaxis()
ax1.legend(fontsize=8)

# Track 2: SI
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(si_twt[win], twt_win, 'k-', lw=0.5)
ax2.plot(si_lfm[win], twt_win, 'b-', lw=2)
ax2.fill_betweenx(twt_win, si_lfm[win], si_twt[win], alpha=0.12, color='green')
ax2.set_xlabel('S-Impedance')
ax2.set_title('SI (S-Impedance)', fontweight='bold')
ax2.set_yticklabels([])

# Track 3: Density
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
ax3.plot(rhob_twt[win], twt_win, 'k-', lw=0.5)
ax3.plot(rhob_lfm[win], twt_win, 'g-', lw=2)
ax3.set_xlabel('Density (g/cc)')
ax3.set_title('RHOB (Density)', fontweight='bold')
ax3.set_yticklabels([])

# Track 4: Reflectivity
ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
ax4.plot(rc_p[win], twt_win, 'k-', lw=0.4)
ax4.axvline(0, color='gray', lw=0.5)
ax4.fill_betweenx(twt_win, 0, rc_p[win], where=rc_p[win]>0, 
                   alpha=0.3, color='red', label='+RC')
ax4.fill_betweenx(twt_win, 0, rc_p[win], where=rc_p[win]<0, 
                   alpha=0.3, color='blue', label='−RC')
ax4.set_xlabel('RC')
ax4.set_title('P-Reflectivity', fontweight='bold')
ax4.set_yticklabels([])
ax4.legend(fontsize=8)

# --- Row 2: Frequency domain + cross-plots ---

# Spectrum
ax5 = fig.add_subplot(gs[1, 0:2])
ax5.plot(f_full, a_full/a_full.max(), 'k-', lw=1.5, label='Full BW AI')
ax5.plot(f_lfm, a_lfm/a_full.max(), 'r-', lw=2, label='LFM')
ax5.plot(f_hfm, a_hfm/a_full.max(), 'b--', lw=1, label='Bandlimited residual')
ax5.axvline(LFM_CUTOFF_HZ, color='red', ls=':', alpha=0.7)
ax5.axvspan(8, 80, alpha=0.06, color='blue')
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Normalized Amplitude')
ax5.set_title('Amplitude Spectra', fontweight='bold')
ax5.set_xlim(0, 100)
ax5.legend(fontsize=8)

# Cross-plot: ln(IS) vs ln(IP)
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(lnIP, lnIS, c='steelblue', s=1, alpha=0.2, rasterized=True)
ax6.plot(x_fit, k*x_fit + k_c, 'r-', lw=2)
ax6.set_xlabel('ln(IP)')
ax6.set_ylabel('ln(IS)')
ax6.set_title(f'k={k:.3f}', fontweight='bold')

# Cross-plot: ln(ρ) vs ln(IP)
ax7 = fig.add_subplot(gs[1, 3])
ax7.scatter(lnIP, lnRho, c='forestgreen', s=1, alpha=0.2, rasterized=True)
ax7.plot(x_fit, m*x_fit + m_c, 'r-', lw=2)
ax7.set_xlabel('ln(IP)')
ax7.set_ylabel('ln(ρ)')
ax7.set_title(f'm={m:.3f}', fontweight='bold')

fig.suptitle(
    'OpenSI Phase 1 — Low-Frequency Model Summary Report\n'
    f'Well 15/9-F-11 T2 | Butterworth LP {LFM_CUTOFF_HZ} Hz Order {FILTER_ORDER} | '
    f'Window {T_WIN_START*1000:.0f}–{T_WIN_END*1000:.0f} ms',
    fontsize=14, fontweight='bold', y=1.02
)
plt.savefig('Phase1_LFM_Summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print('\n' + '='*60)
print('Phase 1 COMPLETE — Low-Frequency Model Built')
print('='*60)
print(f'Outputs:')
print(f'  {OUTPUT_AI_CSV}          — Full-bandwidth logs in TWT')
print(f'  {OUTPUT_LFM_CSV}             — LFM for all inversion phases')
print(f'  background_trend_params.csv  — k, m coefficients for Phase 5')
print(f'  Phase1_LFM_Summary.png       — Summary report figure')
print(f'\nReady for Phase 2 (Recursive/Colored Inversion)')

