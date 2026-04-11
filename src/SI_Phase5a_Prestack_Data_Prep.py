#!/usr/bin/env python3
"""
OpenSI — Phase 5a: Pre-Stack Data Preparation & Background Trends

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase5a_Prestack_Data_Prep.py
"""


# ======================================================================
# # OpenSI — Phase 5a: Pre-Stack Data Preparation & Background Trends
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Prepare angle-dependent inputs for simultaneous pre-stack inversion.
# **Reference:** Fatti et al. (1994), Hampson et al. (2005), Aki & Richards (2002)
#
# ---
#
# ## What This Notebook Does
#
# 1. **Load or generate partial angle stack traces** (Near/Mid/Far) at the well location
# 2. **Extract angle-dependent wavelets** — one per angle range
# 3. **Compute background trend coefficients** (k, m) from well logs
# 4. **Compute per-angle scaling factors** (RMS_synth / RMS_real)
# 5. **Export all inputs** needed by Phase 5b (the simultaneous inversion engine)
#
# ### Two Operating Modes
#
# - **Mode A (Synthetic):** Generate angle-dependent synthetics from well logs using Aki-Richards/Fatti.
# Use this for development and validation.
# - **Mode B (Real SEG-Y):** Load actual Volve angle stacks and extract traces at the well.
# Activate when running locally with the downloaded SEG-Y files.
#
# ### Volve Angle Stack Files
#
# | Stack | File | Angles |
# |-------|------|--------|
# | Near | `ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN...segy` | 3°–17.67° |
# | Mid | `ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN...segy` | 17.67°–32.34° |
# | Far | `ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN...segy` | 32.34°–41° |
# | Full | `ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN...segy` | 3°–41° |
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Operating mode ---
USE_REAL_SEGY    = False   # Set True when running locally with SEG-Y files

# --- Phase 1 outputs ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'
BG_PARAMS_CSV    = 'background_trend_params.csv'
WAVELET_FILE     = 'final_wavelet.csv'

# --- SEG-Y files (Mode B only) ---
SEGY_NEAR = 'ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy'
SEGY_MID  = 'ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy'
SEGY_FAR  = 'ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy'
SEGY_FULL = 'ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy'

# --- Well location (IL/XL for 15/9-F-11 T2) ---
# You may need to verify these from the well header or Petrel project
WELL_INLINE    = 10208
WELL_CROSSLINE = 2320
IL_SEARCH      = 3     # Search radius in IL/XL bins around well
XL_SEARCH      = 3

# --- SEG-Y header byte positions (from the summary file) ---
IL_BYTE = 189
XL_BYTE = 193

# --- Angle definitions ---
ANGLES = {
    'near': {'min': 3.0, 'max': 17.67, 'central': 10.0},
    'mid':  {'min': 17.67, 'max': 32.34, 'central': 25.0},
    'far':  {'min': 32.34, 'max': 41.0, 'central': 37.0},
}

# --- Analysis window ---
T_WIN_START    = 2.60
T_WIN_END      = 3.26

# --- Synthetic noise (Mode A) ---
SNR_DB         = 20

# --- Output ---
OUTPUT_PREFIX  = 'Phase5a'

# %%  [Cell 2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, fftconvolve, welch
from scipy.interpolate import PchipInterpolator
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

vp = df_ai['VP'].values
vs = df_ai['VS'].values
rhob = df_ai['RHOB'].values
ai_full = df_ai['AI'].values
si_full = df_ai['SI'].values
ln_ai = df_ai['LN_AI'].values
ln_si = df_ai['LN_SI'].values
ln_rhob = df_ai['LN_RHOB'].values

ln_ai_lfm = df_lfm['LN_AI_LFM'].values
ln_si_lfm = df_lfm['LN_SI_LFM'].values
ln_rhob_lfm = df_lfm['LN_RHOB_LFM'].values

# Load P-wave wavelet (will be used as basis for angle-dependent wavelets)
wav_df = pd.read_csv(WAVELET_FILE)
wavelet_p = wav_df.iloc[:, 1].values if len(wav_df.columns) >= 2 else wav_df.iloc[:, 0].values
wavelet_p = wavelet_p / np.max(np.abs(wavelet_p))

print(f'Data: {len(twt_s)} samples, dt={dt_s*1000:.1f} ms')


# ======================================================================
# ---
# ## 1. Fatti Reflectivity — Angle-Dependent Forward Model
#
# The Fatti (1994) modification of Aki-Richards gives PP reflectivity as a function of angle:
#
# $$r_{PP}(\theta) = c_1 \cdot r_P + c_2 \cdot r_S + c_3 \cdot r_D$$
#
# where:
# - $c_1 = 1 + \tan^2\theta$
# - $c_2 = -8(V_S/V_P)^2 \sin^2\theta$
# - $c_3 = -0.5 \tan^2\theta + 2(V_S/V_P)^2 \sin^2\theta$
# - $r_P = \Delta I_P / (2 I_P)$, $r_S = \Delta I_S / (2 I_S)$, $r_D = \Delta\rho / \rho$
# ======================================================================

# %%  [Cell 3]
def fatti_reflectivity(vp, vs, rhob, theta_deg):
    """
    Compute angle-dependent PP reflectivity using Fatti et al. (1994).
    
    Parameters
    ----------
    vp, vs, rhob : arrays
        P-wave velocity, S-wave velocity, density.
    theta_deg : float
        Incidence angle in degrees.
    
    Returns
    -------
    rc_pp : array
        PP reflection coefficients at angle theta.
    rc_components : dict
        Individual components {rP, rS, rD, c1, c2, c3}.
    """
    theta = np.radians(theta_deg)
    n = len(vp)
    
    # Impedances
    ip = vp * rhob
    is_ = vs * rhob
    
    # Reflectivities: rP = ΔIP / (2*IP_avg)
    rP = np.zeros(n)
    rS = np.zeros(n)
    rD = np.zeros(n)
    
    rP[1:] = (ip[1:] - ip[:-1]) / (ip[1:] + ip[:-1])
    rS[1:] = (is_[1:] - is_[:-1]) / (is_[1:] + is_[:-1])
    rD[1:] = (rhob[1:] - rhob[:-1]) / (rhob[1:] + rhob[:-1]) * 2  # note: rD = Δρ/ρ
    
    # Average VS/VP ratio
    valid_vpvs = np.isfinite(vs) & (vs > 0) & np.isfinite(vp) & (vp > 0)
    vpvs_avg = np.nanmean(vs[valid_vpvs] / vp[valid_vpvs])
    gamma2 = vpvs_avg**2
    
    # Fatti coefficients
    c1 = 1 + np.tan(theta)**2
    c2 = -8 * gamma2 * np.sin(theta)**2
    c3 = -0.5 * np.tan(theta)**2 + 2 * gamma2 * np.sin(theta)**2
    
    # PP reflectivity
    rc_pp = c1 * rP + c2 * rS + c3 * rD
    
    # Handle NaNs
    rc_pp[np.isnan(rc_pp)] = 0.0
    
    components = {
        'rP': rP, 'rS': rS, 'rD': rD,
        'c1': c1, 'c2': c2, 'c3': c3,
        'gamma': vpvs_avg, 'theta': theta_deg,
    }
    
    return rc_pp, components


# Compute reflectivity for each angle stack
angle_data = {}
for name, adef in ANGLES.items():
    theta_c = adef['central']
    rc_angle, comps = fatti_reflectivity(vp, vs, rhob, theta_c)
    angle_data[name] = {
        'theta': theta_c,
        'rc': rc_angle,
        'components': comps,
    }
    print(f'{name:5s} (θ={theta_c:5.1f}°): c1={comps["c1"]:.3f}, '
          f'c2={comps["c2"]:.3f}, c3={comps["c3"]:.3f}, '
          f'max|rc|={np.max(np.abs(rc_angle)):.4f}')

print(f'\nAvg VS/VP = {comps["gamma"]:.3f}')


# ======================================================================
# ---
# ## 2. Generate / Load Angle Stack Traces
#
# **Mode A:** Convolve angle-dependent RC with wavelet + noise
# **Mode B:** Load from SEG-Y at well location
# ======================================================================

# %%  [Cell 4]
def make_angle_synthetic(rc, wavelet, snr_db=None, dt_sec=0.002):
    """Generate synthetic trace for one angle stack."""
    rc_c = rc.copy()
    rc_c[np.isnan(rc_c)] = 0.0
    s = fftconvolve(rc_c, wavelet, mode='same')
    if snr_db is not None:
        sp = np.mean(s**2)
        np_ = sp / (10**(snr_db/10))
        noise = np.random.randn(len(s)) * np.sqrt(np_)
        nyq = 1/(2*dt_sec)
        sos = butter(4, [5/nyq, min(80/nyq, 0.99)], btype='band', output='sos')
        noise = sosfiltfilt(sos, noise)
        noise *= np.sqrt(np_ / np.mean(noise**2))
        s += noise
    return s


if USE_REAL_SEGY:
    # ========================================
    # MODE B: Load from SEG-Y
    # ========================================
    try:
        import segyio
    except ImportError:
        print('Install segyio: pip install segyio')
        raise
    
    segy_files = {
        'near': SEGY_NEAR,
        'mid': SEGY_MID,
        'far': SEGY_FAR,
    }
    
    for name, filepath in segy_files.items():
        print(f'Loading {name}: {filepath}')
        with segyio.open(filepath, 'r', ignore_geometry=True) as f:
            n_traces = f.tracecount
            dt_segy = segyio.tools.dt(f) / 1e6  # µs → s
            n_samp = f.samples.size
            print(f'  {n_traces} traces, {n_samp} samples, dt={dt_segy*1000:.1f} ms')
            
            # Find trace at well location
            best_dist = 1e9
            best_idx = 0
            for i in range(n_traces):
                hdr = f.header[i]
                il = hdr[segyio.TraceField.INLINE_3D]
                xl = hdr[segyio.TraceField.CROSSLINE_3D]
                dist = abs(il - WELL_INLINE) + abs(xl - WELL_CROSSLINE)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_il, best_xl = il, xl
                if dist == 0:
                    break
            
            trace_data = f.trace[best_idx]
            segy_twt = f.samples / 1000.0  # ms → s
            
            print(f'  Nearest trace: IL={best_il}, XL={best_xl} (dist={best_dist})')
        
        # Resample to match our TWT grid if needed
        if abs(dt_segy - dt_s) > 1e-6:
            interp = PchipInterpolator(segy_twt, trace_data, extrapolate=False)
            trace_resampled = interp(twt_s)
            trace_resampled[np.isnan(trace_resampled)] = 0.0
        else:
            # Align by TWT
            trace_resampled = np.interp(twt_s, segy_twt, trace_data, left=0, right=0)
        
        angle_data[name]['seismic'] = trace_resampled
        angle_data[name]['source'] = 'segy'
    
    print('\nReal SEG-Y data loaded successfully.')

else:
    # ========================================
    # MODE A: Synthetic angle stacks
    # ========================================
    print('Mode A: Generating synthetic angle stacks from well logs')
    
    # Angle-dependent wavelets: apply mild NMO stretch simulation
    # At larger angles, the wavelet broadens slightly
    for name, adef in ANGLES.items():
        theta_c = adef['central']
        # Simple NMO stretch: scale wavelet by 1/cos(theta)
        stretch = 1.0 / np.cos(np.radians(theta_c))
        n_wav = len(wavelet_p)
        t_orig = np.arange(n_wav)
        t_stretched = (t_orig - n_wav//2) * stretch + n_wav//2
        wav_angle = np.interp(t_orig, t_stretched, wavelet_p, left=0, right=0)
        wav_angle = wav_angle / np.max(np.abs(wav_angle))  # renormalize
        
        # Generate synthetic
        rc_angle = angle_data[name]['rc']
        seismic_angle = make_angle_synthetic(rc_angle, wav_angle, SNR_DB, dt_s)
        
        angle_data[name]['seismic'] = seismic_angle
        angle_data[name]['wavelet'] = wav_angle
        angle_data[name]['source'] = 'synthetic'
        angle_data[name]['stretch'] = stretch
    
    print('Synthetic angle stacks generated.')
    for name in ANGLES:
        s = angle_data[name]['seismic']
        print(f'  {name}: RMS={np.sqrt(np.mean(s**2)):.6f}, '
              f'stretch={angle_data[name].get("stretch",1):.3f}')

# %%  [Cell 5]
# --- Visualize angle stacks ---
win = (twt_s >= T_WIN_START) & (twt_s <= T_WIN_END)
t_win = twt_ms[win]

fig, axes = plt.subplots(1, 4, figsize=(16, 10), sharey=True)

for i, (name, color) in enumerate([('near','blue'), ('mid','green'), ('far','red')]):
    s = angle_data[name]['seismic'][win]
    ax = axes[i]
    ax.plot(s, t_win, f'{color[0]}-', lw=0.5)
    ax.fill_betweenx(t_win, 0, s, where=s>0, alpha=0.25, color='red')
    ax.fill_betweenx(t_win, 0, s, where=s<0, alpha=0.25, color='blue')
    theta = angle_data[name]['theta']
    ax.set_title(f'{name.upper()} ({theta:.0f}°)', fontweight='bold', color=color)
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()

# Overlay all
ax = axes[3]
for name, color in [('near','blue'), ('mid','green'), ('far','red')]:
    ax.plot(angle_data[name]['seismic'][win], t_win, '-', color=color, lw=0.5, 
            alpha=0.7, label=f'{name} ({angle_data[name]["theta"]:.0f}°)')
ax.set_title('Overlay', fontweight='bold')
ax.legend(fontsize=9)

source_label = 'Synthetic (from well logs)' if not USE_REAL_SEGY else 'Real SEG-Y'
fig.suptitle(f'Angle Stack Traces at Well — {source_label}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 3. Angle-Dependent Wavelet Extraction
#
# Each angle stack has a different effective wavelet due to NMO stretch and
# processing differences. In Mode A we simulate this; in Mode B we extract
# from log-seismic cross-correlation.
#
# For Mode B (real data), the wavelet is extracted by:
# 1. Computing reflectivity from well logs at the angle (Fatti)
# 2. Solving: TRACE = W * RC → W = least-squares shaping filter
# ======================================================================

# %%  [Cell 6]
def extract_wavelet_from_well(seismic, rc, wav_length=60, dt_sec=0.002):
    """
    Extract wavelet by least-squares shaping filter: s ≈ W * rc.
    
    Uses Wiener-Levinson approach: autocorrelation of RC, cross-correlation with trace.
    """
    n = len(seismic)
    half = wav_length // 2
    
    s = seismic.copy()
    r = rc.copy()
    s[np.isnan(s)] = 0.0
    r[np.isnan(r)] = 0.0
    
    # Build RC convolution matrix
    R_mat = np.zeros((n, wav_length))
    for j in range(wav_length):
        shift = j - half
        if shift >= 0:
            R_mat[shift:, j] = r[:n-shift]
        else:
            R_mat[:n+shift, j] = r[-shift:]
    
    # Least-squares: W = (R^T R + λI)^-1 R^T s
    lam = 0.01 * np.max(np.diag(R_mat.T @ R_mat))
    RtR = R_mat.T @ R_mat + lam * np.eye(wav_length)
    Rts = R_mat.T @ s
    wav = np.linalg.solve(RtR, Rts)
    wav = wav / np.max(np.abs(wav))
    
    # Compute CC
    synth = R_mat @ wav
    cc = np.corrcoef(s, synth)[0, 1]
    
    return wav, cc


# Extract or assign wavelets
wavelets = {}

if USE_REAL_SEGY:
    # Extract from real data using well tie
    for name in ANGLES:
        s_win = angle_data[name]['seismic'][win]
        rc_win = angle_data[name]['rc'][win]
        wav, cc = extract_wavelet_from_well(s_win, rc_win, 
                                             wav_length=len(wavelet_p))
        wavelets[name] = wav
        print(f'{name}: extracted wavelet CC = {cc:.4f}')
else:
    # Use the simulated stretched wavelets
    for name in ANGLES:
        wavelets[name] = angle_data[name]['wavelet']
        print(f'{name}: using NMO-stretched wavelet (stretch={angle_data[name]["stretch"]:.3f})')

# Plot wavelets
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = {'near': 'blue', 'mid': 'green', 'far': 'red'}

for i, name in enumerate(['near', 'mid', 'far']):
    w = wavelets[name]
    t_w = (np.arange(len(w)) - len(w)//2) * dt_s * 1000
    axes[i].plot(t_w, w, '-', color=colors[name], lw=1.5)
    axes[i].axhline(0, color='gray', lw=0.5)
    axes[i].set_xlabel('Time (ms)')
    axes[i].set_title(f'{name.upper()} wavelet ({angle_data[name]["theta"]:.0f}°)',
                       fontweight='bold', color=colors[name])
    if i == 0:
        axes[i].set_ylabel('Amplitude')

fig.suptitle('Angle-Dependent Wavelets', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 4. Background Trend Coefficients (k, m)
#
# From Hampson et al. (2005):
# $$\ln(I_S) = k \cdot \ln(I_P) + k_c + \Delta L_S$$
# $$\ln(\rho) = m \cdot \ln(I_P) + m_c + \Delta L_D$$
#
# These were pre-computed in Phase 1. We reload and verify.
# ======================================================================

# %%  [Cell 7]
# Load or recompute background trends
try:
    bg = pd.read_csv(BG_PARAMS_CSV, index_col=0)
    k = float(bg.loc['k', 'value'])
    k_c = float(bg.loc['k_c', 'value'])
    m = float(bg.loc['m', 'value'])
    m_c = float(bg.loc['m_c', 'value'])
    print(f'Loaded from Phase 1:')
except:
    # Recompute
    mask = np.isfinite(ln_ai) & np.isfinite(ln_si) & np.isfinite(ln_rhob)
    k, k_c = np.polyfit(ln_ai[mask], ln_si[mask], 1)
    m, m_c = np.polyfit(ln_ai[mask], ln_rhob[mask], 1)
    print(f'Recomputed:')

print(f'  ln(IS) = {k:.4f} × ln(IP) + ({k_c:.4f})')
print(f'  ln(ρ)  = {m:.4f} × ln(IP) + ({m_c:.4f})')

# Verify: compute ΔLS and ΔLD
mask = np.isfinite(ln_ai) & np.isfinite(ln_si) & np.isfinite(ln_rhob)
delta_LS = ln_si - (k * ln_ai + k_c)
delta_LD = ln_rhob - (m * ln_ai + m_c)

print(f'\n  ΔLS: mean={np.nanmean(delta_LS):.4f}, std={np.nanstd(delta_LS):.4f}')
print(f'  ΔLD: mean={np.nanmean(delta_LD):.4f}, std={np.nanstd(delta_LD):.4f}')

# Cross-plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x_fit = np.array([np.nanmin(ln_ai[mask]), np.nanmax(ln_ai[mask])])

ax1.scatter(ln_ai[mask], ln_si[mask], c='steelblue', s=2, alpha=0.3, rasterized=True)
ax1.plot(x_fit, k*x_fit + k_c, 'r-', lw=2, label=f'k={k:.3f}')
ax1.set_xlabel('ln(IP)')
ax1.set_ylabel('ln(IS)')
ax1.set_title('S-Impedance vs P-Impedance', fontweight='bold')
ax1.legend()

ax2.scatter(ln_ai[mask], ln_rhob[mask], c='forestgreen', s=2, alpha=0.3, rasterized=True)
ax2.plot(x_fit, m*x_fit + m_c, 'r-', lw=2, label=f'm={m:.3f}')
ax2.set_xlabel('ln(IP)')
ax2.set_ylabel('ln(ρ)')
ax2.set_title('Density vs P-Impedance (Gardner)', fontweight='bold')
ax2.legend()

fig.suptitle('Background Trends for Simultaneous Inversion', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 5. Compute Fatti Coefficients for Each Angle
#
# The modified Fatti coefficients incorporating background trends (Hampson et al., 2005):
#
# $$\tilde{c}_1 = \frac{c_1}{2} + \frac{k \cdot c_2}{2} + m \cdot c_3$$
# $$\tilde{c}_2 = \frac{c_2}{2}$$
#
# These multiply the wavelet and derivative matrices in the simultaneous system.
# ======================================================================

# %%  [Cell 8]
# Compute modified Fatti coefficients for each angle
for name in ANGLES:
    comps = angle_data[name]['components']
    c1, c2, c3 = comps['c1'], comps['c2'], comps['c3']
    
    # Modified coefficients (Hampson et al., 2005)
    c1_tilde = c1/2 + k*c2/2 + m*c3
    c2_tilde = c2/2
    # c3 stays as is (for density deviation)
    
    angle_data[name]['c1_tilde'] = c1_tilde
    angle_data[name]['c2_tilde'] = c2_tilde
    angle_data[name]['c3_fatti'] = c3
    
    print(f'{name:5s} (θ={angle_data[name]["theta"]:5.1f}°): '
          f'c̃₁={c1_tilde:.4f}, c̃₂={c2_tilde:.4f}, c₃={c3:.4f}')


# ======================================================================
# ---
# ## 6. Export All Phase 5a Outputs
#
# Everything Phase 5b needs in one clean package.
# ======================================================================

# %%  [Cell 9]
# --- Export angle stack traces ---
df_angle = pd.DataFrame({'TWT_s': twt_s, 'TWT_ms': twt_ms})
for name in ANGLES:
    df_angle[f'Seismic_{name}'] = angle_data[name]['seismic']
    df_angle[f'RC_{name}'] = angle_data[name]['rc']

df_angle.to_csv(f'{OUTPUT_PREFIX}_angle_traces.csv', index=False, float_format='%.8f')
print(f'Saved: {OUTPUT_PREFIX}_angle_traces.csv')

# --- Export wavelets ---
for name in ANGLES:
    w = wavelets[name]
    t_w = (np.arange(len(w)) - len(w)//2) * dt_s * 1000
    pd.DataFrame({'time_ms': t_w, 'amplitude': w}).to_csv(
        f'{OUTPUT_PREFIX}_wavelet_{name}.csv', index=False, float_format='%.8f')
print(f'Saved: {OUTPUT_PREFIX}_wavelet_near/mid/far.csv')

# --- Export simultaneous inversion parameters ---
sim_params = {
    'k': k, 'k_c': k_c,
    'm': m, 'm_c': m_c,
}
for name in ANGLES:
    sim_params[f'theta_{name}'] = angle_data[name]['theta']
    sim_params[f'c1_tilde_{name}'] = angle_data[name]['c1_tilde']
    sim_params[f'c2_tilde_{name}'] = angle_data[name]['c2_tilde']
    sim_params[f'c3_{name}'] = angle_data[name]['c3_fatti']

pd.Series(sim_params).to_csv(f'{OUTPUT_PREFIX}_sim_params.csv', header=['value'])
print(f'Saved: {OUTPUT_PREFIX}_sim_params.csv')

# Summary
print(f'\n{"="*60}')
print(f'Phase 5a COMPLETE — Pre-Stack Data Preparation')
print(f'{"="*60}')
print(f'  Source: {"Real SEG-Y" if USE_REAL_SEGY else "Synthetic from well logs"}')
print(f'  Angles: Near={ANGLES["near"]["central"]:.0f}°, '
      f'Mid={ANGLES["mid"]["central"]:.0f}°, Far={ANGLES["far"]["central"]:.0f}°')
print(f'  Background: k={k:.4f}, m={m:.4f}')
print(f'\nOutputs:')
print(f'  {OUTPUT_PREFIX}_angle_traces.csv')
print(f'  {OUTPUT_PREFIX}_wavelet_near/mid/far.csv')
print(f'  {OUTPUT_PREFIX}_sim_params.csv')
print(f'\nReady for Phase 5b (Simultaneous Inversion Engine)')

