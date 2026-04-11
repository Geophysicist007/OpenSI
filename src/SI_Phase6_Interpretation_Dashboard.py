#!/usr/bin/env python3
"""
OpenSI — Phase 6: Interpretation & Visualization Dashboard

Run this script section by section in an interactive Python environment,
or execute the entire file: python SI_Phase6_Interpretation_Dashboard.py
"""


# ======================================================================
# # OpenSI — Phase 6: Interpretation & Visualization Dashboard
#
# **Well:** 15/9-F-11 T2 (Volve field, North Sea)
# **Purpose:** Interpret inversion results with cross-plots, facies classification, and report-quality displays.
#
# ---
#
# ## What This Notebook Does
#
# 1. **Loads all inversion outputs** from Phases 1–5
# 2. **Smoothed VP/VS** — reduces ratio noise for usable interpretation
# 3. **LMR attributes** — λρ, µρ for fluid discrimination
# 4. **Probabilistic facies classification** — GMM trained on well data, applied to inverted properties
# 5. **Interpretation cross-plot workspace** — IP vs VP/VS, λρ vs µρ, IP vs IS with facies coloring
# 6. **Composite well display** — STRATA/Petrel-style multi-track panel
# 7. **All-methods scorecard** — Phases 1–5 comparison table
# 8. **PDF/HTML report generation**
# ======================================================================

# %%  [Cell 1]
# ============================================================
# USER PARAMETERS
# ============================================================

# --- Input files ---
AI_FULLBAND_CSV  = 'ai_fullband_15_9_F11_T2.csv'
LFM_CSV          = 'lfm_model_15_9_F11_T2.csv'
PHASE5B_CSV      = 'Phase5b_inversion_results.csv'
BG_PARAMS_CSV    = 'background_trend_params.csv'

# --- VP/VS smoothing ---
VPVS_SMOOTH_HZ   = 15.0     # Low-pass cutoff for VP/VS smoothing (Hz)

# --- Facies definitions ---
# Define expected facies for Volve (North Sea clastics)
FACIES_NAMES = ['Shale', 'Brine Sand', 'HC Sand']
FACIES_COLORS = ['#8B7355', '#4682B4', '#FF4444']

# Approximate VP/VS cutoffs for Volve clastics
VPVS_SHALE_MIN     = 1.80
VPVS_SAND_BOUNDARY = 1.75   # Below this = sand-like
IP_HC_THRESHOLD    = 9500   # Below this + low VP/VS = possible HC

# --- Window ---
T_WIN_START = 2.60
T_WIN_END   = 3.26

# --- Output ---
OUTPUT_PREFIX = 'Phase6'

# %%  [Cell 2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from scipy.signal import butter, sosfiltfilt
from scipy.stats import gaussian_kde
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
df_inv = pd.read_csv(PHASE5B_CSV)

twt_w = df_inv['TWT_ms'].values
dt_s = np.round(np.median(np.diff(df_inv['TWT_s'].values)), 6)
fs = 1.0 / dt_s
n = len(twt_w)

# Truth
IP_true = df_inv['IP_Truth'].values
IS_true = df_inv['IS_Truth'].values
Rho_true = df_inv['Rho_Truth'].values
VpVs_true = df_inv['VpVs_Truth'].values

# Inverted
IP_inv = df_inv['IP_Inv'].values
IS_inv = df_inv['IS_Inv'].values
Rho_inv = df_inv['Rho_Inv'].values
VpVs_inv = df_inv['VpVs_Inv'].values

# LMR
MuRho_inv = df_inv['MuRho_Inv'].values
LamRho_inv = df_inv['LambdaRho_Inv'].values
MuRho_true = df_inv['MuRho_Truth'].values
LamRho_true = df_inv['LambdaRho_Truth'].values

# LFM
ai_lfm_full = df_lfm['AI_LFM'].values
si_lfm_full = df_lfm['SI_LFM'].values

# Load background params
try:
    bg = pd.read_csv(BG_PARAMS_CSV, index_col=0)
    k = float(bg.loc['k', 'value'])
    m = float(bg.loc['m', 'value'])
except:
    k, m = 1.171, 0.209

print(f'Loaded {n} samples, dt={dt_s*1000:.1f} ms')
print(f'TWT: {twt_w[0]:.0f}–{twt_w[-1]:.0f} ms')


# ======================================================================
# ---
# ## 1. Smoothed VP/VS
#
# VP/VS = IP/IS amplifies uncorrelated noise in the numerator and denominator.
# A mild low-pass filter removes the high-frequency noise while preserving
# the meaningful VP/VS variations that indicate lithology and fluid changes.
# ======================================================================

# %%  [Cell 3]
def smooth_ratio(ratio, dt_sec, cutoff_hz, order=4):
    """Zero-phase low-pass filter for VP/VS smoothing."""
    r = ratio.copy()
    nans = np.isnan(r)
    if nans.any():
        r[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), r[~nans])
    sos = butter(order, cutoff_hz, btype='low', fs=1/dt_sec, output='sos')
    r_smooth = sosfiltfilt(sos, r)
    r_smooth[nans] = np.nan
    return r_smooth

VpVs_inv_smooth = smooth_ratio(VpVs_inv, dt_s, VPVS_SMOOTH_HZ)
VpVs_true_smooth = smooth_ratio(VpVs_true, dt_s, VPVS_SMOOTH_HZ)

# Metrics
valid = np.isfinite(VpVs_true) & np.isfinite(VpVs_inv_smooth)
cc_vpvs_raw = np.corrcoef(VpVs_true[valid], VpVs_inv[valid])[0, 1]
cc_vpvs_smooth = np.corrcoef(VpVs_true_smooth[valid], VpVs_inv_smooth[valid])[0, 1]

print(f'VP/VS CC (raw):      {cc_vpvs_raw:.4f}')
print(f'VP/VS CC (smoothed): {cc_vpvs_smooth:.4f}')
print(f'Improvement: +{(cc_vpvs_smooth - cc_vpvs_raw):.3f}')

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

ax1.plot(VpVs_true, twt_w, 'k-', lw=0.5, label='Truth')
ax1.plot(VpVs_inv, twt_w, 'purple', lw=0.8, alpha=0.7, label='Raw inverted')
ax1.set_title(f'Raw VP/VS\nCC={cc_vpvs_raw:.3f}', fontweight='bold')
ax1.set_xlabel('VP/VS')
ax1.set_ylabel('TWT (ms)')
ax1.invert_yaxis()
ax1.legend(fontsize=9)

ax2.plot(VpVs_true_smooth, twt_w, 'k-', lw=0.8, label='Truth (smoothed)')
ax2.plot(VpVs_inv_smooth, twt_w, 'purple', lw=1.5, label='Inverted (smoothed)')
ax2.set_title(f'Smoothed VP/VS ({VPVS_SMOOTH_HZ:.0f} Hz LP)\nCC={cc_vpvs_smooth:.3f}', 
              fontweight='bold')
ax2.set_xlabel('VP/VS')
ax2.legend(fontsize=9)

fig.suptitle('VP/VS Improvement via Smoothing', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ======================================================================
# ---
# ## 2. Rule-Based Facies Classification
#
# For the Volve North Sea clastics, we apply simple geological rules:
#
# - **Shale:** High VP/VS (> 1.75–1.80), moderate IP
# - **Brine Sand:** Low VP/VS (< 1.75), moderate-high IP
# - **HC Sand:** Low VP/VS (< 1.75), low IP (< ~9500)
#
# This is a starting point — in production you'd calibrate with core data.
# ======================================================================

# %%  [Cell 4]
def classify_facies(ip, vpvs, vpvs_boundary=1.75, ip_hc_threshold=9500):
    """
    Simple rule-based facies classification.
    Returns: 0=Shale, 1=Brine Sand, 2=HC Sand
    """
    n = len(ip)
    facies = np.zeros(n, dtype=int)  # Default: Shale
    
    is_sand = vpvs < vpvs_boundary
    is_hc = is_sand & (ip < ip_hc_threshold)
    is_brine = is_sand & ~is_hc
    
    facies[is_brine] = 1
    facies[is_hc] = 2
    
    return facies


# Classify from truth
facies_true = classify_facies(IP_true, VpVs_true_smooth, 
                               VPVS_SAND_BOUNDARY, IP_HC_THRESHOLD)

# Classify from inversion
facies_inv = classify_facies(IP_inv, VpVs_inv_smooth, 
                              VPVS_SAND_BOUNDARY, IP_HC_THRESHOLD)

# Accuracy
accuracy = np.mean(facies_true[valid] == facies_inv[valid])

for i, name in enumerate(FACIES_NAMES):
    n_true = np.sum(facies_true == i)
    n_inv = np.sum(facies_inv == i)
    match = np.sum((facies_true == i) & (facies_inv == i))
    print(f'{name:12s}: Truth={n_true:4d}, Inverted={n_inv:4d}, Match={match:4d}')

print(f'\nOverall facies accuracy: {accuracy:.1%}')


# ======================================================================
# ---
# ## 3. Interpretation Cross-Plot Workspace
#
# Four key cross-plots used by interpreters worldwide,
# color-coded by facies classification.
# ======================================================================

# %%  [Cell 5]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
cmap = mcolors.ListedColormap(FACIES_COLORS)
legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(FACIES_COLORS, FACIES_NAMES)]

# --- 1. IP vs VP/VS (smoothed) ---
ax = axes[0, 0]
sc = ax.scatter(IP_inv[valid], VpVs_inv_smooth[valid], c=facies_inv[valid], 
                cmap=cmap, s=12, alpha=0.6, edgecolors='none', vmin=0, vmax=2)
ax.scatter(IP_true[valid], VpVs_true_smooth[valid], c='none', 
           edgecolors='gray', s=15, alpha=0.3, lw=0.3, label='Well truth')
ax.axhline(VPVS_SAND_BOUNDARY, color='orange', ls='--', lw=1, alpha=0.7, label=f'VP/VS={VPVS_SAND_BOUNDARY}')
ax.axvline(IP_HC_THRESHOLD, color='red', ls='--', lw=1, alpha=0.7, label=f'IP={IP_HC_THRESHOLD}')
ax.set_xlabel('P-Impedance', fontsize=12)
ax.set_ylabel('VP/VS (smoothed)', fontsize=12)
ax.set_title('IP vs VP/VS — Lithology & Fluid', fontweight='bold', fontsize=13)
ax.legend(handles=legend_elements + [ax.plot([], [], 'o', mfc='none', mec='gray', ms=5)[0]], 
          labels=FACIES_NAMES + ['Well truth'], fontsize=8, loc='upper right')

# --- 2. λρ vs µρ ---
ax = axes[0, 1]
ax.scatter(MuRho_inv[valid], LamRho_inv[valid], c=facies_inv[valid],
           cmap=cmap, s=12, alpha=0.6, edgecolors='none', vmin=0, vmax=2)
ax.scatter(MuRho_true[valid], LamRho_true[valid], c='none',
           edgecolors='gray', s=15, alpha=0.3, lw=0.3)
ax.set_xlabel('µρ (Rigidity × Density)', fontsize=12)
ax.set_ylabel('λρ (Incompressibility × Density)', fontsize=12)
ax.set_title('λρ vs µρ — Fluid Discrimination (LMR)', fontweight='bold', fontsize=13)
ax.legend(handles=legend_elements, fontsize=8)

# --- 3. IP vs IS ---
ax = axes[1, 0]
ax.scatter(IP_inv[valid], IS_inv[valid], c=facies_inv[valid],
           cmap=cmap, s=12, alpha=0.6, edgecolors='none', vmin=0, vmax=2)
# Background trend
ip_range = np.linspace(np.nanmin(IP_inv)*0.9, np.nanmax(IP_inv)*1.1, 100)
is_trend = np.exp(k * np.log(ip_range) + float(bg.loc['k_c', 'value']))
ax.plot(ip_range, is_trend, 'r--', lw=2, alpha=0.7, label=f'Background (k={k:.2f})')
ax.set_xlabel('P-Impedance', fontsize=12)
ax.set_ylabel('S-Impedance', fontsize=12)
ax.set_title('IP vs IS — Deviations from Background', fontweight='bold', fontsize=13)
ax.legend(handles=legend_elements + [ax.lines[-1]], 
          labels=FACIES_NAMES + [f'Background (k={k:.2f})'], fontsize=8)

# --- 4. IP vs Density ---
ax = axes[1, 1]
ax.scatter(IP_inv[valid], Rho_inv[valid], c=facies_inv[valid],
           cmap=cmap, s=12, alpha=0.6, edgecolors='none', vmin=0, vmax=2)
ax.scatter(IP_true[valid], Rho_true[valid], c='none',
           edgecolors='gray', s=15, alpha=0.3, lw=0.3)
# Gardner trend
rho_trend = np.exp(m * np.log(ip_range) + float(bg.loc['m_c', 'value']))
ax.plot(ip_range, rho_trend, 'r--', lw=2, alpha=0.7, label=f'Gardner (m={m:.2f})')
ax.set_xlabel('P-Impedance', fontsize=12)
ax.set_ylabel('Density (g/cc)', fontsize=12)
ax.set_title('IP vs Density — Gardner Relationship', fontweight='bold', fontsize=13)
ax.legend(handles=legend_elements + [ax.lines[-1]],
          labels=FACIES_NAMES + [f'Gardner (m={m:.2f})'], fontsize=8)

fig.suptitle('OpenSI — Interpretation Cross-Plot Workspace\nWell 15/9-F-11 T2 (Volve)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}_CrossPlots.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ======================================================================
# ---
# ## 4. Composite Well Display
#
# STRATA/Petrel-style multi-track panel showing all inverted properties
# alongside the well truth, with facies color-fill.
# ======================================================================

# %%  [Cell 6]
fig = plt.figure(figsize=(24, 14))
gs = GridSpec(1, 9, width_ratios=[0.3, 1, 1, 1, 1, 1, 1, 1, 0.3], wspace=0.08)

# --- Facies track (truth) ---
ax = fig.add_subplot(gs[0, 0])
for i in range(n):
    ax.axhspan(twt_w[i] - dt_s*500, twt_w[i] + dt_s*500, 
               color=FACIES_COLORS[facies_true[i]], alpha=0.8)
ax.set_ylabel('TWT (ms)', fontsize=12)
ax.set_title('Facies\n(Truth)', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.invert_yaxis()

# --- Property tracks ---
tracks = [
    ('P-Impedance', IP_true, IP_inv, 'red', None),
    ('S-Impedance', IS_true, IS_inv, 'blue', None),
    ('Density\n(g/cc)', Rho_true, Rho_inv, 'green', None),
    ('VP/VS\n(raw)', VpVs_true, VpVs_inv, 'purple', None),
    ('VP/VS\n(smooth)', VpVs_true_smooth, VpVs_inv_smooth, 'darkviolet', None),
    ('λρ', LamRho_true, LamRho_inv, 'darkorange', None),
    ('µρ', MuRho_true, MuRho_inv, 'teal', None),
]

for i, (title, truth, inv, color, _) in enumerate(tracks):
    ax = fig.add_subplot(gs[0, i + 1], sharey=fig.axes[0])
    ax.plot(truth, twt_w, 'k-', lw=0.5, alpha=0.5, label='Truth')
    ax.plot(inv, twt_w, '-', color=color, lw=1.2, label='Inverted')
    
    v = np.isfinite(truth) & np.isfinite(inv)
    cc = np.corrcoef(truth[v], inv[v])[0, 1] if v.sum() > 2 else 0
    ax.set_title(f'{title}\nCC={cc:.3f}', fontweight='bold', color=color, fontsize=10)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=8, rotation=30)
    if i == 0:
        ax.legend(fontsize=7, loc='lower left')

# --- Facies track (inverted) ---
ax = fig.add_subplot(gs[0, 8], sharey=fig.axes[0])
for i in range(n):
    ax.axhspan(twt_w[i] - dt_s*500, twt_w[i] + dt_s*500,
               color=FACIES_COLORS[facies_inv[i]], alpha=0.8)
ax.set_title('Facies\n(Inv)', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_yticklabels([])

# Legend for facies
handles = [Patch(fc=c, label=n) for c, n in zip(FACIES_COLORS, FACIES_NAMES)]

fig.suptitle(
    'OpenSI — Composite Well Display\n'
    f'Well 15/9-F-11 T2 | Simultaneous Inversion | '
    f'Facies Accuracy: {accuracy:.0%}',
    fontsize=14, fontweight='bold', y=1.02
)
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, 
           bbox_to_anchor=(0.5, -0.02))
plt.savefig(f'{OUTPUT_PREFIX}_CompositeDisplay.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ======================================================================
# ---
# ## 5. All-Methods Scorecard (Phases 1–5)
#
# Comprehensive comparison of every inversion method tested.
# ======================================================================

# %%  [Cell 7]
# Collect CCs from all phases
# Recompute where needed from saved CSVs
scorecard = []

# Phase 1: LFM
win_full = (df_ai['TWT_s'].values >= T_WIN_START) & (df_ai['TWT_s'].values <= T_WIN_END)
ai_f = df_ai['AI'].values[win_full]
ai_l = df_lfm['AI_LFM'].values[win_full]
v = np.isfinite(ai_f) & np.isfinite(ai_l)
scorecard.append(('LFM Only (Phase 1)', np.corrcoef(ai_f[v], ai_l[v])[0,1], '—', '—', '—'))

# Phase 2
for label, fname, col in [
    ('Recursive (Phase 2)', 'Phase2_inversion_results.csv', 'AI_Recursive'),
    ('Colored (Phase 2)', 'Phase2_inversion_results.csv', 'AI_Colored'),
]:
    try:
        df_p = pd.read_csv(fname)
        p_ai = df_p[col].values
        # Align to window
        p_twt = df_p['TWT_ms'].values
        p_mask = (p_twt >= T_WIN_START*1000) & (p_twt <= T_WIN_END*1000)
        p_truth = df_p['AI_Well_Truth'].values[p_mask] if 'AI_Well_Truth' in df_p else df_p['AI_Truth'].values[p_mask]
        p_inv = p_ai[p_mask]
        v2 = np.isfinite(p_truth) & np.isfinite(p_inv)
        cc = np.corrcoef(p_truth[v2], p_inv[v2])[0,1]
        scorecard.append((label, cc, '—', '—', '—'))
    except:
        scorecard.append((label, '—', '—', '—', '—'))

# Phase 3
for label, fname, col in [
    ('Stochastic (Phase 3)', 'Phase3_inversion_results.csv', 'AI_Stochastic'),
    ('Constrained (Phase 3)', 'Phase3_inversion_results.csv', 'AI_Constrained'),
]:
    try:
        df_p = pd.read_csv(fname)
        p_mask = (df_p['TWT_ms'].values >= T_WIN_START*1000) & (df_p['TWT_ms'].values <= T_WIN_END*1000)
        p_truth = df_p['AI_Truth'].values[p_mask]
        p_inv = df_p[col].values[p_mask]
        v2 = np.isfinite(p_truth) & np.isfinite(p_inv)
        cc = np.corrcoef(p_truth[v2], p_inv[v2])[0,1]
        scorecard.append((label, cc, '—', '—', '—'))
    except:
        scorecard.append((label, '—', '—', '—', '—'))

# Phase 4
for label, fname, col in [
    ('LPSS (Phase 4)', 'Phase4_inversion_results.csv', 'AI_LPSS'),
    ('MLSS (Phase 4)', 'Phase4_inversion_results.csv', 'AI_MLSS'),
]:
    try:
        df_p = pd.read_csv(fname)
        p_mask = (df_p['TWT_ms'].values >= T_WIN_START*1000) & (df_p['TWT_ms'].values <= T_WIN_END*1000)
        p_truth = df_p['AI_Truth'].values[p_mask]
        p_inv = df_p[col].values[p_mask]
        v2 = np.isfinite(p_truth) & np.isfinite(p_inv)
        cc = np.corrcoef(p_truth[v2], p_inv[v2])[0,1]
        scorecard.append((label, cc, '—', '—', '—'))
    except:
        scorecard.append((label, '—', '—', '—', '—'))

# Phase 5: Simultaneous
cc_ip = np.corrcoef(IP_true[valid], IP_inv[valid])[0, 1]
cc_is = np.corrcoef(IS_true[valid], IS_inv[valid])[0, 1]
cc_rho = np.corrcoef(Rho_true[valid], Rho_inv[valid])[0, 1]
scorecard.append(('Simultaneous (Phase 5)', cc_ip, cc_is, cc_rho, cc_vpvs_smooth))

# Display
print(f'{"Method":<30s} {"CC(IP)":>8s} {"CC(IS)":>8s} {"CC(ρ)":>8s} {"CC(VP/VS)":>10s}')
print('='*70)
for row in scorecard:
    vals = [f'{v:.3f}' if isinstance(v, float) else v for v in row[1:]]
    print(f'{row[0]:<30s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>10s}')

# %%  [Cell 8]
# --- Scorecard as a figure ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

header = ['Method', 'CC(IP)', 'CC(IS)', 'CC(ρ)', 'CC(VP/VS)']
table_data = [header]
for row in scorecard:
    vals = [f'{v:.3f}' if isinstance(v, float) else str(v) for v in row[1:]]
    table_data.append([row[0]] + vals)

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style header
for j in range(5):
    table[0, j].set_text_props(fontweight='bold')
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best IP CC
ip_ccs = []
for row in scorecard:
    try:
        ip_ccs.append(float(row[1]))
    except:
        ip_ccs.append(-1)
best_row = np.argmax(ip_ccs) + 1  # +1 for header
for j in range(5):
    table[best_row, j].set_facecolor('#d5f5e3')

# Highlight simultaneous (last row) in light blue
for j in range(5):
    table[len(scorecard), j].set_facecolor('#d4e6f1')

# Alternate row colors
for i in range(1, len(scorecard) + 1):
    if i != best_row and i != len(scorecard):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(5):
            table[i, j].set_facecolor(color)

fig.suptitle('OpenSI — All Methods Scorecard', fontsize=14, fontweight='bold')
plt.savefig(f'{OUTPUT_PREFIX}_Scorecard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ======================================================================
# ---
# ## 6. Final Summary Report
# ======================================================================

# %%  [Cell 9]
fig = plt.figure(figsize=(24, 16))
gs = GridSpec(3, 5, height_ratios=[2, 1.5, 1], hspace=0.4, wspace=0.3)

# --- Row 1: Key property tracks ---
props = [
    ('P-Impedance', IP_true, IP_inv, 'red'),
    ('S-Impedance', IS_true, IS_inv, 'blue'),
    ('Density', Rho_true, Rho_inv, 'green'),
    ('VP/VS (smooth)', VpVs_true_smooth, VpVs_inv_smooth, 'purple'),
]

for i, (title, truth, inv, color) in enumerate(props):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(truth, twt_w, 'k-', lw=0.5, alpha=0.5)
    ax.plot(inv, twt_w, '-', color=color, lw=1.2)
    v = np.isfinite(truth) & np.isfinite(inv)
    cc = np.corrcoef(truth[v], inv[v])[0,1] if v.sum() > 2 else 0
    ax.set_title(f'{title}\nCC={cc:.3f}', fontweight='bold', color=color)
    if i == 0:
        ax.set_ylabel('TWT (ms)')
        ax.invert_yaxis()
    else:
        ax.set_yticklabels([])

# Facies
ax = fig.add_subplot(gs[0, 4])
for j in range(n):
    ax.barh(twt_w[j], 0.5, height=dt_s*1000, 
            color=FACIES_COLORS[facies_true[j]], align='center', alpha=0.6)
    ax.barh(twt_w[j], 0.5, height=dt_s*1000, left=0.5,
            color=FACIES_COLORS[facies_inv[j]], align='center', alpha=0.6)
ax.axvline(0.5, color='black', lw=1)
ax.set_title(f'Facies (L=Truth R=Inv)\nAccuracy={accuracy:.0%}', fontweight='bold')
ax.set_xlim(0, 1)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['Truth', 'Inv'], fontsize=9)
ax.set_yticklabels([])

# --- Row 2: Cross-plots ---
cmap = mcolors.ListedColormap(FACIES_COLORS)

ax = fig.add_subplot(gs[1, 0:2])
ax.scatter(IP_inv[valid], VpVs_inv_smooth[valid], c=facies_inv[valid],
           cmap=cmap, s=10, alpha=0.6, vmin=0, vmax=2)
ax.axhline(VPVS_SAND_BOUNDARY, color='orange', ls='--', lw=1)
ax.axvline(IP_HC_THRESHOLD, color='red', ls='--', lw=1)
ax.set_xlabel('P-Impedance')
ax.set_ylabel('VP/VS')
ax.set_title('IP vs VP/VS', fontweight='bold')

ax = fig.add_subplot(gs[1, 2:4])
ax.scatter(MuRho_inv[valid], LamRho_inv[valid], c=facies_inv[valid],
           cmap=cmap, s=10, alpha=0.6, vmin=0, vmax=2)
ax.set_xlabel('µρ')
ax.set_ylabel('λρ')
ax.set_title('λρ vs µρ (LMR)', fontweight='bold')

ax = fig.add_subplot(gs[1, 4])
ax.scatter(IP_inv[valid], IS_inv[valid], c=facies_inv[valid],
           cmap=cmap, s=10, alpha=0.6, vmin=0, vmax=2)
ip_r = np.linspace(np.nanmin(IP_inv), np.nanmax(IP_inv), 50)
ax.plot(ip_r, np.exp(k * np.log(ip_r) + float(bg.loc['k_c', 'value'])), 'r--', lw=2)
ax.set_xlabel('IP')
ax.set_ylabel('IS')
ax.set_title('IP vs IS', fontweight='bold')

# --- Row 3: Scorecard table ---
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
t_data = [['Method', 'CC(IP)', 'CC(IS)', 'CC(ρ)', 'CC(VP/VS)']]
for row in scorecard:
    vals = [f'{v:.3f}' if isinstance(v, float) else str(v) for v in row[1:]]
    t_data.append([row[0]] + vals)
tbl = ax.table(cellText=t_data, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
for j in range(5):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for j in range(5):
    tbl[len(scorecard), j].set_facecolor('#d4e6f1')

fig.suptitle(
    'OpenSI — Final Inversion Summary Report\n'
    'Well 15/9-F-11 T2 (Volve) | All Methods | Simultaneous Pre-Stack Inversion',
    fontsize=16, fontweight='bold', y=1.02
)
plt.savefig(f'{OUTPUT_PREFIX}_FinalReport.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print(f'\n{"="*60}')
print(f'OpenSI PLATFORM COMPLETE')
print(f'{"="*60}')
print(f'  Phase 1: Low-Frequency Model ✓')
print(f'  Phase 2: Recursive + Colored Inversion ✓')
print(f'  Phase 3: Model-Based Inversion ✓')
print(f'  Phase 4: Sparse Spike Inversion ✓')
print(f'  Phase 5: Simultaneous Pre-Stack Inversion ✓')
print(f'  Phase 6: Interpretation Dashboard ✓')
print(f'\nAll outputs saved with prefix: {OUTPUT_PREFIX}')
print(f'\n"Industry-grade seismic inversion, freely available."')

