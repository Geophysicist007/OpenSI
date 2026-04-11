# Data Directory

This directory is where you place the Volve well log and seismic data files. These files are **not included in the repository** due to size — you must download them from Equinor's open data portal.

## Required Files

### Phase 1–4 (Post-Stack Inversion)

| File | Description | Source |
|------|-------------|--------|
| `WLC_PETRO_COMPUTED_INPUT_1.LAS` | Well logs for 15/9-F-11 T2 (VP, VS, RHOB) | Volve well data |
| `td_calibrated_15_9_F11_T2.csv` | Calibrated time-depth table | Generated from AVO workflow |
| `final_wavelet.csv` | Extracted seismic wavelet | Generated from AVO workflow |

### Phase 5 (Pre-Stack / Simultaneous Inversion)

| File | Description | Angles |
|------|-------------|--------|
| `ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | Near angle stack | 3°–17.67° |
| `ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | Mid angle stack | 17.67°–32.34° |
| `ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | Far angle stack | 32.34°–41° |
| `ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy` | Full angle stack | 3°–41° |

## How to Download

1. Visit [Equinor Volve Data Portal](https://www.equinor.com/energy/volve-data-sharing)
2. Register and accept the license terms
3. Navigate to the well data section for well logs
4. Navigate to `Seismic/ST10010/Stacks` for the angle stack SEG-Y files
5. Download the **KIRCH** (Kirchhoff), **T** (time domain), **FIN** (final processed) versions

## Seismic Data Specifications

- **Sample rate:** 4 ms
- **Record length:** 0–3400 ms TWT
- **Samples per trace:** 850
- **Format:** SEG-Y, IBM float
- **Coordinate system:** ED50 / UTM Zone 31N
- **Inline range:** 9961–10361
- **Crossline range:** 1881–2680
- **Bin size:** 12.5m × 12.5m

## File Placement

Place all downloaded files in this `data/` directory, then update the file paths in each notebook's "User Parameters" cell to point here:

```python
LAS_FILE = '../data/WLC_PETRO_COMPUTED_INPUT_1.LAS'
TD_FILE  = '../data/td_calibrated_15_9_F11_T2.csv'
```

Or copy them to the `notebooks/` directory for simpler paths.
