<p align="center">
  <h1 align="center">OpenSI - Open Seismic Inversion</h1>
  <p align="center"><em>Industry-grade seismic inversion, freely available.</em></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/data-Volve%20Open%20Dataset-orange?style=flat-square" alt="Volve Dataset">
  <img src="https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square" alt="Status">
</p>

---

## Overview

**OpenSI** is a complete, open-source seismic inversion platform built in Python. It implements every major post-stack and pre-stack inversion method used in hydrocarbon exploration -- from simple recursive inversion to full simultaneous pre-stack inversion for P-impedance, S-impedance, and density.

Built on the **Volve open dataset** (North Sea, well 15/9-F-11 T2), the platform is designed for geophysicists, researchers, and students who want to understand, validate, and apply seismic inversion without commercial software licenses.

### What Makes This Different

- **Complete workflow** -- Not just algorithms. LFM building, wavelet handling, QC diagnostics, facies classification, LMR cross-plots, and interpretation dashboards.
- **Production-quality outputs** -- Every script produces publication-ready figures matching the display standards of Petrel, STRATA, and Jason.
- **Self-contained scripts** -- Each phase loads its own data. Run any script independently.
- **Theory + code** -- Detailed comments explain the math before every code block. Learn the theory as you use the tools.
- **Benchmarked** -- Every result includes CC, NRMS, and visual overlay against well truth.
- **Interactive-friendly** -- Scripts use `# %%` cell markers, fully compatible with VS Code Interactive Python, Spyder, and PyCharm Scientific Mode.

---

## Inversion Methods

| Phase | Method | Type | Description |
|-------|--------|------|-------------|
| **1** | Low-Frequency Model | Foundation | Butterworth LP filter on impedance logs; horizon-guided interpolation |
| **2** | Recursive / Bandlimited | Post-stack | Lindseth (1979) recursive integration + LFM merge |
| **2** | Colored Inversion | Post-stack | Lancaster & Whitcombe (2000) spectral shaping with -90 deg phase |
| **3** | Model-Based (Stochastic) | Post-stack | Conjugate gradient with soft constraint (weighted model penalty) |
| **3** | Model-Based (Constrained) | Post-stack | Conjugate gradient with hard bounds on impedance deviation |
| **4** | Linear Programming Sparse Spike | Post-stack | FISTA-based L1-norm minimization (Basis Pursuit Denoising) |
| **4** | Maximum Likelihood Sparse Spike | Post-stack | Matching pursuit spike-by-spike addition (Chi et al., 1984) |
| **5** | Simultaneous Pre-Stack | Pre-stack | Joint Fatti equation inversion for IP, IS, density via CG (Hampson et al., 2005) |
| **6** | Interpretation Dashboard | Analysis | LMR transforms, facies classification, cross-plots, composite displays |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Setup

1. Download the **Volve well log data** from [Equinor's open data portal](https://www.equinor.com/energy/volve-data-sharing)
2. Place the following files in the `data/` directory:
   - `WLC_PETRO_COMPUTED_INPUT_1.LAS` -- Well logs (VP, VS, RHOB)
   - `td_calibrated_15_9_F11_T2.csv` -- Calibrated time-depth table
   - `final_wavelet.csv` -- Extracted seismic wavelet
3. For Phase 5 (pre-stack), download the Kirchhoff angle stacks from the Volve seismic data

### Run

```bash
# Run each phase in order
cd src/
python SI_Phase1_LFM_Builder.py
python SI_Phase2_Recursive_Colored_Inversion.py
python SI_Phase3_Model_Based_Inversion.py
python SI_Phase4_Sparse_Spike_Inversion.py
python SI_Phase5a_Prestack_Data_Prep.py
python SI_Phase5b_Simultaneous_Inversion.py
python SI_Phase6_Interpretation_Dashboard.py
```

Or run interactively in **VS Code** (recommended) -- each script uses `# %%` cell markers for cell-by-cell execution, just like a Jupyter notebook.

---

## Repository Structure

```
OpenSI/
|-- README.md
|-- LICENSE
|-- REFERENCES.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- SI_Phase1_LFM_Builder.py
|   |-- SI_Phase2_Recursive_Colored_Inversion.py
|   |-- SI_Phase3_Model_Based_Inversion.py
|   |-- SI_Phase4_Sparse_Spike_Inversion.py
|   |-- SI_Phase5a_Prestack_Data_Prep.py
|   |-- SI_Phase5b_Simultaneous_Inversion.py
|   +-- SI_Phase6_Interpretation_Dashboard.py
|-- outputs/
|   |-- README.md
|   |-- Phase1_LFM_Summary.png
|   |-- Phase2_Comparison.png
|   |-- Phase3_Summary.png
|   |-- Phase4_Summary.png
|   |-- Phase5b_Results.png
|   |-- Phase6_CrossPlots.png
|   |-- Phase6_CompositeDisplay.png
|   |-- Phase6_Scorecard.png
|   +-- Phase6_FinalReport.png
|-- data/
|   +-- README.md
+-- docs/
    +-- Seismic_Inversion_Platform_Roadmap.md
```

---

## Phase Outputs & Data Flow

```
Phase 1 (LFM)
  |-- ai_fullband_*.csv          --> Phases 2, 3, 4, 5, 6
  |-- lfm_model_*.csv            --> Phases 2, 3, 4, 5, 6
  +-- background_trend_params.csv --> Phase 5

Phase 2 (Recursive + Colored)
  +-- Phase2_inversion_results.csv --> Phase 6

Phase 3 (Model-Based)
  +-- Phase3_inversion_results.csv --> Phase 6

Phase 4 (Sparse Spike)
  +-- Phase4_inversion_results.csv --> Phase 6

Phase 5a (Pre-stack Prep)
  |-- Phase5a_angle_traces.csv     --> Phase 5b
  |-- Phase5a_wavelet_*.csv        --> Phase 5b
  +-- Phase5a_sim_params.csv       --> Phase 5b

Phase 5b (Simultaneous)
  +-- Phase5b_inversion_results.csv --> Phase 6

Phase 6 (Dashboard)
  |-- Phase6_CrossPlots.png
  |-- Phase6_CompositeDisplay.png
  |-- Phase6_Scorecard.png
  +-- Phase6_FinalReport.png
```

---

## Key Algorithms

### Forward Model

The linearized forward model for post-stack inversion:

```
s = 0.5 * W * D * Lp
```

where `s` = seismic trace, `W` = wavelet matrix, `D` = derivative matrix, `Lp` = ln(P-impedance).

### Simultaneous Inversion (Fatti Equation)

For each angle theta:

```
s(theta) = c1_tilde * W(theta) * D * Lp + c2_tilde * W(theta) * D * dLS + c3 * W(theta) * D * dLD
```

Solved jointly for `Lp`, `dLS`, `dLD` with variable-specific regularization.

### Property Recovery

```
Ip  = exp(Lp)
Is  = exp(k * Lp + kc + dLS)
rho = exp(m * Lp + mc + dLD)
```

---

## Results Summary (Volve Well 15/9-F-11 T2)

| Method | CC(IP) | CC(IS) | CC(rho) | CC(VP/VS) |
|--------|--------|--------|---------|-----------|
| LFM Only | 0.901 | -- | -- | -- |
| Recursive | 0.918 | -- | -- | -- |
| Colored | 0.899 | -- | -- | -- |
| Model-Based (Stochastic) | 0.901 | -- | -- | -- |
| Model-Based (Constrained) | 0.900 | -- | -- | -- |
| LPSS (Sparse Spike) | 0.910 | -- | -- | -- |
| MLSS (Sparse Spike) | 0.913 | -- | -- | -- |
| **Simultaneous (Pre-stack)** | **0.892** | **0.862** | **0.683** | **0.253** |

> CC = Correlation Coefficient vs well truth. Simultaneous inversion is the only method that recovers IS, density, and VP/VS.

See the [outputs/](outputs/) directory for all result figures.

---

## Using Real Seismic Data

The platform supports both synthetic (well-based) and real SEG-Y inputs.

### Post-Stack (Phases 2-4)

Replace the synthetic trace with a real post-stack trace extracted at the well location.

### Pre-Stack (Phase 5)

Set `USE_REAL_SEGY = True` in Phase 5a and provide paths to your angle stack SEG-Y files:

For the Volve dataset, these are available as:
```
ST10010ZC11_PZ_PSDM_KIRCH_NEAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy  (3-17.67 deg)
ST10010ZC11_PZ_PSDM_KIRCH_MID_T.MIG_FIN.POST_STACK.3D.JS-017536.segy   (17.67-32.34 deg)
ST10010ZC11_PZ_PSDM_KIRCH_FAR_T.MIG_FIN.POST_STACK.3D.JS-017536.segy   (32.34-41 deg)
```

---

## Contributing

Contributions are welcome! Areas where help is needed:

- **Geostatistical inversion** -- Multiple equiprobable realizations via SGS
- **Deep learning inversion** -- CNN/RNN-based impedance prediction
- **Time-lapse (4D) inversion** -- Monitor vs baseline difference inversion
- **Extended Elastic Impedance (EEI)** -- Whitcombe et al. (2002)
- **Multi-well support** -- Lateral interpolation with horizon guidance
- **GUI wrapper** -- Interactive parameter tuning via Panel/Streamlit

---

## References

See [REFERENCES.md](REFERENCES.md) for the complete bibliography. Key references:

- Aki, K. & Richards, P.G. (2002). *Quantitative Seismology*, 2nd ed. University Science Books.
- Fatti, J.L. et al. (1994). Detection of gas in sandstone reservoirs using AVO analysis. *Geophysics*, 59, 1362-1376.
- Hampson, D.P. & Russell, B.H. (1991). Comparison of post-stack seismic inversion methods. *SEG Abstracts*, 876-878.
- Hampson, D.P., Russell, B.H. & Bankhead, B. (2005). Simultaneous inversion of pre-stack seismic data. *SEG Abstracts*, 1633-1637.
- Lancaster, S. & Whitcombe, D. (2000). Fast-track coloured inversion. *SEG Expanded Abstracts*.
- Lindseth, R.O. (1979). Synthetic sonic logs. *Geophysics*, 44, 3-26.
- Oldenburg, D.W., Scheuer, T. & Levy, S. (1983). Recovery of acoustic impedance from reflection seismograms. *Geophysics*, 48, 1318-1337.

---

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Equinor** and the Volve license partners for making the Volve field data publicly available
- The seismic inversion research community whose published methods form the foundation of this platform

---

<p align="center">
  <strong>OpenSI</strong> -- Because seismic inversion shouldn't require a six-figure software license.
</p>
