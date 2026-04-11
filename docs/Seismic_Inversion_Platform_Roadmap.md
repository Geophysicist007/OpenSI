# Open-Source Seismic Inversion Platform — Roadmap

## Vision
A freely available, Jupyter-based seismic inversion suite that delivers interpretation-grade results competitive with Hampson-Russell STRATA, the inversion theory literature GeoSoftware, and Jason. Built on the Volve open dataset (well 15/9-F-11 T2), extensible to any field data.

---

## Foundation: What We Already Have
- T-D calibration (checkshot-based, PCHIP)
- Wavelet extraction (~−2° phase, 33 Hz, CC ≈ 44%)
- Well-tie synthetic with optimization
- Well-tie UQ framework (pending re-run with corrected parameters)
- DTW stretch-and-squeeze notebook (pending/in progress)

---

## Phase 1 — Low-Frequency Model (LFM) Builder
**Why first:** Every inversion method except colored inversion needs an LFM. This is the single biggest quality driver.

### Deliverables (one notebook):
1. **Impedance log computation** — AI = VP × RHOB from the calibrated LAS curves
2. **Horizon-guided interpolation** — Use picked horizons (or auto-picks from seismic) to guide lateral interpolation between wells. For single-well Volve demo: 1-D filtered model.
3. **Low-pass filtering** — Butterworth or Ormsby filter, user-selectable cutoff (default 10 Hz). Show the bandlimited gap clearly.
4. **QC panel** — Overlay: full-bandwidth AI log, filtered LFM, difference (the "missing" seismic band). Frequency-domain comparison.

### Key design choices:
- Filter design must match what commercial tools do (zero-phase Butterworth, order 6–8)
- Support both single-well (1-D) and multi-well (pseudo-2D/3D) modes
- Store LFM as a reusable CSV/NumPy artifact for downstream phases

---

## Phase 2 — Recursive / Bandlimited Inversion
**Why second:** Simplest inversion method; no wavelet needed (or optional dephasing). Establishes the baseline result all other methods must beat.

### Deliverables (one notebook):
1. **Reflectivity-to-impedance recursion** — Implement Eq. 94: I_j = I_{j-1} × (1+r_j)/(1−r_j)
2. **Seismic scaling** — Auto-scale trace amplitudes to reflection-coefficient range (±0.1). Offer RMS-based and max-amplitude-based options.
3. **LFM merge** — Low-pass LFM + bandpass-inverted trace, with crossover frequency as a slider parameter.
4. **Colored inversion operator** (Lancaster & Whitcombe, 2000) — Compute the spectral shaping operator from AI log spectrum vs. seismic spectrum. Apply with −90° phase rotation.
5. **QC & interpretation panel:**
   - Inverted AI trace vs. original AI log
   - Correlation coefficient in the analysis window
   - Frequency spectrum comparison (input seismic vs. inverted AI)
   - Impedance cross-section display (wiggle + color fill) if 2D line available

### Benchmark target:
- CC between inverted AI and well AI (within seismic band) ≥ 0.85

---

## Phase 3 — Model-Based (Constrained) Post-Stack Inversion
**Why third:** This is the workhorse method in STRATA. Requires wavelet + LFM, both of which we now have.

### Deliverables (one notebook):
1. **Forward model engine** — s = W × D × L_P (Eq. 66 from Hampson et al., 2005). Convolution matrix W, derivative matrix D, log-impedance vector L.
2. **Conjugate gradient solver** — Iterative inversion with:
   - Soft constraint (stochastic): weighted sum of seismic misfit + model misfit (Eq. 88)
   - Hard constraint (constrained): maximum impedance change bounds (Eq. 92)
   - User-selectable: number of iterations, prewhitening %, constraint weight
3. **Blocky inversion option** — Layer-based parameterization with adjustable block size
4. **Convergence diagnostics:**
   - Misfit (J) vs. iteration number
   - Synthetic vs. real trace overlay at each iteration
   - Residual trace (real − synthetic)
5. **Result panel:**
   - Inverted AI vs. well AI (full bandwidth)
   - Impedance section with well overlay
   - Correlation and RMS error metrics

### Benchmark target:
- CC ≥ 0.90 at well location
- Visually match STRATA-quality blocky impedance character

---

## Phase 4 — Sparse Spike Inversion
**Why fourth:** Complements model-based; better for areas with fewer reflectors. Two sub-methods.

### Deliverables (one notebook):
1. **Linear Programming Sparse Spike (LPSS)**
   - L1-norm minimization of reflectivity (Eq. 49 area in Hampson et al., 2005)
   - Frequency-domain constraints for low/high frequency recovery
   - Windowed processing with 50% overlap
   - Sparseness control parameter (0–100%)
2. **Maximum Likelihood Sparse Spike (MLSS)**
   - Single-most-likely-addition algorithm
   - Spike detection threshold parameter
   - Iterative spike placement until threshold met
3. **Common QC panel** (shared with Phase 3 for direct comparison)

### Benchmark target:
- Fewer layers than model-based, but CC ≥ 0.85
- Blocky impedance character matching sparse geology

---

## Phase 5 — Simultaneous Pre-Stack Inversion
**Why fifth:** This is the flagship capability. Requires AVO-quality angle gathers — we bridge from our existing AVO work.

### Prerequisites:
- Angle gathers (from Volve pre-stack data, or synthesized partial stacks)
- Angle-dependent wavelets
- Background trend estimation (k, m coefficients from Eqs. 103–104)

### Deliverables (two notebooks):

#### Notebook 5a — Data Preparation & Background Trends
1. **Partial angle stack generation** — Near/mid/far from CDP gathers (or use existing Volve partial stacks if available)
2. **Angle-dependent wavelet extraction** — One wavelet per angle range
3. **Background trend computation:**
   - ln(IS) vs. ln(IP) regression → k, k_c
   - ln(ρ) vs. ln(IP) regression → m, m_c
   - Cross-plot visualization with best-fit lines
4. **Scaling** — Per-angle-range global scalers (RMS_synth / RMS_real)

#### Notebook 5b — Simultaneous Inversion Engine
1. **Modified Fatti equation setup** (Eq. 128):
   - s_θ = c̃₁·W_θ·D·L_P + c̃₂·W_θ·D·ΔL_S + c₃·W_θ·D·ΔL_D
2. **Conjugate gradient solver** for the joint system (Eq. 129)
3. **Output volumes:**
   - P-impedance (I_P)
   - S-impedance (I_S) via I_S = exp(k·L_P + k_c + ΔL_S)
   - Density (ρ) via ρ = exp(m·L_P + m_c + ΔL_D)
   - VP/VS ratio
4. **LMR transform** — λρ and µρ computation from IP, IS
5. **Interpretation cross-plots:**
   - λρ vs. µρ (fluid discrimination)
   - IP vs. VP/VS (lithology + fluid)
   - Color-coded by well facies if available

### Benchmark target:
- IP inversion CC ≥ 0.90
- IS inversion CC ≥ 0.80
- Density CC ≥ 0.60 (inherently noisier — this is realistic)
- Clear fluid anomaly separation on LMR cross-plots

---

## Phase 6 — Interpretation & Visualization Dashboard
**Why last as a build phase, but critical for user satisfaction.**

### Deliverables (one notebook + optional standalone app):
1. **Interactive section viewer** — Matplotlib/Plotly-based, with:
   - Toggle between seismic, AI, IS, VP/VS, λρ, µρ
   - Well log overlay with depth/time toggle
   - Horizon overlay
   - Color bar with petrophysical scale (sandstone/shale/carbonate ranges)
2. **Cross-plot workspace** — Interactive Plotly scatter with:
   - Polygon selection → highlight on section
   - Pre-built templates: IP vs. VP/VS, λρ vs. µρ, IP vs. φ (porosity)
3. **Probabilistic facies classification** — Simple Bayesian or GMM-based, trained on well data, applied to inverted volumes
4. **Report generator** — One-click export of all QC metrics, key figures, and parameters to PDF or HTML

---

## Phase 7 (Stretch Goals) — Advanced Methods
These extend the platform beyond what most users get from commercial tools:

1. **Geostatistical inversion** — Multiple equiprobable realizations using SGS conditioned to seismic
2. **Deep learning-assisted inversion** — CNN/RNN trained on well-seismic pairs to predict impedance beyond seismic bandwidth
3. **Time-lapse (4D) inversion** — Difference inversion for monitor vs. baseline surveys
4. **Elastic impedance inversion** — Connolly (1999) EI parameterization as alternative to simultaneous
5. **Extended elastic impedance (EEI)** — Whitcombe et al. (2002) for optimized fluid/lithology sensitivity

---

## Design Principles (Every Notebook)

| Principle | Implementation |
|-----------|---------------|
| **Self-contained** | Each notebook loads its own data — no shared kernel state |
| **Parameterized** | All key parameters in a single cell at top, clearly documented |
| **Benchmarked** | Every result includes CC, RMS error, visual overlay vs. well truth |
| **Publication-quality plots** | Consistent style, proper axis labels, colorbars, annotations |
| **Theory cells** | Markdown cells explaining the math before each code block |
| **Progressive disclosure** | Simple default run first, advanced tuning options clearly marked |
| **Exportable** | All key outputs saved as CSV/NPY for use in downstream notebooks |
| **Commercial comparison** | Where possible, show expected output from Petrel/STRATA side-by-side |

---

## Data Requirements per Phase

| Phase | Minimum Data | Volve Availability |
|-------|-------------|-------------------|
| 1 — LFM | VP, RHOB, checkshot/TDR | ✅ Have from prior work |
| 2 — Recursive | Stacked seismic (SEG-Y), LFM | ✅ Volve has post-stack |
| 3 — Model-based | + Extracted wavelet | ✅ Have from prior work |
| 4 — Sparse spike | Same as Phase 3 | ✅ |
| 5 — Simultaneous | Partial angle stacks or gathers | ⚠️ Need to verify Volve pre-stack availability |
| 6 — Dashboard | All above outputs | ✅ Generated by Phases 1–5 |

---

## Suggested Build Order & Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| **Phase 1** — LFM | 1 notebook | Prior AVO work (T-D, LAS) |
| **Phase 2** — Recursive + Colored | 1 notebook | Phase 1 |
| **Phase 3** — Model-Based | 1 notebook | Phase 1 + wavelet |
| **Phase 4** — Sparse Spike | 1 notebook | Phase 1 + wavelet |
| **Phase 5** — Simultaneous | 2 notebooks | Phase 1 + angle data |
| **Phase 6** — Dashboard | 1 notebook | Phases 1–5 outputs |

**Recommended start: Phase 1 (LFM Builder)** — it unlocks everything else.

---

## File Naming Convention
```
SI_Phase1_LFM_Builder.ipynb
SI_Phase2_Recursive_Colored_Inversion.ipynb
SI_Phase3_Model_Based_Inversion.ipynb
SI_Phase4_Sparse_Spike_Inversion.ipynb
SI_Phase5a_Prestack_Data_Prep.ipynb
SI_Phase5b_Simultaneous_Inversion.ipynb
SI_Phase6_Interpretation_Dashboard.ipynb
```

---

*Platform name suggestion:* **OpenSI** (Open Seismic Inversion)
*Tagline:* "Industry-grade seismic inversion, freely available."
