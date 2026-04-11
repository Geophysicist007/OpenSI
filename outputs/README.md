# Outputs

This directory contains the results from running the OpenSI inversion pipeline on the Volve well 15/9-F-11 T2.

## Phase 1 — Low-Frequency Model
- `Phase1_LFM_Summary.png` — Composite QC panel: AI/SI/Density/Reflectivity tracks + frequency spectra + background trend cross-plots

## Phase 2 — Recursive & Colored Inversion
- `Phase2_Comparison.png` — Side-by-side comparison of recursive vs. colored inversion with metrics table

## Phase 3 — Model-Based Inversion
- `Phase3_Summary.png` — Stochastic & constrained inversion results with convergence diagnostics

## Phase 4 — Sparse Spike Inversion
- `Phase4_Summary.png` — LPSS vs MLSS comparison + all-methods scorecard

## Phase 5b — Simultaneous Pre-Stack Inversion
- `Phase5b_Results.png` — IP, IS, density, VP/VS inversion results with LMR cross-plots

## Phase 6 — Interpretation Dashboard
- `Phase6_CrossPlots.png` — IP vs VP/VS, lambda-rho vs mu-rho, IP vs IS, IP vs density
- `Phase6_CompositeDisplay.png` — Multi-track well display with facies classification
- `Phase6_Scorecard.png` — All-methods comparison table
- `Phase6_FinalReport.png` — Complete summary report figure

## Results Summary

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
