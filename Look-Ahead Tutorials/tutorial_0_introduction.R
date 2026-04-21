# ════════════════════════════════════════════════════════════════════
# TUTORIAL 0 — INTRODUCTION
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ──────────────────────────────────────────────────────────
# This tutorial series investigates a fundamental limitation of the
# classical minimum Mean Squared Error (MSE) multi-step-ahead predictor 
# in many important application cases:
# it tends to be "stuck at the present", meaning that rather than
# anticipating x_{t+h}, the predictor correlates (most) strongly with
# x_t (the current observation), where h > 0 is the forecast horizon.
#
# This look-ahead failure motivates the development of two novel
# optimisation frameworks that explicitly enforce look-ahead behaviour
# by imposing a lead (left-shift / advancement) over the classic MSE
# predictor:
#
#   - Decouple-From-Present (DFP): decouples the filter output from
#     the current observation to promote genuine anticipation.
#   - Peak Correlation Shifting (PCS): shifts the peak cross-correlation
#     between the filter output and the target toward the desired horizon.
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ── SCOPE AND ASSUMPTIONS ────────────────────────────────────────────
# Both DFP and PCS are designed for stationary univariate time series:
#
#   - Non-stationarity: it is assumed that any non-stationary series
#     has been differenced prior to filtering/forecasting, so that the 
#     predictor targets growth in the original I(1) data.
#   - Univariate focus: extensions to multivariate settings are
#     theoretically possible but have not yet been developed. This
#     tutorial series concentrates on the core look-ahead ideas, which
#     are most clearly illustrated in a simple univariate framework.
#
# In applications to (non-seasonal) economic time series, a typical
# pattern is a monotonically decaying AutoCorrelation Function (ACF).
# Under such persistence, the classic MSE predictor tends to remain
# "stuck at the present" regardless of the forecast horizon h.
# ════════════════════════════════════════════════════════════════════

# ── TUTORIAL OVERVIEW ────────────────────────────────────────────────
#
#   Tutorial 1 — Illustrates the "stuck at present" problem using a
#                simulation example with a slowly, monotonically
#                decaying ACF.
#
#   Tutorial 2 — Presents and discusses various measures of time-shift
#                (lead / advancement / left-shift vs.
#                 lag  / retardation / right-shift).
#
#   Tutorial 3 — Introduces the DFP framework conceptually, without
#                empirical examples.
#
#   Tutorial 4 — Introduces the "unitary DFP" design and discusses
#                its geometric properties.
#
#   Tutorial 5 — Introduces the "MSE-DFP" design and its geometry.
#
#   Tutorial 6 — Examines the interpretability of MSE-DFP by relating
#                its constraint to a formal time-shift measures from
#                Tutorial 2; also identifies special cases in which
#                extreme decoupling may lead to undesirable inversion.
#
#   Tutorial 7 — Applies MSE-DFP to an ARMA process, revealing hidden
#                features of the data-generating process (DGP) that
#                are useful for look-ahead forecasting.
#
#   Tutorial 8 — Applies MSE-DFP to a monthly US employment indicator
#                as a real-world business-cycle forecasting example.
#
#   Tutorial 9 — Introduces the PCS framework and its core mechanics.
# ════════════════════════════════════════════════════════════════════
