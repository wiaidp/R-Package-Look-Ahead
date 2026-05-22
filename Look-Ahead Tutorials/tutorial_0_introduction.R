# ════════════════════════════════════════════════════════════════════
# TUTORIAL 0 — INTRODUCTION
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ──────────────────────────────────────────────────────────

# This tutorial illustrates a fundamental limitation of the classical minimum
# Mean Squared Error (MSE) multi-step-ahead predictor in many practical
# settings: it tends to be "stuck to the present." Specifically, for forecast
# horizons h > 0, the predictor often exhibits a strong correlation with x_t
# (the current observation) rather than with the target quantity x_{t+h},
# which remains unobserved at the time of prediction.
#
# For illustration, consider an AR(1) process: x_t = a1 * x_{t-1} + epsilon_t.
# The optimal h-step-ahead MSE forecast reduces to hat{x}_{t+h|t} = a1^h * x_t,
# meaning the forecast is simply a rescaled version of the current observation.
# This exemplifies the "anchoring" problem: as the horizon h increases, the
# forecast remains structurally tied to x_t, offering no predictive
# value about the true future dynamics. While the AR(1) case is especially
# transparent — the dependence on x_t is explicit and direct — the same
# tendency arises more broadly: for many stationary processes, the forecast
# function places disproportionate weight on the most recent observation(s),
# making deliberate adjustments necessary to look ahead, beyond x_t. 
#
# This potential failure of the classic MSE forecast approach  motivates the 
# development of two novel optimisation frameworks that explicitly enforce 
# look-ahead behaviour by imposing a lead (left-shift / advancement) over the 
# classic MSE predictor:
#
#   - Decouple-From-Present (DFP): decouples the filter output from
#     the current observation to promote genuine anticipation.
#   - Peak Correlation Shifting (PCS): shifts the peak cross-correlation
#     between the predictor and the target toward the desired horizon (when feasible).

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
# In applications to non-seasonal economic time series, a typical
# pattern is a monotonically decaying autocorrelation function (ACF),
# as observed in the AR(1) process. Under such persistence, the classic
# MSE predictor tends to remain stuck (anchored) to the present regardless of
# the forecast horizon h. The proposed DFP and PCS approaches extend naturally 
# to general stationary processes with arbitrarily rich ACF profiles.
#
# ── LIMITATIONS ────────────────────────────────────────────

# While leaning heavily on the most recent observation x_t may appear intuitively
# reasonable — recent data naturally carry the most relevant information — this
# strategy offers no additional predictive value beyond the current observation,
# underscoring the need for richer, more forward-looking approaches.
#
# However, any departure from the optimal MSE predictor necessarily incurs a
# cost in mean-square error performance. Furthermore, decoupling the forecast
# from the most recent observation can feel unnatural, potentially conflicting
# with prior knowledge and domain expertise.
#
# Hence, beyond being technically more demanding, look-ahead designs may also
# run counter to intuition, making them harder to justify in practice.
#
# Finally, in difficult forecasting problems — or in limiting cases where the
# look-ahead structure strongly conflicts with the data-generating process —
# interpretability may become a concern.
#
# ════════════════════════════════════════════════════════════════════

# ── TUTORIAL OVERVIEW ────────────────────────────────────────────────
#
#   Tutorial 1  — Illustrates the "stuck at present" problem using a
#                 simulation example with a slowly, monotonically
#                 decaying ACF.
#
#   Tutorial 2  — Presents and discusses various measures of time-shift
#                 (lead / advancement / left-shift vs.
#                  lag  / retardation / right-shift).
#
#   Tutorial 3  — Introduces the DFP framework conceptually, without
#                 empirical examples.
#
#   Tutorial 4  — Introduces the "unitary DFP" design and discusses
#                 its geometric properties.
#
#   Tutorial 5  — Introduces the "MSE-DFP" design and its geometry.
#
#   Tutorial 6  — Examines the interpretability of MSE-DFP by relating
#                 its constraint to a formal time-shift measures from
#                 Tutorial 2; also identifies special cases in which
#                 extreme decoupling may lead to undesirable inversion.
#
#   Tutorial 7  — Applies MSE-DFP to an ARMA process, revealing hidden
#                 features of the data-generating process (DGP) that
#                 are useful for look-ahead forecasting.
#
#   Tutorial 8  — Applies MSE-DFP to a monthly US employment indicator
#                 as a real-world business-cycle forecasting example.
#
#   Tutorial 9  — Examines a so-called non-standard case in which the optimal
#                 MSE predictor lags behind the nowcast (current data) and where
#                 looking ahead entails counterintuitive adjustments by the DFP.
#
#   Tutorial 10 — Introduces the PCS criterion and provides a direct comparison
#                 with the DFP.
#
#   Tutorial 11 — Discusses a PCS-based leading indicator design applied to US GDP.
#
#   Tutorial 12 — Analyzes Type I, II, and III PCS designs in the context of a
#                 challenging forecast problem, involving the monthly US 
#                 employment indicator.
#
#   Tutorial 13 — Applies Type I, II, and III PCS designs to both an impossible
#                 and an easy forecast problem, discussing the rationale and the
#                 relative strengths and weaknesses of each design.
#
#   Tutorial 14 — Proposes  solutions to impossible forecast problems in which
#                 the structure of the data-generating process does not permit
#                 decoupling or modification of the CCF.
#
#   Tutorial 15 — Introduces an additional, more recent Type IV PCS approach, 
#                 applied to the monthly US employment indicator.

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

