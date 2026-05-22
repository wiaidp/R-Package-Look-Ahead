# ════════════════════════════════════════════════════════════════════
# TUTORIAL 3 — INTRODUCTION TO DECOUPLING FROM PRESENT (DFP)
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────────
# Tutorial 1 showed that the classical MSE multi-step-ahead predictor
# can become "stuck at the present": rather than anticipating x_{t+h},
# it correlates most strongly with x_t (the current observation), where
# h > 0 is the forecast horizon. This look-ahead failure motivates the
# Decouple-From-Present (DFP) criterion — a novel optimisation framework
# that explicitly enforces look-ahead behaviour by controlling how strongly
# the predictor is tied to the present value of the series.
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ── CORE IDEA ─────────────────────────────────────────────────────────
# Let x_t be a stationary time series, h > 0 the forecast horizon, and
# y_t(h) the h-step-ahead predictor. The DFP criterion pursues two
# simultaneous objectives:
#
#   (1) TRACKING  — Maximise correlation of y_t(h) with the target x_{t+h},
#                   equivalently minimise the MSE forecast error.
#
#   (2) DECOUPLING — Control (reduce) the correlation of y_t(h) with x_t,
#                    the value at the CURRENT time point. Decoupling from
#                    the present forces the predictor to "look ahead"
#                    rather than mirroring what is already observed.
#
# These two objectives are placed in explicit tension via a constraint
# hyperparameter, allowing the practitioner to navigate the full
# Accuracy–Timeliness (AT) trade-off frontier.


# ── EDGING ON A TRADEOFF ──────────────────────────────────────────────
# Decoupling y_t(h) from x_t comes at a cost: the DFP correlates less strongly
# with the target x_{t+h} than the classical MSE-optimal predictor does.
#   - The DFP criterion is designed to minimize this loss, i.e., to find the
#     best achievable correlation with x_{t+h} under the decoupling constraint.
#
# Crucially, however, decoupling is not merely a limitation — it is the very
# mechanism that enables y_t(h) to look ahead effectively.
# No other predictor can look as far ahead as the DFP predictor in a given sense 
# and for a given level of tracking accuracy, see Wildi 2026, sections 3.5 and 4.3.

# ── GENERALISATION ────────────────────────────────────────────────────
# The framework extends naturally beyond point forecasting:
#   - The target can be an arbitrary non-causal signal z_{t+h}, where z_t
#     is a trend or business-cycle component extracted from x_t.
#   - This enables the design of LEADING INDICATOR predictors: filters
#     tailored to anticipate a specific signal of interest (e.g., a
#     band-pass filtered cycle) rather than the raw series.
#   - See the dedicated leading-indicator example for a worked application.

# ── SCOPE AND MOTIVATION ──────────────────────────────────────────────
# This tutorial analyses STATIONARY time series only.
#
# Motivation for the stationarity assumption in applied work:
#   Most macroeconomic and financial series are non-stationary (e.g., GDP,
#   prices, asset values). However, it is typically the GROWTH component
#   (first differences) that carries the economically relevant signal.
#   First-differencing renders most economic series approximately stationary,
#   at least over the sample periods relevant for short- to medium-term
#   forecasting. The DFP framework is therefore applied to either natively 
#   stationary data or to data rendered stationary through differencing.

# ── TWO OPTIMISATION FORMS ────────────────────────────────────────────
# DFP can be formulated in two equivalent but complementary ways:
#
#   Form 1 — UNITARY DFP  (Equation 2 in Wildi 2026)
#     A quadratic (squared) optimisation problem. The constraint
#     hyperparameter has a direct, intuitive interpretation: it corresponds
#     to a prescribed correlation between y_t(h) and x_t (the degree of
#     decoupling from the present).
#
#   Form 2 — MSE-DFP  (Equation 9 in Wildi 2026)
#     A linear optimisation problem obtained by relaxing the unit-length 
#     constraint in a MSE formulation of the problem.  Computationally 
#     simpler, but the constraint hyperparameter is less directly 
#     interpretable in isolation.
#
# In both forms, solutions are obtained in CLOSED FORM and correspond to
# the global optimum of the respective problem.

# ── CONNECTION TO TIMELINESS (TUTORIAL 2) ─────────────────────────────
# The time-shift function at zero frequency (omega = 0), introduced in
# Tutorial 2 as a measure of filter timeliness, is directly related to
# the DFP constraint hyperparameter.
#
# Specifically, when the constraint is expressed in terms of the
# zero-frequency time-shift, both Unitary DFP and MSE-DFP acquire a
# concrete, interpretable meaning: the hyperparameter controls the
# LEAD (in time units) of the predictor relative to the present, at the
# frequency that dominates most macroeconomic signals (near zero /
# business-cycle band).

# ── PRIMAL AND DUAL FORMULATIONS ─────────────────────────────────────
#   PRIMAL form:
#     Maximise tracking accuracy (correlation of y_t(h) with x_{t+h} or z_{t+h})
#     subject to a prescribed time-shift (lead) constraint.
#
#   DUAL form:
#     Minimise the link with the present x_t — equivalently, maximise the
#     lead of y_t(h) — subject to a prescribed level of tracking accuracy.
#
# Both formulations trace the same efficient frontier; the choice between
# them is a matter of which quantity (accuracy or lead) is fixed as the
# binding constraint.

# ── THE ACCURACY–TIMELINESS (AT) EFFICIENT FRONTIER ──────────────────
# The DFP predictor sweeps out the complete efficient AT frontier:
#   - No other linear predictor can achieve higher tracking accuracy for
#     a given lead constraint.
#   - Equivalently, no other linear predictor can achieve a greater lead
#     for a given tracking accuracy.
#
# The classical MSE predictor corresponds to a SINGLE POINT on this
# frontier (the accuracy-maximising endpoint).
# DFP generalises MSE to the ENTIRE frontier, giving practitioners
# explicit control over the accuracy–timeliness trade-off.

# ── INTERPRETABILITY AND PRACTICAL ADVANTAGES ─────────────────────────
#   - Both the objective function and the constraint have clear economic
#     interpretations (correlation with target; lead relative to present).
#   - DFP nests MSE as a special case.
#   - Closed-form solutions guarantee global optimality and fast computation.
#   - The framework is modular: swap the target signal z_t to obtain
#     predictors tailored to cycles, trends, or custom band-pass signals.
#   - It is possible to formulate a specialized leading indicator DFP, 
#     as illustrated in one of the tutorials.
# ════════════════════════════════════════════════════════════════════

#---------------------------------------------------------------------
# SHORT NOTE  ON EMPIRICAL EXAMPLES:
#---------------------------------------------------------------------
# Our examples TYPICALLY illustrate challenging forecast problems in which the DGP is
# characterized by a slowly and monotonically decaying autocorrelation (ACF) 
# pattern (see also tutorial 1). The corresponding time series dynamics 
# imply that the classical MSE predictor is effectively trapped at the 
# present time, largely unable to anticipate future movements.
# These forecast problems are inherently difficult and may require aggressive
# decoupling constraints to induce look-ahead behavior. Such extreme settings
# naturally raise the question of interpretability and consistency of strong
# look-ahead designs, as well as the existence of meaningful upper limits on
# the target lead to impose. These topics are addressed in Wildi (2026), 
# section 4.3.