# ═══════════════════════════════════════════════════════════════════
# TUTORIAL 12 — CHALLENGING FORECAST PROBLEM
# ═══════════════════════════════════════════════════════════════════

# This tutorial examines a challenging forecasting problem in which the
# MSE-optimal predictor exhibits two structural limitations:
#
#   1. The MSE-optimal predictor is "stuck" at horizon h = 12: selecting
#      h_tilde > 12 does not affect the predictor up to scaling.
#      Consequently, increasing the forecast horizon beyond h = 12 yields
#      no additional information and leaves the predictor unchanged.
#
#   2. The CCF of the MSE predictor peaks at lag k = 4, which lies
#      significantly to the left of the intended forecast horizon h = 12.
#      This misalignment indicates that the predictor draws most of its
#      explanatory power from short lags rather than from the target horizon,
#      and the peak cannot be pushed further rightward toward h by simply
#      adjusting the MSE objective.


# ── PCS PREDICTOR ─────────────────────────────────────────────────────────────

# The DFP and PCS look-ahead approaches impose constraints on the cross-
# correlation function (CCF) of the resulting predictor with lagged (k < 0),
# coincident (k = 0), or leading targets x_{t+k} (k > 0), while maximizing
# the CCF at the forecast horizon k = h, i.e., CCF(h) = Cor(x̂_t, x_{t+h}).
#
# Specifically:
#   - The DFP controls CCF(0) while maximizing CCF(h).
#   - The PCS ideally shifts the peak of the CCF toward lag k = h while
#     maximizing the peak height at that lag.
#
# In difficult forecasting problems where the MSE-optimal predictor is "stuck
# at the present" (i.e., its CCF peaks at k = 0), two complementary strategies
# can unlock look-ahead behavior:
#   - Decoupling the predictor from the nowcast by penalizing CCF(0), as in
#     the DFP approach.
#   - Shifting the CCF peak away from k = 0 toward k = h, as targeted by the
#     PCS approach, when feasible.
#
# However, the constraints imposed on the CCF may conflict with the internal
# structure of the data-generating process (DGP). In such cases, the degrees
# of freedom available for maximizing the objective — the target correlation
# at the pre-specified horizon h — may shrink, leaving little potential to
# achieve effective look-ahead behavior. In extreme cases the conflict is severe
# enough that no feasible solution exists; see Tutorial 13 for a more
# comprehensive treatment.


# ── TYPOLOGY ──────────────────────────────────────────────────────────────────

#   TYPE I — Monotonically Increasing CCF over {0, …, h}  [Most Restrictive]
#
#       The CCF must be strictly increasing across the full lag interval:
#           CCF(k) - CCF(k-1) = beta_k > 0  for all k = 1, …, h.
#       See Wildi (2026), Section 3.2 and Appendix E.
#       This condition is generally not exactly achievable (see Exercise 1).
#       The principal PCS optimization function PCS_func() enforces it as
#       closely as possible via regularization. Smaller regularization weights
#       allow greater flexibility but may yield a CCF that does not peak at
#       k = h, even in cases where such a peak would otherwise be feasible.
#
#   TYPE II — Positive Local Slope at the Target Lag  [Weaker than Type I]
#
#       The CCF must be increasing over the final step only:
#           CCF(h) - CCF(h-1) = beta > 0.
#       See Wildi (2026), Section 3.2.
#       In cases where the DGP imposes additional structure (e.g., via the
#       Yule-Walker equations of an AR(p) process), Types I and II may become
#       equivalent and may be equally feasible or infeasible; see Tutorials 11 
#       and 13.
#
#   TYPE III — Positive Average Slope from Lag 0 to Lag h  [Weaker than Type I]
#
#       The CCF must be increasing on average from k = 0 to k = h:
#           CCF(h) - CCF(0) = beta > 0.
#       See Wildi (2026), Section 3.2.


# ── CONSTRAINT SUMMARY ────────────────────────────────────────────────────────
#
#   Type I:   CCF(k) > CCF(k-1)  for k = 1, …, h  (h constraints)
#   Type II:  CCF(h) > CCF(h-1)                    (1 constraint)
#   Type III: CCF(h) > CCF(0)                      (1 constraint)
#
# Types II and III are necessary but not sufficient conditions for a global
# CCF maximum at lag k = h. Type I is neither necessary (strict monotonicity
# may be overly restrictive) nor sufficient for a global maximum. Nevertheless,
# in many applications all three constraint types are effective in the sense
# that the resulting PCS predictor exhibits useful look-ahead behavior, even
# when the CCF peak does not fall exactly at k = h.
#
# Among the three types, Type I is the most stringent: it imposes the largest
# number of constraints (one per lag from k = 1 to k = h). While this
# increases the likelihood of attaining a CCF peak at k = h, it simultaneously
# reduces the degrees of freedom available for optimizing the target correlation
# at horizon h and locks the CCF into a rigid monotonic profile. Consequently,
# Type I carries the highest risk of infeasibility, with that risk growing with
# h and depending strongly on the structure of the DGP.
#
# Note, however, that infeasibility under Type I can often be mitigated by
# choosing a moderate regularization weight. This provides sufficient control
# over the desired CCF peak shape while preserving enough flexibility to avoid
# over-constraining the path to the peak (see below).


# ── FEASIBILITY ───────────────────────────────────────────────────────────────
#
# Exercises 4-6  in this tutorial focus on feasible forecast problems;
# Exercise 1-3 and 7 illustrate infeasible problems. Infeasibility and impossibility
# are examined more thoroughly in Tutorial 13. Here the primary focus is on a
# complex look-ahead forecast problem with effective look ahead even when infeasible.
#
#   Feasibility: A valid PCS solution exists; the PCS constraints are satisfied
#     and the target correlation is positive, i.e., CCF(h) > 0.
#
# Note that feasibility does not guarantee that the resulting predictor achieves
# a CCF peak at k = h (in the sense of a global or local maximum), nor that the
# peak height is maximized at that lag.
#
# Examples:
#   - A Type II or III constraint can be satisfied without the CCF peak being
#     relocated to k = h; see Exercises 4 and 5 below.


# ── CHALLENGE ─────────────────────────────────────────────────────────────────
#
# We consider a difficult forecasting problem in which the MSE-optimal predictor
# is stuck at horizon h̃ < h: the CCF cannot be pushed further to the right of
# h̃ in the direction of h. Depending on the problem specification, not all PCS
# types are feasible, and when they are feasible they do not always generate
# look-ahead behavior.
#
# Nevertheless, useful look-ahead can generally be obtained by giving more room
# to the target correlation. Two practical strategies are:
#   1. Avoid unnecessary or potentially misspecified PCS constraints by using
#      the less restrictive Type II or Type III instead of Type I.
#   2. Reduce the regularization weight on the PCS constraints to relax the
#      rigidity of the imposed CCF profile.


# ── PROBLEM FORMULATIONS ──────────────────────────────────────────────────────
#
# We use the monthly US employment (business-cycle) PAYEMS indicator;
# see Tutorials 9 and 11. A parsimonious ARMA(1,1) model is fitted, and we
# consider a one-year-ahead forecast (h = 12) for both yearly and monthly growth.

# 1. Monthly Growth (ARMA(1,1)): Exercise 7 below.
#
#    For k > 0 the ACF of the ARMA(1,1) satisfies the recurrence 
#                       ACF(k+1) = a1 * ACF(k), 
#    so the DGP imposes a rigid linear structure on the autocorrelation function.
#
#    For h > 0 and k >= 0 this implies gamma_h ∝ gamma_{h+k}: the MSE predictor
#    coefficient vectors are mutually proportional for all horizons h > 0.
#
#    Consequently, the column space of the PCS constraint system has at most
#    rank 2: (gamma_0 - gamma_1) and (gamma_1 - gamma_2) ∝ gamma_1 are the
#    only linearly independent directions. If gamma_0 is excluded from the
#    constraints, the rank reduces to 1.
#
#    Because the Type III PCS enforces only a single constraint, the problem
#    remains feasible. However, the CCF cannot peak at h = 12 — this is
#    structurally impossible for the ARMA(1,1) DGP — so the problem is
#    classified as impossible yet feasible. Types I and II are infeasible
#    whenever h > 1.
#
#    More precisely:
#      b' * gamma_{k+1} = a1 * b' * gamma_k = a1^(k-1) * b' * gamma_1  for k > 0.
#    This implies CCF(k+1) = a1^(k-1) * CCF(1), which decays exponentially
#    regardless of the predictor vector b. It is therefore structurally
#    impossible to displace the CCF peak to any horizon h > 1.
#
#    Note: The relation above holds only for k > 0. At k = 0,
#      b' * gamma_1 ≠ a1 * b' * gamma_0,
#    because the MA(1) parameter b1 affects the autocovariance at lag 0 but
#    not at lags k > 0. Consequently, the exponential decay applies only for
#    k > 0, and the rank of the PCS constraint system is 2 provided gamma_0
#    enters the constraints.
#
#    This difficult/impossible problem is analyzed in Tutorial 13, and
#    solutions are presented in Tutorial 14. The impossibility of the problem
#    is illustrated in Exercise 7 below.

# 2. Yearly Growth
#
#    Exercises 1–6 focus on the comparatively simpler problem of forecasting
#    yearly growth:
#
#    - Monthly growth corresponds directly to the ARMA(1,1) DGP, which imposes
#      severe structural constraints on look-ahead designs, as described above.
#
#    - Yearly growth corresponds to applying an equally weighted MA(12) filter
#      (a simple trend smoother) to the monthly differences. The resulting DGP
#      is obtained by convolving the ARMA(1,1) with this equally weighted 
#      filter, see exercise 1 below.
#
#    This convolution substantially enriches the constraint structure: the rank
#    of the PCS constraint system increases from 2 (or 1, if gamma_0 is excluded)
#    under the raw ARMA(1,1) to 13 (or 12) under the convolved DGP. Problems
#    that were previously infeasible or impossible under the monthly growth
#    formulation consequently become feasible and achievable.


# ── SOLUTIONS ─────────────────────────────────────────────────────────────────
#
# To address the challenging forecast problem(s), we employ both exact
# closed-form and regularized PCS optimization criteria. The problem is
# sufficiently complex to reveal a rich solution structure, and we therefore
# explore the solution space across a range of relevant hyperparameter settings,
# including the regularization weight lambda and the slope parameter beta.
#
# For this purpose we exploit the automatic range-selection feature provided by
# PCS_func(), which generates hyperparameter grids in the region of highest
# sensitivity of the PCS predictor. This effectively places a magnifying glass
# over the most informative part of the solution space, allowing fine-grained
# scrutiny of how the predictor responds to changes in the tuning parameters.


# ── EXERCISES AND PCS TYPES ───────────────────────────────────────────────────
#
# The exercises below progressively explore the impact of PCS type and
# regularization strength on look-ahead behavior in this challenging setting.
#
# Exercise 1 — Type I PCS with Strong Regularization:
#   Placing a large regularization weight on the full set of 12 Type I
#   constraints assigns excessive importance to potentially misspecified
#   restrictions, at the direct expense of the target correlation. The
#   resulting predictors are either unusable (e.g., negative target
#   correlation) or exhibit no usable look-ahead behavior.
#
# Exercise 2 — Exact Closed-Form Type I PCS:
#   Replacing regularized Type I PCS with its exact closed-form counterpart
#   enforces the constraints without any relaxation, further suppressing the
#   target correlation and worsening results relative to Exercise 1.
#
# Exercise 3 — Type I PCS with Relaxed Regularization:
#   Reducing the regularization weight lambda on the high-dimensional (12-)
#   constraint system restores degrees of freedom for maximizing the target
#   correlation, yielding predictors with usable look-ahead behavior, despite 
#   the problem being infeasible.
#
# Exercise 4 — Type III PCS with Strong Regularization:
#   Replacing the 12 Type I constraints with the single Type III constraint
#   CCF(h) - CCF(0) = beta concentrates the available degrees of freedom on
#   the target correlation. The problem is feasible and even under strong 
#   regularization, the resulting predictors exhibit usable look-ahead behavior.
#
# Exercise 5 — Type II PCS with Strong Regularization:
#   The single Type II constraint CCF(h) - CCF(h-1) = beta reduces the
#   constraint space to a single equation, making the problem formally feasible.
#   However, Type II proves less effective than Type III at inducing genuine
#   look-ahead behavior in this challenging setting.
#
#   Specifically, designs for which CCF(h) - CCF(h-1) is positive — which would
#   nominally suggest look-ahead — turn out to fall into one of two problematic
#   categories:
#     (a) Sign-inverted predictors: CCF(h) > CCF(h-1) is achieved by flipping
#         the sign of the output, rendering the predictor directionally unusable.
#     (b) Effectively lagging predictors: the CCF peak lies to the left of
#         k = h rather than at or near it, so the predictor lags behind the
#         target rather than anticipating it.
#   In both cases the outcome is difficult to interpret and provides no practical
#   look-ahead benefit, in contrast to the Type III results of Exercise 4.
#
# Exercise 6 — Type II PCS: Exact Closed-Form Solution:
#   The exact closed-form solution corroborates the findings of Exercise 5,
#   confirming that the problem is feasible and that the strongly regularized 
#   solution (large regularization weight lambda) converges to the closed-form 
#   expression in which the Type II constraint is satisfied exactly rather than 
#   approximately.
#
# Exercise 7 — Impossible Problem: Monthly DGP:
#   Switching to monthly data reduces the DGP to the raw ARMA(1,1), whose
#   rigid autocorrelation structure imposes severe constraints on any
#   look-ahead design. In this setting the CCF peak cannot be displaced
#   beyond k = 1, regardless of the PCS type or hyperparameter configuration,
#   rendering a one-year-ahead look-ahead forecast structurally impossible. 
#   This problem is emphasized in Tutorial 13 and solutions are proposed in 
#   Tutorial 14.


# ── MAIN TAKE-AWAYS ───────────────────────────────────────────────────────────
#
# These exercises illustrate the complexity of the look-ahead problem within
# the PCS framework, whose rich structure can produce surprising or
# counter-intuitive results. The principal lessons are:
#
#   1. Over-emphasizing high-dimensional, potentially misspecified constraints
#      — whether through strong regularization or exact closed-form solutions —
#      comes at the direct expense of the target correlation, yielding unusable
#      predictors or predictors with no look-ahead behavior.
#
#   2. Relaxing the constraint space frees degrees of freedom for maximizing
#      the target correlation, and is generally necessary to obtain effective
#      look-ahead behavior in difficult forecasting problems.
#
#   3. Relaxation can be achieved in two complementary ways:
#        (a) Reducing the number of constraints, e.g., using Type II or
#            Type III instead of Type I.
#        (b) Reducing the regularization weight placed on the constraints,
#            allowing the optimizer greater freedom to pursue the target
#            correlation.
#
#   4. A third, structurally motivated route to effective relaxation is
#      proposed in tutorial 15: rather than weakening the constraint system
#      directly, one exploits inherent properties of the DGP to guide a single, 
#      well-chosen constraint toward genuine and efficient look-ahead behaviour.
#
#   5. Reducing the number of constraints is not universally effective: the
#      choice of which single constraint to impose matters considerably.
#      In Tutorials 10 and 11, Type II PCS performed best — precisely shifting the 
#      CCF peak to the forecast horizon h with minimal obstruction to the target
#      correlation. In the present, more challenging setting, Type II PCS is
#      less effective and produces partially unexpected, difficult-to-interpret
#      outcomes (see exercises 5 and 6 below). Type III, which is equally 
#      parsimonious, provides more effective control over the CCF peak location 
#      in this example and avoids an undue negative impact on the target 
#      correlation.


# ════════════════════════════════════════════════════════════════════════════════
# ── BACKGROUND / REFERENCES ──────────────────────────────────────────────────
#
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
#
# ════════════════════════════════════════════════════════════════════════════════







# ── INITIALISATION ───────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 1 — Monthly US Employment:
# Equally Weighted Trend, PCS Type I with Strong Regularization
# ════════════════════════════════════════════════════════════════════════════════
# We adopt the framework from Tutorials 9 and 11: an ARMA(1,1) model fitted to
# the monthly PAYEMS employment indicator.
#
# In this ARMA(1,1) context, imposing strong regularization on the Type I PCS
# constraints severely degrades the target correlation. The monotonicity
# constraints conflict with the DGP structure, consuming the degrees of freedom
# that would otherwise be available for maximizing CCF(h) and pushing the
# predictor into unusable territory (negative or negligible target correlation, 
# lagging instead of leading behaviour).
#
# Nevertheless, the exercise is instructive: it exposes the tension between
# rigid constraint imposition and target correlation, and provides the baseline
# against which the relaxed designs of Exercises 3 and 4 can be evaluated.

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  WARNING: ALL PCS PREDICTORS IN THIS EXERCISE ARE SUBJECT TO            │
# │  MISSPECIFIED CONSTRAINTS AND ARE UNUSABLE FOR LOOK-AHEAD FORECASTING.  │
# └─────────────────────────────────────────────────────────────────────────┘



# ─────────────────────────────────────────────────────────────────────────────
# 1.1 Load the Data
# ─────────────────────────────────────────────────────────────────────────────

# Set reload_data = TRUE to download the latest vintage from FRED;
# set to FALSE to load the previously saved local copy.
reload_data <- FALSE

if (reload_data) {
  PAYEMS <- get_fred_series("PAYEMS", series_name = "GDP")
  PAYEMS <- as.xts(PAYEMS)
  save(PAYEMS, file = file.path(getwd(), "Data", "PAYEMS"))
} else {
  load(file = file.path(getwd(), "Data", "PAYEMS"))
}

# Inspect the series endpoints to confirm the loaded vintage.
head(PAYEMS)
tail(PAYEMS)

# Extract the post-1990, pre-pandemic sub-sample in log-levels.
# The log transformation stabilises the variance as the level of the series
# grows over time. Excluding COVID-era data avoids distortions from extreme
# lockdown outliers.
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y) <- index(PAYEMS["1990::2019"])

par(mfrow = c(2, 2))
plot(y,
     main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = FALSE,
     xlab = "", ylab = "")
axis(1, at = 1:length(y), labels = names(y))
axis(2)
box()

# Compute stationary first differences of the log-series:
#   - The log transformation stabilises the variance.
#   - The first difference removes the trend and stabilises the level.
x <- diff(y)

# The differenced log-PAYEMS series is fairly noisy, with pronounced downturns
# during recession episodes.
ts.plot(x, main = "Diff-log PAYEMS")

# The empirical ACF decays slowly and monotonically — a pattern consistent with
# the dominant AR structure and indicative of an MSE predictor that is "stuck
# at the present" (see Tutorial 1).
acf(x, main = "ACF diff-log PAYEMS")


# ─────────────────────────────────────────────────────────────────────────────
# 1.2 Model Fit
# ─────────────────────────────────────────────────────────────────────────────

L <- 50   # filter length: number of MA coefficients retained

# Fit an ARMA(1,1) model — a parsimonious specification with adequate
# diagnostics for this series.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)
a1 <- arima.obj$coef[1:ar_order]
b1 <- arima.obj$coef[ar_order + 1:ma_order]

# Fix the parameters for replicability of results.
a1 <- 0.95
b1 <- -0.53


# --- Wold Decomposition (MA-infinity representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of the
# fitted ARMA model. The filter length L ensures that the coefficients have
# decayed sufficiently close to zero by lag L.
if (ma_order > 0) {
  xi <- c(1, ARMAtoMA(ar = a1, ma = b1, lag.max = length(x)))
} else {
  xi <- c(1, ARMAtoMA(ar = a1, ma = 0,  lag.max = length(x)))
}

# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition: slowly decaying impulse response (post-1990)")

# The theoretical ACF implied by the Wold decomposition matches the empirical
# ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),
        main = "Model-based ACF", ylab = "", xlab = "Lag")


# ─────────────────────────────────────────────────────────────────────────────
# 1.3 Forecast Horizons
# ─────────────────────────────────────────────────────────────────────────────

# One-year-ahead forecast horizon.
h      <- 12
# Larger horizon retained for benchmarking purposes: htilde > h does not shift 
# the CCF peak further to the right.
htilde <- 24


# ─────────────────────────────────────────────────────────────────────────────
# 1.4 Target: Equally Weighted Trend (Yearly Growth)
# ─────────────────────────────────────────────────────────────────────────────

# The differenced log-PAYEMS series is fairly noisy, with pronounced downturns
# during recession episodes.

# To reduce noise we apply a lowpass filter. Possible choices include:
#   - Classic trend filters, e.g., the HP filter (see Tutorial 11).
#   - Ideal lowpass filter.
#   - AR(1) smoother.
#   - Moving average (MA).
#
# For simplicity, we use an equally weighted MA(12) for the following reasons:
#   - The DFP approach is agnostic to the choice of target filter; results
#     generalise to other target filter designs.
#   - An equally weighted MA(12) applied to differenced log-PAYEMS corresponds
#     directly to yearly growth, making the target readily interpretable.
#   - Averaging over a full year reduces noise and amplifies the relevant
#     business-cycle dynamics.

# Define the equally weighted yearly MA target filter.
gamma_target <- rep(1/12, 12)

# Express the trend target (yearly growth) in MA-equivalent (convolved) form.
gamma <- conv_two_filt_func(xi, gamma_target)$conv

# Visualise the Wold coefficients for both the monthly and yearly representations.
par(mfrow = c(2, 1))
ts.plot(xi,    main = "Wold Decomposition: Monthly Growth (post-1990)")
ts.plot(gamma, main = "Wold Decomposition: Yearly Growth  (post-1990)")

# Set nowcast and h = 12-step-ahead MSE predictors under the yearly target.
gamma0    <- gamma[1:L]
gammah    <- gamma[h + 1:L]
# Longer horizon: to illustrate that MSE cannot improve look ahead behaviour.
gammahtilde    <- gamma[htilde + 1:L]
# This is passed to the PCS functions PCS_func() and PCS_closed_form_func() below.
gamma_pcs <- gamma



# ─────────────────────────────────────────────────────────────────────────────
# 1.5 DGP Structural Constraints on the PCS Solution Space
# ─────────────────────────────────────────────────────────────────────────────

# Construct the matrix with MSE predictors (whose columns are successive shifts 
# of gamma), used to assess the effective rank of the PCS constraint system.
gamma_mat <- gamma[1:L]
for (i in 1:(L - 1)) {
  gamma_i   <- gamma[i + 1:L]
  gamma_mat <- cbind(gamma_mat, gamma_i)
}

eigenvalues <- eigen(gamma_mat)$values

# Effective rank of the PCS constraint system (number of non-negligible
# eigenvalues): determines how many linearly independent constraints can
# be imposed without over-determining the system.
length(which(abs(eigenvalues) > 10^{-10}))

# Implications by PCS type:
#
#   Type II or III:
#     A single constraint of the form b' * (gamma_h - gamma_0) consumes one
#     of the available 13 degrees of freedom, leaving 12 for optimization of
#     the target correlation.
#
#   Type I with h = 12:
#     Strong regularization causes the 12-dimensional constraint system to
#     absorb all but one degree of freedom. Only one degree of freedom remains
#     to maximize the target correlation, severely limiting look-ahead potential.


# ─────────────────────────────────────────────────────────────────────────────
# 1.6 PCS Type I: Parameter Setup
# ─────────────────────────────────────────────────────────────────────────────

# Type I imposes h slope constraints at lags k = 1, …, h.
Delta  <- 1:h
lambda <- 1000000   # very strong regularization

# Each constraint takes the form:
#
#   b' * (gamma_k - gamma_{k-1}) = beta,  k = 1, …, h
#
# where beta is a fixed slope target that is held constant across all h
# constraints; see the regularized criterion in equation (46) of Wildi (2026).
# The solution to the regularized criterion is given by equation (49), and the
# exact closed-form solution by equations (47) and (48). Variable beta = beta_k 
# are analyzed in Tutorial 13.

# Although beta is fixed across constraints, it is informative to compare PCS
# designs over a range of beta values. The candidate grid below spans negative,
# zero, and positive slopes: negative and zero values serve as reference cases
# that illustrate how the CCF profile and peak location respond to different
# slope targets before any look-ahead is imposed. Shifting the peak of the CCF 
# towards h = 12 (to look ahead) assumes a positive slope beta > 0. 
beta_vec <- c(-0.1, -0.0001, -0.00007, -0.00005, -0.000035, -0.00002,
              -0.00001,  0,  0.000003,  0.000009,  0.000012,  0.00002,
              0.00004,  0.00008,  0.00016,  0.1,  0.2,  0.3) / 2

# Selecting informative beta values manually can be difficult. PCS_func()
# addresses this by automatically constructing a candidate grid concentrated
# around the tipping point of the PCS optimization — the region where the
# predictor reacts most sensitively to small changes in beta. Screening
# solutions in this neighbourhood often provides the sharpest insight into the
# structure of the optimization problem.
beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- TRUE

PCS_obj  <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                     Type_III, scaled_constraints, high_resolution)

# We can sweep over either the manually constructed grid or the automatically
# generated one. Here we use the automatic grid as the base, and augment it
# with additional slope values at which the predictor changes
# profile sharply (identified from prior inspection of the solution path).
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec <- c(beta_vec_automatic[1],
              -8e-05, -4e-05, -3e-05, -2e-05,
              beta_vec_automatic[2:length(beta_vec_automatic)])

# ─────────────────────────────────────────────────────────────────────────────
# 1.7 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────────────

# For each beta in beta_vec, compute the regularised Type I PCS predictor
# using criterion (46), solution (49), from Appendix D of Wildi (2026).

b_mat <- NULL   # stores filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the Type I PCS predictor for the current slope target.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint satisfaction check: for a feasible system, the residual of each
  # slope constraint should approach zero as lambda -> Inf. Each printed value
  # is the residual for one of the h = 12 constraints. Large lambda implies
  # small residuals, provided the problem is feasible (assuming sufficient 
  # numerical precision).
  # Note: the exact closed-form solution in exercise 2 will shrink these 
  # deviations to zero.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 7))


# ─────────────────────────────────────────────────────────────────────────────
# 1.8 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above: for a feasible system, the residual of each
# slope constraint should vanish as lambda increases (subject to numerical
# precision).

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# preserves the direction of any trend or level shift present in the data.
# For the smallest positive slopes (beta > 0) the coefficient sum remains positive,
# indicating that trend orientation is preserved. Beyond a certain slope
# threshold the sum turns negative, signaling trend inversion.
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# A sufficiently small positive slope (beta > 0) ensures that the target
# covariance remains positive. Beyond a critical slope threshold this
# positivity is violated and the corresponding predictors become unusable.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 6)))


# ─────────────────────────────────────────────────────────────────────────────
# 1.9 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Observations on Filter Weights and CCF Behavior:
#
# CCFs (right panel):
#   - The CCF profiles conform broadly to the imposed slope constraints:
#     decreasing when beta < 0 and increasing otherwise.
#   - Small deviations from the rigid linear (fixed-slope) growth path are
#     possible because the regularization weight lambda is large but finite.
#     The exact closed-form solution in Exercise 2 will impose a strictly
#     linear path with no such deviations.
#   - Some designs (green and cyan tones) yield a CCF that increases from
#     k = 0 to k = 11 while maintaining CCF(12) > 0: they are almost feasible. 
#     However, the final slope CCF(12) - CCF(11) is not positive for any of 
#     these designs; a positive final slope is achieved only by sign-inverting 
#     designs (blue to violet tones), which are directionally unusable.
#
# Filter weights (left panel):
#   - The classic MSE(12) and MSE(24) predictors are both pure AR(1) filters
#     and overlap exactly in the plot. This illustrates that the MSE predictor
#     is structurally incapable of looking further ahead than MSE(12): extending
#     the horizon merely rescales the same AR(1) coefficient vector.
#   - The simple nearly monotonous and smooth patterns of the CCF in the right 
#     panel contrast with the wildly changing and nearly discontinuous predictor
#     profiles in the left panel. This stark contrast is indicative of 
#     misspecification.
#   - The PCS predictor weight profiles appear irregular and difficult to
#     interpret. Imposing the Type I CCF constraints forces the optimizer to
#     satisfy a monotonic profile that contradicts the DGP structure, resulting
#     in filter weights that lack a coherent economic or statistical
#     interpretation.
#   - Despite the PCS constraints being nearly satisfied (monotonous CCFs), and 
#     despite a rightward shift of the CCF peak relative to the MSE benchmark 
#     (cyan tones), none of the Type I PCS predictors are usable from a
#     look-ahead perspective.
#   - The root cause is the imposition of a monotonic (constant-slope) CCF
#     profile over k = 1, …, h, which the MSE predictor's own CCF reveals to
#     be non-monotonic. Two remedies are available:
#       (a) Relaxing the regularization weight on the Type I constraints allows
#           the CCF to follow a non-monotonic path while still shifting its peak
#           rightward, yielding usable look-ahead behavior; see Exercise 3.
#       (b) Replacing Type I with the simpler Type III constraint is equally
#           effective at restoring look-ahead behavior by avoiding the
#           monotonicity requirement altogether; see Exercise 4.
#     Note, however, that Type II — which imposes a single local slope
#     constraint at the final lag — proves ineffective in this setting and
#     produces difficult-to-interpret outcomes; see Exercises 5 and 6.


# ─────────────────────────────────────────────────────────────────────────────
# 1.10 Compare Forecasts: 
# ─────────────────────────────────────────────────────────────────────────────
###############################################################################
# ALL PCC PREDICTORS IN THIS EXERCISE ARE SUBJECT TO MISSPECIFIED CONSTRAINTS 
#                             AND UNUSABLE
###############################################################################

# ── 1.10.1 Apply Predictors to Data ──────────────────────────────────────────
# All filters are expressed in MA form (applied to the innovations eps_t from
# the Wold decomposition). We therefore convolve each filter with the ARMA
# model residuals, which serve as empirical proxies for the innovations.
x_filt <- arima.obj$residuals

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

# ── 1.10.2 Plot ───────────────────────────────────────────────────────────────

# Full-sample overlay of all predictor outputs (scaled for comparability).
par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)
ts.plot(mplot,
        main = "Predictor Outputs: Full Sample",
        col  = colo, xlab = "", ylab = "",
        lty  = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd  = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)

# Representative excerpt (observations 200–250) for closer visual inspection.
anf   <- 200
enf   <- 250
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[anf:enf, ]
ts.plot(mplot,
        main = "Predictor Outputs: Observations 200–250",
        col  = colo, xlab = "", ylab = "",
        lty  = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd  = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)

# ── 1.10.3 Empirical CCFs ─────────────────────────────────────────────────────
par(mfrow = c(2, 2))
select_vec <- c(2, 3 + 3:5)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ", colnames(y_out_mat)[i]))
}



# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 — Closed-Form Exact PCS
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# As shown in Section 1.5, the PCS constraint system has full rank and can
# therefore be solved exactly in closed form; see equations (47) and (48) in
# Wildi (2026). Here we compute the corresponding closed-form solutions for the 
# same beta values used in Exercise 1, using PCS_closed_form_func(), and then 
# compare the exact solutions with the strongly regularized solutions from 
# Exercise 1. For large lambda, closed-form and regularized solutions should 
# be nearly identical.

# DGP specification for PCS_closed_form_func(): 
# Yearly Growth, i.e. convolution of ARMA(1,1) with equally-weighted MA(12).
gamma_pcs <- gamma

Delta  <- 1:h



# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Closed-Form PCS Based on PCS_closed_form_func()
# ─────────────────────────────────────────────────────────────────────────────

b_closed_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)
  
  b             <- PCS_obj$b
  d_delta       <- PCS_obj$d_delta
  b_closed_mat  <- cbind(b_closed_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly, so residuals should be zero (up to machine
  # precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_closed_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above: residuals are zero by construction for the
# closed-form solution.

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# The outcome is similar to exercise 1.
apply(b_closed_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# The outcome DIFFERS from exercise 1: ALL positive beta lead to NEGATIVE target 
# correlations CCF(h). This illustrates INFEASIBILITY: the Type I constraints 
# are strongly misspecified, conflicting with the data generating process.
t(b_closed_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_closed_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("Closed-form PCS, beta=",
                                 round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_closed_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_closed_mat <- cbind(ccf_closed_mat,
                          compute_acf_at_lags_zero_delta_func(
                            max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_closed_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# While the filter weights (left panel) are virtually identical to those of
# Exercise 1 and therefore equally unusable, the CCF profiles (right panel)
# differ slightly but systematically:
#
#   - beta < 0: the CCF is strictly monotonically decreasing with CCF(h) > 0.
#   - beta > 0: the CCF is strictly monotonically increasing with CCF(h) < 0.
#
# No design simultaneously achieves a positive CCF(h) and an increasing CCF
# profile. A fortiori, no design produces a CCF that peaks at the forecast
# horizon h. This confirms that the Type I constraint system is structurally
# incompatible with useful look-ahead behavior in this ARMA(1,1) setting.
#
# In Exercise 1, the finite regularization weight allowed small departures from
# the rigid closed-form profile, so that certain designs (cyan tones) exhibited
# an increasing CCF with CCF(h) > 0. The closed-form solution eliminates this
# residual flexibility entirely. In any case, all designs remain unusable 
# without exception.


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 Compare CCFs: Strong Regularization vs. Closed-Form Exact PCS
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

mplot <- ccf_mat

plot(mplot[, 1],
     main = "Strong Regularization",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()


mplot <- ccf_closed_mat

plot(mplot[, 1],
     main = "Exact Closed-Form",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Comparison:
#   - Strong regularization (left): because lambda is finite, a small residual
#     degree of freedom remains after satisfying the constraints. This allows
#     the optimizer to marginally inflate CCF(h) while keeping CCF(h) > 0 
#     positive (cyan tone).
#   - Exact closed-form (right): all constraints are satisfied exactly. The CCF 
#     profiles are strictly linear from k=0 to k=h. 
#   - The closed-form CCF illustrates infeasibility: it is not possible to obtain 
#     a linearly increasing CCF with CCF(h)>0. 
#   In both cases the fundamental problem is the same: the rigid Type I
#   constraint system is incompatible with achieving CCF(h) > 0 alongside
#   an increasing slope, confirming the findings of Exercise 1.

# However, out of curiosity, we could try to reduce the number of constraints.


# ─────────────────────────────────────────────────────────────────────────────
# 2.5 Reducing the Constraint Space
# ─────────────────────────────────────────────────────────────────────────────

# In comparison to 2.1 we here reduce the number of constraints from 12 to 11: 

Delta  <- 1:(h-1)

beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- TRUE

PCS_obj  <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                     Type_III, scaled_constraints, high_resolution)

# We can sweep over either the manually constructed grid or the automatically
# generated one. Here we use the automatic grid as the base, and augment it
# with additional slope values at which the predictor changes
# profile sharply (identified from prior inspection of the solution path).
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec <- beta_vec_automatic


# ─────────────────────────────────────────────────────────────────────────────
# 2.6 Reducing the Constraint Space
# ─────────────────────────────────────────────────────────────────────────────

b_closed_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)
  
  b             <- PCS_obj$b
  d_delta       <- PCS_obj$d_delta
  b_closed_mat  <- cbind(b_closed_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly, so residuals should be zero (up to machine
  # precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_closed_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))


# ─────────────────────────────────────────────────────────────────────────────
# 2.7 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above: residuals are zero by construction for the
# closed-form solution.

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# The PCS solutions are more robust against trend inversion: large beta are required to change sign: 
apply(b_closed_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# The PCS solutions are also more robust against negative target correlation:
t(b_closed_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_closed_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("Closed-form PCS, beta=",
                                 round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 2.8 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_closed_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_closed_mat <- cbind(ccf_closed_mat,
                          compute_acf_at_lags_zero_delta_func(
                            max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_closed_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# CCFs (right panel):
#   - The Type I PCS problem with 11 constraints (Delta = 1:11) is feasible:
#     the CCF peak reaches k = h - 1 = 11 while maintaining CCF(h) > 0,
#     confirming that a partial but genuine rightward shift is achievable
#     within the yearly-growth GDP structure.

# Filter weights (left panel):
#   - The filter profiles corresponding to a CCF peak at k = 11 (blue tones)
#     exhibit clear look-ahead behavior: progressively more weight is
#     assigned to the lag-0 observation as the target peak moves closer to
#     the forecast horizon h = 12.

#@@@??? filter the data; try b1>0 to see if feasible with full Delta.


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OUTCOME — Exercises 1 and 2
# ══════════════════════════════════════════════════════════════════════════════
#
# In both exercises the predictors are flawed and unusable. Imposing a rigid,
# high-dimensional Type I constraint system severely impairs forecast
# performance at horizon h. The resulting predictors are unsuitable for any
# practical look-ahead application. Removing a constraint made the problem 
# feasible but with mitigated look ahead sucess. 
#
# ══════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 — As Exercise 1 but with Moderate Regularization
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# The Type I constraint system imposes h = 12 slope constraints:
#
#   b' * (gamma_k - gamma_{k-1}) = beta,  k = 1, …, 12.
#
# All constraints can be satisfied simultaneously (see exercise 2), but at 
# cost of the target correlation CCF(h).
#
# In Exercise 1, a very large lambda forced near-exact constraint satisfaction,
# leaving no room for the target correlation and producing unusable predictors
# (CCF(h) < 0 for beta > 0).
#
# Here we retain the same slope grid beta_vec but reduce lambda to a moderate
# value. This rebalances the criterion between constraint satisfaction and
# target correlation maximization: the constraints are now enforced only
# approximately, but CCF(h) is restored to a positive and meaningful level.

# DGP specification for PCS_func(): 
# Yearly Growth, i.e. convolution of ARMA(1,1) with equally-weighted MA(12).
gamma_pcs <- gamma
# Type I constraints
Delta <- 1:h

# Moderate regularization weight: balances constraint enforcement against
# target correlation, avoiding the sign inversion observed in Exercise 1.
lambda <- 10

# ── Automatic Beta Grid ───────────────────────────────────────────────────────
# Use PCS_func() to generate a candidate beta grid concentrated around the
# tipping point of the PCS optimization (the region of highest sensitivity).
beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- FALSE

PCS_obj            <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                               Type_III, scaled_constraints, high_resolution)
beta_vec_automatic <- PCS_obj$beta_vec

# Manually augment the automatic grid with additional points near beta = 0
# to increase resolution in the region where the predictor profile changes most
# rapidly.
beta_vec <- c(beta_vec_automatic[1:10],
              c(-0.05, 0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.18),
              beta_vec_automatic[12:length(beta_vec_automatic)])


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────────────

# For each beta in beta_vec, compute the regularised Type I PCS predictor
# using criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the regularised Type I PCS predictor for the current slope target.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: with moderate lambda the residuals will not vanish
  # exactly (unlike the closed-form solution). For each beta in the loop, 
  # the 12 printed values are the residuals from the h = 12 constraints.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above. With moderate lambda, non-vanishing residuals are
# expected and acceptable; they reflect the deliberate trade-off between
# constraint satisfaction and target correlation.

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# A strictly positive coefficient sum confirms that the filter preserves the
# direction of any trend or level shift in the data. With moderate lambda,
# the optimizer retains enough freedom to avoid sign inversion for a wider
# range of beta values than in Exercise 1.
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# Designs with strongly positive beta enforce an increasing CCF profile (desirable), which 
# can push the target covariance negative (undesirable) and render the predictor 
# unusable. In contrast to exercises 1 or 2, much larger beta are allowed before 
# the target correlation turns negative.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Observations on Filter Weights and CCF Behavior:
#
# Filter weights (left panel):
#   - The classic MSE(12) and MSE(24) predictors are both pure AR(1) filters
#     and overlap exactly after scaling. This confirms the structural ceiling
#     on MSE look-ahead: extending the horizon beyond h = 12 yields no
#     additional predictive information.
#   - Unlike Exercise 1, the PCS filter weights here are smoother and more
#     interpretable. A fixed moderate lambda=10 allows the optimizer to distribute
#     weights more evenly across lags, rather than concentrating them solely
#     to satisfy the rigid Type I constraints.
#   - Strongly negative beta values (red-to-green tones) remain subject to
#     misspecification. However, filters gradually morph toward the MSE
#     solution as beta increases (cyan tones), signaling
#     progressively lesser and finally resorbed misspecification.
#     Note that results for beta < 0 are included for completeness only;
#     in practice, achieving a rightward CCF peak shift requires beta > 0.
#   - As beta increases above zero, positive weight is progressively
#     concentrated at lag 0, consistent with the intuition that greater
#     look-ahead can, in certain settings such as this one, be achieved by
#     stronger anchoring to the current level of the series (the DFP approach
#     would contradict this intuition in the general case).
#   - Larger positive beta values push weights at higher lags progressively
#     into negative territory. This amplifies the rightward shift of the CCF
#     peak (right panel), but risks inducing trend reversion (see Exercise 3.2
#     above), ultimately driving the target correlation CCF(h) below zero
#     (see right panel) and rendering the predictor unusable.

# CCFs (right panel):
#   - With a fixed moderate lambda = 10, the CCF peak shifts progressively
#     rightward with increasing beta relative to the MSE benchmark, indicating
#     growing look-ahead behavior (blue tones). However, the peak does not
#     reach k = h = 12 exactly: a peak precisely at the target horizon is
#     infeasible for this DGP. The partial shift nonetheless constitutes
#     genuine look-ahead behavior, which is the primary objective of the PCS
#     approach.
#   - Increasing beta further causes the target correlation CCF(h) to shrink
#     progressively, eventually turning negative for excessive beta values
#     (violet tone), at which point the predictor becomes unusable.
#   - The accuracy-timeliness trade-off is clearly visible: larger beta values
#     push the CCF peak further right but simultaneously reduce its height
#     (lower CCF(h)), reflecting the inherent tension between forecast
#     timeliness and accuracy.

# ─────────────────────────────────────────────────────────────────────
# 3.4 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 3.4.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 3.4.2 Plot
#----------------------------------------------------------------------

# Select the PCS designs with positive slope beta>=0
colnames(y_out_mat) 
select_pcs<-15:22
# Add nowcast and the two MSE predictors
select_vec<-c(1:3,select_pcs)


# MSE(12) and MSE(24) overlap exactly after scaling: no additional look-ahead
# behavior can be obtained within the MSE framework by increasing the forecast
# horizon beyond h = 12. In other words, the MSE predictor is effectively
# "stuck at horizon 12", confirming the structural ceiling on MSE look-ahead.

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)[,select_vec]
colnames(mplot)<-colnames(y_out_mat)[select_vec]
coli<-colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=coli[i],line=-i)



# Magnify Dotcom crisis
anf<-100
enf<-170

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,select_vec]
colnames(mplot)<-colnames(y_out_mat)[select_vec]
coli<-colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=coli[i],line=-i)




# Magnify Financial Crisis
anf<-200
enf<-250

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,select_vec]
colnames(mplot)<-colnames(y_out_mat)[select_vec]
coli<-colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=coli[i],line=-i)




# The empirical CCF between the nowcast and each PCS predictor output is used
# to verify that the peak shifts rightward (ideally toward the forecast
# horizon h) as beta increases, confirming that the single Type III PCS
# constraint successfully advances the predictor. 
par(mfrow = c(2, 2))
select_vec<-c(2,16,19,21)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ", colnames(y_out_mat)[i]))
}



# The empirical CCF between MSE(12) and each PCS predictor output is used
# to verify that the peak shifts progressively to the right of the MSE(12)
# benchmark as beta increases, confirming that the single Type III PCS
# constraint successfully advances the predictor. A progressively right-skewed
# CCF with the peak shifting away from zero is the expected signature of
# increasing look-ahead behavior.
par(mfrow = c(2, 2))
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 2]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("MSE(12) vs. ", colnames(y_out_mat)[i]))
}




# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: As Exercise 1 but Type III PCS with Strong Regularization
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


# DGP specification for PCS_func(): 
# Yearly Growth, i.e. convolution of ARMA(1,1) with equally-weighted MA(12).
gamma_pcs <- gamma

# Instead of imposing a monotonically increasing CCF pattern CCF(k) > CCF(k-1)
# for k = 1, ..., h (Type I PCS), we here impose a single constraint on the
# average growth between k = 0 and k = h = 12: CCF(12) > CCF(0), i.e.,
# CCF(12) - CCF(0) = beta > 0. Because only one constraint is imposed rather than
# h constraints, the feasible set is much larger, and very strong
# regularization becomes admissible without risking infeasibility.

# Delta: constraint spans from lag 0 to lag h = 12
Delta <- c(0, h)

# Inform PCS_func that a Type III constraint is used: otherwise Delta <- c(0, h) 
# is not correctly interpreted as b' * (gamma_h - gamma_0) = beta.
Type_III <- TRUE

# Very strong regularization: 
lambda <- 1000000


# ─────────────────────────────────────────────────────────────────────────────
# Select a range of beta values
# ─────────────────────────────────────────────────────────────────────────────
# Start with the automatic beta grid returned by PCS_func, then override it
# with a coarser manual grid that emphasises the main look-ahead designs.
# The broader automatic range is discarded to avoid redundant solutions that
# closely resemble those already examined in Exercise 3.
beta <- 0
high_resolution <- FALSE

PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III,
                    scaled_constraints, high_resolution)

beta_vec_automatic <- PCS_obj$beta_vec
beta_vec           <- beta_vec_automatic

# Coarser manual grid: covers 
# beta < 0 (CCF(h) < CCF(0)), 
# beta = 0 (CCF(h) = CCF(0)), and 
# beta > 0 (CCF(h) > CCF(0)).
beta_vec <- c(-0.1, 0, 0.1, 0.2, 0.5)


# ─────────────────────────────────────────────────────────────────────
# 4.1 PCS Optimisation over the Beta Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised Type III PCS predictor
# using criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the Type III PCS predictor for the current beta
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system the residual of the single Type III
  # constraint should shrink toward zero as lambda -> Inf. The printed value
  # is the absolute deviation of CCF(12) - CCF(0) from its target beta.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 4.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: Type III constraint residuals ────────────────────────────
# Validated in the loop above: for a feasible system, the residual of the
# single slope constraint should vanish as lambda increases.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# does not invert the direction of a trend or level shift in the data.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 4.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Filter weights (left panel):
#   - The classic MSE(12) and MSE(24) predictors overlap (after scaling) and 
#     are pure AR(1) filter, as in previous exercises. The Type III PCS filters 
#     are shaped by the single `average-growth' constraint 
#             CCF(h) - CCF(0) = beta, 
#     rather than the full monotonicity requirement of Type I used in previous 
#     exercises.
#   - Increasing beta progressively concentrates weight at lag 0, consistent
#     with the intuition that stronger look-ahead is achieved by greater
#     anchoring to the current level of the series in this example.

# CCFs (right panel):
#   - The Type III constraint targets the mean growth between k = 0 and
#     k = h = 12: beta > 0 enforces CCF(12) - CCF(0) > 0. This does not
#     guarantee that the CCF peak shifts precisely to k = h (which is
#     infeasible for this DGP), but a progressive rightward shift relative
#     to the MSE benchmark is still achieved, constituting genuine look-ahead
#     behavior.
#   - As in Exercise 3, excessively large beta values reduce the target
#     correlation CCF(h), eventually rendering the predictor unusable.
#   - However, imposing only a single constraint leaves considerably more
#     room for the optimizer, so that CCF(h) remains generally larger for
#     comparable degrees of look-ahead relative to the Type I case.


# ─────────────────────────────────────────────────────────────────────
# 4.4 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────

# ── 4.4.1 Apply Predictors to Data ───────────────────────────────────
# All filters are defined in MA form (as applied to the innovations eps_t
# in the Wold decomposition), so they are applied directly to the model
# residuals from the fitted ARIMA object.
x_filt <- arima.obj$residuals

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


# ── 4.4.2 Full-Sample Plot ────────────────────────────────────────────
# Include all designs:
select_vec <- 1:ncol(y_out_mat)

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── 4.4.3 Dotcom Crisis Episode (observations 100–170) ───────────────
anf <- 100
enf <- 170

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[anf:enf, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs: Dotcom Crisis", col = coli,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── 4.4.4 Financial Crisis Episode (observations 200–250) ────────────
anf <- 200
enf <- 250

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[anf:enf, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs: Financial Crisis", col = coli,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── 4.4.5 Empirical CCF Diagnostics ──────────────────────────────────
# The empirical CCF between the nowcast and each PCS predictor output is used
# to verify that the peak shifts rightward (ideally toward the forecast
# horizon h) as beta increases, confirming that the single Type III PCS
# constraint successfully advances the predictor. 
par(mfrow = c(2, 2))
select_vec <- c(2, 6:8)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ", colnames(y_out_mat)[i]))
}



# The empirical CCF between MSE(12) and each PCS predictor output is used
# to verify that the peak shifts progressively to the right of the MSE(12)
# benchmark as beta increases, confirming that the single Type III PCS
# constraint successfully advances the predictor. A progressively right-skewed
# CCF with the peak shifting away from zero is the expected signature of
# increasing look-ahead behavior.
par(mfrow = c(2, 2))
select_vec <- c(2, 6:8)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 2]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("MSE(12) vs. ", colnames(y_out_mat)[i]))
}


# ════════════════════════════════════════════════════════════════════
# EXERCISE 5: As Exercise 4 but PCS Type II
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# DGP specification for PCS_func():
# Yearly growth, i.e., convolution of ARMA(1,1) with an equally-weighted MA(12).
gamma_pcs <- gamma

# As in Exercise 4, a single constraint is imposed. Type II targets the local
# slope of the CCF at the forecast horizon: CCF(h) - CCF(h-1) = beta.
Delta <- h

# Note: setting Type_III <- TRUE with a scalar Delta would cause an error,
# since Type III requires two lags, i.e., Delta <- c(k, h), imposing
# CCF(h) - CCF(k) = beta (typically with k = 0). Here, Delta = h implies
# the Type II constraint CCF(h) - CCF(h-1) = beta. Either set Type_III <- FALSE
# explicitly or omit it from the function call (the default is FALSE).
Type_III <- FALSE

# Very strong regularization: feasible given the single constraint
lambda <- 10000000


# ─────────────────────────────────────────────────────────────────────────────
# Select a range of beta values
# ─────────────────────────────────────────────────────────────────────────────
# A manual grid is defined first for reference, then overridden by the broader
# automatic grid returned by PCS_func.
beta_vec <- c(-0.1, 0, 0.0001, 0.1, 0.2, 0.5)

# Automatic beta grid
beta           <- 0
high_resolution <- FALSE

PCS_obj            <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III,
                               scaled_constraints, high_resolution)
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec           <- beta_vec_automatic


# ─────────────────────────────────────────────────────────────────────
# 5.1 PCS Optimisation over the Beta Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised Type II PCS predictor
# using criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the Type II PCS predictor; Type_III = FALSE is supplied explicitly.
  # Alternatively, Type_III may be omitted since FALSE is the default:
  #   PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: the problem is feasible so deviations should be small.
  # However, the constraint is numerically demanding, requiring very large
  # lambda to drive the residual close to zero.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 8))


# ─────────────────────────────────────────────────────────────────────
# 5.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: Type II constraint residuals ────────────────────────────
# Validated in the loop above: for a feasible system, the residual of the
# single slope constraint should vanish as lambda increases.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# does not invert the direction of a trend or level shift in the data.
# Note: small positive beta values do not induce trend reversion; verify accordingly.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Small positive beta values preserve a positive filter sum (no trend
# reversion) and a positive target correlation CCF(h) > 0. Larger beta values,
# however, drive the target correlation negative, rendering the predictor
# unusable.
t(b_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 8)))

# ─────────────────────────────────────────────────────────────────────
# 5.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()


# CCFs (right panel):
#   1. The Type II problem is feasible: CCF(h) - CCF(h-1) = beta > 0 and
#      CCF(h) > 0 can be satisfied simultaneously for sufficiently small
#      positive beta (cyan tones). For excessively large beta, CCF(h) turns
#      negative (blue to violet tones) and the predictor becomes unusable.
#   2. A positive local slope CCF(h) - CCF(h-1) does not induce a systematic
#      rightward shift of the CCF peak, contrary to what one might expect 
#      (see tutorial 11).
#   3. Quite the contrary, designs with negative beta (red to yellow tones)
#      produce an unexpected rightward peak shift in this example.
#   4. The filter weight plot (left panel) illustrates the main reason:
#      smaller beta values (red to yellow tones) assign more weight to lag 0,
#      which drives the unanticipated look-ahead behavior.

# Filter weights (left panel):
#   - The filter profile is governed by the numerically demanding constraint
#     CCF(h) - CCF(h-1) at the forecast horizon h = 12.
#   - For small beta (red and orange tones), most weight is concentrated at
#     lag 0, producing unexpected look-ahead behavior precisely in those
#     designs that least conform to the CCF(h) - CCF(h-1) > 0 rule.
#   - As beta increases, weight at lag 0 decreases progressively (yellow to
#     green tones). Eventually, lag-0 weight turns negative (cyan tones) and
#     finally the target correlation CCF(h) itself turns negative (blue to
#     violet tones), rendering the predictor unusable.

# Summary:
#   The outcome is counterintuitive: the Type II constraint conflicts with the
#   DGP structure such that the designs with the strongest apparent look-ahead
#   promise (CCF(h) > CCF(h-1), cyan tones) are in fact lagging designs that
#   most overtly contradict CCF(h) - CCF(h-1) > 0. As in Exercises 1-3, this
#   counterintuitive result stems from a misspecified constraint that conflicts
#   with the internal structure of the DGP.



# ─────────────────────────────────────────────────────────────────────
# 5.4 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 5.4.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the innovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 5.4.2 Plot
#----------------------------------------------------------------------

# Note: this example is atypical in that the Type II PCS constraint does not
# induce a rightward shift of the CCF peak for this DGP, noting that the 
# problem is not feasible (misspecification). As an unexpected consequence,
# designs with negative beta (rather than positive beta) exhibit the
# look-ahead behavior in this case.

# ── Full-sample plot ──────────────────────────────────────────────────
select_vec <- 1:6

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── Dotcom crisis episode (observations 100–170) ─────────────────────
anf <- 100
enf <- 170

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[anf:enf, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs: Dotcom Crisis", col = coli,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# Note: as above, the Type II constraint does not address the rightward CCF
# peak shift for this DGP. Designs with negative beta (rather than positive
# beta) are the ones exhibiting look-ahead behavior in this episode.

# ── Financial crisis episode (observations 200–250) ───────────────────
anf <- 200
enf <- 250

par(mfrow = c(1, 1))
mplot <- scale(y_out_mat, center = FALSE, scale = TRUE)[anf:enf, select_vec]
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- colo[select_vec]
ts.plot(mplot,
        main = "Predictor Outputs: Financial Crisis", col = coli,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(mplot) - 2)),
        lwd = c(1, 2, rep(1, ncol(mplot) - 2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)




# ════════════════════════════════════════════════════════════════════
# EXERCISE 6: Same as Exercise 5 but Closed-Form PCS Type II)
# ════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# DGP specification for PCS_func():
# Yearly growth, i.e., convolution of ARMA(1,1) with an equally-weighted MA(12).
gamma_pcs <- gamma

# As in Exercise 4, a single constraint is imposed. Type II targets the local
# slope of the CCF at the forecast horizon: CCF(h) - CCF(h-1) = beta.
Delta <- h


# ─────────────────────────────────────────────────────────────────────
# 6.1 Closed-Form Solution of Type II PCS
# ─────────────────────────────────────────────────────────────────────

b_closed_mat <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_closed_form_func(h,Delta, gamma_pcs, L, beta)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_closed_mat   <- cbind(b_closed_mat, b)
  
  # Constraint check: for a full-rank constraint system the deviations
  # of the closed-form solution vanish:
  print(abs(d_delta %*% b + beta))
}

colnames(b_closed_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))

# ─────────────────────────────────────────────────────────────────────
# 6.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above.



# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# Similar to exercise 5 above:
apply(b_closed_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
#  Similar to exercise 5 above:
t(b_closed_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_closed_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 6.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_closed_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_closed_mat <- cbind(ccf_closed_mat,
                          compute_acf_at_lags_zero_delta_func(
                            max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_closed_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# The results are similar to the strong regularization in exercise 5.



# ════════════════════════════════════════════════════════════════════
# EXERCISE 7: PCS Type III for Monthly Growth — Impossibility of the
#             Pure ARMA(1,1) Case
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# This exercise targets MONTHLY growth directly, i.e., the Wold decomposition
# of the raw ARMA(1,1) is used without convolving with the equally-weighted
# trend filter employed in Exercise 1.


# ─────────────────────────────────────────────────────────────────────
# 7.1 ARMA(1,1) and Constraint Rank (Monthly Growth)
# ─────────────────────────────────────────────────────────────────────

# Visualise the Wold coefficients for both the monthly and yearly growth
# representations to highlight the structural difference.
par(mfrow = c(2, 1))
ts.plot(xi,    main = "Wold Decomposition: Monthly Growth (Post-1990)")
ts.plot(gamma, main = "Wold Decomposition: Yearly Growth (Post-1990)")


# Replace the yearly-growth gamma by xi, the Wold decomposition of the raw
# ARMA(1,1), omitting the convolution with the equally-weighted trend filter.
gamma0    <- xi[1:L]
gammah    <- xi[h + 1:L]
gamma_pcs <- xi
# Note: the h = 12 step-ahead MSE predictor remains AR(1), with autoregressive
# coefficient a1 determined solely by the ARMA(1,1) parameters.


# Compute the rank of the constraint system by forming the matrix of shifted
# Wold coefficient vectors and inspecting its eigenvalues.
gamma_mat <- xi[1:L]
for (i in 1:(L - 1)) {
  gamma_i   <- xi[i + 1:L]
  gamma_mat <- cbind(gamma_mat, gamma_i)
}

eigenvalues <- eigen(gamma_mat)$values

# Key finding: unlike Exercise 1 (yearly growth, rank 13), the constraint
# system for the raw ARMA(1,1) has rank 2 only:
#   - gamma_0 is linearly independent of gamma_h for all h > 0.
#   - gamma_h and gamma_{h+k} are linearly dependent for all h, k > 0.
# This severely limits the degrees of freedom available for PCS optimisation.
length(which(abs(eigenvalues) > 1e-10))


# Implications for each PCS constraint type:
#
# Type III PCS:
#   Imposing a single constraint b'(gamma_h - gamma_0) = beta is feasible.
#   It consumes one of the two available degrees of freedom, leaving one
#   degree free for optimisation (see Tutorial 13, Exercise 1). Crucially,
#   gamma_0 must enter the constraint system: omitting it reduces the rank
#   to one, collapsing the problem to a pure AR(1) form with no genuine
#   look-ahead flexibility (see Tutorial 14 for effective look-ahead
#   solutions in this setting).
#
# Type II PCS:
#   For h > 1, the ARMA(1,1) structure imposes the recursive relationship
#        b'(gamma_{h+1} - gamma_h) = a1 * b'(gamma_h - gamma_{h-1}),  h > 1,
#   meaning the CCF slope decays at an exponential rate regardless of b.
#   Consequently, either b'(gamma_{h+1} - gamma_h) > 0 and CCF(h) < 0, or
#   b'(gamma_{h+1} - gamma_h) < 0 and CCF(h) > 0. The sign of b is the
#   only free parameter, so one either obtains an increasing but negative
#   CCF, or a decreasing but positive CCF. A positive CCF with a peak at
#   h > 1 is therefore impossible; the peak can only be shifted between
#   lags k = 0 and k = 1. The Type II problem is feasible only when h = 1.
#
# Type I with h = 12:
#   The problem is impossible when h > 1. No closed-form solution exists (the system
#   is non-invertible), and even arbitrarily large lambda cannot drive the
#   constraint residuals toward zero; they remain sizeable regardless of
#   the regularisation strength.



# ─────────────────────────────────────────────────────────────────────
# 7.2 PCS Type I: Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# A grid of target slope values beta is imposed on the CCF. A positive beta
# would require CCF(k) to increase linearly from k = 0 to k = h. However,
# since the ARMA(1,1) structure dictates
#        b'(gamma_{h+1} - gamma_h) = a1 * b'(gamma_h - gamma_{h-1}),  h > 1,
# a uniform slope beta is impossible: only an exponentially decaying slope
# of the form beta_k = beta * a1^k would be compatible with the DGP
# (see Tutorial 14). Due to the rank-2 structure of the ARMA(1,1) constraint
# system, the Type I problem with a fixed beta is therefore infeasible and
# the CCF peak cannot be shifted to the right. Negative and zero beta values
# are included as reference cases in this example to illustrate how the CCF 
# profile responds to the slope target under this impossibility.

Delta    <- 1:12
lambda   <- 10^6

# Automatic beta grid
beta           <- 0
high_resolution <- TRUE
Type_III       <- FALSE

PCS_obj            <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III,
                               scaled_constraints, high_resolution)
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec           <- beta_vec_automatic 


# ─────────────────────────────────────────────────────────────────────
# 7.3 PCS Optimisation over the Beta Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised Type I PCS predictor
# using criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the Type I PCS predictor
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Impossibility check: the deviations can be made very small. How is that possible?
  # The answer is: beta is very small, and therefore the constraints can be tightened 
  # by shrkinking the scale of be towards zero: the deviations are small because everything is zero-shrinked. 
  print(abs(d_delta %*% b + beta))
}

# Let us rescale to unit-length:
b_len<-sqrt(sum(b^2))
# Now we see that the deviations are sizeable and cannot be reduced by increasing lambda.
abs(d_delta %*% b + beta)/b_len

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 8))

# ─────────────────────────────────────────────────────────────────────
# 7.4 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: Type I constraint residuals ─────────────────────────────
# Cannot be eliminated: the constraint system is impossible for this DGP,
# so residuals remain sizeable regardless of the regularisation strength.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A positive CCF slope (beta > 0) forces the filter sum to become negative,
# inverting the direction of any trend or level shift in the data.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Consistent with the sign inversion above, the target correlation CCF(h)
# turns negative for beta > 0, confirming that the corresponding predictors
# are unusable.
t(b_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 8)))

# ─────────────────────────────────────────────────────────────────────
# 7.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Although the problem is impossible, the predictors try to comply with the 
# constraints because the regularization weight lambda is very large. 
# The resulting constraint misspecification leads to strange looking predictors which are 
# not looking ahead.

# We shall see later how to address impossible problems, see tutorial 14.


# ─────────────────────────────────────────────────────────────────────
# 7.6 Zoom Into The Regularized Criterion
# ─────────────────────────────────────────────────────────────────────
# Although the problem is infeasible, the regularized problem remains invertible and smooth. 
# We here zoom in in a region where the solution smoothly changes from sign preverting 
# to sign reverting solutions.


# Improve the resolution for the above fixed lambda
beta_vec_original           <- beta_vec_automatic /10

# Increase the resolution at the passage from + to -
beta_vec<-c(beta_vec_original[1:11],1.8e-06,1.9e-06,2e-06,2.1e-06,2.2e-06,2.3e-06,beta_vec_original[12:length(beta_vec_original)])


b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the Type I PCS predictor
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Impossibility check: the deviations can be made very small. How is that possible?
  # The answer is: beta is very small, and therefore the constraints can be tightened 
  # by shrkinking the scale of be towards zero: the deviations are small because everything is zero-shrinked. 
  print(abs(d_delta %*% b + beta))
}

# Let us rescale to unit-length:
b_len<-sqrt(sum(b^2))
# Now we see that the deviations are sizeable and cannot be reduced by increasing lambda.
abs(d_delta %*% b + beta)/b_len

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 8))


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,center=F,scale=T)
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],
     lty = lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], gamma_pcs)$cor_vec)
mplot <- ccf_mat

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()


# Filter weights (left panel):
#   - None of the above designs yields a usable look-ahead predictor; none 
#     leads the MSE predictor.
#   - The regularised criterion always provides a smooth solution of
#     continuity between the two extreme designs (see Tutorials 13 and 14
#     for additional background). However, very large lambda render the design 
#     nearly singular and numerical computations more difficult (than strictly 
#     necessary).

# CCFs (right panel):
#   - Although the Type I problem is infeasible for this ARMA(1,1) DGP,
#     the CCF peak shifts from k = 0 (red to green tones)
#     toward k = 1 (cyan to violet tones) as beta increases. This unit
#     shift is the maximum rightward displacement achievable under the
#     rank-2 ARMA(1,1) structure (assuming b1 < 0), confirming the
#     structural impossibility of genuine look-ahead beyond lag 1. Low-rank  
#     problems will be discussed in Tutorial 13 and solutions are presented in 
#     Tutorial 14.



















# Main take aways:


# -Forecasting the ARMA(1,1) poses an impossible look ahead problem: the CCF peak 
# cannot be shifted at h>1.

# -Applying an equally-weighted trend specification (yearly growth) allowed to expand 
# the rank 2 to a rank 13 constraint system. The peak of the CCF could still not be shifted to 
#  k=h=12 (but be shifted towards 
# k > 2 but it is still impossible to shift further away than h=12.


# For business-cycle analysis one would typically on an alternative HP trend instead of the equally-weighted trend, 
# see tutorial 12.

# Impossibility and infeasibility will be discussed in Tutorial 13. 


# Finally, while a problem might be effectively impossible (no peak shift at k=h), we contend that it is still 
# possible to generate look ahead beviour out of such problems, see Tutorial 14.

