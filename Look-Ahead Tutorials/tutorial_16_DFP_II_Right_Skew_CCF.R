# ==============================================================================
# TUTORIAL 16 - DFP II: DECOUPLE FROM PAST
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────

# In challenging forecast problems where the MSE predictor exhibits "stuck at
# present" behavior (increasing the forecast horizon does not improve 
# advancement), the original DFP (Decouple From Present) and Peak Correlation 
# Shift (PCS) impose constraints on the predictor to encourage
# "look-ahead behavior". This refers to a left-shift of the predictor (lead or
# advancement) relative to the MSE benchmark, while tracking the target x_{t+h}
# as closely as possible (at the intended forecast horizon h and subject to the
# constraint).

# In some difficult forecast problems, the constraint(s) conflict(s) strongly with
# the Data Generating Process (DGP), thereby unduly degrading performance at
# forecast horizon h. In certain cases, such constraints cannot be implemented
# at all (see Tutorials 12-13).

# One approach to handling difficult or even infeasible forecast problems, where
# the look-ahead constraint(s) cannot be satisfied, is to slightly modify the
# original process to gain additional degrees of freedom for imposing the
# look-ahead constraint(s) (see the perturbation based approach in Tutorial 14).

# The AR(1) process represents a particularly challenging example, since the
# Cross Correlation Function (CCF) at positive lags (future) of the predictor
# is immutable (up to sign and scale), directly conflicting with the classic 
# look-ahead constraints of DFP or PCS. Perturbation-based approaches instead 
# affect the left tail (negative lags, the `past') of the CCF, resulting in a 
# right-skew of the CCF (see Tutorial 14).

# Here we formalize this idea by introducing look-ahead constraints that target
# the left tail, i.e., the past, of the CCF. This approach is referred to as
# Decouple From Past (DFP II).

# DFP II can be viewed as a generalization of the original DFP (Decouple Fom 
# Present), where the constraint targets lag 0 rather than the entire past. As 
# such, DFP II could have been introduced at the end of the original DFP 
# tutorials. However, given that DFP II is particularly relevant and effective 
# for difficult or even infeasible forecast problems (e.g., AR(1)), it is 
# presented after the corresponding sequence of Tutorials 12, 13, and 14.

# We briefly revisit the AR(1) case to illustrate the difficulty of imposing
# constraints on the right tail (future) of the CCF, and to motivate the core
# idea of instead targeting the left tail (past).


# ─────────────────────────────────────────────────────────────────────────────
# THE AR(1) DGP: THE HARDEST CASE
# ─────────────────────────────────────────────────────────────────────────────

# The AR(1) DGP represents the most challenging case for PCS (or DFP). Its
# autocorrelation structure satisfies the Yule-Walker equations:
#
#   ACF(k) = a1 * ACF(k-1),
#
# which define a rank-one system that leaves no room to adjust or reshape the
# profile of the CCF for lags k = 0, ..., h (up to sign change). However, room
# is left for negative lags, which PCS or DFP do not explicitly address. Let's 
# see why.
#
# Denoting the h-step MSE predictor in MA-form as gamma_h, with gamma_0 being
# the nowcast (i.e., the original Wold decomposition of the DGP), the
# Yule-Walker equations imply:
#
#   gamma_{h+k} = a1^k * gamma_h,  for any h, k >= 0.
#
# This property is called self-similarity: all h-step MSE predictors are
# proportional to gamma_0, differing only by the scalar factor a1^h. This 
# proportionality conflicts with DFP and PCS approaches, which assume linear 
# independence (e.g., linearly independent nowcast and predictor).
#
# Self-similarity forces the CCF of any predictor b to satisfy:
#
#   CCF(k) = (b' %*% gamma_k) / (||b|| * ||gamma_0||) = a1^k * CCF(0).
#
# For a1 > 0, the CCF decays monotonically and exponentially from its peak at
# k = 0. This pattern is rigidly enforced by the DGP on every predictor b via
# the Yule-Walker equations. Consequently, reshaping the CCF according to any
# of the PCS constraint types is generally impossible, unless one merely
# replicates the original AR(1) profile (up to sign and scale).
#
# However, for negative lags k = -1, -2, ...:
#
#   CCF(k) = (b[(-k+1):L]' %*% gamma_0[1:(L+k)]) / (||b|| * ||gamma_0||)
#
# This expression depends on the predictor b and the negative lag k, and hence
# can be controlled to some extent by b, irrespective of gamma_0, assuming
# gamma_h != 0.


# ─────────────────────────────────────────────────────────────────────────────
# DFP II: DECOUPLE FROM PAST
# ─────────────────────────────────────────────────────────────────────────────

# Main Ideas

# 1. Decouple From PAST, DFP II, addresses the CCF at NEGATIVE lags k = -1, -2, ...:
#
#      CCF(k) = (b[(-k+1):L]' %*% gamma_0[1:(L+k)]) / (||b|| * ||gamma_0||)
#
#    This expression depends on the predictor b and the negative lag k, and
#    hence can be controlled to some extent by b.

# 2. Specifically, we aim to control the aggregated CCF over a range of past
#    lags k = -l_start, ..., -l_end. This aggregated CCF is proportional to:
#
#      sum over k = -l_start, ..., -l_end  of  b[(-k+1):L]' %*% gamma_0[1:(L+k)]
#         (ignoring the scaling 1/ (||b|| * ||gamma_0||)). 
#
#    This expression simplifies to:
#
#      b' %*% Sigma %*% gamma_0
#
#    where Sigma is the integration operator of order (l_end - l_start).
#    See examples below. Note that when Sigma equals the identity matrix,
#    DFP II replicates the original DFP. DFP II is therefore a generalization
#    of the original Decoupling approach.

# 3. The integration operator Sigma, applied over the range of past lags,
#    increases the rank of the constraint system. Even if the DGP conflicts
#    with the classic DFP or PCS constraint types, as in the low-rank AR(1)
#    case where gamma_0 and gamma_h are linearly dependent for all h, the
#    integration operator of order (l_end - l_start) augments the system rank.
#    That is, Sigma %*% gamma_0 and gamma_h are linearly independent (provided
#    gamma_h != 0). 

# 4. DFP II Criterion: The optimization problem is defined as:
#
#      maximize    b' %*% gamma_h
#      subject to  b' %*% Sigma %*% gamma_0 = alpha_0
#
#    This criterion is feasible provided that gamma_h is not proportional to
#    Sigma %*% gamma_0. The support of Sigma, defined by l_start and l_end,
#    can be chosen such that gamma_h and Sigma %*% gamma_0 are linearly
#    independent. For example, if the DGP is AR(1), then Sigma %*% gamma_0
#    and gamma_h are linearly independent for any Sigma other than the identity
#    (see examples below). When Sigma equals the identity, DFP II reduces to
#    the original DFP.

# 5. Setting alpha_0 to a small value in the constraint effectively reduces
#
#    sum over k = -l_start, ..., -l_end  CCF(k)  ∝  b' %*% Sigma %*% gamma_0
#
#    i.e., it reduces b' %*% Sigma %*% gamma_0. This decouples the
#    predictor from the past of the series at lags k = -l_start, ..., -l_end.
#    At the same time, maximizing the target correlation at the forecast
#    horizon, i.e., maximizing CCF(h) at the positive lag k = h, introduces
#    an asymmetry in the CCF: the CCF tends to be small at negative lags
#    (k < 0) and large at least at the positive lag k = h. As a result, DFP II
#    generally induces a right-skew of the CCF, consistent with the behavior
#    observed under perturbation in Tutorial 14.

# 6. In contrast to the perturbation approach of Tutorial 14, the right-skew
#    of the CCF produced by DFP II is free from extraneous and potentially
#    arbitrary or misspecified modifications to the DGP. DFP II operates
#    entirely on the proper DGP, without any external modifications.

# 7. The right-skewness of the CCF can shift the peak of the CCF toward the
#    forecast horizon h, when feasible (see the MA(9) and HP Examples 5 and 6 
#    below). Right-skewing the CCF via DFP II therefore provides a very general
#    look-ahead approach, effective even in cases where the DGP conflicts with
#    the original DFP or PCS constraints.


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISES OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────


# Exercise 1 — AR(1): Look-Ahead via Perturbation
#              Replicates Exercise 3 of Tutorial 14 to establish a baseline for
#              right-skewness and look-ahead behaviour under coefficient
#              perturbation of the MSE predictor. Serves as a reference point
#              for the DFP II results developed in subsequent exercises.

# Exercise 2 — Role of the Integrator Sigma
#              Using the same AR(1) DGP, illustrates how the choice of the
#              integrator matrix Sigma shapes both the filter coefficient profile
#              and the degree of right-skewness induced in the CCF. Highlights
#              the sensitivity of DFP II to the lag window over which decoupling
#              is enforced.

# Exercise 3 — Replication of the Forward-Looking Identity Filter (MA Form)
#
#              Background:
#              The h-step-ahead MSE predictor for an AR(1) process with 
#              coefficient a1 has the MA representation
#
#                 MSE(h) = a1^h * sum_{k >= 0} a1^k * epsilon_{t-k},
#
#              which weights all past innovations geometrically, with the most 
#              recent innovation epsilon_t receiving the largest weight a1^h and 
#              earlier innovations receiving progressively smaller weights 
#              a1^{h+1}, a1^{h+2}, …
#
#              The identity filter (in MA form) retains only the most recent 
#              innovation:
#
#                b_identity = (1, 0, 0, …, 0)',
#
#              effectively discarding all earlier innovations epsilon_{t-1}, 
#              epsilon_{t-2}, … that enter the MSE predictor. By concentrating 
#              the entire filter weight on epsilon_t, the identity filter 
#              looks ahead: it omits past epsilon_{t-k}. The cost of this 
#              look-ahead is increased noise, since the smoothing provided by 
#              the geometric accumulation of past innovations is abandoned 
#              entirely.
#
#              We demonstrate that DFP II can replicate the identity filter's 
#              look-ahead forecast rule through a suitably constructed 
#              integrator matrix Sigma. As a consequence classic and/or simple 
#              benchmarks (MSE, identity) are embedded in DFP II. 
#
#              We also derive the AR-form representations of the predictors by 
#              inverting the MA form typically used in the context of DFP or PCS. 
#              The MA form highlights the inner structure and interpretation, 
#              while the AR form is typically used in implementation. Both forms 
#              are strictly equivalent in terms of the produced predictor 
#              (assuming invertibility).

# Exercise 4 — DFP II Design for Difficult Forecast Problems
#              Introduces a constraint specification that performs reliably in
#              settings where the standard DFP or PCS approaches are difficult
#              to apply or infeasible, providing a practical fallback strategy
#              for challenging forecast environments. In particular the resulting 
#              predictor inherits look ahead behaviour while retaining smoothness, 
#              in contrast to the identity benchmark of exercise 3. 

# Exercise 5 — DFP II Applied to a Feasible Forecast Problem: MA(9) Process
#              In contrast to Exercises 1–4, which focus on the AR(1) process
#              (a degenerate case for PCS and DFP), Exercises 5 and 6 consider
#              settings in which look-ahead behaviour is genuinely feasible and
#              the standard DFP and PCS frameworks are directly applicable.
#              Exercise 5 revisits the MA(9) process studied in earlier
#              tutorials and demonstrates that DFP II performs competitively
#              in this well-conditioned setting, recovering the expected
#              CCF profile and look-ahead behaviour without difficulty.

# Exercise 6 — DFP II Applied to the Hodrick-Prescott Trend Filter
#              Extends Exercise 5 to the HP trend filter setting, showing that
#              DFP II generalises naturally to this practically important case.
#              Specifically, instead of a straightforward replication of `optimal' 
#              DFP (or PCS) we propose a slightly misspecified design to 
#              highlight some of the specific features of the DFP II approach.


# ═════════════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────────────

#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ═════════════════════════════════════════════════════════════════════════════


# ── INITIALISATION ────────────────────────────────────────────────────────────

rm(list = ls())

library(mFilter)


# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

# Load HP utilities
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))


# ==============================================================================
# EXERCISE 1: Right-Skewing of the CCF Through Perturbation
# ==============================================================================

# We revisit Exercise 3 of Tutorial 14, applied to the AR(1) case.
# The CCF of the AR(1) DGP cannot be manipulated at positive lags (except for
# scaling or sign change): the CCF follows a fixed exponential profile
#
#   CCF(k) = a1^k * CCF(0)
#
# Introducing a perturbation to the DGP cannot alter this profile. Instead,
# the perturbation can be used to right-skew the CCF by reducing it at negative
# lags. This right-skewness generates look-ahead behavior of the predictor,
# as illustrated in the CCF and series plots in Exercise 1.6 below.


#───────────────────────────────────────────────────────────────────────────────
# 1.1 AR(1) DGP
#───────────────────────────────────────────────────────────────────────────────

L  <- 50    # Filter length: number of MA coefficients retained.
a1 <- 0.9   # AR(1) parameter.

# Compute the Wold (MA-infinity) coefficients of the AR(1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = 0, lag.max = 1000))

# Visualize the Wold decomposition: coefficients decay geometrically at rate a1.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: geometrically decaying impulse response")

# Visualize the theoretical ACF: also decays geometrically at rate a1.
ts.plot(ARMAacf(ar = a1, lag.max = L),
        main = "ACF", ylab = "", xlab = "Lag")

# The ACF satisfies ACF(k+1) = a1 * ACF(k) for all k >= 0.
# Consequently, the constraint system has rank one: gamma_h is proportional
# to gamma_{h+k} for any h, k >= 0 (self-similarity property). This linear 
# dependence conflicts with the original DFP (decouple from PRESENT) assumptions.


#───────────────────────────────────────────────────────────────────────────────
# 1.2 PCS Setup
#───────────────────────────────────────────────────────────────────────────────

# Forecast horizon.
h <- 12

# Target: the original AR(1) Wold decomposition.
gamma_pcs <- xi

# Nowcast: MA coefficients at lag 0.
gamma0 <- xi[1:L]

# MSE forecast: MA coefficients shifted to forecast horizon h.
gammah <- xi[h + 1:L]

# Identity predictor: faster than gammah but noisier.
# Used as an additional benchmark.
gamma_I <- c(1, rep(0, L - 1))

# Constrained lag set:
# Type I PCS imposes a non-negative slope at every lag in Delta, enforcing a
# monotonically increasing CCF (when beta > 0 and the problem is feasible) over
# the full interval {0, ..., h}. This is the most restrictive of the three PCS
# types (I, II, and III).
Delta <- 1:h

# Regularization weight (penalty on constraint deviation): strong regularization.
lambda <- 5000000


#───────────────────────────────────────────────────────────────────────────────
# 1.3 AR(1) Perturbation
#───────────────────────────────────────────────────────────────────────────────

# Construct the MSE predictor coefficient vectors gamma_i used to form the
# PCS constraint differences delta_i = gamma_i - gamma_{i-1}.
gamma_all <- xi

# Build the shifted predictor matrix gammah_mat.
# Each row contains the normalized gamma_all = xi coefficients shifted by a
# specific lead drawn from Delta. We begin at Delta[1] - 1 because the first
# constraint difference requires gamma_{Delta[1] - 1}.
gammah_mat <- gamma_all[Delta[1] - 1 + 1:L] / sqrt(sum(gamma_all^2))
if (length(Delta) > 0) {
  for (i in 1:length(Delta)) {
    gammah_mat <- rbind(gammah_mat,
                        gamma_all[Delta[i] + 1:L] / sqrt(sum(gamma_all^2)))
  }
}


# Perturb the AR(1) parameter: a1_perturbate = a1 + delta.
# The resulting Wold decomposition xi_a1_perturbate differs from xi at every
# lag except zero, providing a full-lag perturbation direction.
delta             <- 0.001
a1_perturbate     <- a1 + delta
xi_a1_perturbate  <- c(1, ARMAtoMA(ar = a1_perturbate, ma = 0, lag.max = 1000))
gamma_all_a1_perturbate <- xi_a1_perturbate

# Visualize the difference between the original and perturbed Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi - xi_a1_perturbate,
        main = "Difference: original vs. perturbed Wold coefficients")

# Construct the perturbed constraint matrix by replacing the first row of
# gammah_mat (corresponding to gamma_{Delta[1] - 1}) with its perturbed
# counterpart.
gammah_mat_perturbate      <- gammah_mat
gammah_mat_perturbate[1, ] <-
  (gammah_mat[1, ] + delta * gamma_all_a1_perturbate[1:L]) /
  sqrt(sum(gamma_all_a1_perturbate^2))


#───────────────────────────────────────────────────────────────────────────────
# 1.4 Run PCS
#───────────────────────────────────────────────────────────────────────────────

# PCS_perturbation_func() automatically returns a grid of beta values centered
# on the tipping point of beta, where the sensitivity of the PCS solution with
# respect to beta is highest.
#
# Note: the automatic grid generated by PCS_perturbation_func() is independent
# of the supplied beta value; any value can be passed as a starting input.
beta <- 0.

PCS_obj <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                 gammah_mat_perturbate)

# Extract the automatic grid of beta values.
beta_vec_automatic <- PCS_obj$beta_vec

# Add intermediate values for finer resolution around the tipping point.
beta_vec <- c(beta_vec_automatic[1:10],
              1.86e-07, 1.88e-07, 1.90e-07,
              beta_vec_automatic[11:length(beta_vec_automatic)])

M <- PCS_obj$M
V <- eigen(M)$vectors

# Note on asymptotic behavior:
# As |beta| increases, the PCS predictor b aligns with one of the following
# directions, depending on the interplay between beta, lambda, and delta
# (see Exercise Tutorial 14, 5.3 VIII and Exercise 2):
#   - V1 alone,
#   - a mixture of V1 and V2, or
#   - V2 alone.
# Here, with very large lambda, the asymptotes correspond to +/- V2
# (see Tutorial 14, Exercise  5.3 VIII, case [c]).


# Compute PCS solutions across the full beta grid (Type I constraint).
Delta  <- 1:h
b_mat  <- NULL

for (i in 1:length(beta_vec)) {
  beta    <- beta_vec[i]
  PCS_obj <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                   gammah_mat_perturbate)
  b     <- PCS_obj$b
  b_mat <- cbind(b_mat, b)
}

# Prepend the classical MSE predictor (gamma_0) as a reference benchmark.
filter_mat <- cbind(gamma0, b_mat)

# Column names scale beta by lambda for readability in plots.
colnames(filter_mat) <- c("MSE",
                          paste("lambda =", round(lambda, 2),
                                ", beta*lambda =",
                                round(beta_vec * lambda, 8)))


#───────────────────────────────────────────────────────────────────────────────
# 1.5 Plots
#───────────────────────────────────────────────────────────────────────────────

# See comments to this plot in Exercise 3.4, Tutorial 14.
colo <- plot_func()


#───────────────────────────────────────────────────────────────────────────────
# 1.6 Apply and Compare Predictors
#───────────────────────────────────────────────────────────────────────────────

# Simulate a long realization of the AR(1) DGP for filter evaluation.
len    <- 10000
set.seed(534)
x_filt <- rnorm(len)

# Apply each predictor filter to the simulated series.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


# -- Full-range overview: all predictor outputs --------------------------------
# Display a broad sub-sample to compare the behavior of all predictors.
#
# Observations:
#   - Smaller beta values produce lagging predictors relative to the MSE.
#   - Larger beta values produce increasingly leading predictors.
#   - As the degree of lead increases, predictors tend toward sign inversion,
#     reflecting phase inversion.
#   - Leading predictors and the MSE benchmark are selected for closer
#     inspection.
#   - All series are standardized to simplify visual comparison.
select_pcs <- 11:ncol(y_out_mat)
select_vec <- c(1, select_pcs)

# Longer sub-sample for a broad overview.
anf <- 100
enf <- 500

mplot <- scale(y_out_mat[anf:enf, select_vec])
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- c("black", colo[select_pcs])

par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty = c(2, 1, rep(1, ncol(mplot) - 1)),
        lwd = c(2, rep(1, ncol(mplot) - 1)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# -- Narrow sub-sample: magnifying the look-ahead effect -----------------------
# Zoom into a shorter window to highlight the look-ahead behavior of selected
# predictors, excluding those with pronounced sign inversion.
#
# Note: the look-ahead effect operates primarily on longer swings in the series.
# Short-term random spikes are inherently unpredictable. This long-swing
# look-ahead property may be particularly relevant in business cycle analysis,
# where economically significant episodes such as recessions are typically
# characterized by sustained negative swings rather than isolated shocks.
anf <- 280
enf <- 400

mplot <- scale(y_out_mat[anf:enf, select_vec])
colnames(mplot) <- colnames(y_out_mat)[select_vec]
coli <- c("black", colo[select_pcs])

par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor Outputs (Narrow Window)", col = coli,
        xlab = "", ylab = "",
        lty = c(2, 1, rep(1, ncol(mplot) - 1)),
        lwd = c(2, rep(1, ncol(mplot) - 1)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── Empirical CCF:  ───────────────────────────────────────────────────────────

mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[5])
ccf(mplot_ccf[,1],mplot_ccf[,6],main=colnames(mplot_ccf)[6])
ccf(mplot_ccf[,1],mplot_ccf[,7],main=colnames(mplot_ccf)[7])




# ==============================================================================
#  MAIN TAKE-AWAYS 
# ==============================================================================
#
# 1. As the slope parameter beta (or beta * lambda) increases in the perturbed 
#    PCS constraint, the CCF becomes increasingly right-skewed.
#
# 2. This right-skewness can be achieved more directly by explicitly reducing
#    the CCF at negative lags while simultaneously maximizing it at the
#    forecast horizon k = h > 0.
#
# 3. This is precisely the purpose and intention of DFP II: inducing look-ahead
#    behavior in the predictor by directly targeting and right-skewing the CCF,
#    without relying on perturbations or indirect modifications of the DGP.






# ==============================================================================
# EXERCISE 2: DFP II — Right-Skewing the CCF: The Role of the Integrator Sigma
# ==============================================================================

# ------------------------------------------------------------------------------
# Note: Exercise 1 must be run before this exercise, as it initializes the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ------------------------------------------------------------------------------

# Filter length and forecast horizon.
L <- 50
h <- 12


# Remarks: 
# 1. The optimization criterion of DFP II relies on a "decouple from past" 
#    constraint. The resulting problem can be solved within the original DFP 
#    optimization framework.
# 2. However, in contrast to the standard DFP, which decouples from gamma_0, the 
#    DFP II problem decouples from Sigma %*% gamma_0, where Sigma is an integration
#    operator over a specified range of past lags. 
# 3. When Sigma = Id is the identity, DFP II replicates the original DFP.


#───────────────────────────────────────────────────────────────────────────────
# 2.1 Set Up DFP II
#───────────────────────────────────────────────────────────────────────────────

# Lag support for the integration operator Sigma:
# The CCF is targeted over the left tail from lag k = -l_start to k = -l_end.
# The DFP II predictor pulls the CCF down over this range, producing a
# right-skewed CCF.
#
# Note: the values of l_start and l_end chosen here are selected to illustrate
# and clarify the role of the integrator. They are not necessarily the most
# practically relevant choices.

l_start <- 4
l_end   <- 8

if (l_start >= l_end) {
  print("l_start must be smaller than l_end")
  l_start <- l_end + 1
}

# Construct the integration operator Sigma of order (l_end - l_start).
# Sigma is an L x L matrix that accumulates filter coefficients over the
# specified lag range.
Sigma <- matrix(rep(0, L^2), ncol = L)
for (i in (l_start + 1):l_end)
  Sigma[i, (1 + l_start):i] <- 1
if (l_end < L) {
  for (i in (l_end + 1):L)
    Sigma[i, ((i - l_end + l_start) + 1):i] <- 1
}

# -- Structure of the Integration Operator Sigma -------------------------------
#
# Sigma is an L x L matrix with the following structure:
#
# 1. Integration starts at row (and column) l_start + 1: all rows up to and
#    including row l_start contain only zeros.
#
# 2. From row l_start + 1 onward, the number of ones in each row increases
#    by one per row, so that the integration accumulates progressively more
#    past values.
#
# 3. The length of each sequence of ones is capped at l_end - l_start: no
#    row contains more than l_end - l_start consecutive ones.
#
# 4. Once the maximum sequence length l_end - l_start is reached, the
#    sequences of ones are shifted one position to the right with each
#    subsequent row, acting as a sliding window of fixed width over the
#    past lag range.
Sigma[1:14,1:14]

# Compute the constraint vector gamma_constraint = Sigma %*% gamma_0.
# This vector encodes the aggregated past structure that DFP II targets.
gamma_constraint <- (Sigma %*% c(rep(0, l_start), gamma0[1:(L - l_start)]))

par(mfrow = c(1, 1))
ts.plot(gamma_constraint,
        main = "Constraint vector: Sigma %*% gamma_0",
        xlab = "Lag", ylab = "",
        sub  = "Algebraic constraint vector for controlling the right-skewness of the CCF")
abline(h = 0)

# Baseline coupling: inner product of the MSE predictor gammah with the
# constraint vector gamma_constraint. This serves as a natural reference level,
# since it quantifies how strongly the unconstrained MSE predictor is already
# coupled to the past structure targeted by DFP II.
# Any effective decoupling should enforce a coupling strictly below this value.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Define a grid of decoupling levels alpha0, each strictly smaller than mse_coup.
# Smaller values enforce progressively stronger right-skewing
# of the CCF, corresponding to greater decoupling from the past.
#
# Note: since the predictors are not normalized (||b|| != 1), the rule
# alpha0 < mse_coup does not guarantee stronger decoupling in a strict
# correlation sense, but serves as a useful practical guide.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.5, -1, -2, -3, -4)

# Display alpha0_vec.
# Small alpha_0 indicate that DFP II enforces stronger decoupling from
# the past than the unconstrained MSE predictor, which is expected to right-skew 
# the CCF.
alpha0_vec


#───────────────────────────────────────────────────────────────────────────────
# 2.2 DFP II Decoupling over the alpha0 Grid
#───────────────────────────────────────────────────────────────────────────────

# For each alpha0 in alpha0_vec, compute the DFP II predictor via the
# closed-form solution from Proposition 1 (Wildi 2026):
#
#   b = gammah + lambda * gamma_constraint
#
# where:
#
#   lambda = (alpha0 - gamma_constraint' %*% gammah) /
#            (gamma_constraint' %*% gamma_constraint)
#
# This solution minimizes the MSE subject to the decoupling constraint
# b' %*% gamma_constraint = alpha0.

# Note however that gamma_constraint = Sigma %*% gamma_0 differs notably from 
# gamma_0 of the original DFP, except when Sigma = Id. 

b_mat       <- NULL    # Filter coefficients, one column per alpha0.
cor_vec_mat <- NULL    # Full CCF vectors, one column per alpha0.

# CCF values at lags 0 and h for each alpha0, collected for tabular summary.
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of lags on either side of lag 0 to include in the CCF display.
# This parameter affects visualization only, not the optimization.
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Compute the DFP II predictor for the current decoupling level.
  b      <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  b_mat  <- cbind(b_mat, b)
  
  # Compute the population CCF of b against the process over lags [-max_lag, h].
  cor_vec         <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), xi)$cor_vec
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]        # CCF at lag 0: coupling with present.
    cor_vec_1[i, 2] <- cor_vec[1 + h]    # CCF at lag h: coupling with target.
}

colnames(b_mat)       <- colnames(cor_vec_mat) <-
paste0("alpha0 = ", round(alpha0_vec, 3))
colnames(cor_vec_1)   <- c("Lag 0", paste0("h = ", h))
rownames(cor_vec_1)   <- paste0("alpha0 = ", round(alpha0_vec, 3))


#───────────────────────────────────────────────────────────────────────────────
# 2.3 Routine Checks
#───────────────────────────────────────────────────────────────────────────────

# Check 1: Decoupling constraint satisfaction.
# The constraint b' %*% gamma_constraint = alpha0 should hold exactly for every
# column of b_mat. Residuals should be numerically zero.
t(b_mat) %*% gamma_constraint - alpha0_vec

# Check 2: Sign and orientation preservation.
# A strictly positive sum of filter coefficients preserves the direction of a 
# trend or the sign of a level shift in the data. Increasingly negative alpha_0 
# (stronger decoupling) conflict with this condition. Trend or level inversion 
# is undesirable and requires adjustments in filter/predictor outputs (e.g., apply 
# predictors to zero-centered data).
apply(b_mat, 2, sum)

# Check 3: Positive target covariance.
# The inner product b' %*% gammah should be positive for all designs, confirming
# that each predictor remains positively correlated with the forecast target.
# A negative target correlation indicates excessive sign inversion and an unusable 
# predictor.
t(b_mat) %*% gammah

# Collect all filters (nowcast, MSE, identity, and DFP II variants) into a
# single matrix for joint visualization and comparison.
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          "Identity",
                          paste0("DFP II ", round(alpha0_vec, 2)))


#───────────────────────────────────────────────────────────────────────────────
# 2.4 Plots and Performance Summary
#───────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# -- Left panel: filter coefficients ------------------------------------------
mplot <- filter_mat

plot(mplot[, 1],
     main   = "Filter coefficients: MSE and DFP II variants",
     axes   = FALSE, type = "l",
     xlab   = "Lag", ylab = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(0, min(mplot)), 0.4))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)    # Lag 0.
abline(v = h + 1, lty = 2)    # Lag h.
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# -- Right panel: population CCFs ---------------------------------------------
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag <- l_end
ccf_mat <- NULL

for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)

rownames(ccf_mat) <- -max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat) <- colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main   = "Population CCFs: MSE and DFP II variants",
     axes   = FALSE, type = "l",
     xlab   = "Lag", ylab = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)    # Lag h.
abline(v = max_lag + 1,     lty = 1)    # Lag 0.
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()


# -- Discussion of Results -----------------------------------------------------
#
# Left panel (filter coefficients):
#
# 1. For lags k <= l_start, the predictor coefficients are unaffected and
#    follow the original AR(1) decay pattern (all predictors overlap).
#
# 2. Decreasing alpha0 accelerates the decay of predictor weights for
#    k > l_start, which turn negative for sufficiently small alpha0.
#
# 3. l_start controls the onset of the accelerated decay: coefficients begin
#    to deviate from the AR(1) pattern at lag l_start.
#
# 4. l_end controls the steepness of the decay:
#    - When l_start = 0 and l_end = 1, the constraint vector replicates the
#      AR(1) structure and the system is infeasible, since the constraint and
#      target vectors are collinear.
#    - When l_start = 0 and l_end = 2, the constraint imposes a single
#      discontinuous decay step. This case includes the identity filter
#      (dark green line) as a special case, see Exercise 3 below.
#    - For increasing l_end, the decay of predictor weights operates in the 
#      interval [l_start, l_end]: the decay is longer, more gradual and smoother.


# -- Cross-check of the decoupling constraint ----------------------------------
#
# Step 1: Extract CCF values over the decoupling range l_start to l_end.
#         Note that the CCF matrix starts at max_lag = l_end (first entry
#         corresponds to lag -l_end).
ccf_decouple <- matrix(ccf_mat[1 + 1:(l_end - l_start), ],
                       nrow = (l_end - l_start))
colnames(ccf_decouple) <- colnames(ccf_mat)

# Step 2: Retain only DFP II designs, removing the nowcast and benchmarks.
ccf_decouple_check <- ccf_decouple[,
                                   which("DFP" == substr(colnames(ccf_decouple), 1, 3))[1]:ncol(ccf_decouple),
                                   drop = FALSE]

# Step 3: Sum the CCF over the decoupling range column-wise and compare with
#         alpha0 scaled by the norms of b and xi. Differences should vanish.
#         Note: xi (not gamma0) is used when computing the CCF, so the norm
#         of xi is used for scaling.
#         Note: the case alpha0 = 0 produces a 0/0 singularity and can be
#         ignored.
apply(ccf_decouple_check, 2, sum) -
  (alpha0_vec / (sqrt(apply(b_mat^2, 2, sum)) * sqrt(sum(xi^2))))







# ==============================================================================
# EXERCISE 3: DFP II and the Identity as a Special Look-Ahead Case
# ==============================================================================

# ------------------------------------------------------------------------------
# Note: Exercise 1 must be run before this exercise, as it initializes the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ------------------------------------------------------------------------------

# Filter length and forecast horizon.
L <- 50
h <- 12

# Here we set l_start = 1 and l_end = 2. This choice has the following
# implications:
#
# 1. The resulting Sigma reduces to a simple one-period backshift operator,
#    so the constraint vector gamma_constraint equals the lagged AR(1) process:
#
#      x_{t-1} = epsilon_{t-1} + a1 * epsilon_{t-2} + a1^2 * epsilon_{t-3} + ...
#
# 2. Full decoupling from this lagged process is achieved by the identity
#    filter b = (1, 0, 0, ...), which assigns all weight to epsilon_t and is
#    therefore orthogonal to the lagged x_{t-1}. The identity filter is thus a
#    special case of DFP II (up to MSE-optimal scaling).
#
# 3. Assigning full weight to epsilon_t corresponds to a left-shift (advancement)
#    relative to the MSE predictor, producing look-ahead behavior at the cost
#    of increased noise.


#───────────────────────────────────────────────────────────────────────────────
# 3.1 Set Up DFP II
#───────────────────────────────────────────────────────────────────────────────

# Lag support for the integration operator Sigma:
# The CCF is targeted over the left tail from lag k = -l_start to k = -l_end.
# The DFP II predictor pulls the CCF down over this range, producing a
# right-skewed CCF.
l_start <- 1
l_end   <- 2

if (l_start > l_end) {
  print("l_start must be smaller than or equal to l_end")
  l_start <- l_end
}

# Construct the integration operator Sigma of order (l_end - l_start).
Sigma <- matrix(rep(0, L^2), ncol = L)
for (i in (l_start + 1):l_end)
  Sigma[i, (1 + l_start):i] <- 1
if (l_end < L) {
  for (i in (l_end + 1):L)
    Sigma[i, ((i - l_end + l_start) + 1):i] <- 1
}

# Single back-shift operator (identity except first zero on diagonal) 
Sigma[1:6,1:6]

# Compute the constraint vector gamma_constraint = Sigma %*% gamma_0.
# With l_start = 1 and l_end = 2, this equals the one-period lagged DGP.
gamma_constraint <- (Sigma %*% c(rep(0, l_start), gamma0[1:(L - l_start)]))

par(mfrow = c(1, 1))
ts.plot(gamma_constraint,
        main = "Constraint vector: Sigma %*% gamma_0  (lagged DGP)",
        xlab = "Lag", ylab = "",
        sub  = "Algebraic constraint vector for controlling the right-skewness of the CCF")
abline(h = 0)

# Baseline coupling: see exercise 2
mse_coup <- as.double(gammah %*% gamma_constraint)

# Define a grid of decoupling levels alpha0:
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.1, -0.2, -0.3, -0.5)

# Display alpha0_vec.
alpha0_vec


#───────────────────────────────────────────────────────────────────────────────
# 3.2 Decoupling over the alpha0 Grid
#───────────────────────────────────────────────────────────────────────────────

b_mat       <- NULL    # Filter coefficients, one column per alpha0.
cor_vec_mat <- NULL    # Full CCF vectors, one column per alpha0.

# CCF values at lags 0 and h for each alpha0, collected for tabular summary.
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of lags on either side of lag 0 to include in the CCF display.
# This parameter affects visualization only, not the optimization.
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Compute the DFP II predictor for the current decoupling level.
  b     <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  b_mat <- cbind(b_mat, b)
  
  # Compute the population CCF of b against the process over lags [-max_lag, h].
  cor_vec         <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), xi)$cor_vec
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]        # CCF at lag 0: coupling with present.
    cor_vec_1[i, 2] <- cor_vec[1 + h]    # CCF at lag h: coupling with target.
}

colnames(b_mat)     <- colnames(cor_vec_mat) <-
  paste0("alpha0 = ", round(alpha0_vec, 3))
colnames(cor_vec_1) <- c("Lag 0", paste0("h = ", h))
rownames(cor_vec_1) <- paste0("alpha0 = ", round(alpha0_vec, 3))


#───────────────────────────────────────────────────────────────────────────────
# 3.3 Routine Checks
#───────────────────────────────────────────────────────────────────────────────

# Check 1: Decoupling constraint satisfaction.
t(b_mat) %*% gamma_constraint - alpha0_vec

# Check 2: Sign and orientation preservation.
apply(b_mat, 2, sum)

# Check 3: Positive target covariance.
t(b_mat) %*% gammah

# Collect all filters (nowcast, MSE, identity, and DFP II variants) into a
# single matrix for joint visualization and comparison.
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          "Identity",
                          paste0("DFP II ", round(alpha0_vec, 2)))


#───────────────────────────────────────────────────────────────────────────────
# 3.4 Plots and Performance Summary
#───────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# -- Left panel: filter coefficients ------------------------------------------
mplot <- filter_mat

plot(mplot[, 1],
     main   = "Filter coefficients: MSE and DFP II variants",
     axes   = FALSE, type = "l",
     xlab   = "Lag", ylab = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(0, min(mplot)), 0.4))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)    # Lag 0.
abline(v = h + 1, lty = 2)    # Lag h.
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# -- Right panel: population CCFs ---------------------------------------------
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag <- l_end
ccf_mat <- NULL

for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)

rownames(ccf_mat) <- -max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat) <- colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main   = "Population CCFs: MSE and DFP II variants",
     axes   = FALSE, type = "l",
     xlab   = "Lag", ylab = "",
     col    = colo[1], lwd = 1,
     ylim   = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)    # Lag h.
abline(v = max_lag + 1,     lty = 1)    # Lag 0.
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()



# -- Discussion of Results -----------------------------------------------------
#
# Left panel (filter coefficients):
#
# 1. The lag 0 weight is fixed and identical to the MSE predictor, since
#    l_start = 1 > 0: the constraint does not affect the contemporaneous weight.
#
# 2. Decreasing alpha0 progressively downweights the AR(1) profile at lags
#    k >= 1, with weights turning negative for sufficiently small alpha0.
#
# 3. At alpha0 = 0 (cyan), the DFP II predictor coincides with the identity
#    filter (up to scaling), confirming that the identity is a special case of 
#    DFP II.
#
# Right panel (CCFs):
#
# 1. All predictors except the last retain a positive target correlation at
#    lag k = h, confirming effective look-ahead behavior without sign inversion.
#
# 2. Decoupling is enforced at lag k = -1: the CCF at this lag is progressively
#    reduced and can turn negative, while the target correlation at k = h
#    remains in positive territory.


#───────────────────────────────────────────────────────────────────────────────
# 3.5 Applying the Filters to Simulated Data
#───────────────────────────────────────────────────────────────────────────────

# -- 3.5.1 Forecast Comparison ------------------------------------------------

# Generate a long white-noise series for reliable empirical evaluation.
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes: sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

scale(y_out_mat)[, 1:2]

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each
# predictor relative to the MSE benchmark.
anf <- 500
enf <- 600

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardized): excerpt",
        col  = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
# As alpha0 decreases, the predictors become increasingly noisy and
# mean-reverting: they are unable to track episodic swings away from the
# center line (no or reduced autocorrelation).


# -- 3.5.2 Empirical CCF Comparison -------------------------------------------
# Compute empirical CCFs between the nowcast (column 1, representing x_t) and
# each selected predictor. The CCF is progressively right-skewed.

par(mfrow = c(3, 2))
select_vec <- c(2, 4, 6, 8, 10, 11)

for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main    = colnames(y_out_mat)[i])
}

# Compute target correlations and smoothness for each predictor.
targeth <- c(y_out_mat[(1 + h):len, 1], rep(NA, h))

# Remove NAs before computing summary statistics.
y_outh <- na.exclude(cbind(targeth, y_out_mat))
y_out  <- y_outh[, 2:ncol(y_outh)]
colnames(y_out) <- colnames(y_out_mat)
target <- y_outh[, 1]

# Target correlation: correlation of each predictor with the h-step-ahead target.
target_cor <- NULL
for (i in 1:ncol(y_out_mat))
  target_cor <- c(target_cor, cor(target, y_out[, i]))
names(target_cor) <- colnames(y_out_mat)
target_cor

# Smoothness: measured as the root mean squared second difference of each
# standardized predictor. Larger values indicate noisier predictors.
smooth <- NULL
for (i in 1:ncol(y_out_mat))
  smooth <- c(smooth, sqrt(mean(diff(diff(scale(y_out[, i])))^2)))
names(smooth) <- colnames(y_out_mat)

# Outcome: predictors become increasingly noisy as alpha0 decreases.
smooth


# -- Measuring the Lead of DFP II Predictors ----------------------------------
#
# A natural question is whether and by how much the DFP II predictor leads the
# MSE benchmark in time. The plots above suggest a lead, but how can we quantify 
# this lead? Two complementary approaches are considered:
#
# 1. Zero-crossing lead (Tau statistic):
#    For the AR(1) process, the CCF peak is structurally fixed at lag k = 0
#    and cannot be shifted by any linear predictor. The classic peak correlation 
#    principle (see Tutorial 2) is therefore uninformative about the temporal 
#    lead of DFP II over the MSE predictor.
#    As an alternative, we assess whether the DFP II predictor anticipates
#    zero-crossings (mean-level crossings) of the MSE predictor. An earlier
#    zero-crossing of DFP II relative to MSE indicates look-ahead behavior.
#    See Wildi (2024) and Tutorial 2 for background on the Tau statistic.
#
# 2. Frequency-domain time-shift:
#    The phase of the DFP II filter, relative to the MSE filter, quantifies
#    the lead or lag in time periods as a function of frequency. A negative
#    time-shift at a given frequency confirms that DFP II leads the MSE
#    predictor at that frequency.
#    Note: the simulated AR(1) process does not exhibit cyclical behavior, so
#    the time-shift statistic is not directly interpretable in terms of
#    business-cycle dynamics here. However, if the same predictor filters were
#    applied to macroeconomic data without further modification, the time-shift
#    function would measure an empirically relevant and practically meaningful
#    characteristic of the predictors at business-cycle frequencies.



#───────────────────────────────────────────────────────────────────────────────
# 3.6 Lead at Zero-Crossings: Tau-Statistic 
#───────────────────────────────────────────────────────────────────────────────

# Left-shift of trough (zero-crossing alignment, see Wildi 2024):
#
# The curve below shows the average distance between zero-crossings of the
# DFP II predictor and the MSE(h) benchmark: this is the lead as measured at the 
# mean-level (the zero-crossings). The trough of this curve identifies the lag  
# at which the two series are most closely aligned in time, see Wildi (2024).
# A trough at lag -k indicates that DFP II leads MSE(h) by k time units.
# Increasing decoupling (smaller alpha0) generally shifts the trough to the
# left, reflecting a larger lead, up to a point: for alpha0 below approximately
# 0.18, the lead signal is lost in noise.

max_lead                       <- 14
vicinity                       <- 4
last_crossing_or_closest_crossing <- FALSE
outlier_limit                  <- 40

par(mfrow = c(3, 2))
plot_vec <- NULL

for (i in select_vec) {
  xy_mat <- cbind(y_out[, i],
                  y_out[, paste0("MSE(", h, ")")])
  colnames(xy_mat) <- c(colnames(y_out)[i], paste0("MSE(", h, ")"))
  tau <- compute_min_tau_func(xy_mat, max_lead, vicinity,
                              last_crossing_or_closest_crossing, outlier_limit)
  tau$min_tau_plot
}


# Illustrative example: known phase shift between two cosine waves.
# Used to verify that compute_min_tau_func correctly recovers a known lead of
# k time units. The tau-statistic is discussed in Wildi (2024).
if (F) {
  omega <- pi / 20
  x     <- cos((1:1000) * omega)
  k     <- 3
  y     <- cos((k + 1:1000) * omega)
  
  xy_mat                        <- cbind(x, y)
  max_lead                      <- 10
  vicinity                      <- 4
  last_crossing_or_closest_crossing <- FALSE
  outlier_limit                 <- max(k, 40, 2 * max_lead)
  
  par(mfrow = c(1, 1))
  tau <- compute_min_tau_func(xy_mat, max_lead, vicinity,
                              last_crossing_or_closest_crossing, outlier_limit)
}


#───────────────────────────────────────────────────────────────────────────────
# 3.7 Frequency-Domain Time-Shift 
#───────────────────────────────────────────────────────────────────────────────

# Compute the frequency-domain time-shift function for each selected predictor.
# The time-shift function quantifies the phase lead or lag (in time periods) as
# a function of frequency.

K      <- 600
plot_T <- FALSE
shift_mat <- NULL

for (i in select_vec) {
  b1        <- filter_mat[, i]
  shift_mat <- cbind(shift_mat,
                     amp_shift_func(K, b1, plot_T)$shift)
}
colnames(shift_mat) <- colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat

# Full frequency interval [0, pi].
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: full frequency interval [0, pi]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Low-frequency interval [0, pi/6].
mplot <- shift_mat[1:(K / 6), ]
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: interval [0, pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at = 1 + 0:6 * K / 36,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# Low-frequency interval [0, pi/6], trimmed to [-10, 10] for readability.
mplot <- shift_mat[1:(K / 6), ]
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: interval [0, pi/6], trimmed to [-10, 10]",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at = 1 + 0:6 * K / 36,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# Outcome: decreasing alpha0 progressively reduces the time-shift across a broad 
# frequency band, including the business-cycle frequency range. This confirms that
# stronger decoupling from the past (smaller alpha0, stronger right-skew) 
# generates a larger lead of the DFP II predictor relative to the MSE benchmark.


# -- DFP II Effect on the AR Form of the Predictor ----------------------------
#
# In practice, predictors are typically applied directly to the observed series
# rather than to the residuals of a fitted time series model. All results
# derived above concern the MA (Wold) form of the predictor, which is more
# transparent and directly reflects the underlying problem structure.
# For completeness, we also derive and examine the equivalent AR form of each
# DFP II predictor, which is the representation more commonly encountered in
# applied forecasting (note that both forms are equivalent in the sense that the 
# respective predictors are identical.


#───────────────────────────────────────────────────────────────────────────────
# 3.7 AR Form
#───────────────────────────────────────────────────────────────────────────────

# -- 3.7.1 AR Inversion -------------------------------------------------------

# Compute the AR inversion filter (the inverse of the MA polynomial).
# For the AR(1) process with parameter a1, the AR operator is simply (1, -a1).
ar_inv <- c(a1, rep(0, L - 1))

# Assemble the full AR filter including the leading coefficient of 1.
theta <- c(1, -ar_inv)

# Verify correctness: convolving the AR operator with the Wold (MA) coefficients
# should yield the identity filter (1, 0, 0, ...). Small residual deviations
# are due to finite-length truncation at L.
conv_two_filt_func(xi, theta)$conv[1:10]

# Visualize the AR inversion filter.
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR inversion filter")

# With the identity verified, we proceed to convolve the AR operator with each
# DFP II predictor (expressed in MA form) to obtain the AR-form representation.


# -- 3.7.2 Derivation of AR-Form Predictors -----------------------------------

# Verification: convolving theta with gamma_0 should recover the identity
# filter. Residual deviations from (1, 0, 0, ...) vanish as L increases.
conv_two_filt_func(theta, gamma0)$conv[1:10]

# Convolve the AR operator with each predictor to obtain AR-form coefficients.
filter_mat_ar <- NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar <- cbind(filter_mat_ar,
                         conv_two_filt_func(theta, filter_mat[, i])$conv)
colnames(filter_mat_ar) <- colnames(filter_mat)

# Verification: the first column (nowcast gamma_0) should equal the identity.
filter_mat_ar[1:10, 1]

# Plot the first (h + 3) AR coefficients for each predictor, scaled to unit
# length for comparability.
colo       <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))
first_lags <- h + 3

par(mfrow = c(1, 1))
ts.plot(
  scale(filter_mat_ar[1:first_lags, ], center = FALSE, scale = TRUE),
  col  = colo,
  main = "DFP II predictors in AR form (scaled to unit length)",
  lty  = c(2, 2, rep(1, ncol(filter_mat) - 2)),
  lwd  = c(1, 2, rep(1, ncol(filter_mat) - 2)),xlab="")
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)

# Observations:
# 1. The identity filter and the full-decoupling DFP II design (alpha0 = 0)
#    overlap after scaling, confirming that the identity is a special case
#    of DFP II in the AR(1) setting.
# 2. Increasing decoupling (smaller alpha0, greater right-skewness) assigns
#    progressively less weight to lag 0 and more negative weight to lag 1 in the 
#    AR representation (scaled to unit-length throughout).





# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4:  DFP II — Decoupling From the Past via Integrated Lag Structure
# ══════════════════════════════════════════════════════════════════════════════

# Exercises 2 and 3 considered particular Sigma integrators. Here we present
# an alternative that performs well on difficult forecast problems, such as the
# AR(1). The key idea is to extend the integration range over past lags from
# k = 0 to k = -h (i.e., up to the negative forecast horizon). As shown below,
# this integrator produces a constraint vector Sigma %*% gamma_0 that is
# reminiscent of Tutorial 12, Exercise 1.4, although the context differs.


# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


# Filter length and forecast horizon
L <- 50
h <- 12


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Set-Up: DFP II Integrator and Constraint Vector
# ─────────────────────────────────────────────────────────────────────────────

# Lag support definition: from k = 0 to k = -h
l_start <- 0
l_end   <- h

# Safety check: ensure the lag window is ordered correctly.
if (l_start > l_end) {
  print("l_start must be less than or equal to l_end")
  l_start <- l_end
}


# Construct the integrator matrix Sigma (L x L).
Sigma <- matrix(rep(0, L^2), ncol = L)
for (i in (l_start + 1):l_end)
  Sigma[i, (1 + l_start):i] <- 1

# Rows l_end+1 to L: full-length sliding window sums
if (l_end < L) {
  for (i in (l_end + 1):L)
    Sigma[i, ((i - l_end + l_start) + 1):i] <- 1
}

# Structure of the integrator:
Sigma[1:14,1:14]


# Compute the constraint vector by applying Sigma to the shifted gamma_0.
# This is the key quantity used in the DFP II optimisation constraint:
#   b %*% gamma_constraint = alpha0
gamma_constraint <- Sigma %*% c(rep(0, l_start), gamma0[1:(L - l_start)])


# Plot the constraint vector to visualise its structure.
par(mfrow = c(1, 1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag",
        ylab = "",
        sub  = "Algebraic constraint vector controlling right-skewness of the CCF")
abline(h = 0)


# ── Technical Note ────────────────────────────────────────────────────────────
# The constraint vector gamma_constraint bears a structural resemblance to
# Tutorial 12, Exercise 1.4, specifically to the convolved target gamma used
# there, though the underlying motivation differs:
#
#   Tutorial 12: An equally weighted MA(12) integrator was applied to the
#     original ARMA(1,1) process to expand the rank of the constraint system.
#     The target was the convolved (annually aggregated) DGP.
#
#   Here: The integrator Sigma acts on the CCF's left tail (negative lags)
#     to enforce right-skewness in the CCF. The target remains the original
#     (un-convolved) AR(1) process. The positive-lag CCF profile is
#     exponentially decaying and immutable for the AR(1), making direct
#     improvement at positive lags infeasible.
#
# The conceptual parallel is intriguing but the purposes are distinct.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Define the Decoupling Grid (alpha0_vec)
# ─────────────────────────────────────────────────────────────────────────────

# Baseline coupling: inner product of the unconstrained MSE predictor gammah
mse_coup <- as.double(gammah %*% gamma_constraint)

# Construct a grid of decoupling levels alpha0, each strictly below mse_coup.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.5, -1, -2, -3, -4)

# Display the grid. 
alpha0_vec


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  Optimisation: Compute DFP II Predictors Over the alpha0 Grid
# ─────────────────────────────────────────────────────────────────────────────

b_mat       <- NULL    # Filter coefficients: one column per alpha0 value
cor_vec_mat <- NULL    # Full CCF vectors:    one column per alpha0 value

# Store CCF values at lags 0 and h for each alpha0 (used in tabular summary)
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of leads/lags on either side of lag 0 to include in the CCF output.
# This parameter affects display only, not the optimisation.
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Compute the DFP II predictor for the current decoupling level alpha0.
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  
  b_mat <- cbind(b_mat, b)
  
  # Compute the population CCF of predictor b against the process
  # over lags [-max_lag, h].
  cor_vec <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), xi)$cor_vec
  
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]         # CCF at lag 0 (coupling with present)
    cor_vec_1[i, 2] <- cor_vec[1 + h]     # CCF at lag h (coupling with target)
}

# Attach descriptive column and row names for readability.
colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1) <- c("Lag 0", paste0("h=", h))
rownames(cor_vec_1) <- paste0("alpha0=", round(alpha0_vec, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: Constraint Satisfaction ─────────────────────────────────────────
t(b_mat) %*% gamma_constraint - alpha0_vec

# ── Check 2: Level/Trend Orientation Preservation ─────────────────────────────────
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
t(b_mat) %*% gammah


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Collect All Filters Into a Single Summary Matrix
# ─────────────────────────────────────────────────────────────────────────────

# Combine the nowcast filter, MSE predictor, identity filter, and all DFP II
# variants into one matrix for comparative analysis and plotting.
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c(
  "Nowcast",
  paste0("MSE(", h, ")"),
  "Identity",
  paste0("DFP II ", round(alpha0_vec, 2))
)


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# ── Left Panel: Filter Coefficients ──────────────────────────────────────────
# Display the full filter coefficient vectors for all predictors.
mplot <- filter_mat

plot(mplot[, 1],
     main  = "Filter coefficients: MSE and DFP II variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), 0.4))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right Panel: Population CCFs ─────────────────────────────────────────────
# Compute and display the population CCF for each filter.
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag <- l_end
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
rownames(ccf_mat) <- -max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat) <- colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main  = "Population CCFs: MSE and DFP II variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)   # lag h
abline(v = max_lag + 1,     lty = 1)   # lag 0
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()



# ── Interpretation of Plots ───────────────────────────────────────────────────
#
# Left panel — Filter coefficients:
#   - Decreasing alpha0 accelerates the decay of predictor weights, which
#     eventually turn negative for sufficiently small alpha0.
#   - l_start controls the onset of the accelerated decay: for lags >= l_start,
#     coefficients decay faster than under the unconstrained MSE predictor.
#   - l_end controls the steepness and extent of the decay: the larger l_end 
#     in this setting allows for more gradual and smoother decays of predictor 
#     weights.
#
# Right panel — Population CCFs:
#   - The MSE predictor maximises the CCF at forecast horizon h = 12.
#   - Right-skewing works as intended: as alpha0 decreases, the left tail of
#     the CCF is progressively pulled-down while the CCF at lag h is maintained
#     near its maximum (efficient frontier: no other linear predictor can
#     increase skewness for the same target correlation).
#   - l_end determines the lag interval over which CCF asymmetry is enforced: 
#         the CCF is pulled down on average over [l_start, l_end]
#         via the integrator Sigma, generating progressively stronger asymmetry.




# ── Cross-Check: Decoupling Constraint Verification ──────────────────────────
# Confirm numerically that the DFP II constraint is satisfied by each predictor.
#
# Step 1: Extract the CCF over the decoupling window [l_start, l_end].
#         Note: the CCF matrix starts at lag -max_lag = -l_end, so the first
#         entry corresponds to lag -l_end. Rows 2 to (l_end - l_start + 1)
#         cover lags -l_end+1 to 0, i.e., the decoupling interval.
ccf_decouple <- matrix(ccf_mat[1 + 1:(l_end - l_start), ],
                       nrow = (l_end - l_start))
colnames(ccf_decouple) <- colnames(ccf_mat)

# Step 2: Retain only DFP II designs (exclude nowcast and benchmark filters).
ccf_decouple_check <- ccf_decouple[,
      which("DFP" == substr(colnames(ccf_decouple), 1, 3))[1]:ncol(ccf_decouple),
                                   drop = FALSE]

# Step 3: Sum the CCF over the decoupling window column-wise and compare with
#         alpha0 scaled by the L2-norms of b and xi. Differences should vanish.
#         Note: xi is used in the CCF computation (not gamma0), so we scale by
#         ||xi|| rather than ||gamma0||. The case alpha0 = 0 produces a 0/0
#         singularity and should be disregarded.
apply(ccf_decouple_check, 2, sum) -
  (alpha0_vec / (sqrt(apply(b_mat^2, 2, sum)) * sqrt(sum(xi^2))))

# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────────────

# ── 4.5.1  Forecast Comparison ───────────────────────────────────────────────

# Generate a long white-noise series to support reliable empirical evaluation.
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes: y_t = sum_{k=0}^{L-1} b_k * eps_{t-k}
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each predictor.
anf <- 500
enf <- 600

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col  = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Interpretation:
#   - As alpha0 decreases, predictor output shifts progressively to the left
#     (i.e., leads the series further ahead) relative to the MSE predictor.
#     This visual lead is confirmed quantitatively by the empirical CCFs below.
#   - Strongly skewed predictors (small alpha0) generally lag the identity
#     filter, but the identity filter:
#       * Is a special case of DFP II skewing (l_start = 0, l_end = 2, see Exercise 3).
#       * Is substantially noisier; see smoothness statistics below.
#   - In contrast to Exercise 3, the DFP II predictors here simultaneously
#     achieve a temporal lead AND preserve output smoothness. This double
#     benefit — being both fast and smooth — is a direct consequence of the
#     longer Sigma integrator: by spreading the decoupling constraint over
#     a wider lag window [l_start, l_end], the integrator avoids the abrupt
#     weight attenuation that drives excess noise in shorter-window designs.



# ── 4.5.2  Empirical CCF Comparison ──────────────────────────────────────────
# Compute empirical CCFs between the nowcast output (x_t) and each predictor
# to confirm the progressive right-skewing of the CCF.

par(mfrow = c(3, 2))
select_vec <- c(2, 4, 7, 9, 11, 13)

for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main    = colnames(y_out_mat)[i])
}


# ── 4.5.3  Target Correlation and Smoothness ─────────────────────────────────
# Evaluate each predictor against the h-step-ahead target and assess output
# smoothness (roughness = RMS of second differences of standardised output).

# Construct the h-step-ahead target series (shift nowcast forward by h).
targeth <- c(y_out_mat[(1 + h):len, 1], rep(NA, h))

# Remove rows with NAs introduced by the lag shift.
y_outh  <- na.exclude(cbind(targeth, y_out_mat))
y_out   <- y_outh[, 2:ncol(y_outh)]
colnames(y_out) <- colnames(y_out_mat)
target  <- y_outh[, 1]

# Compute correlation of each predictor with the h-step-ahead target.
target_cor <- NULL
for (i in 1:ncol(y_out_mat))
  target_cor <- c(target_cor, cor(target, y_out[, i]))
names(target_cor) <- colnames(y_out_mat)
target_cor

# Compute output roughness: RMS of second differences of standardised output.
# Smaller values indicate smoother (less noisy) predictor output.
smooth <- NULL
for (i in 1:ncol(y_out_mat))
  smooth <- c(smooth,
              sqrt(mean(diff(diff(scale(y_out[, i])))^2)))
names(smooth) <- colnames(y_out_mat)
smooth

# The DFP II maintains good tracking at the forecast horizon h while being smooth 
#   and leading.




# ─────────────────────────────────────────────────────────────────────
# 4.6 Lead/Time-Shifts
# ─────────────────────────────────────────────────────────────────────



# ── 4.6.1  Lead-Lag Analysis: Zero-Crossing Alignment ────────────────────────
# Quantify the temporal lead of each DFP II predictor relative to MSE(h) by
# computing the average lag at which the zero-crossings of the two series
# are closest (trough in the mean absolute zero-crossing distance curve).
#
# Interpretation:
#   - A trough at lag -k indicates that the DFP II predictor leads MSE(h)
#     by k time units.
#   - Increasing right-skewness (smaller alpha0) produces a progressively
#     larger lead, more systematically than in Exercise 3, where predictors
#     were also (much) noisier.

max_lead <- 8
par(mfrow = c(3, 2))

for (i in select_vec) {
  xy_mat <- cbind(y_out[, i],
                  y_out[, paste0("MSE(", h, ")")])
  colnames(xy_mat) <- c(colnames(y_out)[i], paste0("MSE(", h, ")"))
  tau <- compute_min_tau_func(xy_mat, max_lead)
  tau$min_tau_plot
}


# ── 4.6.2 Time-Shifts ─────────────────────────────────────────────────────────

K<-600
plot_T<-F
shift_mat<-NULL
for (i in select_vec)
{
  
  b1<- filter_mat[,i]
  shift_mat<-cbind(shift_mat,amp_shift_func(K,b1, plot_T)$shift)   # time-shift for lagged filter (b1)
  
}
colnames(shift_mat)<-colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat
colnames(mplot) <- colnames(shift_mat)

plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: Whole frequency interval",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# From 0 to pi/6 
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "In interval [0,pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# From 0 to pi/6 and trimmed to [-10,10]
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Trimmed to -10,10",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# DFP II maintains a lead over the MSE over a broad range of frequencies, including 
# business-cycle frequencies.


# ─────────────────────────────────────────────────────────────────────
# 4.7 AR Form
# ─────────────────────────────────────────────────────────────────────

# See Exercise 3.7 for background.

# ── 4.7.1 AR Inversion ───────────────────────────────────

ar_inv <- c(a1,rep(0,L-1))
theta <- c(1, -ar_inv)


# ── 4.7.2 Derivation of AR Forms ───────────────────────────────────

# Convolve the AR operator with each predictor in filter_mat to obtain
# the AR-form coefficient vectors.
filter_mat_ar <- NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar <- cbind(filter_mat_ar,
                         conv_two_filt_func(theta, filter_mat[, i])$conv)
colnames(filter_mat_ar) <- colnames(filter_mat)

# Plot

colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- h+3
par(mfrow = c(1, 1))
# Plot the first `first_lags` AR coefficients for each predictor.
ts.plot(
  scale(filter_mat_ar[1:first_lags, ],center=F,scale=T),
  col  = colo,
  main = "DFP II Predictors in AR Form (scaled to unit-length)",
  lty  = c(2, 2, rep(1, ncol(filter_mat) - 2)),
  lwd  = c(1, 2, rep(1, ncol(filter_mat) - 2)),xlab="")
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)

# Comments:
#   1. The identity filter in MA form assigns full weight to epsilon_t alone.
#      In AR form, this corresponds to (1 - a1*B) x_t = epsilon_t, yielding
#      weights 1 and -a1 at lags 0 and 1, respectively (red line in plot).
#
#   2. The MSE predictor is the natural identity in AR form (up to a scaling
#      constant): it recovers the target signal without distortion, subject
#      only to the constraints imposed by the forecast horizon h.
#
#   3. The DFP II predictors assign increasingly large and fixed negative
#      weights to lags 1 through h-1 = 11, reflecting the progressive
#      right-skew of the CCF as alpha0 decreases.
#
#   4. In contrast to classic decoupling from the present — where the AR form
#      is modified at lag 0 only — DFP II generally redistributes weight across 
#      multiple lags (or even all lags) simultaneously. This broader
#      redistribution increases flexibility and generality and enables DFP II to 
#      better achieve the conflicting goals of a temporal lead and a smooth 
#      output, by overcoming the necessity to concentrate the 
#      decoupling cost at a single (zero-) lag (as in the original DFP).



# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5:  Applying DFP II to an MA(9) Process
# ══════════════════════════════════════════════════════════════════════════════

# This exercise revisits the DFP II framework developed in Exercise 4, but
# applies it to a truncated MA(9) process rather than an AR(1). The finite
# MA order (q = 9) means the forecast problem is no longer infeasible: the
# MSE predictor is non-trivial for h <= q, and right-skewing via a minimal
# constraint (l_start = 0, l_end = 1) is sufficient to shift the CCF peak
# toward lag h.
#
# More specifically:
#   - The forecast problem is difficult in the sense that the MSE predictor
#     remains anchored at the present (CCF peak remains at lag 0). 
#   - However, the problem is feasible and the peak of the CCF can be left-sifted 
#     — unlike the AR(1) case — as demonstrated in Tutorial 10, Exercise 1: there 
#     exist linear predictors that shift the CCF peak toward lag h.
#   - This exercise shows that DFP II performs well in feasible problems too,
#     not only in the infeasible AR(1) setting of Exercise 4.
#   - A limitation of DFP II is that right-skewing the CCF does not provide
#     explicit control over the location of the CCF peak. In contrast, the
#     PCS framework imposes constraints that are more directly tailored to
#     peak-shifting, offering finer and more interpretable control.
#   - Nevertheless, a suitable decoupling level alpha0 can always be
#     identified empirically in DFP II by scanning the alpha0 grid. Furthermore, 
#     we conjecture that, in the feasible case, a formal closed-form relationship
#     between alpha0 and the CCF peak location exists within the DFP II
#     framework — a direction left open for future investigation.


# ─────────────────────────────────────────────────────────────────────────────
# 5.1  Process Specification and Data Generation
# ─────────────────────────────────────────────────────────────────────────────

# MA order and geometric decay coefficient.
q  <- 9      # MA order: number of lags beyond lag 0
a1 <- 0.9    # Geometric decay rate (mirrors the underlying AR(1) coefficient)

# MA filter weights: b_k = a1^k,  k = 0, 1, ..., q
b_ma <- a1^(0:q)

# Wold decomposition: MA coefficients padded with zeros to a long vector.
# Used in CCF computations that require a fixed-length innovation sequence.
xi <- c(b_ma, rep(0, 1000))

# Plot the MA(9) filter weights to visualise the geometric decay structure.
par(mfrow = c(1, 1))
ts.plot(b_ma,
        main = "MA(9) filter coefficients (geometrically decaying)",
        xlab = "Lag", ylab = "Weight",ylim=c(0,1))

# Simulate one realisation of the MA(9) process:
#   x_t = sum_{k=0}^{q} b_k * epsilon_{t-k},   epsilon_t ~ i.i.d. N(0,1)
len <- 100
set.seed(231)
eps <- rnorm(len + q + 1)    # innovations including burn-in period

# Pre-allocate output vectors for the process and (later) the predictor.
x <- xhat <- rep(NA, len + q + 1)

for (i in (q + 1):(len + q + 1))
  x[i] <- b_ma %*% eps[i:(i - q)]

ts.plot(x, main = "Simulated MA(9) process")


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────────────

# Filter length and forecast horizon.
L <- 20
h <- 5

# Feasibility check: for an MA(q) process, the h-step-ahead MSE predictor is
# identically zero when h > q, because all innovations more than q steps ahead
# are unobservable from the current information set.
if (h > q)
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# Optimal MSE filter: retain MA coefficients from lag h onward; pad to length L.
#   gammah[k] = b_{h+k}  for k = 0, ..., q-h,  and 0 thereafter.
gammah  <- c(b_ma[(h + 1):(q + 1)], rep(0, L - (q - h + 1)))

# Nowcast filter: full MA coefficient vector padded to length L.
gamma0  <- c(b_ma, rep(0, L - (q + 1)))

# Identity filter: unit weight at lag 0, zero elsewhere (used as additional 
# forward looking but noisy predictor).
gamma_I <- c(1, rep(0, L - 1))

# Use gamma0 as the Wold representation for CCF computations.
xi <- gamma0


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  Set-Up: DFP II Integrator and Constraint Vector
# ─────────────────────────────────────────────────────────────────────────────

# Lag support definition:
#   The MA(9) process has rank q = 9 > 1, so the forecast problem is feasible.
#   Right-skewing requires only a minimal constraint: setting l_start <- 0 and 
#   l_end = 1 constrains the CCF at lag k = -1 only, leaving the maximum degrees 
#   of freedom available to maximise the target correlation CCF(h).
l_start <- 0
l_end   <- 1

# Safety check: the window must contain at least one lag.
if (l_start >= l_end) {
  print("l_start must be strictly less than l_end")
  l_start <- l_end + 1
}

# Construct the integrator matrix Sigma (L x L).
Sigma <- matrix(rep(0, L^2), ncol = L)

# Rows l_start+1 to l_end: partial sums (window not yet fully formed).
for (i in (l_start + 1):l_end)
  Sigma[i, (1 + l_start):i] <- 1

# Rows l_end+1 to L: full-length sliding window sums.
if (l_end < L) {
  for (i in (l_end + 1):L)
    Sigma[i, ((i - l_end + l_start) + 1):i] <- 1
}

# Inspect Sigma: a plain identity. In this case, DFP II coincides with and 
# replicates the ordinary DFP.
Sigma[1:9,1:9]

# Compute the constraint vector by applying Sigma to the shifted gamma0.
# This vector encodes the decoupling target used in the DFP II optimisation:
#   b %*% gamma_constraint = alpha0
gamma_constraint <- Sigma %*% c(rep(0, l_start), gamma0[1:(L - l_start)])

# Plot the constraint vector to inspect its structure.
par(mfrow = c(1, 1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag", ylab = "",
        sub  = "Algebraic constraint vector controlling right-skewness of the CCF")
abline(h = 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.4  Define the Decoupling Grid (alpha0_vec)
# ─────────────────────────────────────────────────────────────────────────────

# Baseline coupling: 
mse_coup <- as.double(gammah %*% gamma_constraint)

# Construct a grid of decoupling levels alpha0, each strictly below mse_coup.
alpha0_vec <- c(mse_coup / 1.5^(1:9), 0)

# Display the grid. 
alpha0_vec


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  Optimisation: Compute DFP II Predictors Over the alpha0 Grid
# ─────────────────────────────────────────────────────────────────────────────

b_mat       <- NULL    # Filter coefficients: one column per alpha0 value
cor_vec_mat <- NULL    # Full CCF vectors:    one column per alpha0 value

# Store CCF values at lags 0 and h for each alpha0 (used in tabular summary).
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of leads/lags on either side of lag 0 to include in the CCF output.
# This parameter affects display only, not the optimisation.
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Compute the DFP II predictor for the current decoupling level alpha0.
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  
  b_mat <- cbind(b_mat, b)
  
  # Compute the population CCF of predictor b against the process
  # over lags [-max_lag, h].
  cor_vec <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), xi)$cor_vec
  
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]         # CCF at lag 0 (coupling with present)
    cor_vec_1[i, 2] <- cor_vec[1 + h]     # CCF at lag h (coupling with target)
}

# Attach descriptive column and row names for readability.
colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1) <- c("Lag 0", paste0("h=", h))
rownames(cor_vec_1) <- paste0("alpha0=", round(alpha0_vec, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 5.6  Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: Constraint Satisfaction ─────────────────────────────────────────
t(b_mat) %*% gamma_constraint - alpha0_vec

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# For strongly skewed DFP II predictors (small alpha0), the sum of filter
# coefficients may become negative, indicating that the filter inverts the
# sign of a fixed mean level or a sustained trend in the data. This sign
# inversion can be detected at an early stage by inspecting the coefficient
# sum. Two practical implications follow:
#
#   - Data centring: before applying any orientation-inverting filter to real
#     data, the series must be demeaned. Failing to do so causes the fixed
#     non-zero mean to be sign-inverted in the predictor output, producing
#     a systematic bias of opposite sign.
#   - Interpretation: a negative coefficient sum does not necessarily
#     invalidate the predictor. In mean-zero settings (or after centring),
#     the filter may still produce a useful lead indicator, as confirmed
#     by the target correlation and CCF diagnostics in subsequent checks.
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
t(b_mat) %*% gammah


# Collect All Filters Into a Single Summary Matrix

# Combine the nowcast filter, MSE predictor, identity filter, and all DFP II
# variants into one matrix for comparative analysis and plotting.
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c(
  "Nowcast",
  paste0("MSE(", h, ")"),
  "Identity",
  paste0("DFP II ", round(alpha0_vec, 2))
)


# ─────────────────────────────────────────────────────────────────────────────
# 5.7  Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# ── Left Panel: Filter Coefficients ──────────────────────────────────────────
# Display the full filter coefficient vectors for all predictors.
mplot <- filter_mat

plot(mplot[, 1],
     main  = "Filter coefficients: MSE and DFP II variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])
abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right Panel: Population CCFs ─────────────────────────────────────────────
# Compute and display the population CCF for each filter.
# Vertical lines mark lag 0 (solid) and lag h (dashed).
max_lag <- l_end
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
rownames(ccf_mat) <- -max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat) <- colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main  = "Population CCFs: MSE and DFP II variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)   # lag h
abline(v = max_lag + 1,     lty = 1)   # lag 0
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# The outcome is analogous to Tutorial 4 (Exercise 1) and Tutorial 10 
# (Exercise 1). In particular, sufficiently strong skewness — achieved by 
# pulling CCF(-1) downwards while maximizing CCF(5) — causes the peak of the CCF 
# to shift toward the forecast horizon h (green to violet tones).

# The same outcome can be obtained alternatively by the original DFP as well as 
# the PCS:
#  -The DFP pulls CCF(0) downwards (decouple from present), see Tutorial 4 
#   (Exercise 1).
#  -The PCS imposes CCF(h) - CCF(h-1) > 0, ensuring the CCF is increasing
#   at the forecast horizon h, see Tutorial 10 (Exercise 1). 


# ── Cross-Check: Decoupling Constraint Verification ──────────────────────────
# Confirm numerically that the DFP II constraint is satisfied by each predictor.
#
# Step 1: Extract the CCF over the decoupling window [l_start, l_end].
#         The CCF matrix starts at lag -max_lag = -l_end, so row indices
#         2 to (l_end - l_start + 1) correspond to the decoupling interval.
ccf_decouple <- matrix(ccf_mat[1 + 1:(l_end - l_start), ],
                       nrow = (l_end - l_start))
colnames(ccf_decouple) <- colnames(ccf_mat)

# Step 2: Retain only DFP II designs (exclude nowcast and benchmark filters).
ccf_decouple_check <- ccf_decouple[,
     which("DFP" == substr(colnames(ccf_decouple), 1, 3))[1]:ncol(ccf_decouple),
                                   drop = FALSE]

# Step 3: Sum the CCF over the decoupling window column-wise and compare with
#         alpha0 scaled by the L2-norms of b and xi. Differences should vanish.
#         Note: xi is used in the CCF computation (not gamma0), so we scale by
#         ||xi||. The case alpha0 = 0 produces a 0/0 singularity; disregard it.
apply(ccf_decouple_check, 2, sum) -
  (alpha0_vec / (sqrt(apply(b_mat^2, 2, sum)) * sqrt(sum(xi^2))))


# ── Main Take-Aways ───────────────────────────────────────────────────────────
#
# CCF right-skewing and peak shifting:
#   - DFP II decouples the predictor from the past while maximising target
#     correlation, which right-skews the CCF.
#   - When the problem is feasible, sufficient decoupling (small enough alpha0)
#     shifts the CCF peak from lag 0 to lag h.
#   - Unlike PCS, DFP II does not offer explicit control over the peak location.
#     However, this is not a fundamental limitation:
#       (1) An appropriate alpha0 that places the peak at lag h can always be
#           identified empirically by scanning the alpha0 grid.
#       (2) A closed-form expression linking alpha0 to the peak location is
#           conjectured to exist for feasible problems.
#
# Comparison of the two broad approaches:
#
#   Approach A.1 — Original DFP / PCS (CCF right-tail targeting):
#     Constraints act directly on the right tail of the CCF (lags >= 0),
#     providing explicit control over the peak location. Effective only when
#     the forecast problem is feasible (sufficient rank of constraint system).
#
#   Approach A.2 — DFP II / perturbation (CCF left-tail targeting):
#     The Sigma integrator acts on the left tail of the CCF (lags < 0),
#     inducing right-skewness. Feasible in nearly all practical settings,
#     including the infeasible AR(1) case. Peak shifting to the right is
#     achievable when the problem is feasible; in infeasible cases, only
#     skewness (not peak relocation) is attainable. However, in all cases, the 
#     predictor inherits look ahead behaviour.
#
# Summary comparison:
#   - A.1 offers more direct and interpretable control over the right tail
#     (peak location), but is restricted to feasible problems.
#   - A.2 is more general and robust, applicable even when A.1 fails.
#     Control over peak location is indirect (via skewness) but likely
#     reachable in closed form for feasible problems (conjecture).
#   - The direct peak-location control of A.1 most likely extends to A.2
#     as well, suggesting that A.2 subsumes A.1 as a special case.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 5.8  Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────────────

# ── 5.8.1  Forecast Comparison ───────────────────────────────────────────────

# Generate a long white-noise series to support reliable empirical evaluation.
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes: y_t = sum_{k=0}^{L-1} b_k * eps_{t-k}
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each predictor.
anf <- 100
enf <- 150

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col  = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Note: predictors/filters which invert trend orientation or the sign of a 
# (non-vanishing) mean must be applied to centered data. In the above implementation, 
# the filters are applied to white noise which is centered.



# ── 5.8.2  Empirical CCF Comparison ──────────────────────────────────────────
# Compute empirical CCFs between the nowcast output (x_t) and each predictor
# to confirm in finite-sample data the population-level CCF peak shift
# documented in Section 5.6.

par(mfrow = c(3, 2))
select_vec <- c(2, 4, 7, 10, 12, ncol(y_out_mat))

for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main    = colnames(y_out_mat)[i])
}


# ── 5.8.3  Target Correlation and Smoothness ─────────────────────────────────
# Evaluate each predictor against the h-step-ahead target and assess output
# smoothness (roughness = RMS of second differences of standardised output).

# Construct the h-step-ahead target series (shift nowcast forward by h).
targeth <- c(y_out_mat[(1 + h):len, 1], rep(NA, h))

# Remove rows with NAs introduced by the lag shift.
y_outh <- na.exclude(cbind(targeth, y_out_mat))
y_out  <- y_outh[, 2:ncol(y_outh)]
colnames(y_out) <- colnames(y_out_mat)
target <- y_outh[, 1]

# Compute correlation of each predictor with the h-step-ahead target.
target_cor <- NULL
for (i in 1:ncol(y_out_mat))
  target_cor <- c(target_cor, cor(target, y_out[, i]))
names(target_cor) <- colnames(y_out_mat)
target_cor

# Compute output roughness: RMS of second differences of standardised output.
# Smaller values indicate smoother (less noisy) predictor output.
# Predictors become progressively noisier as alpha0 decreases.
smooth <- NULL
for (i in 1:ncol(y_out_mat))
  smooth <- c(smooth,
              sqrt(mean(diff(diff(scale(y_out[, i])))^2)))
names(smooth) <- colnames(y_out_mat)
smooth

# The DFP II predictors are much smoother than the identity and forward-looking.


# ── 5.8.4  Lead-Lag Analysis: Zero-Crossing Alignment ────────────────────────
# Quantify the temporal lead of each DFP II predictor relative to MSE(h) by
# computing the average lag at which their zero-crossings are closest
# (trough in the mean absolute zero-crossing distance curve).
#
# Interpretation:
#   - A trough at lag -k indicates that the DFP II predictor leads MSE(h)
#     by k time units.
#   - Increasing right-skewness (smaller alpha0) produces a larger lead, up
#     to a point: for alpha0 smaller than approximately 0.18, the lead signal
#     is overwhelmed by noise and the trough disappears.

max_lead   <- h
vicinity   <- 4
last_crossing_or_closest_crossing <- FALSE
outlier_limit <- 40

par(mfrow = c(3, 2))
for (i in select_vec) {
  xy_mat <- cbind(y_out[, i],
                  y_out[, paste0("MSE(", h, ")")])
  colnames(xy_mat) <- c(colnames(y_out)[i], paste0("MSE(", h, ")"))
  tau <- compute_min_tau_func(xy_mat, max_lead, vicinity,
                              last_crossing_or_closest_crossing, outlier_limit)
  tau$min_tau_plot
}

# As in earlier examples, right skewing of the CCF generates a lead at 
# zero-crossings (mean-crossings).


# ─────────────────────────────────────────────────────────────────────────────
# 5.9  Time-Shift Analysis
# ─────────────────────────────────────────────────────────────────────────────
# Compute and plot the frequency-domain time-shift function for each selected
# predictor. The time-shift function quantifies, at each frequency, how many
# periods the predictor leads or lags the target series.

K<-600
plot_T<-F
shift_mat<-NULL
for (i in select_vec)
{
  
  b1<- filter_mat[,i]
  shift_mat<-cbind(shift_mat,amp_shift_func(K,b1, plot_T)$shift)   # time-shift for lagged filter (b1)
  
}
colnames(shift_mat)<-colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat
colnames(mplot) <- colnames(shift_mat)

plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: Whole frequency interval",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# From 0 to pi/6 
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "In interval [0,pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# From 0 to pi/6 and trimmed to [-10,10]
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Trimmed to -10,10",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()


# As in earlier examples, right-skewing the CCF imposes a smaller time-shift at 
# lower frequencies.



# ─────────────────────────────────────────────────────────────────────
# 5.9 AR Form
# ─────────────────────────────────────────────────────────────────────

# ── 5.9.1 AR Inversion ───────────────────────────────────

ar_inv <- -ARMAtoMA(ar = -xi[2:L],
                              ma = 0,
                              lag.max = L)

# Assemble the full AR filter (including the leading coefficient of 1).
theta <- c(1, -ar_inv)

# Verify correctness via a known identity:
# convolving the AR inversion filter with the Wold (MA) coefficients must
# yield the identity filter, i.e., [1, 0, 0, ...]. Negligible deviation from 
# the identity is due to finite length-L inversions.
conv_two_filt_func(xi, theta)$conv[1:L]

# Visualise the AR inversion filter.
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR Inversion Filter")


# Having verified the identity, we convolve the AR operator with each PCS
# predictor (in MA form) to obtain the corresponding AR-form representation.


# ── 5.9.2 Derivation of AR Forms ───────────────────────────────────


# Convolve the AR operator with each predictor in filter_mat to obtain
# the AR-form coefficient vectors.
filter_mat_ar <- NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar <- cbind(filter_mat_ar,
                         conv_two_filt_func(theta, filter_mat[, i])$conv)
colnames(filter_mat_ar) <- colnames(filter_mat)



# Plot

colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- h+3
par(mfrow = c(1, 1))
# Plot the first `first_lags` AR coefficients for each predictor.
ts.plot(
  scale(filter_mat_ar[1:first_lags, ],center=F,scale=T),
  col  = colo,
  main = "DFP II Predictors in AR Form (scaled to unit-length)",
  lty  = c(2, 2, rep(1, ncol(filter_mat) - 2)),
  lwd  = c(1, 2, rep(1, ncol(filter_mat) - 2)))
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)



# ════════════════════════════════════════════════════════════════════
# EXERCISE 6:  Applying DFP II to the Hodrick-Prescott (HP) Filter
# ════════════════════════════════════════════════════════════════════

#----------------------------------------------------------------------
# This concluding exercise applies the DFP II to a feasible
# peak-correlation shifting problem based on the Hodrick-Prescott
# filter.
#
# While the original results from Tutorial 11 could be replicated
# by setting l_start=0 and l_end=1 (which renders the identity
# Sigma integrator: Sigma = Id), we here deliberately depart from
# that pure replication setting by choosing an alternative Sigma
# operator that emphasizes the CCF aggregated over past lags
# k = -1, ..., -h, where h = 4 (yearly forecast in a quarterly
# data framework).
#
# This example provides additional insights into the flexibility
# of the DFP II framework.
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Notes:
#
# 1. Feasibility means that the right tail of the CCF is amenable
#    to modification so that the peak of the CCF can be right-
#    shifted: the HP forecast problem is feasible.
#
# 2. For feasible problems, the original DFP or the PCS can
#    typically be used, relying on minimal constraints to maximize
#    the target correlation, see Tutorial 11.
#
# 3. A natural setting for DFP II is l_start=0 and l_end=1,
#    resulting in Sigma = Id and replication of the original DFP
#    or PCS Types II or III.
#      - The DFP formalism can also implement PCS Types II and III, see 
#        for example Tutorial 10, Exercises 1 and 2. so DFP II can replicate 
#        these PCS Types.
#      - PCS Type I utilizes multiple constraints and differs,
#        i.e., it cannot be replicated by DFP or DFP II.
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Recommendation:
#
# For feasible problems we typically recommend the original DFP,
# the PCS Types II or III, or DFP II with Sigma = Id (l_start=0 and l_end=1).
#
# We here depart from this recommendation to illustrate DFP II
# when operating on a Sigma operator of length h = 4, instead.

# One can easily compare this (suboptimal) setting with the recommended one 
# by selecting l_end = 1.
#----------------------------------------------------------------------


# ── HP FILTER SET-UP ──────────────────────────────────────────────────────────

# Standard HP smoothing parameter for quarterly data (Hodrick & Prescott, 1997).
lambda_hp <- 1600

# Filter half-length: L must be odd so that the symmetric filter is centred
# at position (L - 1)/2 + 1, avoiding ambiguity between two consecutive lags.
L_HP <- 101

# Compute the concurrent (causal, one-sided) HP trend filter coefficients.
# The argument 2*(L_HP - 1) + 1 specifies the full symmetric filter length,
# from which both the classic one-sided and MSE-optimal variants are extracted.
HP_obj <- HP_target_mse_modified_gap(2 * (L_HP - 1) + 1, lambda_hp)

# Classic one-sided (concurrent) HP trend filter.
hp_c <- HP_obj$hp_trend

# Right half of the symmetric HP trend filter (MSE-optimal under white noise).
hp_trend <- HP_obj$hp_mse

par(mfrow = c(1, 1))
ts.plot(
  cbind(hp_trend, hp_c),
  main = paste("Right half and classic concurrent HP(", lambda_hp, ")", sep = ""),
  col  = c("green", "violet")
)
mtext("Right-half HP (MSE-optimal under white noise)", line = -1, col = "green")
mtext("Classic HP-C (one-sided concurrent filter)",   line = -2, col = "violet")

# ── AR(2) Structure of the HP Filter ─────────────────────────────────────────
# The HP filter coefficients satisfy an AR(2) recurrence relation. This is
# confirmed by the regressions below: as L grows, residuals vanish asymptotically,
# and the two predictors account for virtually all variation in the filter weights.
summary(lm(hp_c[2 + 1:L_HP]     ~ hp_c[1 + 1:L_HP]     + hp_c[1:L_HP]))
summary(lm(hp_trend[2 + 1:L_HP] ~ hp_trend[1 + 1:L_HP] + hp_trend[1:L_HP]))

# Extract the AR(2) coefficients (slopes only, intercept dropped): these are 
# used below, when deriving the AR form of the predictors.
ar_vec <- lm(hp_trend[2 + 1:L_HP] ~
               hp_trend[1 + 1:L_HP] + hp_trend[1:L_HP])$coef[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────────────

# Number of lags retained in the truncated filter.
L <- 50

# Forecast horizon: one year equals four quarters for quarterly data.
h <- 4

# MSE-optimal h-step-ahead predictor (gammah):
#   gammah[k] = hp_trend_{h+k}  for k = 0, …, L-1,
# obtained by shifting the symmetric HP weights forward by h lags and
# padding any positions beyond the filter support with zero.
gammah  <- hp_trend[h + 1:L]   # shifted HP weights (the target filter)
gamma0  <- hp_trend[1:L]        # unshifted HP weights (the nowcast filter)
gamma_I <- c(1, rep(0, L - 1))  # identity (pass-through) filter
xi      <- hp_trend              # MA coefficients of the data-generating process

# Notes: 
# - xi is used when computing the population CCF only, see Exercise 6.6. It is not 
#   relevant for optimization. Optimization solely relies on gammah (the target), 
#   and gamma_constraint with alpha0 (the constraint).


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  DFP II Set-Up
# ─────────────────────────────────────────────────────────────────────────────
# ── Constraint Design: Integrated Decoupling over a Full Forecast Horizon ────
#
# Right-shifting of the CCF peak to the forecast horizon h is feasible with 
# a `simple' constraint, where Sigma = Id (original DFP. Here we depart from that 
# simple baseline by setting l_end <- h, so that the integrator Sigma accumulates 
# the CCF over a full year (four quarters for quarterly data).
#
# Operationally, this means that right-skewness of the CCF is induced by
# simultaneously pulling the CCF downward at all lags k = -1, -2, …, -h = -4,
# rather than targeting the single lag k = -1. 
# Lag window over which the decoupling constraint is enforced.
l_start <- 0
l_end   <- 4

if (l_start >= l_end) {
  print("l_start must be strictly less than l_end; resetting l_start.")
  l_start <- l_end + 1
}

# ── Integrator matrix Sigma ───────────────────────────────────────────────────
Sigma <- matrix(0, nrow = L, ncol = L)
for (i in (l_start + 1):l_end)
  Sigma[i, (1 + l_start):i] <- 1

if (l_end < L) {
  for (i in (l_end + 1):L)
    Sigma[i, ((i - l_end + l_start) + 1):i] <- 1
}

# Inspect the structure of Sigma
Sigma[1:8,1:8]


# Constraint vector gamma_constraint: 
gamma_constraint <- Sigma %*% c(rep(0, l_start), gamma0[1:(L - l_start)])

par(mfrow = c(1, 1))
ts.plot(
  gamma_constraint,
  main = expression(gamma[constraint] == Sigma * gamma[0]),
  xlab = "Lag", ylab = "",
  sub  = "Algebraic constraint vector controlling the right-skewness of the CCF"
)
abline(h = 0)


# ── Baseline Coupling (MSE Predictor) ────────────────────────────────────────
mse_coup <- as.double(gammah %*% gamma_constraint)

# ── Grid of Decoupling Levels (alpha0) ───────────────────────────────────────
alpha0_vec <- c(mse_coup / 1.5^(1:9), 0)  # final grid: geometric decay to 0

# Display the grid. 
alpha0_vec


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  Decoupling over the alpha0-Grid
# ─────────────────────────────────────────────────────────────────────────────

b_mat       <- NULL   # columns: filter coefficient vectors, one per alpha0
cor_vec_mat <- NULL   # columns: full CCF vectors,           one per alpha0

# Two-column matrix storing CCF at lag 0 and lag h for each alpha0 value,
# used for a concise tabular comparison of coupling vs. target correlation.
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of lags on each side of lag 0 to include in the displayed CCF.
# This parameter only affects the plotting range, not the optimisation.
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Solve for the DFP II filter coefficient vector at this alpha0 level.
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  
  b_mat <- cbind(b_mat, b)
  
  # Evaluate the population CCF of filter b against the target process xi
  # over lags −max_lag, …, 0, …, h.
  cor_vec <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), xi
  )$cor_vec
  
  cor_vec_mat <- cbind(cor_vec_mat, cor_vec)
}

# Attach descriptive names to columns of both output matrices.
colnames(b_mat)       <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1)   <- c("Lag 0", paste0("h=", h))
rownames(cor_vec_1)   <- paste0("alpha0=", round(alpha0_vec, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 6.5  Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: Constraint satisfaction ─────────────────────────────────────────
t(b_mat) %*% gamma_constraint - alpha0_vec

# ── Check 2: Sign / orientation preservation ─────────────────────────────────
# Predictors that do not preserve trend orientation require centering of the 
# data since they also invert the sign of a non-vanishing mean level.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────────────
t(b_mat) %*% gammah


# ── Consolidated filter matrix ───────────────────────────────────────────────
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c(
  "Nowcast",
  paste0("MSE(", h, ")"),
  "Identity",
  paste0("DFP II ", round(alpha0_vec, 5))
)


# ─────────────────────────────────────────────────────────────────────────────
# 6.6  Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: scaled filter coefficients ───────────────────────────────────
# Scale each filter to unit length so that shape differences are visible
# regardless of overall amplitude. Vertical lines mark lag 0 (solid) and
# lag h (dashed) to facilitate visual alignment with the CCF panel.
mplot <- scale(filter_mat, center = FALSE, scale = TRUE)

plot(mplot[, 1],
     main  = "Scaled filter coefficients: MSE and PCS variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), max(mplot)))

for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])

for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], line = -i, col = colo[i])

abline(h = 0)
abline(v = 1,     lty = 1)   # lag 0
abline(v = h + 1, lty = 2)   # lag h (forecast horizon)
axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────────────
# Each curve is the population cross-correlation function (CCF) between the
# corresponding filter output and the target process. Vertical lines mark
# lag 0 (solid) and lag h (dashed), indicating the target lead location.
max_lag <- l_end
ccf_mat <- NULL

for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi
                   )$cor_vec)

rownames(ccf_mat) <- -max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat) <- colnames(filter_mat)
mplot <- ccf_mat

plot(mplot[, 1],
     main  = "Population CCFs: MSE and PCS variants",
     axes  = FALSE, type = "l",
     xlab  = "Lag", ylab = "",
     col   = colo[1], lwd = 1,
     ylim  = c(min(0, min(mplot)), max(mplot)))

for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])

abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)   # lag h (forecast horizon)
abline(v = max_lag + 1,     lty = 1)   # lag 0 (contemporaneous)
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Cross-check: decoupling constraint consistency ───────────────────────────
# Verify that the CCF-based coupling values match the prescribed alpha0 values.
#
# Step 1: Extract CCF values over the decoupling window [l_start, l_end].
#         The CCF matrix ccf_mat starts at lag −max_lag = −l_end, so the
#         window [l_start+1, l_end] corresponds to rows 2 … (l_end − l_start + 1).
ccf_decouple <- matrix(ccf_mat[1 + 1:(l_end - l_start), ],
                       nrow = (l_end - l_start))
colnames(ccf_decouple) <- colnames(ccf_mat)

# Step 2: Retain only DFP II designs (drop the nowcast and benchmark columns).
ccf_decouple_check <- ccf_decouple[,
    which("DFP" == substr(colnames(ccf_decouple), 1, 3))[1]:ncol(ccf_decouple),
                                   drop = FALSE
]

# Step 3: Sum the CCF over the decoupling window and compare with the
#         normalised alpha0 values: alpha0 / (||b|| * ||xi||).
#         The CCF is computed against xi rather than gamma0, so the
#         denominator uses length(xi), not length(gamma0).
#         All differences should be numerically zero.
#         (The case alpha0 = 0 produces a 0/0 indeterminate form; ignore it.)
apply(ccf_decouple_check, 2, sum) -
  (alpha0_vec / (sqrt(apply(b_mat^2, 2, sum)) * sqrt(sum(xi^2))))


# ── Analysis of Results ───────────────────────────────────────────────────────
#
# 1. Decoupling mechanism (DFP II):
#    Decoupling the filter output from the past — while simultaneously
#    maximising the correlation with the target — right-skews the CCF.
#
# 2. Effects of increasing decoupling strength (decreasing alpha0):
#    Tightening the constraint (i.e., reducing alpha0) produces two
#    interrelated changes in the filter and its CCF:
#      (a) Faster CCF decay over lags k = 0, …, h−1: the one-year integrator
#          Sigma attempts to suppresses the CCF uniformly across the window [−h, −1],
#          so reducing alpha0 accelerates the drop-off from the contemporaneous
#          peak toward zero over this entire range.
#      (b) A more pronounced and left-shifted negative lobe in the filter
#          coefficients: as decoupling intensifies, the filter develops an
#          increasingly negative tail at short lags, a signature of the
#          phase-shift that advances the effective timing of the output.
#    For comparison, one may repeat the exercise with the minimal constraint
#    (Sigma = I, obtained by setting l_end <- 1 instead of l_end <- h) to
#    isolate the contribution of the broader integrator window. This
#    comparison is left as an exercise.
#
# 3. CCF peak migration:
#    Once decoupling is sufficiently strong, the CCF peak migrates from
#    lag 0 to lag h, confirming that the filter output genuinely leads the
#    target series by h steps in population.
#
# 4. Implicit peak control under DFP II:
#    Unlike the PCS approach, DFP II imposes no explicit constraint on the
#    location of the CCF peak. In practice, however, this is not a limitation:
#      (a) One can always search over alpha0 to place the CCF peak at the
#          intended forecast horizon h, provided the problem is feasible at
#          that decoupling level.
#      (b) A closed-form expression for the alpha0 value that positions the
#          CCF peak exactly at lag h very likely exists and is a natural
#          target for future analytical work (derivation pending).



# ─────────────────────────────────────────────────────────────────────
# 6.7 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 6.7.1 Forecast Comparison ────────────────────────────────────────

# Generate a long white-noise series for a reliable empirical evaluation
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

scale(y_out_mat)[,1:2]

colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))


# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 390
enf <- 430
anf <- 500
enf <- 600

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
# As alpha0 decreases, the predictors become left-shifted but they remain 
# smooth overall, see the smoothness measure below.


# ── 6.7.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(3, 2))

select_vec<-c(2,4,7,10,12,ncol(y_out_mat))


for ( i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main = paste(colnames(y_out_mat)[i],sep=""))
}  

# As alpha_0 decreases, the empirical CCF becomes increasingly right-skewed.


# Compute target correlations and smoothness
targeth<-c(y_out_mat[(1+h):len,1],rep(NA,h))
# Remove NAs
y_outh<-na.exclude(cbind(targeth,y_out_mat))
y_out<-y_outh[,2:ncol(y_outh)]
colnames(y_out)<-colnames(y_out_mat)
target<-y_outh[,1]

target_cor<-NULL
for (i in 1:ncol(y_out_mat))
  target_cor<-c(target_cor,cor(target,y_out[,i]))
names(target_cor)<-colnames(y_out_mat)
target_cor


smooth<-NULL
for (i in 1:ncol(y_out_mat))
  smooth<-c(smooth,sqrt(mean(diff(diff(scale(y_out[,i])))^2)))
names(smooth)<-colnames(y_out_mat)
# The DFP II predictors remain smooth despite increasing lead (advancement). 
# Note: the setting l_end = 1 (Sigma = Id) improves smoothness. 
smooth


# Left-shift of trough: 
#  -Curve: average distance between zero-crossings
#  -Trough in curve: lag at which zero-crossings of DFP II and MSE(12) are closest 
#    i.e. at which series are aligned.
# -lead (negative) or lag (positive) of DFP II compared to MSE(12): at trough at -k 
#   indicates a lead of k time units of DFP II over MSE(12).
# -Increased decoupling from paste or skewness (smaller alpha0) generates a left-shift 
#   of the trough (larger lead) of DFP II, up to some point: for alpha_0 smaller than 0.18 
#   the lead at zero-crossings has vanished into the noise.


max_lead   <- 12
vicinity=4
last_crossing_or_closest_crossing=F
outlier_limit=40
par(mfrow = c(3, 2))
plot_vec<-NULL
for (i in select_vec) #i<-2
{
  xy_mat <- cbind(y_out[,i],y_out[,paste("MSE(",h,")",sep="")])
  colnames(xy_mat)<-c(colnames(y_out)[i],paste("MSE(",h,")",sep=""))
  tau<-compute_min_tau_func(xy_mat, max_lead,vicinity,last_crossing_or_closest_crossing,outlier_limit)
  tau$min_tau_plot
}


# ── Notes on Decoupling Strength and Phase Behaviour ─────────────────────────
#
# 1. Lead at zero crossings:
#    Decreasing alpha0 (i.e., enforcing stronger CCF skewness) progressively
#    increases the lead of the filter output relative to the target, as measured
#    by the location of the zero crossings of the CCF (tau statistic, see 
#    Wildi 2024).
#
# 2. Phase reversal at excessive decoupling:
#    Beyond a critical decoupling threshold, the periodic AR(2) structure of
#    the HP filter causes a phase reversal: what were leads (positive time
#    advance) flip into lags (negative time advance). This is a direct
#    consequence of the oscillatory nature of the AR(2) impulse response,
#    which wraps phase continuously and can produce a net phase shift of more
#    than half a cycle if the decoupling constraint is pushed too far.
#
# 3. Loss of interpretability at phase reversal:
#    At the point of phase reversal, the filter output is effectively
#    sign-inverted relative to the target: upturns in the target are
#    signalled as downturns by the filter, and vice versa. This renders
#    the predictor practically uninterpretable and economically misleading.
#    In applied work, alpha0 should therefore be chosen to remain safely
#    on the near side of this threshold, where the gain at zero frequency
#    is still positive and the CCF peak is unambiguously located at lag h.



# ─────────────────────────────────────────────────────────────────────
# 6.8 Time-Shifts
# ─────────────────────────────────────────────────────────────────────

# Time-Shifts

K<-600
plot_T<-F
shift_mat<-NULL
for (i in select_vec)
{
  
  b1<- filter_mat[,i]
  shift_mat<-cbind(shift_mat,amp_shift_func(K,b1, plot_T)$shift)   # time-shift for lagged filter (b1)
  
}
colnames(shift_mat)<-colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat
colnames(mplot) <- colnames(shift_mat)

plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: Whole frequency interval",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# From 0 to pi/6 
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "In interval [0,pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# From 0 to pi/6 and trimmed to [-10,10]
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Trimmed to -10,10",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# Interpretation:
# Increasing right-skewness through smaller alpha_0 decreases the time-shift, as 
# desired.

# ─────────────────────────────────────────────────────────────────────
# 6.9 AR Form
# ─────────────────────────────────────────────────────────────────────

# ── 6.9.1 AR Inversion ───────────────────────────────────

# Compute the AR inversion filter (the inverse of the MA polynomial),
# which transforms the Wold (MA-infinity) representation back into
# an AR representation.
ar_inv <- ar_vec


# Assemble the full AR filter (including the leading coefficient of 1).
theta <- c(1, -ar_inv)

# Verify correctness via a known identity:
# convolving the AR inversion filter with the Wold (MA) coefficients must
# yield the identity filter, i.e., [1, 0, 0, ...]. Negligible deviation from 
# the identity is due to finite length-L inversions.
conv_two_filt_func(xi, theta)$conv[1:L]

# Visualise the AR inversion filter.
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR Inversion Filter")


# Having verified the identity, we convolve the AR operator with each PCS
# predictor (in MA form) to obtain the corresponding AR-form representation.


# ── 6.9.2 Derivation of AR Forms ───────────────────────────────────

# Verification: convolving theta with gamma_0 should recover the identity
# filter. Residual deviations from [1, 0, 0, ...] vanish as L increases,
# reflecting the finite truncation of both the MA and AR representations.
conv_two_filt_func(theta, gamma0)$conv[1:10]

# Convolve the AR operator with each predictor in filter_mat to obtain
# the AR-form coefficient vectors.
filter_mat_ar <- NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar <- cbind(filter_mat_ar,
                         conv_two_filt_func(theta, filter_mat[, i])$conv)
colnames(filter_mat_ar) <- colnames(filter_mat)

# Verification: the first column (nowcast gamma_0) should be the identity.
filter_mat_ar[1:10, 1]


# Plot

colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- h+3
par(mfrow = c(1, 1))
# Plot the first `first_lags` AR coefficients for each predictor.
ts.plot(
  scale(filter_mat_ar[1:first_lags, ],center=F,scale=T),
  col  = colo,
  main = "DFP II Predictors in AR Form (scaled to unit-length)",
  lty  = c(2, 2, rep(1, ncol(filter_mat) - 2)),
  lwd  = c(1, 2, rep(1, ncol(filter_mat) - 2)))
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)


# Comment: Sigma affects the AR form, in particular towards lag h.








