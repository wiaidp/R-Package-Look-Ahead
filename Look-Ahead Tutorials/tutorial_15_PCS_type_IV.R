# ═══════════════════════════════════════════════════════════════════
# TUTORIAL 15 — PCS TYPE IV: EXPLOITING CCF INERTIA
# ═══════════════════════════════════════════════════════════════════

# This tutorial reexamines the forecasting problem introduced in Tutorial 12
# and presents a novel approach, PCS type IV, to resolving the look-ahead problem.

# ── Challenging Forecast Problem ──────────────────────────────────────────────
#
# The forecasting problem is challenging for three interconnected reasons:
#
#   1. The MSE-optimal predictor is "stuck" at horizon h = 12: selecting
#      h_tilde > 12 does not affect the predictor up to scaling, so extending
#      the forecast horizon yields no additional predictive information.
#
#   2. The CCF of the MSE predictor peaks at lag k = 4, which lies significantly
#      to the left of the intended forecast horizon h = 12. A predictor whose CCF 
#      peak is shifted rightward toward h = 12 would more directly target the 
#      intended forecast horizon and is therefore a potentially meaningful 
#       — though not necessarily realisable — objective.
#
#   3. Attempts to displace the CCF peak further to the right via a PCS Type I
#      constraint partly fail, because the linear-growth assumption underlying
#      Type I conflicts with the internal structure of the DGP, leading to
#      heavily misspecified and ultimately unusable predictors (see Tutorial 12).
#
# The formal background — including the DGP specification, the PCS framework,
# and the constraint types — is not repeated here. The reader is referred to
# Tutorial 12 and the comprehensive introduction for full details.

# ── CCF Inertia ───────────────────────────────────────────────────────────────
#
# We briefly replicate Exercise 2 of Tutorial 12 — which derives the closed-form
# solution to the PCS Type I problem — and use it as a diagnostic tool to analyse
# the look-ahead problem in depth. Building on this analysis, we identify specific
# structural features of the DGP that, rather than hindering look-ahead behaviour,
# can be exploited to achieve it more effectively than the original Type I
# formulation allows.
#
# Key observations:
#
#   - Imposing a linear-growth Type I PCS constraint at k = 1, …, 12 encodes a
#     strong misspecification and biases the PCS toward unusable predictors
#     (see Exercises 1 and 2 in Tutorial 12 and exercises 1 and 3 below).
#
#   - In contrast, the new TYPE IV PCS introduced in this Tutorial imposes the 
#     single constraint 
#           CCF(1) - CCF(0) = beta 
#     which is feasible. For h = 1 this constraint coincides with Type I, II, 
#     and III simultaneously, making it a natural look ahead candidate.
#
#   - Imposing a single constraint avoids the linear-growth assumption of Type I,
#     which is a strong misspecification in the considered forecasting problem.
#     Moreover, retaining 12 of the 13 available degrees of freedom for
#     optimisation allows the solver to maximise the target correlation CCF(h)
#     at h = 12 as effectively as possible. The resulting predictor is therefore
#     efficient in a well-defined sense: the minimum number of constraints
#     necessary to induce the desired look-ahead behaviour is imposed, and all
#     remaining degrees of freedom are invested in maximising predictive accuracy
#     at the target horizon.
#
#   - Two structural features of the DGP are of particular interest:
#       1. For k > 12, the CCF slope CCF(k+1) - CCF(k) decays exponentially
#          (see Tutorial 12), so the CCF is guaranteed to turn downward beyond
#          the forecast horizon h = 12.
#       2. Consecutive CCF slopes CCF(k) - CCF(k-1) and CCF(k+1) - CCF(k)
#          cannot differ arbitrarily — a property referred to as CCF inertia —
#          meaning the CCF-slope changes smoothly across lags.
#
#   - Exploiting CCF inertia proceeds in four steps:
#       i)   Setting beta > 0 in the Type IV constraint imposes 
#               CCF(1) - CCF(0) = beta > 0, 
#            introducing an upward `kick' in the CCF slope at the initial lags.
#       ii)  For k > 12, the DGP structure forces CCF(k+1) = a1 * CCF(k) < CCF(k)
#            (assuming positivity), so the CCF must eventually decay exponentially
#            regardless of the constraint imposed at lag k = 1.
#       iii) Given (i) and (ii), CCF inertia implies that CCF(k) for k = 2, …, 12
#            must trace a smooth, continuous path between the initial upward kick
#            at lags 0–1 and the exponentially decaying profile at lags k > 12,
#            thereby inducing a genuine rightward shift of the CCF peak.
#       iv)  This mechanism can be understood via the mean-value theorem: the
#            slope function CCF(k) - CCF(k-1) is `continuous' in k (by inertia),
#            starts positive at k = 1, and must turn negative for k > 12.
#            It must therefore cross zero somewhere in the interval k = 1, …, 13,
#            and this zero-crossing marks the peak of the CCF. Crucially, the
#            larger the initial slope at k = 1 (controlled by beta > 0), the
#            further to the right the zero-crossing of the CCF-slope  — and 
#            hence the CCF peak — will be located.

# ── Summary ───────────────────────────────────────────────────────────────────
#
#   - Larger admissible values of beta push the CCF peak progressively further
#     to the right, generating look-ahead behaviour that extends beyond the MSE
#     benchmark, whose CCF peak is structurally bounded at lag k <= 4.
#
#   - At the same time, the single constraint imposed at k = 1 is minimally
#     invasive: by leaving 12 of the 13 available degrees of freedom free for
#     optimisation, it allows the solver to maximise the target correlation CCF(h)
#     at h = 12 as effectively as possible, ensuring that look-ahead improvement
#     is not achieved at the cost of a large reduction in predictive accuracy.
#

# ── PCS Type IV ───────────────────────────────────────────────────────────────
#
# This tutorial exploits CCF inertia to recover look-ahead behaviour. Knowledge
# of Tutorial 12 is assumed; the novel PCS Type IV approach is defined by:
#
#   TYPE IV — Initial beta kick at k=1:
#
#        CCF(1) - CCF(0) = beta > 0.

# ── Exercise Overview ─────────────────────────────────────────────────────────
#
#   Exercise 1 — Replicates Tutorial 12 (Exercise 2): derives the closed-form
#     solution when imposing all 12 Type I constraints simultaneously. The
#     constraint system has rank 13 (i.e. 13 independent constraints can be
#     enforced), leaving only one degree of freedom for optimisation — which is
#     insufficient to produce usable predictors, because the linear CCF-growth
#     assumption conflicts strongly with the DGP.
#
#   Exercise 2 — Exploits CCF inertia: a single Type IV constraint
#     CCF(1) - CCF(0) = beta > 0 is imposed, and the CCF peak is allowed to
#     shift rightward as beta increases. Retaining only one constraint leaves
#     12 degrees of freedom available for target-correlation maximisation.
#
#   Exercise 3 — Analyses the boundary case in which 13 independent
#     constraints of the rank-13 system are imposed simultaneously (i.e. the
#     constraint system is fully exhausted), leaving no room for optimisation
#     beyond the choice of sign.
#

# ── Main Take-Aways ───────────────────────────────────────────────────────────
#
# The findings extend and complement those of Tutorial 12. The main points are
# summarised below.
#
#   1. The principal novelty of this tutorial is the introduction of the PCS 
#      Type IV constraint. It is based on the recognition and exploitation
#      of two salient structural features of the DGP: CCF smoothness (inertia)
#      and the exponential decay profile for k > 12. Together, these properties
#      can be leveraged to shift the CCF peak rightward beyond lag k = 4 — the
#      maximum achievable under the MSE predictor — without resorting to the
#      heavily misspecified Type I constraint system.
#
#   2. The proposed strategy is generic: it can be applied to any DGP whose
#      tight structural constraints admit a similar combination of smoothness
#      and rigid decay in the CCF beyond a fixed lag.
#
# The following take-aways from Tutorial 12 remain fully applicable here and
# provide the broader methodological context:
#
#   1. Over-emphasising high-dimensional, potentially misspecified constraints
#      — whether through strong regularisation or exact closed-form solutions —
#      comes at the direct expense of the target correlation, yielding unusable
#      predictors or predictors with no look-ahead behaviour.
#
#   2. Relaxing the constraint space frees degrees of freedom for maximising
#      the target correlation, and is generally necessary to obtain effective
#      look-ahead behaviour in difficult forecasting problems.
#
#   3. Relaxation can be achieved in at least two complementary ways:
#        (a) Reducing the number of constraints, e.g., using a single slope
#            constraint rather than the full Type I system, or switching to
#            the less restrictive Type II, Type III or Type IV formulations.
#        (b) Reducing the regularisation weight placed on the constraints,
#            allowing the optimiser greater freedom to pursue the target
#            correlation.
#

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
# Equally Weighted Trend, PCS Type I Closed-Form Solution
# ════════════════════════════════════════════════════════════════════════════════

# We adopt the framework from Tutorial 12: an ARMA(1,1) model fitted to
# the monthly PAYEMS employment indicator. We then consider forecasting of yearly 
# growth one year ahead. We rely on PCS Type I, imposing a linear slope to 
# CCF(k), k=1,...,12 and derive the exact closed-form PCS solution, see equations 
# 47 and 48, Wildi (2026). 

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
# the CCF peak further to the right: the MSE predictor is "stuck at horizon 12" 
# and the MSE paradigm cannot shift the CCF peak to the right of k = 4.
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

# Implications PCS type I with h = 12:
#     The closed-form solution absorbs all but one degree of freedom. Only one 
#     degree of freedom remains to maximize the target correlation, severely 
#     limiting look-ahead potential: the predictors are unusable.

# ─────────────────────────────────────────────────────────────────────────────
# 1.6 Set-Up PCS: Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Twelve constraints at lags 1,...,12
Delta  <- 1:h

# Determine a grid of potentially interesting beta (slope) values. 
# Selecting informative beta values manually can be difficult. PCS_func()
# addresses this by automatically constructing a candidate grid concentrated
# around the tipping point of the PCS optimization — the region where the
# predictor reacts most sensitively to small changes in beta. Screening
# solutions in this neighbourhood often provides the sharpest insight into the
# structure of the optimization problem.

beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- T
lambda<-10^6

PCS_obj  <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                     Type_III, scaled_constraints, high_resolution)

# We use the automatic grid as the base, and augment it
# with additional slope values at which the predictor changes
# profile sharply (identified from prior inspection of the solution path).
beta_vec_automatic <- PCS_obj$beta_vec

# Refine locally grid resolution: beta_vec collects all relevant beta values 
beta_vec <- c(beta_vec_automatic[1],
              -8e-05, -4e-05, -3e-05, -2e-05,
              beta_vec_automatic[2:length(beta_vec_automatic)])

# Note: no regularisation parameter lambda needs to be specified here, since the
# closed-form solution provided by PCS_closed_form_func() is used directly
# (see loop below). The closed-form approach bypasses the need for weighting the 
# constraints and renders lambda caduc in this instance.


# ─────────────────────────────────────────────────────────────────────────────
# 1.7 Run Closed-Form PCS
# ─────────────────────────────────────────────────────────────────────────────


b_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor (no regularization weight lambda needed).
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)
  
  b             <- PCS_obj$b
  d_delta       <- PCS_obj$d_delta
  b_mat  <- cbind(b_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly, so residuals should be zero (up to machine
  # precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("Closed-form PCS, beta=",
                                 round(beta_vec, 2)))



# ─────────────────────────────────────────────────────────────────────────────
# 1.8 Plots and Performance Summary
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


# Filter weights (left panel):
#   - The rank of the constraint system is 13, in principle allowing up to 13
#     linearly independent constraints to be imposed simultaneously (see 
#     exercise 3 below). However, the constant-slope Type I constraints strongly 
#     conflict with the DGP structure and are therefore misspecified. As a 
#     result, the filter profiles are irregular and difficult to interpret. 
#   - Most weight is assigned to lagged data (similar to time reversion). 
#
# CCFs (right panel):
#   - MSE(12) and MSE(24) have identical (overlapping) CCFs peaking at k = 4. 
#     The MSE paradigm cannot shift the peak-CCF further to the right.
#   - A linearly increasing CCF (positive beta) from k=0 to k=12 is only 
#     achievable through sign inversion of the filter, causing the target 
#     correlation CCF(h) to turn negative and rendering the predictor unusable 
#     (cyan to violet color tones).
#   - The DGP structure fundamentally conflicts with a linearly increasing
#     CCF constraint: the Type I problem is therefore misspecified for this
#     DGP.
#   - Due to the convolution of MA(12) and ARMA(1,1) (yearly growth), the CCF 
#     decays exponentially beyond the forecast horizon h = 12: for k > 0,
#           CCF(12 + k) = a1^k * CCF(12),
#     irrespective of the choice of predictor b. This exponential decay is
#     an intrinsic property of the DGP and cannot be overcome by any linear
#     filter, fundamentally limiting the achievable rightward shift of the
#     CCF peak.


# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 —  Exploiting CCF Inertia
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────

# - Imposing a linear-growth Type I PCS constraint at k = 1, ..., 12 encodes
#   a strong misspecification and biases the PCS toward unusable predictors.
# - In contrast, imposing the single constraint CCF(1) - CCF(0) = beta is
#   feasible; for h = 1 this constraint coincides with Type I, II, and III
#   simultaneously.
# - Two structural features of the DGP are of particular interest:
#     1. For k > 12, the CCF slope CCF(k+1) - CCF(k) decays exponentially.
#     2. CCF(k) - CCF(k-1) cannot deviate too far from CCF(k+1) - CCF(k),
#        a property referred to as CCF inertia.
# - Exploiting CCF inertia:
#     i)   Setting beta > 0 imposes CCF(1) - CCF(0) = beta > 0, introducing
#          an upward kick at the shortest lags.
#     ii)  For k > 12, CCF(k+1) = a1 * CCF(k) < CCF(k) (assuming positivity),
#          so the CCF must eventually decay exponentially.
#     iii) Given i) and ii), inertia implies that CCF(k) for k = 2, ..., 12
#          must trace a smooth path of continuity between the initial upward
#          kick at lags 0 and 1 and the exponentially decaying profile at
#          lags k > 12, thereby inducing a genuine rightward shift of the
#          CCF peak.
# - Larger admissible beta values push the peak progressively further to the
#   right (though no further than k=12), generating look-ahead behavior beyond 
#   the MSE benchmark, whose peak is structurally bounded by k = 4 (achieved 
#   by MSE(12)).

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 PCS Set-Up: Hyperparameter Setting
# ─────────────────────────────────────────────────────────────────────────────

# DGP specification as in exercise 1: yearly growth, i.e., the convolution of the ARMA(1,1)
# Wold decomposition with the equally-weighted MA(12) target filter.
gamma_pcs <- gamma

# Novelty: main difference to exercise 1.
# Reduce the constraint system to the single Type II constraint
#   b'(gamma_1 - gamma_0) = beta.
# For beta > 0 this imposes an initial positive beta kick at lag 1 relative
# to lag 0, which, through CCF inertia, propagates smoothly across lags
# k = 2, ..., 12 and shifts the CCF peak rightward beyond the MSE benchmark.
Delta <- 1

# Automatically generate a grid of interesting beta values via PCS_func().
beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- F

PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                    Type_III, scaled_constraints, high_resolution)

# Use the automatically generated grid as the base for subsequent optimisation.
beta_vec_automatic <- PCS_obj$beta_vec
# Focus on a finer grid (divide the original grid-values by 2)
beta_vec           <- beta_vec_automatic/2




# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Closed-Form PCS Based on PCS_closed_form_func()
# ─────────────────────────────────────────────────────────────────────────────

b_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)
  
  b             <- PCS_obj$b
  d_delta       <- PCS_obj$d_delta
  b_mat  <- cbind(b_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly, so residuals should be zero (up to machine
  # precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above: residuals are zero by construction for the
# closed-form solution.

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# The Predictor is much more robust to trend inversion: much larger beta values 
# are required to eventually drive the predictor into trend (or fixed level) 
# inversion:
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# In contrast to exercise 1, the target correlations (covariances) remain 
# positive for all selected beta >  0 (no misspecification).
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("Closed-form PCS, beta=",
                                 round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 Plots and Performance Summary
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


# Filter weights (left panel):
#   - Negative beta values induce a misspecification at early lags k = 0 and k = 1.
#     The corresponding predictors have irregular, uninterpretable profiles
#     and are unusable in practice (red to cyan tones).
#   - Positive beta values produce an interpretable and structured predictor
#     profile: an initial AR(1)-like decay is followed by a discontinuity
#     that progressively removes weight from higher lags (blue to violet tones).
#   - As beta increases further, the weights assigned to higher lags
#     eventually turn negative (violet tones), reflecting the growing
#     tension between the imposed `beta kick' to the CCF at the shortest lags
#     and the structural exponential decay enforced by the DGP at lags k > 12.
#
# CCFs (right panel):
#   - Negative initial `beta kicks' (red to cyan tones) are misspecified,
#     pushing the target correlation CCF(h) toward zero and eventually below
#     it, rendering the corresponding predictors unusable.
#   - Increasing positive beta values shift the CCF peak progressively to
#     the right of the MSE benchmark, generating genuine look-ahead behavior
#     in the PCS designs (blue to violet tones).
#   - Imposing a single constraint leaves more room for maximizing the target 
#     correlation CCF(h), which is positive for all selected positive beta. 
#   - The loss relative to the MSE benchmark (which maximises CCF at k = 12)
#     is small to moderate, depending on the extent of the peak shift.
#     In other words, as we shift the CCF peak rightwards, the degradation in
#     performance relative to the MSE-based predictor remains modest—this reflects
#     a favourable trade-off between look-ahead gains and predictive accuracy.

# ─────────────────────────────────────────────────────────────────────
# 2.5 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 2.5.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 2.5.2 Plot
#----------------------------------------------------------------------

# Select the PCS designs with positive slope beta>=0
colnames(y_out_mat) 
select_pcs<-15:ncol(y_out_mat)
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
# horizon h) as beta increases, confirming that the single Type IV PCS
# constraint successfully advances the predictor. 
par(mfrow = c(2, 2))
select_vec<-c(2,16,18,20)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ", colnames(y_out_mat)[i]))
}

# Out of curiosity, we now examine the exact closed-form solution when ALL
# degrees of freedom of the constraint system (rank 13) are fully exhausted.
# To this end, we impose a Type I PCS with 13 constraints, leaving no room
# for optimisation (beyond eventually the sign of the solution in this example).



# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 — Exhausting the Constraint System: Impose 13 Constraints
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 3.1 PCS Set-Up: Hyperparameter Setting
# ─────────────────────────────────────────────────────────────────────────────

# DGP specification as in Exercise 1: yearly growth, i.e., the convolution of
# the ARMA(1,1) Wold decomposition with the equally-weighted MA(12) target filter.
gamma_pcs <- gamma

# Novelty: impose 13 constraints, fully exhausting the rank-13 constraint
# system. With no degrees of freedom left for optimisation, the solution is
# uniquely determined (up to sign) by the constraints alone.
Delta <- 1:(h+1)

# Automatically generate a grid of interesting beta values via PCS_func().
beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- FALSE

PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                    Type_III, scaled_constraints, high_resolution)

# Use the automatically generated grid, scaled down to focus on central points.
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec           <- beta_vec_automatic / 2


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 Closed-Form PCS Based on PCS_closed_form_func()
# ─────────────────────────────────────────────────────────────────────────────

b_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)
  
  b      <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat  <- cbind(b_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly; residuals should be zero up to machine precision.
  # Note: the problem is challenging and the magnitude of the deviations are 
  # fairly larger than in exercise 1.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("Closed-form PCS, beta=",
                                 round(beta_vec, 7)))


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
abline(v = h+1 + 1, lty = 2)   # lag h
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
rownames(ccf_mat)<-paste("CCF at lag: ",-max_lag-1+1:nrow(ccf_mat),sep="")

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
abline(v = max_lag + 1 + h + 1,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Filter weights (left panel):
#   - All designs are unusable: imposing a linearly increasing CCF forces
#     the filter to assign increasing weight to increasingly distant past
#     observations (time reversion), yielding uninterpretable
#     coefficient profiles.

# CCFs (right panel):
#   - The CCFs are nearly vanishing. To understand this, recall that
#       CCF(k) = b' * gamma_k / (||b|| * ||gamma||),
#     where ||gamma|| * ||b|| is the product of the standard deviations of
#     the process and the predictor. Rearranging the constraint gives
#       beta = b' * (gamma_k - gamma_{k-1}) = (CCF(k) - CCF(k-1)) * ||b|| * ||gamma||.
#     When beta/||b|| is small, CCF(k) is small too. Inspect the ratios 
#     beta / ||b|| explicitly:
beta_vec / sqrt(apply(filter_mat[, 4:ncol(filter_mat)]^2, 2, sum))

# The ratios are fixed (all degrees of freedom are exhausted by the 13 
# constraints) and small. Only the sign of the solution can vary, not its 
# magnitude.

# Rescale the CCF plot to the actual range to make the near-vanishing CCFs
# visible and confirm their structure.
par(mfrow = c(1, 1))
par(mfrow=c(1,1))
mplot <- ccf_mat
plot(mplot[, 1],
     main = "Population CCFs: PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec[1],lty=lwd_vec[1],
     ylim = c(min(ccf_mat[,ncol(ccf_mat)]), -min(ccf_mat[,ncol(ccf_mat)])))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h + 1,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# The rescaled plot confirms that a linearly increasing or decreasing CCF
# slope is technically feasible up to k = 13. However, the scale of the CCF is entirely
# fixed by the constraints, and only the sign can be `optimised', leaving
# no genuine degree of freedom for improving look-ahead performance.

# Confirm the fixed linear increase (or decrease) of the CCF for all PCS designs:
ccf_diff<-ccf_mat[2:(h+2),]-ccf_mat[1:(h+1),]
rownames(ccf_diff)<-paste("CCF(",1:13,")-CCF(",0:12,")",sep="")
ccf_diff


