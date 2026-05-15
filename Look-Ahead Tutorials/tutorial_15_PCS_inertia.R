# ═══════════════════════════════════════════════════════════════════
# TUTORIAL 15 — EXPLOITING CCF INERTIA
# ═══════════════════════════════════════════════════════════════════

# This tutorial reexamines the forecasting problem in Tutorial 12 and provides 
# a novel approach towards resolving the look ahead problem. The forecast 
# problem is challenging 
# 1. The MSE-optimal predictor is "stuck" at a horizon h̃ < h: the cross-correlation
# function (CCF) of the predictor cannot be pushed further toward the target
# horizon h beyond h̃ 
# 2. Attempts of displacing the peak of the CCF further to the right of the MSE predictor 
#   partly failed because the PCS (Type I) constraints conflicted with the internal structure
#   of the DGP.
# We here briefly replicate exercises 1 and 2 of Tutorial 12, analyze the look ahead problem 
# and identify a feature of the DGP, that we call "inertia of the CCF". This inertia 
# renders the original PCS Type I problem infeasible. 
# In this Tutorial we then exploit CCF inertia to recover look ahead behaviour of the predictor.

# We here assume knowledge of Tutorial 12 but briefly summarize PCS Type I) approach:




#   TYPE I — Monotonically Increasing CCF over {0, …, h}  [Most Restrictive]
#
#       The CCF must be strictly increasing across the full lag interval:
#           CCF(k) - CCF(k-1) = beta_k > 0  for all k = 1, …, h.
#       See Wildi (2026), Section 3.2 and Appendix E.
#       This condition is generally not exactly achievable (see Exercise 1).
#


# ── CONSTRAINT SUMMARY ────────────────────────────────────────────────────────
#
#   Type I:   CCF(k) > CCF(k-1)  for k = 1, …, h  (h constraints)
#



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
#   4. Reducing the number of constraints is not universally effective: the
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
# We adopt the framework from Tutorials 12: an ARMA(1,1) model fitted to
# the monthly PAYEMS employment indicator. We then consider forecasting of yearly 
# growth one year ahead.



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

# Implications PCS type I with h = 12:
#     Strong regularization causes the 12-dimensional constraint system to
#     absorb all but one degree of freedom. Only one degree of freedom remains
#     to maximize the target correlation, severely limiting look-ahead potential.




# ─────────────────────────────────────────────────────────────────────────────
# 1.6 Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

Delta  <- 1:h


beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- T

PCS_obj  <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                     Type_III, scaled_constraints, high_resolution)

# We can sweep over either the manually constructed grid or the automatically
# generated one. Here we use the automatic grid as the base, and augment it
# with additional slope values at which the predictor changes
# profile sharply (identified from prior inspection of the solution path).
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec_automatic <- PCS_obj$beta_vec

# Refine locally grid resolution
beta_vec <- c(beta_vec_automatic[1],
              -8e-05, -4e-05, -3e-05, -2e-05,
              beta_vec_automatic[2:length(beta_vec_automatic)])

# ─────────────────────────────────────────────────────────────────────────────
# 1.7 Run Closed-Form PCS
# ─────────────────────────────────────────────────────────────────────────────


b_closed_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta, lambda)
  
  b             <- PCS_obj$b
  d_delta       <- PCS_obj$d_delta
  b_closed_mat  <- cbind(b_closed_mat, b)
  
  # Constraint check: for a full-rank system the closed-form solution satisfies
  # all constraints exactly, so residuals should be zero (up to machine
  # precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_closed_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))

# Assemble all filters (nowcast, MSE references, and closed-form PCS variants)
# into a single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_closed_mat)
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

# Filter weights (left panel):
#   - The rank of the constraint system is 13, in principle allowing up to 13
#     linearly independent constraints to be imposed simultaneously. However,
#     the constant-slope Type I constraints strongly conflict with the DGP
#     structure and are therefore misspecified. As a result, the filter
#     profiles are irregular and difficult to interpret.
#
# CCFs (right panel):
#   - A monotonically increasing CCF (positive beta) is only achievable
#     through sign inversion of the filter, causing the target correlation
#     CCF(h) to turn negative and rendering the predictor unusable.
#   - The DGP structure fundamentally conflicts with a linearly increasing
#     CCF constraint: the Type I problem is therefore misspecified for this
#     ARMA(1,1) DGP.
#   - Due to the ARMA(1,1) structure, the CCF decays exponentially beyond
#     the forecast horizon: for k > 0,
#           CCF(12 + k) = a1^k * CCF(12),
#     irrespective of the choice of predictor b. This exponential decay is
#     an intrinsic property of the DGP and cannot be overcome by any linear
#     filter, fundamentally limiting the achievable rightward shift of the
#     CCF peak.


# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 —  CCF Inertia
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
#   right, generating look-ahead behavior beyond the MSE benchmark, whose
#   peak is structurally bounded at MSE(12).

# While the rigid DGP structure renders a classic Type I constraint
# ineffective, this very structure can be exploited to turn an apparent
# disadvantage into an advantage: by imposing a single well-chosen constraint
# and leveraging CCF inertia, genuine and effective look-ahead behavior can
# still be achieved.

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
beta_vec           <- beta_vec_automatic/2




# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Closed-Form PCS Based on PCS_closed_form_func()
# ─────────────────────────────────────────────────────────────────────────────

b_closed_mat <- NULL   # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the exact closed-form Type I PCS predictor.
  PCS_obj <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta, lambda)
  
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
# 2.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────────────

# ── Check 1: PCS Slope Constraints ───────────────────────────────────────────
# Validated in the loop above: residuals are zero by construction for the
# closed-form solution.

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# The Predictor is much more robust to trend inversion: much larger beta values 
# are required to eventually drive the predictor into trend/level inversion:
apply(b_closed_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# In contrast to exercise 1, the target correlations remain positive for all 
# selected beta >  0 (no misspecification).
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

# Filter weights (left panel):
#   - Negative beta values induce a misspecification at lags k = 0 and k = 1.
#     The corresponding predictors have irregular, uninterpretable profiles
#     and are unusable in practice (red to green tones).
#   - Positive beta values produce an interpretable and structured predictor
#     profile: an initial AR(1)-like decay is followed by a discontinuity
#     that progressively removes weight from higher lags (cyan to violet tones).
#   - As beta increases further, the weights assigned to higher lags
#     eventually turn negative (blue to violet tones), reflecting the growing
#     tension between the imposed beta kick to the CCF at the shortest lags
#     and the structural exponential decay enforced by the DGP at lags k > 12.
#
# CCFs (right panel):
#   - Negative initial beta kicks (red to orange tones) are misspecified,
#     pushing the target correlation CCF(h) toward zero and eventually below
#     it, rendering the corresponding predictors unusable.
#   - Increasing positive beta values shift the CCF peak progressively to
#     the right of the MSE benchmark, generating genuine look-ahead behavior
#     in the PCS designs.

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
# horizon h) as beta increases, confirming that the single Type III PCS
# constraint successfully advances the predictor. 
par(mfrow = c(2, 2))
select_vec<-c(2,15,17,19)
for (i in select_vec) {
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ", colnames(y_out_mat)[i]))
}







# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 — Exhausting the Constraint System: Impose 13 Constraints
# ════════════════════════════════════════════════════════════════════════════════
