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

# The rank of the constraint system is 13: this allows to address a 13-dimensional 
# constraint system.
# However, the constant slope Type I constraints strongly contradict the DGP: they 
# are a misspecification.
# As a result, the predictors (left panel) have a strange uninterpretable profile 
# and positively growing CCFs (right panel) are only possible through sihn inversion, 
# so that the target correlation turns negative: CCF(h) < 0.

# The structure of DGP conflicts with a monotonically (linearly) increasing CCF. 
# However, imposing a single constraint CCF(1) - CCF(0) = beta is fairly easy (for h=1 this 
# corresponds to a Type I, II and III constraint). We can now exploit two specific structural 
# features of the DGP
# 1. For k > 12, CCF(k+1) - CCF(k) decays exponentially.
# 2. CCF(k) - CCF(k-1) cannot be too far away from CCF(k+1) - CCF(k).
# The second property is  called CCF inertia. 

# Exploiting CCF inertia:
# If beta and hence CCF(1) - CCF(0) = beta > 0, inertia 
# implies that CCF(k), k=2,...,12, must draw a solution of continuity between 
# the raising CCF, the initial beta kick at lags 0 and 1, and the exponentially 
# decreasing CCF profile at lags k > 12. Inertia, means that the transition between 
# lag 0 and lag 13 must be smooth, i.e., the CCF will have a peak at a lag between 
# 0 and 13. Increasing beta makes the start steeper, thus producing a right shift 
# of the peak CCF.
# 




# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 — Exploiting CCF Inertia
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

Delta  <- 1

beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- F

PCS_obj  <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                     Type_III, scaled_constraints, high_resolution)

# We can sweep over either the manually constructed grid or the automatically
# generated one. Here we use the automatic grid as the base, and augment it
# with additional slope values at which the predictor changes
# profile sharply (identified from prior inspection of the solution path).
beta_vec_automatic <- PCS_obj$beta_vec
beta_vec <- beta_vec_automatic





# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Closed-Form PCS Based on PCS_closed_form_func()
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


# ════════════════════════════════════════════════════════════════════════════════
# EXERCISE 4 — Exhausting the Constraint System: Impose 13 Constraints
# ════════════════════════════════════════════════════════════════════════════════
