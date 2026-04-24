# ════════════════════════════════════════════════════════════════════
# TUTORIAL 7 — MSE-DFP APPLIED TO ARMA
# PART 4: Exploiting Hidden Structure in a Difficult Forecast Framework
# ════════════════════════════════════════════════════════════════════

# Overview:
# This tutorial applies the DFP procedure introduced in Tutorial 6 to an
# ARMA process designed specifically  for presenting inherent forecasting 
# challenges. 

# Key findings:
#
#   1. FORECAST HORIZON EFFECT:
#      Increasing the forecast horizon (from h = 3 to h = 20) does NOT
#      meaningfully improve the look-ahead behaviour of MSE(20).
#         - The MSE predictor remains strongly tied to x_t regardless of h:
#           the cross-correlation at lag 0 stays high, confirming that the
#           predictor cannot disengage from the current observation: MSE is 
#           "stuck at present".
#         - The primary observable effect of a longer horizon is zero-shrinkage
#           of the predictor coefficients — a signal of growing forecast
#           uncertainty as h increases.
#      Taken together, both findings are symptomatic of a difficult forecast
#      problem in which the MSE predictor is unable to achieve genuine
#      look-ahead behaviour.
#
#   2. DECOUPLING EFFECT:
#      Imposing the DFP constraint (decoupling the predictor from x_t at the
#      present time point) induces significant zero-shrinkage in the DFP
#      predictor coefficients, reflecting the cost of enforcing independence
#      from the current observation. This is a further indication of the
#      inherent difficulty of the forecast problem.
#
#   3. CROSS-CORRELATION CONTROL:
#      Achieving a substantial reduction in the cross-correlation function
#      (CCF) at lag 0 — i.e., enforcing decoupling — requires a
#      strong decrease of the DFP constraint parameter alpha0. This
#      sensitivity is another sign of the complexity of the forecast problem.

# ─────────────────────────────────────────────────────────────────────

# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines:
#   provides mse_dfp_from_alpha0_func() and related solvers
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility:
#   measures lead/lag timing via zero-crossing analysis
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions:
#   helpers for amplitude, time-shift, and CCF computations
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)


# ════════════════════════════════════════════════════════════════════
# Exercise 1: MSE-DFP Applied to ARMA(3,2)
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 1.1 Process Specification
# ─────────────────────────────────────────────────────────────────────

# AR coefficients of the ARMA(3,2) process
ar1 <- 0.4
ar2 <- 0.3
ar3 <- 0.2

# MA coefficients of the ARMA(3,2) process
b1 <- 0.5
b2 <- 0.4

# Filter length (number of lags used in the predictor)
L <- 50

# Forecast horizon — kept the same as in Tutorial 6 for comparability
h <- 3

# Compute the Wold (MA-infinity) representation coefficients via ARMAtoMA();
# prepend 1 for the contemporaneous term (lag 0)
xi <- c(1, ARMAtoMA(
  ar      = c(ar1, ar2, ar3),
  ma      = c(b1, b2),
  lag.max = 100
))

# Plot the Wold decomposition to inspect the impulse-response decay
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition")

# Examine the roots of the AR characteristic polynomial:
#   - Arg(...)/pi gives the cycle length in years (period of oscillation)
#   - abs(...)    gives the modulus; values < 1 confirm stationarity
1 / (Arg(polyroot(c(-ar3, -ar2, -ar1, 1))) / pi)
abs(polyroot(c(-ar3, -ar2, -ar1, 1)))

# The ACF is slowly monotonically decaying: a clear indication 
# of the "stuck at present" problem.
ts.plot(ARMAacf(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=20),main="ACF",
        ylab="",xlab="Lag")

# Work with the MA (Wold) form of the predictors throughout;
# note: the equivalent AR form is derived in exercise 1.4.
gamma <- xi

# Extract L-length coefficient vectors for the nowcast and the h-step forecast
gamma0  <- gamma[1:L]          # nowcast filter  (lag 0 to L-1)
gammah  <- gamma[h + (1:L)]    # h-step MSE forecast filter

# Define a longer reference horizon for comparison purposes (to illustrate 
# "stuck at present" problem).
htilde       <- 20
gammahtilde  <- gamma[htilde + (1:L)]   # htilde-step MSE forecast filter

# Maximum cross-correlation lag (0 = contemporaneous only)
max_lag <- 0

# ── Reference CCF for the h-step MSE predictor ────────────────────────
# Compute the cross-correlation of the h-step MSE predictor output with x_t;
# store for later comparison against DFP designs
cor_vec          <- compute_acf_at_lags_zero_delta_func(max_lag, h, gammah, gamma0)$cor_vec
cor_vec_mat_mse1  <- cor_vec
# Retain CCF at lag 0 (contemporaneous) and at lag h (target horizon)
cor_vec_mse1      <- c(cor_vec[1], cor_vec[1 + h])

# ── Reference CCF for the htilde-step MSE predictor ───────────────────
# Same computation for the longer-horizon MSE predictor
cor_vec          <- compute_acf_at_lags_zero_delta_func(max_lag, h, gammahtilde, gamma0)$cor_vec
cor_vec_mat_mse  <- cbind(cor_vec_mat_mse1, cor_vec)
cor_vec_mse      <- rbind(cor_vec_mse1, c(cor_vec[1], cor_vec[1 + h]))


# ─────────────────────────────────────────────────────────────────────
# 1.2 MSE-DFP Computation
# ─────────────────────────────────────────────────────────────────────

# Compute the unconstrained covariance between the h-step MSE predictor
# and the nowcast; this serves as the baseline (100%) for alpha0 scaling. 
# Decoupling means that alpha0 < alpha0_mse in the DFP constraint.
alpha0_mse <- as.double(gammah %*% gamma0)

# Define a sequence of alpha0 values as decreasing fractions of alpha0_mse:
#   70% → 45% → 22% → 10% → 0% (fully decoupled)
# Smaller alpha0 enforces stronger decoupling from the present observation x_t
alpha0_vec <- round(c(0.7, 0.45, 0.22, 0.1, 0) * alpha0_mse, 2)

# Initialise storage matrices for filter coefficients and CCF summaries
max_lag        <- 0
b_mat          <- b_mat_unscaled <- a_mat <- lambda_vec2 <- NULL
cor_vec_mat_1  <- cor_vec_1 <- NULL

# Loop over alpha0 values and compute the corresponding MSE-DFP filter
for (i in 1:length(alpha0_vec))   
{
  alpha0 <- alpha0_vec[i]
  
  # Solve for the DFP filter b0 and Lagrange multiplier lambda0 (weight on 
  # gamma0) at this alpha0
  dfp_obj <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0)
  
  b0      <- dfp_obj$b
  lambda0 <- dfp_obj$lambda
  
  # Accumulate filter coefficient columns
  b_mat <- cbind(b_mat, b0)
  
  # Compute the CCF of the current DFP predictor output with x_t
  cor_vec       <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, b_mat[, ncol(b_mat)], gamma0)$cor_vec
  cor_vec_mat_1 <- cbind(cor_vec_mat_1, cor_vec)
  
  # Retain CCF at lag 0 and lag h for the summary table
  cor_vec_1 <- rbind(cor_vec_1, c(cor_vec[1], cor_vec[1 + h]))
}

# ── Assemble full filter and CCF summary tables ────────────────────────
# Label DFP columns by their alpha0 value
colnames(b_mat) <- paste("alpha0=", alpha0_vec, sep = "")

# Combine reference MSE filters with all DFP filters into one matrix
filter_mat <- cbind(gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c(
  paste("MSE(", h,      ")", sep = ""),
  paste("MSE(", htilde, ")", sep = ""),
  colnames(b_mat)
)

# Build a summary table: CCF at lag 0 and at the target horizon h
cor_vec_2 <- rbind(cor_vec_mse, cor_vec_1)
colnames(cor_vec_2) <- c("Lag 0", paste("h=", h, sep = ""))
rownames(cor_vec_2) <- colnames(filter_mat)

# Full CCF matrix across all lags for all designs
cor_vec_mat <- cbind(cor_vec_mat_mse, cor_vec_mat_1)
colnames(cor_vec_mat) <- colnames(filter_mat)


# ─────────────────────────────────────────────────────────────────────
# 1.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# CHECK 1 — DFP constraint satisfaction:
#   The inner product b' * gamma0 should equal alpha0 for each design;
#   residuals below numerical tolerance confirm the constraint is met.
#   Note: b_mat and gamma0 are not normalised, so alpha0_vec is a raw
#   covariance (not a correlation coefficient).
t(b_mat) %*% gamma0 - alpha0_vec

# CHECK 2 — Sign/orientation preservation:
#   The sum of filter coefficients indicates how the predictor responds to
#   a unit-level shift in the data. A positive sum means the predicted
#   linear trend retains its direction (no orientation inversion).
#   All DFP designs except the fully decoupled one (alpha0 = 0) should
#   preserve sign; see Section 4.1 of Wildi (2026) for theoretical details.
apply(b_mat, 2, sum)

par(mfrow = c(2, 1))

# ── Illustration: trend orientation under fully decoupled DFP ─────────
trend          <- 1:100
forecast_trend <- NULL

# Fully decoupled DFP (last column of b_mat)
b <- b_mat[, ncol(b_mat)]
for (i in L:100)
  forecast_trend[i] <- b %*% trend[i:(i - L + 1)]

# Expected result: the predicted trend is INVERTED (negative slope)
ts.plot(forecast_trend,
        main = "Fully decoupled DFP inverts trend orientation",
        ylab = "")

# ── Illustration: trend orientation under a partial DFP ───────────────
# Choose column k (1 < k < ncol(b_mat)); clamp to valid range
k <- 3
k <- min(ncol(b_mat) - 1, k)
k <- max(1, k)
b <- b_mat[, k]
for (i in L:100)
  forecast_trend[i] <- b %*% trend[i:(i - L + 1)]

# Expected result: the predicted trend retains the correct (positive) slope
ts.plot(forecast_trend,
        main = "NON fully decoupled DFP does NOT invert trend orientation",
        ylab = "")

# ── Illustration: mean-level sign under fully decoupled DFP ───────────
par(mfrow = c(2, 1))

mu          <- rep(1, 100)   # constant unit-mean series
forecast_mu <- NULL

# Fully decoupled DFP: the predicted mean becomes negative
b_fd <- b_mat[, ncol(b_mat)]
for (i in L:100)
  forecast_mu[i] <- b_fd %*% mu[i:(i - L + 1)]

ts.plot(forecast_mu,
        main = "Fully decoupled DFP changes sign of constant level",
        ylab = "",
        ylim = c(1.1 * forecast_mu[100], 0))
abline(h = 0, lty = 2)

# Partial DFP: the predicted mean remains positive (sign is preserved)
k <- 3
k <- min(ncol(b_mat) - 1, k)
k <- max(1, k)
b <- b_mat[, k]
for (i in L:100)
  forecast_mu[i] <- b %*% mu[i:(i - L + 1)]

ts.plot(forecast_mu,
        main = "NON fully decoupled DFP does NOT change sign of constant level",
        ylab = "",
        ylim = c(0, 1.1 * forecast_mu[100]))
abline(h = 0, lty = 2)


# ── Discussion ────────────────────────────────────────────────────────
# Preserving trend direction and mean sign is a natural and often desirable
# criterion in practice — but it is not universally appropriate.
#
# Counter-example: if the process is periodic with period `per`, the optimal
# predictor at forecast horizon h = per/2 is exactly out-of-phase with the
# signal and therefore changes sign. In such cases, sign inversion is not a
# defect but a correct feature of the optimal design. However, the ARMA(3,2) 
# here is dominated by an aperiodic `near unit-root'. 



# ─────────────────────────────────────────────────────────────────────
# 1.4 AR Form of the Predictors
# ─────────────────────────────────────────────────────────────────────

# ── 1.4.1 Compute the AR Inversion ───────────────────────────────────

# Convert the MA (Wold) representation of the ARMA(3,2) process into its
# equivalent AR representation by inverting the MA polynomial.
# ARMAtoMA() is repurposed here: swapping AR and MA arguments performs
# the inversion, yielding the AR filter coefficients up to lag L-1.
ar_inv <- -ARMAtoMA(ar = -c(b1, b2), ma = -c(ar1, ar2, ar3), lag.max = L - 1)

# Prepend 1 to form the full AR filter (including the lag-0 coefficient)
theta <- c(1, -ar_inv)

# Verify correctness via the MA-AR duality identity:
#   convolving the AR filter (theta) with the Wold MA coefficients (xi)
#   must recover the identity filter — i.e., 1 followed by zeros.
#   Any deviation from {1, 0, 0, ...} indicates an inversion error.
# Increasing L renders the error arbitrarily small.
conv_two_filt_func(xi, theta)$conv[1:10]


# ── Convert MA-form predictors to their AR equivalents ────────────────

# a. AR form of the h-step MSE predictor
filt2        <- gammah
ar_mse_arma32 <- conv_two_filt_func(theta, filt2)$conv

# b. AR form of each DFP predictor (one column per alpha0 value)
ar_dfp_arma32_mat <- NULL
for (i in 1:length(alpha0_vec))
{
  # Convolve the AR inversion filter with the i-th DFP MA filter
  filt2             <- b_mat[, i]
  ar_dfp_arma32_mat <- cbind(ar_dfp_arma32_mat,
                             conv_two_filt_func(theta, filt2)$conv)
}


# ── 1.4.2 Plot: Compare MA and AR Forms ──────────────────────────────

par(mfrow = c(1, 1))

# Plot the first 10 AR coefficients for all predictors (MSE + DFP designs)
ts.plot(cbind(ar_mse_arma32, ar_dfp_arma32_mat)[1:10, ],
        col  = rainbow(length(alpha0_vec) + 1),
        main = "MSE and DFP predictors: AR form (first 10 coefficients)")

# Redraw the MSE predictor in green for visual reference
lines(ar_mse_arma32[1:10], lwd = 1, col = "green")

# DFP affects the first weight of the predictor in AR form, see tutorial 6. 
# and Wildi (2026), end of section 3.1. 


# ── Numerical verification: MA form vs. AR form outputs agree ─────────
# Simulate an ARMA(3,2) realisation and confirm that applying the MA-form
# and AR-form of the same DFP filter to the corresponding input series
# yields virtually identical outputs. Any residual difference is due to
# the finite truncation length L of the MA/AR inversion.

set.seed(1)
len <- 1000
x   <- eps <- rnorm(len)

for (i in 4:len)
  x[i] <- ar1*x[i-1] + ar2*x[i-2] + ar3*x[i-3] +
  eps[i] + b1*eps[i-1] + b2*eps[i-2]

y_dfp_ar32 <- y_dfp_ma32 <- rep(NA, len)

# Select DFP design index k for the comparison
k <- 2
for (i in L:len)
{
  # MA form: applied to the innovation sequence eps
  y_dfp_ma32[i] <- b_mat[, k]             %*% eps[i:(i - L + 1)]
  # AR form: applied to the observed series x
  y_dfp_ar32[i] <- ar_dfp_arma32_mat[1:L, k] %*% x[i:(i - L + 1)]
}

# Visual comparison: both series should be nearly indistinguishable 
# (increasing L renders the approximation error arbitrarily small).
ts.plot(cbind(y_dfp_ma32, y_dfp_ar32)[1:200, ],
        main = "MA-form vs. AR-form DFP output (should overlap)")



# ─────────────────────────────────────────────────────────────────────
# 1.5 DFP Predictors: Coefficient Weights and CCF
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo <- c("green", "darkgreen", "brown", "orange", "blue", "violet", "red")

mplot <- filter_mat

# Left panel: filter coefficient weights for all designs
ts.plot(mplot,
        main = "ARMA(3,2) — Predictor Weights",
        col  = colo, xlab = "", ylab = "")
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], line = -i, col = colo[i])
abline(h = 0)

# Right panel: CCF of each predictor output with x_t
# Scale by the ratio of filter norms to obtain a unit-free correlation measure
mplot <- cor_vec_mat[1:22, ] *
  as.double(sqrt(gamma0 %*% gamma0) / sqrt(gamma %*% gamma))

plot(mplot[, 1], axes = F, type = "l",
     xlab = "", ylab = "", main = "CCF",
     col = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
# Solid vertical line marks lag 0 (contemporaneous coupling)
abline(v = max_lag + 1,     lty = 1)
# Dashed vertical line marks the target forecast horizon h
abline(v = max_lag + 1 + h, lty = 2)
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:(nrow(mplot)))
axis(2)
box()


# ── Outcomes ──────────────────────────────────────────────────────────
#
# COEFFICIENT WEIGHTS (left panel):
#   - Enforcing decoupling (decreasing alpha0) generates progressively
#     stronger zero-shrinkage: the sum of squared coefficients falls markedly
#     as alpha0 approaches zero.
apply(filter_mat^2, 2, sum)
#
#   - Because the overall scale of the DFP shrinks, alpha0 (the covariance
#     between the DFP and the nowcast gamma0) must be decreased substantially
#     to produce a meaningful reduction of the CCF at lag 0. In other words,
#     tighter decoupling demands an increasingly strict alpha0 bound to
#     compensate for the simultaneous zero-shrinkage effect.
#
#   - As alpha0 decreases the smooth, regular shape of the MSE filter (green)
#     becomes progressively more unsmooth and ragged.
#
#   - Increased decoupling (look-ahead behaviour) emphasises features of the 
#     data-generating process that are obscured/ignored by the unconstrained 
#     MSE predictors.
#
# CCF (right panel):
#   - The CCF at lag 0 remains close to 1 (strong contemporaneous coupling)
#     even after substantial reductions of alpha0. This persistence arises
#     because alpha0 is a scale-dependent covariance: strong zero-shrinkage
#     of the DFP offsets the reduction in alpha0, leaving the normalised
#     correlation largely unchanged.
#   - With the exception of the fully decoupled DFP (alpha0 = 0), all other
#     DFP designs achieve only modest decoupling (reductions of the lag-0 CCF), 
#     confirming the inherent difficulty of decoupling from x_t for this
#     process.
#   - alpha0 must be reduced very aggressively to produce a noticeable decrease
#     in the lag-0 CCF — a further indicator of forecast difficulty. 
#     Specifically, shrinking alpha0 from 6.18 to 0.88 decreases the CCF from 
#     0.9977 to 0.9012, as shown in the CCF table below:
round(cor_vec_2, 4)
#   - As alpha0 decreases, the drop in lag-0 CCF spills over to the target
#     horizon h = 3. The DFP criterion minimises this collateral loss: 
#     for a given lag-zero CCF, no other linear predictor can outperform 
#     the DFP CCF at horizon h (optimality).


# ─────────────────────────────────────────────────────────────────────
# 1.6 Verification:
#     Empirical Performances Converge Towards Expected (True) Values
# ─────────────────────────────────────────────────────────────────────
# Apply each filter to a long ARMA(3,2) realisation and compute empirical
# CCF values at lag 0 and at the target horizon h. With len = 100 000 the
# empirical estimates should be very close to the theoretical values in
# cor_vec_2, confirming the analytical derivations above.

len     <- 100000
set.seed(932)

x <- eps <- rnorm(len)
for (i in 4:len)
  x[i] <- ar1*x[i-1] + ar2*x[i-2] + ar3*x[i-3] +
  eps[i] + b1*eps[i-1] + b2*eps[i-2]

y_out_mat <- NULL
perf_mat  <- matrix(ncol = ncol(filter_mat), nrow = 2)
colnames(perf_mat) <- colnames(filter_mat)
rownames(perf_mat) <- c("Lag 0", paste("h=", h, sep = ""))

for (i in 1:ncol(filter_mat))
{
  # Apply the i-th filter (in MA form) to the innovation sequence
  y <- filter(eps, filter_mat[, i], side = 1)
  y_out_mat <- cbind(y_out_mat, y)
  
  # Empirical correlation at lag 0: cor(y_t, x_t)
  perf_mat[1, i] <- cor(y[L:len],       x[L:len])
  # Empirical correlation at lag h: cor(y_t, x_{t+h})
  perf_mat[2, i] <- cor(y[L:(len - h)], x[(h + L):len])
}
colnames(y_out_mat) <- colnames(filter_mat)

# Empirical CCF summary — compare row by row with the theoretical values
t(perf_mat)

# Theoretical CCF for reference: empirical CCFs converge to these values.
cor_vec_2

# ─────────────────────────────────────────────────────────────────────
# 1.7 Look-Ahead Behaviour
# ─────────────────────────────────────────────────────────────────────

# Define the sample window to plot (indices into the simulated series)
anf <- 650
enf <- 750

# ── Step 1: Compare MSE(h) and MSE(htilde) against the target ─────────
# Shift the data left by h so that x_{t+h} aligns visually with the
# predictor output y_t, making lead/lag relationships directly visible.
select_filters <- 1:2   # columns 1-2 of y_out_mat: MSE(h) and MSE(htilde)

mplot <- cbind(x[(h + 1):len],
               y_out_mat[1:(len - h), select_filters])[anf:enf, ]
colnames(mplot) <- c("Data", colnames(y_out_mat)[select_filters])

par(mfrow = c(1, 1))
coli <- c("black", colo)

ts.plot(mplot, col = coli,
        main = "Target (left-shifted by h), MSE(h) and MSE(htilde)")
lines(mplot[, 2], col = "green", lty = 2, lwd = 2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], line = -i, col = coli[i])

# Observation: the dominant difference between MSE(h) and MSE(htilde) is
# scale — MSE(htilde) has a smaller variance due to stronger zero-shrinkage.
# No meaningful gain in look-ahead timing is visible.


# ── Step 2: Standardise to isolate timing differences ─────────────────
# Remove the scale effect by standardising each series to unit variance,
# so that any remaining difference reflects purely look-ahead behaviour.
mplot <- scale(cbind(x[(h + 1):len],
                     y_out_mat[1:(len - h), select_filters])[anf:enf, ])
colnames(mplot) <- c("Data", colnames(y_out_mat)[select_filters])

par(mfrow = c(1, 1))
coli <- c("black", colo)

ts.plot(mplot, col = coli,
        main = "Standardised: Target, MSE(h) and MSE(htilde)")
lines(mplot[, 2], col = "green", lty = 2, lwd = 2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], line = -i, col = coli[i])

# Outcome: after standardisation, MSE(htilde) offers no timing advantage
# over MSE(h). Increasing the forecast horizon in the MSE framework does
# NOT unlock genuine look-ahead behaviour — confirming that this is a
# difficult forecast problem.


# ── Step 3: Add DFP designs ───────────────────────────────────────────
# Include all DFP predictors except the fully decoupled design (last
# column), which inverts trend orientation and is therefore not directly
# comparable on a common scale.
select_filters <- 1:(ncol(y_out_mat) - 1)

mplot <- scale(cbind(x[(h + 1):len],
                     y_out_mat[1:(len - h), select_filters])[anf:enf, ])
colnames(mplot) <- c(paste("Data left-shifted by h =", h),
                     colnames(y_out_mat)[select_filters])

par(mfrow = c(1, 1))
coli <- c("black", colo)

ts.plot(mplot, col = coli,
        main = "Standardised: Target, MSE and DFP Predictors")
lines(mplot[, 2], col = "green", lty = 2, lwd = 2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], line = -i, col = coli[i])


# ── Step 4: Magnifying glass around a turning point ───────────────────
# Zoom into a short window that contains a local turning point to assess
# whether the DFP predictors anticipate the direction change earlier than
# the MSE predictor.
# Note: the black data line is left-shifted by h = 3, so a predictor that
# tracks the black line closely exhibits true h-step look-ahead behaviour.
anf <- 710
enf <- 720

select_filters <- 1:(ncol(y_out_mat) - 1)

mplot <- scale(cbind(x[(h + 1):len],
                     y_out_mat[1:(len - h), select_filters])[anf:enf, ])
colnames(mplot) <- c(paste("Data left-shifted by h =", h),
                     colnames(y_out_mat)[select_filters])
par(mfrow = c(1, 1))
coli <- c("black", colo)

ts.plot(mplot, col = coli, main = "Magnifying glass: turning-point region")
lines(mplot[, 2], col = "green", lty = 2, lwd = 2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], line = -i, col = coli[i])

# The more strongly decoupled DFP are left-shifted, anticipating the 
# lower turning point in the left-shifted data (black). 


# ════════════════════════════════════════════════════════════════════
# Exercise 2: Interpretability — Time-Shift DFP Constraint
# ════════════════════════════════════════════════════════════════════
# The MSE-DFP constraint is formulated in terms of a covariance (alpha0),
# which is scale-dependent: the same alpha0 value has different meanings
# for processes with different variances, making cross-process comparisons
# difficult to interpret.
#
# Two scale-invariant alternatives:
#   - Unitary DFP:   reformulates the constraint directly in terms of the
#                     cross-correlation at lag 0, which is always bounded
#                     in [-1, 1] and has a natural, scale-free interpretation
#                     as a measure of contemporaneous coupling, see tutorial 4.
#               Note: the unitary DFP is quadratic, leading to two distinct 
#               DFP solutions, one lagging, the other leading.
#   - Time-shift DFP: reformulates alpha0 as a function of the time-shift at 
#                     frequency zero, see tutorial 6. Like the correlation, 
#                     the time-shift is scale-free.


# ════════════════════════════════════════════════════════════════════
# Main Take-Aways
# ════════════════════════════════════════════════════════════════════
#
#   1. DIFFICULT FORECAST PROBLEM:
#      The ARMA(3,2) process studied here is inherently difficult to forecast.
#      Neither increasing the forecast horizon (MSE(htilde) vs. MSE(h)) nor
#      applying the unconstrained MSE predictor achieves meaningful look-ahead
#      behaviour: the predictor remains strongly coupled to x_t at lag 0.
#
#   2. DFP INDUCES ZERO-SHRINKAGE:
#      Imposing the decoupling constraint drives strong zero-shrinkage of the
#      DFP coefficients. This shrinkage is itself diagnostic — it quantifies
#      how much information must be sacrificed in order to reduce contemporaneous
#      coupling, and serves as a direct measure of forecast difficulty.
#
#   3. AGGRESSIVE alpha0 REDUCTION REQUIRED:
#      Because of the zero-shrinkage effect, alpha0 (the scale-dependent
#      covariance constraint) must be reduced very substantially before the
#      lag-0 CCF decreases noticeably. This sensitivity underscores the
#      difficulty of decoupling from x_t for this process.
#
#   4. SCALE INTERPRETABILITY:
#      The MSE-DFP criterion is scale-dependent. Scale-invariant formulations
#      (Unitary DFP, Time-shift DFP) are preferable when comparing designs
#      across processes or when a directly interpretable decoupling measure
#      is required.
























# ─────────────────────────────────────────────────────────────────────
# 1.7 Look Ahead Behaviour
# ─────────────────────────────────────────────────────────────────────


anf<-650
enf<-750

# Compare data left-shifted by h=3 with MSE(3) and MSE(20)
select_filters<-1:2
mplot<-cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,]
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])
# The main effect is scaling: MSE(20) has a smaller variance than MSE(3)

# To better evaluate the `look ahead' effect of the MSE(20) over MSE(3) we now standardize the series 
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])
# Outcome: increasing the forecast horizon in MSE does not allow to look ahead.
# The forecast problem is `difficult'.


# We now add the DFP designs. all DFP except fully decoupled (the latter inverts 
# trend orientation)

# Select all filters except fully decoupled
select_filters<-1:(ncol(y_out_mat)-1)
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c(paste("Data left-shifted by h=",h,sep=""),colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])


# Let's apply a magnifying glass at a `turning point':
# Note that the data (black line) is left shifted by h=3
anf<-710
enf<-720
# Select all filters except fully decoupled
select_filters<-1:(ncol(y_out_mat)-1)
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c(paste("Data left-shifted by h=",h,sep=""),colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Magnifying glass")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])


# ════════════════════════════════════════════════════════════════════
# Exercise 2: Interpretability (Time-Shift DFP Constraint)
# ════════════════════════════════════════════════════════════════════
# MSE-DFP is sensitive to scale.
# Unitary DFP is invariant to scale
# Time-shift is invariant to scale



# ════════════════════════════════════════════════════════════════════
# Main Take Aways
# ════════════════════════════════════════════════════════════════════
# The 





















