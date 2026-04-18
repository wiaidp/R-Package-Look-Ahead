# ════════════════════════════════════════════════════════════════════
# TUTORIAL 3 — THE DFP PREDICTOR
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
# No other predictor can look as far ahead as the DFP predictor for a given 
# level of tracking accuracy, see Wildi 2026, sections 3.5 and 4.3.

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
#   forecasting. The DFP framework is therefore applied to differenced data
#   without material loss of generality for practical applications.

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
#   - The framework is modular: swap the target signal z_t to design
#     predictors tailored to cycles, trends, or custom band-pass signals.
#   - It is possible to formulate a specialized leading indicator DFP.
# ════════════════════════════════════════════════════════════════════

#---------------------------------------------------------------------
# SHORT NOTE  ON EMPIRICAL EXAMPLES IN THIS TUTORIAL:
#---------------------------------------------------------------------
# All examples illustrate challenging forecast problems in which the DGP is
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

# ════════════════════════════════════════════════════════════════════
# TUTORIAL 3 — THE DFP PREDICTOR
# Exercise 1: Unit-Length DFP
# ════════════════════════════════════════════════════════════════════

# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines
# Provides DFP_compute_lambda_alpha0_func() and related solvers
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, CCF helpers)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))


# ════════════════════════════════════════════════════════════════════
# CONCEPTUAL BACKGROUND
# ════════════════════════════════════════════════════════════════════
# Let x_t be a stationary time series with a convergent (square-summable)
# Wold decomposition:
#
#   x_t = sum_{k=0}^{inf} gamma_k * epsilon_{t-k}
#
# It is natural to express the DFP predictor in terms of the INNOVATIONS
# epsilon_t rather than in terms of x_t directly.  Exercise ??? will extend 
# this to the observable x_t representation.
#
# Two important notes on the MA / Wold representation:
#   (1) In practice, x_t is observed and the innovations are LATENT;
#       they must be recovered by applying an AR-inversion filter to x_t
#       (requiring stationarity of the AR or ARMA for a convergent Wold decomposition).
#       Here epsilon_t is simulated directly, so no inversion is needed.
#   (2) The DFP framework does NOT require the MA sequence to be invertible
#       (minimum-phase). The MA representation can be non-invertible.
# ════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
# 1.1 Data-Generating Process (DGP) 
# ─────────────────────────────────────────────────────────────────────
# We reuse the MA(9) process from Tutorial 1:
#   x_t = sum_{k=0}^{9} gamma_k * epsilon_{t-k},   gamma_k = 0.9^k
#
# The Wold decomposition (gamma0) plays a dual role here:
#   - It defines the DGP (how x_t is generated from innovations).
#   - It serves as the NOWCAST filter: applying gamma0 to the innovation
#     sequence recovers x_t itself (the present value).

L      <- 10          # filter length (truncation of the Wold decomposition)
a1     <- 0.9         # geometric decay rate of the Wold coefficients
gamma0 <- a1^(0:9)    # Wold coefficients: gamma_k = a1^k, k = 0,...,9

ts.plot(gamma0, main = "Wold decomposition (MA coefficients of the DGP)",
        xlab = "Lag k", ylab = expression(gamma[k]))

# Simulate one realisation of the MA(9) process
len    <- 100
set.seed(21)
eps    <- rnorm(len)               # i.i.d. standard normal innovations
x      <- filter(eps, gamma0)      # MA(9) output: x_t = gamma0 * epsilon_t

par(mfrow = c(2, 1))
ts.plot(x,   main = "One realisation of the MA(9) process", xlab = "")
acf(na.exclude(x), main = "ACF — significant lags up to order q = 9 in long samples")


# ─────────────────────────────────────────────────────────────────────
# 1.2 MSE Predictor 
# ─────────────────────────────────────────────────────────────────────
# Forecast horizon (must satisfy h <= q; otherwise the optimal forecast is 0)
h <- 5

# The h-step-ahead MSE predictor retains only the MA coefficients at lags
# h through q (future shocks are replaced by their conditional mean of zero):
#
#   x̂_{t|t+h} = gamma_h * eps_t + gamma_{h+1} * eps_{t-1} + ... + gamma_q * eps_{t+h-q}
#
# In filter form, the MSE weights are gamma0 shifted forward by h positions:
gammah <- c(gamma0[(h + 1):L], rep(0, h))   # MSE filter (length L, last h entries = 0)

# ── Nomenclature used throughout this tutorial ────────────────────────
# gamma0  : Wold decomposition = NOWCAST filter (recovers x_t from innovations)
# gammah  : h-step-ahead MSE predictor filter
# b0      : DFP predictor filter (to be computed below)
#
# In this simple example the nowcast is trivial (gamma0 applied to eps gives x_t).
# In later exercises the TARGET will be a non-causal signal z_t (e.g., an
# HP-filter trend), and the nowcast will be its causal approximation — a
# non-trivial filter in its own right.

# Measure coupling of MSE predictor to the present (nowcast correlation):
# A value close to 1 signals strong "stuck-at-present" behaviour.
cor_mse_now <- t(gamma0) %*% gammah / sqrt(t(gamma0 %*% gamma0) * t(gammah %*% gammah))
cat("Correlation of MSE predictor with nowcast (present):", round(cor_mse_now, 4), "\n")


# ─────────────────────────────────────────────────────────────────────
# 1.3 DFP Predictor 
# ─────────────────────────────────────────────────────────────────────
# Assumption: gamma0 and gammah are not collinear

# The UNIT-LENGTH DFP predictor b0 is a constrained linear combination of
# the nowcast (gamma0) and the MSE predictor (gammah):
#
#   b0 = lambda2 * gamma0 + lambda1 * gammah,   normalised to ||b0|| = 1
#
# The unit-norm constraint gives the hyperparameter alpha0 a direct
# interpretation: it equals the prescribed correlation between the predictor
# y_t(h) and the current value x_t (the degree of coupling with the present).
# Setting alpha0 < cor_mse_now decouples the predictor from the present.
# See Wildi (2026), Section 3.1 for the closed-form derivation.

alpha0  <- 0.2   # target correlation with the nowcast (present coupling level)
# must satisfy alpha0 < cor_mse_now for decoupling

b0_obj  <- DFP_compute_lambda_alpha0_func(gamma0, gammah, h, L, alpha0)

b0        <- b0_obj$b0          # DFP filter coefficients
lambda1   <- b0_obj$lambda1     # weight on the MSE predictor
lambda2   <- b0_obj$lambda2     # weight on the nowcast

# Plot MSE and DFP filter coefficients side by side
colo  <- c("green", "blue")
mplot <- cbind(gammah, b0)
colnames(mplot) <- c("MSE", "DFP")

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Lag", ylab = "Coefficient",
     main = "Filter coefficients: MSE predictor vs. DFP predictor",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# Observation:
# Unlike the MSE predictor (whose coefficients are zero at lags >= q+1-h),
# the DFP predictor assigns NON-ZERO weight to observations further in the 
# past. 
# Intuitively, this weighting scheme seems to contradict a look-ahead 
# perspective.


# ─────────────────────────────────────────────────────────────────────
# 1.4 Verification Checks 
# ─────────────────────────────────────────────────────────────────────
# Confirm that the computed DFP solution satisfies all theoretical properties.

# Check 1: b0 is a normalised linear combination of gamma0 and gammah
b <- lambda2 * gamma0 + lambda1 * gammah
b <- b / as.double(sqrt(t(b) %*% b))   # enforce unit norm
cat("Check 1 — max deviation from linear combination formula:",
    max(abs(b - b0)), "\n")             # should be (near) zero

# Check 2: Unit-norm constraint (||b0|| = 1)
cat("Check 2 — unit-norm residual:", t(b0) %*% b0 - 1, "\n")   # should be zero

# Check 3: Decoupling constraint (correlation of b0 with gamma0 equals alpha0)
cor_dfp_now <- t(gamma0) %*% b0 / sqrt(t(gamma0 %*% gamma0))
cat("Check 3 — decoupling constraint residual:",
    cor_dfp_now - alpha0, "\n")         # should be zero


# ─────────────────────────────────────────────────────────────────────
# 1.5 Accuracy–Timeliness (AT) Dilemma 
# ─────────────────────────────────────────────────────────────────────
# The decoupling constraint is not free: by reducing correlation with x_t,
# the DFP predictor also sacrifices some correlation with the future target
# x_{t+h}. This is the AT dilemma in action.
# DFP minimises this accuracy loss subject to the prescribed lead constraint.

# A) Correlation with the nowcast (present coupling)
cat("\nCorrelation with nowcast — MSE:", round(cor_mse_now, 4),
    "| DFP:", round(cor_dfp_now, 4), "\n")

# B) Correlation with the effective target x_{t+h}
# (represented by the MSE filter gammah in the innovation space)
target_cor_dfp <- t(b0)    %*% gammah / sqrt((t(b0)    %*% b0)    * (t(gamma0) %*% gamma0))
target_cor_mse <- t(gammah) %*% gammah / sqrt((t(gamma0) %*% gamma0) * (t(gammah) %*% gammah))

cat("Target correlation  — MSE:", round(target_cor_mse, 4),
    "| DFP:", round(target_cor_dfp, 4), "\n")

# C) Correlation with the MSE predictor itself (alternative target representation)
cat("Corr. with MSE filt — MSE:", 1,
    "| DFP:", round(t(b0) %*% gammah / sqrt((t(b0) %*% b0) * (t(gammah) %*% gammah)), 4), "\n")

# Summary table of the AT trade-off
perf_mat <- rbind(
  c(cor_mse_now,   cor_dfp_now),
  c(target_cor_mse, target_cor_dfp)
)
colnames(perf_mat) <- c("MSE", "DFP")
rownames(perf_mat) <- c("Correlation with nowcast (present coupling)",
                        "Target correlation (horizon h)")

cat("\nAccuracy–Timeliness performance summary:\n")
print(round(perf_mat, 3))
# Interpretation:
#   Row 1: DFP achieves the prescribed lower coupling with the present (alpha0).
#   Row 2: The accompanying reduction in target correlation is the AT cost —
#          minimised by DFP's optimal weighting of gamma0 and gammah.


# ─────────────────────────────────────────────────────────────────────
# 1.6 Apply Predictors to the Simulated Data 
# ─────────────────────────────────────────────────────────────────────
# Filter the innovation sequence with each predictor to obtain time series
# of h-step-ahead forecasts, then compare visually.

y_dfp <- filter(eps, b0,    side = 1)   # DFP forecast series
y_mse <- filter(eps, gammah, side = 1)  # MSE forecast series

# Scale for comparability (unit variance) and remove NA burn-in
mplot <- na.exclude(scale(cbind(y_mse, y_dfp)))
colnames(mplot) <- c("MSE", "DFP")

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Time", ylab = "",
     main = "Standardised forecast series: MSE (green) vs. DFP (blue)",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# The DFP (blue) seems left-shifted when compared to the MSE predictor (green).
# We now verify this assertion.


# ─────────────────────────────────────────────────────────────────────
# 1.7 Sample CCF 
# ─────────────────────────────────────────────────────────────────────
# Estimate the cross-correlation between x and each predictor from the
# simulated data. The CCF peak location indicates how far ahead (or behind)
# each predictor is relative to x_t.
# Note: sample CCF is noisy; the true (population) CCF is computed in 1.8.

sample_ccf_dfp <- ccf(na.exclude(x), na.exclude(y_dfp), lag.max = 10,
                      plot = FALSE)$acf
sample_ccf_mse <- ccf(na.exclude(x), na.exclude(y_mse), lag.max = 10,
                      plot = FALSE)$acf

mplot <- cbind(sample_ccf_mse, sample_ccf_dfp)
colnames(mplot) <- c("MSE", "DFP")
rownames(mplot)  <- c(-(L:1), 0:L)

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Lag (−) and Lead (+)", ylab = "CCF",
     main = "Sample CCF of x with MSE (green) and DFP (blue) predictors",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(v = which(rownames(mplot) == "0"),   lty = 2)              # k = 0 (present)
abline(v = which(rownames(mplot) == as.character(h)), col = "blue", lty = 1) # k = h (target horizon)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# Interpretation:
# The black dashed line marks k = 0 (present); the blue solid line marks k = h.
# A peak at the dashed vertical line indicates "stuck to present" problem.
# A CCF peak to the RIGHT of k = 0 indicates that the predictor effectively 
# leads x_t.

# We now verify the above right-shift (lead) of the DFP by computing sample 
# independent true CCFs.

# ─────────────────────────────────────────────────────────────────────
# 1.8 True (Population) CCF
# ─────────────────────────────────────────────────────────────────────
# The population CCF is computed analytically from the filter coefficients,
# avoiding the noise inherent in sample-based estimates.
# This provides a clean, sample-independent comparison of the two predictors.

ccf_dfp <- compute_ccf_func(b0,    gamma0)   # true CCF of DFP predictor with x
ccf_mse <- compute_ccf_func(gammah, gamma0)  # true CCF of MSE predictor with x

mplot <- cbind(ccf_mse, ccf_dfp)
colnames(mplot) <- c("MSE", "DFP")
rownames(mplot)  <- names(ccf_mse)

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Lag (−) and Lead (+)", ylab = "CCF",
     main = "True (population) CCF: MSE (green) vs. DFP (blue) predictor",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(v = which(rownames(mplot) == "0"),   lty = 2)              # k = 0 (present)
abline(v = which(rownames(mplot) == as.character(h)), col = "blue", lty = 1) # k = h (horizon)
abline(h=alpha0,lty=2)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# ──────────────────────────────────────────────────────────────────────────
# REMARKABLY, THE PEAK OF THE CCF SHIFTED FROM k=0 (PRESENT) to k=h (FUTURE)
# ──────────────────────────────────────────────────────────────────────────
# Interpretation:
# The true CCF strenghtens the previous findings:
#   - The MSE predictor's CCF peaks near k = 0: "stuck at present".
#   - The DFP predictor's CCF peak is shifted toward k = h: genuine look-ahead.
#   - The DFP constraint pulls the CCF down to alpha0 at lag k=0 (intersection of 
#     vertical and horizontal black lines). 
#   - The DFP optimization principle minimzes the loss in CCF at k=h (blue vertical line). 
# This connects the DFP to the AT-dilemma, tracing the resulting efficient frontier.


# ─────────────────────────────────────────────────────────────────────
# 1.9 Amplitude and Time-Shift 
# ─────────────────────────────────────────────────────────────────────
# For a linear filter applied to a sinusoid at frequency omega, the output is
# again a sinusoid at the same frequency, but scaled by the filter's AMPLITUDE
# (gain) and delayed by the filter's TIME-SHIFT (phase delay).
#
# Plotting both functions across all frequencies [0, π] allows a direct
# comparison of how each filter distorts the magnitude and timing of signals
# at every frequency — a more informative picture than a single CCF lag.

K      <- 600      # number of frequency grid points spanning [0, π]
plot_T <- FALSE    # suppress internal plotting; custom plots are built below

# Compute amplitude and time-shift for each filter across the frequency grid
as_obj1 <- amp_shift_func(K, b0, plot_T)   # equally-weighted MA
as_obj2 <- amp_shift_func(K, gammah, plot_T)   # exponentially-weighted MA

par(mfrow = c(2, 1))

# ── Amplitude plot ────────────────────────────────────────────────────────────
# Amplitude (gain) at each frequency: values close to 1 mean the filter passes
# that frequency with little attenuation; values near 0 indicate suppression.
colos <- c("blue", "green")
mplot <- cbind(as_obj1$amp, as_obj2$amp)
colnames(mplot) <- c("DFP", "MSE")

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Amplitude (Gain) Function",
     ylim = c(min(mplot), max(mplot)), col = colos[1])
mtext(colnames(mplot)[1], line = -1, col = colos[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colos[i])
  mtext(colnames(mplot)[i], col = colos[i], line = -i)
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ── Time-shift plot ────────────────────────────────────────────────────────────
# Time-shift : we truncate the shift of DFP at values below -3  
# (leads larger than 3 are set to NA). 
mplot <- cbind(as_obj1$shift, as_obj2$shift)
mplot[which(mplot[,1]<(-3)),1]<-NA
colnames(mplot) <- c("DFP", "MSE")

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Time-Shift Function",
     ylim = c(min(na.exclude(mplot)), 4), col = colos[1])
#mtext(colnames(mplot)[1], line = -1, col = colos[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colos[i])
  # Legend labels omitted here to avoid overlap; colours identify each filter
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# The DFP predictor has a smaller time-shift than the MSE predictor toward
# the low-frequency spectral mass of the MA(9) process. While the time-shift
# varies by frequency, the above true CCF provides an exact aggregate lead measure
# (across all frequency components): the left-shifted CCF peak of the DFP
# predictor confirms an effective aggregate lead of h = 5 time points
# relative to the MSE predictor.
#
# The smaller time-shift of DFP is achieved through a deformation of the
# amplitude function toward lower frequencies: the trade-off between
# amplitude and time-shift is an instance of the ATS-trilemma in MDFA
# (Accuracy–Timeliness–Smoothness). See the MDFA tutorial for background.


# ─────────────────────────────────────────────────────────────────────
# 1.10 Geometry of the Unit-Length DFP Predictor
# ─────────────────────────────────────────────────────────────────────
# This figure (reproduced from Wildi 2026) depicts the geometry of the 
# unit-length DFP solution in the plane spanned by gamma0 (nowcast) and 
# gammah (MSE predictor).

# Assumption: gamma0 and gammah are not collinear

# The solution is determined by two constraints:
#   - Unit-length constraint: b0 lies on the unit sphere (blue arc).
#   - DFP constraint: b0 lies on the cone with axis gamma0 and semi-angle
#     arccos(alpha0), i.e. the prescribed correlation with the nowcast
#     (red lines).
#
# The intersection of the cone with the unit sphere in this plane yields
# TWO candidate solutions, corresponding to the two branches of the cone:
#   - Solution 1 (shown): lambda1 > 0, lambda2 < 0 — upper branch.
#   - Solution 2 (not shown): lambda1 < 0, lambda2 > 0 — lower branch.
#
# The two solutions arise because the DFP constraint
#   alpha0 = b0 %*% gamma0 = ||gamma0|| * ||b0|| * cos(theta_{0b})
# depends only on cos(theta_{0b}) and is therefore indifferent to the
# SIGN of the angle theta_{0b} between b0 and gamma0.
#
# The sign of theta_{0b} determines whether the predictor leads or lags:
#   - theta_{0b} > theta_{0h}: b0 is rotated past gammah away from gamma0
#                               → DFP predictor leads x_t  (desired).
#   - theta_{0b} < 0:          b0 lies on the opposite side of gamma0
#                               → DFP predictor lags x_t (undesired).
# The first solution is therefore selected as the operative DFP predictor.

# Intuitively, decoupling b0 from gamma0 requires a negative weight on
# gamma0 (lambda2 < 0): we subtract the nowcast direction from the MSE
# predictor, thereby removing the "present-anchoring" component of gammah
# and retaining only its forward-looking portion.

# Note: for illustration, gamma0 and gammah are drawn as vectors in a
# two-dimensional plane. In general, both vectors live in L-dimensional
# space (where L is the filter length), and the relevant geometry —
# the unit sphere, the cone, and their intersection — is defined within
# the 2-dimensional subspace spanned by gamma0 and gammah in R^L.

vx <- 3
vy <- 2
# Specify gamma0 (gamma0_plot)and gammah (gammah_plot)
gamma0_plot<-c(3,0.5)*3/3.5
gammah_plot<-c(1.5,1)*2/1.5
# Specify lambda0
lambda0<-0.3
# Lengths
l0<-sqrt(sum(gamma0_plot^2))
lh<-sqrt(sum(gammah_plot^2))

# Angle between gammah_plot and gamma0_plot: gammah_plot is above (larger angle)
theta_h <- atan2(gammah_plot[2], gammah_plot[1])-atan2(gamma0_plot[2], gamma0_plot[1])

# Set up plot limits with some padding
x_min<--0.5
x_max<-3
y_min<--0.5
y_max<-1.5
lim <- 1.2 * max(1, abs(c(vx, vy))+0.5)
plot(NA, xlim = c(x_min,x_max), ylim = c(y_min, y_max),
     asp = 1, xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gamma0_plot
arrows(0, 0,gamma0_plot[1],gamma0_plot[2], length = 0.12, lwd=1, col = "black")
text(gamma0_plot[1]+0.1,gamma0_plot[2], labels = expression(gamma[0]), col = "black", 
     cex = 1.2)
# gammah_plot
arrows(0, 0,gammah_plot[1],gammah_plot[2], length = 0.12, lwd=1, col = "black")
text(gammah_plot[1]+0.1,gammah_plot[2], labels = expression(gamma[h]), col = "black", 
     cex = 1.2)
# Insert unit length b0
b0<-c(gammah_plot[1]-lambda0*gamma0_plot[1],gammah_plot[2]-lambda0*gamma0_plot[2])
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.5,b0[2]/lb0+0.1, labels = expression(b==lambda[1]*gamma[h]+
                    lambda[2]*gamma[0]), col = "red", cex = 1.2)
segments(0,0,1.5*(gammah_plot[1]-lambda0*gamma0_plot[1]),
         1.5*(gammah_plot[2]-lambda0*gamma0_plot[2]),  lwd = 1,lty=2, col = "red")

text(b0[1]/lb0,b0[2]/lb0+0.5, "Intersection of cone with plane", 
     col = "red", cex = 1)

# Draw the angle theta_h (between gammah_plot and gamma0_plot)
r <- 0.25 * lh  # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, theta_h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+theta_h / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(theta[0*h]), col = "black", cex = 1.2)
# Draw the angle theta (between b0 and gamma0_plot)
theta <- atan2(b0[2], b0[1])-atan2(gamma0_plot[2], gamma0_plot[1])

r <- 1  # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "blue", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+theta / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(+theta[0*b]), col = "blue", cex = 1.2)

# Draw arrow from gammah_plot to b0
final_b0<-c(r * cos(th_seq[length(th_seq)]), r * sin(th_seq[length(th_seq)]))
lamb<-0.176
from_gammah_plot<-(final_b0+lamb*gamma0_plot)
arrows(from_gammah_plot[1],from_gammah_plot[2],final_b0[1],final_b0[2], length = 0.12, 
       lwd=1, col = "black")
text(from_gammah_plot[1],from_gammah_plot[2]+0.1, labels = expression(lambda[1]*gamma[h]), 
     col = "black", cex = 1.2)
# Draw second intersection of cone at -theta
angle_from_x_axis<--(th_seq[length(th_seq)]-2*atan2(gamma0_plot[2],gamma0_plot[1]))
segments(0,0,1.5*cos(angle_from_x_axis),1.5*sin(angle_from_x_axis),  lwd = 1,
         lty=2, col = "red")

# Draw the angle -theta 
theta <- atan2(b0[2], b0[1])-atan2(gamma0_plot[2], gamma0_plot[1])

r <- 1 # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, -theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "blue", lwd=1,lty=2)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])-theta / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(-theta[0*b]), col = "blue", cex = 1.2)

text(1.15 * r * cos(th_mid)+0.9, 1.15 * r * sin(th_mid)-0.4,
     "Intersection of cone with plane", col = "red", cex = 1)

text(2.,0.05,"Unit sphere (intersection with plane)", col = "blue", cex = 1)




# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: MSE-DFP 
# ════════════════════════════════════════════════════════════════════
# This exercise introduces three novelties relative to Exercise 1:

# Assumption: gamma0 and gammah are not collinear

#
# A) THE MSE-DFP CRITERION (Equation 9, Wildi 2026)
#    Like the unit-length DFP, the MSE-DFP predictor lies in the plane
#    spanned by gamma0 (nowcast) and gammah (MSE predictor), but with
#    the weight on gammah fixed at one:
#
#      b0 = gammah + lambda * gamma0
#
#    Only lambda remains to be determined, making the optimisation LINEAR
#    with a unique closed-form solution (Proposition 1, Wildi 2026).
#    By emphasising the MSE objective (rather than target correlation as
#    in Equation 2), the unit-norm constraint can be dropped entirely.
#
#    Advantages over unit-length DFP in exercise 1:
#      - Reduces to a linear problem with a single globally optimal
#        solution.
#      - The predictor is naturally scaled to minimise MSE rather than
#        constrained to unit length.
#    Disadvantage:
#      - The hyperparameter alpha0 can no longer be interpreted directly
#        as the correlation between the DFP predictor and the nowcast.
#        Its value depends on the scale of gamma0 (and hence gammah).
#      - Except for alpha0=0 (complete decoupling) this might challenge 
#        sensible a priori selection of the DFP constraint parameter.
#      - See however exercise 4 below for an alternative selection criterion.
#
# B) AR FORM OF THE PREDICTORS
#    In addition to the MA form, we derive the AR form of each predictor.
#    The AR form is the natural representation when the predictor is
#    applied to the observed series x_t rather than to the innovation
#    sequence eps_t. We verify that both forms yield identical outputs.
#
# C) A MORE PERSISTENT DGP: AR(3)
#    Exercise 1 used an MA(9) with a finite Wold decomposition. Here we
#    switch to an AR(3), whose Wold decomposition is infinite and must be
#    truncated at length L for practical computation. This exercises the
#    MA-inversion step and tests DFP on a richer, AR-class of processes.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 2.1 Data-Generating Process: AR(3) 
# ─────────────────────────────────────────────────────────────────────
# Construct an AR(3) by specifying its three characteristic roots,
# then recover the AR parameters from Vieta's formulas.
# All roots are real and inside the unit circle → stationary process.

lambda1 <- 0.3
lambda2 <- 0.8
lambda3 <- 0.2

# AR coefficients derived from roots via Vieta's formulas
ar1 <- ar11 <-  lambda1 + lambda2 + lambda3
ar2 <- ar21 <- -(lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3)
ar3 <- ar31 <-  lambda1 * lambda2 * lambda3

# Verify: roots of the characteristic polynomial should recover lambda1/2/3
polyroot(c(-ar3, -ar2, -ar1, 1))

# Wold decomposition: infinite MA representation of the AR(3)
# The ACF-based plot gives a visual check of the decay rate
par(mfrow = c(1, 1))
ts.plot(ARMAacf(ar = c(ar1, ar2, ar3), lag.max = 100),
        main = "Wold decomposition of AR(3) — MA coefficients",
        xlab = "Lag", ylab = expression(gamma[k]))

# Compute the Wold coefficients (truncated MA representation)
# We need more than L coefficients to form MSE forecasts at horizon h
gamma <- c(1, ARMAtoMA(ar = c(ar1, ar2, ar3), lag.max = 1000))

# Sanity check: inverting gamma back to AR should recover ar1, ar2, ar3
# (first three entries match; all subsequent entries are numerically zero)
ts.plot(-ARMAtoMA(ar = -gamma[2:L], lag.max = 40),
        main = "AR inversion check — first three entries should match ar1/ar2/ar3")


# ─────────────────────────────────────────────────────────────────────
# 2.2 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# As in Exercise 1, predictors are first derived in MA (innovation) form.
# Exercise 3 will translate these to the observable AR (data) form.

h <- 5    # forecast horizon
L <- 50   # filter length (truncation of the infinite Wold decomposition)

# Nowcast and MSE predictor filters (truncated to length L)
gamma0 <- gamma01 <- gamma[1:L]         # nowcast: Wold coefficients at lags 0,...,L-1
gammah <- gammah1 <- gamma[h + (1:L)]   # MSE predictor: Wold coefficients shifted by h

# Plot both filters for reference
colo  <- c("green", "black")
mplot <- cbind(gammah, gamma0)
colnames(mplot) <- c(paste0("MSE predictor (h=", h, ")"), "Nowcast")

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Lag", ylab = "Coefficient",
     main = "MA-form filter coefficients: MSE predictor vs. nowcast",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# Compute the population CCF of the MSE predictor as a baseline reference
max_lag    <- 0
cor_vec_mat <- compute_ccf_func(gammah, gamma0)[L:(2 * L - 1)]

# Grid of alpha0 values to sweep the AT frontier
# Note: alpha0 is NOT a correlation here (no unit-norm constraint);
# it is the raw inner product b0 %*% gamma0, controlling decoupling strength
alpha0_vec <- c(0.9, 0.45, 0.22, 0.1, 0)


# ─────────────────────────────────────────────────────────────────────
# 2.3 MSE-DFP: Sweep the AT Frontier 
# ─────────────────────────────────────────────────────────────────────
# For each alpha0, compute the MSE-DFP predictor via Proposition 1
# (Wildi 2026): b0 = gammah + lambda * gamma0, where
#   lambda = (alpha0 - gamma0 %*% gammah) / (gamma0 %*% gamma0)
# This closed-form solution minimises the MSE subject to the decoupling
# constraint b0 %*% gamma0 = alpha0.

b_mat      <- NULL          # stores filter coefficients for each alpha0
lambda_vec1 <- NULL         # stores lambda values
cor_vec_1  <- matrix(ncol = 2, nrow = length(alpha0_vec))  # CCF at lags 0 and h

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # ── Compute MSE-DFP via utility function ─────────────────────────
  b0 <- compute_mse_dfp(alpha0, gamma0, gammah)$b0
  
  # ── Alternative closed-form derivation (Proposition 1) ───────────
  # lambda scales gamma0 to enforce the decoupling constraint exactly
  lambda          <- as.double((alpha0 - t(gamma0) %*% gammah) / (t(gamma0) %*% gamma0))
  b0_alternative  <- gammah + lambda * gamma0
  # Verify both derivations agree (should be zero)
  max(abs(b0 - b0_alternative))
  
  b_mat       <- cbind(b_mat, b0)
  lambda_vec1 <- lambda
  
  # ── Compute population CCF for this predictor ────────────────────
  cor_vec <- compute_acf_at_lags_zero_delta_func(max_lag, h,
                                                 as.vector(b0), gamma0)$cor_vec
  cor_vec_mat      <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1]  <- cor_vec[1]       # CCF at lag 0  (coupling with present)
    cor_vec_1[i, 2]  <- cor_vec[1 + h]   # CCF at lag h  (coupling with target)
}

colnames(b_mat)    <- paste0("alpha0=", alpha0_vec)
colnames(cor_vec_1) <- c("Lag 0", "Lag h")

# ── Verification checks ───────────────────────────────────────────────
# Check 1: DFP constraint b0 %*% gamma0 = alpha0 should hold exactly
# (residuals should be zero for all alpha0)
t(b_mat) %*% gamma0 - alpha0_vec

# Check 2: Equivalent check via the normalised CCF
# Since cor_vec is the CCF, alpha0 must be scaled by the norms of b0 and gamma0
cor_vec_1[, 1] - alpha0_vec / sqrt(diag(t(b_mat) %*% b_mat) *
                                     as.double(t(gamma0) %*% gamma0))


# ─────────────────────────────────────────────────────────────────────
# 2.4 Plots and Performances
# ─────────────────────────────────────────────────────────────────────

# Set up a 1×2 panel layout for side-by-side plots
par(mfrow = c(1, 2))

# --- Disabled diagnostic plots (kept for reference) ---
# ts.plot(gammah1, main=paste("MSE first process: h=", h, sep=""),
#         col="green", xlab="", ylab="")
# ts.plot(gammah, main="Second process", col="green", xlab="", ylab="")

# Define a colour palette for up to 6 predictors
colo <- c("green", "brown", "orange", "blue", "violet", "red")

# Scale filter coefficients (centre=FALSE) and normalise by sqrt(L-1)
# so that all predictors are on a comparable amplitude scale for plotting
mplot <- scale(cbind(gammah, b_mat), center = F, scale = T) / sqrt(L - 1)


ts.plot(mplot,main="Unit-scaled predictors: AR(3)",col=colo,xlab="",ylab="")
mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9, ",rho,"=",0.92)),line=-2,col=colo[2])
#mtext(expression(paste("    ",alpha[0],"=0.45, ",rho,"=",0.76)),line=-3,col=colo[3])
#mtext(expression(paste("    ",alpha[0],"=0.22, ",rho,"=",0.51)),line=-4,col=colo[4])
#mtext(expression(paste("    ",alpha[0],"=0.1, ",rho,"=",0.26)),line=-5,col=colo[5])
#mtext(expression(paste("    ",alpha[0],"=0, ",rho,"=",0.0)),line=-6,col=colo[6])
mtext(expression(paste("DFP ",alpha[0],"=0.9 ")),line=-2,col=colo[2])
mtext(expression(paste("    ",alpha[0],"=0.45 ")),line=-3,col=colo[3])
mtext(expression(paste("    ",alpha[0],"=0.22 ")),line=-4,col=colo[4])
mtext(expression(paste("    ",alpha[0],"=0.1 ")),line=-5,col=colo[5])
mtext(expression(paste("    ",alpha[0],"=0 ")),line=-6,col=colo[6])
abline(h=0)


mplot<-cor_vec_mat[1:22,]*as.double(sqrt(gamma0%*%gamma0)/sqrt(gamma%*%gamma))

plot(mplot[,1],main="",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i])
}
abline(h=0)
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9")),line=-2,col=colo[2])
#mtext(expression(paste("DFP ",alpha[0],"=0.45")),line=-3,col=colo[3])
#mtext(expression(paste("DFP ",alpha[0],"=0.22")),line=-4,col=colo[4])
#mtext(expression(paste("DFP ",alpha[0],"=0.1")),line=-5,col=colo[5])
#mtext(expression(paste("DFP ",alpha[0],"=0")),line=-6,col=colo[6])
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
axis(1,at=1:nrow(mplot),labels=-max_lag-1+1:(nrow(mplot)))
axis(2)
box()


# Round and display the correlation matrix for inspection
mat_cor_vec <- round(cor_vec_1, 2)
mat_cor_vec


# ─────────────────────────────────────────────────────────────────────
# 2.5 Apply Predictors (Filters) to Data
# ─────────────────────────────────────────────────────────────────────

# Fix the random seed for reproducibility
# (set.seed(45) overridden below; set.seed(17) is the active seed)
set.seed(45)
set.seed(17)
len <- 10000

# Generate a white-noise (standard normal) input series of length `len`
x <- rnorm(len)

# Apply each DFP/MSE filter column in b_mat to x using one-sided (causal)
# convolution, and collect all filtered outputs as columns of y_out_mat.
# Each column of b_mat corresponds to one predictor (MSE or a DFP variant).
y_out_mat <- NULL
for (i in 1:ncol(b_mat))
  y_out_mat <- cbind(y_out_mat, filter(x, b_mat[, i], side = 1))


# Reset to single-panel layout and plot a representative excerpt (obs. 300–350)
# of the filtered outputs to visually compare predictor behaviours
par(mfrow = c(1, 1))
ts.plot(y_out_mat[300:350, ],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:length(alpha0_vec))
  mtext(paste("alpha0=", alpha0_vec[i], sep = ""), col = colo[i], line = -i)

# --- Interpretation of the plot ---
# The plot illustrates the practical effect of the DFP regularisation parameter alpha0:
#
#  - High-frequency, short-term noise fluctuations are unpredictable by any of the
#    filters; all predictors follow the series closely at that time-scale.
#
#  - Stronger decoupling (smaller alpha0) causes the predictor to anticipate
#    mean reversion over 'mid-term' dynamics — visible as longer connected
#    intervals where the DFP output stays above or below the zero line relative
#    to the MSE predictor.
#
#  - Note: alpha0 does not have a simple closed-form interpretation in general;
#    its meaning is transparent only at the extreme alpha0 = 0
#    (zero correlation).


# --- Remark on complete decoupling (alpha0 = 0) ---
# Complete decoupling enforces zero correlation between the predictor output
# and x_t (the most recent observation). Since x_t is conventionally regarded
# as highly informative, one might ask: does zero correlation conflict with its
# importance?
#
# The answer is no. Zero correlation does NOT imply that the filter weight
# assigned to x_t is zero. Rather, the decoupling constraint reshapes the
# overall filter dynamics so that the predictor output is structurally
# orthogonal to x_t — while x_t itself may still receive a substantial weight.
#
# Indeed, as the plot below confirms, x_t receives the LARGEST filter weight
# among all lags in the completely decoupled DFP (alpha0 = 0). The decoupling
# is therefore achieved through the interplay of all filter coefficients
# collectively, not by suppressing the weight on x_t.

ts.plot(b_mat[, ncol(b_mat)],
        main = "Complete decoupling: the largest weight is assigned to x_t",
        xlab = "Lag",
        ylab = "Completely decoupled DFP")


# ─────────────────────────────────────────────────────────────────────
# 2.6 MSE-DFP: AR Form
# ─────────────────────────────────────────────────────────────────────
# We now convert the MA-form predictors to their AR equivalents by
# convolving each predictor with the AR(3) operator. A similar proceeding
# applies to the unit-length DFP in exercise 1.

# --------------------------------------------------------------------------
# 2.6.1 Validation of the Convolution Approach
# --------------------------------------------------------------------------
# Specify the predictor matrix: MSE filter and DFP filters.
filter_mat          <- cbind(gammah, b_mat)
colnames(filter_mat) <- c("MSE", colnames(b_mat))

# Verify the approach via a known identity:
# Convolving the AR(3) operator with its Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros).
filt1 <- c(1, -ar1, -ar2, -ar3)  # AR(3) operator
filt2 <- gamma                    # Wold (MA) decomposition of the AR(3)
conv_two_filt_func(filt1, filt2)$conv[1:10]

# Having confirmed the identity, we now convolve the AR(3) operator with
# the MSE and DFP predictors (in MA form) to obtain their AR equivalents.

# --------------------------------------------------------------------------
# 2.6.2 Convolution of the AR(3) Operator with the Predictors
# --------------------------------------------------------------------------

# a. MSE predictor: convolve AR(3) operator with the MSE filter.
filt1      <- c(1, -ar1, -ar2, -ar3)
filt2      <- gammah
ar_mse_ar3 <- conv_two_filt_func(filt1, filt2)$conv

# b. DFP predictors: convolve AR(3) operator with each DFP filter.
ar_dfp_ar3_mat <- NULL
for (i in 1:ncol(filter_mat)) {
  # Use the original (unscaled) DFP predictor.
  filt2          <- filter_mat[, i]
  ar_dfp_ar3_mat <- cbind(
    ar_dfp_ar3_mat,
    conv_two_filt_func(filt1, filt2)$conv
  )
}
colnames(ar_dfp_ar3_mat)<-colnames(filter_mat)

# --------------------------------------------------------------------------
# 2.6.3 Analysis and Plot of DFP Predictors in AR Form
# --------------------------------------------------------------------------
# Key structural property of the DFP predictor in AR form:
#
# Only the FIRST AR coefficient varies across DFP designs; all higher-order
# AR coefficients are identical regardless of the chosen lambda.
#
# Theoretical justification (Wildi 2026, Section 3.1, Equation 19):
#   The DFP filter is defined as:
#     b = gammah + lambda * gamma0
#   When b is converted to AR form by inverting gamma0, the term
#   lambda * gamma0 maps to lambda * identity (a pure scalar shift).
#   Because gammah is fixed and convolution is linear, this scalar shift
#   affects only the first AR coefficient, leaving all higher-order
#   coefficients unchanged across DFP designs.

# Assign colours: green for the baseline (MSE), rainbow for DFP variants
colo <- c("green", rainbow(ncol(filter_mat) - 1))

# Plot the first 5 AR coefficients of each DFP predictor to highlight
# the structural invariance: only the first coefficient differs across designs
ts.plot(
  ar_dfp_ar3_mat[1:5, ],
  col  = colo,
  main = "Method B: DFP Predictors in AR Form"
)

# Add colour-coded in-plot labels for each predictor
for (i in 1:ncol(ar_dfp_ar3_mat))
  mtext(colnames(ar_dfp_ar3_mat)[i], col = colo[i], line = -i)


# --------------------------------------------------------------------------
# 2.6.4 Verification: Comparing MA and AR Forms
# --------------------------------------------------------------------------
# We verify that both forms produce numerically identical outputs when applied 
# to their respective inputs:
#   - AR form applied to observed data x.
#   - MA form applied to white noise innovations eps.

# Illustration: the convolution of an AR-form DFP filter with its MA-form
# counterpart does NOT yield the identity.
# Select any of the filters in filter_mat:
k     <- 4
# k cannot be larger than column dimension of filter_mat
k<-min(k,ncol(filter_mat))

# Simulate an AR(3) process to verify output equivalence.
set.seed(1)
len <- 1000
x   <- eps <- rnorm(len)
for (i in 4:len) {
  x[i] <- ar1 * x[i-1] + ar2 * x[i-2] + ar3 * x[i-3] + eps[i]
}

y_dfp_ma <- y_dfp_ar <- rep(NA, len)

# Apply the MA form to the innovations eps.
y_dfp_ma<-filter(eps,filter_mat[, k])
# Apply the AR form to the observed series x.
y_dfp_ar<-filter(x,ar_dfp_ar3_mat[, k])

# Both outputs should be numerically identical up to negligible errors
# arising from finite-length MA/AR truncation.
ts.plot(cbind(y_dfp_ma, y_dfp_ar)[1:200, ],main="AR- and MA-forms overlap exactly")

# Confirm: the maximum absolute difference is negligible.
max(na.exclude(abs(y_dfp_ma - y_dfp_ar)[1:200]))
# Note on approximation accuracy:
# Any residual discrepancy arises because the filters gamma0 and gammah are
# truncated to a finite length L. As L increases, the difference between the 
# MA and AR implementations vanishes asymptotically.

# ─────────────────────────────────────────────────────────────────────
# --- Interpretation and Remarks on Exercise 2.6 (AR-Form DFP) ---
# ─────────────────────────────────────────────────────────────────────
#
# 1. Structural simplicity of the DFP in AR form:
#    The entire DFP effect is absorbed into a single AR
#    coefficient — the weight on the most recent observation x_t. This makes
#    the AR form compact and seemingly easy to interpret.
#
#    Interpretability — Part I (AR-form perspective):
#      - Reducing the weight on x_t lowers the correlation between the
#        predictor output and x_t, directly controlling the degree of
#        decoupling.
#      - Complete decoupling (alpha0 = 0) corresponds to the design where
#        this correlation is driven to zero. Importantly, zero correlation
#        does NOT imply that the weight on x_t vanishes — it is the
#        interplay of all filter coefficients that achieves orthogonality.
#
# 2. The DFP as a Partial 'Black Box':
#    Despite the clean AR-form structure, the DFP constraint does not directly
#    reveal its look-ahead behaviour. While alpha0 controls the degree of
#    decoupling, it is not immediately obvious how decoupling translates into
#    an interpretable lead of the predictor (over the MSE benchmark). In fact, 
#    one of the two solutions of the unit-length DFP (exercise 1) is 
#    effectively lagging.
#
#    To make this precise: a useful predictor must satisfy two conditions:
#      - NECESSARY condition:  the predictor must be decoupled from the
#                              current data x_t (low correlation with x_t),
#                              so that it is not merely echoing the present, 
#                              see tutorial 1.
#      - SUFFICIENT condition: the predictor must genuinely anticipate future
#                              values, i.e. it must lead the MSE predictor by a
#                              meaningful and quantifiable amount.
#
#    The DFP constraint directly addresses the necessary condition (decoupling)
#    via alpha0, but leaves the sufficient condition (actual lead behaviour)
#    implicit. It is not guaranteed that a given alpha0 produces a predictor
#    that leads the target by any specific or desirable amount. 
#
#    Exercise 3 below partly resolves this gap by re-parameterising alpha0 in terms
#    of a pre-specified lead at the trend frequency (frequency zero), thereby
#    making the sufficient condition explicit — at least for the trend component.
#
# 3. Interpretability — Part II: DFP vs. PCS (Tutorial 4):
#    Even with the frequency-zero re-parameterisation of Exercise 3, the DFP
#    concept remains less directly `look ahead' interpretable than the Peak 
#    Correlation Shifting (PCS) predictor introduced in Tutorial 4. The key 
#    distinction is the scope of the `lead' constraint:
#
#      - PCS addresses the lead of the predictor in an AGGREGATE sense,
#        not only at the trend frequency omega = 0. This makes the PCS 
#        constraint directly meaningful in terms of the predictor's overall 
#        timing behaviour (the `sufficiency' part).
#
#      - As a consequence of this specific `look ahead' constraint, the PCS 
#        effect generally spreads across all AR lags rather than being
#        confined to the first coefficient as in the above DFP. The filter 
#        (predictor) structure is therefore more complex.
#
#    In summary, the two look ahead approaches present an 
#    interpretability-complexity trade-off:
#
#      DFP — simpler filter/predictor structure (only the first AR coefficient 
#            is affected); seemingly easy to interpret (decrease weight on x_t),
#            but the resulting look-ahead effect is less straightforward to 
#            control.
#
#      PCS — more complex filter/predictor structure (all coefficients of the 
#            AR form are affected), but the look-ahead effect is directly 
#            interpretable as a classic (aggregate/overall) lead. 


# ─────────────────────────────────────────────────────────────────────
# 2.7 Geometry of the MSE-DFP Predictor
# ─────────────────────────────────────────────────────────────────────
# The following figure (reproduced from Wildi 2026) illustrates the geometry of
# the MSE-DFP solution in the plane spanned by gamma0 (nowcast filter)
# and gammah (h-step MSE predictor).
#
# Assumption: gamma0 and gammah are linearly independent (not collinear);
# see Wildi (2026) for the degenerate case.
#
# --- Geometric simplification relative to the unit-length DFP ---
#
# The MSE-DFP is geometrically simpler than the unit-length DFP of Exercise 1:
#
#   Unit-length DFP (Exercise 1):
#     Two constraints are active simultaneously — the decoupling constraint
#     AND the unit-length constraint. The DFP lies on the intersection of the 
#     plane spanned by gammah and gamm0, a cone with axis gamma0 (nowcast), 
#     and the unit sphere, see exercise 1.10. 
#
#   MSE-DFP (this exercise):
#     Dropping the unit-length constraint removes the quadratic term entirely.
#     Only the linear decoupling constraint remains, defining an affine
#     hyperplane in R^L. The feasible set is flat (a hyperplane), and the
#     optimisation reduces to a standard orthogonal projection problem.
#
# --- Optimality and closed-form solution ---
#
# By the least-squares optimality principle, the MSE-DFP solution b0 is the
# orthogonal projection of gammah onto the affine hyperplane defined by:
#
#   gamma0' * b = alpha0      (decoupling constraint)
#
# This projection has the closed-form expression (Proposition 1, Wildi 2026):
#
#   b0 = gammah + lambda * gamma0
#
# where the weight on gammah is exactly one (gammah is the unconstrained
# optimum), and lambda is the unique scalar satisfying:
#
#   gamma0' * b0(lambda) = alpha0
#
# Because this equation is linear in lambda, the solution is unique and
# given in closed form.
#
# --- Intuition for the sign of lambda ---
#
# Intuition: decoupling b0 from gamma0 requires a negative weight on
# gamma0 (lambda < 0). We subtract the nowcast direction from the MSE
# predictor, thereby removing its 'present-anchoring' component and
# retaining only the forward-looking portion.
#
# Note on dimensionality:
# For illustration purposes, gamma0 and gammah in the figure below are drawn
# as vectors in a two-dimensional plane. In general, both vectors live in R^L,
# where L is the filter length. This is without loss of generality.


par(mfrow=c(1,1))
# Specify gamma0 and gammah.

gamma0_plot<-c(3,0.5)*3.5/4
gammah_plot<-c(1.5,1)*2/1.5

# Specify lambda0
lambda0<-0.5
# Lengths
l0<-sqrt(sum(gamma0_plot^2))
lh<-sqrt(sum(gammah_plot^2))

# Angle between gammah_plot and gamma0_plot: gammah_plot is above (larger angle)
beta0h <- atan2(gammah_plot[2], gammah_plot[1])-atan2(gamma0_plot[2], 
                                                      gamma0_plot[1])
# Angle between gammah_plot and b
beta <- atan2(gammah_plot[2]-lambda0*gamma0_plot[2], gammah_plot[1]-
                lambda0*gamma0_plot[1])-atan2(gammah_plot[2], gammah_plot[1])

# Set up plot limits with some padding
x_min<-min(0,min(c(gamma0_plot[1],gammah_plot[1]))-1)
x_max<-max(c(gamma0_plot[1],gammah_plot[1]))+0.5
y_min<-min(c(gamma0_plot[2],gammah_plot[2]))-1
y_min<-0
y_max<-max(c(gamma0_plot[2],gammah_plot[2]))+1
y_max<-max(c(gamma0_plot[2],gammah_plot[2]))+0.2
plot(NA, xlim = c(x_min,x_max), ylim = c(y_min, y_max),
     asp=1.5,xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gamma0_plot
arrows(0, 0,gamma0_plot[1],gamma0_plot[2], length = 0.12, lwd=1, col = "black")
text(gamma0_plot[1]+0.1,gamma0_plot[2], labels = expression(gamma[0]), 
     col = "black", cex = 1.2)
# gammah_plot
arrows(0, 0,gammah_plot[1],gammah_plot[2], length = 0.12, lwd=1, col = "red")
text(gammah_plot[1]+0.1,gammah_plot[2], labels = expression(gamma[h]), 
     col = "black", cex = 1.2)
# gammah_plot-lambda0*gamma0_plot
arrows(gammah_plot[1],gammah_plot[2],gammah_plot[1]-lambda0*gamma0_plot[1],
      gammah_plot[2]-lambda0*gamma0_plot[2], length = 0.12, lwd=1, col = "red")
text(gammah_plot[1]-lambda0*gamma0_plot[1]-0.4,gammah_plot[2]-
       lambda0*gamma0_plot[2], labels = expression(b==gamma[h]+
       lambda*gamma[0]), col = "black", cex = 1.2)
text(gammah_plot[1]-lambda0*gamma0_plot[1]+0.4,gammah_plot[2]-
       lambda0*gamma0_plot[2]+0.15, labels = 
       expression(b==~"|"~lambda*gamma[0]~"|"), col = "red", cex = 1)

expression("E" *  "|" ~ Y)

# Insert unit length b0
b0<-c(gammah_plot[1]-lambda0*gamma0_plot[1],gammah_plot[2]-
        lambda0*gamma0_plot[2])
lb0<-sqrt(sum(b0^2))
#  arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
arrows(0,0,b0[1],b0[2], length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.3,b0[2]/lb0-0.2, labels = expression(c==~"|"~gamma[h]+
                          lambda*gamma[0]~"|"), col = "red", cex = 1)

# Draw the angle beta0h (between gammah_plot and gamma0_plot)
r <- 0.3   # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, beta0h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+beta0h / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(theta[0*h]), col = "black", cex = 1.2)

# Draw the angle beta between gammah_plot and b 
r <- 0.35   # arc radius
th_seq <- -beta0h+atan2(gammah_plot[2]-lambda0*gamma0_plot[2], 
  gammah_plot[1]-lambda0*gamma0_plot[1])+seq(0, beta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(beta), col = "red", cex = 1.2)


# Draw the angle theta (between b0 and gamma0_plot)
theta <- atan2(b0[2], b0[1])-atan2(gamma0_plot[2], gamma0_plot[1])
r <- 0.5  # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+theta / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid)-0.1,
     labels = expression(theta[0*b]), col = "black", cex = 1.2)

# Add side naming for side a  
text(1.2 * r * cos(th_mid)+0.5, 1.15 * r * sin(th_mid)+0.3,
     labels = expression(a==~"|"~gamma[h]~"|"), col = "red", cex = 1)

# Draw the angle gamma from the apex gammah_plot
r <- 0.35   # arc radius
th_seq <- seq(pi+0.15, pi+0.6, length.out = 100)
lines(gammah_plot[1]+r * cos(th_seq), gammah_plot[2]+r * sin(th_seq), 
      col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah_plot[1]+1.4 * r * cos(th_mid)+0.15, gammah_plot[2]+
       1.15 * r * sin(th_mid),
     labels = expression(gamma==theta[0*h]), col = "red", cex = 1.2)

# Draw the angle alpha from the apex b
r <- 0.1   # arc radius
th_seq <- seq(-pi/2-0.4, pi/8-0.2, length.out = 100)
lines(gammah_plot[1]-lambda0*gamma0_plot[1]+r * cos(th_seq), gammah_plot[2]-
        lambda0*gamma0_plot[2]+r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah_plot[1]-lambda0*gamma0_plot[1]+1.4 * r * cos(th_mid)+0.05, 
     gammah_plot[2]-lambda0*gamma0_plot[2]+1.15 * r * sin(th_mid),
     labels = expression(alpha), col = "red", cex = 1.2)


# Note on the geometry of the DFP solution:
#
# The figure illustrates that the DFP predictor b lies on the side of gammah
# (the MSE predictor) that is OPPOSITE to gamma0 (the nowcast). Equivalently,
# gammah lies between gamma0 and b, so the angle between gamma0 and b exceeds
# the angle between gamma0 and gammah:
#
#   theta_{0b} > theta_{0h}
#
# This condition is termed (positive) "phase excess" in Wildi (2026). In the
# standard case (typical applications), phase excess is sufficient to guarantee
# that b leads gammah at the trend frequency (frequency zero):
#
#   - Proposition 2 establishes the phase-excess condition formally.
#   - Theorem 2 derives the lead tau > 0 from a set of basic assumptions
#     that include theta_{0b} > theta_{0h} as a key premise.
#
# Intuitively: if gammah already leads gamma0 at frequency zero,  the DFP 
# predictor b accentuates this existing lead by rotating further away from 
# gamma0 and beyond gammah, i.e. it does 'more of the same'. In doing so, 
# b amplifies the phase advance that gammah has relative to gamma0, thereby 
# acquiring an additional lead relative to gammah itself.
#
# However, in some non-standard cases, gammah does not lead gamma0 at frequency 
# zero. This is discussed in Wildi (2026, Appendix A). In such a case, a
# lead by the DFP predictor is obtained when either
#
#   (i)  b lies between gamma0 and gammah, or
#
#   (ii) b lies on the opposite side of gamma0 relative to gammah
#        (gamma0 lies between gammah and b)
#
# Counterintuitively, in the non-standard case the objective function is
# inverted in sign: rather than maximising the correlation with gammah
# (the MSE predictor, as in the standard case), the DFP minimises it.
# This highlights that designing predictors with genuine look-ahead
# behaviour is more challenging and subtle than it may first appear.



# ════════════════════════════════════════════════════════════════════
# Exercise 3: MSE-DFP: Interpretable Time-Shift DFP Constraint
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Relating the DFP Constraint to a Frequency-Zero Time-Shift
# ─────────────────────────────────────────────────────────────────────
#
# Without a unit-length constraint on the filter, the parameter alpha0 in the
# MSE-DFP formulation lacks a straightforward interpretation: its effect on the
# predictor cannot easily be expressed in terms of an observable or intuitive
# quantity (unlike the correlation-based interpretation available for the
# unit-length DFP in Exercise 1). Moreover, as discussed above, it is unclear 
# how the necessary decoupling from the present x_t also addresses 
# look-ahead behaviour in terms of a lead over the MSE benchmark.
#
# To remedy this, we re-parameterise the DFP constraint by linking alpha0 to
# the time-shift (phase delay) at frequency zero — the trend frequency —
# as introduced in Tutorial 2.
#
# Concretely, the re-parameterised DFP predictor is designed so that its
# output LEADS the MSE predictor by a pre-specified number of time steps, tau,
# at frequency zero. This gives alpha0 a clear and interpretable meaning:
#
#   tau = 0  →  no lead relative to the MSE predictor (recovers MSE)
#   tau > 0  →  the DFP output anticipates the MSE predictor by tau periods
#               at the trend frequency
#
# The following sections derive and illustrate this frequency-zero DFP
# formulation.


# ─────────────────────────────────────────────────────────────────────
# 3.1 AR(3) and DFP Settings
# ─────────────────────────────────────────────────────────────────────
# The illustration reuses the AR(3) process from Exercise 2.
# Two forecast horizons are considered:
#   h      — the primary (short) horizon used for the MSE-DFP predictor
#   htilde — a longer horizon used as an additional reference

L <- 50   # filter length (number of MA coefficients retained)
h <- 3    # primary forecast horizon (steps ahead)

# Longer forecast horizon used as a reference benchmark
htilde <- 20

# AR(3) roots (lambda1, lambda2, lambda3) and the corresponding AR coefficients
# obtained by expanding (1 - lambda1*B)(1 - lambda2*B)(1 - lambda3*B)
lambda1 <- 0.3
lambda2 <- 0.8
lambda3 <- 0.2

ar1 <- ar11 <- lambda1 + lambda2 + lambda3
ar2 <- ar21 <- -lambda1*lambda2 - lambda1*lambda3 - lambda2*lambda3
ar3 <- ar31 <-  lambda1*lambda2*lambda3

# Compute a long MA(∞) expansion of the AR(3) process.
# More than L terms are needed to accurately construct the MSE forecast 
# filters below.
gamma <- ARMAtoMA(ar = c(ar1, ar2, ar3), lag.max = 1000)

# Inspect the first L MA coefficients
ts.plot(gamma[1:L])

# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah):
# shift gamma by h positions and pad with zeros at the end.
# The trailing zeros reflect the fact that MA coefficients beyond index L
# are negligible for a finite-length filter of length L.
gammah <- c(gamma[h + (1:(L - h))], rep(0, h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps)
lead <- -2


# ─────────────────────────────────────────────────────────────────────
# 3.2 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)

# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)
dfp_obj <- dfp_from_tau_func(gamma0, gammah, lead)

# Extract the components returned by the function
tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
b       <- dfp_obj$b        # raw DFP filter coefficients

# Normalise b to unit length to obtain the unitary DFP filter
b_opt <- b / as.double(sqrt(b %*% b))


# ─────────────────────────────────────────────────────────────────────
# 3.3 Time-Shifts at Frequency Zero
# ─────────────────────────────────────────────────────────────────────
# The frequency-zero time-shift of a filter with coefficients c is defined as
#   tau_c = sum_{k=0}^{L-1} k * c_k  /  sum_{k=0}^{L-1} c_k
# This is the weighted-average lag, which equals the phase delay at omega = 0.
# See Theorem 2 (equation 34) in Wildi (2026) for the explicit formula linking
# lambda0 to the desired lead tau.

# Frequency-zero time-shifts of the three filters
tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)       # nowcast filter
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)       # h-step MSE filter
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)  # long-horizon filter

# Display tau0 and tauh (the MSE filter leads slightly at frequency zero)
tau0
tauh

# Desired lead of the DFP over the MSE predictor at frequency zero
tau <- lead

# Compute lambda0 from the closed-form expression (Theorem 2, equation 34):
#   lambda0 = -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))

# Construct the raw DFP filter as a regularised combination of gammah and gamma0
b <- gammah + lambda0 * gamma0


# ─────────────────────────────────────────────────────────────────────
# 3.4 Verifications and Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved lead matches the specified lead ---
taub <- sum((0:(L-1)) * b_opt) / sum(b_opt)  # frequency-zero shift of the DFP filter

# Actual lead of the DFP over the MSE predictor at frequency zero
lead_dfp_mse <- taub - tauh

# This difference should be (numerically) zero
lead_dfp_mse - lead

# --- Check 2: round-trip via alpha0 (Proposition 1) ---
# Compute alpha0 = <gamma0, b> (inner product of gamma0 and the DFP filter)
alpha0 <- as.double(t(gamma0) %*% b)

# Recover lambda0 from alpha0 using the formula in Proposition 1:
#   lambda = (alpha0 - <gamma0, gammah>) / <gamma0, gamma0>
lambda <- as.double((alpha0 - t(gamma0) %*% gammah) / (t(gamma0) %*% gamma0))

# Difference between recovered and original lambda0 should vanish
lambda - lambda0

# Reconstruct the DFP filter from the recovered lambda (should match b above)
b <- gammah + lambda * gamma0

# --- Check 3: consistency with dfp_from_alpha0_func ---
# Verify that computing the DFP directly from alpha0 reproduces b_opt
dfp_obj <- dfp_from_alpha0_func(gamma0, gammah, alpha0)
lambda   <- dfp_obj$lambda
b        <- dfp_obj$b

# Normalise to unit length for comparison with b_opt
scale <- as.double(1 / sqrt(t(b) %*% b))
b0    <- scale * b

# Maximum absolute deviation from b_opt should be negligible
max(abs(b_opt - b0))


# ─────────────────────────────────────────────────────────────────────
# 3.5 Compute Complete Decoupling for Additional Reference
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled DFP corresponds to alpha0 = 0, i.e. the DFP filter
# is orthogonal to gamma0. This serves as a reference benchmark alongside the
# time-shift DFP computed above.

alpha0_cd <- 0  # complete decoupling: <gamma0, b_cd> = 0

# Disabled inline computation (kept for reference; superseded by the function call below)
if (F) {
  lambda_cd <- as.double((alpha0_cd - t(gamma0) %*% gammah) / 
                           (t(gamma0) %*% gamma0))
  b_cd      <- gammah + lambda_cd * gamma0
}

# Compute the completely decoupled DFP filter via Proposition 1
dfp_obj  <- dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
lambda_cd <- dfp_obj$lambda
b_cd      <- dfp_obj$b

# Normalise to unit length so that b_cd is comparable to b_opt
scale <- as.double(1 / sqrt(t(b_cd) %*% b_cd))
b_cd  <- scale * b_cd


# ─────────────────────────────────────────────────────────────────────
# 3.6 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# Note on phase reversal:
# The completely decoupled filter b_cd has a negative gain at frequency zero,
# i.e. Gamma(0) = sum(b_cd) < 0, which means it phase-reverses (inverts) the
# trend component of the input. As a consequence, the frequency-zero time-shift
# tau = -sum(k * b_cd) / sum(b_cd) is formally ill-defined in the usual sense
# (a negative denominator implies the filter is inverting, not delaying).
sum(b_cd)

# We can still compute the numerical value of the time-shift formula.
# The result should be interpreted as the shift of the sign-reverted predictor
# (-b_cd), which does have a positive gain at frequency zero.
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd


# ─────────────────────────────────────────────────────────────────────
# 3.7 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (negative = leads the target)
#   Gamma(0)  — gain at frequency zero (sum of filter weights, unit-normalised)
#   lambda    — DFP regularisation weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value)
#
# Rows correspond to:
#   MSE(h)       — h-step MSE predictor
#   MSE(htilde)  — long-horizon MSE predictor (reference)
#   DFP-shifted  — time-shift DFP with specified lead
#   DFP full dec.— completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 4, ncol = 4)

colnames(mat_perf) <- c("$\\tau(0)$", "$\\Gamma(0)$", "$\\lambda$", "$\\alpha_0$")
rownames(mat_perf) <- c(paste("MSE(", h,      ")", sep = ""),
                        paste("MSE(", htilde, ")", sep = ""),
                        "DFP-shifted",
                        "DFP full dec.")

# MSE h-step: time-shift and unit-normalised gain; lambda and alpha0 not applicable
mat_perf[1, 1:2] <- c(-tauh,
                      sum(gammah) / as.double(sqrt(t(gammah) %*% gammah)))

# Long-horizon MSE: same metrics for htilde
mat_perf[2, 1:2] <- c(-tauhtilde,
                      sum(gammahtilde) / as.double(sqrt(t(gammahtilde) %*% gammahtilde)))

# Time-shift DFP: all four metrics available
mat_perf[3, ] <- c(-taub, sum(b_opt), lambda0, alpha0)

# Completely decoupled DFP: time-shift is NA (ill-defined due to phase reversal)
mat_perf[4, ] <- c(NA, sum(b_cd), lambda_cd, alpha0_cd)

mat_perf


# ─────────────────────────────────────────────────────────────────────
# 3.8 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 3.9)
par(mfrow=c(1,2))

colo <- c("black", "green", "blue", "red")

# Collect all four filters into a matrix (no scaling applied)
# Columns: AR(3) nowcast, h-step MSE, time-shift DFP, fully decoupled DFP
mplot     <- scale(cbind(gamma0, gammah, b_opt, b_cd), center = F, scale = F)
col_names <- c("AR(3)", paste("MSE ", h, "-step"), "DFP-shift", "DFP-full-decouple")
colnames(mplot) <- col_names

# Diagnostic: sum of squared coefficients per filter (proxy for filter energy)
apply(mplot^2, 2, sum)

# --- Top-left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters and add colour-coded labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], type = "l")
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the MSE h-step filter on top to ensure visibility
lines(mplot[, 2], col = colo[2])

axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()

# --- Top-right panel: cross-correlation functions (CCF) ---
# Compute the CCF between each predictor and the AR(3) process at lags
# surrounding lag 0 and the h-step-ahead lag
mplot <- cbind(
  compute_acf_at_lags_zero_delta_func(max_lag, h, gammah, gamma0)$cor_vec,
  compute_acf_at_lags_zero_delta_func(max_lag, h, b_opt,  gamma0)$cor_vec,
  compute_acf_at_lags_zero_delta_func(max_lag, h, b_cd,   gamma0)$cor_vec
)
colnames(mplot) <- col_names[2:length(col_names)]

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[2], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[1 + i])
}

# Disabled in-plot legend (filters identified by colour above)
# mtext("MSE", line=-1, col=colo[1+1])
# mtext("DFP", line=-2, col=colo[2+1])

# Vertical reference lines:
#   solid  → lag 0 (current observation, index = max_lag + 1)
#   dashed → h-step-ahead lag
abline(v = max_lag + 1,     lty = 1)
abline(v = max_lag + 1 + h, lty = 2)
abline(h = 0)

axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()


# ─────────────────────────────────────────────────────────────────────
# 3.9 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 3.9.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.
filter_mat <- cbind(
  gamma0  / as.double(sqrt(t(gamma0)  %*% gamma0)),   # unit-normalised nowcast filter
  gammah  / as.double(sqrt(t(gammah)  %*% gammah)),   # unit-normalised h-step MSE filter
  b_opt,                                    # time-shift DFP (already unit-length)
  b_cd                                      # completely decoupled DFP (unit-length)
)
colnames(filter_mat)<-col_names

# Fix the random seed and generate a long white-noise input series
set.seed(345)
len <- 10000
x   <- rnorm(len)

# Apply each filter to x using one-sided (causal) convolution
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 3.9.2 Plot
#----------------------------------------------------------------------
# Disabled earlier diagnostic plot (shorter window, scaled outputs)
# ts.plot(scale(y_out_mat[270:305,], center=F, scale=T),
#         main="AR(3)", col=colo, xlab="", ylab="")
# abline(h=0)

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat[300:350, ],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

#----------------------------------------------------------------------
# 3.9.3 Compute empirical CCFs: referenced against MSE predictor gammah
#----------------------------------------------------------------------

y_out_mat<-na.exclude(y_out_mat)

par(mfrow=c(1,2))
ccf(y_out_mat[,1],y_out_mat[,3],main="CCF: DFP-shift vs. MSE",lag.max=10)
ccf(y_out_mat[,1],y_out_mat[,4],main="DFP-fully-decoupled vs. MSE",lag.max=10)

# Outcome:
# 1. The DFP with a time-shift of -2 at frequency zero exhibits an asymmetric CCF,
#    suggesting that it leads the MSE predictor gammah: it correlates more strongly 
#     with forward shifted than with backward shifted gammah. 
#     However, because the AR(3)
#    process distributes its spectral mass across the entire frequency band, this
#    time-shift lead (-2) is purely local to frequency zero and should not be interpreted as an
#    indicator of an overall aggregate lead over gammah (exercise 3.10
#    below verifies the target lead of -2 at zero frequency).
#
# 2. The fully decoupled DFP has a strongly asymmetric CCF with an effective
#    aggregate lead of two time units, as evidenced by the peak correlation
#    occurring at lead 2. Notably, a strong negative correlation with
#    backward-shifted values of gammah (i.e., lagged values) is observed,
#    suggesting that the fully decoupled DFP may be exploiting the natural
#    mean-reversion tendency of gammah.
#
# 3. With regrads to mean-revesrion, jsut doing the opposite of gammah is not a feasible strategy
#   becuase the correlation with gammah must be `large'. Thus the DFP addresses 
#   an internal consistency in a rationale optimization framework.

#
# 4. The CCF pattern of the fully decoupled DFP pushes to an extreme the
#    asymmetry already introduced by the more mildly decoupled DFP time-shift
#    predictor. This raises the question of interpretability and consistency
#    of aggressive look-ahead designs.


# ─────────────────────────────────────────────────────────────────────
# 3.10 Apply Predictors to a Linear Trend
# ─────────────────────────────────────────────────────────────────────
# Purposes:
# 1) verify pertinence of the time-shift formula at frequency zero.
# 2) Verify pertinence of the imposed DFP time-shift constraint.

# ─────────────────────────────────────────────────────────────────────
# 3.10.1 Generate Trend Predictor Outputs
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a linear trend input x = -len, ..., len.
# A linear trend is the canonical input for assessing frequency-zero
# behaviour: a filter with gain Gamma(0) and time-shift tau(0) will
# reproduce the trend scaled by Gamma(0) and shifted by tau(0) steps.

len <- 100
x   <- (-len):len

# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
ts.plot(y_out_mat,
        main = "Trend Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the completely decoupled DFP (red) INVERTS the trend direction.
# This is a direct consequence of its negative frequency-zero gain:
#   sum(filter coefficients) = Gamma(0) < 0
sum(filter_mat[, "DFP-full-decouple"])


# ─────────────────────────────────────────────────────────────────────
# 3.10.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# Three modifications are applied for a verification of the time-shift 
# constraint:
#   1. Remove the completely decoupled DFP (trend-inverting; not meaningful
#      as a look-ahead predictor for a linear trend).
#   2. Rescale the remaining filters so that each has unit gain at
#      frequency zero, i.e. divide each filter by its coefficient sum.
#      This normalises out amplitude differences and isolates timing (lead)
#      differences across predictors.
#   3. Shorten the trend path (smaller len) for visual clarity.

len <- 30
x   <- (-len):len   # shorter linear trend for clearer visualisation

# Apply each of the three remaining filters, normalised to unit gain at omega=0.
# Dividing by sum(filter) ensures Gamma(0) = 1 for each predictor.
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), side = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward so that the trend lines are separated vertically
# for easier visual comparison (pure display adjustment, no analytical effect)
mplot <- y_out_mat - min(y_out_mat, na.rm = T) - 5


par(mfrow = c(1, 1))
# Plot the rescaled trend outputs; horizontal leads between curves are now
# directly interpretable as time-shifts at frequency zero
ts.plot(na.exclude(mplot),
        main = "Trend Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# The observed left-shifts reflect the time-shift differences: we now verify
# that the observed empirical shifts match the theoretical expressions.

# ─────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# -----------------------------------------------------------------------
# Compute the effective lead from the trend outputs directly:
#   the horizontal offset between the two unit-amplitude trend lines.
# Note: we use the first non-NA offset
na.exclude(y_out_mat[, col_names[1]] - y_out_mat[, col_names[2]])[1]

# Compare with the closed-form time-shift difference at frequency zero
# (derived in Tutorial 2):
tauh - tau0

# The empirical lead from the trend output matches the theoretical
# time-shift difference, confirming that the frequency-zero analysis 
# (tutorial 2) correctly predicts the filter's behaviour on a linear trend.


# Check 2: lead of the DFP predictor (b_opt) over the MSE predictor (gammah)
# ---------------------------------------------------------------------------
# Inspect column names to confirm ordering before differencing
col_names

# Compute the effective lead (horizontal offset) of the DFP over the 
# MSE predictor (use the first non-NA offset):
na.exclude(y_out_mat[, col_names[2]] - y_out_mat[, col_names[3]])[1]
  
# Compare with the lead imposed via the DFP time-shift constraint:
taub - tauh

# The observed lead of the DFP trend output over the MSE trend output
# matches the pre-specified DFP lead at frequency zero, confirming that 
# the TIME-SHIFT CONSTRAINT IS CORRECTLY IMPLEMENTED AND ENFORCED.



# ─────────────────────────────────────────────────────────────────────
# 3.11 Apply Predictors to a (Business-) Cycle
# ─────────────────────────────────────────────────────────────────────
# Purpose:
# Verify spillover of the shift to frequencies nearby trend zero.

# ─────────────────────────────────────────────────────────────────────
# 3.11.1 Generate (Business-) Cycle
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a cosine with periodicity of 5 years in 
# in a monthly data framework.

len <- 120
# Five year periodicity:
omega<-pi/(5*6)
x   <- cos(omega*(1:len))

# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
ts.plot(y_out_mat,
        main = "Cycle Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the completely decoupled DFP (red) INVERTS the phase.


# ─────────────────────────────────────────────────────────────────────
# 3.11.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# Three modifications are applied for a verification of the time-shift 
# constraint:
#   1. Remove the completely decoupled DFP (trend-inverting; not meaningful
#      as a look-ahead predictor for a linear trend).
#   2. Rescale the remaining filters so that each has unit gain at
#      frequency zero, i.e. divide each filter by its coefficient sum.
#      This normalises out amplitude differences and isolates timing (lead)
#      differences across predictors.
#   3. Shorten the trend path (smaller len) for visual clarity.

# Apply each of the three remaining filters, normalised to unit gain at omega=0.
# Dividing by sum(filter) ensures Gamma(0) = 1 for each predictor.
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), side = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward so that the trend lines are separated vertically
# for easier visual comparison (pure display adjustment, no analytical effect)
mplot <- y_out_mat - min(y_out_mat, na.rm = T) - 5


par(mfrow = c(1, 1))
# Plot the rescaled trend outputs; horizontal leads between curves are now
# directly interpretable as time-shifts at frequency zero
ts.plot(na.exclude(mplot),
        main = "Cycle Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# The observed left-shifts reflect the time-shift differences: we now verify
# that the observed empirical shifts match the theoretical expressions.

# ─────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# -----------------------------------------------------------------------
# Compute the effective lead from the trend outputs directly:
# In constrast to the linear trend, the lead here is found be taking the distance between maxima or minima of the cycle
find_max_mat<-head(na.exclude(cbind(y_out_mat[, col_names[1]] , y_out_mat[, col_names[1]])),20)
# Maximum of nowcast
max_now<-which(find_max_mat[,1]==max(find_max_mat[,1]))
# Maximum of DFP-shift (shifted by tau=-2 at frequency zero)
max_mse<-which(find_max_mat[,2]==max(find_max_mat[,2]))
# Zero delay: the predictor is coincident with the nowcast (x_t):
max_mse-max_now
# Compare with the closed-form time-shift difference at frequency zero
# (derived in Tutorial 2):
tauh - tau0

# The original lead at frequency zero of the MSE predictor has vanished
# at frequency omega=pi/30. We now verify if the DFP (shifted by tau=-2 
# at zero-frequency) can maintain part of its lead at omega=pi/30>0



# Check 2: lead of the DFP predictor (b_opt) over the MSE predictor (gammah)
# ---------------------------------------------------------------------------
# Inspect column names to confirm ordering before differencing
col_names

# In constrast to the linear trend, the lead here is found be taking the distance between maxima or minima of the cycle
find_max_mat<-head(na.exclude(cbind(y_out_mat[, col_names[2]] , y_out_mat[, col_names[3]])),20)
# Maximum of MSE (gammah)
max_dfp<-which(find_max_mat[,2]==max(find_max_mat[,2]))

# Outcome: the empirical shift is still approximately 2 time units at omega=pi/30
max_dfp-max_mse

# Compare with the lead imposed via the DFP time-shift constraint:
taub - tauh


# ─────────────────────────────────────────────────────────────────────
# 3.12 Cycle + Noise
# ─────────────────────────────────────────────────────────────────────

# Outcome: the DFP cannot anticipate high-frequency noise. But we know from the previous
# exercise that it can anticipate cyclical turing-points. So if the signal (the cycle) 
# is stron enough, the DFP will lead the classic MSE predictor gammah at the cycle turning-points.

# ─────────────────────────────────────────────────────────────────────
# 3.13 Amplitude and Time-Shifts
# ─────────────────────────────────────────────────────────────────────

# Include complete decoupling: complete decoupling emphasizes leads at higher frequencies!!!!








# ════════════════════════════════════════════════════════════════════
# Exercise 4: ARMA(3,2)
# ════════════════════════════════════════════════════════════════════
# Second process: ARMA(3,2)
# AR-coefficients
ar1<-0.4
ar2<-0.3
ar3<-0.2
b1<-0.5
b2<-0.4

ts.plot(ARMAacf(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=100))


# Roots of characteristic polynomial
1/(Arg(polyroot(c(-ar3,-ar2,-ar1,1)))/pi)
abs(polyroot(c(-ar3,-ar2,-ar1,1)))

# MA inversion
# Compute long sequence: need more values than L for MSE forecasts below
gamma<-c(1,ARMAtoMA(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=1000))
ts.plot(gamma[1:20])
# L-length now- and MSE forecast
gamma0<-gamma[1:L]
gammah<-gamma[h+(1:L)]


# Compute MSE-DFP
cor_vec_mat<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec
alpha0_vec<-c(0.9,0.45,0.22,0.1,0)
#alpha0_vec<-5*c(0.9,0.45,0.22,0.1,0)
b_mat<-b_mat_unscaled<-a_mat<-lambda_vec2<-NULL
cor_vec_2<-matrix(ncol=2,nrow=length(alpha0_vec))

for (i in 1:length(alpha0_vec))#i<-1
{
  alpha0<-alpha0_vec[i]
  # Function for deriving b0  
  b0<-compute_mse_dfp(alpha0,gamma0,gammah)$b0
  # This is the same as 
  lambda<-as.double((alpha0-t(gamma0)%*%gammah)/(t(gamma0)%*%gamma0))
  b0_simpler<-gammah+lambda*gamma0
  max(abs(b0-b0_simpler))
  b_mat_unscaled<-cbind(b_mat_unscaled,b0)
  
  lambda_vec2<-lambda
  # MSE scaling is not used anymore (closed-form solution is already MSE optimal)  
  if (F)
  {
    scale<-as.double(t(b0)%*%gammah/(t(b0)%*%b0))
    b0<-scale*b0
  }
  # Compute minimum phase DFP  
  if (use_min_phase)
  {
    # Compute roots      
    roots<-polyroot(b0)
    # Invert unstable roots      
    roots[which(abs(roots)>1)]<-1/roots[which(abs(roots)>1)]
    # Compute coefficients from roots: minimum phase version of DFP      
    b_min_phase<-Re(rev(poly_from_roots(roots)))
    # Check lag-one ACF: the same
    b0[1:(length(b0)-1)]%*%b0[2:length(b0)]/t(b0)%*%b0-b_min_phase[1:(length(b_min_phase)-1)]%*%b_min_phase[2:length(b_min_phase)]/b_min_phase%*%b_min_phase
    b0<-b_min_phase
  }
  # Use either solution
  b_mat<-cbind(b_mat,b0)
  # AR-inversion (problem: b0 is not always invertible)  
  a_mat<-cbind(a_mat,-ARMAtoMA(ar=-b0[2:L]/b0[1],lag.max=L))
  
  # Compute CCF  
  cor_vec<-compute_acf_at_lags_zero_delta_func(max_lag,h,b_mat[,ncol(b_mat)],gamma0)$cor_vec
  cor_vec_mat<-cbind(cor_vec_mat,cor_vec)
  # Extract CCF at lead 0 and h  
  cor_vec_2[i,1]<-cor_vec[1]
  cor_vec_2[i,2]<-cor_vec[1+h]
}

# Check DFP constraint: should vanish
# Note: neither b_mat nor gamma0 are scaled to unit length and therefore alpha0_vec is not a correlation
t(b_mat)%*%gamma0-alpha0_vec
# Alternative check DFP constraint: should vanish
# Note: since cor_vec is the CCF we have to scale alpha0_vec by inverse lengths of gamma0 and b
cor_vec_1[,1]-alpha0_vec/sqrt(diag((t(b_mat)%*%b_mat))*as.double(t(gamma0)%*%gamma0))
# Check AR-inversion: the max absolute deviations should vanish (all designs are non-invertible and therefore one observes numerical cancellation effects, i.e., absolute deviations do not vanish exactly)
for (i in  1:length(alpha0_vec))#i<-1
{
  b0<-b_mat[,i]
  ar_vec<-a_mat[,i]
  print(max(abs(c(1,ARMAtoMA(ar=ar_vec,lag.max=L-1))-b0/b0[1])))
}



mplot3<-scale(cbind(gammah,b_mat),center=F,scale=T)/sqrt(L-1)
#mplot3<-cbind(gammah,b_mat)

mplot4<-cor_vec_mat[1:22,]*as.double(sqrt(gamma0%*%gamma0)/sqrt(gamma%*%gamma))

#--------------
# AR-inversion of DFP: we rely on possibility B described above
# Solution B: compute DFP AR-weights by convolution of original AR weights with DFP (MA inversion)
# 1. Compute AR-inversion of ARMA
ar_inv<-c(1,ARMAtoMA(ar=-c(b1,b2),ma=-c(ar1,ar2,ar3),lag.max=L))
# 1.1 check: the following should give an identity
filt1<-ar_inv
filt2<-gamma
conv_two_filt_func(filt1,filt2)$conv[1:10]

# 2. Compute AR weights of MSE and DFP predictors
# a. MSE
filt2<-gammah
ar_mse_arma32<-conv_two_filt_func(filt1,filt2)$conv
# b. DFP
ar_dfp_arma32_mat<-NULL
for (i in 1:length(alpha0_vec))
{
  # Use original (unscaled) MSE-DFP
  filt2<-b_mat_unscaled[,i]
  ar_dfp_arma32_mat<-cbind(ar_dfp_arma32_mat,conv_two_filt_func(filt1,filt2)$conv)
  
}
ts.plot(cbind(ar_mse_arma32,ar_dfp_arma32_mat)[1:L,],col=rainbow(ncol(a_mat)),main="Method B: MSE and DFP predictors AR-inverted")
lines(ar_mse_arma32,lwd=2)

# Check: apply filters to MA and AR representations
set.seed(1)
len<-1000
x<-eps<-rnorm(len)
for (i in 4:len)
{
  x[i]<-ar1*x[i-1]+ar2*x[i-2]+ar3*x[i-3]+eps[i]+b1*eps[i-1]+b2*eps[i-2]
}

y_dfp_ar32<-y_dfp_ma32<-rep(NA,len)
# Select DFP design
k<-2
for (i in L:len)
{
  y_dfp_ma32[i]<-b_mat_unscaled[,k]%*%eps[i:(i-L+1)]
  y_dfp_ar32[i]<-ar_dfp_arma32_mat[1:L,k]%*%x[i:(i-L+1)]
}
# Both series are identical up to negligible finite MA/AR inversion errors
ts.plot(cbind(y_dfp_ma32,y_dfp_ar32)[1:200,])
# Maximal error is negligible (due to finite length MA/AR inversions)
max(na.exclude(abs(y_dfp_ma32-y_dfp_ar32)[1:200]))


#-------------------------------------------------
# Plot


par(mfrow=c(2,2))

#ts.plot(gammah1,main=paste("MSE first process: h=",h,sep=""),col="green",xlab="",ylab="")

#ts.plot(gammah,main="Second process",col="green",xlab="",ylab="")


colo<-c("green","brown","orange","blue","violet","red")
# Scale filters
ts.plot(mplot1,main="Predictors: AR(3)",col=colo,xlab="",ylab="")
mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9, ",rho,"=",0.92)),line=-2,col=colo[2])
#mtext(expression(paste("    ",alpha[0],"=0.45, ",rho,"=",0.76)),line=-3,col=colo[3])
#mtext(expression(paste("    ",alpha[0],"=0.22, ",rho,"=",0.51)),line=-4,col=colo[4])
#mtext(expression(paste("    ",alpha[0],"=0.1, ",rho,"=",0.26)),line=-5,col=colo[5])
#mtext(expression(paste("    ",alpha[0],"=0, ",rho,"=",0.0)),line=-6,col=colo[6])
mtext(expression(paste("DFP ",alpha[0],"=0.9 ")),line=-2,col=colo[2])
mtext(expression(paste("    ",alpha[0],"=0.45 ")),line=-3,col=colo[3])
mtext(expression(paste("    ",alpha[0],"=0.22 ")),line=-4,col=colo[4])
mtext(expression(paste("    ",alpha[0],"=0.1 ")),line=-5,col=colo[5])
mtext(expression(paste("    ",alpha[0],"=0 ")),line=-6,col=colo[6])
abline(h=0)

cor_vec_1


ts.plot(mplot3,main="ARMA(3,2)",col=colo,xlab="",ylab="")
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("    ",alpha[0],"=0.9, ",rho,"=",0.99)),line=-2,col=colo[2])
#mtext(expression(paste("    ",alpha[0],"=0.45, ",rho,"=",0.96)),line=-3,col=colo[3])
#mtext(expression(paste("    ",alpha[0],"=0.22, ",rho,"=",0.84)),line=-4,col=colo[4])
#mtext(expression(paste("    ",alpha[0],"=0.1, ",rho,"=",0.58)),line=-5,col=colo[5])
#mtext(expression(paste("    ",alpha[0],"=0, ",rho,"=",0.0)),line=-6,col=colo[6])
abline(h=0)

plot(mplot2[,1],main="CCF",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot2),max(mplot2)))
for (i in 2:ncol(mplot2))
{  
  lines(mplot2[,i],col=colo[i])
}
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9")),line=-2,col=colo[2])
#mtext(expression(paste("DFP ",alpha[0],"=0.45")),line=-3,col=colo[3])
#mtext(expression(paste("DFP ",alpha[0],"=0.22")),line=-4,col=colo[4])
#mtext(expression(paste("DFP ",alpha[0],"=0.1")),line=-5,col=colo[5])
#mtext(expression(paste("DFP ",alpha[0],"=0")),line=-6,col=colo[6])
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
abline(h=0)
axis(1,at=1:nrow(mplot2),labels=-max_lag-1+1:(nrow(mplot2)))
axis(2)
box()

plot(mplot4[,1],main="",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot4),max(mplot4)))
for (i in 2:ncol(mplot4))
{  
  lines(mplot4[,i],col=colo[i])
}
abline(h=0)
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9")),line=-2,col=colo[2])
#mtext(expression(paste("DFP ",alpha[0],"=0.45")),line=-3,col=colo[3])
#mtext(expression(paste("DFP ",alpha[0],"=0.22")),line=-4,col=colo[4])
#mtext(expression(paste("DFP ",alpha[0],"=0.1")),line=-5,col=colo[5])
#mtext(expression(paste("DFP ",alpha[0],"=0")),line=-6,col=colo[6])
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
axis(1,at=1:nrow(mplot4),labels=-max_lag-1+1:(nrow(mplot4)))
axis(2)
box()


mat_cor_vec<-round(cbind(cor_vec_1,cor_vec_2),2)

# Must use a special proceeding when utilizing greek letters in column or rownames

rownames(mat_cor_vec) <- c("$\\alpha_0=0.9$","$\\alpha_0=0.45$","$\\alpha_0=0.22$","$\\alpha_0=0.1$","$\\alpha_0=0$")
colnames(mat_cor_vec) <- c(" $\\textrm{Process 1: CCF }\\delta=0$","$\\delta=5$","$\\textrm{Process 2: } \\delta=0$","$\\delta=5$")

mat_cor_vec


if (F)
{
  tbl <- xtable(mat_cor_vec)
  caption(tbl)<-paste("CCFs of the DFP predictors for the first process (columns 1–2) and the second process (columns 3–4) evaluated at $\\delta=0$ (columns 1 and 3) and at $\\delta=h=5$ (columns 2 and 4), shown for multiple values of the decoupling parameter $\\alpha_0$. Note that $\\alpha_0$ generally differs from the CCF at $\\delta=0$ except in the case of complete decoupling (columns 1 and 3, last row).")
  label(tbl)<-"ar3_decoupling"
  
  print.xtable(tbl, sanitize.text.function = function(x) x)
}






 











# ════════════════════════════════════════════════════════════════════
# Exercise 5: Complete decoupling and limit to look ahead
# ════════════════════════════════════════════════════════════════════
# Intuitively difficult since latest observation most important.



# ════════════════════════════════════════════════════════════════════
# Exercise 6: Leading indicator DFP
# ════════════════════════════════════════════════════════════════════
















####################################################
# R code for solving phase excess theta as a function of betah


# Define lengths of MSE predictors and angle thetah between them
# We use the same gammah, gammahm1 as in above plot
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
betah<--0.1

lh<-sqrt(sum(gammah^2))
lhm1<-sqrt(sum(gammahm1^2))
thetah<-atan2(gammah[2],gammah[1])- atan2(gammahm1[2],gammahm1[1])


a<-lh-cos(thetah)*lhm1
b<-sin(thetah)*lhm1
c<--betah



solve_acos_bsin_eq <- function(a, b, c) {
  R <- sqrt(a^2 + b^2)
  if (R == 0) {
    if (c == 0) return(list(status = "infinite solutions (all theta)", theta = NULL, phi = NA, R = 0))
    return(list(status = "no solution", theta = NULL, phi = NA, R = 0))
  }
  phi <- atan2(b, a)                # phase shift
  x <- c / R
  # Clamp for numerical safety
  x <- max(min(x, 1), -1)
  if (abs(c) > R + .Machine$double.eps^0.5) {
    return(list(status = "no real solution (|c| > R)", theta = NULL, phi = phi, R = R))
  }
  if (abs(abs(x) - 1) < 1e-14) {
    # Single solution modulo 2π
    theta <- if (x > 0) phi else (phi + pi)
    theta <- atan2(sin(theta), cos(theta))  # wrap to (-pi, pi]
    return(list(status = "one solution modulo 2π", theta = theta, phi = phi, R = R))
  }
  alpha <- acos(x)
  theta1 <- phi + alpha
  theta2 <- phi - alpha
  # Wrap to (-pi, pi]
  wrap <- function(t) atan2(sin(t), cos(t))
  theta <- sort(c(wrap(theta1), wrap(theta2)))
  list(status = "two solutions modulo 2π", theta = theta, phi = phi, R = R)
}

# Find theta for given a,b,c
solve_acos_bsin_eq(a, b, c )

#########################################################























#######################################################################################################

# Example: DFP for a given tau
# Example in the paper but no plot




max_lag<-2
L<-50
h<-3
# First AR(3)
lambda1<-0.3
lambda2<-0.8
lambda3<-0.2
ar1<-ar11<-lambda1+lambda2+lambda3
ar2<-ar21<--lambda1*lambda2-lambda1*lambda3-lambda2*lambda3
ar3<-ar31<-lambda1*lambda2*lambda3

# Compute long sequence: need more values than L for MSE forecasts below
gamma<-ARMAtoMA(ar=c(ar1,ar2,ar3),lag.max=1000)

ts.plot(gamma[1:L])

gamma0<-gamma[1:L]
# MSE: last entries are vanishing (we could also insert the longer MA-expansion but this would not be the MSe estimate in the finite length MA case)
gammah<-c(gamma[h+(1:(L-h))],rep(0,h))

# Select lead over MSE
lead<--1

if (F)
{
  # Old code
  # Compute shifts at frequency zero
  tau0<-sum((0:(L-1))*gamma0)/sum(gamma0)
  tauh<-sum((0:(L-1))*gammah)/sum(gammah)
  
  # MSE is slightly leading
  tau0
  tauh
  tau<-lead
  # Formula for lambda0
  lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
  # Compute b
  b<-gammah+lambda0*gamma0
  
}

# Compute shifts at frequency zero

dfp_obj<-dfp_from_tau_func(gamma0,gammah,lead)
tau0=dfp_obj$tau0
tauh=dfp_obj$tauh
lambda0=dfp_obj$lambda0
b=dfp_obj$b
# Unitary DFP
b_opt<-b/as.double(sqrt(b%*%b))

# Check lead
taub<-sum((0:(L-1))*b_opt)/sum(b_opt)
# Should equal lead (or tau): this is an exact result
taub-tauh

# Compute alpha0
alpha0<-compute_alpha_0_func(gamma0,gammah,lambda0)$alpha0
# Replicate DFP predictor with DFP criterion in MSE_LA_closed_form_rank_two_func
criterion_number<-1
# Select large lambda
lambda<-1000
val_vec_target<-1
# Need to scale alpha0 since the optimization routine assumes scaled gamma0,gammah
val_vec_constraint<-alpha0/as.double(sqrt(gamma0%*%gamma0))

MSE_LA_obj<-MSE_LA_closed_form_rank_two_func(criterion_number,h,lambda,gammah,gamma0,val_vec_target,val_vec_constraint,L)

# Check: ratio should be nearly one (up to negligible errors due to roots of quartic equation)  
b_opt/MSE_LA_obj$b


#--------------------------------------------------------
# Plots

par(mfrow=c(3,2))

colo<-c("black","green","blue")

mplot<-scale(cbind(gamma0,gammah,b_opt),center=F,scale=F)#/sqrt((L-1))
colnames(mplot)<-c("Nowcast",paste("MSE ",h,"-step"),"DFP")
apply(mplot^2,2,sum)
plot(mplot[,1],main="Filters",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
lines(mplot[,2],col=colo[2])
axis(1,at=1:nrow(mplot),labels=0:(nrow(mplot)-1))
axis(2)
box()

mplot<-cbind(compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec,compute_acf_at_lags_zero_delta_func(max_lag,h,b_opt,gamma0)$cor_vec)




K<-600
mplot<-scale(cbind(gamma0,gammah,b_opt),center=F,scale=F)#/sqrt((L-1))
apply(mplot^2,2,sum)
colnames(mplot)<-c("Nowcast",paste("MSE ",h,"-step"),"DFP")
shift_mat<-amp_mat<-matrix(ncol=ncol(mplot),nrow=K+1)
colnames(shift_mat)<-colnames(amp_mat)<-colnames(mplot)
for (i in 1:ncol(mplot))
{  
  filt_obj<-amp_shift_func(K,mplot[,i],F)
  shift_mat[,i]<-apply(cbind(rep(0,K+1),filt_obj$shift),1,max)
  amp_mat[,i]<-filt_obj$amp
}  


mplot<-amp_mat
plot(mplot[,1],ylim=c(0,max(mplot)),axes=F,col=colo[1],type="l",xlab="Frequency",ylab="",main="Amplitude")
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()

# Positive numbers signify left shift
mplot<-cbind(shift_mat[,1]-shift_mat[,1],shift_mat[,2]-shift_mat[,1],shift_mat[,3]-shift_mat[,1])
plot(mplot[,1],axes=F,col=colo[1],type="l",xlab="",ylab="",main="Leads over nowcast",ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()


# 1. Linear trend
len<-10000
x<-1:len
# Scale all filters to unit-length
filter_mat<-cbind(gamma0/mean(gamma0),gammah/mean(gammah),b_opt/mean(b_opt))/L
apply(filter_mat,2,sum)
y_out_mat<-filter(x,filter_mat[,1],side=1)
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,2],side=1))
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,3],side=1))
colnames(y_out_mat)<-c("Process=nowcast",paste("MSE ",h,"-step",sep=""),"DFP")
colo<-c("black","green","blue")

anf<-100
enf<-110
ts.plot(y_out_mat[anf:enf,]-(anf-tau0-1),col=colo,main="Linear trend",xlab="",ylab="")
#mtext("Nowcast",line=-1,col=colo[1])
#mtext("MSE",line=-2,col=colo[2])
#mtext("DFP",line=-3,col=colo[3])
abline(h=4)

set.seed(345)

x<-rnorm(len)
y_out_mat<-filter(x,filter_mat[,1],side=1)
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,2],side=1))
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,3],side=1))
colnames(y_out_mat)<-c("Process=nowcast",paste("MSE ",h,"-step",sep=""),"DFP")

ts.plot(scale(y_out_mat[290:320,],center=F,scale=T),main="White noise",col=colo,xlab="",ylab="")
abline(h=0)

#######################################################################################################




# Example DFP applied to MA(9)
# It relies on quadratic DFP in lambda1,lambda2: function DFP_compute_lambda_alpha0_func above.



# We use the solution to the first unit-length DFP criterion: quadratic in lambda
# Advantage: alpha0 in decoupling constraint is lag-zero CCF (theta)

# Design
h<-5
L<-10
ar1<-0.9
ar2<-0.
# Use c(1,ARMAtoMA(ar=c(ar1,ar2),lag.max=L)) since the weight 1 of epsilon_t is omitted
gamma<-c(1,ARMAtoMA(ar=c(ar1,ar2),lag.max=L-1))
# Forecast horizon
delta<-h

# Compute MSE forecast and nowcast (the latter is the DGP since x_t is causal)
gamma0<-gamma
gammah<-c(gamma0[(h+1):L],rep(0,h))

# CCF of MSE predictor at delta=0
ccf_mse0<-as.double(t(gammah)%*%gamma0/sqrt(t(gamma0)%*%gamma0*t(gammah)%*%gammah))

# Compute alpha0 in decoupling constraint: this is also the lag zero CCF (or theta)
# Impose mild decoupling and complete decoupling
alpha0_vec<-c(ccf_mse0/2,0)
# Compute DFP predictors
b0_mat<-matrix(nrow=L,ncol=length(alpha0_vec))
lambda1<-lambda2<-NULL
for (i in 1:length(alpha0_vec))
{ 
  alpha0<-alpha0_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b0_obj<-DFP_compute_lambda_alpha0_func(gamma0,gammah,h,L,alpha0)
  b0_mat[,i]<-b0_obj$b0
  lambda1<-c(lambda1,b0_obj$lambda1)
  lambda2<-c(lambda2,b0_obj$lambda2)
  
}
colnames(b0_mat)<-c("DFP mild","DFP complete")

ts.plot(b0_mat)
# Check: should be one on diagonal (unit length)
diag(t(b0_mat)%*%b0_mat)


#----------------------------------------------
# Apply filters to data
# generate filtered series
len1<-100000

set.seed(4)
eps<-rnorm(len1)
mat_out<-matrix(nrow=len1,ncol=3)
z<-mse<-fast_for<-rep(NA,len)
for (i in L:len1)
{
  # DGP  
  z[i]<-gamma0%*%eps[i:(i-L+1)]
  # MSE and DFP predictors  
  mat_out[i,1]<-gammah%*%eps[i:(i-L+1)]
  mat_out[i,2]<-b0_mat[,1]%*%eps[i:(i-L+1)]
  mat_out[i,3]<-b0_mat[,2]%*%eps[i:(i-L+1)]
}
colnames(mat_out)<-c("mse",colnames(b0_mat))
colo<-c("black","green","royalblue")
ts.plot(scale(mat_out[500:min(1000,len1),],scale=T,center=F),col=colo)
for (i in 1:ncol(mat_out))
  mtext(colnames(mat_out)[i],col=colo[i],line=-i)

#-----------------------
max_lag<-0

# Simon's proposal: MSE-predictor of x_{t+h}-x_t
#gammah<-gammah-gamma0


# Compute CCFs of predictors
# MSE
gamma1<-gammah
# Add zeroes to avoid NAs
gamma_ref<-c(gamma0,rep(0,100))
ccf_mat<-compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec
# DFP mild
# Add zerors to avoid NAs
gamma1<-b0_mat[,1]
ccf_mat<-cbind(ccf_mat,compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec)
# DFP mild
gamma1<-b0_mat[,2]
ccf_mat<-cbind(ccf_mat,compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec)

colnames(ccf_mat)<-c("MSE",colnames(b0_mat))


#----------------------------------------
# Generate plot and table


colo<-c("green","red","royalblue")

layout(matrix(c(1,2,3,3), 2, 2, byrow = T)) 


mplot<-cbind(gammah,b0_mat)
plot(mplot[,1],main="Predictor",axes=F,type="l",xlab="Lags",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-ccf_mat
plot(mplot[,1],main="CCF",axes=F,type="l",xlab="Leads",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(v=max_lag+1,col="royalblue")
abline(v=max_lag+1+h)
abline(h=0)
text(5,0.7,"MSE",col=colo[1])
text(5,0.5,"Partial decoupling",col=colo[2])
text(5,0.1,"Complete decoupling",col=colo[3])

axis(1,at=1:nrow(mplot),labels=-(max_lag+1)+1:nrow(mplot))
axis(2)
box()

anf<-900
anf<-1300
anf<-1500
anf<-1800
anf<-3200
mplot<-scale(mat_out[anf:min(anf+80,len1),],scale=T,center=F)

plot(mplot[,1],main="Forecast",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(h=0)
axis(1,at=(1:(nrow(mplot)/10))*10,labels=(1:(nrow(mplot)/10))*10)
axis(2)
box()




mplot<-cbind(gammah,b0_mat)

# Performances: CCFs and lead at frequency zero
mat_cor_vec<-ccf_mat[c(max_lag+1,max_lag+1+h),]
colnames(mat_cor_vec)<-c("MSE","DFP weak decoupling","DFP complete decoupling")
rownames(mat_cor_vec)<-c("CCF at lag=0",paste("CCF at h=",h,sep=""))
mat_cor_vec
# Impose a positive sign of zero
mat_cor_vec[1,3]<-abs(mat_cor_vec[1,3])


# Time shifts at omega=0: DFP with complete decoupling is not meaningful because of phase reversal at zero
tauh<-(1:(L-1))%*%gammah[2:L]/sum(gammah)
tau_pd<-(1:(L-1))%*%b0_mat[2:L,1]/sum(b0_mat[,1])
tau_cd<-(1:(L-1))%*%b0_mat[2:L,2]/sum(b0_mat[,2])



mat_cor_vec<-rbind(mat_cor_vec,c(0,2.12,4.03))
mat_cor_vec[3,]<-round(mat_cor_vec[3,])

rownames(mat_cor_vec)[3]<-"Relative lead over MSE"
mat_cor_vec


#######################################################################################################


# Application of MSE-DFP to AR(3) and ARMA






