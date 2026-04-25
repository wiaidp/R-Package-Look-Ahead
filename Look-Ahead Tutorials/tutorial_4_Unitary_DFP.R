# ════════════════════════════════════════════════════════════════════
# TUTORIAL 4 — DECOUPLE FROM PRESENT (DFP) PREDICTOR
# PART 1: UNITARY DFP
# ════════════════════════════════════════════════════════════════════

# A brief overview is provided in tutorial_3_introduction.r

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
#----------------------------------------------------------------------
# This tutorial discusses the first form: the unitary DFP.
#----------------------------------------------------------------------
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# Exercise 1: MA-PROCESS
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
# epsilon_t rather than in terms of x_t directly.  Tutorial 4 will extend 
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
# 1.2 Classical MSE Predictor 
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

# The unitary DFP predictor b0 is a constrained linear combination of
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

b0_obj  <- unitary_DFP_func(gamma0, gammah, alpha0)

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
# REMARKABLY, THE PEAK OF THE (blue) CCF SHIFTED FROM k=0 (PRESENT) to k=h (FUTURE)
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
# 1.10 Geometry of the Unitary DFP Predictor
# ─────────────────────────────────────────────────────────────────────
# This figure (reproduced from Wildi 2026) depicts the geometry of the 
# unitary DFP solution in the plane spanned by gamma0 (nowcast) and 
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




