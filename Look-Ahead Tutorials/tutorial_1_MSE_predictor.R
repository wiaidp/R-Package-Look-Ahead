# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1 — MSE: THE LOOK-AHEAD PROBLEM
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────────
# This tutorial demonstrates a simple forecasting problem to illustrate
# a fundamental limitation of the classical Mean Squared Error (MSE)
# multi-step-ahead predictor: it can become "stuck" at the current time
# point, meaning it fails to project the series forward in time and instead
# tracks the present observation rather than anticipating future values.
#
# This failure is a manifestation of the broader "Forecast Trilemma":
# the inherent tension between three desirable predictor properties —
#   (1) Accuracy    — low MSE with respect to the true future target
#   (2) Timeliness  — the predictor leads rather than lags the target
#   (3) Smoothness  — the predictor does not over-react to noise
#
# The classical MSE criterion optimises (1) alone, often at the expense
# of (2). This tutorial makes that trade-off visible and motivates the
# "look-ahead" predictors introduced in Wildi (2026).

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ── INITIALISATION ────────────────────────────────────────────────────
# Clear the workspace to ensure a clean environment before execution
rm(list = ls())

# Load the tau-statistic utility
# Provides compute_min_tau_func(): measures lead/lag at zero crossings
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction and filter utility functions from the JBCY paper
# Dependency: the mFilter package must be installed
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: MSE FORECAST
# ════════════════════════════════════════════════════════════════════
#
# Data-Generating Process (DGP):
#   x_t = sum_{k=0}^{9} (0.9^k * ε_{t-k})     [MA(9) process]
#
# The MA(9) is constructed as a truncated version of an AR(1) with
# coefficient a1 = 0.9. Its moving-average weights decay geometrically:
#   b_k = a1^k,  k = 0, 1, ..., 9
#
# This DGP is simple enough to allow closed-form derivation of the
# optimal h-step-ahead MSE predictor, making it ideal for illustration.
# ════════════════════════════════════════════════════════════════════

# ── 1.1 Data Generation ───────────────────────────────────────────────
# Define MA(9) filter coefficients: b_k = a1^k, k = 0,...,q
q  <- 9          # MA order
a1 <- 0.9        # AR(1) coefficient (controls persistence of the process)
b  <- a1^(0:9)   # geometrically decaying MA weights

par(mfrow = c(1, 1))
ts.plot(b, main = "MA(9) filter coefficients (geometrically decaying)",
        xlab = "Lag", ylab = "Weight")

# Simulate a realisation of the MA(9) process
len     <- 100
set.seed(231)
eps <- rnorm(len + q + 1)   # innovations (i.i.d. standard normal)

# Pre-allocate series and predictor vectors
x <- xhat <- rep(NA, len + q + 1)

# Compute x_t = sum_{k=0}^{q} b_k * ε_{t-k}  for each t
for (i in (q + 1):(len + q + 1)) {
  x[i] <- b %*% eps[i:(i - q)]
}

ts.plot(x, main = "Simulated MA(9) process")

# Inspect the autocorrelation structure:
# ACF should show significant lags up to order q = 9 and then cut off
acf(na.exclude(x), main = "ACF of simulated MA(9) — cuts off at lag 9")


# ── 1.2 MSE-Optimal h-Step-Ahead Predictor ────────────────────────────
# Forecast horizon
h <- 5

# Feasibility check: if h > q + 1, all MA coefficients at horizon h are zero
# and the optimal MSE forecast degenerates to the unconditional mean (zero)
if (h > q + 1)
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# ── Derivation of the MSE predictor ───────────────────────────────────
# In practice ε_t is latent; here it is observed, so the predictor is treated
# as a linear filter applied directly to the innovation sequence.
#
# Target decomposition:
#   x_{t+h} = ε_{t+h} + b_1 ε_{t+h-1} + ... + b_{h-1} ε_{t+1}   [future shocks]
#           + b_h ε_t  + b_{h+1} ε_{t-1} + ... + b_q ε_{t+h-q}   [available shocks]
#
# The MSE criterion replaces all future shocks (ε_{t+k} for k > 0) with their
# conditional expectation of zero. The remaining terms define the predictor:
#
#   x̂_{t|t+h} = b_h ε_t + b_{h+1} ε_{t-1} + ... + b_q ε_{t+h-q}
#
# In filter notation, the optimal MSE weights are simply the MA coefficients
# at lags h through q:
b_MSE <- b[(h + 1):(q + 1)]   # optimal MSE filter weights (length L = q+1-h)
L     <- q + 1 - h            # number of non-zero filter taps

# Compute the MSE predictor for each time point
for (i in (q + 1):(len + q + 1)) {
  xhat[i] <- b_MSE %*% eps[i:(i - q + h)]
}

# ── Visual comparison ─────────────────────────────────────────────────
colo <- c("black", "green")
par(mfrow = c(2,1))

# Panel 1: x shifted forward by h (the true target) vs. the MSE forecast
# If the predictor were perfectly timely, the two lines would coincide.
mplot <- na.exclude(cbind(c(x[(h+1):length(x)], rep(NA, h)), xhat))
plot(mplot[,1],
     main = paste("x_t shifted forward", h, "steps (target) vs MSE forecast (green)"),
     axes = FALSE, type = "l", xlab = "Time", ylab = "",
     col = colo[1], lwd = 1)
lines(mplot[,2], col = colo[2])
mtext("Target", line = -1)
mtext("MSE predictor", col = "green", line = -2)
axis(1, at = 1:nrow(mplot), labels = 1:nrow(mplot))
axis(2)
box()

# Panel 2: original series x_t vs. the MSE forecast (unshifted)
# The predictor tracks x_t closely rather than x_{t+h}, revealing the
# "stuck-at-present" behaviour: the forecast is effectively contemporaneous.
mplot <- na.exclude(cbind(x, xhat))
plot(mplot[,1],
     main = "Original series x_t and 5-step-ahead MSE forecast (green)",
     axes = FALSE, type = "l", xlab = "Time", ylab = "",
     col = colo[1], lwd = 1)
lines(mplot[,2], col = colo[2])
mtext("Observed MA(9) process", line = -1)
mtext("MSE predictor", col = "green", line = -2)
axis(1, at = 1:nrow(mplot), labels = 1:nrow(mplot))
axis(2)
box()


# ── 1.3 Cross-Correlation Function (CCF) of the MSE Predictor ─────────
# We compute the theoretical (population) correlation between the MSE predictor
# x̂_{t|t+h} and x_{t+k} for a range of leads (+k) and lags (-k).
#
# Key question: at which k does the predictor correlate most strongly with x?
#
# If the predictor were truly "looking ahead" by h periods, the CCF should
# peak at k = h. 

max_lag <- 5   # number of lag steps to include on the negative side

# Accumulate correlation values across leads and lags
cor_vec_lead <- cor_vec_lag <- NULL

# ── Leads k = 0, ..., h (predictor vs. x_{t+k} for k <= h) ──────────
for (i in 0:h)
  cor_vec_lead <- c(
    cor_vec_lead,
    b_MSE[1:(min(L + i, (q + 1)) - i)] %*% b[(i + 1):min(L + i, (q + 1))] /
      (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

# ── Leads k = h+1, ..., h+L-1 (predictor vs. x_{t+k} for k > h) ─────
for (i in 1:(L - 1))
  cor_vec_lead <- c(
    cor_vec_lead,
    b_MSE[1:(L - i)] %*% b[(h + i) + 1:(L - i)] /
      (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

# ── Lags k = -1, ..., -(max_lag-1) (predictor vs. x_{t-k}) ──────────
for (i in 1:(max_lag - 1))
  cor_vec_lag <- c(
    cor_vec_lag,
    b_MSE[(i + 1):L] %*% b[1:(L - i)] /
      (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

# Combine: lags (reversed) | contemporaneous | leads
cor_vec <- c(cor_vec_lag[length(cor_vec_lag):1], cor_vec_lead)

par(mfrow = c(1, 1))
plot(cor_vec,
     main = "Theoretical CCF: MSE predictor vs. x_{t+k}",
     axes = FALSE, type = "l", xlab = "k (negative = lag, positive = lead)",
     ylab = "Correlation", col = "green", lwd = 1)
abline(v = max_lag,       lty = 2,          col = "black")   # vertical line at k = 0
abline(v = max_lag + h,   lty = 2,          col = "green")   # vertical line at k = h
axis(1, at = 1:length(cor_vec), labels = -(max_lag) + 1:length(cor_vec))
axis(2); box()

# Interpretation:
# The green dashed line marks k = h = 5 (the forecast horizon).
# The MSE predictor maximises the CCF(h) at k = h: no other predictor 
# leads to a larger CCF(h). 
# However, the CCF does not peak at k=h but instead at k=0:
#   -CCF(0)>CCF(h)
#   -The predictor correlates most strongly with x_t (k = 0), not with x_{t+h}.
# This peak of the CCF at k = 0 is the "stuck-at-present" problem.


# ════════════════════════════════════════════════════════════════════
# Exercise 2
# ════════════════════════════════════════════════════════════════════
# The forecast horizon h is the only tuning (hyper-) parameter of MSE.

# Q: Can we shift the CCF peak of the MSE predictor to the left by 
#    increasing h?

# Left as an exercise.


# ════════════════════════════════════════════════════════════════════
# DISCUSSION: THE FORECAST TRILEMMA AND THE LOOK-AHEAD PROBLEM
# ════════════════════════════════════════════════════════════════════
#
# The classical MSE-optimal predictor optimises ACCURACY alone.
# It ignores the other two legs of the Forecast Trilemma:
#
#   - TIMELINESS : the predictor should lead the target, not lag it.
#   - SMOOTHNESS : the predictor should not over-react to high-frequency noise.
#
# Consequence observed above:
#   The MSE predictor is effectively "stuck at the present" — its CCF peaks
#   at k = 0 rather than k = h, so it tracks x_t rather than anticipating
#   x_{t+h}. Increasing h cannot fix this: it is a structural
#   consequence of the MSE criterion and the data generating process.
#
# In applications, processes with monotonically decaying ACF (as our example) 
# are typical.

# Increasing the forecast horizon generally does not address the 
# "stuck-at-present" problem for such processes.

#
# Remedies addressed in subsequent tutorials:
#   - Tutorial 2 (Timeliness) : introduces actionable measures for timeliness 
#                               (lead/lag) to be used in DFP/PCS optimization.
#   - DFP predictor
#   - PCS predictor

# ════════════════════════════════════════════════════════════════════

















