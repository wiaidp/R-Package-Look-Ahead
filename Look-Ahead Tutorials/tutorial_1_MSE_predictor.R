# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1 — MSE: THE LOOK-AHEAD PROBLEM
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial demonstrates a simple forecasting problem to illustrate
# a key limitation of the classical Mean Squared Error (MSE) multi-step-ahead
# predictor: it can become "stuck" at the current time point, meaning it
# fails to project future values beyond the present observation (i.e.,
# it lacks the ability to "look ahead").

# ════════════════════════════════════════════════════════════════════
# Theoretical background:
#   Wildi, M. (2026). Forecasting on the Accuracy–Timeliness Frontier:
#   Two Novel "Look-Ahead" Predictors.
#   https://doi.org/10.48550/arXiv.2602.23087

# ════════════════════════════════════════════════════════════════════

rm(list = ls())

# Load tau-statistic (measures lead/lag performance)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used in JBCY (requires mFilter)
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))


# ============================================================
# EXERCISE 1: MSE FORECAST
# ============================================================
# Data-generating process (DGP):
#   x_t = sum_{k=0}^9 (0.9^k * ε_{t-k})   → MA(9)
# ============================================================

#---------------------------------------------------------------
# 1.1 Data generation
#---------------------------------------------------------------
# Construct an MA(9) using a truncated MA representation of an AR(1),
# i.e., coefficients b_k = a1^k for k = 0,...,9

q <- 9
a1 <- 0.9
b <- a1^(0:9)

par(mfrow = c(1,1))
ts.plot(b, main = "MA(9) coefficients", xlab = "Lag")

# Simulate data
len <- 100
set.seed(231)
eps <- rnorm(len + q + 1)

# Initialize series and MSE predictor
x <- xhat <- rep(NA, len + q + 1)

# Generate MA(9) process
for (i in (q+1):(len+q+1)) {
  x[i] <- b %*% eps[i:(i-q)]
}

ts.plot(x, main = "Simulated MA(9) process")

# Inspect dependence structure
acf(na.exclude(x))


#---------------------------------------------------------------
# 1.2 MSE predictor
#---------------------------------------------------------------
# Forecast horizon
h <- 5

# Ensure that the forecast horizon is feasible
if (h > q + 1)
  print("(q + 1) must exceed h; otherwise the optimal forecast is zero.")

# In practice, ε_t is unobserved and must be recovered (e.g., via AR inversion).
# Here, ε_t is known, so we interpret the predictor as a linear filter applied
# directly to ε_t. The goal is to choose filter weights that minimize the MSE
# between the predictor and x_{t+h}.
#
# Expansion of the target:
#   x_{t+h} = ε_{t+h} + b1 ε_{t+h-1} + ... + b_{h-1} ε_{t+1}
#             + b_h ε_t + ... + b_q ε_{t+h-q}
#
# Under MSE optimality, future shocks (ε_{t+k}, k > 0) are replaced by zero.
# This yields the h-step-ahead predictor:
#   x̂_{t|t+h} = b_h ε_t + ... + b_q ε_{t+h-q}
#
# Corresponding optimal weights:
b_MSE <- b[(h+1):(q+1)]

# Length of the prediction filter
L <- q + 1 - h

# Compute MSE predictor
for (i in (q+1):(len+q+1)) {
  xhat[i] <- b_MSE %*% eps[i:(i-q+h)]
}


# Compare shifted target and MSE predictor
colo <- c("black", "green")
par(mfrow = c(2,1))

# Plot: shifted target vs forecast
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

# Plot: original series vs forecast
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


#---------------------------------------------------------------
# 1.3 Cross-correlation function (see Wildi, 2026)
#---------------------------------------------------------------
# Compute correlations between the MSE predictor x̂_{t|t+h} and x_{t+k}
# for leads and lags k with -4 <= k <= 9.
#
# Ideally, the predictor x̂_{t|t+h} should correlate most strongly with 
# x_{t+h} (the target), when k=h. In fact, the MSE predictor maximizes 
# this correlation: no alternative linear predictor can improve it.

# Maximum lag considered
max_lag <- 5

# Compute cross-correlations
cor_vec_lead <- cor_vec_lag <- NULL

# Leads: k = 0,...,h
for (i in 0:h)
  cor_vec_lead <- c(cor_vec_lead,
                    b_MSE[1:(min(L+i, (q+1)) - i)] %*% b[(i+1):min(L+i, (q+1))] /
                      (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

# Leads beyond h
for (i in 1:(L-1))
  cor_vec_lead <- c(cor_vec_lead,
                    b_MSE[1:(L-i)] %*% b[(h+i)+1:(L-i)] /
                      (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

# Lags
for (i in 1:(max_lag-1))
  cor_vec_lag <- c(cor_vec_lag,
                   b_MSE[(i+1):L] %*% b[1:(L-i)] /
                     (sqrt(b_MSE %*% b_MSE) * sqrt(b %*% b))
  )

cor_vec <- c(cor_vec_lag[length(cor_vec_lag):1], cor_vec_lead)

par(mfrow = c(1,1))
plot(cor_vec,
     main = "Cross-correlation at leads (+) and lags (−)",
     axes = FALSE, type = "l", xlab = "Lag/Lead", ylab = "Correlation",
     col = "green", lwd = 1)
abline(v = max_lag)
abline(v = max_lag + h, col = "green")
axis(1, at = 1:length(cor_vec), labels = -(max_lag) + 1:length(cor_vec))
axis(2)
box()

# The slight kink in the CCF at lag k = h = 5 (green vertical line in the plot)
# illustrates that the MSE predictor achieves its maximum correlation
# at k = h, as expected by construction.

# ============================================================
# DISCUSSION: MSE FILTER & THE FORECAST TRILEMMA
# ============================================================
# The classical MSE-optimal predictor focuses exclusively on forecast accuracy.
# It does not account for timeliness (phase shift) or smoothness,
# highlighting a fundamental trade-off — the so-called "Forecast Trilemma" —
# inherent in the design of any forecasting system.

# In the example above, the MSE predictor is effectively "stuck" at the current
# time point: the cross-correlation function (CCF) attains its maximum at lag
# k = 0, meaning the predictor correlates most strongly with x_t rather than
# the target x_{t+h}.

# The M-SSA tutorial addresses the smoothness component of this trilemma.

# Here, we propose novel forecasting algorithms that explicitly emphasize
# timeliness, effectively forcing the predictor to "look ahead" beyond the
# current time point.









