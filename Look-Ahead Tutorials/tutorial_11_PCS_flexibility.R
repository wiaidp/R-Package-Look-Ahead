# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS APPLIED TO MONTHLY US EMPLOYMENT 
# ════════════════════════════════════════════════════════════════════

#
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════





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


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1 — Monthly US Employment: Equally Weighted Trend
# ════════════════════════════════════════════════════════════════════

# We adopt the framework from Tutorial 9: an ARMA(1,1) model fitted to the
# monthly PAYEMS employment indicator. 


# ─────────────────────────────────────────────────────────────────────
# 1.1 Load the Data
# ─────────────────────────────────────────────────────────────────────

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
# The log transformation stabilises the variance as the level of the
# series grows over time. Skipping COVID data avoids distortions by extreme 
# lockdown outliers.
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y) <- index(PAYEMS["1990::2019"])

par(mfrow=c(2,2))
plot(y,
     main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = FALSE,
     xlab = "", ylab = "")
axis(1, at = 1:length(y), labels = names(y))
axis(2)
box()

# Compute stationary first differences of the log-series:
#   - The log transformation stabilises the variance.
#   - The first difference stabilises the level (removes the trend).
x <- diff(y)

# The differenced log-PAYEMS series is fairly noisy, with pronounced
# downturns during recession episodes.
ts.plot(x,main="Diff-log PAYEMS")

# The empirical ACF decays slowly and monotonically — a pattern
# consistent with the dominant AR structure and indicative of an MSE
# predictor that is 'stuck at the present' (see Tutorial 1).
acf(x,main="ACF diff-log PAYEMS")


# ─────────────────────────────────────────────────────────────────────
# 1.2 Model Fit
# ─────────────────────────────────────────────────────────────────────

L <- 50   # filter length (number of MA coefficients retained)

# Fit an ARMA(1,1) model: a parsimonious specification with adequate diagnostics.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)
a1 <- arima.obj$coef[1:ar_order]
b1 <- arima.obj$coef[ar_order + 1:ma_order]

# Fix the parameters for replicability of results
a1<-0.95
b1<--0.53
# --- Wold Decomposition (MA-infinity representation) ---
# Compute the infinite-order MA coefficients (impulse response weights)
# of the fitted ARMA model. The filter length L ensures that the
# coefficients have decayed sufficiently close to zero by lag L.
if (ma_order > 0) {
  xi <- c(1, ARMAtoMA(
    ar      = a1,
    ma      = b1,
    lag.max = length(x)))
} else {
  xi <- c(1, ARMAtoMA(
    ar      = a1,
    ma      = 0,
    lag.max = length(x)))
}

# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: slowly decaying impulse response (post-1990)")

# The theoretical ACF implied by the Wold decomposition matches the
# empirical ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),
        main = "Model-based ACF", ylab = "", xlab = "Lag")

# For k > 0 the ACF satisfies the recurrence ACF(k+1) = a1 * ACF(k), so
# the DGP imposes a rigid linear structure on the autocorrelation function.
#
# For h > 0 and k >= 0 this implies gamma_h ∝ gamma_{h+k}: the MSE predictor
# coefficient vectors are mutually proportional for all horizons h > 0.
#
# Consequently, the column space of the constraint system has rank 2:
# (gamma_0 - gamma_1) and (gamma_1 - gamma_2) ∝ gamma_1 are the only
# linearly independent directions. Because the Type III PCS enforces only a
# single constraint, the problem remains feasible. Unfortunately, the CCF
# cannot peak at h = 12 — this is structurally impossible for the
# ARMA(1,1) DGP — so the problem is classified as impossible but feasible.


# ─────────────────────────────────────────────────────────────────────
# 1.3 MSE Now- and Forecast
# ─────────────────────────────────────────────────────────────────────
# One-year-ahead forecast horizon.
h <- 12
# Larger horizon for benchmarking
htilde<-24

# Truncate the Wold coefficients to length L to obtain the nowcast
# filter (gamma_0).
gamma0 <- xi[1:L]

# h-step-ahead MSE predictor (gamma_h):
# shift the Wold coefficients forward by h positions.
gammah <- xi[h + 1:L]
gammahtilde <- xi[htilde + 1:L]




# ─────────────────────────────────────────────────────────────────────
# 1.4 Target: Equally-Weighted Trend (Yearly Growth)
# ─────────────────────────────────────────────────────────────────────


# The differenced log-PAYEMS series is fairly noisy, with pronounced
# downturns during recession episodes.

# To reduce noise we apply a lowpass filter. Possible choices include:
#   - Classic trend filters, e.g., the HP filter (see tutorial 10)
#   - Ideal lowpass filter
#   - AR(1) smoother
#   - Moving average (MA)
#
# For simplicity, we use an equally-weighted MA(12) for the following reasons:
#   - The DFP approach is agnostic to this choice; results generalize to
#     other target filter designs.
#   - An equally-weighted MA(12) applied to differenced log-PAYEMS corresponds
#     directly to yearly growth, making the target readily interpretable.
#   - Averaging over a full year reduces noise and amplifies the relevant
#     business-cycle dynamics.

# Define the equally-weighted yearly MA target filter
gamma_target <- rep(1/12, 12)

# Express trend in MA-equivalent form:
gamma <- conv_two_filt_func(xi, gamma_target)$conv


# Visualize the Wold coefficients for both representations.
par(mfrow = c(2, 1))
ts.plot(xi,    main = "Wold Decomposition: Monthly Growth (Post-1990)")
ts.plot(gamma, main = "Wold Decomposition: Yearly Growth (Post-1990)")

# Set MSE nowcast and h=12-step ahead predictors:
gamma0<-gamma[1:L]
gammah<-gamma[h+1:L]

# Note: The h=12-step ahead MSE predictor is AR(1) with a1 determined by the ARMA(1,1).

# ─────────────────────────────────────────────────────────────────────
# 1.5 DGP Structural Constraints on PCS Solution Space
# ─────────────────────────────────────────────────────────────────────

# For h>=1 gammah<-a1^(h+0:L)
gamma_mat<-gamma[1:L]
for (i in 1:(L-1))
{
  gamma_i<-gamma[i+1:L]
  gamma_mat<-cbind(gamma_mat,gamma_i)
}  

eigenvalues<-eigen(gamma_mat)$values

# Maximal rank of an equation system involving the CCF in PCS
length(which(abs(eigenvalues)>10^{-10}))

# Type II or III PCS:
# In these cases, imposing a single constraint of the type b' * (gamma_h-gamma_{0}) steals 
# one of the available 13 degrees of freedom: 13 degrees of freedom are left for 
# optimization.

# Type I with h=12:
# Strong regularization implies that all but one degrees of freedom are eaten-up by the 
# 12-dimensional constraint system of PCS type I. One degree can be used to maximize 
# the target correlation.

# ─────────────────────────────────────────────────────────────────────
# 1.6 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Forecast horizon
h<-12



# Using yearly differences increases the rank of the constraint system.

# However, when length(Delta) >12 the system is still overdetermined: the PCS equation system is infeasible.
# If length(Delta)=12 the equations system is regular and can be solved with a unique solution. 
# Hence no degree of freedom is left for optimiaztion and the target correlation could become negative (as is the case here).
# In both cases the PCS is infeasible.
# If length(Delta)<12 the equation system is underdetermined and henece there are degrees of freedom
# left for optimization which implies that the target correlation is positive.

# Proceeding
# Selecting Delta=1:11  and lambda arbitrarily large allows for positive target correlation
# Selecting Delta=2:12  and lambda arbitrarily large does not always allow for positive target correlation
# because the rank of the system without lag 0 is reduced.
# Obviously, selecting Delta=1:12 and lambda arbitrarily large does not always allow for positive target correlation
# Selecting Delta=1:12 and lambda moderate allows for positive target correlation and increased flexibility 
#   in addressing the constraints: preferred solution.
# We here select 


# Since one degree of freedom is left for optimization, the
Delta <- 1:h
lambda<-1000000000
lambda<-1000000

# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.


beta_vec <- c(-0.1,-0.0001,-0.00007, -0.00005, -0.000035,-0.00002, -0.00001,0,0.000003,0.000009,0.000012,0.00002,0.00004,0.00008,0.00016, 0.1, 0.2, 0.3)/2



beta<-0
Type_III<-F
scaled_constraints<-F
high_resolution<-T


PCS_obj<-PCS_func(h, Delta, gamma, L, beta, lambda,Type_III,scaled_constraints,high_resolution)

beta_vec<-PCS_obj$beta_vec




# ─────────────────────────────────────────────────────────────────────
# 1.7 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 7))

# ─────────────────────────────────────────────────────────────────────
# 1.8 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above: for a feasible system, residuals of each slope
# constraint should vanish as lambda increases (conditional on numerical 
# precision).

# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# preserves the direction of any trend or level shift present in the data.
# For moderate positive slopes (beta > 0), the coefficient sum remains positive,
# indicating that trend orientation is preserved. As beta increases beyond a
# certain threshold, the coefficient sum becomes negative, signaling trend
# inversion.
apply(b_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# A sufficiently small positive slope (beta > 0) ensures that the target
# covariance remains positive. Beyond a certain slope threshold, positivity
# is violated and the corresponding predictors become unusable.
t(b_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 1.9 Plots and Performance Summary
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
                     max_lag, h, filter_mat[, i], gamma)$cor_vec)
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

# It looks as if the last CCF(12)-CCF(11) is never positive. The crux is:
# beta becomes so small that the negative slope error slips through the finite-sized regularization.
# Whatever the size of lambda, beta will be correspondingly down-sized to slip through. 
# This suggests that the only solution with non-negative slope throughout is zero...

# We will look at the closed-form solution in exercise 2 below.

# For very large lambda the slope is nearly constant.



ccf_mat[13,]-ccf_mat[12,]

# Predictor weights:
# -The classic MSE(12) is AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# -If lambda is moderate (i.e. lambda=1) the peak of the CCF is shifted rightwards but 
#   it is not located at k=h=12. Nevertheless, the PCS has look ahead behaviour which is 
#   is the main purpose of PCS.


# ─────────────────────────────────────────────────────────────────────
# 1.10 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.10.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 1.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)


anf<-200
enf<-250
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,]
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)




# Empirical CCF: positive beta induce an increasing CCF by sign inversion: 
# The type I constraints are satisfied but CCF(h) < 0 and the predictor is unusable.
# This forecast problem is impossible and unfeasible, see Tutorial 13 for background.
par(mfrow = c(2, 2))
select_vec<-c(2,3+3:5)
for (i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ",colnames(y_out_mat)[i],sep=""))
  
}




# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: Closed Form Exact PCS
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# As shown in exercise 1.5, the constraint system has full rank and can be 
# solved in closed-form (exactly), see equations 47 and 48 in Wildi (2026).
# We here compute the closed-form solutions  for the beta values in exercise 1, 
# based on the function PCS_closed_form(). We then compare exact solutions with 
# the strongly regularized solutions obtained with PCS_func() from exercise 1.



# ─────────────────────────────────────────────────────────────────────
# 2.1 Closed Form PCS
# ─────────────────────────────────────────────────────────────────────


b_closed_mat <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_closed_form_func(h,Delta, gamma, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_closed_mat   <- cbind(b_closed_mat, b)
  
  # Constraint check: for a full-rank constraint system the deviations
  # of the closed-form solution vanish:
  print(abs(d_delta %*% b + beta))
}

colnames(b_closed_mat) <- paste0("Closed-form PCS, beta=", round(beta_vec, 7))

# ─────────────────────────────────────────────────────────────────────
# 2.1 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above.



# ── Check 2: Sign / Orientation Preservation ─────────────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# preserves the direction of any trend or level shift present in the data.
# For moderate positive slopes (beta > 0), the coefficient sum remains positive,
# indicating that trend orientation is preserved. As beta increases beyond a
# certain threshold, the coefficient sum becomes negative, signaling trend
# inversion.
apply(b_closed_mat, 2, sum)

# ── Check 3: Positive Target Covariance ──────────────────────────────────────
# 
t(b_closed_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_closed_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 1.9 Plots and Performance Summary
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
                     max_lag, h, filter_mat[, i], gamma)$cor_vec)
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

# The CCF of the closed-form (exact) PCS is either decreasing and positive 
# (beta<0) or increasing and negative (beta > 0). This result can be contrasted 
# with the strong regularization in the previous exercise: we here compare the 
# CCF profiles of both solutions:


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

# Since lambda < ∞, strong regularization (left panel, exercise 1) leaves 
# some limited room for maximizing the target correlation CCF(h). Consequently, 
# the CCF profiles are slightly upward biased toward k = h. In contrast, the 
# closed-form solutions satisfy all constraints exactly, leaving no residual 
# freedom to inflate the CCF at k = h.

# In both cases, the predictors are flawed and unusable, 










# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: As Exercise 1 but Medium Regularization
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# The Type 1 constraint system imposes h=12 constraints: 
# b' * (gamma_k-gamma_{k-1}= = beta, for k=1,...,12

# The rank of the constraint system is 12: hence all constraints can be satisfied but 
# there is no room left for target optimization.

# A very large lambda (exercise 1 above) emphasizes the constraints only, to the detriment 
# of the target correlation CCF(h).

# For beta > 0 the predictor changes sign and the target correlation is negative, CCF(h) < 0, 
# leading to an unusable PCS predictor.

# Here, we retain the same slope parameters beta, but we select a medium-sized lambda. This 
# allows the target correlation to enter the optimization and hence avoids a negative CCF(h) 
# when beta > 0.

# Smaller Lambda: allows to address the target correlation as a valid objective besides the PCS constraints, addressing the 
# PCS constraints in a more flexible way
lambda<-30


# ─────────────────────────────────────────────────────────────────────
# 3.1 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────






# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 3.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above: for a feasible system, residuals of each slope
# constraint should vanish as lambda increases.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# does not invert the direction of a trend or level shift in the data.
# Here, all peak-shifting designs (beta > 0) produce negative coefficient
# sums, indicating trend inversion. This is a direct and potentially
# undesirable cost of aggressive look-ahead behaviour under strong
# regularisation: the linear CCF constraint forces the filter to assign
# sufficiently negative weights to older lags that the overall orientation
# of the filter is reversed. Milder regularisation (explored in Exercise 4)
# can alleviate this potentially undesirable effect.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah

# Infeasibility: the constraints with negative slope beta<0 can be enforced, 
# but the target correlation turns negative.

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 3.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
                     max_lag, h, filter_mat[, i], gamma)$cor_vec)
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

# Predictor weights:
# -The classic MSE(12) and MSE(24) are both AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# -If lambda is moderate (i.e. lambda=1) the peak of the CCF is shifted rightwards but 
#   it is not located at k=h=12. Nevertheless, the PCS has look ahead behaviour which is 
#   is the main purpose of PCS.


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

# MSE(12) and MSE(24) overlap exactly after scaling: no look ahead behaviour 
# for h > 12.
par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)


# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually.
# Increasing beta generates look ahead behaviour by left-shifting the PCS predictors 
# relative to the MSE benchmark(s).
anf<-200
enf<-250

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,]
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)




# The empirical CCF peak is effectively left-shifted towards h=12.  

par(mfrow = c(2, 2))
select_vec<-c(2,3+3:5)
for (i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, i]),
    lag.max = 20, plot = TRUE,
    main = paste0("Nowcast vs. ",colnames(y_out_mat)[i],sep=""))

}





# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: As Exercise 1 but Type III) PCS based on PCS_func()
# ════════════════════════════════════════════════════════════════════

# New feature: we can address a positive average growth between k=0 and k=h=12.
# This generalizes exercise 1 based on unitary DFP, since we can use PCS_func()
# which allows fine-tuning of the constraint (in contrast to DFP in exercise 1).

# Since we impose a single aggregate constraint (between k=0 and k=h) the PCS constraints 
# are underdetermined and allow for maximization of the target correlation for arbitrarily large lambda. 
# Delta: from k=0 to k=12
Delta <- c(0,12)
# New feature: we must inform PCS_func below that we use type III
Type_III<-T

lambda<-1000000



if (length(Delta)>=length(which(abs(eigenvalues)>10^{-10}))-1)
{
  print("PCS system has no degrees of freedom left for optimization")
  print("Select regularization weight lambda not too large or reduce length of Delta (number of constraints")
}

beta_vec <- c(-0.1, 0, 0.1, 0.2, 0.5)



# ─────────────────────────────────────────────────────────────────────
# 4.1 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor. We supply the additional initialize_with_null whose default value is F (when omitted in the previous exercises)
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda,Type_III)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 4.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above: for a feasible system, residuals of each slope
# constraint should vanish as lambda increases.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# does not invert the direction of a trend or level shift in the data.
# Here, all peak-shifting designs (beta > 0) produce negative coefficient
# sums, indicating trend inversion. This is a direct and potentially
# undesirable cost of aggressive look-ahead behaviour under strong
# regularisation: the linear CCF constraint forces the filter to assign
# sufficiently negative weights to older lags that the overall orientation
# of the filter is reversed. Milder regularisation (explored in Exercise 4)
# can alleviate this potentially undesirable effect.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah

# Infeasibility: the constraints with negative slope beta<0 can be enforced, 
# but the target correlation turns negative.

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 4.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
                     max_lag, h, filter_mat[, i], gamma)$cor_vec)
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


# Predictor weights:
# -The classic MSE(12) is AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# - We address the mean-growth between k=0 and k=12: beta>0 means that CCF(h)-CCF(0)>0.
# - This does not imply that the CCF peak is shifted to k=h.
# - But we can still gain look ahead behaviour


# ─────────────────────────────────────────────────────────────────────
# 4.4 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 4.4.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the innovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 4.4.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)


# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually.
# Increasing beta generates look ahead behaviour by left-shifting the PCS predictors 
# relative to the MSE benchmark(s).
anf<-200
enf<-250

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,]
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)



# The empirical CCF peak is effectively left-shifted towards h=12.  

par(mfrow = c(2, 2))
select_vec<-c(2,3+3:5)
for (i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ",colnames(y_out_mat)[i],sep=""))
  
}



# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: IMPOSSIBILITY
# ════════════════════════════════════════════════════════════════════
# We rely on exercise 1 but we target monthly growth, i.e., we ignore the 
# equally weighted trend


# Visualize the Wold coefficients for both representations.
par(mfrow = c(2, 1))
ts.plot(xi,    main = "Wold Decomposition: Monthly Growth (Post-1990)")
ts.plot(gamma, main = "Wold Decomposition: Yearly Growth (Post-1990)")



# We replace gamma by xi, the original Wold decomposition of the ARMA(1,1), skipping 
# the convolution with the equally weighted trend
gamma0<-xi[1:L]
gammah<-xi[h+1:L]
# Note: The h=12-step ahead MSE predictor is AR(1) with a1 determined by the ARMA(1,1).

# ─────────────────────────────────────────────────────────────────────
# 1.5 DGP Structural Constraints on PCS Solution Space
# ─────────────────────────────────────────────────────────────────────

# For h>=1 gammah<-a1^(h+0:L)
gamma_mat<-xi[1:L]
for (i in 1:(L-1))
{
  gamma_i<-xi[i+1:L]
  gamma_mat<-cbind(gamma_mat,gamma_i)
}  

eigenvalues<-eigen(gamma_mat)$values

# In contrast to exercise 1, the degrees of freedom shrink to 2.
length(which(abs(eigenvalues)>10^{-10}))

# Type II or III PCS:
# In these cases, imposing a single constraint of the type b' * (gamma_h-gamma_{0}) steals 
# one of the available 2 degrees of freedom: 1 degree is left for 
# optimization.

# Type I with h=12:
# Strong regularization implies that the system is generally overdetermined: if 
# the vector does not lie in the column space of the constraints, the constraints are infeasible. 
# This is the case with the type I 

# ─────────────────────────────────────────────────────────────────────
# 1.6 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.1, 0, 0.1, 0.2, 0.3)/2



# Using yearly differences increases the rank of the constraint system.

# However, when length(Delta) >12 the system is still overdetermined: the PCS equation system is infeasible.
# If length(Delta)=12 the equations system is regular and can be solved with a unique solution. 
# Hence no degree of freedom is left for optimiaztion and the target correlation could become negative (as is the case here).
# In both cases the PCS is infeasible.
# If length(Delta)<12 the equation system is underdetermined and henece there are degrees of freedom
# left for optimization which implies that the target correlation is positive.

# Proceeding
# Selecting Delta=1:11  and lambda arbitrarily large allows for positive target correlation
# Selecting Delta=2:12  and lambda arbitrarily large does not always allow for positive target correlation
# because the rank of the system without lag 0 is reduced.
# Obviously, selecting Delta=1:12 and lambda arbitrarily large does not always allow for positive target correlation
# Selecting Delta=1:12 and lambda moderate allows for positive target correlation and increased flexibility 
#   in addressing the constraints: preferred solution.
# We here select 


# This is infeasible for positive slopes (the CCF peak cannot be shifted)
Delta <- 1:12
lambda<-1000000







# ─────────────────────────────────────────────────────────────────────
# 1.7 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 1.8 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above: for a feasible system, residuals of each slope
# constraint should vanish as lambda increases.

# ── Check 2: Sign / orientation preservation ─────────────────────────
# A strictly positive sum of filter coefficients confirms that the filter
# does not invert the direction of a trend or level shift in the data.
# Here, all peak-shifting designs (beta > 0) produce negative coefficient
# sums, indicating trend inversion. This is a direct and potentially
# undesirable cost of aggressive look-ahead behaviour under strong
# regularisation: the linear CCF constraint forces the filter to assign
# sufficiently negative weights to older lags that the overall orientation
# of the filter is reversed. Milder regularisation (explored in Exercise 4)
# can alleviate this potentially undesirable effect.
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah

# Infeasibility: the constraints with negative slope beta<0 can be enforced, 
# but the target correlation turns negative.

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,gammahtilde,  b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 1.9 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))
lwd_vec = c(2,2,2,rep(1,ncol(b_mat)))
# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
                     max_lag, h, filter_mat[, i], gamma)$cor_vec)
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


# Predictor weights:
# -The classic MSE(12) is AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# -If lambda is moderate (i.e. lambda=1) the peak of the CCF is shifted rightwards but 
#   it is not located at k=h=12. Nevertheless, the PCS has look ahead behaviour which is 
#   is the main purpose of PCS.


# ─────────────────────────────────────────────────────────────────────
# 1.10 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.10.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 1.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)


anf<-200
enf<-250
mplot<-scale(y_out_mat,center=F,scale=T)[anf:enf,]
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)




# Empirical CCF: positive beta induce an increasing CCF by sign inversion: 
# The type I constraints are satisfied but CCF(h) < 0 and the predictor is unusable.
# This forecast problem is impossible and unfeasible, see Tutorial 13 for background.
par(mfrow = c(2, 2))
select_vec<-c(2,3+3:5)
for (i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = 20, plot = TRUE,
      main = paste0("Nowcast vs. ",colnames(y_out_mat)[i],sep=""))
  
}

























# Forecasting the ARMA(1,1) poses an impossible look ahead problem: the CCF peak 
# cannot be shifted at h>1.

# Applying an equally-weighted trend specification (yearly growth) allowed to expand 
# the rank 2 to a rank 12 constraint system. The peak of the CCF could be shifted towards 
# k > 2 but it is still impossible to shift further away than h=12.


# For business-cycle analysis one would typically on an alternative HP trend instead of the equally-weighted trend, 
# see tutorial 12.

# Impossibility and infeasibility will be discussed in Tutorial 13. 


# Finally, while a problem might be effectively impossible (no peak shift at k=h), we contend that it is still 
# possible to generate look ahead beviour out of such problems, see Tutorial 14.

