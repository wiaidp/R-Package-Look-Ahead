# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — BUSINESS CYCLE ANALYSIS AND LEADING INDICATOR DESIGN
# ════════════════════════════════════════════════════════════════════

# ── MACRO INDICATOR DESIGN ───────────────────────────────────────────
#
# This section follows the Leading Indicator Design (LID) framework
# introduced in Wildi (2026), Section 3.5.
#
# Let x_t be a stationary indicator of interest — for example, the first
# difference of a non-stationary macro series such as industrial production,
# employment, income, or GDP. The target signal is defined as:
#
#   Phi' * X_t  (AR form)   ≡   gamma' * Epsilon_t  (MA form)
#
# Where Phi and gamma are vectors of length L and X_t=(x_t,...,X-{t-L+1}), 
# Epsilon_t=(epsilon_t,...,epsilon_{t-L+1}).

# Typical targets Phi (or gamma) include trend-, cycle-, or
# seasonal adjustment filters and  Phi' * X_t represents signal-growth: trend-, 
# cycle-, or seasonally adjusted growth. 

# Stationary signal-growth is often more relevant to analysts and decision 
# makers than non-stationary signal-level.

# Throughout this tutorial we work in the MA form. Here, gamma is the 
# convolution of the Wold decomposition xi of x_t with the filter Phi:
#
#   gamma = Phi ∘ xi,   where ∘ denotes convolution.
#
# Let gamma_k denote the minimum mean-squared-error (MSE) predictor of the
# signal at horizon k.

# ── LEADING INDICATOR DESIGN (LID) ─────────────────────────────────────────
#
# The optimization problem is:
#
#   Minimize  (b - gamma)' (b - gamma)          [MSE objective]
#   subject to  b' * (gamma_h - gamma_{h-1}) = beta   [lead constraint]
#
# See section 3.3, Wildi (2026). This problem is related (though not identical) 
# to the Type II PCS approach introduced in Tutorial 12. 

# The objective minimizes the distance from the causal (nowcast) filter
# gamma; no explicit forecasting step is involved. The hyperparameters
# h > 0 and beta >= 0 in the constraint jointly govern the profile of the 
# cross-correlation function (CCF) at the specified lead:
#
#   - beta > 0 : the CCF peak cannot be located in h-1.
#   - beta = 0 : the CCF is constrained to be flat between lags h-1 and h.
# 
# Under some circumstances, these constraints can determine an effective shift 
# of the CCF at k >= h, see examples below.
#
# ── HP TREND: BUSINESS-CYCLE ANALYSIS ─────────────────────────
#
# When the HP trend (Phi) is applied to the first differences (X_t) of the data, 
# the resulting indicator estimates current growth (drift):
#
#   Positive values → economic expansion
#   Negative values → economic contraction / recession
#
# This constitutes a straightforward form of business-cycle analysis. 
#
# - HP-trend tracks the level of the first differences (trend-growth), 
#   and `cyclical' up- and down-turns are triggered by changes in the 
#   underlying data growth-rate due to transitions between economic phases 
#   of expansion and contraction.  
# - These dynamics are endogenous to the data, not the filter.
# - Applying a trend filter to first differences tracks effective growth, 
#   thereby mitigating the problem of spurious cycle of conventional 
#   business-cycle designs, see Wildi (2014).

# Background is provided in the M-SSA Tutorial Series (Tutorial 2 of M-SSA).  

# ── SIMPLE VS. CHALLENGING FORECAST PROBLEMS ─────────────────────────
#
# In general, imposing a flat CCF at lag h (beta = 0) through the LID does 
# not guarantee that the global CCF peak occurs exactly at lag h. However, for 
# the present business-cycle application — which combines the HP filter with
# the above LID design — the problem is relatively well-conditioned:
# the CCF peak is naturally shifted to h = 0 as a direct consequence of
# the DGP gamma = HP ∘ xi.
#
# In more demanding forecasting settings (covered in Tutorials 12–15),
# such a peak shift may not arise automatically and may require more
# elaborate constraint specifications or alternative PCS designs.

# ═════════════════════════════════════════════════════════════════════
# ── REFERENCES ───────────────────────────────────────────────────────
#
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     arXiv preprint. https://doi.org/10.48550/arXiv.2602.23087
#
# ═════════════════════════════════════════════════════════════════════



# ── INITIALISATION ─────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

library(mFilter)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDP"))
} else {
  load(file = file.path(getwd(), "Data", "GDP"))
}
head(GDPC1)
tail(GDPC1)

is.xts(GDPC1)

# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application 
# more challenging (as an example, applying a filter to a xts-object reverts time).
# We here skip the pandemic: outliers affect the design 

end_year<-2024
start_year<-1992
y<-as.double(log(GDPC1[paste(start_year,"/",end_year,sep="")]))
y_xts<-log(GDPC1[paste(start_year,"/",end_year,sep="")])
len<-length(y)
use_adjusted_hamilton<-F
use_new_i2_adjustment<-F


# Plot
par(mfrow=c(2,2))
plot(GDPC1,main="US GDP")
plot(y_xts,main="Log-GDP")
plot(diff(y_xts),main="Diff-log")
acf(na.exclude(diff(y_xts)),main="ACF of log-diff")

x<-na.exclude(as.double(diff(y_xts)))
names(x)<-index(na.exclude(diff(y_xts)))

# The ACF suggests that log_differences of GDP are (close to) white noise.
# We assume that xi = 1.
# Note: applying a trend filter to white noise would generate spurious cycles. 
# However, log-differences of GDP are not white noise (in contradiction to empirical ACF).
#  - The mean level (long term growth) is different from zero
#  - Protracted down-turns (recessions) dominate the noisy dynamics.
#  - The trend filter removes the noise and emphasizes the relevant features: non-zero mean level and downturns.
# While the data is not white noise, for the purpose of filtering, at least, we may assume xi = 1.


# ─────────────────────────────────────────────────────────────────────
# HP Set-Up
# ─────────────────────────────────────────────────────────────────────


# HP setting for quarterly data: lambda = 1600
lambda_hp<-1600
# L is an odd integer such that the symmetric filter is centered at (L-1)/2+1  
L<-31
# One year ahead forecast horizon
h<-4
# Setting for computing the CCF: has no effect on predictor.
max_lag<-0


# Here we compute only the two-sided filter for double length 2*(L-1)+1
# This is used when comparing one-sided to right tail of two-sided
HP_obj<-HP_target_mse_modified_gap(2*(L-1)+1,lambda_hp)
HP_two=HP_obj$target
hp_gap=HP_obj$hp_gap[1:L]
modified_hp_gap=HP_obj$modified_hp_gap[1:L]
# Concurrent HP assuming I(2)-process
hp_trend_long=HP_obj$hp_trend
hp_trend=hp_trend_long[1:L]
# MSE estimate of bi-infinite HP assuming white noise
hp_mse_long=HP_obj$hp_mse
hp_mse<-hp_mse_long[1:L]



# The LID design relies on a single constraint and can be operationalized either 
# in DFP form or in PCS form. Exercise 1 presents the DFP implementation. The PCS implementation 
# is covered in exercise 2.


# ════════════════════════════════════════════════════════════════════
# Exercise 1: LID Based on DFP Optimization
# ════════════════════════════════════════════════════════════════════

# Type II) requires CCF(h) > CCF(h-1), i.e., the CCF must increase
# over the final step to the forecast horizon h. Using CCF(k) = b' * gamma_k, 
# this becomes:
#
#   b' * gamma_{h-1} < b' * gamma_h
#   <=>  b' * (gamma_{h-1} - gamma_h) < 0
#
# Equivalently, setting alpha0 = b' * gamma_constraint with
#   gamma_constraint = gamma_{h-1} - gamma_h,
# Type II) requires alpha0 < 0.
#
# This maps exactly onto a standard DFP decoupling problem: minimise MSE
# subject to b' * gamma_constraint = alpha0, with gamma_constraint playing
# the role of gamma_0. We therefore apply compute_mse_dfp() with
# gamma_constraint in place of gamma_0, and decrease alpha0 below the MSE
# baseline to progressively enforce the slope condition.
#
# Remark on interpretation:
#   Unlike the classic DFP constraint (which decouples b from the observable
#   present via gamma_0), the constraint vector 

#       gamma_constraint = gamma_{h-1} - gamma_h

#   is a difference of two forecast vectors and has no direct physical interpretation
#   as a "present-value" filter. Its role is purely algebraic: zeroing out
#   b' * gamma_constraint forces CCF(h-1) = CCF(h), and driving it negative
#   enforces CCF(h-1) < CCF(h). The resulting filter may therefore look
#   unusual (e.g., non-monotone weights), which is expected and not a cause
#   for concern: the constraint is meaningful even if gamma_constraint itself
#   is not intuitively interpretable.


# ─────────────────────────────────────────────────────────────────────
# 1.1 DFP Set-Up
# ─────────────────────────────────────────────────────────────────────



# Specify gamma at forecast horizon sup_vec_target=h and at lead/lag sup_vec_constraint
# Classic h-step ahead predictor
gamma<-hp_trend_long
gamma0<-hp_trend_long[1:L]
gammah<-hp_trend_long[h+1:L]
gammahm1<-hp_trend_long[h-1+1:L]
# Plot
colo<-c("black","blue","cyan")
ts.plot(cbind(gamma0, gammahm1,gammah),main="Nowcast, h and h-1 MSE predictors",col=colo)

if (F)
{
# Some Checks and diagnostics:
# The sum of filter coefficients of the trend should add to one
  sum(hp_trend_long)
# Standard error when applied to standardized white noise
  sqrt(t(hp_trend_long)%*%hp_trend_long)
} 



# LID constraint vector: difference between consecutive MSE predictors at lags h-1 and h
gamma_constraint <- gammahm1 - gammah
# Note: the sign in gamma_constraint is arbitrary and could be reversed, 
# together with alpha0 in the constraint.

par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == gamma[h-1] - gamma[h]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector encoding the CCF slope condition at lag h")
abline(h = 0)

# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor. 
# Purpose: mse_coup is a natural upper bound for the DFP constraint, i.e., 
# DFP should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, strictly smaller than the above mse_coup. 
# Smaller (more negative) values enforce progressively stronger CCF slope at 
# lag h (a right-shift of the peak towards h=1).
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0)

# The DFP constraint enforces stronger decoupling form gamma_constraint than 
# the MSE predictor gammah: the last negative value suggests that the peak CCF
# should be shifted to the right: from k=0 to k=h=1.
alpha0_vec





# ─────────────────────────────────────────────────────────────────────
# 1.2 Run DFP
# ─────────────────────────────────────────────────────────────────────

# For each alpha0 in alpha0_vec, compute the MSE-optimal PCS predictor via
# Proposition 1 (Wildi 2026):
#
#   b = gammah + lambda * gamma_constraint,
#   lambda = (alpha0 - gamma_constraint' * gammah) / (gamma_constraint' * gamma_constraint)
#
# This closed-form solution minimises the MSE subject to the modified
# decoupling constraint b' * gamma_constraint = alpha0.

b_mat       <- NULL    # filter coefficients, one column per alpha0
lambda_vec1 <- NULL    # corresponding Lagrange multipliers
cor_vec_mat <- NULL    # full CCF vectors, one column per alpha0

# CCF values at lags 0 and h for each alpha0 (for tabular summary)
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of leads on either side of lag 0 to include in the CCF
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Compute MSE-PCS predictor with modified constraint vector
  b <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0
  b_mat <- cbind(b_mat, b)
  
  # Compute the population CCF of b against the process over lags [-max_lag, h]
  cor_vec <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), gamma)$cor_vec
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1 + h-1]         # CCF at lag 0 (coupling with present)
  cor_vec_1[i, 2] <- cor_vec[1 + h]     # CCF at lag h (coupling with target)
}

colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1) <- paste("Lag ", (h-1):h,sep="")
rownames(cor_vec_1)<-paste0("alpha0=", round(alpha0_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 1.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS constraint ──────

# Verification: the constraint b' * gamma_constraint = alpha0 should hold
# exactly for every column of b_mat (residuals should be numerically zero).
t(b_mat) %*% gamma_constraint - alpha0_vec

# ── Check 2: sign / orientation preservation ──────────────────────────────
# A strictly positive sum of filter coefficients confirms that none of the
# PCS filters inverts the direction of a trend or level shift in the data.
apply(b_mat, 2, sum)

# CHECK 3 — Positive Target Covariance
t(b_mat)%*%gammah


# Collect all filters (nowcast, MSE, and PCS variants) into a single matrix
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast", paste("MSE(",h,")",sep=""),
                          paste0("LID ", round(alpha0_vec, 8)))



# ─────────────────────────────────────────────────────────────────────
# 1.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green", rainbow(ncol(b_mat)))

lwd_vec<-c(2,2,rep(1,ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncate to the first q+1 lags where the MA process has support.
mplot <- scale(filter_mat)
plot(mplot[, 1], main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec,lty=lwd_vec,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i],line = -i, col = colo[i])
abline(h = 0)
abline(v =   1,     lty = 1)   # lag 0
abline(v =   h+1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).

ccf_mat<-NULL
for (i in 1:ncol(filter_mat))
  ccf_mat<-cbind(ccf_mat,compute_acf_at_lags_zero_delta_func(
    max_lag, h, filter_mat[,i], gamma)$cor_vec)
mplot   <- ccf_mat

plot(mplot[, 1], main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec,lty=lwd_vec,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,     lty = 1)   # lag 0
abline(v = max_lag + 1 + h, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Outcomes ─────────────────────────────────────────────────────────
# Left panel (filter coefficients):
#   - Unlike the MSE predictor, the PCS/DFP filters assign non-zero weight
#     to the farthest lag k = q.
#   - Stronger decoupling (smaller alpha0) progressively shifts weight away
#     from recent observations toward the oldest lag. This is counter-intuitive
#     but is a direct consequence of enforcing the CCF slope constraint.
#
# Right panel (CCFs):
#   - The MSE predictors maximize the CCF at their respective forecast horizons.
#   - Enforcing the slope constraint via decoupling works as intended: as
#     alpha0 decreases, the slope between lags 0 and h=1 flattens and eventually
#     inverts, confirming a peak shift toward lag h=1 (violet line).
#   - Increasing the forecast horizon (any admissible htilde<=9) does not 
#     shift the peak of the CCF of the MSE predictor.
#   - The loss in target correlation at lag h=1 is minimised subject to the
#     modified decoupling constraint (efficient frontier).

# Tabular summary: CCF at lag 0 and lag h for each decoupling level
round(cor_vec_1, 2)


# ─────────────────────────────────────────────────────────────────────
# 1.5 Compute DFP-Based LID (Leading Indicators)
# ─────────────────────────────────────────────────────────────────────



# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(x, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)
rownames(y_out_mat) <- names(x)


# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 1
enf <- nrow(y_out_mat)
mplot<-scale(y_out_mat[anf:enf, ])
par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)

# Outcome:
#   As the PCS decoupling weight increases (alpha0 decreases), the predictor
#   output shifts progressively to the left (looks further ahead) relative to
#   the MSE predictor. This visual lead is confirmed quantitatively by the
#   empirical CCFs below.


# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(1, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))


ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))


# ════════════════════════════════════════════════════════════════════
# Exercise 2: LID Based on PCS Optimization
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────
# Two equivalent solution paths to the LID exist:
#
#   Path 1 – Modified DFP (exercise 1):
#     The DFP function can solve this problem by supplying a suitably
#     modified constraint vector, as demonstrated in the sections above.
#
#   Path 2 – Direct PCS (used here):
#     PCS_func() solves the same problem natively. Unlike the DFP
#     approach, PCS_func() is more general (multiple constraints 
#     can be implemented).
#
# In this exercise we use Path 2 to replicate the LID
# solution obtained earlier, confirming that both paths yield the same
# filter coefficients.


# ── 2.1 Full Decoupling ─────────────────────────────────────────────
# Under full decoupling (alpha0 = 0), the MSE-DFP predictor and the PCS
# predictor coincide exactly, so no sign or scale adjustment is needed.

# Set the decoupling parameter to zero (full decoupling)
alpha0 <- 0

# Compute the MSE-DFP filter coefficients using the specified constraint
# vector (gamma_constraint) and the target cross-covariance (gammah)
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# ── PCS hyperparameter settings  ────────────────────────────────────

# Under full decoupling, beta equals alpha0 directly (no rescaling required)
beta <- alpha0

# Use strong regularization to enforce the constraint tightly
lambda <- 100000

# Specify the constraint:
Delta <- h


# Use the true DGP 
gamma_pcs <- gamma

# Compute the PCS filter coefficients: equation 49 in Wildi (2026)
b_pcs_regularized <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)$b

# We can also compute the exact closed-form solution: equations 47 and 48 in Wildi (2026) 
b_pcs_closed_form<- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)$b


# Plot coefficients to verify they overlap
par(mfrow=c(1,1))
ts.plot(cbind(b_dfp, b_pcs_regularized, b_pcs_closed_form), main = "Both Predictors Overlap")

# Note: closed-form PCS and DFP overlap exactly; the regularized PCS is virtually identical: 
# it would coincide exactly when lambda \to \infty (assuming infinite numerical precision).




# ── 2.2 Partial Decoupling ──────────────────────────────────────────
# When alpha0 ≠ 0 (partial decoupling), the DFP and PCS parameterizations
# use different sign conventions and scaling for the slope constraint.
# A manual sign flip and rescaling of beta are therefore required before
# the two filters will agree.

alpha0 <- 0.5

# Compute the MSE-DFP filter coefficients for the partially decoupled case
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# Adjust beta: flip the sign and apply the empirical rescaling factor (2.54 in this example)
# that accounts for the difference in normalization between the two frameworks
beta <- -2.54* alpha0

# Compute the rescaled PCS filter coefficients
# Compute the PCS filter coefficients: equation 49 in Wildi (2026)
b_pcs_regularized <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)$b

# We can also compute the exact closed-form solution: equations 47 and 48 in Wildi (2026) 
b_pcs_closed_form<- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)$b


# Plot both sets of coefficients; they should be nearly identical
ts.plot(cbind(b_dfp, b_pcs_regularized, b_pcs_closed_form), main = "Both Predictors Overlap")




# 2.3 PCS  Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases from k = h-1 to k = h: this might be 
# too extreme for the problem cinsidered here.
beta_vec <- c(-0.1,-0.02,-0.007,-0.002, 0)

# Selecting informative beta values manually can be difficult. PCS_func()
# addresses this by automatically constructing a candidate grid concentrated
# around the tipping point of the PCS optimization — the region where the
# predictor reacts most sensitively to small changes in beta. Screening
# solutions in this neighbourhood often provides the sharpest insight into the
# structure of the optimization problem.
beta               <- 0
Type_III           <- FALSE
scaled_constraints <- FALSE
high_resolution    <- F

PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                    Type_III, scaled_constraints, high_resolution)

# Use the automatically generated grid as the base for subsequent optimisation.
beta_vec_automatic <- PCS_obj$beta_vec
# Focus on negative beta only (positive beta are too extreme in this example)
beta_vec           <- c(beta_vec_automatic[which(beta_vec_automatic<0)],0)




# Constrained lag set: Type I) imposes a positive slope at every lag in Delta, 
# here from 1 to h, enforcing a monotonically increasing CCF (if beta is 
# positive and the problem is feasible) over the full interval
# {0, …, h}. This is the most restrictive of the three PCS types (I, II and III).
Delta <- h

# Very large regularisation weight: drives the solution toward exact
# satisfaction of all h slope constraints simultaneously, producing a CCF
# that increases linearly from k = 0 to k = h with uniform slope
# beta / (b' * b). In practice, this level of regularisation is typically
# more restrictive than necessary and may reduce target correlation unduly
# (see the discussion in Exercises 3.5 and 4).
lambda <- 100000


# ─────────────────────────────────────────────────────────────────────
# 2.4 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h,Delta, gamma_pcs, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the residual of each slope
  # constraint — defined as the deviation from the target value beta —
  # should converge to zero as lambda -> Inf. Each printed value corresponds
  # to the residual for one of the h = 5 constraints. Under large lambda,
  # small residuals confirm feasibility; persistent large residuals would
  # indicate infeasibility. Note that numerical precision imposes a practical
  # lower bound on the achievable residuals: deviations cannot be driven
  # arbitrarily close to zero in finite-precision arithmetic.
  print(abs(d_delta %*% b + beta))
}

# Note: PCS_func() also computes the MSE-optimal PCS:
PCS_obj$b_mse
# The MSE-optimal PCS differs from the 'ordinary' PCS b only by an MSE-optimal
# scaling factor. The ordinary PCS is based on the regularised criterion (46)
# in Wildi (2026), which does not intrinsically scale to optimal MSE performance.

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 2.5 Routine Checks
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

# ── Check 3: Positive target covariance ──────────────────────────────────────
# Verifies that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, confirming a positive target correlation at
# lag h. A negative inner product would indicate sign inversion, rendering
# the predictor unusable.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))

# ─────────────────────────────────────────────────────────────────────
# 2.6 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green", rainbow(ncol(b_mat)))

lwd_vec<-c(2,2,rep(1,ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncate to the first q+1 lags where the MA process has support.
mplot <- scale(filter_mat)
plot(mplot[, 1], main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec,lty=lwd_vec,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i],line = -i, col = colo[i])
abline(h = 0)
abline(v =   1,     lty = 1)   # lag 0
abline(v =   h+1, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Right panel: population CCFs ─────────────────────────────────────
# Vertical lines mark lag 0 (solid) and lag h (dashed).

ccf_mat<-NULL
for (i in 1:ncol(filter_mat))
  ccf_mat<-cbind(ccf_mat,compute_acf_at_lags_zero_delta_func(
    max_lag, h, filter_mat[,i], gamma)$cor_vec)
mplot   <- ccf_mat

plot(mplot[, 1], main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = lwd_vec,lty=lwd_vec,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i],lty=lwd_vec[i],lwd=lwd_vec[i])
abline(h = 0)
abline(v = max_lag + 1,     lty = 1)   # lag 0
abline(v = max_lag + 1 + h, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()


# ─────────────────────────────────────────────────────────────────────
# 2.7 Compute PCS-Based LID (Leading Indicators)
# ─────────────────────────────────────────────────────────────────────



# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(x, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)
rownames(y_out_mat) <- names(x)


# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 1
enf <- nrow(y_out_mat)
mplot<-scale(y_out_mat[anf:enf, ])
par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)

# Outcome:
#   As the PCS decoupling weight increases (alpha0 decreases), the predictor
#   output shifts progressively to the left (looks further ahead) relative to
#   the MSE predictor. This visual lead is confirmed quantitatively by the
#   empirical CCFs below.


# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(1, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))


ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))













#???????????????????????????????????????????????????????
# 2. Interpretability: DFP vs. PCS (Tutorial 4):
#    Even with the frequency-zero re-parameterisation of Exercise 3, the DFP
#    concept remains less directly interpretable than the Peak Correlation 
#    Shifting (PCS) predictor introduced in Tutorial 4, which, in its simplest 
#    form, is defined as:

#      MSE-PCS = gammah + lambda * (gamma_{h-1} - gammah)

#    In PCS, the look-ahead modification weighted by lambda addresses the lead of the predictor 
#    in AGGREGATE, i.e., not only at the trend-frequency omega=zero.  

# Moreover, PCS does not interpolate between gamma_{h-1} and gammah but instead 
# extrapolate, because lambda<0 whenn looking ahead.
#    Note also that because (gamma_{h-1} - gammah) is not proportional to
#    gamma0, AR-inversion no longer yields an identity convolution, so both
#    the MA and AR forms of the PCS predictor involve multiple coefficients
#    varying across designs — more complex than the DFP, but more interpretable.

# Finally, the above simplest form might not suffice to shift the CCF peak: then 
# the more complex PCS is required, i.e., PCS is inherently more complex than 
# just decoupling at present: PCS controls monotonicity of CCF from present to h,
# while DFP only controls present.
#?????????????????????????????????????????????????????????




#?????????????????????????????????????????????????
# MSE-PCS is     MSE-PCS = gammah + lambda * (gamma_{h-1} - gammah)
# Check that this is indeed MSE...
# Problem:
# The weight on gammah should be 1.
# On the other hand the MSE predictor is given by projecting gammah orthogonally to plan 
# spanned by PCS constraint, i.e. above formula.

# Which argument is correct? The following piece shows that MSE-PCS as in paper is correct
#???????????????????????????????????????????????????????
# Note: select either gamma_target<-hp_trend_long[h+1:L]
# or gamma_target<-hp_trend_long[1:L] as targets above.

if (F)
{
  # Verify which PCS is MSE optimal
  k<-ncol(b0_mat)
  k<-4
  b<-b0_mat[,k]
  lambda1<-lambda1_vec[k]
  lambda2<-lambda2_vec[k]
  # 1. MSE-PCS as in paper: gammah+lambda(gamma_{h-1}-gammah) (divide above b0 by lambda1)
  b<-b/lambda1
  mean((b-gamma_target)^2)
  # 1. MSE-PCS with overall weight 1 on gammah: gammah+lambda(gamma_{h-1}-gammah)
  b<-b0_mat[,k]
  b<-b/(lambda1-lambda2)
  mean((b-gamma_target)^2)
  
}








