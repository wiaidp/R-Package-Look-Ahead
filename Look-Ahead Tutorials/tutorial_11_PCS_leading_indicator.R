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

# ── INITIALISATION ────────────────────────────────────────────────────────────
rm(list = ls())

# Load DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)
library(mFilter)

# Install and load the alfred package for direct FRED data access (no API key required).
install.packages("alfred")
library(alfred)


# ── DATA ──────────────────────────────────────────────────────────────────────
# Toggle reload_data to TRUE to fetch fresh data from FRED and overwrite the
# locally saved file; set to FALSE to load the previously saved copy.
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

# Convert to a plain numeric vector.
# xts objects carry implicit index-handling conventions that can interfere with
# downstream computations (e.g., applying a filter to an xts object may silently
# reverse the time axis). Working with plain doubles avoids these pitfalls.
start_year <- 1992
end_year   <- 2024

y     <- as.double(log(GDPC1[paste(start_year, "/", end_year, sep = "")]))
y_xts <-           log(GDPC1[paste(start_year, "/", end_year, sep = "")])
len   <- length(y)



# ── EXPLORATORY PLOTS ─────────────────────────────────────────────────────────
par(mfrow = c(2, 2))
plot(GDPC1,                          main = "US Real GDP (levels)")
plot(y_xts,                          main = "Log GDP")
plot(diff(y_xts),                    main = "Log-differences of GDP")
acf(na.exclude(diff(y_xts)),         main = "ACF of log-differences")


# ── WHITENESS ASSUMPTION (xi = 1) ─────────────────────────────────────────────
# Construct a named numeric vector of log-differences for subsequent analysis.
x        <- na.exclude(as.double(diff(y_xts)))
names(x) <- index(na.exclude(diff(y_xts)))

# The sample ACF of log-GDP-differences is broadly consistent with white noise,
# motivating the assumption xi = 1 (i.e., the Wold decomposition is the identity).
#
# In practice, log-differences of GDP are not strictly white noise:
#   - The sample mean is positive, reflecting long-run trend growth.
#   - Protracted contractions (recessions) introduce low-frequency persistence
#     that a simple white-noise model does not capture.
#
# Nevertheless, for the purpose of trend filtering, xi = 1 is a reasonable
# working assumption: the filter attenuates the high-frequency noise and
# preserves the economically relevant features (trend growth and downturns).


# ── HP FILTER SET-UP ──────────────────────────────────────────────────────────

# Standard HP smoothing parameter for quarterly data.
lambda_hp <- 1600

# Filter half-length: L must be odd so that the symmetric filter is centred
# at position (L - 1)/2 + 1 (not in the middle between two consecutive lags).
L <- 31

# Compute the concurrent (causal, one-sided) HP trend filter.
HP_obj   <- HP_target_mse_modified_gap(2 * (L - 1) + 1, lambda_hp)
# Classic one-sided HP
hp_c <- HP_obj$hp_trend
# Right tail of symmetric HP trend filter
hp_trend <- HP_obj$hp_mse

ts.plot(cbind(hp_trend,hp_c),main=paste("Right half and classic concurrent HP(",lambda_hp,")",sep=""))

# Background
# - The LID should target the two-sided (acausal) HP trend, while ideally
#   being slightly faster (left-shifted, advanced, leading) than its MSE-optimal 
#   one-sided (causal) nowcast.
# - The classic concurrent HP filter, hp_c above, is not an MSE-optimal nowcast 
#   when the data are (close to) white noise.
# - Under white noise, the MSE-optimal HP nowcast is given by the right-tail
#   output of the acausal two-sided HP filter applied to the full sample,
#   i.e., hp_trend.
# - The optimisation is invariant to the choice of target: substituting the
#   two-sided HP filter for its MSE-optimal nowcast hp_trend in the objective
#   function yields the same LID solution.
# - Accordingly, the target adopted here is hp_trend: the MSE-optimal
#   one-sided predictor of the two-sided HP trend.
# - When the data are autocorrelated (xi ≠ 1), the MSE-optimal nowcast is
#   constructed as follows:
#     1. Compute the convolution HP_two ∘ xi, yielding the MA representation
#        of the two-sided HP filter adapted for the autocorrelation structure.
#     2. Replace all MA coefficients assigned to future (not-yet-observed)
#        innovations with zero — their MSE-optimal forecast under linearity.
# - When xi = 1 (white noise), steps 1 and 2 reduce to hp_trend directly.


# ════════════════════════════════════════════════════════════════════════════════
# Exercise 1: DFP-Based LID
# ════════════════════════════════════════════════════════════════════════════════
# The LID (Lead-Indicator Design) imposes a single linear constraint on the
# filter coefficients b:
#
#   b' * (gamma_{h-1} - gamma_h) = beta
#
# This maps onto the standard DFP (Decoupling Filter Problem) framework:
# minimise MSE subject to
#
#   b' * gamma_constraint = alpha0,
#
# where  gamma_constraint = gamma_{h-1} - gamma_h  and  alpha0 = beta.
#
# In the classical DFP, the constraint vector is gamma_0 (decoupling from the
# nowcast). Here it is the difference of two consecutive forecast vectors, so
# the LID problem is not a classical DFP problem, but it can be solved by the
# same DFP optimisation routine.
#
# Interpretation of the constraint vector:
#   gamma_constraint = gamma_{h-1} - gamma_h is a purely algebraic construct,
#   unlike the classical DFP vector gamma_0, which admits a direct physical
#   interpretation as a present-value filter (Decoupling From Present).
#
#   It encodes the CCF slope condition at lag h:
#
#     b' * gamma_constraint = 0   ⟺   CCF(h-1) = CCF(h)   [CCF flat at lag h]
#
#   A flat CCF at lag h implies (in this example at least) that the peak is 
#   shifted rightward from lag 0 toward lag h, indicating look-ahead behaviour 
#   (x as a leading indicator).
# ════════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
# 1.1 DFP Set-Up 
# ─────────────────────────────────────────────────────────────────────

# Leading horizon: one year ahead for quarterly data.
h <- 4

# Extract the target (MSE-optimal) filter coefficients at the relevant lags.
gamma    <- hp_trend            # Full filter (reference)
gamma0   <- hp_trend[1:L]       # Length L Nowcast filter (lag 0)
gammah   <- hp_trend[h   + 1:L] # Length L h-step-ahead MSE predictor
gammahm1 <- hp_trend[h-1 + 1:L] # Length L (h-1)-step-ahead MSE predictor

# Plot the three target filters for visual comparison.
colo <- c("black", "blue", "cyan")
ts.plot(cbind(gamma0, gammahm1, gammah),
        main = "Nowcast, h-step and (h-1)-step MSE predictors",
        col  = colo)

if (FALSE) {
  # Diagnostic checks on the target filter:
  #   (i)  Coefficients of a trend filter should sum to one.
  #   (ii) Root-MSE when applied to standardised white noise.
  sum(hp_trend)
  sqrt(t(hp_trend) %*% hp_trend)
}

# ── LID constraint vector ─────────────────────────────────────────────────────
# Defined as the difference between consecutive MSE predictors.
# The sign convention is arbitrary; reversing it requires reversing the sign
# of alpha0 accordingly.
gamma_constraint <- gammahm1 - gammah

par(mfrow = c(1, 1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == gamma[h-1] - gamma[h]),
        xlab = "Lag", ylab = "",
        sub  = "Algebraic constraint vector encoding the CCF slope condition at lag h")
abline(h = 0)

# ── Baseline decoupling and constraint levels ───────────────────────────────────
# Compute the inner product of the h-step MSE predictor with the constraint
# vector. This serves as a natural upper bound for alpha0: if the DFP constraint
# enforces a stronger decoupling (a smaller alpha0), the leading indicator will 
# look further ahead (left-shift/advancement).
mse_coup <- as.double(gammah %*% gamma_constraint)

# Construct a sequence of decoupling levels alpha0, some above, others below 
# mse_coup. Progressively smaller values enforce a stronger rightward
# shift of the CCF peak toward lag k = h (when alpha0 = 0).
alpha0_vec <- c(0.00123,0.001, 0.00086,mse_coup / 1.5^(1:5), 0)



# ─────────────────────────────────────────────────────────────────────
# 1.2 Run DFP 
# ─────────────────────────────────────────────────────────────────────
# For each alpha0 in alpha0_vec, compute the MSE-optimal filter via the
# closed-form DFP solution (Proposition 1, Wildi 2026):

b_mat       <- NULL   # Filter coefficient matrix  (L × |alpha0_vec|)
cor_vec_mat <- NULL   # Full CCF matrices, one column per alpha0

# CCF values at lags (h-1) and h for tabular summary.
cor_vec_1 <- matrix(ncol = 3, nrow = length(alpha0_vec))

# Number of negative lags to include in the CCF (does not affect DFP calculation).
max_lag <- 1

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Closed-form DFP solution with the LID constraint vector.
  b     <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0
  b_mat <- cbind(b_mat, b)
  
  # Population CCF of b against the process over lags [-max_lag, h].
  cor_vec <- compute_acf_at_lags_zero_delta_func(
    max_lag, h, as.vector(b), gamma)$cor_vec
  
  cor_vec_mat      <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1]  <- cor_vec[1 +max_lag ]   # CCF at lag h-1
  cor_vec_1[i, 2]  <- cor_vec[1 +max_lag + h - 1]   # CCF at lag h-1
  cor_vec_1[i, 3]  <- cor_vec[1 + max_lag+h]        # CCF at lag h
}

# Attach descriptive names to columns and rows.
colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 5))
colnames(cor_vec_1) <- c("Lag 0",paste("Lag", (h - 1):h))
rownames(cor_vec_1) <- paste0("alpha0=", round(alpha0_vec, 8))


# ─────────────────────────────────────────────────────────────────────
# 1.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# Check 1 — Constraint satisfaction:
# The residual  b' * gamma_constraint - alpha0  should be numerically zero
# for every column of b_mat.
t(b_mat) %*% gamma_constraint - alpha0_vec

# Check 2 — Filter orientation:
# A strictly positive sum of filter coefficients ensures that the filter preserves
# the direction of any trend or level shift present in the input data (i.e., an
# upward movement in the input produces an upward movement in the output, and vice
# versa). A negative sum would indicate that the filter inverts such directional
# patterns — effectively flipping the sign of trends or shifts. Here we observe
# that sufficiently small values of alpha0 can cause this inversion, which has
# direct implications for forecast behaviour (see Exercise 1.5).
apply(b_mat, 2, sum)

# Check 3 — Positive target covariance.
# A key distinction of the LID formulation here is that we do not verify that 
# b' * gammah > 0 but b' * gamma0 > 0. Indeed, the target is not the
# h-step-ahead MSE predictor, gammah (as used in DFP and PCS
# applications), but rather the nowcast hp_trend of the two-sided HP trend. 
# Consequently, the LID filter should closely approximate hp_trend (the
# finite-length gamma0) while being left-shifted (i.e., time-advanced) relative
# to it. This anchors the LID to the contemporaneous indicator itself, rather
# than to an h-step-ahead forecast of it (as gammah would imply).
# A non-positive value of b' * gamma0 <= 0 therefore signals misspecification
# of the LID, even if the h-step-ahead criterion b' * gammah > 0 is satisfied.
t(b_mat) %*% gamma0

# ── Collect all filters for downstream comparison ─────────────────────────────
# Columns: nowcast, h-step ahead MSE predictor, classic concurrent HP-C, 
# and LID-constrained variants.
filter_mat <- cbind(gamma0, gammah, hp_c[1:L],b_mat)
colnames(filter_mat) <- c("Nowcast",  paste0("MSE(", h, ")"),
                          "HP-C",paste0("LID: alpha0= ", round(alpha0_vec, 8)))

# ─────────────────────────────────────────────────────────────────────
# 1.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green","violet", rainbow(ncol(b_mat)))

lwd_vec<-c(2,2,2,rep(1,ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Scale to unit length for better visual inspection.
mplot <- scale(filter_mat,center=F,scale=T)
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
#
# Left panel (filter coefficients):
#   - As alpha0 decreases, the filter coefficients decay faster, turn negative 
#     sooner and with greater magnitude (deeper negative swing).
#   - The LID design generalizes HP-MSE and HP-C:
#       • alpha0 = 0.0123  → LID is nearly equivalent to MSE(4).
#       • alpha0 = 0.00086 → LID is virtually identical to HP-C.
#       • alpha0 ∈ (0.00086, 0.0123) → LID interpolates smoothly between
#         MSE(4) and HP-C (e.g., alpha0 = 0.001).
#       • alpha0 < 0.00086  → LID coefficients decay faster than HP-C and
#         exhibit a stronger negative swing.
#       • Tighter approximations to either MSE(4) or HP-C can be achieved
#         by fine-tuning alpha0 accordingly.
#
# Right panel (cross-correlation functions, CCFs):
#   - The CCFs of MSE(4) and HP-C are both replicable by the LID to within
#     negligible deviation. Both CCFs peak at lag k = 0.
#   - Decreasing alpha0 progressively flattens the CCF; at the limit alpha0 = 0,
#     the CCF peak shifts to h = 4, as expected by construction.
#   - A peak shifted to the right of lag 0 introduces look-ahead behaviour,
#     as illustrated in the predictor plots below.
#
# CCFs evaluated at lags h-1 and h:
round(cor_vec_1, 2)
# As alpha0 decreases, the difference CCF(4) - CCF(3) decreases in magnitude and 
# eventually vanishes as alpha0 → 0, reflecting a rightward shift of the CCF peak.
# Smaller absolute differences (i.e., a flatter CCF) imply a reduced CCF at
# lag k = 0 (the nowcast correlation) — revealing an inherent trade-off:
# a flatter CCF buys lead time (left-shift or advancement) at the cost of 
# nowcast accuracy.

# ─────────────────────────────────────────────────────────────────────
# 1.5 Compute DFP-Based LID (Leading Indicators)
# ─────────────────────────────────────────────────────────────────────



# Apply each filter to the data.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(x, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)
rownames(y_out_mat) <- names(x)


# Plot the entire history
anf <- 1
enf <- nrow(y_out_mat)
mplot<-scale(y_out_mat[anf:enf, ])
par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "",lwd=lwd_vec,lty=lwd_vec)
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)

# Financial crisis:
anf <- 50
enf <- 80
mplot<-scale(y_out_mat[anf:enf, ])
par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "",lwd=lwd_vec,lty=lwd_vec)
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)


# Outcome:
#   Stronger decoupling from gamma_constraint (smaller alpha0), shifts the 
#   predictor progressively leftwards (looks further ahead) relative to
#   MSE(4) or HP-C. 

# Note: LID designs that approach full decoupling invert the level
# or the trend direction of the signal (see exercise 1.3 above). In such cases, 
# mean centering (i.e., standardization) is recommended, as non-zero baseline 
# levels (due to positive long-term GDP growth) would otherwise undergo 
# sign inversion.



# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))


k<-7
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: ",colnames(y_out_mat)[k],sep=""))

k<-10
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: ",colnames(y_out_mat)[k],sep=""))

k<-ncol(y_out_mat)
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: ",colnames(y_out_mat)[k],sep=""))

# ════════════════════════════════════════════════════════════════════
# Exercise 2: PCS-Based LID
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────
# Two equivalent solution paths to the LID exist:
#
#   Path 1 – Modified DFP (exercise 1):
#     The DFP function can solve this problem by supplying a suitably
#     modified constraint vector, as demonstrated in exercise 1 above.
#
#   Path 2 – Direct PCS (used here):
#     PCS_func() solves the same problem natively. Unlike the DFP
#     approach, PCS_func() is more general (multiple constraints 
#     can be implemented).
#
# In this exercise we use Path 2 to replicate the LID
# solution obtained earlier, confirming that both paths yield the same
# filter coefficients.

# ─────────────────────────────────────────────────────────────────────
# 2.1 Full Decoupling: Verify that DFP and PCS coincide.
# ─────────────────────────────────────────────────────────────────────
# Under full decoupling (alpha0 = 0), the MSE-DFA predictor and the PCS
# predictor coincide exactly, so no sign or scale adjustment is needed.

# Set the decoupling parameter to zero (full decoupling)
alpha0 <- 0

# Compute the MSE-DFA filter coefficients using the specified constraint
# vector (gamma_constraint) and the target cross-covariance (gammah)
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# ── PCS hyperparameter settings ──────────────────────────────────────

# Under full decoupling, beta equals alpha0 directly (no rescaling required)
beta <- alpha0

# Use strong regularisation to enforce the constraint tightly
lambda <- 100000

# Lag set for constraints: Delta = h means a single constraint 
#  b' * (gamma_h-gamma_{h-1}) = beta.
Delta <- h

# Use the true target (HP) autocorrelation structure as the PCS target
gamma_pcs <- gamma

# First PCS: regularized criterion (equation 49 in Wildi (2026)).
b_pcs_regularized <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)$b

# Second PCS: exact closed-form solution (equations 47 and 48 in Wildi (2026))
b_pcs_closed_form <- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)$b

# Plot all three sets of coefficients to verify mutual overlap
par(mfrow = c(1, 1))
ts.plot(cbind(b_dfp, b_pcs_regularized, b_pcs_closed_form),
        main = "Full Decoupling: MSE-DFA, PCS (Regularised) and PCS (Closed-Form) Overlap")

# Note: in the case of full decoupling (peak CCF shifted to k = h = 4), 
# the closed-form PCS and DFP solutions overlap exactly.
# The regularised PCS is virtually identical and would coincide exactly
# as lambda → ∞, assuming sufficient numerical precision.




# ─────────────────────────────────────────────────────────────────────
# 2.2 Partial Decoupling: Verify that DFP and PCS coincide.
# ─────────────────────────────────────────────────────────────────────
# When alpha0 ≠ 0 (partial decoupling), the DFP and PCS parameterizations
# use different sign conventions and scaling for the slope constraint.
# A manual sign flip and rescaling of beta are therefore required before
# the two filters will agree.

alpha0 <- 0.03

# Compute the DFP filter coefficients for the partially decoupled case
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# Adjust beta: flip the sign and apply the empirical rescaling factor (~6.65 in this example)
# that accounts for the difference in normalization between the two frameworks
beta <- -6.65* alpha0

# Compute the rescaled PCS filter coefficients
# Compute the PCS filter coefficients: equation 49 in Wildi (2026)
b_pcs_regularized <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)$b

# We can also compute the exact closed-form solution: equations 47 and 48 in Wildi (2026) 
b_pcs_closed_form<- PCS_closed_form_func(h, Delta, gamma_pcs, L, beta)$b


# Plot both sets of coefficients; they should be nearly identical
ts.plot(cbind(b_dfp, b_pcs_regularized, b_pcs_closed_form), main = "Both Predictors Overlap")


# Having verified that the DFP and PCS solutions are equivalent (under an
# appropriate transformation of the hyperparameters alpha0 and beta), we now
# proceed to evaluate the PCS across a grid of beta values. The grid points
# are generated automatically by PCS_func(), which selects a range of
# potentially relevant values.

# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS  Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF.
# A positive beta would require the CCF to increase from lag k = h-1 to lag
# k = h — an overly strong constraint in this context. Accordingly, the largest
# beta on the grid is zero, which corresponds to a flat CCF slope and yields the
# largest lead (time advancement).
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




# PCS Constraint: a single constraint at k = h.
Delta <- h

# Very large regularisation weight: drives the solution toward nearly exact
# satisfaction of the single constraint. 
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
  # to the residual of the single constraint. Under large lambda,
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

# Check 2 — Filter orientation:
# A strictly positive sum of filter coefficients ensures that the filter preserves
# the direction of any trend or level shift present in the input data (i.e., an
# upward movement in the input produces an upward movement in the output, and vice
# versa). A negative sum would indicate that the filter inverts such directional
# patterns — effectively flipping the sign of trends or shifts. Here we observe
# that the `flat' CCF specification (beta = 0) causes this inversion, which has
# direct implications for forecast behaviour (recall Exercise 1.5).
apply(b_mat, 2, sum)

# Check 3 — Positive target covariance.
# A key distinction of the LID formulation here is that we do not verify that 
# b' * gammah > 0 but b' * gamma0 > 0. Indeed, the target is not the
# h-step-ahead MSE predictor, gammah (as used in DFP and PCS
# applications), but rather the nowcast hp_trend of the two-sided HP trend. 
# Consequently, the LID filter should closely approximate hp_trend (the
# finite-length gamma0) while being left-shifted (i.e., time-advanced) relative
# to it. This anchors the LID to the contemporaneous indicator itself, rather
# than to an h-step-ahead forecast of it (as gammah would imply).
# A non-positive value of b' * gamma0 <= 0 therefore signals misspecification
# of the LID, even if the h-step-ahead criterion b' * gammah > 0 is satisfied.
t(b_mat) %*% gamma0

# ── Collect all filters for downstream comparison ─────────────────────────────
# Columns: nowcast, h-step ahead MSE predictor, classic concurrent HP-C, 
# and LID-constrained variants.
filter_mat <- cbind(gamma0, gammah, hp_c[1:L],b_mat)
colnames(filter_mat) <- c("Nowcast",  paste0("MSE(", h, ")"),
                          "HP-C",paste0("LID: beta= ", round(beta_vec, 8)))

# ─────────────────────────────────────────────────────────────────────
# 2.6 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green","violet", rainbow(ncol(b_mat)))

lwd_vec<-c(2,2,2,rep(1,ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Scale to unit length for better visual inspection.
mplot <- scale(filter_mat,center=F,scale=T)
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
# Note: select either gamma_target<-hp_trend[h+1:L]
# or gamma_target<-hp_trend[1:L] as targets above.

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








