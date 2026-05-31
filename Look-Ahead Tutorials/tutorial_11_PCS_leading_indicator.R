# ════════════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — BUSINESS CYCLE ANALYSIS AND LEADING INDICATOR DESIGN
# ════════════════════════════════════════════════════════════════════════════

# ── MACRO INDICATOR DESIGN ───────────────────────────────────────────────────
#
# This section follows the Leading Indicator Design (LID) framework introduced
# in Wildi (2026), Section 3.3.
#
# Let x_t be a stationary indicator of interest — for example, the first
# difference of a non-stationary macro series such as industrial production,
# employment, income, or GDP. The target signal is defined as:
#
#   Phi' * X_t  (AR form)   ≡   gamma' * Epsilon_t  (MA form)
#
# where Phi and gamma are coefficient vectors of length L,
#       X_t       = (x_t, ..., x_{t-L+1})' is the observation vector, and
#       Epsilon_t = (epsilon_t, ..., epsilon_{t-L+1})' is the innovation vector.
#
# Typical targets Phi (or gamma) include trend, cycle, or seasonal adjustment
# filters, so that Phi' * X_t represents the corresponding signal-growth:
# trend-adjusted, cycle-adjusted, or seasonally adjusted growth.
#
# Notes:
#   1. Useful signals can be derived from acausal two-sided filters. Here we
#      consider the symmetric bi-infinite Hodrick-Prescott (HP) trend filter 
#      as the target. In this case, Phi or gamma represent the optimal causal 
#      approximations:
#        - the MSE-optimal one-sided HP trend filter, or
#        - the classic concurrent HP-C filter.
#   2. Stationary signal-growth is often more relevant to analysts and
#      decision-makers than the non-stationary signal level:
#        - Negative growth indicates contraction; positive growth indicates
#          expansion. This emphasis of growth dynamics is
#          generally more informative than the absolute signal level.
#
# Throughout this tutorial we work in the MA form, where gamma is obtained as
# the convolution of the Wold decomposition xi of x_t with the filter Phi:
#
#   gamma = Phi ∘ xi,   where ∘ denotes convolution.
#
# Let gamma_k denote the MSE-optimal predictor of the target signal at horizon
# k, so that gamma_0 corresponds to the nowcast (k = 0) of a possibly acausal 
# target (e.g., the two-sided HP trend).

# ── LEADING INDICATOR DESIGN (LID) ─────────────────────────────────────────
#
# The optimization problem is:
#
#   Minimize  (b - gamma)' (b - gamma)          [MSE objective]
#   subject to  b %*% (gamma_h - gamma_{h-1}) = beta   [lead constraint]
#
# See section 3.3, Wildi (2026). This problem is related (though not identical) 
# to the Type II PCS approach (see Tutorial 12). 

# The objective minimizes the distance from the causal (nowcast) filter
# gamma; no explicit forecasting step is involved. The hyperparameters
# h > 0 and beta >= 0 in the constraint jointly govern the profile of the 
# cross-correlation function (CCF) at the specified lead:
#
#   - beta > 0 : the CCF peak cannot be located in h-1.
#   - beta = 0 : the CCF is constrained to be flat between lags h-1 and h.
# 
# Under some circumstances, these constraints can determine an effective shift 
# of the CCF peak at k = h (more exactly: between h-1 and h), see examples below.
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
# the CCF peak is naturally shifted to k = h (between h-1 and h) as a direct 
# consequence of the DGP gamma = HP ∘ xi.
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
# preserves the economically relevant features (long term trend growth and 
# `cyclical' downturns).


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

par(mfrow=c(1,1))
ts.plot(cbind(hp_trend,hp_c),main=paste("Right half and classic concurrent HP(",lambda_hp,")",sep=""),
        col=c("green","violet"))
mtext("Right-half HP (MSE optimal under white noise)",line=-1,col="green")
mtext("Classic HP-C",line=-2,col="violet")

# Note on the HP Filter: An AR(2) Design
# The HP filter satisfies an AR(2) difference equation, as illustrated by the
# regression equations below (as L increases, the residuals vanish asymptotically):
summary(lm(hp_c[2+1:L]~hp_c[1+1:L]+hp_c[1:L]))
summary(lm(hp_trend[2+1:L]~hp_trend[1+1:L]+hp_trend[1:L]))

# Background
# - The LID should target the two-sided (acausal) HP trend, while ideally
#   being slightly faster (left-shifted, advanced, leading) than its MSE-optimal 
#   one-sided (causal) nowcast.
# - The classic concurrent HP filter (violet line in above plot), is not an 
#   MSE-optimal nowcast when the data are (close to) white noise.
# - Under white noise, the MSE-optimal HP nowcast is given by the right-tail
#   output of the acausal two-sided HP filter (green line), i.e., hp_trend.
# - The PCS optimisation is invariant to the choice of target: substituting the
#   two-sided HP filter for its MSE-optimal nowcast hp_trend in the objective
#   function yields the same LID solution (under white noise).
# - Accordingly, the target adopted here is hp_trend: the MSE-optimal
#   one-sided predictor of the two-sided HP trend (green line).
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
#   b %*% (gamma_{h-1} - gamma_h) = beta
#
# This maps onto the standard DFP (Decoupling Filter Problem) framework:
# minimise MSE subject to
#
#   b %*% gamma_constraint = alpha0,
#
# where  gamma_constraint = gamma_{h-1} - gamma_h  and  alpha0 = beta.
#
# In the classical DFP, the constraint vector is gamma_0 (decoupling from present,
# represented by the nowcast). Here it is the difference of two consecutive 
# forecast vectors, so the LID problem is not a classical DFP problem, but it 
# can be solved by the same DFP optimisation routine.
#
# Interpretation of the constraint vector:
#   gamma_constraint = gamma_{h-1} - gamma_h is a purely algebraic construct,
#   unlike the classical DFP vector gamma_0, which is the nowcast.
#
#   It encodes the CCF slope condition at lag h:
#
#     b %*% gamma_constraint = 0   ⟺   CCF(h-1) = CCF(h)   [CCF flat at lag h]
#
#   A flat CCF at lag h implies (in this example at least) that the peak is 
#   shifted rightward from lag 0 toward lag h, indicating look-ahead behaviour: 
#   a LEADING INDICATOR.
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
        main = "Nowcast, h=4-step and (h-1=3)-step MSE predictors",
        col  = colo)
mtext("Nowcast",line=-1,col="black")
mtext("MSE(3)",line=-2,col="blue")
mtext("MSE(4)",line=-3,col="cyan")

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
# Note: since the predictors are not normalized (||b|| != 1), the
# rule is not exact — alpha0 < mse_coup does not guarantee stronger decoupling
# of b (PCS) from gamma_constraint — but it serves as a useful practical proxy.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Construct a sequence of decoupling levels alpha0, starting at alpha_0 = max_coup 
# and ending at alpha0 = 0 (full constraint decoupling: flat CCF at h).  
# Progressively smaller alpha0 enforce a stronger rightward
# shift of the CCF peak toward lag k = h (when alpha0 = 0), i.e. a left-shift or 
# advancement of the corresponding LID which becomes effectively leading.
alpha0_vec <- c(mse_coup,0.00087, mse_coup/ 1.5^(1:5), 0)
alpha0_vec


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
# The residual  b %*% gamma_constraint - alpha0  should be numerically zero
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
# b %*% gammah > 0 but b %*% gamma0 > 0. Indeed, the target is not the
# h-step-ahead MSE predictor, gammah (as used in DFP and PCS
# applications), but rather the nowcast hp_trend of the two-sided HP trend. 
# Consequently, the LID filter should closely approximate hp_trend (the
# finite-length gamma0) while being left-shifted (i.e., time-advanced) relative
# to it. This anchors the LID to the contemporaneous indicator itself, rather
# than to an h-step-ahead forecast of it (as gammah would imply).
# A non-positive value of b %*% gamma0 <= 0 therefore signals misspecification
# of the LID, even if the h-step-ahead criterion b %*% gammah > 0 is satisfied.
# Here all LIDs are admissible (positive target correlation).
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
#       • alpha0 = 0.001214  → LID is nearly equivalent to MSE(4) (dashed green line).
#       • alpha0 = 0.00087 → LID is virtually identical to HP-C (dashed violet line).
#       • alpha0 ∈ (0.00087,0.01214) → LID interpolates smoothly between
#         MSE(4) and HP-C.
#       • alpha0 < 0.00087  → LID coefficients decay faster and
#         exhibit a stronger negative swing.
#
# Right panel (cross-correlation functions, CCFs):
#   - The CCFs of MSE(4) and HP-C both peak at lag k = 0.
#   - Decreasing alpha0 progressively flattens the CCF of the LID; at the limit 
#     alpha0 = 0, the CCF peak shifts to h = 4, as expected by construction.
#   - A peak shifted to the right of lag 0 introduces look-ahead behaviour,
#     as illustrated in the predictor plots below.
#
# CCFs evaluated at lags h-1 and h:
round(cor_vec_1, 2)
# As alpha0 decreases, the difference CCF(3) - CCF(4) decreases in magnitude and 
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
# Note: LID designs that approach full decoupling invert the level
# or the trend direction of the signal (see exercise 1.3 above). In such cases, 
# mean centering (or standardization as above) is recommended, as non-zero baseline 
# levels (due to positive long-term GDP growth) would otherwise undergo 
# sign inversion.
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
#   Stronger decoupling from the gamma constraint (smaller alpha0) shifts the
#   predictor progressively leftward (i.e., further ahead in time) relative to
#   MSE(4) or HP-C. Very small values of alpha0 (close to zero) may become
#   difficult to interpret due to excessive lead caused by phase inversion
#   (some LIDs already invert the trend direction or the sign of a fixed,
#   non-vanishing level, see exercise 1.3).


# Empirical CCF:
# Compute empirical CCFs between the (HP-) nowcast and each predictor to
# confirm that the population peak-shift is reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0",sep=""))


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
# Exercise 2: PCS Type II) Based LID
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
# Under full decoupling (alpha0 = 0), the DFP predictor and the PCS (Type II)
# predictor coincide exactly, so no sign or scale adjustment is needed for 
# replication.

# Set the decoupling parameter to zero (full decoupling)
alpha0 <- 0

# Compute the MSE-DFP filter coefficients using the specified constraint
# vector (gamma_constraint) and the target cross-covariance (gammah)
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# ── PCS hyperparameter settings ──────────────────────────────────────

# Under full decoupling, beta equals alpha0 directly (no rescaling required)
beta <- alpha0

# Use strong regularisation to enforce the constraint tightly
lambda <- 100000

# Lag set for constraints: Delta = h means a single constraint 
#  b %*% (gamma_h-gamma_{h-1}) = beta.
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
        main = "Full Decoupling: MSE-DFP, PCS (Regularised) and PCS (Closed-Form) Overlap")

# Note: in the case of full decoupling (alpha0 = 0), 
# the closed-form PCS and DFP solutions overlap exactly.
# The regularised PCS is virtually identical and would coincide exactly
# as lambda → ∞, assuming sufficient numerical precision.




# ─────────────────────────────────────────────────────────────────────
# 2.2 Partial Decoupling: Verify that DFP and PCS Type II coincide.
# ─────────────────────────────────────────────────────────────────────
# When alpha0 ≠ 0 (partial decoupling), the DFP and PCS parameterizations
# use different sign conventions and scaling for the slope constraint.
# A manual sign flip and rescaling of beta are therefore required before
# the two filters will agree.

# Specify alpha0
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
# potentially relevant values (the values are centered at the tipping point, 
# where PCS is most sensitive to changes in beta).

# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS  Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Selecting informative beta values (slope parameter) manually can be difficult. PCS_func()
# addresses this by automatically constructing a candidate grid concentrated
# around the tipping point of the PCS optimization — the region where the
# predictor reacts most sensitively to small changes in beta. Screening
# solutions in this neighbourhood often provides the sharpest insight into the
# structure of the optimization problem.

# Generate automatic grid points: 
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

# Note: a positive beta would require the CCF to increase from lag k = h-1 to lag
# k = h — an overly strong constraint in this context. Accordingly, the largest
# beta on the grid is zero, which corresponds to a flat CCF slope and yields the
# largest lead (time advancement). Larger beta would generate phase inversion in 
# this example.


# PCS Constraint: a single constraint at k = h.
Delta <- h

# Very large regularisation weight: drives the solution toward nearly exact
# satisfaction of the single constraint. 
lambda <- 100000


# ─────────────────────────────────────────────────────────────────────
# 2.4 PCS Type II Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type II) predictor.
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

#==============================================================================
# Notes on PCS Computation
#==============================================================================

# Note 1: Closed-Form vs. Regularized PCS
# ----------------------------------------
# Instead of the regularized criterion, one can alternatively rely on the
# closed-form solution, which is virtually identical when lambda is large.

# Select a beat on the grid:
beta <- beta_vec[length(beta_vec)]

# Closed-form PCS solution
b_pcs_closed_form <- PCS_closed_form_func(
  h, Delta, gamma_pcs, L, beta
)$b

# Regularized PCS solution
b_pcs_regularized <- PCS_func(
  h, Delta, gamma_pcs, L, beta, lambda
)$b

# Sanity check: the difference between both solutions should be negligible 
# (the difference vanishes asymptotically, as lambda increases, provided sufficient
# numerical precision).
max_diff <- max(abs(b_pcs_closed_form - b_pcs_regularized))
cat("Max absolute difference (closed-form vs. regularized):", max_diff, "\n")

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
# b %*% gammah > 0 but b %*% gamma0 > 0. Indeed, the target is not the
# h-step-ahead MSE predictor, gammah (as used in classic DFP and PCS
# applications), but rather the nowcast hp_trend of the two-sided HP trend. 
# Consequently, the LID filter should closely approximate hp_trend (the
# finite-length gamma0) while being left-shifted (i.e., time-advanced) relative
# to it. This anchors the LID to the contemporaneous indicator itself, rather
# than to an h-step-ahead forecast of it (as gammah would imply).
# A non-positive value of b %*% gamma0 <= 0 therefore signals misspecification
# of the LID, even if the h-step-ahead criterion b %*% gammah > 0 is satisfied.
# In this example, all LIDs produce a positive target correlation and are usable:
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

# Discussion of outcome: see exercise 1.4.

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

# Discussion of outcome: see exercise 1.5.



#==============================================================================
# MAIN TAKE AWAYS
#==============================================================================

#------------------------------------------------------------------------------
# 1. PCS vs. DFP
#------------------------------------------------------------------------------
# Classic DFP (Tutorials 1 - 9):
#   - Decouples from the nowcast gamma_0.
#   - Interpreted as a time shift at frequency zero (see Tutorial 6).
#
# Modified DFP (Exercise 1 above):
#   - Decouples from gamma_h - gamma_{h-1} (Type II PCS constraint).
#   - Aims to shift the peak of the cross-correlation function (CCF):
#       an aggregate lead effect that extends beyond frequency zero
#       and captures shifts across all frequencies.
#
# PCS:
#   - Aims to shift the peak of the CCF as an aggregate timing adjustment.
#   - Interpreted as a global time shift affecting all frequencies, so that
#     the aggregate CCF dependence measure is modified (see Tutorial 2).
#   - PCS Types II), III), and IV) can be replicated within a modified DFP setup.
#     Type I) addresses multiple constraints that cannot be handled by the
#     single-constraint DFP framework.



#------------------------------------------------------------------------------
# 2. Interpretability
#------------------------------------------------------------------------------
# - The single PCS constraint considered here enforces a flat CCF at lag k = h.
# - If the process has a single CCF peak, the constraint effectively shifts
#   that peak to k = h (more precisely: between h-1 and h).
# - The CCF peak is one of the timeliness measures introduced in Tutorial 2
#   (Exercise 1.2):
#     * A RIGHT-shift of the CCF peak  <=>  LEFT-shift (advancement) of the predictor.

#------------------------------------------------------------------------------
# 3. LID (Look-ahead Indicator Design)
#------------------------------------------------------------------------------
# - Unlike usual DFP or PCS forecasting (which target the MSE predictor gamma_h), 
#   the LID targets the nowcast gamma_0, anchoring the design to the current 
#   indicator while generating a lead through the single PCS constraint.
# - Anchoring to gamma_0 underscores the importance of tracking the original
#   indicator directly, rather than a derivative such as its forecast.

#------------------------------------------------------------------------------
# 4. Look-Ahead and Inversion
#------------------------------------------------------------------------------
# - Strong look-ahead behavior (large right-shift of the CCF) can generate
#   INVERSION: reversal of trend direction or negative target correlation with
#   the nowcast, despite anchoring at gamma_0.
# - Inversion is undesirable and reduces interpretability.
# - In principle, sign inversion is mitigated by nowcast anchoring; however,
#   assigning excessive weight to potentially misspecified constraints can
#   enforce inversion regardless.

#------------------------------------------------------------------------------
# 5. Closed-Form vs. Strongly Regularized PCS
#------------------------------------------------------------------------------
# FEASIBLE problem (constraint compatible with the DGP):
#   - The closed-form and strongly regularized PCS solutions converge as the
#     regularization weight lambda increases (given sufficient numerical
#     precision).
#
# INFEASIBLE problem (constraint conflicts with the DGP):
#   - The closed-form solution does not always exist.
#   - The regularized solution ALWAYS exists (the problem remains invertible),
#     but will not satisfy the constraint exactly, irrespective of the
#     magnitude of lambda.
#   - In this case, the constraint should be interpreted as a MISSPECIFICATION:
#     assigning excessive weight to it (selecting lambda too large) is not
#     recommended — see the following tutorials for a detailed treatment.
#
# Summary:
#
#   Scenario     | Closed-Form Solution | Regularized Solution | Constraint Satisfied
#   -------------|----------------------|----------------------|---------------------
#   Feasible     | Exists               | Exists               | Exactly (large lambda)
#   Infeasible   | Does NOT exist       | Always exists        | Never exactly

#------------------------------------------------------------------------------
# 6. Problem Difficulty
#------------------------------------------------------------------------------
# Easy case (current tutorial):
#   - The CCF of the process has a single peak: flattening of the CCF at h thus 
#     moves the peak to k = h.
#   - In such a case, the single PCS constraint generates effective look-ahead 
#     behavior (lead) while maintaining optimal tracking of the target (nowcast).
#
#
# Hard / impossible cases (upcoming tutorials):
#   - In some cases the peak of the CCF cannot be moved to k = h.
#   - In some cases the imposed constraint conflicts with the data-generating
#     process (DGP). Assigning too much weight to such a constraint — or
#     relying on the exact closed-form solution — can:
#       * Generate strong losses in target correlation.
#       * Drive target correlation negative, rendering the predictor unusable.
#   - Next tutorials will analyze more challenging and even impossible PCS
#     problems, and demonstrate how to obtain useful look-ahead behavior even
#     under severe misspecification or impossibility.


#------------------------------------------------------------------------------
# Concluding Remark: AR-Form Structure — DFP vs. PCS
#------------------------------------------------------------------------------
#
# DFP (simpler AR form):
#   - The original DFP (Tutorials 1–9) decouples from gamma_0, the nowcast.
#   - In AR form, only the lag-0 weight (assigned to x_t) is affected by the
#     decoupling constraint; lags k = 1, ..., L-1 remain unaffected.
#     (see Tutorial 5, Exercise 2)
#   - This follows because AR-inversion of the constraint in gamma_0 yields the 
#     identity operator, which acts exclusively on the lag-0 weight.
#
# PCS (richer AR form):
#   - The PCS constraint involves differences of the kind gamma_k - gamma_{k-1}.
#   - The AR-forms of these differences generally differ from the identity and
#     affect the ENTIRE lag sequence of the predictor, not only lag 0.
#   - The PCS is therefore richer and more complex, addressing all lags and
#     potentially multiple constraints simultaneously.
#
# Why PCS is more interpretable despite its complexity:
#   - The shift of the CCF peak is an AGGREGATE time-shift measure, covering
#     all frequencies — easier to interpret in practice than the local,
#     zero-frequency shift of the DFP (see Tutorial 6).
#
# Summary comparison:
#
#   Feature                  | DFP                         | PCS
#   -------------------------|-----------------------------|------------------------------
#   Decouples from           | gamma_0 (nowcast)           | PCS constraint vector (single or multiple differences of MSE predictors at various lags)
#   Frequency coverage       | Explicitly zero frequency   | All frequencies (aggregate)
#   AR-form lags affected    | Lag 0 only                  | Generally all lags
#   Interpretability         | Local (freq. zero shift)    | Aggregate (CCF peak shift): generally easier to interpret
#   Complexity               | Lower                       | Higher












