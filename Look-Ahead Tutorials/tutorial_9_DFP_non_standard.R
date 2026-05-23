# ════════════════════════════════════════════════════════════════════
# TUTORIAL 9 — DFP: NON-STANDARD CASE
# ════════════════════════════════════════════════════════════════════
#
# A brief overview of the DFP framework is provided in
# tutorial_3_DFP_overview.r.
#
# We revisit the PAYEMS application introduced in Tutorial 8.
# This tutorial introduces two modifications relative to Tutorial 8:
#
#   1. A simpler ARMA(1,1) model: unlike the ARMA(2,2) used in
#      Tutorial 8, the ARMA(1,1) is aperiodic, meaning the MSE
#      predictor cannot exploit any phase effect and therefore behaves
#      as a nowcast — effectively "stuck at present" regardless
#      of the forecast horizon.
#
#   2. A minor modification to the original h-step-ahead MSE
#      predictor, which markedly affects the DFP solution.
#
# Unlike the ARMA(2,2) in Tutorial 8, the aperiodic ARMA(1,1) does
# not support phase hunting: increasing the forecast horizon has no
# effect on lead or lag, so the MSE predictor cannot deliver genuine
# look-ahead behaviour.
#
# The DFP, by contrast, is able to generate a genuine lead even in
# the absence of phase or periodicity. This look-ahead behaviour is
# intrinsic to the optimisation principle: tracking the target
# optimally subject to a time-shift constraint (a lead) at frequency
# zero.
#
#
# ── IMPLICATIONS OF THE TWO MODIFICATIONS ───────────────────────────
#
# Modification 1 (ARMA(1,1) model) gives rise to the so-called non-standard 
# case discussed in Wildi (2026), Appendix A. This case has two defining
# features:
#
#   (a) The MSE predictor effectively LAGS the nowcast at frequency
#       zero — a counterintuitive forecast problem.
#
#   (b) As a consequence, the sign of the optimisation objective must
#       be inverted: the DFP solution is obtained by MINIMISING
#       (rather than maximising) tracking accuracy. This is an
#       unusual outcome predicated on a counterintuitive forecast case; see 
#       Wildi (2026), Appendix A, for the theoretical background.
#
# Modification 2 reinstates the standard case via a minor adjustment
# to the MSE predictor. However, DFP solutions optimised for larger
# leads tend to lag behind the classical MSE predictor — another
# counterintuitive outcome. This effect stems from the inherent
# difficulty of the prediction problem: the DFP exploits every
# available opportunity to satisfy the time-shift constraint while
# maximising target correlation, an uncompromising strategy that can
# lead to overfitting and, consequently, to undesirable lagging
# rather than genuine look-ahead behaviour.
#
# This suggests that, in difficult "stuck at present" forecast problems, which 
# are also potentially prone to overfitting, anchoring the DFP constraint at 
# frequency zero alone is insufficient to generate a meaningful and effective 
# lead across the full range of relevant frequencies.
#
#
# ── POTENTIAL REMEDIES ───────────────────────────────────────────────
#
#   1. Optimise over an aggregate lead measure rather than a
#      zero-frequency constraint alone; see the PCS criterion
#      introduced in Tutorial 10.
#
#   2. Anchor the time-shift constraint at a frequency other than
#      zero — for example, at the business-cycle frequency — to
#      directly target the frequency band of primary interest.


# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────
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

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: DFP — PAYEMS SETTINGS
# ════════════════════════════════════════════════════════════════════

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

# Fit an ARMA(1,1) model: parsimonious specification with adequate diagnostics.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)

# --- Wold Decomposition (MA-infinity representation) ---
# Compute the infinite-order MA coefficients (impulse response weights)
# of the fitted ARMA model. The filter length L ensures that the
# coefficients decay sufficiently close to zero by lag L.
if (ma_order > 0) {
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = arima.obj$coef[ar_order + 1:ma_order],
    lag.max = length(x)))
} else {
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = 0,
    lag.max = length(x)))
}


# Visualise the Wold coefficients: 
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: slowly decaying impulse response (post-1990)")

# Note: the large weight assigned to lag 0, compared to lags k > 0, is responsible 
# for the non-standard case discussed below.

# The theoretical ACF implied by the Wold decomposition matches the
# empirical ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),main="Model-based ACF",ylab="",xlab="Lag")

# A slowly and monotonically decaying ACF pattern suggests that the MSE
# predictor will be 'stuck at the present'; see Tutorial 1.

# Optionally target a smoothed version of x rather than x itself.
if (FALSE) {
  # Acausal moving average over the preceding and following year
  # (symmetric filter of length 23).
  L_target    <- 12 * 2 - 1
  gamma_target <- rep(1 / L_target, L_target)
  gamma        <- conv_two_filt_func(xi, gamma_target)
} else {
  # Default: target the raw differenced series directly.
  gamma <- xi
}


# ─────────────────────────────────────────────────────────────────────
# 1.3 MSE Benchmark(s) and DFP Settings
# ─────────────────────────────────────────────────────────────────────

# Two forecast horizons are considered:
#   h      — the primary (short) horizon for the MSE-DFP predictor.
#   htilde — a longer horizon used as an additional reference.
h      <- 12       # primary forecast horizon (12 months = one year ahead)
htilde <- 2 * h    # extended horizon (24 months = two years ahead)

# Truncate the Wold coefficients to length L to obtain the nowcast
# filter (gamma0).
gamma0 <- gamma[1:L]

# h-step-ahead MSE predictor (gammah):
# Shift gamma forward by h positions:
gammah <- gamma[h + 1:L]                        

# Analogous htilde- (=24) MSE predictor 
gammahtilde <- gamma[htilde + 1:L]

# After scaling, the two MSE predictors align perfectly.
#   - Increasing the forecast horizon cannot generate look ahead behaviour.
ts.plot(scale(cbind(gammah,gammahtilde),scale=T,center=F),
        main="After scaling, 12-step and 24-step MSE predictors overlap")

# Desired lead of the DFP output over the MSE predictor at frequency zero.
# Negative values indicate that the DFP leads the MSE predictor by 
# |lead| time steps at the zero (trend) frequency: trend is left-shifted by
# |lead| compared to MSE predictor. 
# Note: the lead is relative to the MSE predictor. The absolute shift is 
# tauh-|lead|. If tauh-|lead|>0 the DFP is still lagging in absolute terms, 
# but leading when compared to gammah.
lead_vec <- -2^((-1):3)
lead_vec


# ─────────────────────────────────────────────────────────────────────
# 1.4 Time-Shifts: NON-STANDARD CASE
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift for three reference filters:
#   - gammahtilde : the modified MSE predictor (see Section 1.3)
#   - gammah      : the original h-step-ahead MSE predictor
#   - gamma0      : the nowcast filter
# The time-shift at frequency zero is defined as the centroid of the
# filter coefficients; see Tutorial 2, Exercise 3.3.2.
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)
tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)

# Diagnose whether we are in the non-standard case:
# If tauh > tau0, the MSE predictor lags the nowcast at frequency zero,
# which triggers the sign-inversion logic described in Wildi (2026), Appendix A.
if (tauh > tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
}

# ── VERIFICATION ─────────────────────────────────────────────────────────────

# 1. Compute a linear trend.
trend <- -50:50

# 2. Normalize the filters to isolate the time-shift effect from the
#    compounded filter effect.
normalized_gammah <- gammah / sum(gammah)
normalized_gamma0 <- gamma0 / sum(gamma0)

# 3. Apply the normalized filters to the linear trend.
trend_h <- filter(trend, normalized_gammah, side = 1)
trend_0 <- filter(trend, normalized_gamma0, side = 1)

# 4. Plot the filtered outputs.
par(mfrow = c(1, 1))
ts.plot(na.exclude(cbind(trend_0, trend_h)), col = c("black", "green"),
        main = paste("Outputs of MSE(", h, ") (green) and Nowcast (black)", sep = ""))
abline(h = 0)
# Outcome: the MSE predictor (green) is lagging (right-shifted).

# 5. Compute the time difference as the lag of the MSE predictor
#    relative to the nowcast at the last available observation.
(trend_0 - trend_h)[length(trend_0)]

# 6. Confirm that this matches the difference of time-shifts at frequency zero.
tauh - tau0


# ─────────────────────────────────────────────────────────────────────
# 1.5 Run MSE-DFP Based on (Zero-Frequency) Lead Constraint
# ─────────────────────────────────────────────────────────────────────

# ── Compute the time-shift DFP filter for each specified lead ──────────────
# For each target lead in lead_vec, call mse_dfp_from_tau_func(), which
# implements the closed-form DFP solution from Theorem 2 of Wildi (2026).
# Results are collected column-wise in b_mat; the corresponding regularisation
# weights and target-correlation values are stored in lambda_vec and alpha_vec.

# If unit_length = TRUE, each DFP filter vector is rescaled to unit Euclidean
# norm before storage. Normalisation eases visual comparison across leads.
unit_length <- TRUE

b_mat      <- lambda_vec <- alpha_vec <- NULL
for (i in 1:length(lead_vec))#i<-1
{
  # Target lead of the DFP filter over the MSE predictor at frequency zero
  lead <- lead_vec[i]
  
  # Solve for the DFP filter that achieves the specified lead
  # A Warning is issued in the non-standard case.
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    # ── Unpack the output of mse_dfp_from_tau_func ──────────────────────
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of the nowcast (gamma0)
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of the MSE predictor (gammah)
    lambda0 <- dfp_obj$lambda0  # DFP regularisation weight (scalar multiplier on gamma0)
    b       <- dfp_obj$b        # raw DFP filter coefficients (length-L vector)
    
    # Compute the inner product <gamma0, b>: measures the projection of the
    # DFP filter onto the nowcast direction.
    alpha0 <- as.double(t(gamma0) %*% b)
    
    # Optionally normalise b to unit Euclidean length
    if (unit_length)
    {
      b_tau <- b / as.double(sqrt(b %*% b))
    } else
    {
      b_tau <- b
    }
    
    # Accumulate results across leads
    b_mat      <- cbind(b_mat, b_tau)
    lambda_vec <- c(lambda_vec, lambda0)
    alpha_vec  <- c(alpha_vec, alpha0)
    
  } else
  {
    # mse_dfp_from_tau_func returns NULL when gamma0 and gammah are nearly
    # collinear; the DFP problem is then numerically ill-conditioned.
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}

# Note: the function mse_dfp_from_tau_func() warns that the case is non-standard.
# Internally, the problem is addressed by inverting the direction of optimisation.

# ─────────────────────────────────────────────────────────────────────
# 1.6 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved leads matches the specified leads ---
tau_vec<-NULL
for (i in 1:length(lead_vec))
{
  taub <- sum((0:(L-1)) * b_mat[,i]) / sum(b_mat[,i])  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2
  tau_vec<-c(tau_vec,taub)
  # Actual lead of the DFP over the MSE predictor at frequency zero
  lead_dfp_mse <- taub - tauh
  
  # This difference should be (numerically) zero
  print(lead_dfp_mse - lead_vec[i])
}

# CHECK 2 — Sign/orientation preservation: If the sum of filter weights 
# is strictly positive, the DFP does not reverse
# the direction (sign) of a trend signal (or a fixed (mean) level).
apply(b_mat, 2, sum)
# Note: When the DFP constraint is formulated in terms of the lead at frequency
# zero, as in this exercise, the resulting filter is guaranteed not to invert
# the trend direction. This is a practically useful property of expressing the
# DFP constraint through the lead rather than through alpha0, see Tutorial 6.
# However, full decoupling is not guaranteed: even as lead -> -Inf, the
# predictor may remain positively correlated with x_t (which is the case here). 

# CHECK 3 — Positive Target Covariance

t(b_mat)%*%gammah


# ─────────────────────────────────────────────────────────────────────
# 1.7 Standard vs. Non-Standard DFP Solutions (Tricky!)
# ─────────────────────────────────────────────────────────────────────

# This exercise is delicate and may be skipped on a first reading.
# It illustrates the non-trivial character of the look-ahead DFP design 
# particularly in the non-standard case relevant in this exercise.

# ─────────────────────────────────────────────────────────────────────
# 1.7.1 STANDARD VS. NON-STANDARD CASES
# ─────────────────────────────────────────────────────────────────────

# A formal treatment of the non-standard case is given in Appendix A, Wildi (2026).

# I) Standard case (tauh < tau0):
#   - The MSE predictor gammah leads gamma0 (this is not the case here).
#   - To obtain a DFP lead over gammah, rotate b away from gammah in the
#     direction opposite to gamma0: gammah lies between b and gamma0, i.e.,
#         b = gammah + lambda0 * gamma0,  lambda0 < 0
#     see Exercise 1.6, Tutorial 5.
#   - Critically: gammah is positively weighted and gamma0 negatively weighted.

# II) Non-standard case (relevant here):
#   - To obtain a lead by DFP, rotate b the other way round.
#   - Two sub-cases arise:
#
#   a) b rotates away from gammah towards gamma0:
#      - b lies between gammah and gamma0, approaching gamma0.
#
#   b) b rotates further (still the other way round), now away from gamma0:
#      - gamma0 now lies between b and gammah.

# Case a) => DFP leads gammah but LAGS gamma0.
# Case b) => DFP leads gammah AND gamma0.

# In case a), a lead by b is obtained by coupling more strongly with x_t (not decoupling).
# In case b), a more substantial lead is obtained by effectively decoupling from x_t.

# In the standard case, increasing coupling with x_t is never an option: we always decouple.

# ─────────────────────────────────────────────────────────────────────
# ALGEBRAIC FORMULATIONS
# ─────────────────────────────────────────────────────────────────────

# Non-standard case a) can be expressed as:
#
#   i)  b = gammah + lambda0 * gamma0,  lambda0 > 0
#
#   Critically: both gamma0 and gammah are positively weighted.
#   In this case b lies between gamma0 and gammah.

# Non-standard case b) can be expressed in two equivalent forms:
#
#   ii)  b = gamma0 + lambdah * gammah,  lambdah < 0
#
#   iii) b = -gammah + lambda0 * gamma0,  lambda0 > 0
#
#   In both forms, gamma0 sits between gammah and b.
#   Critically: gammah is negatively weighted and gamma0 is positively weighted.
#
#   In ii)  the roles of gamma0 and gammah are inverted relative to the standard case;
#           this is equivalent to swapping gamma0 and gammah in the objective and constraints.
#
#   In iii) the signs are inverted relative to the standard case;
#           this is equivalent to maximising MSE instead of minimising it. Note that since
#           the weight -1 on gammah is fixed, solving the DFP constraint b %*% gamma0 = alpha0
#           still yields a finite solution despite the maximisation.
#           As a result, maximising MSE is a well-defined criterion under case b) / formulation iii)
#           and does not produce an infinite solution.
#
#   Note: in case b), formulations ii) and iii) are equivalent after suitable implementation;
#         formulation iii) is used in our code.

# ─────────────────────────────────────────────────────────────────────
# MSE OPTIMALITY NOTE
# ─────────────────────────────────────────────────────────────────────

# - In the standard case, b = gammah + lambda0 * gamma0 with lambda0 < 0
#   is always MSE-optimal by design.
#
# - In the non-standard case, formulations i), ii), and iii) are generally
#   NOT MSE-optimal. To recover the MSE-optimal solution, rescale as follows:
#
#       b <- b * (gammah %*% b) / (b %*% b)
#
#   The scaling factor (gammah %*% b) / (b %*% b) ensures MSE optimality
#   in the non-standard case for all three formulations.
# - Positive scaling does not affect the time-shift constraint at frequency zero
#   which is a scale independent lead-time measure.

# ─────────────────────────────────────────────────────────────────────
# TECHNICAL NOTE ON SINGULARITIES
# ─────────────────────────────────────────────────────────────────────

# If b is proportional to gamma0:
#   - In formulation ii), lambdah -> 0 (solution is feasible).
#   - In formulation iii), lambda0  -> Inf (solution is singular in raw form).
# However, the MSE-optimal rescaling b * (gammah %*% b) / (b %*% b)
# resolves the singularity. Both ii) and iii) are therefore feasible, though
# some care may be needed to avoid numerical issues near this boundary.


# ─────────────────────────────────────────────────────────────────────
# 1.7.2 IDENTIFYING THE RELEVANT CASES FOR THIS EXERCISE
# ─────────────────────────────────────────────────────────────────────

# We begin by identifying which sub-cases a) and b) are relevant here,
# by comparing the imposed leads against tau0 - tauh.

# Display the imposed leads and the reference shift difference:
abs(lead_vec)
tauh - tau0

# Observation:
# - The first lead (0.5) is smaller in magnitude than (tauh - tau0);
#   this corresponds to case a): b lies between gamma0 and gammah.
# - All remaining leads are larger in magnitude and correspond to case b):
#   gamma0 lies between gammah and b.

# Note: in both cases the length of b must be corrected for MSE optimality:
#     b <- b * (gammah %*% b) / (b %*% b)
# In the non-standard case the raw scaling from formulations i), ii), or iii)
# is arbitrary and must always be corrected.

# Our function mse_dfp_from_tau_func() differentiates all cases and generates 
# an MSE optimal solution in standard as well as in non-standard cases.


# ─────────────────────────────────────────────────────────────────────
# 1.7.3 Case a): b sits between gammah and gamma0
# ─────────────────────────────────────────────────────────────────────

# Case a) corresponds to the first lead.
k   <- 1
tau <- lead_vec[k]

# ── Step 1: Construct b ───────────────────────────────────────────────────

# Standard formula for lambda0 (Wildi 2026, Theorem 2), under the assumptions:
#   1. sum(gammah) > 0 and sum(gamma0) > 0 (neither filter eliminates or inverts the trend).
#     The function mse_dfp_from_tau_func() checks the conditions (it issues a warning and stops computation)
#   2. abs(tau + tauh - tau0) > 0: b is not proportional to gamma0.
#     The function mse_dfp_from_tau_func() checks the conditions and resolves the potential singularity.
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))

# Verify that lambda0 > 0 (as required by case a); note lambda0 < 0 in the standard case or 
# in the non-standard case b), see 1.7.4 below).
lambda0

# Construct b via formulation i): b lies between gamma0 and gammah
# in the plane spanned by (gamma0, gammah).
b <- gammah + lambda0 * gamma0

# ── Step 2: Verify the time-shift constraint ──────────────────────────────

# The frequency-zero time-shift of b is its coefficient-weighted centroid.
# The achieved lead of b over the MSE predictor (taub - tauh) should equal tau.
# The printed residual should be numerically zero.
taub         <- sum((0:(L-1)) * b) / sum(b)   # Frequency-zero time-shift of b
lead_dfp_mse <- taub - tauh                   # Achieved lead over MSE predictor
print(lead_dfp_mse - tau)                     # Expected: ~0

# ── Step 3: Diagnose the sign of the target correlation ───────────────────

# In case a), the correlation is positive:
b %*% gammah / sqrt(b %*% b * gammah %*% gammah)   # Expected: positive

# ── Step 4: MSE-optimal rescaling ─────────────────────────────────────────

# MSE before rescaling (for reference):
(b - gammah) %*% (b - gammah)

# Compute the optimal scaling factor:
optimal_scaling <- as.double(b %*% gammah / (b %*% b))
b <- b * optimal_scaling

# MSE after rescaling (minimised):
(b - gammah) %*% (b - gammah)


# ─────────────────────────────────────────────────────────────────────
# 1.7.4 Case b): gamma0 sits between gammah and b
# ─────────────────────────────────────────────────────────────────────

# Case b) corresponds to any of the larger leads (k = 2 : length(lead_vec)).
k <- 2
# Clamp k to a valid case-b index:
k <- max(k, 2)
k <- min(k, length(lead_vec))
tau <- lead_vec[k]

# ── Step 1: Compute lambda0 ───────────────────────────────────────────────

# Same formula as in Section 1.7.3 (Wildi 2026, Theorem 2).
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))
# In contrast to case a), lambda0 is now NEGATIVE.
lambda0

# ── Step 2 (Incorrect attempt): Apply formulation i) from case a) ─────────

# Applying formulation i) with a negative lambda0 is incorrect for case b).
b <- gammah + lambda0 * gamma0

# Verify the time-shift constraint (algebraically satisfied, but geometrically wrong):
taub         <- sum((0:(L-1)) * b) / sum(b)
lead_dfp_mse <- taub - tauh
print(lead_dfp_mse - tau)                     # Expected: ~0

# ── Step 3: Diagnose the failure of formulation i) in case b) ────────────

# The target correlation is negative: formulation i) does not apply for case b).
b %*% gammah / sqrt(b %*% b * gammah %*% gammah)   # Expected: negative (incorrect)

# Outcome: applying the case-a formula in case b) yields a negative target correlation.

# ── Step 4: Apply the correct formulation iii) for case b) ───────────────

# Note that -lambda0 > 0, as required by formulation iii).
b <- -gammah - lambda0 * gamma0

# Confirm that the target correlation is now positive:
b %*% gammah / sqrt(b %*% b * gammah %*% gammah)   # Expected: positive

# ── Step 5: Re-verify the time-shift constraint after the sign correction ──

taub         <- sum((0:(L-1)) * b) / sum(b)
lead_dfp_mse <- taub - tauh
print(lead_dfp_mse - tau)                     # Expected: ~0

# ── Step 6: MSE-optimal rescaling ─────────────────────────────────────────

# MSE before rescaling (for reference):
(b - gammah) %*% (b - gammah)

# Compute the optimal scaling factor:
optimal_scaling <- as.double(b %*% gammah / (b %*% b))
b <- b * optimal_scaling

# MSE after rescaling (minimised):
(b - gammah) %*% (b - gammah)

# ── Summary ──────────────────────────────────────────────────────────────────
#
# Standard formulation, assuming tauh < tau0 (MSE predictor gammah leads nowcast 
#   gamma0 at zero-frequency).

# Formulation:

#   b = gammah + lambda0 * gamma0,  lambda0 < 0

#   The negative lambda0 rotates b away from gammah in the direction opposite to
#   gamma0, so that gammah sits between b and gamma0 in the plane spanned by 
#   (gamma0,gammah).
#   If tauh < tau0 (standard case), this rotation yields an additional lead with
#   a positive target correlation.
#
# Non-standard case (tauh > tau0):
#   Here gammah lags gamma0. Rotating b away from gammah on the side
#   opposite to gamma0 — as in the standard case — magnifies the LAG rather than
#   producing a LEAD.
#   To obtain a genuine LEAD with a POSITIVE TARGET CORRELATION, the rotation
#   must go the other way:
#
#   Case a) b lies between gammah and gamma0:
#           b = gammah + lambda0 * gamma0,  lambda0 > 0
#
#   Case b) gamma0 lies between gammah and b:
#           b = -gammah + lambda0 * gamma0,  lambda0 > 0
#
# Comparison — standard vs. non-standard:
#
#   Standard:
#     - lambda0 < 0
#     - b is MSE-optimal by construction
#
#   Non-standard:
#     - lambda0 > 0
#     - Depending on case a) or b), gammah or -gammah participates
#     - b must be rescaled to restore MSE optimality
#
# The non-standard case is intriguing because of the inversions required to
# recover the optimal design: a filter that is both leading and has a positive
# (maximised) target correlation. In particular, swapping gamma0 and gammah in
# the objective and constraints, or replacing minimisation of MSE by
# maximisation, is counter-intuitive. Yet the structure of the problem ensures
# a well-defined solution even under these seemingly contradictory requirements;
# see Appendix A in Wildi (2026).



# ─────────────────────────────────────────────────────────────────────
# 1.8 Compute Complete Decoupling for Additional Reference
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled DFP corresponds to alpha0 = 0, i.e. the DFP filter
# is orthogonal to gamma0. This serves as a reference benchmark alongside the
# time-shift DFP computed above.

alpha0_cd <- 0  # complete decoupling: <gamma0, b_cd> = 0


if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
  print("Supply the sign inverted -gammah to MSE-DFP: minimize target correlation")
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, -gammah, alpha0_cd)
  
} else
{
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
  
}

lambda_cd <- dfp_obj$lambda
b      <- dfp_obj$b
if (unit_length)
{
  # Normalise b to unit length to obtain the unitary DFP filter
  b_cd <- b / as.double(sqrt(b %*% b))
} else
{
  b_cd<-b
}



# ─────────────────────────────────────────────────────────────────────
# 1.9 Check Complete Decoupling
# ─────────────────────────────────────────────────────────────────────

# ── Check: orthogonality ──────────────────────────────────────────────────
# The inner product <b_cd, gamma0> should be numerically zero, confirming
# that the completely decoupled filter is orthogonal to the nowcast gamma0
# by construction (alpha0 = 0 constraint).
t(b_cd) %*% gamma0

# ── Check: orientation (trend direction) ─────────────────────────────────
# The sum of filter coefficients equals the filter gain at frequency zero.
# A negative sum indicates that the fully decoupled DFP inverts the direction
# of an underlying trend or level shift in the data — an undesirable property
# that limits its practical usefulness in the present case.
sum(b_cd)

# ── Check: target covariance ──────────────────────────────────────────────
# The inner product <b_cd, gammah> measures the (unnormalised) target
# covariance. A negative value indicates that complete decoupling does not work 
# in the present non-standard case.
b_cd %*% gammah

# Since the predictor inverts the trend, its shift is not well-defined.
tau_cd<-NA

# Out of curiosity we can compute the fully decoupled DFP based on the standard 
# case: this design will exhibit a very large lag and is therefore useless
if (F)
{
  # ── Reference: fully decoupled DFP in the standard case ─────────────────
  # For completeness, we compute the fully decoupled DFP using the original
  # (un-negated) gammah. This corresponds to the standard case (tauh < tau0)
  # and illustrates the contrast with the non-standard solution above.
  b <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)$b
  
  if (unit_length)
  {
    b_cd <- b / as.double(sqrt(b %*% b))
  } else
  {
    b_cd <- b
  }
  
  # Verify orthogonality to gamma0
  t(b_cd) %*% gamma0
  
  # In the standard case, the sum of coefficients is positive:
  # no trend orientation inversion occurs.
  sum(b_cd)
  
  # The target correlation is positive in the standard case.
  b_cd %*% gammah
  
  # However, because tauh>0 is large in this example, the fully decoupled DFP
  # lags strongly:
  tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
  tau_cd
}

# ── Summary ───────────────────────────────────────────────────────────────
# In this ARMA(1,1) example, enforcing full decoupling in the non-standard
# case asks too much of the predictor. The completely decoupled DFP:
#   (i)  inverts trend direction (sum of coefficients < 0), and
#   (ii) achieves a negative target correlation.
# Both properties make it unsuitable as a practical forecasting tool here.

# We now assemble all relevant predictors, skipping the fully decoupled design
# which is unusable in this example.
filter_mat<-cbind(gamma0,gammah,b_mat)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""))



# ─────────────────────────────────────────────────────────────────────
# 1.10 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise three key performance metrics for each predictor in a single
# table. Columns:
#   tau(0)  — frequency-zero time-shift: positive values indicate a lag
#             when the filter is applied to a linear trend (right-shift).
#   lambda  — DFP regularisation weight on gamma0 (NA for MSE predictors).
#   alpha0  — inner product <gamma0, b>; the DFP constraint value
#             (NA for MSE predictors; zero for the fully decoupled DFP).
#
# Rows:
#   MSE(h)               — h-step-ahead MSE predictor
#   MSE(htilde)          — long-horizon (htilde-step) MSE predictor
#   DFP-shifted (×leads) — time-shift DFP filters, one per entry in lead_vec
#   DFP fully decoupled  — completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 2 + length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)", "lambda", "alpha0")
rownames(mat_perf) <- c(
  paste("MSE(", h,      ")", sep = ""),
  paste("MSE(", htilde, ")", sep = ""),
  paste("DFP-shifted by ", lead_vec, sep = ""))

# MSE h-step: record time-shift only; lambda and alpha0 are not applicable
mat_perf[1, 1] <- tauh

# Long-horizon MSE: record time-shift only
mat_perf[2, 1] <- tauhtilde

# Time-shift DFP filters and fully decoupled DFP: all three metrics available.
# DFP time-shifts are tauh + lead_vec by construction (time-shift constraint).
mat_perf[3:nrow(mat_perf), 1] <- lead_vec + tauh
mat_perf[3:nrow(mat_perf), 2] <- lambda_vec
mat_perf[3:nrow(mat_perf), 3] <- alpha_vec

# NAs in the MSE rows indicate that the corresponding metrics are not defined
# for pure MSE predictors.
mat_perf

# ── Discussion ────────────────────────────────────────────────────────────
#
# tau(0) column:
#   - MSE(12): time-shift ≈ 15. When applied to a linear trend, the predictor
#     delays the trend by approximately 15 time points — a substantial lag that
#     reflects the aperiodic nature of the underlying ARMA(1,1) model.
#   - MSE(htilde): increasing the forecast horizon to htilde does not reduce
#     the lag in this example; the ARMA(1,1) model offers no phase structure
#     that longer horizons could exploit.
#   - DFP-shifted: each filter's time-shift equals tauh + lead, exactly as
#     prescribed by the time-shift constraint (lead_vec entries added to tauh).
#
# lambda column:
#   - The regularisation weight lambda governs the rotation of b relative to
#     gammah in filter space (see Tutorial 5, Exercise 1.6). In the standard
#     case (tauh < tau0), a negative lambda places gammah between b and gamma0.
#     In the non-standard case (tauh > tau0) addressed here, lambda is positive,
#     placing b between gamma0 and gammah (case a: lead = -0.5), or gamma0
#     between gammah and b (case b: leads < -0.5 in the table) — the opposite
#     rotation direction required to generate a genuine lead.
#   - In formulation iii) of the non-standard case, lambda0 can become
#     arbitrarily large when b aligns with gamma0. This effect can be observed
#     for shift = -1 in the table: the imposed lead of -1 is close to the
#     shift between gamma0 and gammah (tau0 - tauh ~ -0.94). At exactly this
#     shift, b would be perfectly aligned with gamma0 (see the discussion
#     in Section 1.7.1 above).

#
# alpha0 column:
#   - The DFP constraint parameter <gamma0, b>. It should not be interpreted
#     as a correlation coefficient except when it equals zero. Expressing the
#     DFP constraint in terms of lead (at frequency zero) rather than alpha0 is
#     generally more interpretable. 


# ─────────────────────────────────────────────────────────────────────
# 1.11 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo     <- c("black", "green", rainbow(ncol(filter_mat) - 2))
col_names <- colnames(filter_mat)

# ── Left panel: filter coefficient profiles ───────────────────────────
# Display original nowcast and MSE as well as unit-length DFP filter coefficients.
mplot <- filter_mat
# Check: sum of squared coefficients per filter (filter energy proxy).
# DFP are unit-length adjusted.
apply(mplot^2, 2, sum)

plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters with colour-coded legend labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lwd=ifelse(colnames(mplot)[i]=="MSE",2,1),lty=ifelse(colnames(mplot)[i]=="MSE",2,1))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the MSE h-step filter on top to ensure visibility
lines(mplot[, 2], col = colo[2])

axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
     labels = c(0, (1:(nrow(mplot) / 10)) * 10))
axis(2)
box()

# ── Right panel: cross-correlation functions (CCF) ────────────────────
# For each predictor, compute the CCF against the nowcast gamma0 at lags
# 0, 1, …, max_lag - 1. A vertical dashed line marks the target horizon h;
# a horizontal line marks zero correlation.
max_lag <- 20
mplot   <- NULL

for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot, compute_ccf_func(filter_mat[, i], gamma0)[L - 1 + 1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lwd=ifelse(colnames(mplot)[i]=="MSE",2,1),lty=ifelse(colnames(mplot)[i]=="MSE",2,1))
}

abline(v = 1 + h, lty = 2)   # vertical marker at target horizon h
abline(h = 0)                 # zero-correlation reference line

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Discussion ───────────────────────────────────────────────────────────────
#
# Filter coefficient panel (left):
#   - The MSE filter (green shaded) decays slowly (the exponential decay rate
#     is governed by a1, which is large).
#   - With increasing lead, the DFP designs assign progressively more weight
#     to the last innovation epsilon_t (recall that we are looking at the 
#     MA form of the predictor: the predictor is applied to epsilon_t, not x_t). 
#     This outcome is intuitively appealing.
#
# CCF panel (right):
#   - MSE predictor (green dashed): the CCF peaks at the target horizon h = 12,
#     confirming MSE optimality. The predictor also correlates strongly with
#     the contemporaneous signal (lag 0), a direct consequence of the
#     aperiodic ARMA(1,1) dynamics.
#   - DFP with small leads (-0.5, -1 and -2): these filters COUPLE MORE STRONGLY
#     with x_t than the MSE predictor. This behaviour — increasing lead
#     accompanied by stronger coupling — CANNOT be observed in the STANDARD case.
#   - DFP-shifted filters: as the specified lead increases, the CCF at lag 0
#     declines and the target correlation at h = 12 decreases accordingly.
#     The DFP minimises this loss in target correlation subject to the
#     time-shift constraint.
#
#   - Compare these findings with Exercise 1.16 (AR form) and 2.8. In the 
#     latter Exercise, the target is almost the same
#     but the predictors are very different, and the CCFs differ accordingly.


# ─────────────────────────────────────────────────────────────────────
# 1.12 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.12.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 1.12.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
# With increasing imposed lead, the DFP predictors are more systematically and 
# strongly left-shifted but noisier (ATS-dilemma). 



# ─────────────────────────────────────────────────────────────────────
# 1.13 Amplitude and Time-Shifts
# ─────────────────────────────────────────────────────────────────────

K      <- 600      # number of frequency grid points
plot_T <- FALSE    # suppress internal plotting; we build a custom plot below
amp_mat<-shift_mat<-NULL
for (i in 1:ncol(filter_mat))
{
  as_obj <- amp_shift_func(K, filter_mat[,i], plot_T)   # time-shift for lagged filter (b1)
  amp_mat<-cbind(amp_mat,as_obj$amp)
  shift_mat<-cbind(shift_mat,as_obj$shift)
}
colnames(amp_mat)<-colnames(shift_mat)<-colnames(filter_mat)


# Plot time-shift functions for both filters across frequencies [0, π]
par(mfrow = c(1, 2))
# Scale amplitudes for better visual inspection.
mplot <- amp_mat
lty_vec<-c(2,2,rep(1,ncol(filter_mat)-2))
plot(mplot[, 1], type = "l", axes = FALSE,lty=lty_vec[1],
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude functions",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lty=lty_vec[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()


mplot <- shift_mat
mplot[which(mplot[,ncol(mplot)]<(-2)),ncol(mplot)]<-NA

plot(mplot[, 1], type = "l", axes = FALSE,lty=lty_vec[1],
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
#mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lty=lty_vec[i])
  #  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Amplitude functions:
#   - Increasing the lead assigns progressively less weight to lower frequencies.
#     This aligns with the filter coefficients plotted in Exercise 1.11: more
#     weight is assigned to epsilon_t, making the predictor increasingly noisy.
#
# Time shifts:
#   - The decrease in time-shift at frequency zero corresponds exactly to
#     lead_vec as imposed by the DFP constraint.
#   - Part of the lead propagates to business-cycle frequencies, so that the
#     forecasts effectively lead at recessions; see Exercise 1.12.


# ─────────────────────────────────────────────────────────────────────
# 1.14 AR Form: AR Inversion of ARMA(1,1)
# ─────────────────────────────────────────────────────────────────────

# AR inversion:

ar_inv <- -ARMAtoMA(ar = -arima.obj$coef[ar_order + 1:ma_order], 
                    ma = -arima.obj$coef[1:ar_order], lag.max = L)
# AR-filter
theta<-c(1,-ar_inv)

# Verify the approach via a known identity:
# Convolving the AR inversion with the Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros). 
conv_two_filt_func(xi, theta)$conv[1:10]


# Visualise theta: 
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR inversion")

# The first weight is always 1 (the weight assigned to x_t)
# The other weights are decaying: the decay is regular
theta[2:L]/theta[1:(L-1)]
# First element matches a1+b1 (up to sign)
arima.obj$coef[ar_order + 1:ma_order]+arima.obj$coef[1:ar_order]
# After that, the exponential decay matches b1 (up to sign)
arima.obj$coef[ar_order + 1:ma_order]

# Having confirmed the identity, we now convolve the AR operator with
# the MSE and DFP predictors (in MA form) to obtain their AR form equivalents.


# ─────────────────────────────────────────────────────────────────────
# 1.15 Convolution of the AR inversion with the Predictors
# ─────────────────────────────────────────────────────────────────────

# a. MSE predictor: convolve the AR operator with the predictors.

# Check: convolution of theta with gamma0 should be the identity: smaller 
# deviations become vanishing with increasing L (length of finite MA and AR inversions)
conv_two_filt_func(theta, gamma0)$conv[1:10]

# b. DFP predictors
filter_mat_ar<-NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar<-cbind(filter_mat_ar,conv_two_filt_func(theta, filter_mat[,i])$conv)

colnames(filter_mat_ar)<-colnames(filter_mat)

# Check: the first column (corresponding to the nowcast gamma0) should be the identity.
filter_mat_ar[1:10,1]

# ─────────────────────────────────────────────────────────────────────
# 1.16 Analysis and Plot of DFP Predictors in AR Form
# ─────────────────────────────────────────────────────────────────────
#

# Assign colors
colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- 10
par(mfrow=c(1,1))
# Plot the first first_lags AR coefficients of each predictor.
ts.plot(
  filter_mat_ar[1:first_lags, ],
  col  = colo,
  main = "DFP (MSE-Optimal) Predictors in AR Form",
  lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)

# Outcome:
# - The MSE predictor in AR-form decays monotonically, following an exponential 
#   form b1^k:
filter_mat_ar[2:L,"MSE"]/filter_mat_ar[1:(L-1),"MSE"]
# - Mild decoupling (small lead) gives more weight to x_t, which is intuitive.
# - Stronger decoupling (larger leads) further increases the weight on x_t
#   but assigns increasingly negative weights to lagged observations.
# -Compare with the MA form  in exercise 1.11.

# ─────────────────────────────────────────────────────────────────────────────
# Technical note on AR form and scaling
# ─────────────────────────────────────────────────────────────────────────────
#
# In the standard case, only the first weight of the AR-form of the DFP is
# affected (see Exercise 2.3, Tutorial 5). In the non-standard case this rule
# does not hold in general: it depends on whether the MSE-optimal scaling is
# applied and on whether case a) or case b) holds (see Exercise 1.7.1).
#
# We show here that when the MSE-optimal scaling is NOT applied, only the first
# weight is affected for all case b) designs.

# ── Step 1: Recompute the unscaled DFP predictors ────────────────────────────
b_mat_unscaled <- NULL
for (i in 1:length(lead_vec))
{
  # Lead over the MSE predictor at frequency zero
  lead <- lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    b              <- dfp_obj$b_unscaled   # Raw (unscaled) DFP filter coefficients
    b_mat_unscaled <- cbind(b_mat_unscaled, b)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}

# ── Step 2: Compute the AR form of the unscaled DFP filters ──────────────────
filter_mat_ar_unscaled <- NULL
for (i in 1:ncol(b_mat_unscaled))
  filter_mat_ar_unscaled <- cbind(filter_mat_ar_unscaled,
                                  conv_two_filt_func(theta, b_mat_unscaled[, i])$conv)

# ── Step 3: Plot ──────────────────────────────────────────────────────────────
# For the unscaled case b) designs, only the first weight differs across filters.
# The case a) design (red line) is an exception, as expected.
ts.plot(filter_mat_ar_unscaled, col = rainbow(ncol(filter_mat_ar_unscaled)),
        main = "Un-scaled (non MSE-optimal) DFP Predictors:
        Only the first weight of unscaled case b) designs is affected.
        Case a), red line, is different, though.")
# Note: the scaling in the last plot is not MSE optimal. The MSE optimal 
# predictors were shown in the previous plot.



##############################################################################

# Introduction to Exercise 2:

# ── Special case: sensitivity of DFP to MSE predictor specification ───────
# We now examine a special sub-case of Exercise 1, obtained by a small
# modification of the MSE predictor. Despite its apparent simplicity,
# this minor change has three substantial consequences for the DFP
# optimisation outcome:
#
#   1. The previous non-standard case (tauh > tau0) morphs into the
#      standard case (tauh < tau0), restoring the conventional DFP
#      optimisation direction without requiring a sign correction.
#
#   2. The fully decoupled DFP design becomes usable: it no longer inverts 
#      trend orientation and achieves a positive target correlation, 
#      making it a viable predictor rather than a degenerate solution, 
#      as in exercise 1 above.
#
#   3. Larger leads imposed at frequency zero do not translate into an
#      effective lead of the predictor at business-cycle or other
#      policy-relevant frequencies — a cautionary finding that motivates
#      the more comprehensive PCS `look ahead' approach introduced in Tutorial 10.

##############################################################################




# ════════════════════════════════════════════════════════════════════
# Exercise 2: Standard Case — Slight Modification of the MSE Predictor
# ════════════════════════════════════════════════════════════════════
#
# This exercise mirrors Exercise 1 in structure but applies a minor
# modification to the MSE predictor that reinstates the standard DFP
# case (tauh < tau0). All other settings (data, model, lead_vec) are
# inherited from Exercise 1; please run Exercise 1 before proceeding.


# ─────────────────────────────────────────────────────────────────────
# 2.1 MSE Predictor Modification
# ─────────────────────────────────────────────────────────────────────
# The modified h-step-ahead MSE predictor is constructed by shifting the
# AR coefficient vector forward by h positions and padding with h trailing
# zeros. Relative to the original gammah used in Exercise 1, this
# modification advances the filter's centre of mass toward lag 0, reducing
# the frequency-zero time-shift and thereby reinstating tauh < tau0
# (the standard case).

# Modified h-step-ahead MSE predictor: truncate MA expansion at length L-h.
gammah <- c(gamma[h + (1:(L - h))], rep(0, h))

# Modified long-horizon MSE predictor (same construction, horizon htilde)
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))

# The only difference between Exercise 1 and Exercise 2 is that we now truncate
# the last h weights of the MSE predictor. The intuition is that this modification
# should have negligible effects on the DFP, since the last few weights are small
# and therefore possibly negligible. However, these last few weights are in fact
# non-negligible, and the effect on the DFP will be substantial — a rather
# unexpected result confirming that decoupling is non-trivial.

# ─────────────────────────────────────────────────────────────────────
# 2.2 Compute Frequency-Zero Time-Shifts: A SURPRISING OUTCOME
# ─────────────────────────────────────────────────────────────────────
# Recompute the frequency-zero time-shifts for the modified predictors.

tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)

tauh-tau0
# -------------- A SURPRISING OUTCOME --------------------------

# In contrast to Exercise 1, the modification ensures tauh < tau0, which
# defines the standard DFP case. 


# ─────────────────────────────────────────────────────────────────────
# 2.3 Compute DFP Predictors (Standard Case)
# ─────────────────────────────────────────────────────────────────────
# For each target lead in lead_vec, compute the time-shift DFP filter via
# mse_dfp_from_tau_func(). The function automatically detects whether the
# standard or non-standard case applies and adapts the optimisation sign
# accordingly; no manual sign correction is needed here.

b_mat      <- lambda_vec <- alpha_vec <- NULL

# If unit_length = TRUE, each DFP filter is rescaled to unit Euclidean norm,
# which facilitates visual comparison of coefficient profiles across leads.
unit_length <- TRUE

for (i in 1:length(lead_vec))
{
  # Target lead of the DFP filter over the MSE predictor at frequency zero
  lead <- lead_vec[i]
  
  # Solve for the DFP filter achieving the specified lead
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    # ── Unpack output ────────────────────────────────────────────────
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # regularisation weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    # Inner product <gamma0, b>: projection of the DFP filter onto gamma0
    alpha0 <- as.double(t(gamma0) %*% b)
    
    # Optionally rescale b to unit Euclidean length
    if (unit_length)
    {
      b_tau <- b / as.double(sqrt(b %*% b))
    } else
    {
      b_tau <- b
    }
    
    # Accumulate results across leads
    b_mat      <- cbind(b_mat, b_tau)
    lambda_vec <- c(lambda_vec, lambda0)
    alpha_vec  <- c(alpha_vec, alpha0)
    
  } else
  {
    # mse_dfp_from_tau_func returns NULL when gamma0 and gammah are nearly
    # collinear; the DFP problem is numerically ill-conditioned in this case.
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}


# ─────────────────────────────────────────────────────────────────────
# 2.4 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: verify that each DFP filter achieves its specified lead ──────
# The residual (lead_dfp_mse - lead_vec[i]) should be numerically zero for
# every column of b_mat, confirming that the time-shift constraint is met.
tau_vec <- NULL

for (i in 1:length(lead_vec))
{
  # Frequency-zero time-shift of the i-th DFP filter
  taub         <- sum((0:(L-1)) * b_mat[, i]) / sum(b_mat[, i])
  tau_vec      <- c(tau_vec, taub)
  
  # Lead of the DFP filter over the MSE predictor at frequency zero
  lead_dfp_mse <- taub - tauh
  
  # Residual: expected to be ~0
  print(lead_dfp_mse - lead_vec[i])
}

# ── Check 2: sign / orientation preservation ──────────────────────────────
# A strictly positive sum of filter coefficients confirms that none of the
# DFP filters inverts the direction of a trend or level shift in the data.
# When the DFP constraint is formulated as a lead at frequency zero (rather
# than directly as a constraint on alpha0), orientation preservation is
# guaranteed by construction — a practically useful property discussed in
# Tutorial 6.
# Note: full decoupling is NOT guaranteed. Even as lead → -Inf, the DFP
# predictor may remain positively correlated with x_t in this example.
apply(b_mat, 2, sum)

# CHECK 3 — Positive Target Covariance

t(b_mat)%*%gammah



# ─────────────────────────────────────────────────────────────────────
# 2.5 Compute Completely Decoupled DFP (Reference Benchmark)
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled DFP imposes alpha0 = <gamma0, b> = 0, making the
# filter orthogonal to the nowcast direction. It serves as a reference
# benchmark representing extreme decoupling. 

alpha0_cd <- 0  # orthogonality constraint

if (tauh > tau0)
{
  # Non-standard case: negate gammah to flip the optimisation sign
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
  print("Supplying sign-inverted -gammah to MSE-DFP: minimising target correlation")
  dfp_obj <- mse_dfp_from_alpha0_func(gamma0, -gammah, alpha0_cd)
} else
{
  # Standard case: pass gammah directly
  dfp_obj <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
}

# Unpack the completely decoupled DFP solution
lambda_cd <- dfp_obj$lambda   # regularisation weight for the CD solution
b         <- dfp_obj$b        # raw CD filter coefficients

# Optionally rescale to unit Euclidean length
if (unit_length)
{
  b_cd <- b / as.double(sqrt(b %*% b))
} else
{
  b_cd <- b
}



# ─────────────────────────────────────────────────────────────────────
# 2.6 Checks: Completely Decoupled DFP
# ─────────────────────────────────────────────────────────────────────

# ── Check: orthogonality ──────────────────────────────────────────────────
# The inner product <b_cd, gamma0> should be numerically zero, confirming
# that the completely decoupled filter satisfies the alpha0 = 0 constraint.
t(b_cd) %*% gamma0

# ── Check: orientation (trend direction) ─────────────────────────────────
# The sum of filter coefficients equals the filter gain at frequency zero.
# In contrast to Exercise 1, the sum is POSITIVE here: the modified MSE
# predictor shifts the problem into the standard case, and  the
# fully decoupled DFP preserves trend orientation without inversion.
sum(b_cd)

# ── Check: frequency-zero time-shift ─────────────────────────────────────
# Because the fully decoupled DFP does not invert trend orientation
# (sum > 0), the time-shift is directly interpretable — unlike in
# Exercise 1, where the sign of the sum required special treatment.
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd


# ── Check: positive target covariance  ─────────────────────────────────────
b_cd%*%gammah

# ── Key observation ───────────────────────────────────────────────────────
# In contrast to Exercise 1, the fully decoupled design is well-behaved
# here: it neither inverts the trend direction nor produces a negative
# target correlation. Furthermore, full decoupling lies within the range
# of solutions attainable by the time-shift DFP: it is less extreme than
# DFP filters with very large specified leads, confirming that alpha0 = 0
# is an interior (rather than outer/extreme) point of the DFP solution space
# in this standard-case example.

# We now assemble all relevant predictors:
filter_mat_2<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat_2)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"Fully decoupled")



# ─────────────────────────────────────────────────────────────────────
# 2.7 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise three key performance metrics for each predictor. Columns:
#   tau(0)  — frequency-zero time-shift: positive values indicate a lag
#             when the filter is applied to a linear trend (right-shift).
#   lambda  — DFP regularisation weight on gamma0 (NA for MSE predictors).
#   alpha0  — inner product <gamma0, b>; the DFP constraint value
#             (NA for MSE predictors; zero for the fully decoupled DFP).
#
# Rows:
#   MSE(h)               — h-step-ahead MSE predictor
#   MSE(htilde)          — long-horizon (htilde-step) MSE predictor
#   DFP-shifted (×leads) — time-shift DFP filters, one per entry in lead_vec
#   DFP fully decoupled  — completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 3 + length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)", "lambda", "alpha0")
rownames(mat_perf) <- c(
  paste("MSE(", h,      ")", sep = ""),
  paste("MSE(", htilde, ")", sep = ""),
  paste("DFP-shifted by ", lead_vec, sep = ""),
  "DFP fully decoupled"
)

# MSE h-step: record time-shift only; lambda and alpha0 are not applicable
mat_perf[1, 1] <- tauh

# Long-horizon MSE: record time-shift only
mat_perf[2, 1] <- tauhtilde

# Time-shift DFP and completely decoupled DFP: all three metrics available
mat_perf[3:nrow(mat_perf), 1] <- c(tau_vec,    tau_cd)
mat_perf[3:nrow(mat_perf), 2] <- c(lambda_vec, lambda_cd)
mat_perf[3:nrow(mat_perf), 3] <- c(alpha_vec,  alpha0_cd)

# NAs in the MSE rows indicate that the corresponding metrics are not
# defined for pure MSE predictors.
mat_perf

# ── Discussion ────────────────────────────────────────────────────────────
# The table mirrors Exercise 1 in structure but differs in two important
# respects:
#
#   1. MSE(htilde=24) now has a strictly smaller time-shift than MSE(h=12):
#      the predictor modification allows the longer forecast horizon to
#      meaningfully reduce the frequency-zero lag, a gain that was absent
#      in the original ARMA(1,1) setting of Exercise 1, see 1.10.
#
#   2. The fully decoupled DFP yields a usable predictor with an effective
#      absolute lead at frequency zero (tau_cd < tau0), without inverting
#      trend direction or producing a negative target correlation — in
#      direct contrast to the degenerate outcome observed in Exercise 1.


# ─────────────────────────────────────────────────────────────────────
# 2.8 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo      <- c("black", "green", rainbow(ncol(filter_mat_2) - 2))
col_names <- colnames(filter_mat_2)

# ── Left panel: filter coefficient profiles ───────────────────────────────
# Display the unit-normalised filter coefficients for every predictor.
# The sum of squared coefficients (filter energy) is printed as a diagnostic.
mplot <- filter_mat_2
apply(mplot^2, 2, sum)

plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters with colour-coded legend labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the MSE h-step filter on top to ensure visibility
lines(mplot[, 2], col = colo[2])

axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
     labels = c(0, (1:(nrow(mplot) / 10)) * 10))
axis(2)
box()

# ── Right panel: cross-correlation functions (CCF) ────────────────────────
# For each predictor, compute the CCF against the nowcast gamma0 at lags
# 0, 1, …, max_lag - 1. A vertical dashed line marks the target horizon h;
# a horizontal reference line marks zero correlation.
max_lag <- 20
mplot   <- NULL

for (i in 1:ncol(filter_mat_2))
  mplot <- cbind(mplot, compute_ccf_func(filter_mat_2[, i], gamma0)[L - 1 + 1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
}

abline(v = 1 + h, lty = 2)   # vertical marker at target horizon h
abline(h = 0)                 # zero-correlation reference line

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()


# ── Discussion ────────────────────────────────────────────────────────────────
#
# Filter coefficient panel (left):
#   - Despite the minor modification to the MSE predictor, the DFP filter
#     coefficients look markedly different from those in Exercise 1. 
#   - The CCF in the right panel explains the unexpected profile of the DFP 
#     predictors.
#
# CCF panel (right):
#   - The CCF is roughly inverted relative to Exercise 1.11:
#       - The CCF starts small at lag 0 (strong decoupling).
#       - It then rises quickly.
#       - It arrives large at the forecast horizon h.
#     This means the DFP optimisation performed well: strong decoupling at
#     lag 0 combined with maximal coupling at forecast horizon h.
#   - This double score justifies the seemingly counter-intuitive pattern of
#     the forecast weights in the left panel.
#   - Strong decoupling combined with a large target correlation suggests a
#     predictor with desirable properties.


# Notes:
#   1. DFP filters with very large specified leads can have negative CCFs at 
#      lag 0 (try e.g. lead <- -16, not shown currently), indicating that they
#      are more extreme than the fully decoupled design in this example.
#      Full decoupling (alpha0 = 0) therefore represents an interior solution 
#      in this example (in contrast to exercise 1 above) : it is less 
#      aggressively look-ahead than large-lead DFP filters (e.g. lead < -16).
#   2. The sensitivity to small modifications of the target filter (gammah)
#      illustrates:
#      a) That the forecast problem is complex and difficult.
#      b) That the DFP optimisation fully exploits whatever structure is
#         available in the modified predictor to satisfy the time-shift
#         constraint while maximising tracking accuracy. A small change in
#         the reference filter can therefore open up substantially different
#         regions of the DFP solution space. This behaviour is coupled 
#         to overfitting in the present case.

# ─────────────────────────────────────────────────────────────────────
# 2.9 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────

# ── 2.9.1 Apply Predictors to Data ───────────────────────────────────────
# All filters are expressed in MA form (coefficients applied to the Wold
# innovations eps_t). 

x_filt <- arima.obj$residuals

y_out_mat <- NULL
for (i in 1:ncol(filter_mat_2))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat_2[, i], side = 1))
colnames(y_out_mat) <- col_names

# ── 2.9.2 Plot Predictor Outputs ─────────────────────────────────────────
# Three time windows are shown: the full sample, the dot-com recession
# (obs. 120–170), and the global financial crisis (obs. 200–250).
# Reference filters (nowcast, MSE) are drawn with dashed lines; DFP filters
# with solid lines.

par(mfrow = c(1, 1))

# Full sample
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, rep(1, ncol(filter_mat_2) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Dot-com recession (approx. 2000–2002)
ts.plot(y_out_mat[120:170, ],
        main = "Predictor Outputs — Dot-com Recession", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, rep(1, ncol(filter_mat_2) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Global financial crisis (approx. 2007–2009)
ts.plot(y_out_mat[200:250, ],
        main = "Predictor Outputs — Financial Crisis", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, rep(1, ncol(filter_mat_2) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# ── Discussion ────────────────────────────────────────────────────────────
# Counterintuitively, the DFP filters with the largest specified leads at
# frequency zero appear to LAG the classic MSE predictor in the time-domain
# plots. The amplitude and time-shift analysis in Section 2.10 below reveals
# why: a lead imposed strictly at frequency zero does not necessarily propagate 
# to the business-cycle frequencies that are relevant in this application.


# ─────────────────────────────────────────────────────────────────────
# 2.10 Amplitude and Time-Shift Functions
# ─────────────────────────────────────────────────────────────────────
# Compute and plot the amplitude and time-shift functions for all predictors
# across the full frequency range [0, π]. A zoomed panel then magnifies the
# low-frequency region [0, π/6] to expose behaviour at business-cycle
# frequencies (around π/30, corresponding to a 5-year period).

K      <- 600     # number of frequency grid points
plot_T <- FALSE   # suppress internal plotting in amp_shift_func

amp_mat <- shift_mat <- NULL
for (i in 1:ncol(filter_mat_2))
{
  as_obj    <- amp_shift_func(K, filter_mat_2[, i], plot_T)
  amp_mat   <- cbind(amp_mat,   as_obj$amp)
  shift_mat <- cbind(shift_mat, as_obj$shift)
}
colnames(amp_mat) <- colnames(shift_mat) <- colnames(filter_mat_2)

lty_vec <- c(2, 2, rep(1, ncol(filter_mat_2) - 2))

# ── Full-range plots: amplitude and time-shift ────────────────────────────
par(mfrow = c(1, 2))

# Left: amplitude functions
mplot <- amp_mat
plot(mplot[, 1], type = "l", axes = FALSE, lty = lty_vec[1],
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude Functions",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lty = lty_vec[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

# Right: time-shift functions (extreme negative outliers clipped for clarity)
mplot <- shift_mat
mplot[which(mplot[, ncol(mplot)] < (-2)), ncol(mplot)] <- NA

plot(mplot[, 1], type = "l", axes = FALSE, lty = lty_vec[1],
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-Shift Functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lty = lty_vec[i])
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2); box()

# ── Zoomed plot: time-shift at low frequencies [0, π/6] ──────────────────
# This magnified view reveals how the time-shift constraint at frequency zero
# propagates (or fails to propagate) to the business-cycle band.
par(mfrow = c(1, 1))

plot(mplot[1:(K/6), 1], type = "l", axes = FALSE, lty = lty_vec[1],
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-Shift Functions — Low-Frequency Detail",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[1:(K/6), i], col = colo[i], lty = lty_vec[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 36,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2); box()

# ── Discussion ────────────────────────────────────────────────────────────
# The zoomed time-shift plot exposes the key limitation of anchoring the DFP
# constraint strictly at frequency zero:
#
#   - At omega = 0, the achieved leads match the specified lead_vec entries
#     exactly, confirming that the time-shift constraint is satisfied. 
#
#   - At business-cycle frequencies (around pi/30, corresponding to a
#     5-year cycle), the DFP filters with the largest specified leads
#     actually LAG behind the classic MSE predictor. The lead imposed at
#     frequency zero does not spill over into the economically relevant
#     frequency band.
#
# This is the central cautionary finding of Exercise 2: in this example,
# a larger lead at frequency zero is not only insufficient to generate an
# effective lead at business-cycle frequencies — it actively worsens
# timeliness there. This motivates the aggregate lead criteria (PCS)
# introduced in Tutorial 10, which implicitly optimise for the whole 
# frequency band rather than at a single point.


# ═════════════════════════════════════════════════════════════════════
# Concluding Remarks on the DFP Framework
# ═════════════════════════════════════════════════════════════════════

# ── 1. Structural Form ────────────────────────────────────────────────
# The DFP filter admits a remarkably compact closed form:
#
#   b = lambda_h * gamma_h + lambda_0 * gamma_0
#
# That is, the optimal DFP is always a linear combination of exactly two
# filters/predictors: the h-step MSE filter (gamma_h) and the nowcast
# filter (gamma_0).
#
# Standard case (the MSE predictor leads the nowcast at frequency zero,
# i.e., a trend component is left-shifted):
#   - The pure MSE predictor is recovered by setting lambda_0 = 0
#     (no decoupling penalty) and lambda_h = 1.
#   - Imposing a look-ahead constraint yields lambda_0 < 0, which
#     down-weights the contemporaneous component and induces the
#     desired lead relative to x_t.
#   - With lambda_h = 1 fixed, the DFP is MSE-optimal subject to the
#     imposed decoupling constraint: no other predictor with the same
#     degree of decoupling can achieve a lower MSE.
#
# Non-standard case (the MSE predictor lags the nowcast at frequency zero):
#   - The sign of lambda_h flips, and that of lambda_0 may also flip
#     depending on whether the problem falls into sub-case (a) or
#     sub-case (b) (see main text for details).
#   - In this regime the DFP is no longer MSE-optimal and must be
#     re-scaled to restore optimality.


# ── 2. Core Idea and Computational Complexity ─────────────────────────
# The DFP approach rests on a simple, intuitive principle:
#   reduce the attraction of the forecast toward the nowcast (x_t)
#   while maintaining optimal alignment with the h-step MSE filter (gamma_h).
#
# This dual objective is particularly challenging when gamma_h and gamma_0
# are nearly collinear:
#   - The difficulty of a forecast problem can be characterised by the
#     proximity of gamma_h to gamma_0 in filter space.
#   - A necessary condition for the DFP approach is that gamma_0 and
#     gamma_h are linearly independent, so that they span a plane within
#     which the optimal DFP b resides. If this condition fails, the two
#     defining requirements of the DFP — decoupling from gamma_0 and
#     optimal alignment with gamma_h — cannot be satisfied simultaneously,
#     rendering the DFP construction infeasible.
#
# Optimal target tracking (of gamma_h) subject to decoupling (from gamma_0)
# yields a tractable optimisation problem provided gamma0 and gammah are not 
# collinear:
#   - Unitary DFP  → quadratic optimisation problem.
#   - MSE DFP      → linear optimisation problem.
#
# Despite this apparent simplicity, the structure of look-ahead optimisation
# is considerably more intricate than it first appears, and a number of
# counter-intuitive results can emerge — particularly in difficult forecast
# settings. A striking example from Exercise 1 is the non-standard case,
# in which the DFP criterion effectively maximises (rather than minimises)
# the MSE, an outcome that runs directly counter to classical intuition.
# Such counter-intuitive outcomes typically reflect the conflicting
# requirements of simultaneous decoupling and target tracking, whose
# tension intensifies as the forecast problem becomes more difficult.


# ── 3. When Counter-Intuitive Results Arise ───────────────────────────
# Unexpected behaviour is most prevalent when the underlying forecast
# problem is inherently difficult — i.e., when the classical MSE
# predictor is essentially anchored to the present and unable to look
# ahead effectively (gamma_0 and gamma_h are nearly collinear). 
# In such cases the decoupling constraint may force
# the optimiser into regions of the parameter space that violate
# standard expectations regarding the properties, patterns, and profiles
# of the resulting predictor, a phenomenon akin to overfitting.


# ── 4. Role of Full Decoupling ────────────────────────────────────────
# The fully decoupled DFP (alpha_0 = 0, i.e., zero contemporaneous
# correlation between the predictor and x_t) represents a natural
# extremal design within the DFP family, defining an outer boundary
# for meaningful look-ahead behaviour.
#
# In practice, full decoupling is often too extreme to be directly useful:
#   - It can invert the sign of a non-zero mean level and reverse the
#     direction of a linear trend (as demonstrated in Section 1.3).
#   - It may not be practically attainable in difficult forecast settings.
#
# Even more aggressive designs — requiring *negative* contemporaneous
# correlation with x_t — are theoretically possible but lie outside a sensible
# (interpretable) look-ahead perspective, being rarely of practical relevance.


# ── 5. Interpretability of the Constraint Parameter alpha_0 ───────────
# The interpretation of alpha_0 differs across the two DFP variants:
#
#   - Unitary DFP:  alpha_0 is a *correlation* (bounded in [-1, 1]) and
#                   is therefore directly interpretable as the degree of
#                   contemporaneous coupling between the predictor and x_t.
#
#   - MSE DFP:      alpha_0 is a *covariance*, whose magnitude depends on
#                   the scale of the process, making comparisons across
#                   processes or design choices less straightforward.
#
# To restore interpretability in the MSE variant of the DFP, alpha_0 
# (equivalently, lambda_0, the weight placed on gamma_0) can be re-expressed 
# as an implied *time-shift at frequency zero* (Tutorial 6). This 
# re-parameterisation is meaningful provided that the look-ahead gain at 
# frequency zero spills over to frequencies of practical interest 
# — for example, business-cycle frequencies in macroeconomic applications.


# ── 6. AR Form ────────────────────────────────────────────────────────
# In the standard case, the effect of decoupling on the AR form of the
# MSE-optimal DFP is straightforward: only the first (contemporaneous)
# coefficient is affected, leaving all remaining AR coefficients unchanged.
# Although this result has intuitive appeal, we argue that examining the
# effect of decoupling on the MA form of the DFP is more informative.
#   The MA representation is driven by white-noise innovations (epsilon_t),
#   whose i.i.d. structure ensures that every coefficient directly and
#   transparently reflects the impact of decoupling, without the confounding
#   influence of the serial dependence inherent in the AR-form regressor x_t.
# In the non-standard case, the above simple AR-form of the DFP does not hold 
# in general (depending on cases a and b). 


# ── 7. PCS vs. DFP ────────────────────────────────────────────────────────
# The PCS, to be introduced in Tutorial 10, addresses a different timeliness
# measure: the CCF (see Tutorial 2). In contrast to the time-shift tau at 
# frequency zero — a point measure — the CCF provides an aggregate measure of 
# timeliness, implicitly accounting for the average lead across the entire 
# frequency band.
# Consequently, the PCS constraint system is more general and more complex,
# and the resulting AR-form of the predictor will generally depart from the
# simple lag-zero structure of the DFP in the standard case (or the non-standard 
# case b).
