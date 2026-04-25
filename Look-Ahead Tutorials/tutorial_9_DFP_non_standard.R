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
#      Tutorial 8, the ARMA(1,1) is 'aperiodic', meaning the MSE
#      predictor cannot exploit any phase effect and is effectively
#      'stuck at the present' (i.e., it behaves as a nowcast).
#
#   2. A minor modification applied to the original h-step-ahead MSE
#      predictor (see Section 1.3 below), which restores the standard
#      DFP case.
#
# Modification 1 gives rise to the so-called 'non-standard' case
# discussed in Wildi (2026), Appendix A. This case has two defining
# features:
#
#   (a) The MSE predictor effectively lags the signal at frequency
#       zero — a practically uncommon situation.
#
#   (b) As a consequence, the sign of the optimisation objective must
#       be inverted: the DFP solution is obtained by minimising
#       (rather than maximising) tracking accuracy. This is a highly
#       unusual outcome; see Wildi (2026), Appendix A, for the
#       theoretical background.
#
# Modification 2 reinstates the standard case via a minor adjustment
# to the MSE predictor. However, DFP solutions whose time-shifts
# satisfy the zero-frequency lead constraint may still lag at
# business-cycle frequencies, because the imposed phase restriction
# does not propagate beyond frequency zero.
#
# This suggests that, in some applications, anchoring the DFP
# constraint at frequency zero alone is insufficient to generate an
# effective lead across the full range of policy-relevant frequencies.
#
# Two potential remedies:
#
#   1. Optimise over an aggregate lead measure (rather than a
#      zero-frequency lead constraint alone); see the PCS criterion
#      introduced in Tutorial 10.
#
#   2. Anchor the time-shift constraint at a frequency other than
#      zero — for example, at the business-cycle frequency — to
#      directly target the frequency band of primary interest.
#

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
# 1.3 DFP Settings
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

# Desired lead of the DFP output over the MSE predictor at frequency zero.
# Negative values indicate that the DFP leads the MSE predictor by 
# |lead| time steps at the zero (trend) frequency.
lead_vec <- -2^(0:3)

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

#---------------------------------------------------------------------
# VERIFICATION:

# 1. Compute linear trend
trend<--50:50
# 2. Normalize: disentangle the time-shift effect from the compouned filter effect.
normalized_gammah<-gammah/sum(gammah)
normalized_gamma0<-gamma0/sum(gamma0)
# 3. Apply filters 
trend_h<-filter(trend,normalized_gammah,side=1)
trend_0<-filter(trend,normalized_gamma0,side=1)
# 4. Plot
par(mfrow=c(1,1))
ts.plot(na.exclude(cbind(trend_0,trend_h)),col=c("black","green"),
        main=paste("Outputs of MSE(",h,") (green) and Nowcast (black)",sep=""))
abline(h=0)
# Outcome: the MSE predictor (green) is lagging (right shifted)
# 5. Time difference: lag of MSE predictor
(trend_0-trend_h)[length(trend_0)]
# 6. Matches exactly the difference of time-shifts at frequency zero:
tauh-tau0

#---------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────
# 1.5 Run MSE-DFP Based on Zero-Frequency Lead
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
for (i in 1:length(lead_vec))
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
    # DFP filter onto the nowcast direction (target-correlation numerator)
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

# ─────────────────────────────────────────────────────────────────────
# 1.6 Standard vs. Non-Standard DFP Solutions
# ─────────────────────────────────────────────────────────────────────

# ── Verification of the non-standard case sign correction ─────────────────
# This block manually reconstructs the DFP solution for the first entry of
# lead_vec. Its purpose is to illustrate, step by step, why the sign of the
# DFP objective must be inverted when the non-standard case applies
# (i.e., when tauh > tau0; see Wildi 2026, Appendix A).

# Select the first lead for illustration
k   <- 1
tau <- lead_vec[k]

# ── Step 1: Standard-case DFP ─────────────────────────────────────────────
# In the standard case, the DFP filter takes the form
#
#   b = gammah + lambda0 * gamma0
#
# where lambda0 is chosen so that the frequency-zero time-shift of b
# equals tauh + tau (i.e., the MSE time-shift plus the desired lead).
# The expression below follows directly from the time-shift constraint;
# see Wildi (2026), Theorem 2, for the derivation.
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))

# Construct the standard-case DFP filter
b <- gammah + lambda0 * gamma0

# ── Step 2: Verify the time-shift constraint ──────────────────────────────
# The frequency-zero time-shift of b is its coefficient-weighted centroid.
# The achieved lead over the MSE predictor (taub - tauh) should equal tau;
# the printed residual should therefore be numerically zero.
taub         <- sum((0:(L-1)) * b) / sum(b)   # frequency-zero time-shift of b
lead_dfp_mse <- taub - tauh                   # achieved lead over the MSE predictor
print(lead_dfp_mse - tau)                     # expected output: ~0

# ── Step 3: Diagnose the sign of the target correlation ───────────────────
# In the standard case, the DFP filter maximises the target correlation
# (inner product between b and gammah, normalised to [-1, 1]) subject to
# the time-shift constraint. In the non-standard case (tauh > tau0),
# however, this correlation is NEGATIVE, which means the standard formula
# inadvertently MINIMISES tracking accuracy instead of maximising it.
# The standard DFP solution is therefore misspecified here.
b %*% gammah / sqrt(b %*% b * gammah %*% gammah)   # expected output: negative value

# ── Step 4: Apply the non-standard sign correction ────────────────────────
# Negating both lambda0 and the gammah term flips the objective from
# minimisation back to maximisation, yielding the correct DFP solution
# for the non-standard case (Wildi 2026, Appendix A).
lambda0 <- -lambda0
b       <- -gammah + lambda0 * gamma0

# Confirm that the corrected target correlation is now positive,
# confirming that the filter maximises (rather than minimises) tracking accuracy.
b %*% gammah / sqrt(b %*% b * gammah %*% gammah)   # expected output: positive value

# ── Step 5: Re-verify the time-shift constraint after sign correction ──────
# The sign correction must not disturb the time-shift constraint.
# The residual below should again be numerically zero.
taub         <- sum((0:(L-1)) * b) / sum(b)   # frequency-zero time-shift of corrected b
lead_dfp_mse <- taub - tauh                   # achieved lead over the MSE predictor
print(lead_dfp_mse - tau)                     # expected output: ~0

# ── Geometric interpretation of the sign correction ───────────────────────
# In the standard formulation
#
#   b = gammah + lambda0 * gamma0
#
# a negative lambda0 rotates b away from gammah in the direction opposite to
# gamma0. Geometrically, gammah sits between b and gamma0 in filter space
# (see Tutorial 5, Exercise 1.6). In the standard case (tauh < tau0), gammah
# already leads gamma0, so this rotation introduces an additional lead at
# the cost of a reduced (but still positive) target correlation.
#
# In the non-standard case (tauh > tau0), the roles are reversed: gammah
# LAGS gamma0. Rotating b away from gamma0 in the same direction therefore
# magnifies the existing lag of gammah rather than introducing a lead —
# the opposite of what is intended. Technically, a lead can still be obtained
# via this construction (as demonstrated above), but only at the cost of a
# NEGATIVE target correlation, meaning the filter minimises rather than
# maximises tracking accuracy.
#
# To achieve a genuine LEAD with a POSITIVE TARGET CORRELATION, the rotation
# must go the other way: gamma0 should sit between b and gammah. This is
# accomplished by the sign correction
#
#   b = -gammah + lambda0 * gamma0,   lambda0 > 0
#
# Negating gammah reflects the construction about gamma0, rotating b in the
# opposite direction — away from the lagging gammah and toward (and beyond)
# leading (in relative terms) gamma0. The result is a DFP predictor (filter)
# that leads the MSE predictor (as well as the nowcast gamma0) at frequency
# zero while maintaining a POSITIVE target correlation, as required.
#
# Rotating in the opposite direction (to the standard-case without the 
# sign correction) is equivalent to negating the target
# correlation in the DFP objective. Interestingly, and perhaps
# counterintuitively, this does NOT yield a negative target correlation.
# Instead, the negated objective attains a factually positive value under
# the time-shift constraint — meaning the non-standard case filter does achieve
# a positive correlation with gammah, despite objective minimization, and does 
# so by rotating b in the anti-direction: generating a lead rather than 
# magnifying a pre-existing lag. The time-shift constraint is satisfied in 
# both cases, but only the sign-corrected (minimizing) non-standard solution 
# keeps the target correlation positive.



# ─────────────────────────────────────────────────────────────────────
# 1.7 Compute Complete Decoupling for Additional Reference
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
# 1.8 Check Complete Decoupling
# ─────────────────────────────────────────────────────────────────────

# ── Check: orthogonality ──────────────────────────────────────────────────
# The inner product <b_cd, gamma0> should be numerically zero, confirming
# that the completely decoupled filter is orthogonal to the nowcast gamma0
# by construction (alpha0 = 0 constraint).
t(b_cd) %*% gamma0

# ── Check: orientation (trend direction) ─────────────────────────────────
# The sum of filter coefficients equals the filter gain at frequency zero.
# A negative sum indicates that the fully decoupled DFP inverts the direction
# of any underlying trend or level shift in the data — an undesirable property
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
  
  # The target correlation is (marginally) positive in the standard case.
  b_cd %*% gammah
  
  # However, because tauh > tau0 in this example, the fully decoupled DFP
  # lags even more: tau_cd > tauh.
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
# 1.9 Performance Table
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
#     case (tauh < tau0), a negative lambda places gammah between b and
#     gamma0. In the non-standard case (tauh > tau0) addressed here, lambda
#     is positive, placing gamma0 between gammah and b — the opposite
#     rotation direction required to generate a genuine lead.
#
# alpha0 column:
#   - The DFP constraint parameter <gamma0, b>. It should not be interpreted
#     as a correlation coefficient except when it equals zero. Expressing the
#     DFP constraint in terms of tau (time-shift) rather than alpha0 is
#     generally more interpretable. 


# ─────────────────────────────────────────────────────────────────────
# 1.10 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo     <- c("black", "green", rainbow(ncol(filter_mat) - 2))
col_names <- colnames(filter_mat)

# ── Left panel: filter coefficient profiles ───────────────────────────
# Display the raw (unit-normalised) filter coefficients for every predictor.
# Diagnostic: sum of squared coefficients per filter (filter energy proxy)
mplot <- filter_mat
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
  lines(mplot[, i], col = colo[i])
}

abline(v = 1 + h, lty = 2)   # vertical marker at target horizon h
abline(h = 0)                 # zero-correlation reference line

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Discussion ────────────────────────────────────────────────────────────
#
# CCF panel (right):
#   - MSE predictor (green): the CCF peaks at the target horizon h = 12,
#     confirming MSE optimality. The predictor also correlates strongly with
#     the contemporaneous signal (lag 0), a direct consequence of the
#     aperiodic ARMA(1,1) dynamics.
#   - DFP-shifted filters: as the specified lead increases, the CCF at lag 0
#     declines and the target correlation at h = 12 decreases accordingly.
#     The DFP minimises this loss in target correlation subject to the
#     time-shift constraint.
#
# Filter coefficient panel (left):
#   - All DFP filters are subject to implicit zero-shrinkage (unit-length
#     normalisation). As the specified lead increases (more negative entries
#     in lead_vec), the predictor progressively overweights the most recent
#     innovation — an intuitively sensible strategy for gaining timeliness.
#   - Once the DFP lead more than compensates for the MSE predictor's lag
#     (i.e., the DFP leads in absolute terms), the weights on older lags
#     turn negative (not shown in this plot: one can set lead<--32 to verify 
#     this claim). 



# ─────────────────────────────────────────────────────────────────────
# 1.11 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.11.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.


# All filters are defined in MA form (as applied to the einnovations eps_t in the Wold decomposition)
# Therefore we apply the filters to model residuals.
# Note: example 2.4 in tutorial 6 applied the MA form to x_t instead, which is not optimal.
x_filt   <- arima.obj$residuals

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 1.11.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:



# ─────────────────────────────────────────────────────────────────────
# 1.11 Amplitude and Time-Shifts
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


# Discussion

# Need a more refined design which addresses aggregate lead (not only at frequency 0): PCS



###################################################################################
###################################################################################
###################################################################################
# ════════════════════════════════════════════════════════════════════
# Exercise 2: Same as Exercise 1 but a slight modification of gammah
# ════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────
# 2.1 Load the Data
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
# Log transformation addresses non-stationarity in the variance as the level 
# of the series evolves. 
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y)<-index(PAYEMS["1990::2019"])
plot(y,main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = F,
     xlab = "", ylab = "")
axis(1, at = 1:length(y),
     labels = names(y))
axis(2)
box()

# We consider (stationary) first differences of the log-series:
# The log stabilizes the variance.
# The difference stabilizes the level.
x<-diff(y)

# diff-log PAYEMS is fairly noisy with some strong downturns on recession 
# episodes:
ts.plot(x)
# The dependence structure is similar to above AR(3): slowly monotonically 
# decaying ACF: 
acf(x)

# ─────────────────────────────────────────────────────────────────────
# 2.2 Model Fit
# ─────────────────────────────────────────────────────────────────────

L <- 50   # filter length (number of MA coefficients retained)

# ARMA(1,1): simple model with OK diagnostics 
ar_order<-1
ma_order<-1

arima.obj<-arima(x,order=c(ar_order,0,ma_order))

tsdiag(arima.obj)

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 100 was chosen to ensure that
# the coefficients decay sufficiently close to zero by lag L.
if (ma_order>0)
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = arima.obj$coef[ar_order + 1:ma_order],
    lag.max = length(x)))
} else
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = 0,
    lag.max = length(x)
  ))
  
}


xi[1:(L-1)]/xi[2:L]

# Visualise xi: the slow decay confirms the longer-memory character of the
# post-1990 log-returns relative to the full post-WWII sample.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")

# The theoretical ACF based on the Wold-decomposition (ARMA-model) matches the 
# above empirical ACF.
ts.plot(ARMAacf(ar=0,ma=xi,lag.max=L))

# A slowly monotonically decaying pattern suggest that the MSe predictor will be 
# stuck at present, see tutorial 1.

# Target the series (x) or a smoothed version of the series
if (F)
{
  # Mean over last and next year: acausal filter of length 23  
  L_target<-12*2-1
  gamma_target<-rep(1/L_target,L_target)
  gamma<-conv_two_filt_func(xi,gamma_target)
  
} else
{
  gamma<-xi
}


# ─────────────────────────────────────────────────────────────────────
# 2.3 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# Two forecast horizons are considered:
#   h      — the primary (short) horizon used for the MSE-DFP predictor
#   htilde — a longer horizon used as an additional reference

h <- 12    # primary forecast horizon (one year ahead)
# Longer forecast horizon used as a long-term benchmark
htilde <- 2*h

# Nowcast:
# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah):
# shift gamma by h positions and pad with zeros at the end.
# The trailing zeros reflect the fact that MA coefficients beyond index L
# are negligible for a finite-length filter of length L.

#?????????????
# The first is doable and leads to `interesting' solution that induces a 
# a lead at omega=0 but a lag at business-cycle frequencies
gammah <- c(gamma[h + (1:(L-h))],rep(0,h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L-htilde))],rep(0,htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead_vec <- c(-2,-4,-6,-8,-10,-20,-100,-1000000)
lead_vec <- -2^(0:6)


# ─────────────────────────────────────────────────────────────────────
# 2.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh <- sum((0:(L-1)) * gammah) / sum(gammah)

# In contrast to exercise 1 above, tauh>tau0: standard case
if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
}


# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
# Normalize DFP to unit-length (or not)
# Normalization eases visual inspection below.
unit_length<-T
for (i in 1:length(lead_vec))#i<-1
{
  # Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0,gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    
    # Extract the components returned by the function
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    alpha0 <- as.double(t(gamma0) %*% b)
    
    if (unit_length)
    {
      # Normalise b to unit length to obtain the unitary DFP filter
      b_tau <- b / as.double(sqrt(b %*% b))
    } else
    {
      b_tau<-b
    }
    b_mat<-cbind(b_mat,b_tau)
    lambda_vec<-c(lambda_vec,lambda0)
    alpha_vec<-c(alpha_vec,alpha0)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}







# ─────────────────────────────────────────────────────────────────────
# 2.5 Validation
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



# ─────────────────────────────────────────────────────────────────────
# 2.6 Compute Complete Decoupling for Additional Reference
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




filter_mat<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 2.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# In contrast to previous example, the fully decoupled DFP does not invert trend direction!
sum(b_cd)



# We can compute the time-shift at frequency zero. 
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

# In contrast to exercise 3 above, the fully decoupled design is OK here.
# Also, full decoupling is less extreme than time-shift DFP with large leads




# ─────────────────────────────────────────────────────────────────────
# 2.8 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (positive = right-shift or lag when applied to linear trend)
#   lambda    — DFP regularisation weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value)
#
# Rows correspond to:
#   MSE(h)       — h-step MSE predictor
#   MSE(htilde)  — long-horizon MSE predictor (reference)
#   DFP-shifted  — time-shift DFP with specified lead
#   DFP full dec.— completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 3+length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)",  "lambda", "alpha0")
rownames(mat_perf) <- c(paste("MSE(", h,      ")", sep = ""),
                        paste("MSE(", htilde, ")", sep = ""),
                        paste("DFP-shifted by ",lead_vec,sep=""),
                        "DFP fully decoupled")

# MSE h-step: time-shift and unit-normalised gain; lambda and alpha0 not applicable
mat_perf[1, 1] <- tauh

# Long-horizon MSE: same metrics for htilde
mat_perf[2, 1] <- tauhtilde
# Time-shift DFP and completely decoupled DFP: all metrics available
mat_perf[3:nrow(mat_perf),1]<-c(tau_vec,tau_cd)
mat_perf[3:nrow(mat_perf),2]<-c(lambda_vec,lambda_cd)
mat_perf[3:nrow(mat_perf),3]<-c(alpha_vec,alpha0_cd)


# Note: NAs indicate that the corresponding entries are left blank (not meaningful)
mat_perf

# Discussion:
# 1. tau(0)-column:
#     -The classic h=12 steps ahead MSE predictor has a time-shift of ~13.01
#       Interpretation: when the filter (predictor) is applied to a linear trend, 
#       then the trend will be right-shifted (delayed) by 13 time points.
#     -Increasing the forecast horizon to h=24 (MSE(24) predictor) 
#       reduces the lag: the forecast horizon can be used to some extent to address 
#       timeliness (look ahead) but the effect remains limited overall.
#     -DFP-shifted have time-shifts that differ by lead_vec from MSE(12).
#     -DFP fully decoupled: the time-shift is also well defined since the fully-decoupled DFP
#       does not invert the trend direction, see exercise 1.5 (to be contrasted with 
#       tutorial 6, exercise 1.9 (where the fully decoupled DFP inverts trend direction).
#       Note: the forecast horizon, h=12, is larger which puts some distance between 
#       lag 0 (decoupling) and forecast horizon h=12 (maximization of target correlation). 
#       In a way, the forecast problem is less conflicting (less complex) here.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 5, exercise 1.6. More negative lambda indicate stronger rotation 
#       in the figure of tutorial 5, exercise 1.6.
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 2.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 2.9)
par(mfrow=c(1,2))

colo <- c("black", "green", rainbow(ncol(filter_mat)-2))

# Collect all four filters into a matrix (no scaling applied)
mplot<-filter_mat
col_names <- colnames(filter_mat)
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
max_lag<-20
mplot<-NULL
for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot,compute_ccf_func(filter_mat[,i], gamma0)[L-1+1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[ i])
}
abline(v =  1 + h, lty = 2)
abline(h = 0)

axis(1, at = 1:nrow(mplot),
     labels = -1+1:nrow(mplot))
axis(2)
box()

# Discussion:
# CCF (right panel)
#   -The MSE predictor (green) maximizes the CCF at the target horizon h=12.
#    However, the MSE predictor correlates very strongly with x_t (lag 0).
#   -The tau-shifted DFP correlate less strongly with x_t with increasing lead in lead_vec.
#     As a consequence, the correlation with the target at h=12 decreases.
#     But the DFP minimizes this loss at h=12.
#   -Finally, the fully decoupled (red) has a vanishing CCF at lag 0. As 
#     a consequence, the target correlation at h=12 is pulled down even more strongly, 
#     though it is still maximal subject to the imposed full decoupling.

# ─────────────────────────────────────────────────────────────────────
# 2.10 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 2.10.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.


# All filters are defined in MA form (as applied to the einnovations eps_t in the Wold decomposition)
# Therefore we apply the filters to model residuals.
# Note: example 2.4 in tutorial 6 applied the MA form to x_t instead, which is not optimal.
x_filt   <- arima.obj$residuals

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 2.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
# We observe a lag!!!!!!!!!!!!!!!!!!!!!
# How is that possible??


# ─────────────────────────────────────────────────────────────────────
# 2.11 Amplitude and Time-Shifts
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


# Discussion

# Need a more refined design which addresses aggregate lead (not only at frequency 0): PCS



