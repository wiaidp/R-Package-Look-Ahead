# ════════════════════════════════════════════════════════════════════
# TUTORIAL 6 — DECOUPLE FROM PRESENT (DFP) PREDICTOR
# PART 3: Time-Shift Constraint and Interpretability
# ════════════════════════════════════════════════════════════════════

# A brief overview on the DFP is provided in tutorial_3_DFP_overview.r

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
# As in tutorial 5, we here discuss the second form: the MSE-DFP.
#----------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────
# MOTIVATION:
# ─────────────────────────────────────────────────────────────────────
# In the MSE-DFP formulation (unlike the unitary DFP of Tutorial 4,
# which imposes a unit-length constraint on the filter), the parameter
# alpha0 corresponds to a covariance rather than a correlation. This
# scale-dependence of the covariance makes the absolute value of alpha0 
# more difficult to interpret: one cannot assess the degree of decoupling 
# from alpha0 alone without knowing the scale of the process.
#
# By contrast, the unitary DFP expresses the effect of alpha0 in terms
# of a scale-independent correlation, providing a more transparent
# and directly comparable measure of decoupling strength.
#
# A second limitation, discussed in Tutorial 5, is that it is not
# immediately clear how enforcing decoupling from the present x_t
# translates into genuine look-ahead behaviour of x_{t+h} — that is, 
# whether and to what extent the MSE-DFP predictor achieves a measurable 
# lead over the classical MSE benchmark.

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
# The following exercises derive and illustrate this frequency-zero DFP
# formulation.
#----------------------------------------------------------------------

# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines
# Provides DFP_compute_lambda_alpha0_func() and related solvers
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, CCF helpers)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED using the alfred library (no API key required).
install.packages("alfred")
library(alfred)



# ════════════════════════════════════════════════════════════════════
# Exercise 1: MSE-DFP
# Interpretable Time-Shift DFP Constraint
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 1.1 AR(3) and DFP Settings
# ─────────────────────────────────────────────────────────────────────
# The example reuses the AR(3) process from tutorial 5.
# This process represents a challenging forecast problem, as its ACF decays
# slowly and monotonically. In such cases, the classical MSE predictor is
# typically trapped at the present, unable to anticipate future movements.

# Two forecast horizons are considered:
#   h      — the primary (short) horizon used for the MSE-DFP predictor
#   htilde — a longer horizon used as an additional reference

L <- 50   # filter length (number of MA coefficients retained)
h <- 3    # primary forecast horizon (steps ahead)

# Longer forecast horizon used as a long-term benchmark
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
gamma <- c(1,ARMAtoMA(ar = c(ar1, ar2, ar3), lag.max = 1000))

# Inspect the first L MA coefficients
# Slowly monotonically decaying
par(mfrow=c(1,1))
ts.plot(gamma[1:L],main="Wold decomposition of AR(3)")

# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead predictor (gammah):
# shift gamma by h positions.
gammah <- gamma[h + 1:L]

# Analogous long MSE forecast: horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))
gammahtilde <- gamma[htilde + 1:L]


# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead <- -2


# ─────────────────────────────────────────────────────────────────────
# 1.2 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────


# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)
dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)

# Extract the components returned by the function
tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
b       <- dfp_obj$b        # raw DFP filter coefficients

# Normalise b to unit length (corresponds to unitary DFP)
b_unit <- b / as.double(sqrt(b %*% b))


# ─────────────────────────────────────────────────────────────────────
# 1.3 Replicate Formula for b
# ─────────────────────────────────────────────────────────────────────
# The frequency-zero time-shift of a filter with coefficients c is
# defined as:
#
#   tau_c = sum_{k=0}^{L-1} k * c_k  /  sum_{k=0}^{L-1} c_k
#
# This quantity equals the phase shift at frequency omega = 0; see
# Tutorial 2, Exercise 3 for a derivation.

# The MSE-DFP predictor is obtained as:
#
#   b <- gammah + lambda0 * gamma0
#
# See Tutorial 5 and Wildi (2026), Proposition 1. The scalar lambda0
# can be linked to the frequency-zero time-shift tau via Theorem 2
# (equation 34) of Wildi (2026). The main implementation steps are
# summarised below.

# --- Step 1: Compute frequency-zero time-shifts of the three filters ---
# (See Tutorial 2 for the definition and interpretation of tau.)

tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)       # nowcast filter (gamma0)
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)       # h-step MSE predictor (gammah)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)  # long-horizon MSE predictor (gammahtilde)

# Display tau0 and tauh:
# tauh < tau0 confirms that the MSE predictor gammah leads the process
# slightly at frequency zero — but the lead is small (approximately 
# half a time unit).
tau0-tauh

# Display tauhtilde:
# Increasing the forecast horizon from h to htilde does not substantially 
# reduce the time-shift, confirming that longer-horizon MSE prediction alone 
# does not resolve the look-ahead problem.
tau0-tauhtilde


# The above leads are minor (approximately half a time unit), and are
# therefore insufficient for practical look-ahead purposes. We impose a
# larger lead via the DFP constraint. The desired lead of the DFP over
# the MSE predictor at frequency zero is specified as:

tau <- lead     # target lead (in time units) to be enforced by DFP
abs(tau)        # absolute magnitude of the desired lead


# --- Step 2. Compute lambda0 from the closed-form expression (Theorem 2, equation 34): ---
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))


# --- Step 3. Construct the DFP predictor/filter ---
b_tau <- gammah + lambda0 * gamma0

# Check: difference should vanish
max(abs(b-b_tau))

# Geometry:
# A negative lambda0 indicates that the DFP predictor b lies on the
# opposite side of gammah from gamma0 — that is, gammah lies between
# b and gamma0 in the filter coefficient space. This configuration
# corresponds to a phase excess: the DFP predictor overshoots the
# MSE benchmark gammah in the direction away from the process filter
# gamma0. See Tutorial 5, Exercise 1.6 for a detailed geometric
# derivation.



# ─────────────────────────────────────────────────────────────────────
# 1.4 Express alpha0 as a Function of tau
# ─────────────────────────────────────────────────────────────────────
# The DFP constraint parameter alpha0 can be expressed as a function of tau, see
# Wildi 2026, corollary 3, equation 36. Note that alpha0 depends on tau through b_tau 

alpha0 <- as.double(t(gamma0) %*% b_tau)
alpha0
# Note: alpha0 is the covariance of b and gamma0 (up to the variance sigma^2 
# of epsilon_t)


# ─────────────────────────────────────────────────────────────────────
# 1.5 Roundtrips: Apply MSE-DFP Based on alpha0
# ─────────────────────────────────────────────────────────────────────

# --- Consistency Check: Recovering b_tau from the MSE-DFP via alpha0 ---
#
# Purpose: verify that the time-shift-based DFP formulation (b_tau)
# is consistent with the covariance-constrained MSE-DFP formulation
# (based on alpha0), and that the linking formulae in Wildi (2026)
# are mutually coherent.
#
# Specifically, we confirm that the two routes to the DFP predictor —
#   (i)  specifying the desired frequency-zero lead tau directly, and
#   (ii) specifying the decoupling threshold alpha0 —
# yield identical filter coefficients b_tau when the correspondence
# established in Theorem 2 (equation 34) of Wildi (2026) is applied.

dfp_obj <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0)

lambda   <- dfp_obj$lambda
b_mse        <- dfp_obj$b

# Difference should vanish
max(abs(b_tau-b_mse))

# Recover lambda0 from alpha0 using the formula in Proposition 1:
#   lambda = (alpha0 - <gamma0, gammah>) / <gamma0, gamma0>
lambda <- as.double((alpha0 - t(gamma0) %*% gammah) / (t(gamma0) %*% gamma0))

# Difference between recovered and original lambda0 should vanish
lambda - lambda0

# Reconstruct the MSE-DFP filter from the recovered lambda (should match b_tau)
b_r <- gammah + lambda * gamma0

max(abs(b_tau-b_r))




# ─────────────────────────────────────────────────────────────────────
# 1.6 Verification of Time-Shift Constraint
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved lead matches the specified lead ---
taub <- sum((0:(L-1)) * b_tau) / sum(b_tau)  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2

# Actual lead of the DFP over the MSE predictor at frequency zero
lead_dfp_mse <- taub - tauh

# This difference should be (numerically) zero
lead_dfp_mse - lead



# ─────────────────────────────────────────────────────────────────────
# 1.7 Comparison with Unitary DFP: Cautionary Note on alpha0
# ─────────────────────────────────────────────────────────────────────
# This section presents a counter-example highlighting the different
# interpretations of alpha0 in the two DFP formulations:
#
#   - Unitary DFP: alpha0 is a correlation (scale-independent) between
#     the predictor output and the nowcast filter gamma0.
#   - MSE-DFP:     alpha0 is a covariance (scale-dependent); it no
#     longer has a direct correlation interpretation.
#
# Imposing the same numerical value of alpha0 in both formulations
# therefore produces two different predictors.

# --- Compute the unitary DFP predictor (naive, incorrect translation) ---
# Passing the MSE-DFP covariance alpha0 directly to unitary_DFP_func()
# is incorrect: the unitary DFP expects a correlation, not a covariance.
# This call is retained for illustration purposes only. If |alpha0|>1 an 
# error message is issued and the function stops.
bu <- unitary_DFP_func(gamma0, gammah, alpha0)$b0

# --- Correct translation: convert alpha0 to a correlation ---
# The unitary DFP constraint requires a scale-independent correlation.
# We obtain alpha0_unitary by normalising the inner product <gamma0, b_tau>
# by the product of the L2-norms of gamma0 and b_tau.
alpha0_unitary <- as.double(t(gamma0) %*% b_tau) /
  sqrt(as.double(t(gamma0) %*% gamma0) *
         as.double(t(b_tau)  %*% b_tau))

bu <- unitary_DFP_func(gamma0, gammah, alpha0_unitary)$b0

# Verify unit-length constraint: should equal 1
t(bu) %*% bu

# Verify unitary DFP decoupling constraint (correlation form): should equal 0
t(bu) %*% gamma0 / sqrt(as.double(t(gamma0) %*% gamma0)) - alpha0_unitary

# Confirm that bu does not satisfy the MSE-DFP covariance constraint:
# the following should NOT equal 0 (different scale)
t(bu) %*% gamma0 / sqrt(as.double(t(gamma0) %*% gamma0)) - alpha0

# Verify MSE-DFP decoupling constraint (covariance form): should equal 0
t(b_tau) %*% gamma0 - alpha0

# --- Compare the two DFP predictors (both rescaled to unit length) ---
ts.plot(cbind(bu, b_tau / sqrt(as.double(t(b_tau) %*% b_tau))),
        col  = c("red", "blue"),
        xlab = "",
        main = "Unitary DFP vs. MSE-DFP: correctly matched constraints")
mtext("Unitary DFP", col = "red",  line = -1)
mtext("MSE-DFP",     col = "blue", line = -2)

# When alpha0 and alpha0_unitary are correctly matched (i.e., they encode
# the same decoupling strength in their respective scales), the two
# predictors differ only up to a unit-length rescaling — confirming that
# the formulations are equivalent once the constraint is expressed on a
# common scale.
#
# By contrast, imposing the same raw numerical value of alpha0 (assuming 
# |alpha0|<1) in both formulations does not impose the same
# degree of decoupling, because alpha0 carries a different meaning in
# each framework (correlation vs. covariance).
#
# Note: the frequency-zero time-shift tau provides the most natural
# common language for specifying decoupling strength across DFP
# variants. If the same tau is imposed on both the unitary DFP and the
# MSE-DFP, the two formulations yield identical filter coefficients up
# to unit-length rescaling, regardless of the differing alpha0 scales.



# ─────────────────────────────────────────────────────────────────────
# 1.8 Compute Complete Decoupling for Additional Reference
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
dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
lambda_cd <- dfp_obj$lambda
b_cd      <- dfp_obj$b

# Normalise to unit length so that b_cd is comparable to b_unit
scale <- as.double(1 / sqrt(t(b_cd) %*% b_cd))
b_cd  <- scale * b_cd


# ─────────────────────────────────────────────────────────────────────
# 1.9 Checks: Complete Decoupling DFP
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
# Notes:
#
# 1. Complete decoupling (alpha0 = 0) does not automatically imply
# that the predictor reverses the trend direction or the level of the
# series. Whether trend reversal occurs may depend on the data generating 
# process or the forecast horizon. Tutorial 8 presents a counter-example 
# in which the fully decoupled DFP preserves the trend direction despite 
# alpha0 = 0.
#
# 2. We can still compute the numerical value of the time-shift formula.
# The result should be interpreted as the shift of the sign-reverted predictor
# (-b_cd), which does have a positive gain at frequency zero, see Wildi 2026, 
# section 4.1.
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd


# ─────────────────────────────────────────────────────────────────────
# 1.10 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#
#   tau(0)   — frequency-zero time-shift: positive values indicate a
#              right-shift (lag/delay) when the filter is applied to
#              a linear trend; negative values indicate a left-shift
#              (lead/advancement).
#   Gamma(0) — gain at frequency zero: the sum of filter weights,
#              normalised to unit filter length. Positive values
#              indicate that the filter preserves the sign of a linear
#              trend; negative values indicate trend reversal.
#   lambda   — DFP scalar weight on gamma0 in the decomposition
#              b = gammah + lambda * gamma0 (Proposition 1,
#              Wildi 2026).
#   alpha0   — inner product <gamma0, b>: the DFP decoupling
#              constraint value, interpretable as a covariance
#              between the predictor and the nowcast filter gamma0.
#
# Rows correspond to:
#   MSE(h)        — h-step ahead MSE predictor (classical benchmark).
#   MSE(htilde)   — long-horizon MSE predictor (timeliness reference).
#   DFP-shifted   — MSE-DFP with a specified frequency-zero lead tau.
#   DFP full dec. — completely decoupled MSE-DFP (alpha0 = 0).

mat_perf <- matrix(nrow = 4, ncol = 4)

colnames(mat_perf) <- c("tau(0)", "Gamma(0)", "lambda", "alpha0")
rownames(mat_perf) <- c(paste0("MSE(", h,      ")"),
                        paste0("MSE(", htilde, ")"),
                        "DFP-shifted",
                        "DFP full dec.")

# MSE h-step: compute time-shift and unit-normalised gain.
# lambda and alpha0 are not applicable (left as NA).
# Unit-normalisation is applied so that all predictors are compared
# on the same scale (b_unit and b_cd are also of unit length).
mat_perf[1, 1:2] <- c(tauh,
                      sum(gammah) / as.double(sqrt(t(gammah) %*% gammah)))

# Long-horizon MSE: same metrics computed for the htilde-step predictor.
mat_perf[2, 1:2] <- c(tauhtilde,
                      sum(gammahtilde) / as.double(sqrt(t(gammahtilde) %*% gammahtilde)))

# Time-shift DFP: all four metrics are available and meaningful.
mat_perf[3, ] <- c(taub, sum(b_unit), lambda0, alpha0)

# Completely decoupled DFP: tau(0) is set to NA because the
# frequency-zero time-shift is ill-defined when the filter reverses
# the trend direction (i.e., Gamma(0) < 0, see Wildi 2026, section 4.1).
mat_perf[4, ] <- c(NA, sum(b_cd), lambda_cd, alpha0_cd)

# Note: NA entries indicate that the metric is not meaningful for
# that predictor, not that the computation failed.
mat_perf

# ── Discussion ────────────────────────────────────────────────────────
#
# 1. tau(0) — Frequency-zero time-shift
#
#    - MSE(h): the classical h-step MSE predictor has a time-shift of
#      approximately 4.01, meaning that when applied to a linear trend
#      the output is delayed by ~4 time units (right-shifted).
#
#    - MSE(htilde): increasing the forecast horizon to h = htilde does
#      not materially reduce the time-shift. A longer horizon alone
#      cannot address timeliness — the look-ahead problem persists
#      within the MSE paradigm regardless of h.
#
#    - DFP-shifted (row 3): the time-shift differs from MSE(h) by the
#      imposed lead tau = -2 time units. The DFP constraint directly
#      controls frequency-zero timeliness, achieving look-ahead
#      behaviour (relative to gammah) that is inaccessible to the MSE 
#      predictor at any forecast horizon.
#
#    - DFP full dec. (row 4): tau(0) is undefined because the fully
#      decoupled DFP reverses the trend direction, as confirmed by the
#      negative Gamma(0) in column 2. A time-shift is not meaningful
#      when the filter inverts the sign of the trend.
#
# 2. Gamma(0) — Gain at frequency zero
#
#    - Positive values indicate that the predictor preserves the
#      direction of a linear trend (sign-consistent, see Wildi 2026, 
#      section 4.1).
#    - All predictors except the fully decoupled DFP have positive
#      Gamma(0), confirming sign consistency at frequency zero.
#    - The fully decoupled DFP has Gamma(0) < 0, indicating trend
#      reversal — an extreme consequence of complete decoupling in 
#      this particular example.
#
# 3. lambda — DFP weight on gamma0
#
#    - A negative lambda indicates that gammah lies between b and
#      gamma0 in filter coefficient space (phase excess); see
#      Tutorial 5, Exercise 1.6 for the geometric interpretation.
#    - The fully decoupled DFP has a more negative lambda than the
#      shifted DFP, reflecting stronger look-ahead behaviour (a larger
#      rotation in the geometry of Tutorial 5, Exercise 1.6).
#
# 4. alpha0 — DFP decoupling constraint
#
#    - alpha0 is a covariance, not a correlation, and cannot be
#      interpreted on an absolute scale except at the boundary
#      alpha0 = 0 (complete decoupling).
#    - Reparametrising the DFP constraint in terms of the frequency-
#      zero time-shift tau (instead of alpha0) substantially improves
#      interpretability and comparability across predictors.
#    - The fully decoupled DFP corresponds to alpha0 = 0 by
#      construction, confirming exact satisfaction of the constraint.


# ─────────────────────────────────────────────────────────────────────
# 1.11 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two side-by-side panels —
#   Left:  filter coefficient profiles for all four predictors.
#   Right: cross-correlation functions (CCF) between each predictor
#          and the nowcast gamma0, i.e., x_t.
par(mfrow = c(1, 2))

colo <- c("black", "green", "blue", "red")

# Collect all four filters into a matrix (no rescaling applied).
# Columns: AR(3) nowcast | h-step MSE | time-shift DFP | fully decoupled DFP
mplot     <- cbind(gamma0, gammah, b_unit, b_cd)
col_names <- c("Nowcast: AR(3)", paste0("MSE ", h, "-step"), "DFP-shifted", "DFP-full-dec.")
colnames(mplot) <- col_names

# --- Left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Nowcast and Predictor Filter Coefficients",
     axes = F, type = "l",
     xlab = "Lag", ylab = "Filter coefficient",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
# Overlay remaining filters with colour-coded legend labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the h-step MSE filter on top to ensure it is not occluded
lines(mplot[, 2], col = colo[2])

axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
     labels = c(0, (1:(nrow(mplot) / 10)) * 10))
axis(2)
box()

# --- Right panel: cross-correlation functions (CCF) ---
# For each predictor, compute the CCF with the AR(3) process and
# retain max_lag values starting from lag 0.
max_lag <- 10
mplot <- cbind(
  compute_ccf_func(gammah, gamma0),
  compute_ccf_func(b_unit, gamma0),
  compute_ccf_func(b_cd,   gamma0))[L - 1 + 1:max_lag, ]
colnames(mplot) <- col_names[2:length(col_names)]

plot(mplot[, 1],
     main = "Cross-Correlation Functions (CCF)",
     axes = F, type = "l",
     xlab = "Lag", ylab = "CCF",
     col  = colo[2], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for the time-shift DFP and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[1 + i])
}
# Reference lines:
#   solid  vertical → lag 0 (current observation x_t)
#   dashed vertical → lag h (target forecast horizon)
#   solid horizontal → CCF = 0 baseline
abline(v = 1,       lty = 1)
abline(v = 1 + h,   lty = 2)
abline(h = 0)

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ── Discussion: CCF panel (right) ────────────────────────────────────
#
# - MSE predictor (green): by construction, the MSE predictor maximises
#   the CCF at the target horizon h. However, it also correlates very
#   strongly with x_t at lag 0, confirming the "stuck at the present"
#   phenomenon.
#
# - DFP-shifted (blue): imposing a frequency-zero lead of tau = -2
#   reduces the correlation with x_t at lag 0. As a consequence, the
#   CCF at the target horizon h decreases relative to the MSE
#   predictor. The DFP-shifted solution minimises this accuracy loss
#   at horizon h subject to the imposed lead constraint.
#
# - DFP fully decoupled (red): enforcing alpha0 = 0 drives the CCF
#   at lag 0 to zero by construction. The resulting CCF at the target
#   horizon h is substantially reduced — it represents the maximum
#   attainable correlation at h under complete decoupling from x_t.




# ─────────────────────────────────────────────────────────────────────
# 1.12 Compare Predictors
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 1.12.1 Apply Predictors to Data
# ─────────────────────────────────────────────────────────────────────

# Assemble the filter matrix, normalising gamma0 and gammah to unit
# L2-norm so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after
# normalisation (Gamma(0) < 0; see Section 1.10).
filter_mat <- cbind(
  gamma0 / as.double(sqrt(t(gamma0) %*% gamma0)),  # unit-normalised nowcast filter
  gammah / as.double(sqrt(t(gammah) %*% gammah)),  # unit-normalised h-step MSE filter
  b_unit,                                          # time-shift DFP (already unit-length)
  b_cd                                             # fully decoupled DFP (already unit-length)
)
colnames(filter_mat) <- col_names

# --- Simulate the AR(3) process and the h-step MSE predictor ---
# Fix the random seed for reproducibility and generate a long
# white-noise (standard normal) input series of length `len`.
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Initialise the AR(3) process x and the h-step MSE predictor xhat.
# Both are initialised to the innovation sequence; the loop below
# overwrites entries from index 4 onwards.
x    <- eps
xhat <- eps

for (i in 4:len) {
  # AR(3) recursion: x_t = ar1*x_{t-1} + ar2*x_{t-2} + ar3*x_{t-3} + eps_t
  x[i]    <- ar1 * x[i-1] + ar2 * x[i-2] + ar3 * x[i-3] + eps[i]
  # h-step MSE predictor: truncated MA applied to the innovation sequence
  xhat[i] <- gammah[1:min(i, L)] %*% eps[i:max(1, i - L + 1)]
}

# --- Apply each filter to the innovation sequence via causal convolution ---
# All filters are applied to eps using one-sided (causal) convolution,
# consistent with the MA-form representation of the AR(3) process.
y_out_mat <- filter(eps, filter_mat[, 1], sides = 1)                    # nowcast (gamma0)
y_out_mat <- cbind(y_out_mat, filter(eps, filter_mat[, 2], sides = 1))  # h-step MSE
y_out_mat <- cbind(y_out_mat, filter(eps, filter_mat[, 3], sides = 1))  # DFP-shifted
y_out_mat <- cbind(y_out_mat, filter(eps, filter_mat[, 4], sides = 1))  # DFP fully decoupled
colnames(y_out_mat) <- col_names

# --- Verification: gamma0 approximates the AR(3) process ---
# The nowcast output (column 1 of y_out_mat, based on gamma0) is
# compared with the directly simulated AR(3) series x over a short
# window. The two series should be nearly indistinguishable: any
# remaining discrepancy arises from the finite truncation length L of
# the Wold decomposition and can be made arbitrarily small by
# increasing L.
anf <- 350
enf <- 415
par(mfrow = c(1, 1))
ts.plot(scale(cbind(x, y_out_mat[, 1]))[anf:enf, ],
        main = "Nowcast gamma0 replicates the AR(3) process")

# ─────────────────────────────────────────────────────────────────────
# 1.12.2 Plot: Predictor Outputs
# ─────────────────────────────────────────────────────────────────────

# Plot a representative excerpt (obs. anf:enf) of all four scaled
# predictor outputs. The true AR(3) process (dashed black) and the
# h-step MSE predictor (dashed green) are overlaid for reference.
par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Scaled Predictor Outputs",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
lines(scale(x[anf:enf]),              lty = 2, lwd = 2)            # true AR(3) process
lines(scale(xhat[anf:enf]), col = "green", lty = 2, lwd = 2)       # h-step MSE predictor
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# --- Interpretation of the Plot ---
#
# The plot compares the outputs of all four predictors over the sample
# window [anf, enf], illustrating the practical consequences of the
# DFP constraint when alpha0 = alpha0(tau) is derived from the desired
# frequency-zero lead tau.
#
# - DFP-shifted (blue): the predictor anticipates mean reversion over
#   mid-term dynamics. This is visible as sustained intervals where the
#   DFP-shifted output leads the MSE predictor (green) across the zero
#   line — i.e., it signals turning points before the MSE predictor
#   does. By construction, the DFP-shifted filter leads the MSE
#   predictor by exactly tau time units on a linear trend at frequency
#   zero. Subsequent tutorials and applications demonstrate how this
#   translates into a pratically relevant look-ahead advantage.
#
# - DFP fully decoupled (red): complete decoupling (alpha0 = 0)
#   aggressively anticipates maxima, minima, and zero-crossings of the
#   process. However, two costs emerge simultaneously:
#
#     (i)  Amplitude loss — the predictor becomes anchored near the
#          mean during sustained swings, losing the ability to track
#          the true amplitude of the process.
#
#     (ii) Increased noise — the predictor output is noisier than the
#          DFP-shifted variant, as complete decoupling discards all
#          signal content associated with x_t.
#
#   Together, these two costs reduce the cross-correlation (CCF)
#   between the fully decoupled predictor and the target x_{t+h} at
#   forecast horizon h — as confirmed in the CCF panel of Section 1.11.

# ─────────────────────────────────────────────────────────────────────
# 1.12.3 Empirical CCFs: DFP Predictors Referenced Against MSE (gammah)
# ─────────────────────────────────────────────────────────────────────

# Remove leading NAs introduced by the causal convolution before
# computing empirical cross-correlation functions.
y_out_mat <- na.exclude(y_out_mat)

par(mfrow = c(1, 2))
ccf(y_out_mat[, 2], y_out_mat[, 3],
    main    = "CCF: DFP-shifted vs. MSE",
    lag.max = 10)
ccf(y_out_mat[, 2], y_out_mat[, 4],
    main    = "CCF: DFP fully decoupled vs. MSE",
    lag.max = 10)
# --- Discussion ---
#
# 1. DFP-shifted (tau = -2 at frequency zero):
#    The empirical CCF is asymmetric, peaking at lag 0 rather than
#    lead 2. This apparent discrepancy arises because the imposed
#    time-shift of -2 is local to frequency zero: it guarantees a
#    left-shift of exactly 2 time units only for signal components at
#    or near zero frequency. The AR(3) process, however, distributes
#    its spectral mass across the entire frequency band, so higher-
#    frequency components are left-shifted by less than 2 units,
#    pulling the aggregate CCF peak toward 0.
#    Components near zero frequency — such as business-cycle dynamics
#    with typical periodicities of 4–6 years — are shifted by close
#    to 2 units. Exercise 2 below verifies the target lead of -2
#    explicitly for a linear trend and for a cosine with a 5-year
#    business-cycle periodicity.
#
# 2. DFP fully decoupled (alpha0 = 0):
#    The empirical CCF is strongly asymmetric, with the peak
#    correlation still occurring at lag 0. A pronounced
#    negative correlation with backward-shifted (lagged) values of
#    gammah is also observed, suggesting that the fully decoupled DFP
#    exploits the natural mean-reversion tendency of the AR(3) process.
#
# 3. Mean reversion and the DFP rationale:
#    Simply inverting the MSE predictor (i.e., doing the opposite of
#    gammah) is not a viable look-ahead strategy, because the predictor
#    must still maintain a sufficiently large positive (maximal) correlation 
#    with the target x_{t+h}. The DFP resolves this fundamental tension —
#    between decoupling from x_t and retaining predictive accuracy at
#    horizon h — within a coherent optimisation framework.
#
# 4. Consistency requirement:
#    When targeting forecast horizon h (here h = 3), a consistent
#    predictor must correlate positively with x_{t+h}. Both DFP
#    designs satisfy this requirement. However, the fully decoupled
#    design achieves only a small positive correlation at h = 3 —
#    the price of enforcing complete decoupling from x_t at lag 0.
#    The DFP addresses this accuracy–timeliness tradeoff optimally,
#    operating on the efficient frontier between the two objectives.
#
# 5. Escalating asymmetry and interpretability:
#    The CCF pattern of the fully decoupled DFP pushes to an extreme
#    the asymmetry already introduced by the DFP-shifted predictor.
#    This raises natural questions about the interpretability and
#    statistical consistency of aggressive look-ahead designs. These
#    questions are examined further in subsequent tutorials; see also
#    Sections 4.3 and 5 of Wildi (2026) for a formal treatment.



# ════════════════════════════════════════════════════════════════════
# Exercise 2: Applications to Trend, Cycle, Cycle+Noise, and Macro
#             Indicator
# ════════════════════════════════════════════════════════════════════
# Prerequisite: the code in Exercise 1 must be executed before
# running this exercise, as filter objects and parameters defined
# there are used throughout.

# ─────────────────────────────────────────────────────────────────────
# 2.1 Apply Predictors of Exercise 1 to a Linear Trend
# ─────────────────────────────────────────────────────────────────────
# Purposes:
#   1. Verify the pertinence of the frequency-zero time-shift formula.
#   2. Verify that the imposed DFP time-shift constraint is correctly
#      enforced on a linear trend input.
#
# A linear trend is the canonical input for assessing frequency-zero
# behaviour: a filter with gain Gamma(0) and time-shift tau(0) applied
# to a linear trend reproduces the trend scaled by Gamma(0) and
# shifted by tau(0) time steps.

# ─────────────────────────────────────────────────────────────────────
# 2.1.1 Generate Trend Predictor Outputs
# ─────────────────────────────────────────────────────────────────────
len <- 100
x   <- (-len):len   # symmetric linear trend of length 2*len + 1

# Apply each filter to x via one-sided (causal) convolution and
# collect the outputs as columns of y_out_mat.
y_out_mat <- filter(x, filter_mat[, 1], sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], sides = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors.
ts.plot(y_out_mat,
        main = "Trend Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the fully decoupled DFP (red) INVERTS the trend
# direction. This is a direct consequence of its negative frequency-
# zero gain: sum(filter coefficients) = Gamma(0) < 0.
# See the performance table in Section 1.10, column Gamma(0).
sum(filter_mat[, "DFP-full-dec."])   # confirms Gamma(0) < 0

# ─────────────────────────────────────────────────────────────────────
# 2.1.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# To isolate timing differences and verify the time-shift constraint,
# three modifications are applied:
#
#   1. Remove the fully decoupled DFP: its negative Gamma(0) inverts
#      the trend, making a timing comparison meaningless.
#   2. Rescale the remaining three filters to unit gain at frequency
#      zero by dividing each by its coefficient sum. This normalises
#      out amplitude differences and isolates horizontal (timing) leads.
#   3. Use a shorter trend (smaller len) for visual clarity.

len <- 30
x   <- (-len):len   # shorter linear trend for clearer visualisation

# Apply the three remaining filters, each normalised to unit gain
# at omega = 0 (i.e., Gamma(0) = 1 after dividing by sum of weights).
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), sides = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward to separate the trend lines vertically
# for easier visual comparison (display adjustment only; no analytical
# effect on the time-shift verification).
mplot <- y_out_mat - min(y_out_mat, na.rm = TRUE) - 5

par(mfrow = c(1, 1))
# Horizontal offsets between curves are now directly interpretable as
# time-shift differences at frequency zero.
ts.plot(na.exclude(mplot),
        main = "Trend Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# ─────────────────────────────────────────────────────────────────────
# Verification Checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# ----------------------------------------------------------------------
# Compute the empirical lead as the horizontal offset between the two
# unit-amplitude trend outputs (first non-NA difference).
na.exclude(y_out_mat[, col_names[1]] - y_out_mat[, col_names[2]])[1]
  
# Compare with the theoretical time-shift difference at frequency zero
# (derived in Tutorial 2):
tauh - tau0

# The empirical lead matches the theoretical value, confirming that the
# frequency-zero time-shift formula correctly predicts filter behaviour
# on a linear trend input.

# Check 2: lead of the DFP predictor (b_unit) over the MSE predictor (gammah)
# ---------------------------------------------------------------------------
# Inspect column names to confirm ordering before differencing.
col_names

# Compute the empirical lead of the DFP-shifted output over the MSE
# output (first non-NA difference):
na.exclude(y_out_mat[, col_names[2]] - y_out_mat[, col_names[3]])[1]
  
# Compare with the lead imposed by the DFP time-shift constraint:
taub - tauh

# The observed lead matches the pre-specified DFP lead at frequency
# zero, confirming that the TIME-SHIFT CONSTRAINT IS CORRECTLY
# IMPLEMENTED AND ENFORCED.

# ─────────────────────────────────────────────────────────────────────
# 2.2 Apply Predictors to a Cosine (Synthetic Business Cycle)
# ─────────────────────────────────────────────────────────────────────
# Purpose:
# Verify that the frequency-zero time-shift spills over to frequencies
# near zero — specifically to business-cycle frequencies. A cosine with
# a 5-year periodicity (in a monthly data framework) serves as a
# stylised representation of business-cycle dynamics.

# ─────────────────────────────────────────────────────────────────────
# 2.2.1 Generate Cosine Input (Business-Cycle Periodicity)
# ─────────────────────────────────────────────────────────────────────
len   <- 120
omega <- pi / (5 * 6)          # angular frequency: 5-year cycle, monthly data
x     <- cos(omega * (1:len))  # cosine input of length len

# Apply each filter to x via one-sided (causal) convolution.
y_out_mat <- filter(x, filter_mat[, 1], sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], sides = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
ts.plot(y_out_mat,
        main = "Cycle Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the fully decoupled DFP (red) seems to invert the phase of the
# cosine — the counterpart, at business-cycle frequency, of the trend
# reversal observed in Section 2.1.1. However, the frequency-domain analysis 
# at the end of exercise 2.3.3 suggests an alternative interpretation.


# ─────────────────────────────────────────────────────────────────────
# 2.2.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# The same modifications as in Section 2.1.2 are applied:
#   1. Remove the fully decoupled DFP whose shift is excessive.
#   2. Rescale the remaining three filters to unit gain at frequency
#      zero by dividing each by its coefficient sum. This normalises
#      out amplitude differences and isolates horizontal timing leads.

# Apply the three remaining filters, each normalised to unit gain
# at omega = 0 (Gamma(0) = 1 after dividing by sum of weights).
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), sides = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

par(mfrow = c(1, 1))
# Horizontal offsets between curves are directly interpretable as
# time-shift differences (leads) at the cosine frequency omega.
ts.plot(na.exclude(y_out_mat),
        main = "Cycle Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the MSE predictor (green) is nearly coincident with the
# nowcast (black), indicating that its lead is negligible
# at the business-cycle frequency omega = pi/30. By contrast, the
# DFP-shifted predictor (blue) is visibly left-shifted relative to the
# MSE predictor, demonstrating that the imposed frequency-zero lead
# carries over to nearby frequencies.

# ─────────────────────────────────────────────────────────────────────
# Verification Checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# ----------------------------------------------------------------------
# For a cosine input, the lead between two filters is measured as the
# distance between their respective maxima (or minima).
find_max_mat <- head(na.exclude(cbind(y_out_mat[, col_names[1]],
                                      y_out_mat[, col_names[2]])), 20)

max_now <- which(find_max_mat[, 1] == max(find_max_mat[, 1]))  # maximum of nowcast
max_mse <- which(find_max_mat[, 2] == max(find_max_mat[, 2]))  # maximum of MSE predictor

# Empirical lead of MSE over nowcast at the cosine frequency:
max_mse - max_now
# Expected lead based on the frequency-zero time-shift difference:
tau0-tauh 

# The MSE predictor's frequency-zero lead has vanished at omega = pi/30:
# the two maxima are coincident. This confirms that the MSE predictor
# does not genuinely lead the process at business-cycle frequencies,
# regardless of the forecast horizon h.

# Check 2: lead of the DFP predictor (b_unit) over the MSE predictor (gammah)
# ----------------------------------------------------------------------------
col_names   # inspect ordering before differencing

find_max_mat <- head(na.exclude(cbind(y_out_mat[, col_names[2]],
                                      y_out_mat[, col_names[3]])), 20)

max_mse <- which(find_max_mat[, 1] == max(find_max_mat[, 1]))  # maximum of MSE predictor
max_dfp <- which(find_max_mat[, 2] == max(find_max_mat[, 2]))  # maximum of DFP-shifted

# Empirical lead of DFP-shifted over MSE at the cosine frequency:
max_mse - max_dfp
# Lead at frequency zero:
tauh-taub 

# The DFP-shifted predictor retains approximately 2 time units of lead
# over the MSE predictor even at omega = pi/30, confirming that the
# frequency-zero time-shift constraint spills over to nearby (business-
# cycle) frequencies by continuity.

# ─────────────────────────────────────────────────────────────────────
# Main Takeaways
# ─────────────────────────────────────────────────────────────────────
#
# 1. MSE lead: the classical MSE predictor has a negligible lead at
#    both frequency zero and business-cycle frequencies. Increasing
#    the forecast horizon h does not generate a materially larger lead.
#
# 2. DFP-shifted: the tau-shifted DFP works as intended — a linear
#    trend is left-shifted by exactly tau time units. By continuity,
#    this lead also applies at non-zero frequencies, and in particular
#    at the 5-year business-cycle frequency.
#
# 3. Fully decoupled DFP: complete decoupling is too extreme to be
#    meaningfully applied to either a linear trend or a pure cosine 
#    in this example —it inverts the signal direction in both cases.
#    Tutorial 8 illustrates a case of non-inverting fully decoupled DFP.
#
# Next: apply all three designs to a cosine contaminated with noise.

# ─────────────────────────────────────────────────────────────────────
# 2.3 Cycle + Noise
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 2.3.1 Generate Business-Cycle Cosine Contaminated with Noise
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a signal consisting of a cosine at the
# 5-year business-cycle frequency (monthly data) plus white noise.
# The noise component tests whether the DFP's look-ahead advantage
# survives in a more realistic, noisy environment.

len   <- 240
omega <- pi / (5 * 6)   # angular frequency: 5-year cycle, monthly data
set.seed(1)
eps <- rnorm(len)
x   <- cos(omega * (1:len)) + eps   # cycle-plus-noise input

# Apply each filter to x via one-sided (causal) convolution.
y_out_mat <- filter(x, filter_mat[, 1], sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], sides = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
ts.plot(y_out_mat,
        main = "Cycle + Noise Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the fully decoupled DFP (red) inverts the phase and is
# excluded from further analysis in this section.

# ─────────────────────────────────────────────────────────────────────
# 2.3.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# Apply the three remaining filters, each normalised to unit gain
# at omega = 0 (Gamma(0) = 1 after dividing by sum of weights).
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), sides = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), sides = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), sides = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

mplot <- na.exclude(y_out_mat)

par(mfrow = c(1, 1))
ts.plot(mplot,
        main = "Cycle + Noise Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: The DFP cannot anticipate high-frequency noise —
# short-term peaks and troughs are not left-shifted. However, as
# established in Section 2.2, the DFP does anticipate cyclical
# turning points. When the signal-to-noise ratio is sufficiently
# large, the DFP leads the MSE predictor at cyclical turning points
# even in the presence of noise. The plot suggests that this lead is
# preserved here. A formal quantitative assessment via amplitude and
# time-shift statistics is provided in the next section, where the
# empirical findings above are confirmed through a data-independent
# frequency-domain analysis.


# ─────────────────────────────────────────────────────────────────────
# 2.3.3 Amplitude and Time-Shift Functions
# ─────────────────────────────────────────────────────────────────────
# Compute the amplitude and time-shift functions for all four filters
# across the full frequency range [0, pi]. These spectral summaries
# complement the time-domain verification in Sections 2.1–2.3 and
# quantify the accuracy–timeliness tradeoff across frequencies.

K      <- 600    # number of frequency grid points in [0, pi]
plot_T <- FALSE  # suppress internal plotting; a custom plot is built below

amp_mat   <- NULL
shift_mat <- NULL

for (i in 1:ncol(filter_mat)) {
  as_obj    <- amp_shift_func(K, filter_mat[, i], plot_T)
  amp_mat   <- cbind(amp_mat,   as_obj$amp)
  shift_mat <- cbind(shift_mat, as_obj$shift)
}
colnames(amp_mat) <- colnames(shift_mat) <- colnames(filter_mat)

# --- Plot amplitude and time-shift functions side by side ---
par(mfrow = c(1, 2))
colo <- c("black", "green", "blue", "red")

# Left panel: amplitude functions
mplot <- amp_mat
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude Functions",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Right panel: time-shift functions
# Truncate extreme negative values of the fully decoupled DFP (red)
# for display purposes: values below -2 are replaced with NA to
# prevent the y-axis scale from being dominated by outliers.
mplot <- shift_mat
mplot[mplot[, ncol(mplot)] < -2, ncol(mplot)] <- NA

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-Shift Functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))),
     col  = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at     = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# --- Discussion ---
#
# 1. Amplitude functions (left panel):
#    - The first three filters (excluding the fully decoupled design) 
#      are low-pass: amplitude declines toward higher frequencies.
#    - The fully decoupled DFP (red) is the only filter whose amplitude
#      function is not monotonically decreasing.
#    - The MSE predictor (green) closely tracks the process amplitude
#      (black) across all frequencies, illustrating the "stuck at the
#      present" problem: it adds little filtering relative to the
#      nowcast.
#    - Both DFP designs exhibit reduced amplitude at low frequencies
#      and relatively larger amplitude at high frequencies, reflecting
#      the accuracy–timeliness (ATS) trilemma: gaining timeliness
#      comes at the cost of amplifying high-frequency noise relative
#      to the low-frequency signal.
#
# 2. Time-shift functions (right panel):
#    - The MSE predictor (green) and the nowcast (black) have nearly
#      identical time-shift profiles, confirming the "stuck at the
#      present" problem across the full frequency band.
#    - The DFP-shifted predictor (blue) has a time-shift exactly
#      tau = -2 units smaller than the MSE predictor at frequency
#      zero, consistent with the imposed constraint. This confirms
#      the time-domain results reported in the performance table of
#      Section 1.10.
#    - The DFP lead over the MSE predictor is approximately stable in
#      a neighbourhood of frequency zero, which explains the observed
#      lead at business-cycle frequencies in Sections 2.2 and 2.3.
#    - The fully decoupled DFP (red) exhibits a negative time-shift at
#      low frequencies, confirming its anticipative behaviour.
#      Specifically, its lead at omega = pi/30 (the 5-year
#      business-cycle frequency) is:

shift_mat[1 + K/30, ncol(shift_mat)]

# This corresponds to the left-shift visible in the output (red curve)
# in the figure of Exercise 2.2.1: the maxima and minima are left-shifted
# by approximately 20 time units relative to the process (black curve).
# Since the process itself lags by 

shift_mat[1 + K/30, 1]

# the net lead is (as can be verified from the figure):

shift_mat[1 + K/30, 1] - shift_mat[1 + K/30, ncol(shift_mat)]


# ─────────────────────────────────────────────────────────────────────
# 2.4 Application to a Monthly Macro Indicator: PAYEMS
# ─────────────────────────────────────────────────────────────────────
# Note: this is a preliminary application of the DFP to the monthly
# US non-farm payroll employment series (PAYEMS). It is intentionally
# incomplete and flawed and serves as an initial illustration only. 
# A fully refined and corrected application is provided in Tutorial 8.

# Set reload_data = TRUE to download the latest vintage from FRED;
# set to FALSE to load a previously saved local copy.
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

# --- Extract and transform the sub-sample ---
# Restrict to the post-1990, pre-pandemic period (1990–2019) in
# log-levels. The log transformation stabilises the variance as the
# level of the series grows over time.
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y) <- index(PAYEMS["1990::2019"])

plot(y, main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = FALSE,
     xlab = "", ylab = "")
axis(1, at = 1:length(y), labels = names(y))
axis(2)
box()

# First-difference the log series to obtain a stationary growth-rate
# series. The log transformation stabilises the variance; differencing
# stabilises the level.
x <- diff(y)

# Inspect the ACF of the differenced series.
# The slowly, monotonically decaying ACF closely resembles the AR(3)
# structure used in Exercise 1, justifying the direct application of
# the filters derived there.
acf(x, main = "ACF of log-differenced PAYEMS")

# --- Apply existing filters without re-fitting ---
# Rather than re-estimating a model for PAYEMS, we apply the AR(3)-
# based filters from Exercise 1 directly. This is valid given the
# similarity in ACF structure noted above.
# We select the nowcast (gamma0), h-step MSE predictor (gammah), and
# DFP-shifted predictor (b_unit); the fully decoupled DFP is excluded
# as it inverts the signal direction.
# Based on the cycle analysis in Sections 2.2–2.3, we expect the
# nowcast and MSE predictor to be nearly coincident, while the DFP
# should lead both.

select_predictors <- 1:3
filter_payems     <- filter_mat[, select_predictors]
colnames(filter_payems) <- colnames(filter_mat)[select_predictors]
coli <- colo[select_predictors]

# Apply each filter to x via one-sided (causal) convolution.
y_out_mat <- NULL
for (i in 1:ncol(filter_payems))
  y_out_mat <- cbind(y_out_mat, filter(x, filter_payems[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_payems)
rownames(y_out_mat) <- names(x)

# --- Plot predictor outputs ---
par(mfrow = c(1, 1))
mplot <- y_out_mat
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "", ylab = "",
     main = "Nowcast, MSE Predictor, and DFP Applied to Log-Diff PAYEMS",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))),
     col  = coli[1])
mtext(colnames(mplot)[1], line = -1, col = coli[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = coli[i])
  mtext(colnames(mplot)[i], col = coli[i], line = -i)
}
abline(h = 0)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2)
box()

# Outcome:
# The DFP-shifted predictor (blue) leads the MSE predictor (green) and
# the nowcast (black), consistent with the results in Sections 2.2–2.3.
# However, the DFP output is noisier in relative terms: the signal-to-
# noise ratio is lower than for the MSE predictor. This is a direct
# consequence of the amplitude function behaviour documented in
# Section 2.3.3 — the DFP amplifies high-frequency components relative
# to the low-frequency signal as the price of its look-ahead advantage.

# CAVEAT:
# The implementation above is fundamentally flawed (why?). An explanation of
# the issue and the corresponding correction are provided in Tutorial 8.
