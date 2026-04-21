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
# translates into genuine look-ahead behaviour — that is, whether and
# to what extent the MSE-DFP predictor achieves a measurable lead over
# the classical MSE benchmark.

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
gamma <- ARMAtoMA(ar = c(ar1, ar2, ar3), lag.max = 1000)

# Inspect the first L MA coefficients
# Slowly monotonically decaying
par(mfrow=c(1,1))
ts.plot(gamma[1:L],main="Wold decomposition of AR(3)")

# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah):
# shift gamma by h positions and pad with zeros at the end.
# The trailing zeros reflect the fact that MA coefficients beyond index L
# are negligible for a finite-length filter of length L.
gammah <- c(gamma[h + (1:(L - h))], rep(0, h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead <- -2


# ─────────────────────────────────────────────────────────────────────
# 1.2 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)

# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)
dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)

# Extract the components returned by the function
tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
b       <- dfp_obj$b        # raw DFP filter coefficients

# Normalise b to unit length to obtain the unitary DFP filter
b_unit <- b / as.double(sqrt(b %*% b))


# ─────────────────────────────────────────────────────────────────────
# 1.3 Replicate Formula for b
# ─────────────────────────────────────────────────────────────────────
# The frequency-zero time-shift of a filter with coefficients c is defined as
#   tau_c = sum_{k=0}^{L-1} k * c_k  /  sum_{k=0}^{L-1} c_k. This equals the 
# phase shift at omega = 0, see tutorial 2, exercise 3.

# The MSE DFP predictor is obtained as
#   b <- gammah + lambda0 * gamma0,
# see tutorial 5 and Wildi 2026, proposition 1. We can link lambda0 to the 
# time-shift tau at frequency zero, see Theorem 2 (equation 34) in 
# Wildi (2026). We here briefly summarize the main implementation steps:

# Step 1. Compute frequency-zero time-shifts of the three filters (see tutorial 2)
tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)       # nowcast filter
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)       # h-step MSE filter
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)  # long-horizon filter

# Display tau0 and tauh (the MSE filter leads slightly at frequency zero)
tau0
tauh

# Interpretation of tau0: the AR(3) process can be represented as an
# AR(3) filter 1/(1 - a1*B - a2*B^2 - a3*B^3) applied to the
# innovation sequence epsilon_t. Accordingly, gamma0 represents the
# inverted finite-length approximation of this filter (i.e., a
# truncated Wold decomposition).
#
# When applied to a linear trend, this filter shifts the trend by tau0
# time units to the right (after rescaling; see Tutorial 2 for
# details). The MSE predictor gammah has a slightly smaller time-shift
# value tauh, meaning that a linear trend filtered by gammah lies
# slightly to the left of a trend filtered by gamma0 (after proper
# rescaling).
#
# In this sense, the MSE predictor gammah effectively leads the process gamma0
# at frequency zero, as measured by the tau-statistic: the predictor
# anticipates the trend component by (tau0 - tauh) time units relative
# to the process itself:

tau0 - tauh   # natural lead of the MSE predictor over the process (in time units)

# This lead is minor (approximately a quarter of a time unit), and is
# therefore insufficient for practical look-ahead purposes. We impose a
# larger lead via the DFP constraint. The desired lead of the DFP over
# the MSE predictor at frequency zero is specified as:

tau <- lead     # target lead (in time units) to be enforced by DFP
abs(tau)        # absolute magnitude of the desired lead


# Step 2. Compute lambda0 from the closed-form expression (Theorem 2, equation 34):
lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))


# Step 3. Construct the DFP predictor/filter
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


# ─────────────────────────────────────────────────────────────────────
# 1.5 Roundtrips: Apply MSE-DFP Based on alpha0
# ─────────────────────────────────────────────────────────────────────

# Check: we can recover b_tau from the ordinary MSE-DFP based on alpha0

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
# 1.7 Comparison with Unitary DFP: Cautionary Warning
# ─────────────────────────────────────────────────────────────────────
# This is a counter example to emphasize the different meaning of alpha0 
# in unitary DFP and MSE-DFP.

# Compute unitary DFP
bu<-unitary_DFP_func(gamma0, gammah, alpha0)$b0
# Check unit length
t(bu)%*%bu
# Check unitary DFP constraint: should vanish
t(bu)%*%gamma0/sqrt(t(gamma0)%*%gamma0)-alpha0

# Check MSE-DFP constraint: should vanish
t(b_tau)%*%gamma0-alpha0

# Compare both DFP predictors
ts.plot(cbind(bu,b_tau/sqrt(as.double(t(b_tau)%*%b_tau))),col=c("red","blue"),xlab="",main="Unitary and MSE DFP for a given (the same) alpha0")
mtext("Unitary DFP",col="red",line=-1)
mtext("MSE DFP",line=-2,col="blue")


# The difference is due to the fact that alpha0 does not have the same meaning:
# In the unitary DFP, alpha0 is the correlation between DFP and the nowcast gamma0. 
# In the MSE-DFP alpha0 is not a correlation anymore.
# Therefore, imposing the same alpha0 generates two different solutions.

# However, imposing the same (zer0-frequency) time-shift to unitary and MSe DFP would lead to the same solution.




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

# We can still compute the numerical value of the time-shift formula.
# The result should be interpreted as the shift of the sign-reverted predictor
# (-b_cd), which does have a positive gain at frequency zero.
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd


# ─────────────────────────────────────────────────────────────────────
# 1.10 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (positive = right-shift or lag when applied to linear trend)
#   Gamma(0)  — gain at frequency zero (sum of filter weights, unit-normalised)
#   lambda    — DFP regularisation weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value)
#
# Rows correspond to:
#   MSE(h)       — h-step MSE predictor
#   MSE(htilde)  — long-horizon MSE predictor (reference)
#   DFP-shifted  — time-shift DFP with specified lead
#   DFP full dec.— completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 4, ncol = 4)

colnames(mat_perf) <- c("tau(0)", "Gamma(0)", "lambda", "alpha0")
rownames(mat_perf) <- c(paste("MSE(", h,      ")", sep = ""),
                        paste("MSE(", htilde, ")", sep = ""),
                        "DFP-shifted",
                        "DFP full dec.")

# MSE h-step: time-shift and unit-normalised gain; lambda and alpha0 not applicable
mat_perf[1, 1:2] <- c(tauh,
                      sum(gammah) / as.double(sqrt(t(gammah) %*% gammah)))

# Long-horizon MSE: same metrics for htilde
mat_perf[2, 1:2] <- c(tauhtilde,
                      sum(gammahtilde) / as.double(sqrt(t(gammahtilde) %*% gammahtilde)))

# Time-shift DFP: all four metrics available
mat_perf[3, ] <- c(taub, sum(b_unit), lambda0, alpha0)

# Completely decoupled DFP: time-shift is NA (ill-defined due to phase reversal)
mat_perf[4, ] <- c(NA, sum(b_cd), lambda_cd, alpha0_cd)

# Note: NAs indicate that the corresponding entries are left blank (not meaningful)
mat_perf

# Discussion:
# 1. tau(0)-column:
#     -The classic h=3 steps ahead MSE predictor has a time-shift of ~4.01
#       Interpretation: when the filter (predictor) is applied to a linear trend, 
#       then the trend will be right-shifted (delayed) by 4.01 time points.
#     -Increasing the forecast horizon to h=20 (MSE(20) predcitor) does not 
#       reduce the lag: the forecast horizon cannot be used to address timeliness (look ahead)
#     -DFP-shifted (3-rd row) has a time-shift that differs by tau=-2 from MSE(3).
#       DFP implements the time-shift constraint at frequency zero and it can address
#       look ahead behaviour which cannot be obtained in the MSE paradigm (increasing 
#       the foreacst horizon is not effective.)
#     -DFP fully decoupled: the time-shift is not defined because the fully-decoupled DFP
#       inverts the trend direction: this is confirmed by the negative sign of Gamma(0) 
#       in the second column (the weights add to a negative number)
# 2. Gamma(0) column
#     -Positive numbers indicate that the filter (predictor) preserves the direction of a linear trend,
#     -All predictors except the fully-decoupled preserve sign orientation at frequency zero.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 5, exercise 1.6. The fully decoupled has a more negative lambda indicating stronger
#       look ahead behaviour (stronger rotation in the figure of tutorial 5, exercise 1.6).
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 1.11 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 3.9)
par(mfrow=c(1,2))

colo <- c("black", "green", "blue", "red")

# Collect all four filters into a matrix (no scaling applied)
# Columns: AR(3) nowcast, h-step MSE, time-shift DFP, fully decoupled DFP
mplot     <- scale(cbind(gamma0, gammah, b_unit, b_cd), center = F, scale = F)
col_names <- c("AR(3)", paste("MSE ", h, "-step"), "DFP-shift", "DFP-full-decouple")
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
max_lag<-10
mplot <- cbind(
  compute_ccf_func(gammah, gamma0),
  compute_ccf_func( b_unit,  gamma0),
  compute_ccf_func( b_cd,   gamma0))[L-1+1:max_lag,]
colnames(mplot) <- col_names[2:length(col_names)]

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[2], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[1 + i])
}

# Disabled in-plot legend (filters identified by colour above)
# mtext("MSE", line=-1, col=colo[1+1])
# mtext("DFP", line=-2, col=colo[2+1])

# Vertical reference lines:
#   solid  → lag 0 (current observation, index = max_lag + 1)
#   dashed → h-step-ahead lag
abline(v = 1,     lty = 1)
abline(v =  1 + h, lty = 2)
abline(h = 0)

axis(1, at = 1:nrow(mplot),
     labels = -1+1:nrow(mplot))
axis(2)
box()

# Discussion:
# CCF (right panel)
#   -The MSE predictor (green) maximizes the CCF at the target horizon h=3.
#    However, the MSE predictor correlates very strongly with x_t (lag 0).
#   -The tau-shifted DFP (blue) correlates less strongly with x_t as an effect of imposing a lead tau=-2
#     at frewuency zero. As a consequence, the correlation with the target at h=3 decreases.
#     But the DFP minimizes this loss at h=3.
#   -Finally, the fully decoupled (red) has a vanishing CCF at lag 0. As 
#     a consequence, the traget correlation at h=3 is pulled severly down: it is maximal 
#     subject to the imposed full decoupling.

# ─────────────────────────────────────────────────────────────────────
# 1.12 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.12.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.
filter_mat <- cbind(
  gamma0  / as.double(sqrt(t(gamma0)  %*% gamma0)),   # unit-normalised nowcast filter
  gammah  / as.double(sqrt(t(gammah)  %*% gammah)),   # unit-normalised h-step MSE filter
  b_unit,                                    # time-shift DFP (already unit-length)
  b_cd                                      # completely decoupled DFP (unit-length)
)
colnames(filter_mat)<-col_names

# Fix the random seed and generate a long white-noise input series
set.seed(345)
len <- 10000
x   <- rnorm(len)

# Apply each filter to x using one-sided (causal) convolution
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 1.12.2 Plot
#----------------------------------------------------------------------
# Disabled earlier diagnostic plot (shorter window, scaled outputs)
# ts.plot(scale(y_out_mat[270:305,], center=F, scale=T),
#         main="AR(3)", col=colo, xlab="", ylab="")
# abline(h=0)

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat[300:350, ],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Discussion:
# The DFP designs (blue and red) tend to lie to the right of the MSE predictor
#   (green) especially at longer swings above or below the zero (mean) line.
# Short term high-frequency noise cannot be anticipated.
# The time-shifted DFP (blue) leads the MSE predictor on a linear trend, by design of the constraint.
# We shall see below how this materializes as useful look ahead feature for other series/applications.

#----------------------------------------------------------------------
# 1.12.3 Compute empirical CCFs: referenced against MSE predictor gammah
#----------------------------------------------------------------------

y_out_mat<-na.exclude(y_out_mat)

par(mfrow=c(1,2))
ccf(y_out_mat[,1],y_out_mat[,3],main="CCF: DFP-shift vs. MSE",lag.max=10)
ccf(y_out_mat[,1],y_out_mat[,4],main="DFP-fully-decoupled vs. MSE",lag.max=10)

# Outcome:
# 1. The DFP with a time-shift of -2 at frequency zero exhibits an asymmetric CCF,
#    suggesting that it leads the MSE predictor gammah: it correlates more strongly 
#     with forward shifted than with backward shifted gammah. 
#     However, because the AR(3)
#    process distributes its spectral mass across the entire frequency band, this
#    time-shift lead (-2) is purely local to frequency zero and should not be interpreted as an
#    indicator of an overall aggregate lead over gammah (exercise 3.10
#    below verifies the target lead of -2 at zero frequency).
#
# 2. The fully decoupled DFP has a strongly asymmetric CCF with an effective
#    aggregate lead of two time units, as evidenced by the peak correlation
#    occurring at lead 2 (to the right of zero). Notably, a strong negative correlation with
#    backward-shifted values of gammah (i.e., lagged values) is observed,
#    suggesting that the fully decoupled DFP may be exploiting the natural
#    mean-reversion tendency of gammah.
#
# 3. With regards to mean-reversion, just doing the opposite of gammah is not a feasible strategy
#   because the correlation with gammah must be `large'. Thus the DFP reconciles 
#   an internal (fundamental) inconsistency in a consistent rationale optimization framework.
#
# 4. When targeting a forecast horizon h (here h=3) we expect a consistent predictor to correlate 
#     positively with x_{t+h}. Both DFP comply with this fundamental consistency rule,
#     though the correlation of the fully decoupled design is rather small at h=3. This 
#     is the price to be paid for complete decoupling at lag 0 (present time). 
#     The DFp addresses this tradeoff optimally (efficient frontier).
#
# 5. The CCF pattern of the fully decoupled DFP pushes to an extreme the
#    asymmetry already introduced by the more mildly decoupled DFP time-shift
#    predictor. This raises the question of interpretability and consistency
#    of aggressive look-ahead designs.





# ════════════════════════════════════════════════════════════════════
# Exercise 2: Applications to Trend, Cycle, Cycle+Noise and Macro Indicator
# ════════════════════════════════════════════════════════════════════

# It is assumed that the code in exercise 1 has been executed.

# ─────────────────────────────────────────────────────────────────────
# 2.1 Apply Predictors of Exercise 1 to a Linear Trend
# ─────────────────────────────────────────────────────────────────────
# Purposes:
# 1) verify pertinence of the time-shift formula at frequency zero.
# 2) Verify pertinence of the imposed DFP time-shift constraint.

# ─────────────────────────────────────────────────────────────────────
# 2.1.1 Generate Trend Predictor Outputs
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a linear trend input x = -len, ..., len.
# A linear trend is the canonical input for assessing frequency-zero
# behaviour: a filter with gain Gamma(0) and time-shift tau(0) will
# reproduce the trend scaled by Gamma(0) and shifted by tau(0) steps.

len <- 100
x   <- (-len):len

# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
ts.plot(y_out_mat,
        main = "Trend Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the completely decoupled DFP (red) INVERTS the trend direction.
# This is a direct consequence of its negative frequency-zero gain:
#   sum(filter coefficients) = Gamma(0) < 0, see the performance matrix in 
#   exercise 1.10, column Gamma(0):
sum(filter_mat[, "DFP-full-decouple"])


# ─────────────────────────────────────────────────────────────────────
# 2.1.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# Three modifications are applied for a verification of the time-shift 
# constraint:
#   1. Remove the completely decoupled DFP (trend-inverting; not meaningful
#      as a look-ahead predictor for a linear trend).
#   2. Rescale the remaining filters so that each has unit gain at
#      frequency zero, i.e. divide each filter by its coefficient sum.
#      This normalises out amplitude differences and isolates timing (lead)
#      differences across predictors.
#   3. Shorten the trend path (smaller len) for visual clarity.

len <- 30
x   <- (-len):len   # shorter linear trend for clearer visualisation

# Apply each of the three remaining filters, normalised to unit gain at omega=0.
# Dividing by sum(filter) ensures Gamma(0) = 1 for each predictor.
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), side = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward so that the trend lines are separated vertically
# for easier visual comparison (pure display adjustment, no analytical effect)
mplot <- y_out_mat - min(y_out_mat, na.rm = T) - 5


par(mfrow = c(1, 1))
# Plot the rescaled trend outputs; horizontal leads between curves are now
# directly interpretable as time-shifts at frequency zero
ts.plot(na.exclude(mplot),
        main = "Trend Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# The observed left-shifts reflect the time-shift differences: we now verify
# that the observed empirical shifts match the theoretical expressions.

# ─────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# -----------------------------------------------------------------------
# Compute the effective lead from the trend outputs directly:
#   the horizontal offset between the two unit-amplitude trend lines.
# Note: we use the first non-NA offset
na.exclude(y_out_mat[, col_names[1]] - y_out_mat[, col_names[2]])[1]

# Compare with the closed-form time-shift difference at frequency zero
# (derived in Tutorial 2):
tauh - tau0

# The empirical lead from the trend output matches the theoretical
# time-shift difference, confirming that the frequency-zero analysis 
# (tutorial 2) correctly predicts the filter's behaviour on a linear trend.


# Check 2: lead of the DFP predictor (b_unit) over the MSE predictor (gammah)
# ---------------------------------------------------------------------------
# Inspect column names to confirm ordering before differencing
col_names

# Compute the effective lead (horizontal offset) of the DFP over the 
# MSE predictor (use the first non-NA offset):
na.exclude(y_out_mat[, col_names[2]] - y_out_mat[, col_names[3]])[1]

# Compare with the lead imposed via the DFP time-shift constraint:
taub - tauh

# The observed lead of the DFP trend output over the MSE trend output
# matches the pre-specified DFP lead at frequency zero, confirming that 
# the TIME-SHIFT CONSTRAINT IS CORRECTLY IMPLEMENTED AND ENFORCED.



# ─────────────────────────────────────────────────────────────────────
# 2.2 Apply Predictors to a Cosine (Synthetic Business-Cycle)
# ─────────────────────────────────────────────────────────────────────
# Purpose:
# Verify spillover of the shift to frequencies nearby the trend (zero-) frequency.

# ─────────────────────────────────────────────────────────────────────
# 2.2.1 Generate Cosine (Business-Cycle Periodicity)
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a cosine with a typical business-cycle periodicity 
# of 5 years in a monthly data framework.

len <- 120
# Five year periodicity:
omega<-pi/(5*6)
x   <- cos(omega*(1:len))

# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
ts.plot(y_out_mat,
        main = "Cycle Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the completely decoupled DFP (red) INVERTS the phase.


# ─────────────────────────────────────────────────────────────────────
# 2.2.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────
# The same modifications as in exercise 2.1.2 are applied: 
#   1. Remove the completely decoupled DFP (trend-inverting; not meaningful
#      as a look-ahead predictor for a linear trend).
#   2. Rescale the remaining filters so that each has unit gain at
#      frequency zero, i.e. divide each filter by its coefficient sum.
#      This normalises out amplitude differences and isolates timing (lead)
#      differences across predictors.

# Apply each of the three remaining filters, normalised to unit gain at omega=0.
# Dividing by sum(filter) ensures Gamma(0) = 1 for each predictor.
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), side = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward so that the trend lines are separated vertically
# for easier visual comparison (pure display adjustment, no analytical effect)
mplot <- y_out_mat 


par(mfrow = c(1, 1))
# Plot the rescaled trend outputs; horizontal leads between curves are now
# directly interpretable as time-shifts at frequency zero
ts.plot(na.exclude(mplot),
        main = "Cycle Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# The observed left-shifts reflect the time-shift differences: 
# Note that the MSE (green) is almost coincident with the (AR(3)-) data. In 
# constrast, the DFP is faster (left-shifted). we now compute 
# the empirical left-shifts (leads) for the business-cycle cosine.

# ─────────────────────────────────────────────────────────────────────
# Verification checks
# ─────────────────────────────────────────────────────────────────────

# Check 1: lead of the MSE predictor (gammah) over the nowcast (gamma0)
# -----------------------------------------------------------------------
# Compute the effective lead from the trend outputs directly:
# In constrast to the linear trend, the lead here is found be taking the distance between maxima or minima of the cycle
find_max_mat<-head(na.exclude(cbind(y_out_mat[, col_names[1]] , y_out_mat[, col_names[1]])),20)
# Maximum of nowcast
max_now<-which(find_max_mat[,1]==max(find_max_mat[,1]))
# Maximum of DFP-shift (shifted by tau=-2 at frequency zero)
max_mse<-which(find_max_mat[,2]==max(find_max_mat[,2]))
# Zero delay: the predictor is coincident with the nowcast (x_t):
max_mse-max_now
# Compare with the closed-form time-shift difference at frequency zero
# (derived in Tutorial 2):
tauh - tau0

# The original lead at frequency zero of the MSE predictor has vanished
# at frequency omega=pi/30. We now verify if the DFP (shifted by tau=-2 
# at zero-frequency) can maintain part of its lead at omega=pi/30>0



# Check 2: lead of the DFP predictor (b_unit) over the MSE predictor (gammah)
# ---------------------------------------------------------------------------
# Inspect column names to confirm ordering before differencing
col_names

# In constrast to the linear trend, the lead here is found be taking the distance between maxima or minima of the cycle
find_max_mat<-head(na.exclude(cbind(y_out_mat[, col_names[2]] , y_out_mat[, col_names[3]])),20)
# Maximum of MSE (gammah)
max_dfp<-which(find_max_mat[,2]==max(find_max_mat[,2]))

# Outcome: the empirical shift is still approximately 2 time units at omega=pi/30
max_dfp-max_mse

# Compare with the lead imposed via the DFP time-shift constraint:
taub - tauh

#############################################################################
# MAIN TAKE AWAYS:

# -The left-shift (lead) of the classic MSE predictor is small and increasing the forecast horizon does 
#   help in generating a lerger lead.
# -The tau-shifted DFP works as intended: a linear trend is left-shifted by tau time units.
# -Due to continuity, this left shift also applied to non-zero frequencies, in 
#  particular for a cosine with a typical 5-year business-cycle frequency.
# -The fully-decoupled DFP is too extreme to be meaningfully applied to either the linear trend of the cosine.

# Next we apply the designs to a cosine with noise

# ─────────────────────────────────────────────────────────────────────
# 2.3 Cycle + Noise
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 2.3.1 Generate (Business-Cycle Frequency) Cosine + Noise
# ─────────────────────────────────────────────────────────────────────
# Apply all four filters to a cosine with periodicity of 5 years in 
# in a monthly data framework.

len <- 240
set.seed(1)
eps<-rnorm(len)
# Five year periodicity:
omega<-pi/(5*6)
# Cycle plus noise
x   <- cos(omega*(1:len))+eps


# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat <- filter(x, filter_mat[, 1], side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3], side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 4], side = 1))
colnames(y_out_mat) <- col_names

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
ts.plot(y_out_mat,
        main = "Cycle Outputs: Unscaled Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Observation: the completely decoupled DFP (red) INVERTS the phase and is discarded 
# from further analysis here.


# ─────────────────────────────────────────────────────────────────────
# 2.3.2 Rescale to Unit Amplitude
# ─────────────────────────────────────────────────────────────────────

# Apply each of the three remaining filters, normalised to unit gain at omega=0.
# Dividing by sum(filter) ensures Gamma(0) = 1 for each predictor.
y_out_mat <- filter(x, filter_mat[, 1] / sum(filter_mat[, 1]), side = 1)
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 2] / sum(filter_mat[, 2]), side = 1))
y_out_mat <- cbind(y_out_mat, filter(x, filter_mat[, 3] / sum(filter_mat[, 3]), side = 1))
colnames(y_out_mat) <- col_names[-length(col_names)]   # drop the fully decoupled label

# Shift all outputs downward so that the trend lines are separated vertically
# for easier visual comparison (pure display adjustment, no analytical effect)
mplot <- na.exclude(y_out_mat) 


par(mfrow = c(1, 1))
# Plot the rescaled trend outputs; horizontal leads between curves are now
# directly interpretable as time-shifts at frequency zero
ts.plot(mplot,
        main = "Cycle Outputs: Unit-Amplitude Predictors",
        col  = colo, xlab = "", ylab = "")
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)


# Outcome: the DFP cannot anticipate high-frequency noise: short term peaks or troughs are not left-shifted.
# But we know from the previous
# exercise that it can anticipate cyclical turning-points. So if the signal (the cycle) 
# is strong enough, the DFP will lead the classic MSE predictor gammah at the cycle turning-points.
# The plot suggests indeed that the DFP is left-shifted (leading) even when subject to noise. 
# Amplitude and time-shifts will provide more insight.

# ─────────────────────────────────────────────────────────────────────
# 2.3.3 Amplitude and Time-Shifts
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
colo <- c("black", "green", "blue", "red")
mplot <- amp_mat

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude functions",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()


mplot <- shift_mat
mplot[which(mplot[,ncol(mplot)]<(-2)),ncol(mplot)]<-NA

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
#mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
#  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Outcome:
# 1. Amplitude:
#   -All filters are lowpass. Only the fully-decoupled (red) does not have a monotonically decaying amplitude.
#   -The MSE amplitude (green) is close to the process (black) illustrating the `stuck to present' problem.
#   -The DFP amplitudes are smaller at lower frequencies and larger at higher frequencies: ATS trilemma.
#     -They generate more high-frequency noise relative to low frequency signal.
# 2. Time shifts
#   -MSE (green) and process (black) are close: `stuck at present problem'.
#   -The DFP (blue) time-shift is exactly tau=-2 smaller than MSE at frequency zero.
#   -The frequency-domain shifts at frequency zero confirm the time-domain time-shifts computed in the performance table of 
#    exercise 1.10: the MSE predictor lags a linear time trend by 4 time units.
#   -The lead of DFP (blue) over MSE (green) is fairly stable in a vicinity of frequency zero: 
#    this confirms the lead at business-cycles frequencies in exercise 2.2 and 2.3.
#   -The fully-decoupled DFP (red) is anticipative (negative shift for lower frequencies).



# ─────────────────────────────────────────────────────────────────────
# 2.4 Application to Monthly Macro Indicator: PAYEMS
# ─────────────────────────────────────────────────────────────────────
# This is a first incomplete and partly false application of DFP to the 
# monthly US employment PAYEMS indicator.

# A more refined complete and corrected application to the indicator is 
# given in tutorial 8.

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
# The dependence structure is similar to above AR(3): slowly monotonically 
# decaying ACF: 
acf(x)

# We can fit a model to the data and run the code in exercise 1 or
# we can apply the existing predictors, without data fitting.
# Specifically, we select AR(3), MSE and DFP (shifted by tau) and compare
# their effects on the data.
# Given the above `cycle' analysis, we expect AR(3) and its 3-step MSE 
# predictor to be coincident, while the DFP should lead. 
select_predictors<-1:3
filter_payems<-filter_mat[,select_predictors]
colnames(filter_payems)<-colnames(filter_mat)[select_predictors]
coli<-colo[select_predictors]




# Apply each filter to x using one-sided (causal) filters and
# collect outputs as columns of y_out_mat
y_out_mat<-NULL
for (i in 1:ncol(filter_payems))
  y_out_mat <- cbind(y_out_mat, filter(x, filter_payems[, i], side = 1))
colnames(y_out_mat)<-colnames(filter_payems)
rownames(y_out_mat)<-names(x)

par(mfrow = c(1, 1))
# Plot all four trend outputs (unscaled) to reveal differences in
# amplitude and timing across predictors
mplot<-y_out_mat
plot(mplot[, 1], type = "l", axes = FALSE, xlab = "", ylab = "",
     main = "MSE and DFP applied to log-diff PAYEMS",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = coli[1])
mtext(colnames(mplot)[1], line = -1, col = coli[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = coli[i])
  mtext(colnames(mplot)[i], col = coli[i], line = -i)
}
abline(h=0)
axis(1, at     = 1:nrow(mplot),
     labels = rownames(mplot))
axis(2)
box()



# Outcome: 
# The DFP is left-shifted (leading) but the noise is stronger in relative terms.
# This effect (worse signal to noise ratio) has been explained with the 
# amplitude functions above.


