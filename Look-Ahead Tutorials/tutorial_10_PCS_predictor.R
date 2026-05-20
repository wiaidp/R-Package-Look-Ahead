# ════════════════════════════════════════════════════════════════════
# TUTORIAL 10 — PCS: INTRODUCTION
# ════════════════════════════════════════════════════════════════════
#
# Background
# ──────────
# As shown in Tutorial 1, Exercise 1.3, the h-step-ahead MSE predictor tends
# to be "anchored" at the present in difficult forecasting problems, in the
# sense that its Cross-Correlation Function (CCF) peaks at
# lag 0. The Peak Correlation Shifting (PCS) approach designs the h-step-ahead
# predictor so that the CCF peak is shifted, ideally toward the forecast 
# horizon h, thereby improving timeliness and look-ahead behaviour. While a 
# full shift from lag 0 to the forecast horizon h is not always feasible, even 
# a partial attenuation of the original peak at lag 0 in difficult forecast 
# problems can be effective. This weaker objective is precisely the strategy 
# of the DFP (Decoupling From Present) predictor in earlier tutorials.
#
# ── DFP Recap ────────────────────────────────────────────────────────
#
# - Decoupling From Present (DFP) weakens the CCF at lag 0 while maximising
#   it at the forecast horizon h (dilemma, conflicting objectives).
#
#     - In some cases, the CCF peak migrates toward lag h as a by-product, see 
#       exercise 5 below.
#
#     - However, peak-shifting is not an explicit DFP design objective; it is
#       an indirect and partially uncontrollable outcome.
#
#     - In terms of lead, DFP directly addresses the time-shift at frequency
#       zero (see Tutorials 6–9).
#
#     - Although DFP can always left-shift a linear trend by an arbitrary 
#       amount (arbitrarily large lead at frequency zero), the induced lead 
#       may not extend to adjacent relevant frequencies, e.g., business-cycle
#       frequencies (see Tutorial 9).
#
# ── PCS Approach ─────────────────────────────────────────────────────
#
# - Peak Correlation Shifting (PCS) simultaneously addresses two features
#   of the CCF:
#
#     - Peak location: ideally, the CCF peak is moved toward a precise target
#       location, typically the forecast horizon h.
#
#     - Peak height: in addition to relocating the peak, PCS maximises the
#       CCF value at lag h.
#
#     - To obtain an effective peak shift, PCS controls the slope of the CCF
#       via one of three design choices (Type I, II, and III below),
#       which differ in the number of lags constrained, the specific lags
#       targeted, and the degree of flexibility allowed in enforcing the
#       slope requirements.
#
#     - Shifting the peak (ideally toward h) and maximising its height
#       (always evaluated at h) are conflicting requirements, forming an
#       inherent dilemma: a predictor optimised purely for peak height at h
#       — i.e., the classic MSE approach — may fail to shift the peak (the
#       "stuck at present" problem), while one aggressively constrained to
#       shift the peak may sacrifice height.
#
#     - For given slope or average-slope constraints, PCS maximises the target
#       correlation at h, thereby tracing an efficient frontier that makes the
#       accuracy–timeliness trade-off explicit and controllable.
#
#     - By directly controlling the peak location, PCS addresses one of the
#       lead measures introduced in Tutorial 2.
#
#     - In contrast to DFP, which controls the lead locally at frequency zero
#       (see Tutorial 6), PCS targets an aggregate lead effect that is strong 
#       enough to relocate the CCF peak if the problem is feasible. The CCF is 
#       itself an aggregate, frequency-integrated measure of lead, so PCS 
#       effectively acts across the entire frequency band rather than 
#       specifically at frequency zero (as for the DFP).
#
#
# ── DFP vs. PCS ──────────────────────────────────────────────────────
#
# - Although in certain applications the resulting CCF profiles may look
#   similar, DFP and PCS are geometrically distinct:
#     - DFP: the predictor lies in the plane spanned by (gamma_0, gamma_h).
#     - PCS: depending on the implementation variant (I, II, or III below),
#       the predictor lies in the plane spanned by gamma_h and a single
#       gamma_k for k in {0, …, h-1}, or in a combination of such planes.
#   - In practice, PCS can often enforce stronger effective look-ahead
#     behaviour than DFP, though generally at the cost of some reduction in
#     target correlation.
#
# ── Necessary Conditions for a Peak Shift from k = 0 to k = h ────────
#
# Three type of constraints are considered to shift the CCF peak from 0 to h:
#
#   TYPE I (neither necessary nor sufficient condition) 
#       Monotonically increasing CCF over {0, …, h} (most restrictive):
#       The CCF must be strictly increasing across the full interval, i.e.,
#       CCF(k-1) < CCF(k) for all k = 1, …, h. See Wildi (2026), Section 3.2 
#       and Appendix E. This condition is generally not exactly feasible 
#       (see Exercise ???); The principal PCS optimization function 
#       PCS_func() enforces it as closely as possible via regularisation.
#
#   TYPE II (necessary but not sufficient)
#       Local positive slope at the target lag (weaker than I):
#       The CCF must be increasing over the final step only, i.e.,
#       CCF(h-1) < CCF(h). See Wildi (2026), Section 3.2.
#       In some cases where additional structure is imposed by the data-
#       generating process (e.g., from the Yule-Walker equations of an AR(p)),
#       conditions I) and II) may become equivalent.
#
#   TYPE III (necessary but not sufficient) 
#       Positive average slope from lag 0 to lag h (weaker than I):
#       The CCF must be increasing on average from k = 0 to k = h, i.e.,
#       CCF(0) < CCF(h). In some cases where additional structure is imposed 
#       by the data-generating process, conditions I) and III) may become 
#       equivalent.
#
#   A link to decoupling:
#       Even the weaker conditions of types II) and III) are not always exactly
#       feasible. When they are, however, both can be imposed within the DFP
#       optimisation framework by using a suitably modified constraint vector
#       (see Exercise 1 below).
#
# Technical note on Type I) condition:
#   When the full monotonicity constraint in Type I is feasible, strong 
#   regularisation enforces a fixed positive CCF slope across all lags in 
#   {0, …, h},  which may be unnecessarily costly in terms of target 
#   correlation. In practice, therefore, mild regularisation is typically 
#   preferred: it nudges the CCF toward monotonicity while retaining enough 
#   flexibility to balance the inherent dilemma between peak shifting and peak 
#   height, see exercise 4 below.
#

# ── Unit Norm ─────────────────────────────────────────────────────────
#
# The CCF at lag k is given by:
#
#     CCF(k) = b' * gamma_k / ||b||
#
# where:
#   b       : filter coefficient vector (the predictor to be designed),
#   gamma_k : k-step-ahead MSE predictor coefficients (forward-shifted Wold
#             coefficients), normalised to unit length.
#
# CCF and unit-length scaling:
#   - The CCF requires both b and gamma_k to be of unit length.
#   - In practice, b (the PCS predictor) is generally not of unit length;
#     imposing this constraint would complicate the geometry and lead to
#     multiple solutions (cf. the unitary DFP in Tutorial 4).
#   - Normalising only gamma_k to unit length is sufficient to ensure that
#     b' * gamma_k is proportional to CCF(k) for all k, with ||b|| serving as
#     a common (k-invariant) scaling factor. This makes the slope conditions
#     below well-defined and comparable across lags.
#
# ── PCS Optimisation Criterion ────────────────────────────────────────
#
# Let beta > 0 be the PCS slope parameter. Conditions I)–III) translate 
# into linear constraints on b:
#
#   Type I)   requires  b' * (gamma_k - gamma_{k-1}) > 0,  k = 1, …, h
#   Type II)  requires  b' * (gamma_h - gamma_{h-1}) > 0   (k = h only)
#   Type III) requires  b' * (gamma_h - gamma_0)     > 0   (k = h only)
#
# Note: the sign convention adopted here is the opposite of Wildi (2026).
#
# The PCS criterion enforces these constraints with target slope beta:
#
#   max  b' * gamma_h                          (maximise target correlation)
#   s.t. b' * (gamma_k - gamma_{k-1}) = beta,  k in Delta
#
# where:
#   Delta = {h}       corresponds to Condition II) (local slope at lag h), and
#   Delta = {1, …, h} corresponds to Condition I)  (monotonicity from 0 to h).
#
# Type III) is handled by setting Delta = {h} and replacing the ordinary
# differencing vector (gamma_h - gamma_{h-1}), i.e., delta = 1, with the
# lag-delta differencing vector (gamma_h - gamma_0), where delta = h.
#
# Notes: 
# 1. The objective (maximising target correlation) can alternatively be
#    replaced by minimum MSE (assuming a standard case, see the discussion in 
#    Tutorial 9). The two are equivalent up to simple (MSE-optimal) scaling. Our 
#    function implements this MSE optimal scaling, see exercises below.
# 2. Interpretation: beta/||b|| can be interpreted as the slope of the CCF.
#
# ── Regularised PCS (Implemented in the main function PCS_func) ────────
#
# The equality constraints above may be infeasible when the system is rank-
# deficient — for example, because of additional structure imposed by the
# Yule-Walker equations. To handle this, Wildi (2026) proposes a regularised
# version of the criterion (Equation 46, Appendix D). The regularised
# formulation replaces the hard equality constraints with a soft penalty term,
# yielding a criterion that is always well-defined and admits a unique solution
# regardless of feasibility. PCS_func() implements this regularised PCS
# criterion. The regularisation weight acts as the key tuning parameter: a
# larger weight pushes the solution closer to satisfying the slope constraints
# (stronger peak shifting) at the expense of target correlation, while a
# smaller weight relaxes the slope requirements and prioritises correlation
# height at h. Selecting an appropriate regularisation weight therefore
# provides the flexibility needed to navigate the inherent accuracy–timeliness
# dilemma of the PCS look-ahead approach.
#
# ── Tutorial Structure ────────────────────────────────────────────────
#
#   Exercise 1 — PCS type II) via a modified DFP approach (h = 1, one-step ahead):
#                Imposes a single local slope constraint at the forecast horizon 
#                h by replacing the standard DFP decoupling vector gamma_0 
#                (see DFP tutorials) with the difference
#                (gamma_{h} - gamma_{h-1}), directly enforcing a positive CCF
#                slope at the target horizon. While the MSE predictor remains
#                stuck at the present irrespective of the forecast horizon,
#                the PCS predictor successfully moves the CCF peak from k = 0
#                to k = h = 1, inducing effective look-ahead behaviour.
#
#   Exercise 2 — PCS type III) for h = 5: 
#                Imposes a single `on average' positive slope constraint from 
#                k=0 to k=h, replacing the standard DFP decoupling vector gamma_0 
#                (see DFP tutorials) with the difference
#                (gamma_{h} - gamma_0), directly enforcing an average positive 
#                slope CCF(h) - CCF(0) >=0.
#
#   Exercise 3 — PCS type I) for h = 5 with a fixed and very large
#                regularisation weight and a range of slope parameters:
#                Relocates the CCF peak by enforcing a linearly increasing
#                CCF over the full interval {0, ..., h}. This is more
#                constraining than Exercise 2, as it imposes a uniform
#                positive slope at every intermediate lag, effectively
#                prescribing a linear path for the CCF from k = 0 to
#                k = h = 5. The multi-constraint system is handled by
#                the novel PCS function PCS_func().
#
#   Exercise 4 — PCS type I) for h = 5 with a single (fixed) slope and a range of
#                regularisation weights:
#                Repeats Exercise 3 but varies the regularisation strength across
#                a range of values, holding the target slope fixed. This
#                illustrates how the penalty strength governs the
#                accuracy–timeliness trade-off: lighter regularisation permits
#                departures from the strict linear CCF path imposed in
#                Exercise 3, providing fine-tuning flexibility to better balance
#                peak shifting against target correlation.
#
#   Exercise 5 — PCS type I) vs. fully decoupled DFP:
#                Compares the look-ahead behaviour and CCF profiles produced
#                by PCS and the fully decoupled DFP predictor, highlighting
#                differences in the resulting CCF profiles.
#
#   Exercise 6 — Geometry of the PCS predictor:
#                Visualises the PCS type II) predictor in the plane spanned by
#                gamma_h and gamma_{h-1}, illustrating the geometric
#                interpretation of the local slope constraint. A comparison
#                with Tutorial 5, Exercise 1.6 reveals that DFP and PCS are
#                conceptually distinct approaches — operating in different
#                geometric planes and enforcing different constraints — even
#                when their resulting predictors may appear similar in some 
#                applications.
#
# ════════════════════════════════════════════════════════════════════
# REFERENCE
# ─────────
# Wildi, M. (2026).
#   Forecasting on the Accuracy–Timeliness Frontier:
#   Two Novel "Look-Ahead" Predictors.
#   https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════
# INITIALISATION
# ════════════════════════════════════════════════════════════════════

rm(list = ls())

# Load the DFP optimisation routines.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: PCS II) APPLIED TO AN MA(9) PROCESS
# ════════════════════════════════════════════════════════════════════
#
# We revisit the MA(9) example from Tutorial 1 and address Condition II)
# (local positive slope at the target horizon h) by adapting the DFP
# decoupling framework. Instead of decoupling b from the nowcast predictor
# gamma_0 (classic DFP), we decouple b from the difference vector
# (gamma_{h} - gamma_{h-1}), which directly enforces a positive CCF slope
# at lag h.
#
# Data-Generating Process (DGP), see Tutorial 1:
#   x_t = sum_{k=0}^{9} a1^k * epsilon_{t-k}        [MA(9) process]
#
# The MA(9) is a truncated AR(1) with coefficient a1 = 0.9; its moving-
# average weights decay geometrically: b_k = a1^k, k = 0, …, 9.
# ════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
# 1.1 Process Specification and Data Generation
# ─────────────────────────────────────────────────────────────────────

# MA order and geometric decay coefficient
q  <- 9        # MA order (number of lags beyond lag 0)
a1 <- 0.9      # geometric decay rate (equal to the underlying AR(1) coefficient)

# MA filter weights: b_k = a1^k, k = 0, …, q
b_ma <- a1^(0:q)

# Wold decomposition: b_ma with appended zeroes (useful when operating shifts).
xi<-c(b_ma,rep(0,1000))

par(mfrow = c(1, 1))
ts.plot(b_ma,
        main = "MA(9) filter coefficients (geometrically decaying)",
        xlab = "Lag", ylab = "Weight")

# Simulate a realisation of the MA(9) process:
#   x_t = sum_{k=0}^{q} b_k * ε_{t-k},  ε_t ~ i.i.d. N(0,1)
len <- 100
set.seed(231)
eps <- rnorm(len + q + 1)          # innovations (burn-in included)

# Pre-allocate output vectors
x <- xhat <- rep(NA, len + q + 1)

for (i in (q + 1):(len + q + 1)) {
  x[i] <- b_ma %*% eps[i:(i - q)]
}

ts.plot(x, main = "Simulated MA(9) process")


# ─────────────────────────────────────────────────────────────────────
# 1.2 MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────

# Filter length and forecast horizon
L <- 20
h <- 1
# Reference: applying a range of forecast horizons (1 <= h_tilde <= 9) confirms
# the absence of look-ahead behaviour in the MSE predictor — the CCF peak
# remains stuck at lag k = 0 regardless of the chosen horizon, illustrating
# the "stuck at present" problem that PCS is designed to overcome.
htilde<-5

# Feasibility check: for an MA(q) process, the h-step-ahead MSE predictor
# is identically zero when h > q, because all innovations more than q steps
# ahead are unobservable.
if (h > q )
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# Optimal MSE filter: retain MA coefficients from lag h onward; pad to length L.
# gammah[k] = b_{h+k} for k = 0, …, q-h, and 0 thereafter.
gammah <- c(b_ma[(h + 1):(q + 1)], rep(0, L - (q - h + 1)))
gammahtilde <- c(b_ma[(htilde + 1):(q + 1)], rep(0, L - (q - htilde + 1)))

par(mfrow=c(1,1))
ts.plot(cbind(gammah,gammahtilde),main=paste("MSE(",h,") and MSE(",htilde,") 
                                             predictors",sep=""))

# ─────────────────────────────────────────────────────────────────────
# 1.3 PCS Type II): Setting Up the Modified DFP Constraint
# ─────────────────────────────────────────────────────────────────────
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

# For h = 1: gamma_{h-1} = gamma_0 (nowcast predictor, padded to length L)
gammahm1 <- gamma0 <- c(b_ma, rep(0, L - length(b_ma)))

# Constraint vector: difference between consecutive MSE predictors at lags h-1 and h
gamma_constraint <- gammahm1 - gammah
# Note: the sign in gamma_constraint is arbitrary and could be reversed, 
# together with alpha0 in the constraint.

par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == gamma[h-1] - gamma[h]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector encoding the CCF slope condition at lag h")
abline(h = 0)

# The shape of gamma_constraint may raise doubts about whether decoupling b
# from it will effectively enforce look-ahead behaviour in the PCS predictor.
# The plots below will clarify this point.

# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor. 
# Purpose: mse_coup is a natural upper bound for the DFP constraint, i.e., 
# DFP should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, strictly smaller than the above mse_coup. 
# Smaller (more negative) values enforce progressively stronger CCF slope at 
# lag h (a right-shift of the peak towards h=1).
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.1)

# The DFP constraint enforces stronger decoupling form gamma_constraint than 
# the MSE predictor gammah: the last negative value suggests that the peak CCF
# should be shifted to the right: from k=0 to k=h=1.
alpha0_vec


# ─────────────────────────────────────────────────────────────────────
# 1.4 PCS Type II) Optimisation over the Decoupling Grid
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
    max_lag, h, as.vector(b), xi)$cor_vec
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]         # CCF at lag 0 (coupling with present)
    cor_vec_1[i, 2] <- cor_vec[1 + h]     # CCF at lag h (coupling with target)
}

colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1) <- c("Lag 0", paste0("h=", h))
rownames(cor_vec_1)<-paste0("alpha0=", round(alpha0_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 1.5 Routine Checks
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
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast", paste("MSE(",h,")",sep=""),
                          paste("MSE(",htilde,")",sep=""),
                          paste0("PCS ", round(alpha0_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 1.6 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green","darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncate to the first q+1 lags where the MA process has support.
mplot <- filter_mat[1:(q + 1), ]

plot(mplot[, 1], main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
    max_lag, h, filter_mat[,i], xi)$cor_vec)
mplot   <- ccf_mat[1:q, ]

plot(mplot[, 1], main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
# 1.7 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 1.7.1 Forecast Comparison ────────────────────────────────────────


# Generate a long white-noise series for a reliable empirical evaluation
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 390
enf <- 430

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   As the PCS decoupling weight increases (alpha0 decreases), the predictor
#   output shifts progressively to the left (looks further ahead) relative to
#   the MSE predictor. This visual lead is confirmed quantitatively by the
#   empirical CCFs below.


# ── 1.7.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",htilde,"): Peak at lag k = 0 (no peak-shift)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))






# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: PCS III) Impose an Average Positive Slope of the CCF
# ════════════════════════════════════════════════════════════════════

# We apply the PCS based on case III), assuming h=5. The single constraint 
# is implemented by a modified DFP.

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 2.1 MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────

# Filter length and forecast horizon
L <- 20
h <- 5
# Reference: apply a higher forecast lead for look ahead behaviour in the MSE
htilde<-5

# Feasibility check: for an MA(q) process, the h-step-ahead MSE predictor
# is identically zero when h > q, because all innovations more than q steps
# ahead are unobservable.
if (h > q + 1)
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# Optimal MSE filter: retain MA coefficients from lag h onward; pad to length L.
# gammah[k] = b_{h+k} for k = 0, …, q-h, and 0 thereafter.
gammah <- c(b_ma[(h + 1):(q + 1)], rep(0, L - (q - h + 1)))
gammahtilde <- c(b_ma[(htilde + 1):(q + 1)], rep(0, L - (q - htilde + 1)))

# ─────────────────────────────────────────────────────────────────────
# 2.2 PCS Type III): Setting Up the Modified PCS Constraint
# ─────────────────────────────────────────────────────────────────────
# Type III) requires CCF(h) > CCF(0), i.e., the CCF must increase
# on average from lag 0 to the forecast horizon h. Using 
# CCF(k) = b' * gamma_k, this becomes:
#
#   b' * gamma_0 < b' * gamma_h
#   <=>  b' * (gamma_0 - gamma_h) < 0
#
# Equivalently, setting alpha0 = b' * gamma_constraint with
#   gamma_constraint = gamma_0 - gamma_h,
# Type III) requires alpha0 < 0.
#

# For h = 1: gamma_{h-1} = gamma_0 (nowcast predictor, padded to length L)
gamma0 <- c(b_ma, rep(0, L - length(b_ma)))

# Constraint vector: difference between consecutive MSE predictors at lags h-1 and h
gamma_constraint <- gamma0 - gammah

par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == gamma[0] - gamma[h]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector encoding the average CCF slope condition from k=0 to h")
abline(h = 0)


# The shape of gamma_constraint may raise doubts about whether decoupling b
# from it will effectively enforce look-ahead behaviour in the PCS predictor.
# The plots below will clarify this point.

# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor. 
# Purpose: mse_coup is a natural upper bound for the DFP constraint, i.e., 
# DFP should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0: strictly smaller than mse_coup. 
# Smaller (more negative) values enforce progressively stronger average CCF 
# slope from k=0 to k=h=5 (a right-shift of the peak towards h=5).
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.1)

# The PCS constraint enforces stronger decoupling form gamma_constraint than 
# the MSE predictor gammah: the last negative value suggests that the peak CCF
# should be shifted to the right: from k=0 to k=h=1.
alpha0_vec


# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS Optimisation over the Decoupling Grid
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
    max_lag, h, as.vector(b), xi)$cor_vec
  cor_vec_mat     <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1] <- cor_vec[1]         # CCF at lag 0 (coupling with present)
  cor_vec_1[i, 2] <- cor_vec[1 + h]     # CCF at lag h (coupling with target)
}

colnames(b_mat) <- colnames(cor_vec_mat) <- paste0("alpha0=", round(alpha0_vec, 3))
colnames(cor_vec_1) <- c("Lag 0", paste0("h=", h))
rownames(cor_vec_1)<-paste0("alpha0=", round(alpha0_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 2.4 Routine Checks
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
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast", paste("MSE(",h,")",sep=""),
                          paste("MSE(",htilde,")",sep=""),
                          paste0("PCS ", round(alpha0_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 2.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green","darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncate to the first q+1 lags where the MA process has support.
mplot <- filter_mat[1:(q + 1), ]
plot(mplot[, 1], main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
    max_lag, h, filter_mat[,i], xi)$cor_vec)
mplot   <- ccf_mat[1:q, ]

plot(mplot[, 1], main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1,     lty = 1)   # lag 0
abline(v = max_lag + 1 + h, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Outcomes ─────────────────────────────────────────────────────────
# Left panel (filter coefficients):
#   - Unlike the MSE predictor, the PCS/DFP filters assign non-zero weight
#     up to the farthest lag k = q.
#   - Stronger decoupling (smaller alpha0) progressively shifts weight away
#     from recent observations toward the oldest lag. This is counter-intuitive
#     but is a direct consequence of enforcing the CCF slope constraint.
#
# Right panel (CCFs):
#   - The MSE predictors maximize the CCF at their respective forecast horizons.
#   - Enforcing the average slope constraint via decoupling works as intended: as
#     alpha0 decreases, the average slope between lag 0 and h flattens and eventually
#     inverts (violet line), confirming a peak shift toward h=5.
#   - Increasing the forecast horizon (any admissible htilde<=9) does not 
#     shift the peak of the CCF of the MSE predictor.
#   - The loss in target correlation at h is minimised subject to the
#     modified decoupling constraint (efficient frontier).

# Tabular summary: CCF of PCS at lag 0 and lag h for each decoupling level
round(cor_vec_1, 2)


# ─────────────────────────────────────────────────────────────────────
# 2.6 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 2.6.1 Forecast Comparison ────────────────────────────────────────


# Generate a long white-noise series for a reliable empirical evaluation
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 390
enf <- 430

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   As the PCS decoupling weight increases (alpha0 decreases), the predictor
#   output shifts progressively to the left (looks further ahead) relative to
#   the MSE predictor. This visual lead is confirmed quantitatively by the
#   empirical CCFs below.


# ── 2.6.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",htilde,"): Peak at lag k = 0 (no peak-shiftd)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))

# ─────────────────────────────────────────────────────────────────────
# 2.7 Replicating DFP Solution via PCS
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────
# The forecasting problem at hand is a PCS Type III specification with a
# single constraint. Two equivalent solution paths exist:
#
#   Path 1 – Modified DFP:
#     The DFP function can solve this problem by supplying a suitably
#     modified constraint vector, as demonstrated in the sections above.
#
#   Path 2 – Direct PCS (used here):
#     PCS_func() solves the same problem natively. Unlike the DFP
#     approach, PCS_func() is more general:
#       • Single constraints : Type II or Type III specifications
#       • Multiple constraints: Type I specifications
#
# In this section we use Path 2 to replicate the "modified DFP" Type III
# solution obtained earlier, confirming that both paths yield the same
# filter coefficients.


# ── 2.7.1 Full Decoupling ─────────────────────────────────────────────
# Under full decoupling (alpha0 = 0), the MSE-DFP predictor and the PCS
# predictor coincide exactly, so no sign or scale adjustment is needed.

# Set the decoupling parameter to zero (full decoupling)
alpha0 <- 0

# Compute the MSE-DFP filter coefficients using the specified constraint
# vector (gamma_constraint) and the target cross-covariance (gammah)
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# ── PCS hyperparameter settings (see Exercise 3 for a full derivation) ──

# Under full decoupling, beta equals alpha0 directly (no rescaling required)
beta <- alpha0

# Use strong regularization to enforce the constraint tightly
lambda <- 100000

# Define the two-point constraint grid: origin and forecast horizon h
Delta <- c(0, h)

# Use a Type III PCS specification
Type_III <- TRUE

# Use the true DGP 
gamma_pcs <- gamma0

# Compute the PCS filter coefficients
b_pcs <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III)$b

# Plot both sets of coefficients to verify they overlap
par(mfrow=c(1,1))
ts.plot(cbind(b_dfp, b_pcs), main = "Both Predictors Overlap")


# ── 2.7.2 Partial Decoupling ──────────────────────────────────────────
# When alpha0 ≠ 0 (partial decoupling), the DFP and PCS parameterizations
# use different sign conventions and scaling for the slope constraint.
# A manual sign flip and rescaling of beta are therefore required before
# the two filters will agree.

alpha0 <- 0.5

# Compute the MSE-DFP filter coefficients for the partially decoupled case
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0

# Adjust beta: flip the sign and apply the empirical rescaling factor (0.465)
# that accounts for the difference in normalization between the two frameworks
beta <- -0.465 * alpha0

# Retain strong regularization to keep the constraint active
lambda <- 100000

# Constraint grid: origin and forecast horizon h (unchanged from above)
Delta <- c(0, h)

# Type III PCS specification (unchanged)
Type_III <- TRUE

# Extend gamma_pcs with trailing zeros to match the required vector length
# (PCS_func needs a longer ACF vector for the partial-decoupling case)
gamma_pcs <- c(gamma0, rep(0, 1000))

# Compute the rescaled PCS filter coefficients
b_pcs <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda, Type_III)$b

# Plot both sets of coefficients; they should be nearly identical
ts.plot(cbind(b_dfp, b_pcs), main = "Both Predictors are Nearly Overlapping")




# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: PCS I) — Regularised Criterion with Strong Regularisation
# ════════════════════════════════════════════════════════════════════
# 
# A PCS Type I design is applied with forecast horizon h = 5 and a large
# regularisation weight lambda.
#
# - In contrast to Exercises 1 and 2, which relied on the DFP framework to
#   impose a single (simpler) constraint, the more complex five-dimensional
#   constraint system determined by h = 5 and PCS Type I is handled here
#   via PCS_func().
#
# - The optimisation is based on Equation 46 (Appendix D, Wildi 2026), with
#   the closed-form solution given in Equation 49.
#
# - Strong regularisation imposes an overly restrictive constraint: the CCF
#   of the resulting predictor increases linearly from lag k = 0 to k = h,
#   leaving little flexibility in the shape of the CCF profile.
#
# - The slope parameter beta is varied to examine its effect on the CCF
#   profile and on the degree of look-ahead behaviour achieved.

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 3.1 MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────

# Filter length and forecast horizon.
L <- 20
h <- 5

# Additional reference horizon htilde > h: used later to confirm that the MSE predictor
# remains stuck at k = 0 (CCF peaks at k=0) regardless of htilde.
htilde <- 7

# Feasibility check: for an MA(q) process, the h-step-ahead MSE predictor is
# identically zero when h > q, because all innovations more than q steps ahead
# are unobservable.
if (h > q + 1)
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# MSE filter: retain MA coefficients from lag h onward and pad to length L.
# gammah[k] = b_{h+k} for k = 0, …, q-h, and 0 thereafter.
gammah      <- c(b_ma[(h      + 1):(q + 1)], rep(0, L - (q - h      + 1)))
gammahtilde <- c(b_ma[(htilde + 1):(q + 1)], rep(0, L - (q - htilde + 1)))

# ─────────────────────────────────────────────────────────────────────
# 3.2 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.
beta_vec <- c(-0.2, -0.1, 0, 0.01, 0.02, 0.05)

# Constrained lag set: Type I) imposes a positive slope at every lag in Delta, 
# here from 1 to h, enforcing a monotonically increasing CCF (if beta is 
# positive and the problem is feasible) over the full interval
# {0, …, h}. This is the most restrictive of the three PCS types (I, II and III).
Delta <- 1:h

# Very large regularisation weight: drives the solution toward exact
# satisfaction of all h slope constraints simultaneously, producing a CCF
# that increases linearly from k = 0 to k = h with uniform slope
# beta / (b' * b). In practice, this level of regularisation is typically
# more restrictive than necessary and may reduce target correlation unduly
# (see the discussion in Exercises 3.5 and 4).
lambda <- 100000


# ─────────────────────────────────────────────────────────────────────
# 3.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h,Delta, xi, L, beta, lambda)
  
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

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 3.4 Routine Checks
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
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))

# ─────────────────────────────────────────────────────────────────────
# 3.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncated to the first q+1 lags where the MA process has support.
mplot <- filter_mat[1:(q + 1), ]
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat[1:q, ]

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1,       lty = 1)   # lag 0
abline(v = max_lag + 1 + h,   lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# ── Outcomes ─────────────────────────────────────────────────────────
#
# Left panel (filter coefficients):
#   - Unlike the MSE predictor, the PCS filters assign non-zero weight up to
#     the farthest lag k = q.
#   - As the slope parameter beta increases, weight is progressively shifted
#     away from recent observations toward older lags. This is counter-intuitive
#     but is a direct algebraic consequence of enforcing the CCF slope constraint.
#   - The most extreme PCS designs exhibit negative weights on older lags,
#     which causes trend inversion — a direct cost of aggressive look-ahead.
#
# Right panel (CCFs):
#   - The MSE predictors maximise the CCF at their respective horizons h and
#     h_tilde, but the peak remains at k = 0 in both cases (stuck at present).
#   - Very large lambda forces the CCF to follow a strictly linear path from
#     k = 0 to k = h = 5, regardless of whether this is necessary (assuming 
#     the problem is feasible).
#   - A positive slope parameter beta relocates the CCF peak toward h = 5,
#     as intended by the PCS design.
#   - Both the linear CCF constraint and a large positive beta are more
#     restrictive than necessary for look-ahead purposes, and both reduce
#     the achievable target correlation at k = h (the lowest peak value 
#     is attained by the largest beta PCS, violet line).
#   - The loss in target correlation at lag h is minimised subject to the
#     imposed constraints; however, the constraints themselves are overly
#     tight in this example, making the trade-off unnecessarily costly.

# Technical note:
#   As the regularisation weight lambda increases, the CCF slope at each
#   constrained lag converges to beta / (b' * b), i.e., the target slope
#   beta divided by the squared norm of the filter vector b. The dependence
#   on ||b||^2 is undesirable; however, eliminating it would require an
#   additional unit-length constraint b' * b = 1, which would complicate
#   the geometry and typically lead to multiple solutions (cf. the unitary
#   DFT in Tutorial 4). Crucially, this unwanted scaling effect does not
#   affect the look-ahead properties of the predictor, since the CCF peak
#   location is invariant to the scaling of b.



# ─────────────────────────────────────────────────────────────────────
# 3.6 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 3.6.1 Forecast Comparison ────────────────────────────────────────

# Generate a long white-noise series for reliable empirical evaluation.
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each
# predictor output relative to the target series.
anf <- 390
enf <- 430

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   As beta increases, the predictor output shifts progressively to the left
#   (further ahead in time) relative to the MSE predictor. This visual lead
#   is confirmed quantitatively by the empirical CCFs below.

# ── 3.6.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 3.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", h, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", htilde, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))

# Outcome:
#   The empirical CCF confirms the population-level peak shift toward h = 5,
#   consistent with the population results in Section 3.5. However, because
#   lambda is very large and the slope beta is also unnecessarily large, the
#   imposed linear CCF constraint is overly tight: the predictor is forced to
#   satisfy restrictive slope requirements across all lags in {0, …, h},
#   reducing the achievable target correlation at h more than necessary.
#   Exercise 4 addresses this by holding beta fixed to a smaller value and 
#   varying the regularisation weight across a range of values, demonstrating 
#   how a lighter design can recover target correlation while preserving the
#   essential peak-shifting behaviour.



# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: PCS I) — Relaxing Constraints via the Regularisation Weight
# ════════════════════════════════════════════════════════════════════
#
# We apply PCS type I) with h = 5, fixing the slope parameter beta=0.1 to a
# moderately small positive value and varying the regularisation weight lambda
# across a range of values. The goal is to explore how relaxing the structural
# constraints on the CCF — imposed rigidly in Exercise 3 — affects the
# balance between peak shifting and target correlation, see Wildi (2026), 
# Appendix D for background. 

# ─────────────────────────────────────────────────────────────────────
# Note: Exercises 1 & 3 must be run before this exercise, as they initialise
# the empirical framework (process specification, filter length, forecast
# horizon, MA coefficient vector an constraints) required by the subsequent 
# exercise.
# ─────────────────────────────────────────────────────────────────────

#
# ─────────────────────────────────────────────────────────────────────
# 4.1 Fixed Beta, Variable Regularisation Weight Lambda
# ─────────────────────────────────────────────────────────────────────

# Fix the target slope and vary the regularisation weight.
# A moderately small positive beta encourages a peak shift toward h without
# being as restrictive as the large-beta cases in Exercise 3.
beta <- 0.05

# Constrained lag set: Type I) requires a positive slope at every lag
# from 1 to h, enforcing a monotonically increasing CCF over {0, …, h}.
Delta <- 1:h

# Range of regularisation weights: from near-zero (loose constraints, high
# flexibility) to very large (tight constraints, near-exact slope enforcement).
lambda_vec <- c(0.1, 0.5, 1, 5, 10, 30,10000)

b_mat <- NULL    # filter coefficients, one column per lambda value

for (i in seq_along(lambda_vec)) {
  
  lambda <- lambda_vec[i]
  
  # Compute the regularised PCS Type I) predictor.
  PCS_obj <- PCS_func(h,Delta, xi, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: the residual abs(d_delta %*% b + beta) measures how
  # closely each slope constraint is satisfied. As lambda increases, these
  # residuals shrink toward zero (exact constraint satisfaction). Smaller
  # lambda permits larger residuals, allowing the predictor more freedom to
  # maximise target correlation at the cost of looser slope control.
  print(abs(d_delta %*% b + beta))
}


# The MSE-optimal PCS differs from the 'ordinary' PCS b only by an MSE-optimal
# scaling factor. The ordinary PCS is based on the regularised criterion (46)
# in Wildi (2026), which does not intrinsically scale to optimal MSE performance.


colnames(b_mat) <- paste0("lambda=", round(lambda_vec, 3), ", beta=", beta)

# ─────────────────────────────────────────────────────────────────────
# 4.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS slope constraints ───────────────────────────────────
# Validated in the loop above: the magnitude of the constraint residuals is 
# larger than in example 3 since lambda is smaller (less strong regularization). 

# ── Check 2: Sign / orientation preservation ─────────────────────────
# In contrast to exercise 3 all but the last design preserve trend orientation 
# at frequency zero. It takes a very large lambda to invert trend 
# orientation. 
apply(b_mat, 2, sum)

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms a positive inner product with gamma_h for all PCS variants,
# ensuring a positive target correlation at lag h.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, gammahtilde, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("MSE(", htilde, ")"),
                          paste0("lambda=", round(lambda_vec, 3),
                                 ", beta=", beta))

# ─────────────────────────────────────────────────────────────────────
# 4.3 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncated to the first q+1 lags where the MA process has support.
mplot <- filter_mat[1:(q + 1), ]
plot(mplot[, 1],
     main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat[1:q, ]

plot(mplot[, 1],
     main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1,     lty = 1)   # lag 0
abline(v = max_lag + 1 + h, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Outcome:
#   The results are qualitatively similar to Exercise 3, but with two key
#   differences:
#
#   - For fixed beta = 0.05, increasing lambda progressively tightens the
#     slope constraints, pushing the CCF peak toward k = h = 5. Values of
#     lambda > 100 can be regarded as strong regularisation, producing near-
#     linear CCF profiles as in Exercise 3.
#
#   - Reducing lambda relaxes the monotonicity constraints, granting the
#     optimiser more freedom to maximise target correlation at lag h (peak
#     height) without being forced to maintain a rigid linear CCF path. 
#     As an example, the designs based on lambda=30 (blue) and lambda=10000 
#     (violet) both shift the peak CCF to h. But the less strong regularization 
#     lambda=30 (blue) achieves a larger target correlation (a higher peak value).
#     This illustrates the core accuracy–timeliness trade-off: lighter
#     regularisation favours correlation height; heavier regularisation
#     affirms `clearer' peak location through tighter (linear) control of the 
#     CCF slope.


# ─────────────────────────────────────────────────────────────────────
# 4.4 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 4.4.1 Forecast Comparison ────────────────────────────────────────

# Generate a long white-noise series for reliable empirical evaluation.
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each
# predictor output relative to the target series.
anf <- 390
enf <- 430

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   Qualitatively similar to Exercise 3. The key advantage here is that using
#   a weaker slope (beta = 0.05) combined with a moderate regularisation weight
#   (lambda ~ 30) is sufficient to achieve full look-ahead behaviour — shifting
#   the CCF peak to k = h = 5 — while maintaining a tighter tracking of the
#   target at horizon h, i.e., a higher peak height than in Exercise 3.

# ── 4.4.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 4.3 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", h, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", htilde, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat) - 1]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: relaxed regularisation PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))

# Outcome:
#   Qualitatively similar to Exercise 3. The key difference is that the
#   relaxed regularisation (smaller lambda) allows the predictor to shift
#   the CCF peak to k = h = 5 while achieving a higher peak height,
#   reflecting stronger target tracking and a better balance between
#   look-ahead behaviour and target correlation.






# ════════════════════════════════════════════════════════════════════
# Exercise 5 Peak Correlation Shifting AND Complete Decoupling
# ════════════════════════════════════════════════════════════════════
# We compare a DFP fully decoupled design (exercise 5.2) with a PCS design 
# whose slope beta is selected to provide virtual full decoupling, up to 
# negligible deviation (exercise 5.3).

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 5.1 MSE-Optimal h-Step-Ahead Predictor
# ─────────────────────────────────────────────────────────────────────

# Filter length and forecast horizon
L <- 20
h <- 5

# Feasibility check: for an MA(q) process, the h-step-ahead MSE predictor
# is identically zero when h > q, because all innovations more than q steps
# ahead are unobservable.
if (h > q )
  print("(q + 1) must exceed h; otherwise the MSE-optimal forecast is zero.")

# Optimal MSE filter: retain MA coefficients from lag h onward; pad to length L.
# gammah[k] = b_{h+k} for k = 0, …, q-h, and 0 thereafter.
gammah <- c(b_ma[(h + 1):(q + 1)], rep(0, L - (q - h + 1)))


# ─────────────────────────────────────────────────────────────────────
# 5.2 DFP Full Decoupling
# ─────────────────────────────────────────────────────────────────────

# DFP contraint: b' * gamma0=alpha0. Full decoupling means alpha0 = 0. 
gamma_constraint <- gamma0
alpha0<-0

# Compute MSE-PCS predictor 
b_dfp <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0


# ─────────────────────────────────────────────────────────────────────
# 5.3 PCS Peak Shifting with Full Decoupling
# ─────────────────────────────────────────────────────────────────────

# The following combination of constraint parameters has been determined 
# empirically to align with quasi full decoupling of the PCS:
beta<-c(0.05265)
Delta <- 1:h
lambda <- 1000

# Compute PCS Type I) predictor.
PCS_obj <- PCS_func(h,Delta, xi, L, beta, lambda)

# Retrieve MSE optimally scaled PCS: to align with scale of MSE-DFP above. 
b_pcs <- PCS_obj$b


b_mat<-cbind(b_dfp,b_pcs)
colnames(b_mat) <- c("DFP","PCS")

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          colnames(b_mat))

# ─────────────────────────────────────────────────────────────────────
# 5.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncated to the first q+1 lags where the MA process has support.
mplot <- filter_mat[1:(q + 1), ]
plot(mplot[, 1],
     main = "Filter coefficients: Fully Decoupled DFP and PCS",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
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
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")
mplot <- ccf_mat[1:q, ]

plot(mplot[, 1],
     main = "Population CCFs",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1,     lty = 1)   # lag 0
abline(v = max_lag + 1 + h, lty = 2)   # lag h
axis(1, at = 1:nrow(mplot), labels = -max_lag - 1 + 1:nrow(mplot))
axis(2)
box()

# Inspect the CCF at lags k=0 and k=h:
ccf_mat

# Outcome:
#   Both designs achieve full or near-full decoupling: the CCF (right plot) is 
#   exactly zero (DFP) or nearly zero (PCS) at lag k = 0.
#
#   Under an equivalent full-decoupling constraint, the DFP predictor is
#   guaranteed to at least equal (in general outperform) PCS (or any other 
#   predictor) in terms of target correlation at the forecast horizon k = h = 5
#   (see ccf_mat above).
#   In practice, however, the margin of outperformance may be small. Here, 
#   Inspecting the CCF plot (or the corresponding table) reveals a subtle
#   qualitative difference: the PCS CCF (cyan) is marginally more linear over
#   {0, …, h} (the degree of linearity governed by the choice of lambda),
#   whereas the DFP CCF (red) is slightly flexed (convex) over the same interval. 
#   DFP imposes no particular path between k = 0 and k = h — it only requires 
#   CCF(0) = 0 and maximises CCF(h), leaving the intermediate profile 
#   unconstrained. In the considered MA(9) example, this amounts to an almost 
#   linear (very slightly flexed) path.
#
#   When full decoupling is the primary objective, DFP offers a simpler
#   implementation (setting alpha0 = 0 directly enforces CCF(0) = 0) and
#   is guaranteed to maximise CCF height at lag h subject to the decoupling
#   constraint. However, DFP provides no guarantee that the CCF peak is
#   actually located at k = h: depending on the process structure, the peak
#   may remain near k = 0 or at some intermediate (or higher) lag. 


# ════════════════════════════════════════════════════════════════════
# Exercise 6 Geometry of the PCS Predictor
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 6.1 Geometry Case II)
# ─────────────────────────────────────────────────────────────────────
# See Wildi section 3.2 for background.
#
# Vector components (edit these)
# Vector components (edit these)
vx <- 2
vy <- 2
vx <- 2
vy <- 2

# Specify gammahm1 and gammah
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
# Specify lambda0
lambda0<-0.3
# Lengths
l0<-sqrt(sum(gammahm1^2))
lh<-sqrt(sum(gammah^2))

# Angle between gammah and gammahm1: gammah is above (larger angle)
theta_h <- atan2(gammah[2], gammah[1])-atan2(gammahm1[2], gammahm1[1])

# Set up plot limits with some padding
x_min<-0
x_max<-1.5
y_min<--0.2
y_max<-1.2
lim <- 1.2 * max(1, abs(c(vx, vy))+0.5)
par(mfrow=c(1,1))
plot(NA, xlim = c(x_min,x_max+0.3), ylim = c(y_min, y_max),xlab = "", ylab = "", axes = TRUE,asp=1)

#     asp = 1, xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gammahm1
arrows(0, 0,gammahm1[1],gammahm1[2], length = 0.12, lwd=1, col = "black")
text(gammahm1[1]+0.1,gammahm1[2], labels = expression(gamma[h-1]), col = "black", cex = 1.2)
# gammah
arrows(0, 0,gammah[1],gammah[2], length = 0.12, lwd=1, col = "black")
text(gammah[1]+0.1,gammah[2], labels = expression(gamma[h]), col = "black", cex = 1.2)
# Insert unit length b1: first solution corresponding to beta=0
b0<-c(0.7,1.05)
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.1,b0[2]/lb0, labels = expression(b[1]), col = "red", cex = 1.2)
ls<-2
segments(0,0,ls*b0[1],ls*b0[2],  lwd = 1,lty=2, col = "red")

# Orthogonal projection of gammah onto b0
x1<--0.27
theta_gammah<-atan(b0[2]/b0[1])
segments(gammah[1],gammah[2],gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah), lwd = 1,lty=2, col = "red")
#text(gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah)+0.03, labels =expression(b[0]*gamma[h])) 
# Orthogonal projection of gammahm1 onto b0
x1<--1
theta_gammahm1<-atan(b0[2]/b0[1])
segments(gammahm1[1],gammahm1[2],gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1), lwd = 1,lty=2, col = "red")
text(gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[1]*gamma[h-1]),col="red") 
text(gammahm1[1]+x1*sin(theta_gammahm1)+0.1,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels ="=",col="red") 
text(gammahm1[1]+x1*sin(theta_gammahm1)+0.18,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[1]*gamma[h]),col="red") 


#text(gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[0]*gamma[h-1]-b[0]*gamma[h]=0)) 

# Angle between gammah and gammahm1
theta_h <- atan2(gammah[2], gammah[1])-atan2(gammahm1[2], gammahm1[1])
# Draw the angle 
r <- 0.4 * lh  # arc radius
th_seq <- atan2(gammahm1[2], gammahm1[1])+seq(0, theta_h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)
text(1.15 * r * cos(th_seq[length(th_seq)]), 1.15 * r * sin(th_seq[length(th_seq)]-0.1),
     labels = expression(theta[hh-1]), col = "black", cex = 1.2)

# Angle between gammah and b1
theta_hm1 <- atan2(b0[2], b0[1])-atan2(gammah[2], gammah[1])
# Draw the angle theta_h (between gammah and b0)
r <- 0.3 * lh  # arc radius
thm1_seq <- atan2(gammah[2], gammah[1])+seq(0, theta_hm1, length.out = 100)
lines(r * cos(thm1_seq), r * sin(thm1_seq), col = "red", lwd=1)
text(1.15 * r * cos(thm1_seq[length(thm1_seq)])+0.05, 1.15 * r * sin(thm1_seq[length(thm1_seq)])-0.05,
     labels = expression(theta[hb1]), col = "red", cex = 1.2)

# Unit circle
theta <- atan2(b0[2], b0[1])-atan2(gammahm1[2], gammahm1[1])
r <- 1  # arc radius
th_seq <- seq(-0.2, pi/2+0.2, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1,lty=2)


# Insert second unit length b2: with beta<0
b0<-c(0.4,1.05)
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "blue")
text(b0[1]/lb0-0.1,b0[2]/lb0-0.05, labels = expression(b[2]), col = "blue", cex = 1.2)
ls<-2
segments(0,0,ls*b0[1],ls*b0[2],  lwd = 1,lty=2, col = "blue")

# Orthogonal projection of gammah onto b0
x1<--0.83
theta_gammah<-atan(b0[2]/b0[1])
segments(gammah[1],gammah[2],gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah), lwd = 1,lty=2, col = "blue")
text(gammah[1]+x1*sin(theta_gammah)-0.05,gammah[2]-x1*cos(theta_gammah)+0.03, labels =expression(b[2]*gamma[h]),col="blue") 
# Orthogonal projection of gammahm1 onto b0
x1<--1.29
theta_gammahm1<-atan(b0[2]/b0[1])
segments(gammahm1[1],gammahm1[2],gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1), lwd = 1,lty=2, col = "blue")
text(gammahm1[1]+x1*sin(theta_gammahm1)-0.08,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[2]*gamma[h-1]),col="blue") 

# Angle between gammah and b2
theta_hm1 <- atan2(b0[2], b0[1])-atan2(gammah[2], gammah[1])
# Draw the angle theta_h (between gammah and b0)
r <- 0.2 * lh  # arc radius
thm1_seq <- atan2(gammah[2], gammah[1])+seq(0, theta_hm1, length.out = 100)
lines(r * cos(thm1_seq), r * sin(thm1_seq), col = "blue", lwd=1)
text(1.15 * r * cos(thm1_seq[length(thm1_seq)])+0.08, 1.15 * r * sin(thm1_seq[length(thm1_seq)])-0.02,
     labels = expression(theta[hb2]), col = "blue", cex = 1.2)

# Explanation:
#   The PCS predictor (Type II) lives in the plane spanned by gamma_h and
#   gamma_{h-1} (see Wildi 2026, section 3.2). Therefore, we require gammah and 
#   gamma_{h-1} to be linearly independent. If not, a perturbation based solution is proposed in Tutorial ???. 
#   The PCS type II is defined by the following 
#   optimisation:
#
#     Objective:  b' * gamma_h  ->  max        (maximise target correlation)
#     Constraint: b' * (gamma_h - gamma_{h-1}) = beta  (enforce CCF slope at h)
#
#   The geometric construction in the plot above illustrates how these two
#   requirements are reconciled simultaneously, showing two PCS solutions:
#
#     Red solution (b1):   b1' * (gamma_h - gamma_{h-1}) = beta = 0.
#                          No slope is imposed; b1 lies as close as possible
#                          to gamma_h within the plane.
#
#     Blue solution (b2):  b2' * (gamma_h - gamma_{h-1}) = beta > 0.
#                          A strictly positive slope is enforced, rotating b2
#                          away from gamma_h.
#
#   A larger beta rotates b further away from gamma_h, on the side opposite
#   to gamma_{h-1}, increasing the angle theta_{h,b}: theta_{h,b2} >
#   theta_{h,b1}. This rotation may be linked to the distinction between
#   standard and non-standard forecasting cases discussed in Tutorial 9.
#
#   Geometrically, for a given beta, the unit-length PCS predictor is obtained by
#   projecting gamma_h orthogonally onto the line (a radius of the unit
#   sphere) in the plane spanned by gamma_h and gamma_{h-1} that satisfies
#   the slope constraint b' * (gamma_h - gamma_{h-1}) = beta. The resulting
#   predictor maximises the inner product with gamma_h — and hence the target
#   correlation — among all unit vectors (predictors) satisfying the constraint.
#


# ─────────────────────────────────────────────────────────────────────
# 6.2 Recovering the Phase Excess Theta from the PCS Constraint Beta
# ─────────────────────────────────────────────────────────────────────
# See Appendix C, Wildi (2026).
#
# The angle theta_{hb} (phase excess) visible in the plot above can be
# recovered analytically from the PCS slope parameter beta. The derivation
# follows Appendix C of Wildi (2026).

# Define the gamma vectors and slope parameter used in the plot above.
# These match the values used in Section 6.1.
gammahm1 <- c(3, 1) * 0.44 * 1.2     # gamma_{h-1}: (h-1)-step MSE predictor
gammah   <- c(1.5, 1) * 0.66 * 1.2   # gamma_h:     h-step MSE predictor
betah    <- -0.1                       # target PCS slope parameter

# Compute lengths and the angle between gamma_h and gamma_{h-1}.
lh    <- sqrt(sum(gammah^2))           # ||gamma_h||
lhm1  <- sqrt(sum(gammahm1^2))        # ||gamma_{h-1}||
thetah <- atan2(gammah[2],   gammah[1]) -
  atan2(gammahm1[2], gammahm1[1])  # angle between the two MSE predictors

# Define the coefficients a, b, c of the trigonometric equation
# a*cos(theta) + b*sin(theta) = R*cos(theta-phi), as derived in Appendix C, Wildi (2026).
# Solving this equation for theta yields the phase excess theta_{hb}.
a <-  lh - cos(thetah) * lhm1   # coefficient of cos(theta)
b <-  sin(thetah) * lhm1        # coefficient of sin(theta)
c <- -betah                      # right-hand side (negated slope parameter)

# General solver for the above equation. Returns 0, 1, or 2 solutions in 
# (-pi, pi], together with diagnostic information (feasibility status, 
# phase shift phi, and amplitude R).
solve_acos_bsin_eq <- function(a, b, c) {
  R <- sqrt(a^2 + b^2)   # amplitude of the right-hand side
  
  # Degenerate case: both coefficients are zero.
  if (R == 0) {
    if (c == 0)
      return(list(status = "infinite solutions (all theta)", theta = NULL,
                  phi = NA, R = 0))
    return(list(status = "no solution", theta = NULL, phi = NA, R = 0))
  }
  
  phi <- atan2(b, a)     # phase shift: rewrites RHS as R * cos(theta - phi)
  x   <- c / R           # normalised right-hand side; must lie in [-1, 1]
  
  # Clamp x for numerical safety before calling acos().
  x <- max(min(x, 1), -1)
  
  # Infeasible: |c| exceeds the maximum amplitude R.
  if (abs(c) > R + .Machine$double.eps^0.5)
    return(list(status = "no real solution (|c| > R)", theta = NULL,
                phi = phi, R = R))
  
  # Boundary case: exactly one solution modulo 2*pi.
  if (abs(abs(x) - 1) < 1e-14) {
    theta <- if (x > 0) phi else (phi + pi)
    theta <- atan2(sin(theta), cos(theta))   # wrap to (-pi, pi]
    return(list(status = "one solution modulo 2pi", theta = theta,
                phi = phi, R = R))
  }
  
  # Generic case: two solutions modulo 2*pi.
  alpha  <- acos(x)
  theta1 <- phi + alpha
  theta2 <- phi - alpha
  
  # Helper to wrap an angle to (-pi, pi].
  wrap <- function(t) atan2(sin(t), cos(t))
  theta <- sort(c(wrap(theta1), wrap(theta2)))
  
  list(status = "two solutions modulo 2pi", theta = theta, phi = phi, R = R)
}

# Solve for the phase excess theta_{hb} corresponding to the given beta.
# The two solutions (when they exist) correspond to the two intersection
# points of the constraint line with the unit circle in the (gamma_h, gamma_{h-1})
# plane; the geometrically meaningful solution is the one consistent with
# the optimisation direction (maximising b' * gamma_h).
solve_acos_bsin_eq(a, b, c)






# ════════════════════════════════════════════════════════════════════
# MAIN TAKE-AWAYS
# ════════════════════════════════════════════════════════════════════
#
# 1. Aggregate lead measure:
#    PCS targets an aggregate, frequency-integrated measure of lead — the
#    location of the CCF peak. In principle, it can therefore enforce stronger
#    and more systematic look-ahead behaviour than DFP, which controls the
#    lead only locally at frequency zero (see Tutorials 6-9).
#
# 2. Three PCS design types:
#    Three design variants (Types I, II, and III) were presented, each
#    conditioning an effective CCF peak shift — under suitable conditions —
#    with varying degrees of restrictiveness:
#      - Type II  (weakest):    requires a positive CCF slope at lag h only,
#                               i.e., CCF(h-1) < CCF(h).
#      - Type III (moderate):   requires a positive average slope from k = 0
#                               to k = h, i.e., CCF(0) < CCF(h).
#      - Type I   (strongest):  requires a strictly increasing CCF over the
#                               full interval {0, …, h}.
#    More restrictive designs offer stronger guarantees on peak relocation
#    but at a greater potential cost to target correlation. In some cases the
#    problem is infeasible — either because the constraints cannot be
#    satisfied or because the target correlation at h is negative at the  
#    constraint values. In other cases, even when the constraints are 
#    satisfied, none of the three types succeeds in relocating the CCF peak, 
#    i.e., the global maximum, exactly to k = h. Nevertheless,
#    enforcing any of the above types generally produces measurable look-ahead
#    behaviour, provided the problem is at least approximately feasible. Even
#    fully infeasible problems can be addressed by the regularised Type I PCS,
#    though sizable constraint residuals may persist even as the regularisation
#    weight is increased.
#
# 3. Feasibility and efficiency:
#    When a peak shift is feasible, it can often be achieved without imposing
#    unnecessarily severe structural constraints, e.g., a linearly increasing 
#    CCF. Choosing the mildest design type that still relocates the peak — 
#    and tuning the regularisation weight accordingly — preserves as much target 
#    correlation as possible, keeping the accuracy–timeliness trade-off 
#    efficient. As demonstrated in Exercises 4 and 5, the gain in target 
#    correlation from relaxing constraints may be minor in some settings and 
#    substantial in others, depending on the problem formulation, i.e., the 
#    process structure and forecast horizon.
#
# 4. Geometric distinction between DFP and PCS:
#    DFP and PCS operate in geometrically distinct subspaces: the DFP
#    predictor lies in the plane spanned by (gamma_0, gamma_h), whereas PCS
#    predictors lie in planes spanned by gamma_h and one or more gamma_k for
#    k in {0, …, h-1}. This geometric difference underlies their distinct
#    behaviour in terms of CCF shape, peak location, and target correlation.
#    As illustrated in Exercise 5, however, the practical differences between
#    the two approaches can be marginal in certain process structures and
#    forecast settings.





