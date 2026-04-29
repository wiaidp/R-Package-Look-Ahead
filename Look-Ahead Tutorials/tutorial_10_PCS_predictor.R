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
# predictor so that the CCF peak is shifted, ideally toward lag h, thereby
# improving timeliness and look-ahead behaviour. While a full shift from lag 0
# to lag h is not always feasible, even a partial attenuation of the original
# peak at lag 0 in difficult forecast problems can be effective. This weaker 
# objective is precisely the strategy of the DFP (Decoupling From Present) 
# predictor in earlier tutorials.
#
# ── DFP Recap ────────────────────────────────────────────────────────
#
# - Decoupling From Present (DFP) weakens the CCF at lag 0 while maximising
#   it at the forecast horizon h (dilemma, conflicting objectives).
#
#     - In some cases, the CCF peak migrates toward lag h as a by-product.
#         - As an example, under full decoupling, CCF(0) = 0. If the target 
#           correlation at lag h remains positive (not always possible; 
#           see Tutorial 9), the CCF peak can no longer be at lag 0.
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
#       via one of three design choices (Conditions I, II, and III below),
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
#       (see Tutorial 6), PCS targets an aggregate lead effect that is ideally
#       strong enough to relocate the CCF peak. The CCF is itself an aggregate,
#       frequency-integrated measure of lead, so PCS effectively acts across
#       the entire frequency band rather than at frequency zero alone.
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
# Three necessary (but not sufficient) conditions must hold for the CCF peak
# to shift from lag 0 to lag h > 0:
#
#   I)  Monotonically increasing CCF over {0, …, h} (most restrictive):
#       The CCF must be strictly increasing across the full interval, i.e.,
#       CCF(k-1) < CCF(k) for all k = 1, …, h. See Wildi (2026), Section 3.2 
#       and Appendix E. This condition is generally not exactly feasible 
#       (see Exercise ???); The principal PCS optimization function 
#       PCS_shift_func() enforces it as closely as possible via regularisation.
#
#   II) Local positive slope at the target lag (weaker than I):
#       The CCF must be increasing over the final step only, i.e.,
#       CCF(h-1) < CCF(h). See Wildi (2026), Section 3.2.
#       In some cases where additional structure is imposed by the data-
#       generating process (e.g., from the Yule-Walker equations of an AR(p)),
#       conditions I) and II) may become equivalent.
#
#   III) Positive average slope from lag 0 to lag h (weaker than I):
#       The CCF must be increasing on average from k = 0 to k = h, i.e.,
#       CCF(0) < CCF(h).
#
#   A link to decoupling:
#       Even the weaker conditions II) and III) are not always exactly
#       feasible. When they are, however, both can be imposed within the DFP
#       optimisation framework by using a suitably modified constraint vector
#       (see Exercise 1 below).
#
# Technical note on Condition I):
#   When the full monotonicity constraint is feasible, strong regularisation
#   enforces a fixed positive CCF slope uniformly across all lags in {0, …, h},
#   which may be unnecessarily costly in terms of target correlation. In
#   practice, therefore, mild regularisation is typically preferred: it nudges
#   the CCF toward monotonicity while retaining enough flexibility to balance
#   the inherent dilemma between peak shifting and peak height, see 
#   exercise 4 below.
#

# ── Key Identity ─────────────────────────────────────────────────────
#
# The CCF at lag k is given by:
#
#     CCF(k) = b' * gamma_k
#
# where:
#   b       : filter coefficient vector (the predictor to be designed),
#   gamma_k : k-step-ahead MSE predictor coefficients (forward-shifted Wold
#             coefficients), normalised to unit length.
#
# CCF and unit-length scaling:
#   - Strictly speaking, the formula above requires both b and gamma_k to be
#     of unit length.
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
# Let beta > 0 be the PCS slope parameter. Using the key identity above,
# Conditions I)–III) translate into linear constraints on b:
#
#   Condition I)   requires  b' * (gamma_k - gamma_{k-1}) > 0,  k = 1, …, h
#   Condition II)  requires  b' * (gamma_h - gamma_{h-1}) > 0   (k = h only)
#   Condition III) requires  b' * (gamma_h - gamma_0)     > 0   (k = h only)
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
# Condition III) is handled by setting Delta = {h} and replacing the
# differencing vector with (gamma_h - gamma_{h-delta}), with delta = h.
#
# Note: the objective (maximising target correlation) can alternatively be
# replaced by minimum MSE (assuming a standard case, see the discussion in 
# Tutorial 9). The two are equivalent up to simple (MSE-optimal) scaling.
#
# ── Regularised PCS (Implemented in the main function PCS_shift_func) ────────
#
# The equality constraints above may be infeasible when the system is rank-
# deficient — for example, because of additional structure imposed by the
# Yule-Walker equations. To handle this, Wildi (2026) proposes a regularised
# version of the criterion (Equation 46, Appendix D). The regularised
# formulation replaces the hard equality constraints with a soft penalty term,
# yielding a criterion that is always well-defined and admits a unique solution
# regardless of feasibility. PCS_shift_func() implements this regularised PCS
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
#   Exercise 1 — PCS II) via a modified DFP approach (h = 1, one-step ahead):
#                Imposes a local slope constraint at lag h by replacing the
#                standard DFP decoupling vector gamma_0 with the difference
#                (gamma_{h-1} - gamma_h), directly enforcing a positive CCF
#                slope at the target horizon. While the MSE predictor remains
#                stuck at the present irrespective of the forecast horizon,
#                the PCS predictor successfully moves the CCF peak from k = 0
#                to k = h = 1, inducing effective look-ahead behaviour.
#
#   Exercise 2 — PCS III) for h = 5 via PCS_shift_func():
#                Relocates the CCF peak at k=h=5 by enforcing a positive 
#                average growth of the CCF over lags {0, …, h}, i.e., 
#                requiring CCF(0) < CCF(h), without constraining intermediate 
#                lags.
#
#   Exercise 3 — PCS I) for h = 5 with a fixed an very large regularisation 
#                weight and a range of slope parameters:
#                Relocates the CCF peak by enforcing a linearly increasing CCF
#                over the full interval {0, …, h}. This is more constraining
#                than Exercise 2, as it imposes a uniform positive slope at
#                every intermediate lag, effectively prescribing a linear path
#                for the CCF from k = 0 to k = h = 5.
#
#   Exercise 4 — PCS I) for h = 5 with a single (fixed) slope and a range of
#                regularisation weights:
#                Repeats Exercise 3 but varies the regularisation strength across
#                a range of values, holding the target slope fixed. This
#                illustrates how the penalty strength governs the
#                accuracy–timeliness trade-off: lighter regularisation permits
#                departures from the strict linear CCF path imposed in
#                Exercise 3, providing fine-tuning flexibility to better balance
#                peak shifting against target correlation.
#
#   Exercise 5 — PCS vs. fully decoupled DFP:
#                Compares the look-ahead behaviour and CCF profiles produced
#                by PCS and the fully decoupled DFP predictor, highlighting
#                differences in peak location, peak height, and target
#                correlation.
#
#   Exercise 6 — Geometry of the PCS predictor:
#                Visualises the PCS II predictor in the plane spanned by
#                gamma_h and gamma_{h-1}, illustrating the geometric
#                interpretation of the local slope constraint. A comparison
#                with Tutorial 5, Exercise 1.6 reveals that DFP and PCS are
#                conceptually distinct approaches — operating in different
#                geometric planes and enforcing different constraints — even
#                when their resulting predictors may appear similar in some 
#                applications.
#
#
#   Exercise 7 — Infeasibility:???
#                Illustrates cases where the monotonicity constraints are
#                rank-deficient and the exact PCS criterion admits no solution,
#                demonstrating how regularisation recovers a best-effort
#                approximation in the presence of structural infeasibility.
#

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
ts.plot(cbind(gammah,gammahtilde),main=paste("MSE(",h,") and MSE(",htilde,") predictors",sep=""))

# ─────────────────────────────────────────────────────────────────────
# 1.3 PCS Condition II): Setting Up the Modified DFP Constraint
# ─────────────────────────────────────────────────────────────────────
# Condition II) requires CCF(h) > CCF(h-1), i.e., the CCF must increase
# over the final step to the forecast horizon h. Using CCF(k) = b' * gamma_k, 
# this becomes:
#
#   b' * gamma_{h-1} < b' * gamma_h
#   <=>  b' * (gamma_{h-1} - gamma_h) < 0
#
# Equivalently, setting alpha0 = b' * gamma_constraint with
#   gamma_constraint = gamma_{h-1} - gamma_h,
# Condition II) requires alpha0 < 0.
#
# This maps exactly onto a standard DFP decoupling problem: minimise MSE
# subject to b' * gamma_constraint = alpha0, with gamma_constraint playing
# the role of gamma_0. We therefore apply compute_mse_dfp() with
# gamma_constraint in place of gamma_0, and decrease alpha0 below the MSE
# baseline to progressively enforce the slope condition.
#
# Remark on interpretation:
#   Unlike the classic DFP constraint (which decouples b from the observable
#   present via gamma_0), the constraint vector gamma_constraint = gamma_{h-1} - gamma_h
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
# 1.4 PCS (DFP) Optimisation over the Decoupling Grid
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
    max_lag, h, as.vector(b), gamma0)$cor_vec
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
    max_lag, h, filter_mat[,i], gamma0)$cor_vec)
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
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",htilde,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MA(9) vs. strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))






# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: PCS III) Impose an Average Positive Slope of the CCF
# ════════════════════════════════════════════════════════════════════

# We apply the PCS based on case III), assuming h=5.

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
# 2.2 PCS Condition III): Setting Up the Modified PCS Constraint
# ─────────────────────────────────────────────────────────────────────
# Condition III) requires CCF(h) > CCF(0), i.e., the CCF must increase
# on average from lag 0 to the forecast horizon h. Using 
# CCF(k) = b' * gamma_k, this becomes:
#
#   b' * gamma_0 < b' * gamma_h
#   <=>  b' * (gamma_0 - gamma_h) < 0
#
# Equivalently, setting alpha0 = b' * gamma_constraint with
#   gamma_constraint = gamma_0 - gamma_h,
# Condition III) requires alpha0 < 0.
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
    max_lag, h, as.vector(b), gamma0)$cor_vec
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
    max_lag, h, filter_mat[,i], gamma0)$cor_vec)
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
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",htilde,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MA(9) vs. strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))






# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: PCS I) 
# Variable Beta, Fixed Large Lambda
# ════════════════════════════════════════════════════════════════════

# We apply the PCS based on case I), assuming h=5 and regularization weight 
# lambda large (and fixed).

# ─────────────────────────────────────────────────────────────────────
# 3.1 MSE-Optimal h-Step-Ahead Predictor
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
# 3.2 PCS Condition I): Setting Up the Modified PCS Constraint
# ─────────────────────────────────────────────────────────────────────
#

beta_vec<-c(-0.2,-0.1,0,0.1,0.2,0.3)
Delta<-1:h
lambda<-100000

# ─────────────────────────────────────────────────────────────────────
# 3.3 PCS Optimisation over the Decoupling Grid
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

for (i in seq_along(beta_vec)) {#i<-1
  
  beta <- beta_vec[i]
 
# Compute PCS case I)   
  PCS_obj<-PCS_shift_func(Delta, xi, L, beta, lambda)
    
  b = PCS_obj$b
  d_delta = PCS_obj$d_delta
  b_mat <- cbind(b_mat, b)
  
# Check PCS constraints: when the system is feasible, these should converge to 
# zero as lambda \to \infty
  print(abs(d_delta %*% b + beta))
  
  
}

colnames(b_mat) <-  paste0("lambda=",lambda,", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 3.4 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS constraint  ──────

# This check is validated in the above loop: for a feasible system, the 
# deviations of the CCF-slope from the imposed slope (beta) should vanish 
# with increasing regularisation weight lambda


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
                          paste0("PCS lambda=",lambda,", beta=", round(beta_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 3.5 Plots and Performance Summary
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
    max_lag, h, filter_mat[,i], gamma0)$cor_vec)
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
#     alpha0 decreases, the average slope between lags 0 and h flattens and eventually
#     inverts, confirming a peak shift toward lag h.
#   - Increasing the forecast horizon (any admissible htilde<=9) does not 
#     shift the peak of the CCF of the MSE predictor.
#   - The loss in target correlation at lag h is minimised subject to the
#     modified decoupling constraint.
#   - 

# Tabular summary: CCF of PCS at lag 0 and lag h for each decoupling level
round(cor_vec_1, 2)


# ─────────────────────────────────────────────────────────────────────
# 3.6 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 3.6.1 Forecast Comparison ────────────────────────────────────────


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


# ── 3.6.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",htilde,"): Peak at lag k = 0 (no look-ahead)",sep=""))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MA(9) vs. strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))






# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: PCS I) 
# Variable Lambda, Fixed Beta 
# ════════════════════════════════════════════════════════════════════

# We apply the PCS based on case I), assuming h=5 and slope beta fixed. 

# ─────────────────────────────────────────────────────────────────────
# 4.1 Fixed Beta, Variable Regularisation Weight Lambda
# ─────────────────────────────────────────────────────────────────────

# Fix beta and vary lambda
beta<-0.3
Delta<-1:h
lambda_vec<-c(0.1,0.5,1,5,10,10000)


b_mat       <- NULL    # filter coefficients, one column per alpha0

for (i in seq_along(lambda_vec)) {#i<-1
  
  lambda <- lambda_vec[i]
  
  # Compute PCS case I)   
  PCS_obj<-PCS_shift_func(Delta, xi, L, beta, lambda)
  
  b = PCS_obj$b
  d_delta = PCS_obj$d_delta
  b_mat <- cbind(b_mat, b)
  
  # Check PCS constraints: when the system is feasible, these should converge to 
  # zero as lambda \to \infty
  print(abs(d_delta %*% b + beta))
  
  
}

# Note: as lambda increases, the magnitude of the errors 
# abs(d_delta %*% b + beta) computed withing the loop decreases. 

colnames(b_mat) <-  paste0("lambda=", round(lambda_vec, 3),", beta=",beta,sep="")


# ─────────────────────────────────────────────────────────────────────
# 4.2 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: PCS constraint  ──────

# This check is validated in the above loop: for a feasible system, the 
# deviations of the CCF-slope from the imposed slope (beta) should vanish 
# with increasing regularisation weight lambda


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
                          paste0("lambda=", round(lambda_vec, 3),", beta=",beta,sep=""))



# ─────────────────────────────────────────────────────────────────────
# 4.3 Plots and Performance Summary
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
    max_lag, h, filter_mat[,i], gamma0)$cor_vec)
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

# Outcome: for increasing regularisation weitght lambda, the slope 
# converges to beta/(b%*%b), i.e., beta divided by the length of b.

# Removing the dependence on the length of b would amount to more intricate 
# optimization with multiple solutions. However, this would not improve 
# the look ahead perspective.


###################################################################################
# In look ahead applications 
# 1. lambda not too large:  
#   -Gives more flexibility, higher target correlation, less overfitting, like 
#     exercise 3 which regulated the mean-growth (instead of each individual step).
# 2. beta not too negative. 
# Idea: we mainly want to displace the peak to obtain look ahead behaviour. 
# -Important: peak displacement
# -Maximal peak (target correlation)
# -Otherwise the shape of the CCF is not relevant.
# -Giving morre flexibility allows higher peak value.

# Connection to DFP:
# -Can also achieve full decoupling at lag 0, depending on h and slope. But that's not the purpose of PCS.
# -Note however that fully decoupled DFP and PCS would generally differ: geometrically, 
# the spans spanned by (gamma, gamma), i.e. the DFP, differ from (gammah,gamma_{h-1}) in the PCS.
###################################################################################


# ════════════════════════════════════════════════════════════════════
# Exercise 5 PCS vs. DFP
# ════════════════════════════════════════════════════════════════════
# Example: full decoupling and one of the PCS with positive slope and near full decoupling



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



# ─────────────────────────────────────────────────────────────────────
# 6.2 Recover Phase Excess Theta From PCS Constraint Beta
# ─────────────────────────────────────────────────────────────────────
# See Appendix C, Wildi (2026).


# R code for solving phase excess theta as a function of betah


# Define lengths of MSE predictors and angle thetah between them
# We use the same gammah, gammahm1 as in above plot
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
betah<--0.1

lh<-sqrt(sum(gammah^2))
lhm1<-sqrt(sum(gammahm1^2))
thetah<-atan2(gammah[2],gammah[1])- atan2(gammahm1[2],gammahm1[1])

# Define sides a,b,c:  Appendix C, Wildi (2026).
a<-lh-cos(thetah)*lhm1
b<-sin(thetah)*lhm1
c<--betah



solve_acos_bsin_eq <- function(a, b, c) {
  R <- sqrt(a^2 + b^2)
  if (R == 0) {
    if (c == 0) return(list(status = "infinite solutions (all theta)", theta = NULL, phi = NA, R = 0))
    return(list(status = "no solution", theta = NULL, phi = NA, R = 0))
  }
  phi <- atan2(b, a)                # phase shift
  x <- c / R
  # Clamp for numerical safety
  x <- max(min(x, 1), -1)
  if (abs(c) > R + .Machine$double.eps^0.5) {
    return(list(status = "no real solution (|c| > R)", theta = NULL, phi = phi, R = R))
  }
  if (abs(abs(x) - 1) < 1e-14) {
    # Single solution modulo 2π
    theta <- if (x > 0) phi else (phi + pi)
    theta <- atan2(sin(theta), cos(theta))  # wrap to (-pi, pi]
    return(list(status = "one solution modulo 2π", theta = theta, phi = phi, R = R))
  }
  alpha <- acos(x)
  theta1 <- phi + alpha
  theta2 <- phi - alpha
  # Wrap to (-pi, pi]
  wrap <- function(t) atan2(sin(t), cos(t))
  theta <- sort(c(wrap(theta1), wrap(theta2)))
  list(status = "two solutions modulo 2π", theta = theta, phi = phi, R = R)
}

# Find theta for given a,b,c
solve_acos_bsin_eq(a, b, c )



# ════════════════════════════════════════════════════════════════════
# EXERCISE 7: Infeasibility
# ════════════════════════════════════════════════════════════════════
# PCS applied to ARMA(1,1)








