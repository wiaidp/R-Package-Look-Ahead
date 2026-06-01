# ══════════════════════════════════════════════════════════════════════════════
# TUTORIAL 16 — RSC : RIGHT SKEWING THE CCF: 
#               APPLICATION TO THE HARDEST LOOK AHEAD FORECAST PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

# Fo reference see Tutorial 14.


# ── BACKGROUND: DFP AND PCS FORECASTING ───────────────────────────────────────

# The Decouple From Present (DFP) and Peak Correlation Shifting (PCS) approaches
# are designed to look ahead of the classical MSE predictor in difficult forecasting
# problems where the MSE predictor is "stuck at the present": its cross-correlation
# function (CCF) with the target x_{t+h} peaks at lag k=0 for any horizon h.
#
# PCS attempts to shift the CCF peak away from k=0, ideally placing it at k=h, while
# maintaining maximal correlation with x_{t+h}. This is achieved by constraining
# the CCF path over lags k=0,...,h. Four constraint types are considered:
#
#   Type I   PCS: CCF(k) - CCF(k-1) = beta_k >= 0,  for k = 1, ..., h
#   Type II  PCS: CCF(h) - CCF(h-1) = beta    >= 0
#   Type III PCS: CCF(h) - CCF(0)   = beta    >= 0
#   Type IV will be introduced and discussed in Tutorial 15.
#
# For further details, see Tutorial 13.
#
# Terminology:
#   - Feasible:    The constraint system is exactly solvable and the resulting
#                  CCF(h) > 0 (which does not ensure that the CCF peaks at h).
#   - Impossible:  The CCF peak cannot be shifted to k=h while maintaining a
#                  positive height. Impossible problems are generally also
#                  infeasible, though not always; see Tutorial 13, Exercise 1
#                  for a counterexample.
#───────────────────────────────────────────────────────────────────────────────

# ── THE AR(1) DGP: THE HARDEST CASE ───────────────────────────────────────────

# The AR(1) DGP represents the most challenging case for PCS (or DFP). Its 
# autocorrelation structure satisfies the Yule-Walker equations:
#
#   ACF(k) = a1 * ACF(k-1),
#
# which define a rank-one system that leaves no room to adjust or
# reshape the profile of the CCF for lags k=0,...,h (up to sign change). But 
# room is eventually left for NEGATIVE lags which PCS or DFP do not explicitly 
# address.
#
# Denoting the h-step MSE predictor in MA-form as gamma_h (with gamma_0 being
# the nowcast, i.e., the original Wold decomposition of the DGP), the Yule-Walker
# equations imply:
#
#   gamma_{h+k} = a1^k * gamma_{h},  for any h, k >= 0.
#
# This property is called self-similarity: all h-step MSE predictors are
# proportional to gamma_0, differing only by the scalar factor a1^h.
#
# Self-similarity forces the CCF of any predictor b to satisfy:
#
#   CCF(k) = (b %*% gamma_k) / (||b|| * ||gamma_0||) = a1^k * CCF(0).
#
# For a1 > 0, the CCF decays monotonically and exponentially from its peak at
# k=0. This pattern is rigidly enforced by the DGP on every predictor b via the
# Yule-Walker equations. Consequently, reshaping the CCF according to any of the
# PCS constraint types above is generally impossible (unless one merely replicates
# the original AR(1) profile).
#
# However, for negative lags k=-1,-2,...
#
#   CCF(k) = (b[-k+(1:L)] %*% gamma_0) / (||b|| * ||gamma_0||) 
#
# This expression depends on the predictor b and the (negative) lag k and hence
# could be controlled somehow.

#───────────────────────────────────────────────────────────────────────────────

# ── RANK-ONE CONSTRAINT SYSTEM AND ITS CONSEQUENCES ───────────────────────────

# Due to self-similarity, the PCS constraint system has rank one for the AR(1)
# DGP. While it is possible to impose b %*% (gamma_k - gamma_{k-1}) = beta for a
# single k > 0, doing so with beta > 0 forces the sign of b in a direction that 
# yields CCF(h) < 0 at the forecast horizon h — a negative target correlation, 
# which is unacceptable for a predictor of x_{t+h}, see Tutorial 14 
# (introduction) for further discussion.
#
# As a result, for k >= 0, no degree of freedom remains to reshape the CCF away 
# from the exponentially decaying path imposed by the AR(1) structure. For a 
# non-AR(1) DGP, residual degrees of freedom would exist and could be exploited 
# to at least partially address the PCS constraints, even if the constraint 
# system is not exactly feasible. For demonstration purposes we here consider 
# the hardest AR(1) forecast problem.
#
# Rather than solving the PCS constraint system exactly — which would consume
# too many degrees of freedom when feasible — we adopt the penalized criterion
# from Wildi (2026), Equation 46 (Appendix D), with the solution given by
# Equation 49. The penalty weight nu > 0 (renamed lambda in these tutorials)
# balances target correlation maximization against constraint deviation. It acts
# as a regularization hyperparameter, not as a Lagrange multiplier. Large 
# lambda enforce the constraints to the detriment of the target correlation 
# (dilemma).
#───────────────────────────────────────────────────────────────────────────────

# ── PERTURBATION STRATEGY: EXPANDING THE RANK ─────────────────────────────────

# To overcome the rank-one limitation of the AR(1) DGP, we introduce infinitesimal
# and imperceptible departures from its rigid structure. These perturbations expand
# the constraint system from rank 1 to rank 2 or higher. For simplicity, only
# rank-two perturbations are considered here.
#
# The perturbation magnitude is controlled by a scaling parameter delta > 0.
# Its choice must balance two competing objectives:
#   (i)  Small enough to remain imperceptible and to preserve clean geometric
#        interpretability.
#   (ii) Large enough to avoid numerical precision (near singularity) issues.
#
# In practice, values of delta in the range [1e-5, 1e-3] are recommended.
# Smaller values sharpen the geometric interpretability of the design and can 
# be implemented provided sufficient numerical precision is available.
#───────────────────────────────────────────────────────────────────────────────

# ── GEOMETRY (INTERPRETABILITY) OF THE SOLUTION ───────────────────────────────

# In the perturbated rank-two setting, the PCS solution (Wildi (2026), Equation 
# 49) lives in a two-dimensional subspace spanned by the two leading 
# eigenvectors V1 and V2 of the matrix M, which encodes the PCS constraint 
# system.
#
# Interpretability: when delta is small
#   - V1 aligns closely with gamma_0 (up to sign), capturing the original AR(1)
#     DGP direction.
#   - V2 is orthogonal to gamma_0 and captures the component of the perturbation
#     lying outside the AR(1) subspace. This orthogonality — V2 %*% gamma_0 = 0
#     — implies that V2 fully decouples from the present (see DFP tutorials).
#
# The PCS predictor takes the general form:
#
#   b = lambda1 * V1 + lambda2 * V2,
#
# where lambda1 and lambda2 are determined by the PCS hyperparameters beta and
# lambda (corresponding to beta and nu in Equation 46 of Wildi (2026)). 
#
# Important: not all linear combinations of V1 and V2 are valid PCS solutions.
# The solution is constrained to a specific locus (subspan) within the plane 
# spanned by V1 and V2.
#
# When delta ≈ 0, V1 ∝ gamma_0, and the PCS predictor can be interpreted as a
# linear combination of:
#   - The classical MSE predictor direction gamma_h (proportional to gamma_0 
#     or V1 in the AR(1) case), and
#   - The full decoupling vector V2, obtained from perturbation.
#
# Full decoupling (b ∝ V2) is a theoretically valid strategy for generating
# look-ahead behaviour, but it is often too extreme in practice, potentially
# leading to sign inversion and uninterpretable predictions, see DFP tutorials. 
# The weight lambda2 on V2 — in combination with lambda1, both governed by the 
# PCS hyperparameters lambda and beta — enables controlled, partial decoupling: 
# the predictor departs from gamma_0 toward full decoupling in a graduated and 
# controllable manner to generate look ahead behaviour.
#───────────────────────────────────────────────────────────────────────────────

# ── DFP VS. PERTURBATION-BASED DECOUPLING ─────────────────────────────────────

# When delta ≈ 0, V1 ≈ gamma_0 so that V2 is orthogonal to gamma_0 and induces
# full decoupling from the present. The PCS hyperparameters modulate the predictor
# between two extremes:
#   - No look-ahead:      b ∝ V1 ∝ gamma_0 ∝ gamma_h (replicates the MSE predictor).
#   - Maximum look-ahead: b ∝ V2             (full decoupling from the present).
#
# A key distinction from the original DFP approach:
#   - In original DFP, decoupling arises intrinsically from the DGP structure,
#     provided the data-generating process is sufficiently flexible to support it.
#   - Here, decoupling is exogenously imposed via the perturbation vector, which
#     introduces an artificial direction orthogonal to the AR(1) subspace.
#
# The common element across all perturbation-based PCS predictors in this
# tutorial is V1 ∝ gamma_0 ∝ gamma_h (in the AR(1) case): the shared MSE 
# predictor direction, valid in the limit delta -> 0. The distinguishing element 
# is V2, the fully decoupling direction, which is specific to the perturbation 
# chosen.
#
# Excessively large regularization weight lambda over-weights V2 at the expense 
# of V1, potentially yielding unusable predictors. To see why, note that if 
# b ≈ V2, then:
#
#   0 = b %*% gamma_0 = b %*% gamma_h 
#
# since gamma_0 ∝ gamma_h under the AR(1) structure. That is, the target
# correlation vanishes entirely, i.e., the predictor is unusable.
#───────────────────────────────────────────────────────────────────────────────

# ── PERTURBATION TYPES ────────────────────────────────────────────────────────

# Two main perturbation strategies are considered, both expanding the rank from
# 1 to 2:
#
#   1. Single-lag perturbation:
#      A single coefficient of gamma_0 is slightly modified. This approach tends
#      to be less effective at generating look-ahead behaviour.
#
#   2. Multi-lag perturbation:
#      The perturbation affects the entire lag structure. Two specific forms are
#      considered:
#        (a) gamma_0 + delta * DGP1: DGP1 is an AR(1) with parameter (a1 + delta),
#            yielding a perturbed but structurally similar process to gamma_0.
#        (b) gamma_0 + delta * DGP2: DGP2 is based on an AR(2) specification,
#            introducing a qualitatively different lag structure.
#
# Both multi-lag forms perform roughly equivalently in terms of look-ahead
# behaviour. A formal criterion for selecting an "optimal" perturbation does not
# yet exist, and it remains unclear whether such optimality is even well-defined
# within this framework. Both forms are of independent interest and could
# potentially be combined into a unified, mixed look-ahead PCS specification 
# (forecast combination).
#───────────────────────────────────────────────────────────────────────────────

# ── CAVEAT: NUMERICAL SENSITIVITY FOR SMALL DELTA ─────────────────────────────

# Very small values of delta lead to nearly singular designs, where the predictor
# becomes highly sensitive to small changes in the hyperparameters. It is 
# not strictly necessary to choose delta very small in order to induce look-ahead
# behaviour in the PCS predictor.
#
# Nevertheless, we deliberately work with small delta in this tutorial in order to
# preserve the geometric interpretability of the solution:
#   - V1 remains closely aligned with gamma_0 (equivalently, gamma_h), retaining
#     its interpretation as the classical MSE predictor direction.
#   - V2 retains its interpretation as the full decoupling direction, orthogonal
#     to gamma_0 and determined by the specific perturbation chosen.
#
# As delta increases, these geometric correspondences gradually loosen, and the
# clean decomposition of the PCS predictor into an MSE component (V1) and a full
# decoupling component (V2) becomes less transparent. However, the design is less 
# sensitive to small changes in the hyperparameters which eventually facilitates 
# the search for interesting look ahead alternatives by making designs more 
# regular.
#───────────────────────────────────────────────────────────────────────────────

# ── EXTENSION TO DFP ──────────────────────────────────────────────────────────
#
# The perturbation framework developed above for PCS extends naturally to the DFP
# approach. Like PCS, DFP operates by manipulating the CCF of the predictor, but
# its constraints are generally less restrictive — particularly when compared to
# Type I PCS at longer horizons (h > 1).
#
# Consequently, the rank-expanding perturbation strategy in this tutorial can be 
# applied to DFP in exactly the same way: by introducing a small perturbation 
# delta to the AR(1) DGP (or any other `difficult' DGP), the rank-one constraint 
# system is expanded to rank two, unlocking the decoupling direction V2 and 
# enabling controlled look-ahead behaviour under the DFP criterion as well.
# 
#───────────────────────────────────────────────────────────────────────────────

# ── EXAMPLES OVERVIEW ─────────────────────────────────────────────────────────
#
# Example 1 — Impossibility
#             Demonstrates that, for a pure AR(1) DGP, the PCS constraint system
#             is impossible: the CCF peak cannot be shifted to k=h>0 while
#             maintaining a positive target correlation. 
#
# Example 2 — Single-lag perturbation.
#             The first lag coefficient of gamma_0 is slightly modified. This is
#             analogous to applying PCS to an ARMA(1,1) DGP, where the MA(1)
#             parameter b1 perturbs only the lag-0 weight of gamma_0 (see, e.g., 
#             Tutorial 13).
#
# Example 3 — Multi-lag AR(1) perturbation.
#             The perturbation is based on a second AR(1) process with a slightly
#             different parameter (a1 + delta), affecting the entire lag structure.
#             Although both the original and perturbed processes are acyclical,
#             the resulting PCS predictor adopts a nearly cyclical coefficient
#             profile, reflecting the influence of the full decoupling vector V2.
#
# Example 4 — Multi-lag AR(2) perturbation.
#             The perturbation is based on an AR(2) specification, introducing
#             a qualitatively different lag structure relative to the original
#             AR(1) DGP. As in Example 3, the decoupling direction V2 is
#             determined by the perturbation and shapes the look-ahead profile
#             of the PCS predictor accordingly. However, since the AR(2)
#             perturbation introduces a more severe misspecification than its
#             AR(1) counterpart, the regularization weight lambda should be
#             kept small to moderate: excessively large lambda assigns
#             disproportionate weight to the misspecified constraints, compared 
#             to the proper forecast target, hence risking unusable predictors.
#
# Example 5 — Parameter Interplay:
#             Analyses and formalites the complex entanglement and interaction 
#             of the hyperparameters:
#               - lambda : regularisation weight,
#               - beta   : CCF slope appearing in the constraints,
#               - delta  : size of the perturbation.

################################################################################
# Main Take-Aways:
#
# ─────────────────────────────────────────────────────────────────────────────
# I) Inducing Look-Ahead Behaviour in the Context of Impossible Problems
# ─────────────────────────────────────────────────────────────────────────────
#
# A forecast problem is called impossible when the CCF peak cannot be relocated
# to the forecast horizon k = h while remaining of positive height. Even so,
# the problem generally remains amenable to look-ahead behaviour: predictors
# exist that are LEFT-SHIFTED (leading) relative to the MSE benchmark while simultaneously
# maximising tracking accuracy. The CCF peak may be fixed at the origin (k = 0),
# yet effective anticipatory behaviour can still be recovered by reshaping the
# LEFT TAIL (negative lags) of the CCF. This tutorial demonstrates the 
# effectiveness of perturbation approaches in achieving this dual objective.

#
# ─────────────────────────────────────────────────────────────────────────────
# II) Left Tail of the CCF
# ─────────────────────────────────────────────────────────────────────────────
#
# The AR(1) structure of the hardest forecast problem examined in this
# tutorial makes it impossible to shift the CCF peak to the right. More
# precisely, no linear predictor can alter the exponentially decaying profile
# of the CCF at positive lags -- confirming that the pure AR(1) represents
# the most challenging look-ahead problem.
#
# ─────────────────────────────────────────────────────────────────────────────
# How to Achieve Look-Ahead Behaviour When the CCF Peak Cannot Be Shifted?
# ─────────────────────────────────────────────────────────────────────────────
#
# Although the right tail of the CCF is entirely determined by the AR(1)
# structure and is therefore immutable, the left tail remains accessible to
# manipulation through the choice of predictor. The following observation,
# carried over from Exercise 3.5, elaborates on this point:
#
#   - Only the left tail of the CCF is amenable to modification. While the
#     MSE predictor yields a symmetric CCF, the PCS predictor becomes
#     progressively more asymmetric as beta increases. Effective look-ahead
#     behaviour is thus achieved by skewing the CCF rightward -- that is, by
#     suppressing (or inverting) the contribution of negative lags.
#
# By introducing suitable perturbations to the original DGP -- which may be
# made arbitrarily small -- the left tail can be shaped to induce look-ahead
# behaviour. This offers a principled resolution to an otherwise infeasible
# problem: rather than attempting to shift the CCF peak rightward, one instead
# recovers effective lead behaviour by redistributing CCF mass away from
# negative lags.
#
# Perturbation-based approaches are, however, just one instance of a broader
# strategy: modifying the left tail of the CCF is a general and flexible
# pathway to look-ahead behaviour whenever the right tail is structurally
# immutable or too rigid to admit meaningful modification. 


# ─────────────────────────────────────────────────────────────────────────────
# III) Severity of Perturbation Misspecification
# ─────────────────────────────────────────────────────────────────────────────
#
# In general, a perturbation introduces a degree of model misspecification:
# without it, the constraint space determined by the DGP would remain of
# insufficient rank to admit meaningful look-ahead behaviour. The severity
# of this misspecification, however, can vary considerably across designs.
#
# Exercise 3 illustrates a mild case: the perturbation modifies the DGP via
# a closely related model assumption (also AR(1) in structure), keeping the
# misspecification minimal. Exercise 4 demonstrates a moderately more severe
# case, in which an AR(2)-type perturbation is applied to the original AR(1)
# DGP. While the AR(2) is structurally close to the AR(1), the increased
# distance between the two model classes introduces a non-negligible degree
# of misspecification relative to Exercise 3, justifying smaller regularization 
# weights (the hyperparameter lambda in this tutorial or nu in Wildi 
# (2026), Appendix D).
#
# When the misspecification of the constraint equations is substantial, it is
# advisable to limit the regularization weight to small or moderate values,
# so that the target correlation retains meaningful influence in the
# optimisation. If the regularization is too strong, excessive weight is
# assigned to the misspecified perturbed constraints, potentially yielding
# unusable predictors. Exercise 4 illustrates these points.

################################################################################




# ═════════════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ═════════════════════════════════════════════════════════════════════════════


# ── INITIALISATION ────────────────────────────────────────────────────────────

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
# EXERCISE 1: RANK-ONE CONSTRAINT SYSTEM
# ════════════════════════════════════════════════════════════════════

#───────────────────────────────────────────────────────────────────────────────
# 1.1 AR(1) DGP
#───────────────────────────────────────────────────────────────────────────────

L  <- 50    # Filter length: number of MA coefficients retained.
a1 <- 0.9   # AR(1) parameter.

# Compute the Wold (MA-infinity) coefficients of the AR(1) process.
xi <- c(1, ARMAtoMA(ar = a1, ma = 0, lag.max = 1000))

# Visualise the Wold decomposition: coefficients decay geometrically at rate a1.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: geometrically decaying impulse response")

# Visualise the theoretical ACF: also decays geometrically at rate a1.
ts.plot(ARMAacf(ar = a1, lag.max = L),
        main = "ACF", ylab = "", xlab = "Lag")

# The ACF satisfies ACF(k+1) = a1 * ACF(k) for all k >= 0.
# Consequently, the constraint system has rank one: gamma_h is proportional
# to gamma_{h+k} for any h, k >= 0 (self-similarity).


#───────────────────────────────────────────────────────────────────────────────
# 1.2 PCS Setup
#───────────────────────────────────────────────────────────────────────────────

# Forecast horizon.
h <- 12

# Target: the original AR(1) Wold decomposition.
gamma_pcs <- xi
# Nowcast
gamma0 <- xi[1:L]
# MSE forecast
gammah <- xi[h+1:L]
# Identity forecast: this is faster than gammah but noisier:
# It is used as an additional benchmark.
gamma_I<-c(1,rep(0,L-1))



# Constrained lag set:
# Type I PCS imposes a non-negative slope at every lag in Delta, enforcing a
# monotonically increasing CCF (when beta > 0 and the problem is feasible) over
# the full interval {0, ..., h}. This is the most restrictive of the three PCS
# types (I, II, and III).
Delta <- 1:h

# Regularisation weight (penalty on constraint deviation): strong regularisation.
lambda <- 5000000



# ─────────────────────────────────────────────────────────────────────
# 1.3 AR(1) Perturbation
# ─────────────────────────────────────────────────────────────────────

# Construct the MSE predictor coefficient vectors gamma_i used to form the
# PCS constraint differences delta_i = gamma_i - gamma_{i-1}.
gamma_all <- xi

# Build the shifted predictor matrix 'gammah_mat':
# Each row contains the normalised gamma_all = xi coefficients shifted by a
# specific lead drawn from Delta. We begin at Delta[1] - 1 because the first
# constraint difference requires gamma_{Delta[1]-1}.
gammah_mat <- gamma_all[Delta[1] - 1 + 1:L] / sqrt(sum(gamma_all^2))
if (length(Delta) > 0) {
  for (i in 1:length(Delta)) {
    gammah_mat <- rbind(gammah_mat,
                        gamma_all[Delta[i] + 1:L] / sqrt(sum(gamma_all^2)))
  }
}

# Notes on normalisation:
#
# 1. The divisor sqrt(sum(gamma_all^2)) equals the standard deviation of the
#    process (up to the innovation variance). Normalising by this quantity is
#    consistent with interpreting the PCS constraints as conditions on
#    correlations rather than raw covariances.
#
# 2. Strictly speaking, this normalisation is not required for the PCS
#    optimisation to be well-defined; it is a scaling convention that simplifies
#    the interpretation of the constraint parameter beta.
#
# 3. Note that b is not normalised by 1/sqrt(sum(b^2)), so beta does not
#    represent a pure correlation in the strict sense. 


# Perturb the AR(1) parameter: a1_perturbate = a1 + delta.
# The resulting Wold decomposition xi_a1_perturbate differs from xi at every lag 
# except zero, providing a full-lag perturbation direction.
delta          <- 0.001
a1_perturbate  <- a1 + delta
xi_a1_perturbate <- c(1, ARMAtoMA(ar = a1_perturbate, ma = 0, lag.max = 1000))
gamma_all_a1_perturbate <- xi_a1_perturbate

# Visualise the difference between the original and perturbed Wold coefficients.
par(mfrow=c(1,1))
ts.plot(xi - xi_a1_perturbate,
        main = "Difference: original vs. perturbed Wold coefficients")

# Construct the perturbed constraint matrix by replacing the first row of
# gammah_mat (corresponding to gamma_{Delta[1]-1}) with its perturbed counterpart.
gammah_mat_perturbate <- gammah_mat

gammah_mat_perturbate[1, ] <-
  (gammah_mat[1, ] + delta * gamma_all_a1_perturbate[1:L]) /
  sqrt(sum(gamma_all_a1_perturbate^2))



# ─────────────────────────────────────────────────────────────────────
# 1.4 Run PCS
# ─────────────────────────────────────────────────────────────────────

# PCS_perturbation_func() automatically returns a grid of beta values centred on 
# the tipping point of beta — where the sensitivity of the PCS solution with 
# respect to beta is highest. 
#
# Note: the automatic grid generated by PCS_perturbation_func is independent of 
# beta: any value can be supplied:
beta<-0.

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)

# Grid of beta values 
beta_vec_automatic<-PCS_obj$beta_vec
# Add some intermediary values for better resolution:
beta_vec<-c(beta_vec_automatic[1:10],1.86e-07,1.88e-07,1.90e-07,beta_vec_automatic[11:length(beta_vec_automatic)])
M         <- PCS_obj$M
V<-eigen(M)$vectors

# Note:
# The asymptotic behaviour of the grid tails — i.e. as |beta| -> infinity —
# depends on the interplay between beta, lambda, and delta (see Exercise 5.3 VIII
# and Exercise 2 above). Depending on the particular combination of these
# hyperparameters, the PCS predictor b aligns with one of the following
# directions as |beta| increases:
#   - V1 alone,
#   - a mixture of V1 and V2, or
#   - V2 alone.
# Here, the asymptotes ( |beta| -> infinity) correspond to +/-V2 since lambda is 
# very large (see exercise 5.3 VIII, case [c]).


# PCS Type I constraint:
Delta <- 1:h

b_mat <- NULL

for (i in 1:length(beta_vec)) {
  beta    <- beta_vec[i]
  PCS_obj <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                   gammah_mat_perturbate)
  b     <- PCS_obj$b
  b_mat <- cbind(b_mat, b)
}

# Prepend the classical MSE predictor (gamma_0) as a reference.
filter_mat           <- cbind(gamma0, b_mat)
# Beta is scaled by lambda in the column names: to allow readability in plots.
colnames(filter_mat) <- c("MSE",
                          paste("lambda =", round(lambda, 2),
                                ", beta*lambda =", round(beta_vec*lambda, 8)))

# ─────────────────────────────────────────────────────────────────────
# 1.5 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors: Full-Lag AR(1) Perturbation ─────────────────
#

# ─────────────────────────────────────────────────────────────────────
# 1.6 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

# Simulate a long realisation of the AR(1) DGP for filter evaluation.
len     <- 10000
set.seed(534)
x_filt  <- rnorm(len)

# Apply each predictor filter to the simulated series.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


# ── Full-range overview: all predictor outputs ─────────────────────────────────
# Display a broad sub-sample to compare the behaviour of all predictors.
# Observations:
#   - Smaller beta values produce lagging predictors (relative to the MSE).
#   - Larger beta values produce increasingly leading predictors.
#   - As the degree of lead increases, predictors tend toward sign inversion,
#     reflecting phase inversion.
#   - We select the leading predictors as well as the MSE benchmark predictor.
#   - All series are standardized to simplify visual inspection.
select_pcs<-11:ncol(y_out_mat)
select_vec<-c(1,select_pcs)

# Longer sub-sample
anf<-100
enf<-500

mplot<-scale(y_out_mat[anf:enf,select_vec])
colnames(mplot)<-colnames(y_out_mat)[select_vec]

coli<-c("black",colo[select_pcs])

par(mfrow = c(1, 1))

ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty = c(2, 1, rep(1, ncol(mplot) - 1)),
        lwd = c(2, rep(1, ncol(mplot) - 1)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── Narrow sub-sample: magnifying the look-ahead effect ───────────────────────
# Zoom into a shorter window to highlight the look-ahead behaviour of selected
# predictors, avoiding those with pronounced sign inversion.
#
# Note: the look-ahead effect operates primarily on longer swings in the series.
# Short-term random spikes are inherently unpredictable. This long-swing
# look-ahead property may be particularly relevant in business cycle analysis,
# where economically significant episodes — such as recessions — are typically
# characterised by sustained negative swings rather than isolated shocks.
anf<-280
enf<-400

mplot<-scale(y_out_mat[anf:enf,select_vec])
colnames(mplot)<-colnames(y_out_mat)[select_vec]

coli<-c("black",colo[select_pcs])

par(mfrow = c(1, 1))

# Full-sample overview of all predictor outputs.
ts.plot(mplot,
        main = "Predictor Outputs", col = coli, xlab = "", ylab = "",
        lty = c(2, 1, rep(1, ncol(mplot) - 1)),
        lwd = c(2, rep(1, ncol(mplot) - 1)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i], col = coli[i], line = -i)


# ── Empirical CCF:  ───────────────────────────────────────────────────────────
# Note: the true (expected) CCF was shown in exercise 3.4. In contrast we here 
# compute the empirical CCF based on the sample correlations of the filtered series. 

# Compute the empirical cross-correlation function (CCF) between the MSE
# predictor output (column 1) and the selected (leading) PCS predictor output.
#
# Key observations:
#   - As beta increases, the empirical CCF becomes increasingly right-skewed,
#     reflecting a growing lead of the PCS predictor relative to the MSE predictor.
#
#   - The right tail of the CCF (lag > 0) is immutable and always follows the 
#     AR(1) decay:
#     b %*% gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
#     consequence of the Yule-Walker equations and holds for any linear predictor b.
#     No non-zero predictor can alter this decay shape.
#
#   - Consequently, shifting the CCF peak strictly to the right of lag 0 is
#     impossible under the AR(1) DGP (see Exercise 1).
#
#   - Only the left tail of the CCF (lags < 0) is amenable to modification in 
#     the AR1) case. 
#     Whereas the MSE predictor (first panel) yields a symmetric CCF, the PCS 
#     predictor becomes progressively more asymmetric as beta increases. 
#     Effective look-ahead behaviour (illustrated in the predictor plot above) 
#     is thus achieved by skewing the CCF rightward — that is, by down-weighting 
#     or inverting the contribution of negative lags.

mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[5])
ccf(mplot_ccf[,1],mplot_ccf[,6],main=colnames(mplot_ccf)[6])
ccf(mplot_ccf[,1],mplot_ccf[,7],main=colnames(mplot_ccf)[7])





# ─────────────────────────────────────────────────────────────────────
# 2. RSC (Right-Skewing CCF): Relying on the DFP Framework
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 2.1 Set-Up RSC 
# ─────────────────────────────────────────────────────────────────────

# Lag support: negative lags from k=0 to k=-l0
l0<-10
# Integrator
Sigma<-matrix(rep(0,L^2),ncol=L)
for (i in 1:l0)
  Sigma[i,1:i]<-1

gamma_constraint<-(Sigma%*%gamma0)



par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector encoding the CCF slope condition at lag h")
abline(h = 0)

# Technical note:
# - The constraint vector gamma_constraint exhibits a strong, albeit partly
#   unintended, link to Tutorial 12, Exercise 1.4, and in particular to the
#   convolved target gamma used in this exercise.
# - The main idea in Tutorial 12 was to expand the rank of the constraint system
#   by applying an equally weighted MA(12) (i.e., a sum or integrator of length 12)
#   to the original ARMA(1,1) process.
# - In contrast, here we rely on the integrator Sigma to affect the left tail of
#   the cross-covariance function (CCF), thereby producing a right-skewed CCF.
# - While the concepts and ideas are reminiscent, and to some extent intriguing, 
#   there is also a notable difference:
#   in Tutorial 12, Exercise 1.4, the target is the convolved (integrated from 
#   original monthly to yearly-growth) DGP, whereas here the target remains the 
#   original (un-convolved) AR(1) process.





# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor gammah.
# Purpose: mse_coup serves as a natural upper bound for the DFP constraint,
# i.e., any effective decoupling should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, each strictly smaller than mse_coup.
# Smaller (more negative) values enforce a progressively stronger right skewing 
# of  the CCF.  
# Note: since the predictors are not normalized (||b|| != 1), the
# rule is not exact — alpha0 < mse_coup does not guarantee stronger decoupling
# of b from gamma_constraint — but it serves as a useful practical proxy.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.5,-1,-2,-4)

# Display alpha0_vec: the last (negative) entry indicates that the DFP
# constraint enforces stronger decoupling than the MSE predictor gammah,
# which should shift the CCF peak to the right from k = 0 to k = h = 1.
alpha0_vec


# ─────────────────────────────────────────────────────────────────────
# 2.2 Decoupling over the alpha0-Grid
# ─────────────────────────────────────────────────────────────────────
# For each alpha0 in alpha0_vec, compute the MSE-optimal PCS predictor via
# Proposition 1 (Wildi 2026):
#
#   b = gammah + lambda * gamma_constraint,
#   lambda = (alpha0 - gamma_constraint' * gammah) / (gamma_constraint' * gamma_constraint)
#
# This closed-form solution minimises the MSE subject to the modified
# decoupling constraint b %*% gamma_constraint = alpha0.

b_mat       <- NULL    # filter coefficients, one column per alpha0
cor_vec_mat <- NULL    # full CCF vectors, one column per alpha0

# CCF values at lags 0 and h for each alpha0 (for tabular summary)
cor_vec_1 <- matrix(ncol = 2, nrow = length(alpha0_vec))

# Number of leads on either side of lag 0 to include in the CCF (does not affect 
# optimization)
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
# 2.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# ── Check 1: Decoupling constraint ──────

# Verification: the constraint b %*% gamma_constraint = alpha0 should hold
# exactly for every column of b_mat (residuals should be numerically zero).
t(b_mat) %*% gamma_constraint - alpha0_vec

# ── Check 2: sign / orientation preservation ──────────────────────────────
# A strictly positive sum of filter coefficients confirms that none of the
# PCS filters inverts the direction of a trend or level shift in the data.
apply(b_mat, 2, sum)

# CHECK 3 — Positive Target Covariance
t(b_mat)%*%gammah


# Collect all filters (nowcast, MSE, and PCS variants) into a single matrix
filter_mat <- cbind(gamma0, gammah, gamma_I, b_mat)
colnames(filter_mat) <- c("Nowcast", paste("MSE(",h,")",sep=""),"Identity",
                          paste0("RSC ", round(alpha0_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 2.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo  <- c("black","green","darkgreen", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
# Truncate to the first q+1 lags where the MA process has support.
mplot <- filter_mat

plot(mplot[, 1], main = "Filter coefficients: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), 0.4))
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
max_lag<-l0
ccf_mat<-NULL
for (i in 1:ncol(filter_mat))
  ccf_mat<-cbind(ccf_mat,compute_acf_at_lags_zero_delta_func(
    max_lag, h, filter_mat[,i], xi)$cor_vec)
mplot   <- ccf_mat

plot(mplot[, 1], main = "Population CCFs: MSE and PCS variants",
     axes = FALSE, type = "l", xlab = "Lag", ylab = "",
     col = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))
for (i in 2:ncol(mplot))
  lines(mplot[, i], col = colo[i])
abline(h = 0)
abline(v = max_lag + 1 + h, lty = 2)   # lag h
abline(v = max_lag + 1 , lty = 1)   # lag h
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
# 2.5 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 2.5.1 Forecast Comparison ────────────────────────────────────────


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


# ── 2.5.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(3, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste("CCF: MSE(",h,"): Peak at lag k = 0 (no peak-shift)",sep=""))

k<-4
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste(colnames(y_out_mat)[k],sep=""))
k<-7
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste(colnames(y_out_mat)[k],sep=""))
k<-9
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste(colnames(y_out_mat)[k],sep=""))
k<-11
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[,k]),
    lag.max = 10, plot = TRUE,
    main = paste(colnames(y_out_mat)[k],sep=""))
k<-13
ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, k]),
    lag.max = 10, plot = TRUE,
    main = paste(colnames(y_out_mat)[k],sep=""))


targeth<-c(y_out_mat[(1+h):len,1],rep(NA,h))
# Remove NAs
y_outh<-na.exclude(cbind(targeth,y_out_mat))
y_out<-y_outh[,2:ncol(y_outh)]
colnames(y_out)<-colnames(y_out_mat)
target<-y_outh[,1]

target_cor<-NULL
for (i in 1:ncol(y_out_mat))
  target_cor<-c(target_cor,cor(target,y_out[,i]))
names(target_cor)<-colnames(y_out_mat)
target_cor

smooth<-NULL
for (i in 1:ncol(y_out_mat))
  smooth<-c(smooth,mean(diff(diff(scale(y_out[,i])))^2))
names(smooth)<-colnames(y_out_mat)
smooth

max_lead   <- 12

for (i in 2:ncol(y_out))
{
  filter_mat <- cbind(y_out[,i],y_out[,"Identity"])
  colnames(filter_mat)<-c(colnames(y_out)[i],"Identity")
  tau<-compute_min_tau_func(filter_mat, max_lead)
}





