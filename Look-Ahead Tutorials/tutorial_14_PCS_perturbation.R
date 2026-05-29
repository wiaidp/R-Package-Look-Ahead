# ══════════════════════════════════════════════════════════════════════════════
# TUTORIAL 14 — HANDLING IMPOSSIBILITY THROUHG PERTURBATION: 
#               APPLICATION TO THE HARDEST LOOK AHEAD FORECAST PROBLEM
# ══════════════════════════════════════════════════════════════════════════════


# This tutorial is still under construction (26-May-2026).


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
#   CCF(k) = (b[-k+(1:L)] %*% gamma_0) / (||b[-k+(1:L)]|| * ||gamma_0||) 
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
# immutable or too rigid to admit meaningful modification. However, we here do 
# not explore alternative directions for affecting the left CCF tail. 


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

# Constrained lag set:
# Type I PCS imposes a non-negative slope at every lag in Delta, enforcing a
# monotonically increasing CCF (when beta > 0 and the problem is feasible) over
# the full interval {0, ..., h}. This is the most restrictive of the three PCS
# types (I, II, and III).
Delta <- 1:h

# Regularisation weight (penalty on constraint deviation): strong regularisation.
lambda <- 1000000

# Constraint slope parameter (negative here to probe the impossible regime).
beta <- -0.0001

b_mat <- NULL

# Compute the Type I PCS predictor.
PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol


#───────────────────────────────────────────────────────────────────────────────
# 1.3 Linear Algebra of the Rank-One System
#───────────────────────────────────────────────────────────────────────────────
# The closed-form PCS solution is: b <- solve(M) %*% gamma_sol, see equation 49, Wildi (2026).
#   - M depends on lambda (nu in the cited literature) but not on beta.
#   - gamma_sol depends on both lambda and beta.

# Verify the closed-form solution: the residual should vanish.
max(abs(b - solve(M) %*% gamma_sol))

# M is symmetric and admits an eigendecomposition M = V %*% diag(d) %*% t(V).
eigenM <- eigen(M)
V      <- eigenM$vectors

# Verify the eigendecomposition: residual should vanish.
max(abs(M - V %*% diag(eigenM$values) %*% t(V)))

# Verify the matrix inverse via eigendecomposition: residual should vanish.
max(abs(solve(M) - V %*% diag(1 / eigenM$values) %*% t(V)))

# Since solve(M) = V %*% diag(1/d) %*% t(V), the solution b = solve(M) %*% gamma_sol
# can be written as b = V %*% g, where g = diag(1/d) %*% t(V) %*% gamma_sol.

# gamma_sol is a weighted linear combination of gamma_h (the proper target) and 
# the PCS constraints.
# In the AR(1) case, gamma_h and all constraints are linearly dependent (rank one),
# so gamma_sol is itself AR(1) with geometrically decaying coefficients.
ts.plot(gamma_sol, main = "gamma_sol: AR(1) exponential decay")

# Confirm geometric decay: consecutive ratios should equal a1.
gamma_sol[2:L] / gamma_sol[1:(L - 1)]

# M = I + lambda * N, where 
# N = sum_{k in Delta} (gamma_k - gamma_{k-1}) %*% t(gamma_k - gamma_{k-1}), 
# see Wildi (2026) Appendix D.
# Verify: residual should vanish.
max(abs(M - diag(rep(1, L)) - lambda * N))
# Note:  (gamma_k - gamma_{k-1}) represents the PCS constraint at lag k: we want 
# b %*% (gamma_k - gamma_{k-1}) = beta.

# Since (gamma_k - gamma_{k-1}) is AR(1) for all k, the L x L matrix N has rank one.
eigenN <- eigen(N)

# Confirm rank one: only one eigenvalue exceeds the numerical threshold 10^{-10}.
which(abs(eigenN$values) > 1e-10)

# Visualise the eigenvector corresponding to the single non-vanishing eigenvalue.
ts.plot(N[, 1], main = "Leading eigenvector of N (rank-one structure)")

# Key spectral relationships between M and N:
#   - Eigenvalues of M = I + lambda * N are: 1 + lambda * n_i, where n_i are eigenvalues of N.
#   - Eigenvalues of M^{-1} are: 1 / (1 + lambda * n_i).
#   - M and N share the same eigenvectors.
#   - rank(N) = 1, rank(M) = L.
#
# Note: the eigenvector orderings of M and N may differ, so the following need not vanish.
max(abs(V - eigenN$vectors))

# Inspect eigenvalues of M and its leading eigenvector. 
# Note: the sign of V is arbitrary since the diagonalization is quadratic in V.
eigenM$values
ts.plot(V[,1], main = "Leading eigenvector of M")

# Since V is orthogonal and gamma_sol is proportional to V1:=V1, all projections
# t(V[,k]) %*% gamma_sol vanish for k > 1.

# Verify: the projection onto V[,k] for k > 1 should vanish.
k <- 2
V[, k] %*% gamma_sol        # Should be (near) zero for k > 1.
t(V)[k, ] %*% gamma_sol     # Equivalent formulation.

# Full projection vector: only the first element should be non-zero.
t(V) %*% gamma_sol

# Intermediate vector g = diag(1/d) %*% t(V) %*% gamma_sol:
# only its first element is non-zero, so b = V %*% g = g[1] * V1.
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Verify: V %*% g equals g[1] * V1 up to numerical precision.
max(abs(V %*% g - g[1] * V[,1]))

# Conclusion: b = solve(M) %*% gamma_sol is proportional to V1, the leading
# eigenvector of M, which itself has an AR(1) (geometrically decaying) profile.
b <- V %*% diag(1 / eigenM$values) %*% t(V) %*% gamma_sol

ts.plot(b, main = "PCS predictor: AR(1) profile")

# Confirm geometric decay: consecutive ratios should equal a1.
b[2:L] / b[1:(L - 1)]

# This result holds for any value of lambda > 0 and any beta < 0.
#
# However, for beta > 0 and lambda sufficiently large, the PCS constraint forces
# an increasing CCF path by imposing:
#
#   b %*% (gamma_k - gamma_{k-1}) = beta > 0,  for k = 1, ..., h.
#
# In the AR(1) case, satisfying this constraint with beta > 0 requires the sign
# of b to flip relative to the beta < 0 case. To verify this, re-run the code
# above with beta = 0.001 instead of beta = -0.0001.
#
# The consequence is that while the CCF is indeed monotonically increasing from
# k=0 to k=h (as required by Type I PCS), all CCF values — including the peak
# CCF(h) — are negative. A predictor that correlates negatively with its target
# x_{t+h} is clearly unacceptable, regardless of where the CCF peak is located.
#
# This confirms the impossibility result: for the AR(1) DGP, the PCS constraint
# system cannot simultaneously achieve a positive CCF(h) and a peak at k=h.
# The rank-one structure of the DGP leaves no degree of freedom to escape this
# trade-off.

#───────────────────────────────────────────────────────────────────────────────
# 1.4 Impossibility
#───────────────────────────────────────────────────────────────────────────────
# It is impossible to shift the CCF peak to k > 0 for the AR(1) DGP.
#
# The self-similarity of the AR(1) process (rank-one constraint system) implies
# that gamma_sol is proportional to V1, the leading eigenvector of M. As a
# result, b = solve(M) %*% gamma_sol is proportional to V1, which is itself
# AR(1). This holds irrespective of the choice of lambda > 0 and beta can only 
# trigger the sign of b. 
#
# Consequently, CCF(k) = a1^k * CCF(0) for any predictor b derived from the PCS
# criterion: the CCF always peaks at k=0 and decays geometrically, regardless of
# the hyperparameter settings.

# However, this result applies to positive lags k > 0 only.




# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: INCREASING THE RANK — A PERTURBATION-BASED APPROACH
# ════════════════════════════════════════════════════════════════════
# A single perturbation of magnitude delta is introduced at lag 0 of the
# Wold decomposition. Provided gamma_0 enters the PCS constraint system,
# this expands the rank of the constraint system from 1 to 2 (otherwise the rank 
# is stuck at one).

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


#───────────────────────────────────────────────────────────────────────────────
# 2.1 Single-Lag Perturbation
#───────────────────────────────────────────────────────────────────────────────

# Forecast horizon.
h <- 12

# Construct the perturbation vector: delta times the first unit vector e_1,
# so that only the lag-0 coefficient of xi is modified.
e1                <- c(1, rep(0, length(xi) - 1))
delta             <- 0.0001
perturbation_vec  <- delta * e1
xi_perturbated    <- xi + perturbation_vec

# Perturbed target vector for PCS.
gamma_pcs_perturbated <- xi_perturbated

# Extract the perturbed h-step predictor coefficient vectors.
gamma0_perturbated <- xi_perturbated[1:L]
gamma1_perturbated <- xi_perturbated[1 + 1:L]
gamma2_perturbated <- xi_perturbated[2 + 1:L]

# The perturbation affects only lag 0 of gamma0
gamma0_perturbated[2:L]/gamma0_perturbated[1:(L-1)]
# gamma1 or any gammah, h>0, is not affected:
gamma1_perturbated[2:L] / gamma1_perturbated[1:(L - 1)]

# The first two PCS constraints involve the differences:
#   delta_1 = gamma_0 - gamma_1   (affected by the perturbation at lag 0)
#   delta_2 = gamma_1 - gamma_2   (unaffected; both are pure AR(1))
delta1 <- gamma0_perturbated - gamma1_perturbated
delta2 <- gamma1_perturbated - gamma2_perturbated

# For k > 2, delta_k = gamma_{k-1} - gamma_k is proportional to delta_{k-1},
# since the perturbation affects only the lag-0 coefficient of xi.

# delta1 lies in the span of {gamma_0, perturbation_vec}: both components
# are statistically significant and the residual vanishes.
gamma0 <- xi[1:L]
summary(lm(delta1 ~ gamma0 + perturbation_vec[1:L] - 1))

# delta_2 lies in the span of {gamma_0} alone: the perturbation has no
# effect on delta_2 or any delta_k, k>1.
summary(lm(delta2 ~ gamma0 - 1))

# Type I PCS: constrain all lags from k=1 to k=h (most restrictive type).
Delta <- 1:h

# Regularisation weight: very strong regularization.
lambda <- 10000000

# Constraint slope parameter.
beta <- -0.0001

# Compute the Type I PCS predictor for the perturbed system.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol
beta_vec  <- PCS_obj$beta_vec

#───────────────────────────────────────────────────────────────────────────────
# 2.2 Rank-Two Structure
#───────────────────────────────────────────────────────────────────────────────
# The closed-form PCS solution is: b = solve(M) %*% gamma_sol.
# In contrast to Exercise 1, the perturbed gamma_sol is no longer exactly AR(1).

# Confirm: the first ratio deviates from exact AR(1):
gamma_sol[2:L] / gamma_sol[1:(L - 1)]

# gamma_sol lies in the span of {gamma_0, perturbation_vec}: both components
# are significant and the residual vanishes.
summary(lm(gamma_sol ~ perturbation_vec[1:L] + gamma0 - 1))

# Eigendecomposition of M and N.
eigenM <- eigen(M)
V      <- eigenM$vectors
eigenN <- eigen(N)

# Confirm rank two: exactly two eigenvalues of N exceed the threshold 10^{-10}.
which(abs(eigenN$values) > 1e-10)

# The column space of N is now two-dimensional, spanned by delta_1 and delta_2.

# Visualise the two leading eigenvectors of N.
par(mfrow = c(1, 1))
ts.plot(eigenN$vectors[, 1:2],
        main = "Leading eigenvectors of N (rank-two structure)",
        lty  = 1:2)
# When delta is small, V1 is aligned with gamma0 (up to sign) and V2 is the 
# orthogonal full decoupling vector, i.e. V2 %*% gamma0 = 0 (up to a small 
# deviation depending on delta).


# Full projection of gamma_sol onto the eigenbasis of M:
# only the first two elements should be non-negligible.
# This should be contrasted with exercise 1, where only the first element 
# did not vanish.
t(V) %*% gamma_sol

# Intermediate vector g = diag(1/d) %*% t(V) %*% gamma_sol:
# only its first two elements are non-zero.
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Consequently, b = V %*% g = g[1]*V1 + g[2]*V2.
# Verify: the residual should vanish.
max(abs(V %*% g - g[1] * V[,1] - g[2] * V[,2]))

# The PCS predictor b is a linear combination of V1 and V2, or
# equivalently, of gamma_0 and perturbation_vec.
b <- V %*% diag(1 / eigenM$values) %*% t(V) %*% gamma_sol

ts.plot(b, main = paste("PCS Predictor, lambda=",lambda,", beta=",beta," : 
                        No longer AR(1)",sep=""))

# Confirm departure from AR(1): lag 0 is affected by the perturbation.
b[2:L] / b[1:(L - 1)]

# Verify the decomposition of b into {gamma_0, perturbation_vec}: perfect fit.
summary(lm(b ~ gamma0 + perturbation_vec[1:L] - 1))


# We now explore the Rank-Two System under various regularization settings.

#───────────────────────────────────────────────────────────────────────────────
# 2.3  VERY STRONG Regularisation
#───────────────────────────────────────────────────────────────────────────────
# The PCS predictor lies in the span of V1 and V2, with coefficients
# governed by lambda, beta and delta. 
#
# - The behaviour of the regularised perturbated solution depends on the 
#   interplay between the hyperparameters: 
#     - the regularisation weight lambda, 
#     - the perturbation size delta, and 
#     - the slope beta; 
#   see Exercise 5 (more precisely Exercise 5.3, VIII) for theoretical background 
#   on the various regularisation regimes.
#
# - Here we set lambda = 1/delta^2, corresponding to a very strong regularisation
#   regime, see  case [c] in Exercise 5.3, VIII. In principle, any lambda of this 
#   very large order or larger (assuming delta is small) emphasises the perturbation and the full decoupling
#   direction V2:=V2 (which is determined by and depends on the perturbation). 
#   However, special values of beta allow for alternative combinations of V1 
#   and V2, including the special case b = O(delta^2) * V1 (subject to very strong 
#   shrinkage O(delta^2)), see  case [c] in Exercise 5.3, VIII.

# Very strong regularisation, see case [c] in Exercise 5.3, VIII.
lambda <- 1/delta^2

# Compute a grid of 'interesting' beta values for the given lambda.
#
# PCS_func() automatically generates a grid of beta values that concentrates
# points near the tipping point — the region where the PCS design is most
# sensitive to changes in beta and may become near-singular. Adequate resolution
# in this region is important, since small changes in beta can produce large
# changes in the predictor there.
#
# The grid depends only on lambda, not on beta. We therefore call PCS_func()
# with an arbitrary beta value solely to retrieve the grid.

# The beta value passed here is arbitrary; only the resulting grid is used.
beta <- 0

# Call PCS_func() to generate the beta grid for the specified lambda.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

# Extract the generated beta grid from the returned object.
beta_vec_automatic <- PCS_obj$beta_vec
# Slightly refine the grid at the right boundary to emphasize special cases
beta_vec<-c(beta_vec_automatic[1:(length(beta_vec_automatic)-1)],2.768514e-08,2.775e-08,2.777e-08,2.78e-08,beta_vec_automatic[length(beta_vec_automatic)])

# Type I) PCS: impose constraints from lags k=1 to h:
Delta <- 1:h

b_mat <- NULL
# Compute PCS predictors for given lambda and beta in beta_vec
for (i in 1:length(beta_vec)) {
  beta    <- beta_vec[i]
  PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)
  b       <- PCS_obj$b
  b_mat   <- cbind(b_mat, b)
}

filter_mat           <- b_mat
colnames(filter_mat) <- paste("lambda =", round(lambda, 2),
                              ", beta*lambda =", round(beta_vec*lambda, 8))

#───────────────────────────────────────────────────────────────────────────────
# 2.4 Plots
#───────────────────────────────────────────────────────────────────────────────

# The following function displays predictors and cross-correlation functions (CCFs)
# across four panels:
#
#   Panel 1 (top left):     Predictor profiles. All predictors are scaled to unit
#                           length to facilitate visual comparison.
#   Panel 2 (top right):    CCF of the predictor against xi, the original AR(1) process.
#   Panel 3 (bottom left):  CCF of the predictor against V1, the perturbed AR(1)
#                           direction (approximately equal to the original AR(1)).
#                           For small delta, Panels 2 and 3 are visually nearly
#                           indistinguishable, differing only by an arbitrary sign
#                           (the sign of V1 is not identified) and the small
#                           delta-perturbation, which is too subtle to be visible.
#   Panel 4 (bottom right): CCF of the predictor against V2, the full decoupling
#                           direction.

colo<-plot_func()


# ── Interpretation of Predictors ──────────────────────────────────────────────
#
# The predictor profiles displayed in Panel 1 illustrate the effect of the
# single-lag perturbation at lag 0. Because the perturbation affects only the
# lag-0 coefficient of xi, the PCS predictor b follows the standard AR(1)
# decay profile for all lags k >= 1, up to a possible sign inversion.
# The perturbation introduces a single degree of freedom — a fine-tuning
# adjustment at lag 0 only — leaving the predictor structure at all other
# lags unchanged (again, up to sign).
#
# This limited flexibility is the defining constraint of the single-lag
# perturbation: it expands the rank from 1 to 2, but the additional degree of
# freedom is confined entirely to lag 0. Consequently, the look-ahead behaviour
# along the full decoupling direction V2 (Panel 4) can shift the predictor
# peak from k = 0 to k = 1 at most — a relatively modest effect. By contrast,
# multi-lag perturbations reshape the predictor profile across all lags,
# generating richer decoupling effects and more pronounced peak shifts along
# their full decoupling directions V2; see Exercises 3 and 4 below.
#
# Under very strong regularisation, i.e., lambda = O(1/delta^2), the full
# decoupling direction V2 is emphasised; though V1 can be recovered from 
# particular beta values, see Exercise 5.3 VIII, case [c] for background.
#
#
# Notes
#  1. The coefficient vector b admits the reparametrisation:
#
#        b = lambda1 * V1 + lambda2 * V2,
#
#     where lambda1 and lambda2 are determined by lambda and beta (for given 
#     delta). Crucially, not all combinations of (lambda1, lambda2) are 
#     admissible: optimising the PCS criterion imposes constraints on their 
#     joint values.
#
#  2. Under very strong regularisation, lambda = O(1/delta^2), the full
#     decoupling direction V2 typically dominates V1. In particular, for 
#     large |beta| (asymptotically) the PCS predictor aligns with +/- V2. 
#     However, specific values of beta can unlock V1, allowing combinations of 
#     V1 and V2 — including V1 alone as a special case (violet lines in
#     Panel 1: beta * lambda ≈ 2.777). Such combinations are subject to
#     stronger shrinkage; see Exercise 5.3 VIII, case [c]. Note that the 
#     predictors in the first panel are scaled to unit-length (so shrinkage is 
#     not rendered).
#
#  3. The signs of V1 and V2 are arbitrary, as is standard for
#     eigenvectors. This is compensated by the signs of lambda1 and lambda2,
#     which adjust the orientation of b accordingly.

# ── Interpretation of CCF Panels ──────────────────────────────────────────────
#
# The PCS predictor b is a linear combination of V1 and V2, and therefore
# also of gamma_0 and perturbation_vec.
#
# Since V1 and V2 are orthogonal, the total CCF decomposes additively:
#
#   CCF(b, xi) = CCF(b, V1) + CCF(b, V2).
#
# Panel 3 — CCF against V1:
#   Isolates the gamma_0 contribution to the CCF. This reflects the immutable,
#   fixed AR(1) profile (up to the delta perturbation which is invisible when 
#   delta is small). It coincides with the CCF profile in Panel 2 (CCF against 
#   xi), up to an invisible delta-effect and a possible sign change due to 
#   arbitrary sign of the eigenvector V1.  
#
# Panel 4 — CCF against V2:
#   Isolates the `full decoupling' direction (V2) introduced by the perturbation.
#   This component captures the idiosyncratic look-ahead effect that arises
#   exclusively from the extraneous perturbation. Unlike the V1 component, it
#   depends directly on the type of the perturbation chosen. Different 
#   perturbations yield different V2 vectors and therefore different look-ahead 
#   profiles in this panel, see exercises 3 and 4 below.


# ─────────────────────────────────────────────────────────────────────
# 2.5 STRONG Regularization
# ─────────────────────────────────────────────────────────────────────

# Same as Exercise 2.3 but with strong regularisation, lambda = O(1/delta); see
# Exercise 5.3, VIII, case [b] for background.
#
# In contrast to Exercise 2.3 (very strong regularisation, lambda = O(1/delta^2)),
# a mixture of V[,1] and V[,2] is the natural outcome here: both directions
# contribute with similar importance. Pure V[,1] or pure V[,2] directions are
# possible only for special values of beta.
#
# As |beta| -> Inf, b remains a mixture of V[,1] and V[,2], unlike the very
# strong regularisation setting of Exercise 2.3, where b aligns asymptotically
# with +/- V[,2].

lambda<-1/delta

# Instead of manually adjusting beta (which can be tedious) we can 
# rely on grid computed by PCS_func():

# Arbitrary beta used only to trigger the grid computation (value is irrelevant).
beta <- 0

# Call PCS_func() to obtain the automatically generated beta grid.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

# Extract the beta grid from the PCS object.
beta_vec_automatic <- PCS_obj$beta_vec
# Add two extreme values at the boundaries
beta_vec<-c(0,beta_vec_automatic,10)
# Either the manual grid defined above or the automatic grid extracted here
# may be used in subsequent computations (the manually computed has slightly 
# better resolution).


b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}

filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta*lambda=",round(beta_vec*lambda,5))

# ─────────────────────────────────────────────────────────────────────
# 2.6 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors (Strong Regularisation) ──────────────────────
#
# The key differences relative to the very strong regularisation case in
# Section 2.4 are:
# - For very large |beta|, the solution is now an intermediate mix of V1 and
#   V2, rather than being dominated by V2 alone. 
# - The blend of V1 and V2 arises naturally, and shrinkage is generally less
#   pronounced (note that predictors are scaled to unit length in the first 
#   panel, so shrinkage effects are not directly visible).


# ─────────────────────────────────────────────────────────────────────
# 2.7 MILD Regularization
# ─────────────────────────────────────────────────────────────────────

# Same as Exercise 2.3 but with mild regularisation (lambda = 1); see
# Exercise 5.3 VIII case [a] for background. The natural solution aligns with
# V1, although mixtures of V1 and V2 are also possible. A pure V2 direction
# remains attainable, but is subject to strong shrinkage. Asymptotically, as
# |beta| -> Inf, b aligns with V1, in contrast to the very strong and strong
# regularisation cases discussed above.

# Medium regularization
lambda<-1


# Instead of manually adjusting beta (which can be tedious) we can 
# rely on grid computed by PCS_func():

# Arbitrary beta used only to trigger the grid computation (value is irrelevant).
beta <- 0

# Call PCS_func() to obtain the automatically generated beta grid.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

# Extract the beta grid from the PCS object.
beta_vec_automatic <- PCS_obj$beta_vec
# Add two extreme values at the boundaries
beta_vec<-c(0,beta_vec_automatic,10)
# Either the manual grid defined above or the automatic grid extracted here
# may be used in subsequent computations (the manually computed has slightly 
# better resolution).


b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}

filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta*lambda=",round(beta_vec*lambda,5))

# ─────────────────────────────────────────────────────────────────────
# 2.8 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors (Strong Regularisation) ──────────────────────
#
# Compared to the very strong (lambda = 1/delta^2) and strong (lambda = 1/delta)
# regularisation cases, the key differences are:
#
# - For very large |beta|, the solution aligns with +/- V1 rather than a mix
#   of V1 and V2 (under strong regularization), or the pure V2 direction (under 
#   very strong regularization).
#
# - V1 emerges as the natural solution direction, although mixtures of V1 and V2
#   remain possible. The pure V2 direction is also attainable, but the
#   corresponding PCS is subject to strong shrinkage in that case;
#   see Exercise 5.3 VIII, case [a]. Note that predictors are scaled to unit
#   length in the first panel, so shrinkage effects are not directly visible.

#
# Notes:
#
# 1. Solution space vs. full span:
#    All PCS predictors in this exercise are linear combinations of the form:
#
#      b = lambda1 * gamma_0 + lambda2 * perturbation_vec.
#
#    However, not every linear combination of gamma_0 and perturbation_vec is
#    a valid PCS solution: the PCS criterion additionally requires maximisation
#    of the target correlation subject to the penalty on constraint deviation.
#    The weights lambda1 and lambda2 are therefore implicitly determined by the
#    hyperparameters lambda, delta and beta via the penalised criterion, and should not
#    be set independently.
#
# 2. Extending to Higher Rank via Additional Perturbations:
#    The rank of the constraint system can be increased beyond two by introducing
#    additional linearly independent perturbations of the form delta_i * e_i,
#    where e_i is the i-th unit vector and delta_i != 0 are small scaling
#    factors. Each such perturbation introduces a new eigenvector Vi into the
#    solution space, expanding it from rank 2 to rank 3, 4, and so on.
#
#    Adding further perturbations at lags i > 0 would not affect the CCF against
#    gamma_0 or V1 (Panels 2 and 3), since these are immutable: they are fully
#    determined by the rigid rank-one AR(1) structure, which no perturbation can
#    alter. The effect of additional perturbations would instead be visible in
#    the CCF against V2 (Panel 4) or against the newly introduced eigenvectors
#    Vi (i > 2). Each additional perturbation direction introduces a new
#    full-decoupling component — orthogonal to gamma_0 and to all previously
#    introduced decoupling directions — thereby expanding the space of achievable
#    look-ahead profiles.
#
#    Exploring the effect of such higher-rank perturbations on the PCS predictor
#    and its CCF decomposition is left as an exercise.





# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: FULL-LAG PERTURBATION — THE AR(1) CASE
# ════════════════════════════════════════════════════════════════════

# In contrast to Exercise 2, the perturbation introduced here affects all lags
# of the Wold decomposition simultaneously, rather than a single lag only.
# This guarantees that the rank of the constraint system is expanded from 1 to 2
# for any PCS constraint type, without requiring gamma_0 itself to appear
# explicitly in the constraint system. (In Exercise 2, the lag-0 perturbation
# had no effect unless gamma_0 entered the constraints directly.)

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# 3.1 Full-Lag AR(1) Perturbation
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

# Select initial parameter values to analyse the geometry of the perturbed
# PCS problem; for illustration we select a larger lambda, prioritizing the constraints 
# over the target correlation.
lambda <- 10000
beta   <- 0

PCS_obj <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                 gammah_mat_perturbate)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol


# 3.2 Linear Algebra of the Rank-Two System
#----------------------------------------------------------------------------------
# The closed-form PCS solution is: b = solve(M) %*% gamma_sol.
#   - M depends on lambda but not on beta.
#   - gamma_sol depends on both lambda and beta (via lambda * beta).

# gamma_sol is no longer perfectly AR(1): the consecutive ratio is not constant.
ts.plot(gamma_sol, main = "gamma_sol: departure from AR(1) decay")
# The ratio slowly evolves: this change is determined by delta.
gamma_sol[2:L] / gamma_sol[1:(L - 1)]

# Eigendecomposition of M.
eigenM <- eigen(M)
V      <- eigenM$vectors

# Verify M = I + lambda * N: residual should vanish.
max(abs(M - diag(rep(1, L)) - lambda * N))

# Confirm rank two: exactly two eigenvalues of N exceed the threshold 10^{-10}.
eigenN <- eigen(N)
which(abs(eigenN$values) > 1e-13)

# Key spectral properties:
#   - Eigenvalues of M = I + lambda * N are: 1 + lambda * n_i (n_i from N).
#   - Eigenvalues of M^{-1} are: 1 / (1 + lambda * n_i).
#   - M and N share the same eigenvectors.
#   - rank(N) = 2,  rank(M) = L.
#   - The eigenvectors depend only on the constraint matrix (independent of lambda and beta).
#   - The eigenvalues of N depend on the constraint matrix; those of M depend additionally on lambda.
#   - gamma_sol depends on the product lambda * beta.

# Visualise the two leading eigenvectors of N.
par(mfrow = c(1, 1))
ts.plot(eigenN$vectors[, 1:2],
        main = "Leading eigenvectors of N (rank-two structure)",lty=1:2)

# Visualise the two leading eigenvectors of M.
ts.plot(V[, 1:2], main = "Leading eigenvectors of M",lty=1:2)

# Confirm that V1 decays geometrically (AR(1) direction) if delta is small.
V[2:L, 1] / V[1:(L - 1), 1]

# Since gamma_sol lies in the column space of V[,1:2], all projections
# t(V[,k]) %*% gamma_sol should vanish for k > 2.
t(V) %*% gamma_sol

# Intermediate vector g = diag(1/d) %*% t(V) %*% gamma_sol:
# only its first two elements are non-zero.
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Verify: b = V %*% g = g[1]*V1 + g[2]*V2. Residual should vanish.
max(abs(V %*% g - g[1] * V[,1] - g[2] * V[,2]))

# The PCS predictor b lies in the plane spanned by V1 and V2, or
# equivalently, by xi[1:L] and xi_a1_perturbate[1:L].
b <- V %*% diag(1 / eigenM$values) %*% t(V) %*% gamma_sol

ts.plot(b, main = "PCS predictor: rank-two, full-lag perturbation")

# Confirm departure from AR(1): consecutive ratios are no longer constant if 
# lambda is large.
b[2:L] / b[1:(L - 1)]

# This result holds for any lambda > 0 and any beta but larger lambda emphasize 
# the constraints (which are perturbated) more strongly, leading to stronger 
# departure of b from the original AR(1) profile.


# 3.3 Exploring the Rank-Two System: Strong Regularisation
#----------------------------------------------------------------------------------
# Under strong regularisation, the PCS predictor transitions smoothly between
# the two extremes ±V2 as beta varies, passing through -V1 at the
# tipping point where the V2 contribution vanishes.
# The weights on V1 and V2 are governed by lambda and beta. A complete analysis 
# is provided in Exercise 5.3 VIII case [c].

# Strong regularisation weight.
lambda <- 5000000

# Notes:
# - In contrast to Exercise 2, which emphasised very strong (lambda = 1/delta^2),
#   strong (lambda = 1/delta), and mild (lambda = 1) regularisation, here we
#   focus on strong (lambda = 5,000,000; see above) and medium (lambda = 5;
#   see Exercise 3.6 below) regularisation, not directly connected to delta.
# - We do not attempt to cover the full interplay between lambda, beta, and
#   delta, which is analysed comprehensively in Exercise 5.3 VIII.
# - Instead, we demonstrate that meaningful look-ahead behaviour can be
#   obtained under strong and medium regularisation, without requiring an
#   exhaustive analysis of the complex hyperparameter interactions arising
#   from the Full-lag AR(1) perturbation considered here.

# Remark:
# The design exhibits marked sensitivity to beta: small perturbations in beta
# can induce substantial changes in the PCS solution. This is a consequence of
# the near-singularity of the PCS criterion as delta shrinks toward zero.
# This singularity is not merely a numerical inconvenience; it is intimately
# tied to interpretability. Specifically, as delta -> 0:
#   - The first eigenvector V1 converges to gamma0 (the AR(1) DGP or nowcast).
#   - The second eigenvector V2 aligns with the full decoupling direction
#     induced by the perturbation.
# Small delta therefore sharpens the geometric interpretation of these two
# directions, but at the cost of an increasingly ill-conditioned optimisation
# landscape. This trade-off is deliberate: interpretability is here prioritised
# over numerical stability.

if (F)
{
# Manual grid
  beta_vec <- c(0 ,0.5,0.8,0.83,0.85,0.87,0.88,0.9,0.95,0.97,1,1.1,10) / lambda
}

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

# Note:
# The asymptotic behaviour of the grid tails — i.e. as |beta| -> infinity —
# depends on the interplay between beta, lambda, and delta (see Exercise 5.3 VIII
# and Exercise 2 above). Depending on the particular combination of these
# hyperparameters, the PCS predictor b aligns with one of the following
# directions as |beta| increases:
#   - V1 alone,
#   - a mixture of V1 and V2, or
#   - V2 alone.



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
# 3.4 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors: Full-Lag AR(1) Perturbation ─────────────────
#
# In contrast to Exercise 2, where the perturbation was confined to lag 0, the
# full-lag AR(1) perturbation affects every coefficient of the PCS predictor b.
# This richer perturbation structure expands the solution space more broadly,
# enabling, among other things, a non-monotonic seemingly cyclical coefficient 
# profile (violet tones, larger beta values) — a notably non-trivial outcome in an 
# inherently aperiodic framework where the underlying Wold coefficients of original 
# and perturbated systems decay monotonically.
#
# Under strong regularisation, the PCS predictors transition smoothly from -V2
# to +V2 as beta increases. The two extremes, ±V2, correspond to configurations
# where constraint satisfaction is driven entirely by the decoupling direction
# V2, with V1 (the gamma_0 component) receiving zero weight. V2 is the primary
# enabler of the PCS constraints and is therefore emphasised when lambda is
# large: the optimiser allocates most of its budget to satisfying the constraints
# via V2, at the expense of the target correlation carried by V1.
#
# Between the two extremes, the predictor traces a continuum of optimal linear
# combinations of V1 and V2, with the relative weights determined by the
# interplay between target correlation maximisation (governed by V1) and
# constraint satisfaction (governed by V2), for the specified values of lambda
# and beta, see Exercise 5.3 VIII for a detailed analysis.
#


#
# ── Interpretation of CCF: Full-Lag AR(1) Perturbation ────────────────────────
#
# In contrast to Exercise 2, the CCF in Panel 4 (CCF against V2) now exhibits
# genuine look-ahead behaviour: as beta increases, the peak of the CCF shifts
# progressively to the right. This right-shifting of the CCF peak reflects the 
# increasing weight placed on the full decoupling direction V2 by increasing 
# beta, which — unlike the lag-0 perturbation in Exercise 2 — reshapes
# the predictor across all lags and thereby induces a meaningful lead or 
# left-shift as demonstrated in the predictor plots in the next exercise 3.5.

# Notes:
# 1. At lag 0, the CCF of the MSE predictor vanishes in panel 4:
#    CCF_MSE(0) = 0. This is a consequence of full decoupling along V2, i.e.,
#    the MSE predictor stands orthogonal to V2.
# 2. The left-shift of the CCF peak of the PCS predictors along the decoupling 
#    direction V2 can exceed the forecast horizon h. 
#
# We now explore look ahead behaviour of the perturbated PCS approach.

# ─────────────────────────────────────────────────────────────────────
# 3.5 Apply and Compare Predictors
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
# 3.6 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Lambda is fixed at a moderate regularization strength. 

lambda<-5


# PCS_perturbation_func() automatically returns a grid of beta values centred on 
# the tipping point — where the sensitivity of the PCS solution with respect to beta is
# highest. 

# The grid is independent of beta: any value can be supplied
beta<-0.

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)

# Grid of beta values 
beta_vec<-PCS_obj$beta_vec
# PCS constraint system (as above: lags 1 to h).
Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta*lambda=",round(beta_vec*lambda,8)))

# ─────────────────────────────────────────────────────────────────────
# 3.7 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V1 and +V1, passing through V2 at an intermediate tipping point, see Exercise 
# 5.3 VIII case [a].

# The CCF in the fourth panel illustrates look ahead behaviour: the CCF peak 
# is shifted rightwards along the fully decoupled V2 direction. However, 
# too strong look ahead (larger beta) induce sign inversion  (negative CCF 
# against xi).

colo<-plot_func()


# ─────────────────────────────────────────────────────────────────────
# 3.8 Apply and Compare Predictors
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
# We select the first 5 PCS designs with look ahead (right-shift of CCF peak 
# along full decoupling V2 direction) but without sign inversion. For 
# illustration  we also include the sixth PCS which is subject to sign 
# inversion (negative CCF against xi). 
select_pcs<-c(2:6)
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

select_vec<-c(1,select_pcs)
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
# Compute the empirical cross-correlation function (CCF) between the MSE
# predictor output (column 1) and the selected (leading) PCS predictor output.
#
# Key observations:
#   - As beta increases, the empirical CCF becomes increasingly right-skewed,
#     reflecting a growing lead of the PCS predictor relative to the MSE predictor.
#
#   - The right tail of the CCF (lag > 0) is immutable and always follows the AR(1) decay:
#     b %*% gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
#     consequence of the Yule-Walker equations and holds for any linear predictor b.
#     No non-zero predictor can alter this decay shape.
#
#   - Consequently, shifting the CCF peak strictly to the right of lag 0 is
#     impossible under the AR(1) DGP (see Exercise 1).
#
#   - Only the left tail of the CCF is amenable to modification. Whereas the MSE
#     predictor (first panel) yields a symmetric CCF, the PCS predictor becomes
#     progressively more asymmetric as beta increases. Effective look-ahead 
#     behaviour (illustrated in the predictor plot above) is thus achieved by 
#     skewing the CCF rightward — that is, by down-weighting the contribution 
#     of negative lags.

mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,1],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])




# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: ALTERNATIVE AR(2) PERTURBATION
# ════════════════════════════════════════════════════════════════════

# In analogy to Exercise 3, we propose a perturbation acting across all lags.
# Here, however, the perturbation takes the form of an AR(2) modification:
# the original AR(1) autocovariance structure is overlaid with an AR(2)
# component whose weight can be made arbitrarily small, rendering its
# effect on the DGP imperceptible.

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────
# 4.1 Perturbation: AR(2) Type
# ─────────────────────────────────────────────────────────────────────
#

# MSE predictor coefficients derived from the original AR(1) DGP,
# used to define the PCS constraints
gamma_all <- xi

# Build the shifted covariance matrix 'gammah_mat':
# each row contains the MSE predictor coefficients (gamma_all) shifted by
# a specific lead value drawn from 'Delta'.
# The initialisation starts at Delta[1] - 1 because the constraints are
# expressed as first differences (gamma_h - gamma_{h-1}), requiring
# gamma_{Delta[1] - 1} to define the first such difference.
gammah_mat <- gamma_all[Delta[1] - 1 + 1:L] / sqrt(sum(gamma_all^2))
if (length(Delta) > 0) {
  for (i in 1:length(Delta)) {
    gammah_mat <- rbind(gammah_mat,
                        gamma_all[Delta[i] + 1:L] / sqrt(sum(gamma_all^2)))
  }
}

# Specify a periodic AR(2) process as the perturbation component

omega<-pi/48
mod<-0.95
a1_ar2<-2*mod*cos(omega)
a2_ar2<--mod^2

xi_ar2_all <- c(1, ARMAtoMA(ar = c(a1_ar2, a2_ar2), ma = 0, lag.max = 2000))
k_start <- 12
xi_ar2  <- xi_ar2_all[k_start + 1:1001]

gamma_all_ar2 <- xi_ar2

# Visual comparison of the AR(1) and AR(2) autocovariance sequences
par(mfrow = c(1, 1))
ts.plot(cbind(xi, xi_ar2), col = c("black", "red"),
        main = "AR(1) (black) and AR(2) (red)")

# Construct the imperceptible perturbation by scaling the AR(2) component
# by a small weight delta
delta <- 0.0001

gammah_mat_perturbate_ar2 <- gammah_mat

# Modify only the first row of gammah_mat (corresponding to gamma_0):
# the constraint system has rank 2, so it suffices to perturb gamma_0 alone.
# All remaining rows entering the constraints retain the original AR(1) DGP.
gammah_mat_perturbate_ar2[1, ] <- gammah_mat[1, ] +
  delta * gamma_all_ar2[1:L] / sqrt(sum(gamma_all_ar2^2))

# Select initial parameter values to analyse the geometry of the perturbed
# PCS problem; any reasonably sized (beta, lambda) pair is suitable here
lambda <- 5
beta   <- 0

# Compute the PCS predictor under the AR(2) perturbation
PCS_obj  <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                  gammah_mat_perturbate_ar2)

b        <- PCS_obj$b
d_delta  <- PCS_obj$d_delta
b_mat    <- cbind(b_mat, b)
M        <- PCS_obj$M
N        <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol


# ─────────────────────────────────────────────────────────────────────
# 4.2 Background: Linear Algebra of the PCS Solution
# ─────────────────────────────────────────────────────────────────────
#
# The closed-form PCS solution is: b <- solve(M) %*% gamma_sol
#   - M depends on lambda but not on beta.
#   - gamma_sol depends on both lambda and beta.
# ─────────────────────────────────────────────────────────────────────

# Inspect gamma_sol: its decay is not purely exponential, confirming
# that it does not follow the AR(1) autocovariance structure
ts.plot(gamma_sol)
gamma_sol[2:L] / gamma_sol[1:(L - 1)]

eigenM <- eigen(M)
V      <- eigenM$vectors

# M = I + lambda * N, where N = sum_{k=1}^{h} (gamma_k - gamma_{k-1})(gamma_k - gamma_{k-1})'
# Verify: the following maximum absolute difference should be (near) zero
max(abs(M - diag(rep(1, L)) - lambda * N))

# N has rank 2: only two eigenvalues exceed the numerical threshold 1e-10
eigenN <- eigen(N)
which(abs(eigenN$values) > 1e-10)

# Inspect the two eigenvectors associated with the non-vanishing eigenvalues of N
par(mfrow = c(1, 1))
ts.plot(eigenN$vectors[, 1:2],
        main = "Eigenvectors of non-zero eigenvalues of N", lty = 1:2)

# Key spectral results:
#   - Eigenvalues of M = I + lambda*N are 1 + lambda * n_i, where n_i are
#     the eigenvalues of N.
#   - Eigenvalues of M^{-1} are 1 / (1 + lambda * n_i).
#   - M and N share the same eigenvectors.
#   - Rank(N) = 2; Rank(M) = L (M is full rank for any lambda > 0).
# Hint: the sign of the eigenvectors is arbitrary: the eigenvectors of M and N 
# may differ with regards to signs.
ts.plot(V[, 1:2], main = "First two eigenvectors of M", lty = 1:2)

# gamma_sol lies in the column space of V[,1:2]:
# consequently, t(V[,k]) %*% gamma_sol = 0 for all k > 2.
# Verification: only the first two elements of t(V) %*% gamma_sol are non-zero
t(V) %*% gamma_sol

# Compute the projected and scaled coefficients g
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Since b = solve(M) %*% gamma_sol = V %*% diag(1/eigenM$values) %*% t(V) %*% gamma_sol,
# and g has only two non-zero elements, it follows that:
#   b = V %*% g = g[1] * V1 + g[2] * V2
# Verification: the following maximum absolute deviation should be (near) zero
abs(max(V %*% g - g[1] * V[,1] - g[2] * V[,2]))

# Conclusion: the PCS predictor b lies in the space spanned by V1 and V2
# (equivalently, by xi[1:L] and the perturbed AR(2) direction).
# It is therefore a linear combination of these two eigenvectors,
# irrespective of the choice of lambda.
b <- V %*% diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
ts.plot(b)

# Confirm that b does not follow AR(1) decay in general: larger lambda emphasize 
# the constraints (which are perturbated) and favors departure from the original 
# AR(1) profile.
b[2:L] / b[1:(L - 1)]


# ─────────────────────────────────────────────────────────────────────
# 4.3 Rank-Expansion: Strong Regularization
# ─────────────────────────────────────────────────────────────────────
#
# Lambda is fixed at a very large value (strong regularization) and beta
# is varied across a grid. The two boundary values of beta correspond to
# -V2 and +V2, with intermediate solutions mixing V1 and  V2, see 
# Exercise 5.3 VIII case [c] for details.

# When the perturbation conditions the constraints into a misspecified design 
# (here AR(2) instead of AR(1)), emphasizing the constraints at the detriment of 
# the target correlation through a large lambda might cause troubles.
# ─────────────────────────────────────────────────────────────────────

# Strong regularization
lambda <- 5000000

# Note: the design exhibits marked sensitivity to beta when delta is small.
if (F)
{
  # Manual grid
  beta_vec <- c(-10, 1,  2,2.2, 2.3,2.4,2.5,2.6,2.7,2.85,3,3.5,10) / lambda
}

# Instead of cumbersome manual tuning of beta, PCS_perturbation_func()
# automatically returns a grid of beta values centred on the tipping point
# — where the sensitivity of the PCS solution with respect to beta is
# highest. 

# The grid is independent of beta: any value can be supplied
beta<-0.

# We must supply the new AR(2) perturbation : gammah_mat_perturbate_ar2

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)

# Grid of beta values 
beta_vec<-PCS_obj$beta_vec
# PCS constraint system (as above: lags 1 to h).
Delta<-1:h

# Compute PCS predictor coefficients for each value of beta
b_mat <- NULL
for (i in 1:length(beta_vec)) {
  beta    <- beta_vec[i]
  PCS_obj <- PCS_perturbation_func(h, Delta, gamma_pcs, L, beta, lambda,
                                   gammah_mat_perturbate_ar2)
  b       <- PCS_obj$b
  b_mat   <- cbind(b_mat, b)
}

# Combine MSE baseline with PCS predictors; label columns accordingly
filter_mat <- cbind(gamma0, b_mat)
colnames(filter_mat) <- c("MSE",
                          paste("lambda =", round(lambda, 2),
                                ", beta*lambda =", round(lambda*beta_vec, 8)))

# ─────────────────────────────────────────────────────────────────────
# 4.4 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V2 and +V2, passing through mixes of V1 and V2, see Exercise 5.3 VIII case [c]. 
#
# In contrast to Exercise 3, the CCF against xi (second panel)
# is either near zero or negative throughout. This indicates that placing
# excessive weight on the (misspecified) perturbed AR(2) constraints via large lambda
# induces misspecification in this example: the predictor loses meaningful 
# correlation with the target. A more balanced strategy is to employ small to
# moderate values of lambda, so that target correlation remains a
# relevant and influential component of the optimisation objective.

colo<-plot_func()


# ─────────────────────────────────────────────────────────────────────
# 4.5 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

# The PCS predictors obtained under strong regularization (large lambda)
# are biased toward sign inversion in this example. This arises primarily from the
# misspecification of the perturbation: unlike Exercise 3, where the
# perturbation is aligned with the AR(1) structure, the AR(2) perturbation
# introduced here is not, causing the constraint penalty to dominate and
# distort the solution. The CCF against xi (second panel) confirms this
# sign inversion, and forecast comparisons are therefore omitted for
# this configuration.




# ─────────────────────────────────────────────────────────────────────
# 4.6 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a medium regularization
# We then vary beta: the two extreme beta values correspond to +/-V1 with mixes 
# of V1 and V2 in between, see exercise 5.3 VIII, case [a] for details.



# Medium regularization
lambda<-5


# PCS_perturbation_func()
# automatically returns a grid of beta values centred on the tipping point
# — where the sensitivity of the PCS solution with respect to beta is
# highest. 

# The grid is independent of beta: any value can be supplied
beta<-0.

# We must supply the new AR(2) perturbation : gammah_mat_perturbate_ar2

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)

# Grid of beta values 
beta_vec<-PCS_obj$beta_vec
# PCS constraint system (as above: lags 1 to h).
Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta*lambda=",round(lambda*beta_vec,8)))



# ─────────────────────────────────────────────────────────────────────
# 4.7 Plots
# ─────────────────────────────────────────────────────────────────────

# In contrast to Exercise 4.4 (strong regularization, large lambda), a weak
# or moderate regularization (small to medium lambda) assigns meaningful
# weight to the AR(1) target correlation objective, thereby avoiding the overt
# misspecification induced by the misspecified AR(2) perturbation — provided beta does
# not become too large. Excessively large beta places disproportionate
# emphasis on the constraints, eventually driving the predictor into sign-
# inversion territory, as evidenced by the negative CCF against xi
# (second panel).
#
# The CCF along the fully decoupling direction V2 (fourth panel) confirms
# peak shifting of the CCF: as beta increases, the CCF peak shifts progressively to
# the right, up to the point at which sign inversion occurs.

# The predictor comparisons presented below are restricted to PCS designs
# that do not exhibit sign inversion.

colo<-plot_func()


# ─────────────────────────────────────────────────────────────────────
# 4.8 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

len<-10000
set.seed(534)

x_filt <- rnorm(len)

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


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

# Select PCS designs that do not exhibit sign inversion: all
# of these maintain a positive CCF against xi (second panel in the plot
# above). For completeness, the first sign-inverting design (column 6) is 
# also included to illustrate the onset of sign inversion.
select_pcs<-2:7

select_vec<-c(1,select_pcs)
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
# Compute the empirical cross-correlation function (CCF) between the MSE
# predictor output (column 1) and the selected (leading) PCS predictor output.
#
# Key observations:
#   - As beta increases, the empirical CCF becomes increasingly right-skewed,
#     reflecting a growing lead of the PCS predictor relative to the MSE predictor.
#
#   - The right tail of the CCF (lag > 0) is immutable and always follows the AR(1) decay:
#     b %*% gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
#     consequence of the Yule-Walker equations and holds for any linear predictor b.
#     No non-zero predictor can alter this decay shape.
#
#   - Consequently, shifting the CCF peak strictly to the right of lag 0 is
#     impossible under the AR(1) DGP (see Exercise 1).
#
#   - Only the left tail of the CCF is amenable to modification. Whereas the MSE
#     predictor (first panel) yields a symmetric CCF, the PCS predictor becomes
#     progressively more asymmetric as beta increases. Effective look-ahead 
#     behaviour (illustrated in the predictor plot above) is thus achieved by 
#     skewing the CCF rightward — that is, by down-weighting the contribution 
#     of negative lags.

mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,1],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])




# ════════════════════════════════════════════════════════════════════
# EXERCISE 5: PARAMETER INTERPLAY   (under construction)
# ADVANCED ANALYSIS BASED ON PERTURBATION OF EXERCISE 2. 
# ════════════════════════════════════════════════════════════════════

# We analyse here the complex interplay between three key parameters:
#   - delta:  the perturbation size 
#   - lambda: the regularization weight (denoted nu in Eq. 49 of Wildi (2026))
#   - beta:   the slope constraint weight
#
# The relationship between lambda and beta is governed by Eq. 49 in Wildi (2026).
# Two important notational points regarding that formula:
#   (i)  The regularization weight is denoted nu in Eq. 49, corresponding to lambda here.
#   (ii) lambda (i.e. nu) appears in two distinct roles: as a multiplicative weight on the
#        slope term AND as a constituent of the regularized matrix M^{-1} = (I + lambda * N)^{-1}.

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


#───────────────────────────────────────────────────────────────────────────────
# 5.1 Replicate the Framework of Exercise 2  
#───────────────────────────────────────────────────────────────────────────────

# Forecast horizon.
h <- 12

# Construct the perturbation vector: delta times the first unit vector e_1,
# so that only the lag-0 coefficient of xi is modified.
e1                <- c(1, rep(0, length(xi) - 1))
delta             <- 0.0001
perturbation_vec  <- delta * e1
xi_perturbated    <- xi + perturbation_vec

# Perturbed target vector for PCS.
gamma_pcs_perturbated <- xi_perturbated

# Extract the perturbed h-step predictor coefficient vectors.
gamma0_perturbated <- xi_perturbated[1:L]
gamma1_perturbated <- xi_perturbated[1 + 1:L]
gamma2_perturbated <- xi_perturbated[2 + 1:L]

# The perturbation affects only lag 0 of gamma0
gamma0_perturbated[2:L]/gamma0_perturbated[1:(L-1)]
# gamma1 or any gammah, h>0, is not affected:
gamma1_perturbated[2:L] / gamma1_perturbated[1:(L - 1)]

# The first two PCS constraints involve the differences:
#   delta_1 = gamma_0 - gamma_1   (affected by the perturbation at lag 0)
#   delta_2 = gamma_1 - gamma_2   (unaffected; both are pure AR(1))
delta1 <- gamma0_perturbated - gamma1_perturbated
delta2 <- gamma1_perturbated - gamma2_perturbated

# For k > 2, delta_k = gamma_{k-1} - gamma_k is proportional to delta_{k-1},
# since the perturbation affects only the lag-0 coefficient of xi.

# delta1 lies in the span of {gamma_0, perturbation_vec}: both components
# are statistically significant and the residual vanishes.
gamma0 <- xi[1:L]
summary(lm(delta1 ~ gamma0 + perturbation_vec[1:L] - 1))

# delta_2 lies in the span of {gamma_0} alone: the perturbation has no
# effect on delta_2 or any delta_k, k>1.
summary(lm(delta2 ~ gamma0 - 1))

# Type I PCS: constrain all lags from k=1 to k=h (most restrictive type).
Delta <- 1:h

# Regularisation weight: very strong regularization.
lambda <- 10000000

# Constraint slope parameter.
beta <- -0.0001

# Compute the Type I PCS predictor for the perturbed system.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol
beta_vec  <- PCS_obj$beta_vec


#───────────────────────────────────────────────────────────────────────────────
# 5.2 Set-Up and Dependence on Perturbation Size (under construction)
#───────────────────────────────────────────────────────────────────────────────

#───────────────────────────────────────────────────────────────────────────────
# 5.2.1 Set-Up (under construction)
#───────────────────────────────────────────────────────────────────────────────

# Extract core quantities from the PCS object
d_delta     <- PCS_obj$d_delta       # Constraint system
M           <- PCS_obj$M             # Inversion matrix
sum_d_delta <- PCS_obj$sum_d_delta   # Cumulated PCS type-I constraint vector

gamma_sol <- PCS_obj$gamma_sol  # Target used in the regularized criterion
gammah    <- PCS_obj$gammah     # MSE h-step predictor

# Compute the filter coefficient vector b via the regularized normal equations
# (see Equation 49 in Wildi (2026)):  b = M^{-1} * gamma_sol
b_formula <- solve(M) %*% gamma_sol

# Verify consistency: the difference between b_formula and b should be negligible
max(abs(b_formula - b))

par(mfrow = c(1, 1))
# Plot b: note that, unlike the pure AR(1) case, b no longer has an AR(1) structure
ts.plot(b)

# Compute the eigenvectors and eigenvalues of M and N = M - I
V   <- eigen(M)$vectors   # Eigenvectors of M (columns)
eta <- eigen(N)$values    # Eigenvalues of N
# Note: M = I + lambda * N, see Wildi Appendix D.

# Verify that M is correctly diagonalized: max deviation should vanish
max(abs(M - V %*% diag(eigen(M)$values) %*% t(V)))

# Verify rank-2 structure of perturbated system: only the first two entries of 
# t(V) %*% gammah are non-vanishing (without perturbation only the first entry 
# is non-vanishing: the second entry is of order O(delta))
t(V) %*% gammah


#───────────────────────────────────────────────────────────────────────────────
# 5.2.2 Dependence on Perturbation Size (delta) (under construction)
#───────────────────────────────────────────────────────────────────────────────

# Ratio of the first two eigenvalues of N is of order O(1/delta^2)
eta[1]/eta[2]
# Multiplied by delta^2: O(1)
delta^2*eta[1]/eta[2]
# Multiplying by delta^2 yields an O(1) quantity, confirming the scaling

# Explanation:
# - N is a sum of outer products of first differences of the ACF:
#       N = sum_{k in Delta} (gamma_k - gamma_{k-1}) %*% t(gamma_k - gamma_{k-1}).
# - Each outer product (gamma_k - gamma_{k-1}) %*% t(gamma_k - gamma_{k-1}) is an L x L
#   (rank-1) matrix, so N is an L x L matrix (of rank at most |Delta|).
# - In the absence of perturbation, (gamma_k - gamma_{k-1}) %*% t(gamma_k - gamma_{k-1}) 
#   are proportional, i.e., the rank of N is one.
# - In the presence of a perturbation:
#   - The column rank of N equals 2 if 1 in Delta (since gamma_1 - gamma_{0} and the
#     remaining differences gamma_k - gamma_{k-1}, k>1, are linearly independent in that case)
#   - The column rank of N equals 1 otherwise.
# - Because N is built from squared (outer-product) expressions in the perturbation
#   magnitude delta, its non-zero eigenvalues are O(1) and O(delta^2).
# - Specifically: eta[1] = O(1), eta[2] = O(delta^2), and eta[k] = 0 for k >= 3,
#   which implies eta[1] / eta[2] = O(1/delta^2).

# V1 (first eigenvector) is nearly AR(1): consecutive ratios deviate from a
# constant by O(delta) only
V[2:L, 1] / V[1:(L-1), 1]

# Because V2 is orthogonal to V1 and V1 nearly aligns with gammah, V2 is nearly
# orthogonal to gammah: the inner product t(V2) %*% gammah is O(delta).
# Dividing by delta yields an O(1) quantity, confirming the scaling
V[,2] %*% gammah / delta

# Equivalently, using the second row of t(V) (denoted tV2 below)
t(V)[2, ] %*% gammah / delta

# Similarly, tV2 %*% sum_d_delta is O(delta):
#   sum_d_delta = O(1) * gammah + perturbation 
#   tV2 %*% gammah   = O(delta)  (shown above)
#   tV2 %*% perturbation = O(delta)  (since perturbation = O(delta))
t(V)[2, ] %*% sum_d_delta / delta
# Note: sum_d_delta = sum_{k in Delta} (gamma_k - gamma_{k-1}) encodes the
#    cumulated PCS type-I constraints, see Wildi (2026), Appendix D. 
#    

#───────────────────────────────────────────────────────────────────────────────
# Summary of delta-scalings (tVi denotes the i-th row of t(V))
#   tV2 %*% gammah        = O(delta)
#   eta_2                 = O(delta^2)  [second non-zero eigenvalue of N]
#   tV2 %*% sum_d_delta   = O(delta)
#───────────────────────────────────────────────────────────────────────────────


#───────────────────────────────────────────────────────────────────────────────
# 5.3 Analysis: Interplay of Hyperparameters delta, lambda, and beta
#───────────────────────────────────────────────────────────────────────────────

# Recall that:
#   t(V2) %*% gamma_h       = O(delta)
#   eta_2                   = O(delta^2)   [second non-zero eigenvalue of N]
#   t(V2) %*% sum_d_delta   = O(delta)
# See Exercise 5.2.2.

# I) Solution Formula
#
#    The regularised solution is:
#
#      b = M^{-1} %*% (gamma_h + lambda * beta * sum_d_delta),
#
#    where gamma_h is the ideal target ACF vector and
#    sum_d_delta = sum_{k in Delta} (gamma_k - gamma_{k-1}) encodes the
#    cumulated PCS type-I constraints. Since k = 1 lies in Delta, the 
#    perturbation enters sum_d_delta through the term involving gamma_0.


# II) Rank-2 Structure
#
#     For an AR(1) process with a single perturbation, both gamma_h and the
#     difference vectors (gamma_k - gamma_{k-1}), k = 1, ..., L, span a
#     rank-2 subspace. The matrix N shares this rank-2 column space, with
#     two non-zero eigenvalues eta_1 and eta_2 and corresponding leading
#     eigenvectors V1 = V1 and V2 = V2 lying in this subspace,
#     while V[, k] for k = 3, ..., L are orthogonal to it.
#
#     Since M = I + lambda * N, M has full rank for any positive
#     regularisation weight lambda > 0. Moreover, M and N share the same
#     eigenvectors. As a symmetric matrix, M admits the eigendecomposition
#
#       M = V %*% D %*% t(V),
#
#     where D is diagonal with entries
#
#       (1 + lambda * eta_1, 1 + lambda * eta_2, 1, ..., 1)  [length L],
#
#     and the inverse is given by
#
#       M^{-1} = V %*% diag(1/diag(D)) %*% t(V),
#
#     with reciprocal diagonal entries. When delta is small, the leading
#     eigenvector satisfies V1 ≈ gamma_h / ||gamma_h||, i.e., it is
#     approximately equal to the normalised gamma_h.

# III) Eigendecomposition of the Solution b (from I)
#
#      The solution can be written as:
#
#        b = V %*% (1/D) %*% t(V) %*% (gamma_h + lambda * beta * sum_d_delta),
#
#      where 1/D has diagonal entries 1 / (1 + lambda * eta_i), and the
#      eigenvalues of N are ordered as:
#
#        eta_1 = O(1),  eta_2 = O(delta^2),  eta_3 = ... = eta_L = 0.
#
#      Note that the eigenvalues of M = I + lambda * N are 1 + lambda * eta_i,
#      which invert to 1 / (1 + lambda * eta_i) under M^{-1}.
#
#      Defining the coordinate vectors of the projections onto the eigenbasis:
#
#        G1 = (t(V1) %*% gamma_h,      t(V2) %*% gamma_h,      0, ..., 0),
#        G2 = (t(V1) %*% sum_d_delta,  t(V2) %*% sum_d_delta,  0, ..., 0),
#
#      the solution decomposes as:
#
#        b = V %*% (1/D) %*% (G1 + lambda * beta * G2).
#
#      The trailing zero entries in G1 and G2 follow from the orthogonality of
#      the eigenvectors V[, 3], ..., V[, L] to both gamma_h and sum_d_delta.

# IV) Expanded Form of b
#
#      Expanding the eigendecomposition explicitly, the solution b reads:
#
#        b =   V1 * (t(V1) %*% gamma_h      / (1 + lambda * eta_1))
#            + V2 * (t(V2) %*% gamma_h      / (1 + lambda * eta_2))
#            + lambda * beta
#            * (  V1 * (t(V1) %*% sum_d_delta / (1 + lambda * eta_1))
#               + V2 * (t(V2) %*% sum_d_delta / (1 + lambda * eta_2)) ).
#
#      That is, b is expressed as a weighted sum of the two leading eigenvectors
#      V1 and V2, with weights determined by their inner products with gamma_h
#      and sum_d_delta, respectively, scaled by the inverse eigenvalue factors.

# V)  Simplification and the Effect of delta
#
#     When delta is small, V1 ≈ gamma_h / ||gamma_h||, and hence:
#
#       V1 * (t(V1) %*% gamma_h) ≈ gamma_h.
#
#     Note that the sign of V1 is arbitrary (eigenvector signs are
#     undetermined), so it cancels in this product. Additionally,
#
#       V2 * (t(V2) %*% gamma_h) = O(delta) * V2.
#
#     Moreover, sum_d_delta = O(1) * gamma_h - perturbation_vec, so that:
#
#       V1 * (t(V1) %*% sum_d_delta) ≈ O(1) * V1,
#
#     assuming perturbation_vec is of small size O(delta). Finally,
#
#       V2 * (t(V2) %*% sum_d_delta) = O(delta) * V2,
#
#     since both t(V2) %*% gamma_h = O(delta) and
#     t(V2) %*% perturbation_vec = O(delta).

# VI) Approximate Form of b (combining IV and V)
#
#     Substituting the approximations from V) into the expanded form IV),
#     the solution simplifies to:
#
#       b =   [1 / (1 + lambda * eta_1) * O(1)    ] * V1
#           + [1 / (1 + lambda * eta_2) * O(delta)] * V2
#           + lambda * beta
#           * (  [1 / (1 + lambda * eta_1) * O(1)    ] * V1
#              + [1 / (1 + lambda * eta_2) * O(delta)] * V2 )
#
#         =: F1 + lambda * beta * F2,
#
#     where F1 captures the unpenalised direction (approximately aligned with
#     V1) and F2 captures the penalised perturbation direction. Note that in general 
#     F1 and F2 are not collinear, i.e., they point in different directions in the 
#     plane spanned by V1 and V2.

# VII) Interplay of lambda and beta
#
#      Effect of the product lambda * beta on the decomposition b = F1 + lambda * beta * F2:
#
#        - beta = 0:               b = F1.
#        - lambda * beta = O(1):   b = F1 + O(1) * F2.
#        - lambda * beta >> 1:     b ≈ lambda * beta * F2.
#
#      Summary: the product lambda * beta controls the relative weight between
#      the directions F1 and F2. As shown next in VIII), the internal compositions 
#      of F1 and F2 themselves depend on lambda and delta: both F1 and F2 interpolate
#      (in different ways) between V1 and V2 as these parameters vary.

# VIII) Interplay of lambda, beta and delta
#
#      Effect of lambda on the eigenvalue scaling factors
#      1 / (1 + lambda * eta_i):
#
#        - lambda = O(1):         1 / (1 + lambda * eta_1) = O(1),
#                                 1 / (1 + lambda * eta_2) ≈ 1.
#
#        - lambda = O(1/delta):   1 / (1 + lambda * eta_1) = O(delta),
#                                 1 / (1 + lambda * eta_2) ≈ 1.
#
#        - lambda = O(1/delta^2): 1 / (1 + lambda * eta_1) = O(delta^2),
#                                 1 / (1 + lambda * eta_2) = O(1).
#
#      Implied structure of F1, F2, and b across regimes:
#
#      [a] lambda = O(1)  —  moderate regularisation:
#
#            F1 ≈ O(1) * V1 + O(delta) * V2,
#            F2 ≈ O(1) * V1 + O(delta) * V2.
#
#          Both F1 and F2 are dominated by V1 (the original AR(1) direction).
#
#          - If lambda * beta = O(1): b = F1 + O(1) * F2. If beta is chosen
#            such that the dominating V1 components of F1 and F2 cancel exactly, then
#            b = O(delta) * V2 (full decoupling direction) and b is subject to 
#            strong zero-shrinkage. Otherwise, b is a mix of
#            V1 and V2 (and subject to less or none zero-shrinkage).
#
#          - If lambda * beta >> 1: b ≈ lambda * beta * F2 ≈ O(beta) * V1.
#
#          - If beta = o(1):        b ≈ F1 ≈ O(1) * V1 (≈ O(1) * gamma_h)
#         
#          - Asymptotically, as |beta| -> Inf, b is proportional to F2 ≈ O(1) * V1.
#
#      [b] lambda = O(1/delta)  —  large regularisation:
#
#            F1 = O(delta) * V1 + O(delta) * V2,
#            F2 = O(delta) * V1 + O(delta) * V2.
#
#          Both F1 and F2 are of order O(delta), representing an intermediate
#          mix of V1 and V2. 

#          - The product lambda * beta modulates between F1 and F2 directions. 
#
#          - For beta = O(delta), beta * lambda = O(1) and 
#             b = F1 + lambda * beta * F2 
#            remains an O(delta)-weighted combination of V1 and V2, implying that 
#            b is subject to strong zero-shrinkage. 
#
#          - Larger or smaller beta affect this shrinkage as well as the relative 
#            weight assigned to F2 (relative to F1).
#
#          - Asymptotically, as |beta| -> Inf, b is proportional to F2, a mix 
#            of V1 and V2. 
#
#          - Pure V1 or V2 solutions arise for special cases of beta (not 
#            asymptotically |beta| -> Inf).

#
#      [c] lambda = O(1/delta^2)  —  very strong regularisation:
#
#            F1 = O(delta^2) * V1 + O(delta) * V2,
#            F2 = O(delta^2) * V1 + O(delta) * V2.
#
#          Both F1 and F2 are dominated by V2 (the full decoupling direction),
#          with all components subject to strong or very strong shrinkage towards zero.
#
#          - If beta = O(delta^2), so that lambda * beta = O(1):
#            b = F1 + O(1) * F2. If beta is chosen such that the dominant V2
#            components of F1 and F2 (each of order O(delta)) cancel exactly, then
#
#               b = O(delta^2) * V1,
#
#            i.e., b aligns with the AR(1) direction, though subject to very
#            strong shrinkage towards zero.
#
#          - If the V2 components do not cancel exactly: b is a mixture of
#            V1 and V2, still subject to strong or very strong shrinkage.
#
#          - If lambda * beta >> 1: b ≈ lambda * beta * F2 ∝ V2, with the
#            magnitude of b controlled by lambda * beta.
#
#          - If lambda * beta = o(1): b ≈ F1 = O(delta) * V2, i.e., b is
#            dominated by V2 and subject to large (but not necessarily very 
#            large) shrinkage.
#
#          - Asymptotically, as |beta| -> Inf, b is proportional to +/- V2.
#
#          - Mixtures of V1 and V2, including the pure V1 direction, arise when 
#            beta = O(delta^2) is very small in absolute value.  
#
#
#      Summary of cases [a], [b] and [c]: as lambda increases through the regimes
#
#        O(1)  ->  O(1/delta)  ->  O(1/delta^2),
#
#      the weights assigned to V1 and V2 within F1 and F2 transition as:
#
#        O(1)*V1 + O(delta)*V2  ->  O(delta)*V1 + O(delta)*V2  ->  O(delta^2)*V1 + O(delta)*V2.
#
#      While either V1 or V2 may dominate in F1 and F2, depending on lambda, the 
#      slope parameter beta can always be tuned to cancel the dominant term, so that
#      b ultimately aligns with either V1, or V2, or any (PCS-optimal) 
#      intermediate combination.
#      Throughout, the product lambda * beta governs the relative weight
#      between the F1 and F2 directions in the expression for b.




#───────────────────────────────────────────────────────────────────────────────
# 5.4 Analysis: Case Studies (under construction)
#───────────────────────────────────────────────────────────────────────────────

# Recall that 
#   tV2 %*% gammah        = O(delta)
#   eta_2                 = O(delta^2)  [second non-zero eigenvalue of N]
#   tV2 %*% sum_d_delta   = O(delta)
# See exercise 5.2.2.

#───────────────────────────────────────────────────────────────────────────────
# 5.4.1 Case A: Vanishing Slope (beta = 0) (under construction)
#───────────────────────────────────────────────────────────────────────────────


# With beta = 0 the G2 term drops out, leaving:
#   b = V1 * (tV1 %*% gammah / (1 + lambda * eta_1))
#     + V2 * (tV2 %*% gammah / (1 + lambda * eta_2))

# A.1) lambda = O(1),  delta ~ 0  (very small perturbation)
#   eta_2 = O(delta^2) => 1 / (1 + lambda * eta_2) ~ 1
#   tV2 %*% gammah = O(delta) => the V2 term vanishes
#   => b ~ V1 * (tV1 %*% gammah / (1 + lambda * eta_1)) = O(1) * gammah
#      (V1 aligns with gammah as delta -> 0)

# A.2) lambda = O(1/delta^2)  (very large regularization)
#   1 / (1 + lambda * eta_1) = O(delta^2),   1 / (1 + lambda * eta_2) = O(1)
#   tV1 %*% gammah = O(1),                   tV2 %*% gammah = O(delta)
#   => V1 term = O(delta^2),                 V2 term = O(delta)
#   => b ~ V2 * O(delta)  (b aligns with V2 and shrinks to zero with delta)
#   Note that even stronger regularization (larger lambda) will keep the same 
#   direction but shrink b even stronger to zero (recall that the constraint 
#   imposed by beta = 0 is a misspecification).

# A.3) lambda = O(1/delta)  (large regularization but not as large as in A.2)
#   1 / (1 + lambda * eta_1) = O(delta),     1 / (1 + lambda * eta_2) = O(1)
#   tV1 %*% gammah = O(1),                   tV2 %*% gammah = O(delta)
#   => V1 term = O(delta),                   V2 term = O(delta)
#   => b is an O(delta) linear combination of V1 and V2



#───────────────────────────────────────────────────────────────────────────────
# 5.4.2 Case B: Non-Vanishing Slope (beta != 0) (under construction)
#───────────────────────────────────────────────────────────────────────────────

# With beta != 0 the additional G2 contribution must be considered:
#   lambda * beta * [ V1 * (tV1 %*% sum_d_delta / (1 + lambda * eta_1))
#                   + V2 * (tV2 %*% sum_d_delta / (1 + lambda * eta_2)) ]
# Note: sum_d_delta = O(1) * gammah + perturbation, with perturbation = O(delta),
# so G2 has the same order structure as G1.

# B.1) lambda * beta ~ 0
#   The G2 contribution vanishes; the analysis reduces to Case A.

# B.2) lambda * beta = O(1)
#
#   B.2.1) lambda ~ 0,  beta very large
#     G1 part (from A.1): b_G1 ~ O(1) * gammah
#     G2 part: tV2 %*% sum_d_delta / (1 + lambda * eta_2) = O(delta), negligible
#     => b aligns with V1 ~ gammah
#
#   B.2.2) lambda = O(1/delta^2),  beta = O(delta^2)
#     G1 part (from A.2): b_G1 ~ V2 * O(delta)
#     G2 part: same order structure as G1 => also aligns with V2
#     => b = O((1+beta * lambda) * delta) * V2 = O(delta+beta/delta) * V2 = 
#     O(delta) * V2.  
#     - Shrinks to zero since beta = O(delta^2) when lambda * beta = O(1). 
#
#   B.2.3) lambda = O(1)
#     Similar to A.1: b proportional to V1 ~ O(1) * gammah
#
#   B.2.4) lambda = O(delta),  beta = O(1/delta)
#     Similar to A.3: b = V1 * O(delta) + V2 * O(delta)  (shrinks to zero with delta)


#───────────────────────────────────────────────────────────────────────────────
# 5.4.3 Case C: Very Large Slope (beta >> 1) (under construction)
#───────────────────────────────────────────────────────────────────────────────

# C.1) lambda such that lambda * beta = O(1)  =>  see Case B.2
# C.2) lambda = O(1)                          =>  see Case A.1
# C.3) lambda = O(1/delta)                    =>  see Case A.3
# C.4) lambda = O(1/delta^2)                  =>  see Case A.2


#───────────────────────────────────────────────────────────────────────────────
# 5.5 Summary
#───────────────────────────────────────────────────────────────────────────────

# The interplay of delta, lambda, and beta is complex. However, a unifying structural
# insight is that the PCS predictor b can always be expressed as a linear combination
# of just two vectors: gammah and the perturbation vector (or, equivalently, V1 and V2).
# This follows directly from the rank-2 structure of the equation system (matrix N) 
# established above.
#
# The coefficients of this linear combination are determined jointly by delta, lambda,
# and beta. Importantly, not all coefficient pairs are attainable: the feasible set is
# restricted to solutions that are optimal with respect to the regularized criterion
# (Eq. 49 in Wildi (2026)), so the reachable region within the (V1, V2)-plane is
# constrained accordingly.






# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. The AR(1) forecast problem is self-similar and one-dimensional: the PCS
#    problem is impossible and the constraint system is rank-deficient.
#
# 2. Perturbing the DGP expands the column space of the PCS constraint system,
#    resolving at least partially the rank deficiency.
#
# 3. Type and shape of the perturbation are main determinants of look-ahead behaviour:
#      - Single-lag vs. multi-lag perturbations induce different constraint structures.
#      - AR(1) vs. AR(2) perturbations yield qualitatively different decoupling
#        directions V2, and hence different look-ahead profiles.
#      - Other DGPs would induce their own specific perturbation pattern.
#
# 4. Three perturbation types were analysed:
#      i.  delta-type (lag 0 only): structurally analogous to an ARMA(1,1)
#          process with a small MA(1) coefficient when delta is small, see 
#          exercise 2.
#      ii. AR(1)-type (all lags): the perturbation spreads across all lags
#          according to a slightly modified AR(1) parameter, see exercise 3.
#      iii.AR(2)-type (all lags, periodic): this perturbation differs from the 
#          original AR(1) DGP in shape (periodicity), introducing a more severe
#          misspecification, see exercise 4.
#
# 5. In all cases considered, the rank of the constraint system increases from
#    1 to 2. The PCS solution lies in the space spanned by the original AR(1)
#    direction and the perturbation vector — the first two eigenvectors of the
#    constraint matrix corresponding to its two non-vanishing eigenvalues.
#    The full decoupling direction V2 is generated respectively by: 
#    the unit vector e1 (delta-type; exercise 2), the modified AR(1) vector 
#    (exercise 3), or the AR(2) vector (exercise 4).
#
# 6. When the perturbation is weakly misspecified (as in Exercise 3),
#    meaningful look-ahead PCS predictors can be obtained across a wide
#    range of regularization weights lambda — both large (strong) and
#    small (weak). When the misspecification is more severe (as in
#    Exercise 4), large lambda amplifies the distortion by over-weighting
#    the misspecified perturbed constraints, potentially yielding unusable
#    predictors. In such cases, small to moderate lambda is recommended,
#    so that the target correlation retains sufficient influence in the
#    optimisation.
#
# 7. Navigating this two-dimensional space via lambda and beta — rather than
#    by directly weighting V1 and V2 — ensures optimality: not all linear
#    combinations of V1 and V2 correspond to optimal PCS predictors.
#
# 8. In principle, multiple perturbations could further increase the rank of
#    the constraint system. However, this risks placing excessive weight on
#    the constraint objectives at the expense of the target correlation
#    CCF(h), potentially degrading forecast performance.
#
# 9. In principle, multiple look ahead predictors could be derived from 
#    different perturbations and aggregated into a combined look ahead design.
# ─────────────────────────────────────────────────────────────────────────────

# Note: A closed-form analogue to the regularized perturbed PCS function 
# PCS_perturbation_func() is not provided yet.





