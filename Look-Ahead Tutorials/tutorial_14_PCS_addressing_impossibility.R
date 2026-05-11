# ══════════════════════════════════════════════════════════════════════════════
# TUTORIAL 14 — IMPOSSIBILITY: 
#               HANDLING THE HARDEST LOOK AHEAD FORECAST PROBLEM
# ══════════════════════════════════════════════════════════════════════════════


# ── BACKGROUND: DFP AND PCS FORECASTING ───────────────────────────────────────

# The Decouple From Present (DFP) and Peak Correlation Shifting (PCS) approaches
# are designed to look ahead of the classical MSE predictor in difficult forecasting
# problems where the MSE predictor is "stuck at the present": its cross-correlation
# function (CCF) with the target x_{t+h} peaks at lag k=0 for any horizon h.
#
# PCS attempts to shift the CCF peak away from k=0, ideally placing it at k=h, while
# maintaining maximal correlation with x_{t+h}. This is achieved by constraining
# the CCF path over lags k=0,...,h. Three constraint types are considered:
#
#   Type I   PCS: CCF(k) - CCF(k-1) = beta_k >= 0,  for k = 1, ..., h
#   Type II  PCS: CCF(h) - CCF(h-1) = beta    >= 0
#   Type III PCS: CCF(h) - CCF(0)   = beta    >= 0
#
# For further details, see Tutorial 13.
#
# Terminology:
#   - Feasible:    The constraint system is exactly solvable and the resulting
#                  peak CCF(h) > 0.
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
# room is eventually left for negative lags which PCS or DFP do not explicitly 
# address.
#
# Denoting the h-step MSE predictor in MA-form as gamma_h (with gamma_0 being
# the nowcast, i.e., the original Wold decomposition of the DGP), the Yule-Walker
# equations imply:
#
#   gamma_h = a1^k * gamma_{h+k},  for any h, k >= 0.
#
# This property is called self-similarity: all h-step MSE predictors are
# proportional to gamma_0, differing only by the scalar factor a1^h.
#
# Self-similarity forces the CCF of any predictor b to satisfy:
#
#   CCF(k) = (b' %*% gamma_k) / (||b|| * ||gamma_0||) = a1^k * CCF(0).
#
# For a1 > 0, the CCF decays monotonically and exponentially from its peak at
# k=0. This pattern is rigidly enforced by the DGP on every predictor b via the
# Yule-Walker equations. Consequently, reshaping the CCF according to any of the
# PCS constraint types above is generally impossible (unless one merely replicates
# the original process).
#
# However, for negative lags k=-1,-2,...
#
#   CCF(k) = (b[-k+(1:L)]' %*% gamma_0) / (||b[-k+(1:L)]|| * ||gamma_0||) 
#
# This expression depends on the predictor b and the (negative) lag k and hence
# can be controlled somehow.

#───────────────────────────────────────────────────────────────────────────────

# ── RANK-ONE CONSTRAINT SYSTEM AND ITS CONSEQUENCES ───────────────────────────

# Due to self-similarity, the PCS constraint system has rank one for the AR(1)
# DGP. While it is possible to impose b' * (gamma_k - gamma_{k-1}) = beta for a
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
# from Wildi (2026), Equation 46 (Appendix E), with the solution given by
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

# ── GEOMETRY OF THE SOLUTION: EIGENVECTORS V1 AND V2 ──────────────────────────

# In the rank-two setting, the PCS solution (Wildi (2026), Equation 49) lives in
# a two-dimensional subspace spanned by the two leading eigenvectors V1 and V2
# of the matrix M, which encodes the PCS constraint system.
#
# When delta is small:
#   - V1 aligns closely with gamma_0 (up to sign), capturing the original AR(1)
#     DGP direction.
#   - V2 is orthogonal to gamma_0 and captures the component of the perturbation
#     lying outside the AR(1) subspace. This orthogonality — V2 %*% gamma_0 = 0
#     — implies that V2 fully decouples from the present.
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
#     or V1), and
#   - The full decoupling vector V2, obtained from perturbation.
#
# Full decoupling (b ∝ V2) is a theoretically valid strategy for generating
# look-ahead behaviour, but it is often too extreme in practice, potentially
# leading to sign inversion and uninterpretable predictions. The weight lambda2
# on V2 — in combination with lambda1, both governed by the PCS hyperparameters 
# lambda and beta — enables controlled, partial decoupling: the predictor 
# departs from gamma_0 toward full decoupling in a graduated and controllable  
# manner to generate look ahead behaviour.
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
# tutorial is V1 ∝ gamma_0 ∝ gamma_h: the shared MSE predictor direction,
# valid in the limit delta -> 0. The distinguishing element is V2, the fully
# decoupling direction, which is specific to the perturbation chosen.
#
# Excessively large lambda over-weights V2 at the expense of V1, potentially
# yielding unusable predictors. To see why, note that if b ≈ V2, then:
#
#   0 = b' * gamma_0 = b' * gamma_h 
#
# since gamma_0 ∝ gamma_h under the AR(1) structure. That is, the target
# correlation vanishes entirely.
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
# clean decomposition of the PCS predictor into an MSE component (V1) and a
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
# Type III PCS at longer horizons (h > 1).
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
#             parameter b1 perturbs only the lag-0 weight of gamma_0.
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
# exist that are left-shifted relative to the MSE benchmark while simultaneously
# maximising tracking accuracy. The CCF peak may be fixed at the origin (k = 0),
# yet effective anticipatory behaviour can still be recovered by reshaping the
# LEFT tail of the CCF. This tutorial demonstrates the effectiveness of
# perturbation approaches in achieving this dual objective.

#
# ─────────────────────────────────────────────────────────────────────────────
# II) Left Tail of the CCF
# ─────────────────────────────────────────────────────────────────────────────
#
# The AR(1) structure of the "hardest forecast problem" examined in this 
# tutorial renders it impossible to shift the CCF peak to the right.
# More precisely, no linear predictor can alter the exponentially decaying
# profile of the CCF at positive lags. In this sense, the AR(1) forecast
# problem can be considered as the hardest look-ahead problem.
#
#---------------------------------------------------------------------------
# How to Address Look-Ahead Behaviour When the CCF Peak Cannot Be Shifted?
#---------------------------------------------------------------------------
#
# Although the right tail of the CCF is entirely determined by the AR(1)
# structure and is therefore immutable, the LEFT tail remains accessible to
# manipulation via the choice of predictor. The following observation,
# carried over from Exercise 3.5, elaborates on this point:
#
#   - Only the left tail of the CCF is amenable to modification. Whereas the MSE
#     predictor yields a symmetric CCF, the PCS predictor becomes progressively
#     more asymmetric as beta increases. Effective look-ahead behaviour is thus
#     achieved by skewing the CCF rightward — that is, by down-weighting the
#     contribution of negative lags.
#
# By introducing suitable perturbations to the original DGP (which may be made 
# arbitrarily small), the left tail can be shaped to induce look-ahead behaviour. 
# This offers a principled resolution to an otherwise impossible problem: rather
# than attempting to shift the CCF peak rightwards, one instead recovers
# effective lead behaviour by redistributing CCF mass away from negative lags.
#
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
# (2026), Appendix E).
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
lambda <- 10000

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
# see Wildi (2026) Appendix E.
# Verify: residual should vanish.
max(abs(M - diag(rep(1, L)) - lambda * N))
# Note:  (gamma_k - gamma_{k-1}) represents the PCS constraint at lag k: we want 
# b' * (gamma_k - gamma_{k-1}) = beta.

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
eigenM$values
ts.plot(V[, 1], main = "Leading eigenvector of M")

# Since V is orthogonal and gamma_sol is proportional to V[,1], all projections
# t(V[,k]) %*% gamma_sol vanish for k > 1.

# Verify: the projection onto V[,k] for k > 1 should vanish.
k <- 2
V[, k] %*% gamma_sol        # Should be (near) zero for k > 1.
t(V)[k, ] %*% gamma_sol     # Equivalent formulation.

# Full projection vector: only the first element should be non-zero.
t(V) %*% gamma_sol

# Intermediate vector g = diag(1/d) %*% t(V) %*% gamma_sol:
# only its first element is non-zero, so b = V %*% g = g[1] * V[,1].
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Verify: V %*% g equals g[1] * V[,1] up to numerical precision.
max(abs(V %*% g - g[1] * V[, 1]))

# Conclusion: b = solve(M) %*% gamma_sol is proportional to V[,1], the leading
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
#   b' * (gamma_k - gamma_{k-1}) = beta > 0,  for k = 1, ..., h.
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
# that gamma_sol is proportional to V[,1], the leading eigenvector of M. As a
# result, b = solve(M) %*% gamma_sol is proportional to V[,1], which is itself
# AR(1). This holds irrespective of the choice of lambda > 0 and beta can only 
# trigger the sign of b. 
#
# Consequently, CCF(k) = a1^k * CCF(0) for any predictor b derived from the PCS
# criterion: the CCF always peaks at k=0 and decays geometrically, regardless of
# the hyperparameter settings.




# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: INCREASING THE RANK — A PERTURBATION-BASED APPROACH
# ════════════════════════════════════════════════════════════════════
# A single perturbation of magnitude delta is introduced at lag 0 of the
# Wold decomposition. Provided gamma_0 enters the PCS constraint system,
# this expands the rank of the constraint system from 1 to 2 (otherwise the rank 
# is stuck at one).


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

# delta_1 lies in the span of {gamma_0, perturbation_vec}: both components
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

# Consequently, b = V %*% g = g[1]*V[,1] + g[2]*V[,2].
# Verify: the residual should vanish.
max(abs(V %*% g - g[1] * V[, 1] - g[2] * V[, 2]))

# The PCS predictor b is a linear combination of V[,1] and V[,2], or
# equivalently, of gamma_0 and perturbation_vec.
b <- V %*% diag(1 / eigenM$values) %*% t(V) %*% gamma_sol

ts.plot(b, main = paste("PCS Predictor, lambda=",lambda,", beta=",beta," : 
                        No longer AR(1)",sep=""))

# Confirm departure from AR(1): lag 0 is affected by the perturbation.
b[2:L] / b[1:(L - 1)]

# Verify the decomposition of b into {gamma_0, perturbation_vec}: perfect fit.
summary(lm(b ~ gamma0 + perturbation_vec[1:L] - 1))


#───────────────────────────────────────────────────────────────────────────────
# 2.3 Exploring the Rank-Two System Under Strong Regularisation
#───────────────────────────────────────────────────────────────────────────────
# The PCS predictor lies in the span of V[,1] and
# V[,2], with coefficients governed by beta and lambda. As beta varies, the
# predictor traces a path between two limiting directions:
#
#   beta → -∞ :  predictor aligns with  -V[,2], if lambda is large
#   beta → +∞ :  predictor aligns with  +V[,2], if lambda is large
#
# If lambda is small (which is not the case here), the limiting directions 
# are determined by +/- V1. instead.

# Regularisation weight (strong regularisation regime).
lambda <- 5000000

# Construct a manual grid of beta values that spans the two limiting directions
# and resolves the transition region near the tipping point.
#
# The values are scaled by 1/lambda so that the effective perturbation
# beta * lambda remains on a meaningful scale.
#
beta_vec <- c(-5.000e+05, 0, 2, 2.5, 2.7, 2.8, 2.9, 3, 4, 5) / lambda

# Alternatively, PCS_func() can generate a beta grid automatically for a given
# lambda, concentrating points near the tipping point where the PCS design is
# most sensitive to beta (and may become near-singular). Near singularity,
# arbitrarily small changes in beta can produce large changes in the predictor,
# so adequate resolution in this region is important.
#
# The automatic grid depends only on lambda, not on beta itself. We therefore
# call PCS_func() with an arbitrary beta value solely to extract the grid.

# Arbitrary beta used only to trigger the grid computation (value is irrelevant).
beta <- 0

# Call PCS_func() to obtain the automatically generated beta grid.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

# Extract the beta grid from the PCS object.
beta_vec <- PCS_obj$beta_vec

# Either the manual grid defined above or the automatic grid extracted here
# may be used in subsequent computations (the manually computed has slightly 
# better resolution).


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

# The following plots will be used several times. We therefore define a specific
# Plot function for later usage:
  

plot_func<-function()
{
  colo <- rainbow(ncol(filter_mat))
  par(mfrow = c(2, 2))
  
  # Scale all filters to unit energy for visual comparability.
  mplot <- scale(filter_mat, center = FALSE, scale = TRUE)
  
  # Verify filter energies after scaling (should all equal 1).
  apply(mplot^2, 2, sum)
  
  # ── Panel 1: Scaled predictor coefficient profiles ────────────────────────────
  plot(mplot[, 1],
       main = "Scaled Predictors", axes = FALSE, type = "l",
       xlab = "Lags", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(mplot), max(mplot)))
  mtext(colnames(mplot)[1], col = colo[1], line = -1)
  
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
  lines(mplot[, 2], col = colo[2])   # Redraw second filter on top for visibility.
  
  axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
       labels = c(0, (1:(nrow(mplot) / 10)) * 10))
  axis(2)
  box()
  
  # ── Panel 2: CCF against xi (Wold decomposition) ──────────────────────────────
  # For each predictor, compute the CCF against xi at lags 0, 1, ..., h.
  # The dashed vertical line marks the target horizon h; the horizontal line
  # marks zero correlation.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], xi)$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against xi", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)   # Vertical marker at target horizon h.
  abline(h = 0)                 # Zero-correlation reference line.
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  
  # ── Panel 3: CCF against V1 (leading eigenvector of M) ────────────────────────
  # V[,1] ≈ gamma_0 (up to sign) when delta is small, so this panel isolates
  # the gamma_0 contribution to the CCF.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], V[, 1])$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against V1", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)
  abline(h = 0)
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  
  # ── Panel 4: CCF against V2 (second eigenvector of M) ─────────────────────────
  # V[,2] is orthogonal to V[,1] ≈ gamma_0 and captures the full decoupling
  # direction introduced by the perturbation.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], V[, 2])$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against V2", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)
  abline(h = 0)
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  return(colo)
}

colo<-plot_func()


# ── Interpretation of Predictors ──────────────────────────────────────────────
#
# The predictor profiles displayed in Panel 1 illustrate the effect of the
# single-lag perturbation at lag 0. Because the perturbation affects only the
# lag-0 coefficient of xi, the PCS predictor b follows the standard AR(1)
# decay profile for all lags k >= 1, i.e., strictly greater than zero 
# (up to possible sign inversion). The perturbation provides a single degree
# of freedom — a fine-tuning adjustment at lag 0 only — without altering the
# structure of the predictor at any other lag.
#
# This limited flexibility is the key limitation of the single-lag perturbation:
# it expands the rank from 1 to 2, but the additional degree of freedom is
# confined entirely to lag 0. As a result, the look-ahead behaviour along the 
# full decoupling direction V2 (Panel 4) can shift the peak from k=0 to k=1, at 
# most. This effect is relatively modest compared to multi-lag perturbations, which
# reshape the predictor profile across all lags and thereby generate a richer
# decoupling effect, generating strong peak-shifts along their full decoupling 
# directions V2, see exercises 3 and 4 below.
#
# Under strong regularisation, the PCS predictors transition smoothly from -V2
# to +V2 as beta increases. The two extremes, ±V2, reflect configurations where
# the constraint is driven entirely by the decoupling direction V2, with V1
# (the target gamma_h direction) receiving zero weight. V2 is the primary
# enabler of the PCS constraints and is therefore emphasized when lambda is 
# large (as is the case here): for sufficiently large positive or negative beta, 
# the predictor collapses onto ±V2 and the contribution of V1 vanishes.
#
# Between these two extremes, the predictor traces a continuum of optimal linear
# combinations of V1 and V2, with the weights determined by the
# interplay between the target correlation (governed by V1) and the constraint
# satisfaction (governed by V2) for the specified values of lambda and beta.
#
# Note that we can reparametrize b as
# b = - lambda1 * V1 + lambda2 * V2, where lambda1,lambda2>0. The parameters 
# lambda1 and lambda2 are determined by lambda and beta: not all combinations of 
# lambda1 and lambda2 are optimal (solutions of the PCS criterion). The negative 
# sign - lambda1 * V1 of V1 is due to placing very strong emphasis on the 
# constraints relative to the target when lambda is very large. Sign inversion of V1 (equivalently, of gamma_0
# or xi) is the only mechanism by which a monotonically increasing CCF over
# lags k = 0, ..., h can be achieved under this configuration (see panel 2). 

# ── Interpretation of CCF Panels ──────────────────────────────────────────────
#
# The PCS predictor b is a linear combination of V[,1] and V[,2], and therefore
# also of gamma_0 and perturbation_vec.
#
# Since V[,1] and V[,2] are orthogonal, the total CCF decomposes additively:
#
#   CCF(b, xi) = CCF(b, V[,1]) + CCF(b, V[,2]).
#
# Panel 3 — CCF against V1:
#   Isolates the gamma_0 contribution to the CCF. This reflects the immutable,
#   fixed AR(1) profile: it is identical across all predictors b, regardless of
#   the choice of hyperparameters beta and lambda. It coincides with the CCF
#   profile in Panel 2 (CCF against xi), up to a possible sign change induced
#   by the constraints (when lambda is large). NOte that the sign is inverted when 
#   compared to panel 2 because of the negative sign of V1 in b, see above comment. 
#
# Panel 4 — CCF against V2:
#   Isolates the `full decoupling' direction (V2) introduced by the perturbation.
#   This component captures the idiosyncratic look-ahead effect that arises
#   exclusively from the extraneous perturbation. Unlike the V1 component, it
#   varies with the hyperparameters and depends directly on the type and
#   direction of the perturbation chosen. Different perturbations yield different
#   V2 vectors and therefore different look-ahead profiles in this panel, see 
#   exercises 3 and 4 below.


# ─────────────────────────────────────────────────────────────────────
# 2.5 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Same as Exercise 2.3 but with a medium-sized lambda:
# Medium regularization
lambda<-5
# Tipping points: the two extremes are -V[,1] and +V[,1]
# Note: in exercises 3 and 4, the new function PCS_perturbation_func() will 
# generate automatically a grid of relevant beta values.
beta_vec<-c(0.900000, 0.902000, 0.902500, 0.902750, 0.902850, 0.902940, 0.902945, 0.902975, 0.903000, 0.903050, 0.903250, 0.905500)/lambda

# Note:
# The PCS design exhibits marked sensitivity to the choice of beta: small
# changes in beta drive the transition from -V1 to +V1. This sensitivity
# reflects the near-singularity of the design as delta (the perturbation) shrinks toward zero —
# a fundamental trade-off inherent to the interpretability of the
# parameterisation. Specifically, as delta -> 0, V1 converges to gamma0 (the
# direction associated with the AR(1) process), while V2 aligns with the fully
# decoupling direction induced by the perturbation, rendering the design
# geometrically interpretable. Conversely, larger values of delta alleviate
# the near-singularity — reducing sensitivity to beta — but at the cost of
# a less structured and less interpretable geometry. This trade-off is
# deliberate: interpretability is here prioritised over numerical regularity.


# Instead of manually adjusting beta (which can be tedious) we can 
# rely on grid computed by PCS_func():

# Arbitrary beta used only to trigger the grid computation (value is irrelevant).
beta <- 0

# Call PCS_func() to obtain the automatically generated beta grid.
PCS_obj <- PCS_func(h, Delta, gamma_pcs_perturbated, L, beta, lambda)

# Extract the beta grid from the PCS object.
beta_vec <- PCS_obj$beta_vec

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

filter_mat<-scale(b_mat)
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta*lambda=",round(beta_vec*lambda,6))

# ─────────────────────────────────────────────────────────────────────
# 2.6 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors (Medium Regularisation) ──────────────────────
#
# The key difference relative to the strong regularisation case in Section 2.4
# is that the extremal predictors now align with +V1 and -V1 rather than ±V2.
# Under medium regularisation, the PCS constraints carry less weight relative
# to the target correlation objective. As a result, the target correlation
# component — represented by V1 ≈ gamma_h — dominates, and the solution space
# morphs smoothly between -V1 and +V1 as beta varies. The decoupling direction
# V2 still contributes intermediate solutions but no longer defines the extremes.
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
#    hyperparameters lambda and beta via the penalised criterion, and should not
#    be set independently.
#
# 2. Invariance to the perturbation size delta:
#    The PCS solution space is invariant to the magnitude of the perturbation
#    delta: the same family of predictors is obtained for any delta > 0 (or <0). 
#    A change in delta shifts the implicit mapping between (lambda, beta) and 
#    the weights (lambda1, lambda2), requiring recalibration of the 
#    hyperparameters, but leaves the geometric structure of the solution space 
#    unchanged.
#
# 3. Extending to Higher Rank via Additional Perturbations:
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
which(abs(eigenN$values) > 1e-10)

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

# Confirm that V[,1] decays geometrically (AR(1) direction) if delta is small.
V[2:L, 1] / V[1:(L - 1), 1]

# Since gamma_sol lies in the column space of V[,1:2], all projections
# t(V[,k]) %*% gamma_sol should vanish for k > 2.
t(V) %*% gamma_sol

# Intermediate vector g = diag(1/d) %*% t(V) %*% gamma_sol:
# only its first two elements are non-zero.
g <- diag(1 / eigenM$values) %*% t(V) %*% gamma_sol
g

# Verify: b = V %*% g = g[1]*V[,1] + g[2]*V[,2]. Residual should vanish.
max(abs(V %*% g - g[1] * V[, 1] - g[2] * V[, 2]))

# The PCS predictor b lies in the plane spanned by V[,1] and V[,2], or
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
# the two extremes ±V[,2] as beta varies, passing through -V[,1] at the
# tipping point where the V[,2] contribution vanishes.
# The weights on V[,1] and V[,2] are governed by lambda and beta.

# Strong regularisation weight.
lambda <- 5000000

# Note: the design exhibits marked sensitivity to beta — small perturbations
# in beta can induce substantial changes in the PCS solution — a consequence
# of the near-singularity of the PCS criterion as delta shrinks toward zero.
# This singularity is not merely a numerical inconvenience; it is intimately
# tied to interpretability: as delta -> 0, the first eigenvector V1 converges
# to gamma0 (the AR(1) autocovariance direction), while V2 aligns with the
# full decoupling direction induced by the perturbation. Small delta thus
# sharpens the geometric separation between these two directions, at the cost
# of an increasingly ill-conditioned optimisation landscape. This trade-off is
# deliberate: interpretability is here prioritised over numerical stability.

if (F)
{
# Manual grid
  beta_vec <- c(0 ,0.5,0.8,0.83,0.85,0.87,0.88,0.9,0.95,0.97,1,1.1,10) / lambda
}

# Instead of cumbersome manual tuning of beta as in exercise 2, PCS_perturbation_func()
# automatically returns a grid of beta values centred on the tipping point
# — where the sensitivity of the PCS solution with respect to beta is
# highest. Any initial beta may be supplied; the function locates the
# tipping point internally and constructs a symmetric grid around it.
#
# The asymptotic behaviour of the grid tails depends on lambda:
#   - Large lambda: the perturbed constraint system dominates, and the
#     left and right tails converge to -V[,2] and +V[,2] respectively,
#     since V2 is determined by the perturbation.
#   - Small lambda: the constraints are effectively down-weighted and the
#     target correlation dominates, causing the tails to converge to
#     -V[,1] and +V[,1] respectively, since V1 aligns with the MSE
#     predictor direction gamma_h.

# The grid is independent of beta: any value can be supplied
beta<-0.

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)

# Grid of beta values 
beta_vec<-PCS_obj$beta_vec

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

head(scale(filter_mat))
# ─────────────────────────────────────────────────────────────────────
# 3.4 Plots
# ─────────────────────────────────────────────────────────────────────

colo<-plot_func()

# ── Interpretation of Predictors: Full-Lag AR(1) Perturbation ─────────────────
#
# In contrast to Exercise 2, where the perturbation was confined to lag 0, the
# full-lag AR(1) perturbation affects every coefficient of the PCS predictor b.
# This richer perturbation structure expands the solution space more broadly,
# enabling, among other things, a seemingly cyclical coefficient profile — a
# notably non-trivial outcome in an inherently aperiodic framework where the
# underlying Wold coefficients decay monotonically.
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
# and beta.
#
# The PCS design exhibits sensitivity to the choice of beta when delta is small. 
# Small delta allow a better geometric interpretation. 
# Specifically, as delta -> 0, V1 converges to gamma0 (the direction
# associated with the AR(1) process), while V2 aligns with the fully decoupling
# direction. For larger delta, the design becomes less singular (less sensitive 
# to beta) but also less interpretable.

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
#    CCF_MSE(0) = 0. This is a consequence of full decoupling along V2:
#    the MSE predictor has no instantaneous correlation with the target
#    at lag 0 in this direction.
# 2. The left-shift of the CCF peak of the PCS predictors along the decoupling 
#    direction V2 can exceed the forecast horizon h. 
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
#     reflecting the fundamental difficulty of the AR(1) forecasting problem.
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
# Compute the empirical cross-correlation function (CCF) between the MSE
# predictor output (column 1) and the selected (leading) PCS predictor output.
#
# Key observations:
#   - As beta increases, the empirical CCF becomes increasingly right-skewed,
#     reflecting a growing lead of the PCS predictor relative to the MSE predictor.
#
#   - The right tail of the CCF (lag > 0) always follows the AR(1) decay:
#     b' * gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
#     consequence of the Yule-Walker equations and holds for any linear predictor b.
#     No non-zero predictor can alter this decay shape.
#
#   - Consequently, shifting the CCF peak strictly to the right of lag 0 is
#     impossible under the AR(1) DGP (see Exercise 1).
#
#   - Only the left tail of the CCF is amenable to modification in the AR1) case. 
#     Whereas the MSE
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





# ─────────────────────────────────────────────────────────────────────
# 3.6 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Lambda is fixed at a moderate regularization strength. 

lambda<-5

# Note: the design exhibits marked sensitivity to beta when delta is small.
if (F)
{
  # Manual grid
  beta_vec<-c(4.25,4.251,4.2513,4.2516,4.2518,4.252,4.2521,4.2522,4.2523,4.2524,4.2525,4.253,4.26,4.27)/lambda
}

# Instead of cumbersome manual tuning of beta, PCS_perturbation_func()
# automatically returns a grid of beta values centred on the tipping point
# — where the sensitivity of the PCS solution with respect to beta is
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

head(scale(filter_mat))
# ─────────────────────────────────────────────────────────────────────
# 3.7 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V1 and +V1, passing through V2 at an intermediate tipping point.

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
#   - The right tail of the CCF (lag > 0) always follows the AR(1) decay:
#     b' * gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
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
#   b = V %*% g = g[1] * V[,1] + g[2] * V[,2]
# Verification: the following maximum absolute deviation should be (near) zero
abs(max(V %*% g - g[1] * V[, 1] - g[2] * V[, 2]))

# Conclusion: the PCS predictor b lies in the space spanned by V[,1] and V[,2]
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
# -V[,2] and +V[,2], with intermediate solutions of the form
# -V[,1] + lambda1 * V[,2], where lambda1 depends continuously on beta.

# In this example, the minus sign -V[,1] on V1 is due to the large lambda: emphasizing strongly 
# an increasing CCF(k), from k=0,...,h, through the constraints, is only possible 
# through sign inversion of gamma_0, i.e., -V1.

# When the perturbation conditions the constraints into a misspecified design 
# (here AR(2) instead of AR(1)), emphasizing the constraints at the detriment of the target correlation through a large lambda might be problematic.
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

head(scale(filter_mat))
# ─────────────────────────────────────────────────────────────────────
# 4.4 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V2 and +V2, passing through -V1 at an intermediate tipping point.
#
# Note: V1 (corresponding to beta*lambda around 2.78 in the first panel below) 
# appears with an inverted sign -V1 because the monotonically increasing
# CCF required by the constraints cannot be achieved without reversing the
# sign of the DGP direction encoded in V1. The very large lambda selected
# here amplifies this effect, driving the solution toward the sign-inverted
# direction as the constraint penalty dominates the optimisation objective.
#
# However, in contrast to Exercise 3, the CCF against xi (second panel)
# is either near zero or negative throughout. This indicates that placing
# excessive weight on the perturbed AR(2) constraints via large lambda
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
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# first eigenvector V[,1] with combinations V[,2]+lambda1*V[,1] in between, where 
# lambda1 depends on beta.



# Medium regularization
lambda<-5

# Note: the design exhibits marked sensitivity to beta when delta is small.
if (F)
{
  # Manual grid
  beta_vec<-c(0.9,0.9026,0.9027,0.90275,0.9028,0.90282,0.90285,0.90286,0.90288,0.9029,0.90292,0.90295,0.903,0.904,0.905)/lambda
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

head(scale(filter_mat))


# ─────────────────────────────────────────────────────────────────────
# 4.7 Plots
# ─────────────────────────────────────────────────────────────────────

# In contrast to Section 4.4 (strong regularization, large lambda), a weak
# or moderate regularization (small to medium lambda) assigns meaningful
# weight to the target correlation objective, thereby avoiding the overt
# misspecification induced by the AR(2) perturbation — provided beta does
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

# Select PCS designs that do not exhibit sign inversion (columns 1:5): all
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
#   - The right tail of the CCF (lag > 0) always follows the AR(1) decay:
#     b' * gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
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
# 3. While the magnitude of the perturbation is irrelevant
#    (it can be scaled arbitrarily through delta), its type and shape are consequential:
#      - Single-lag vs. multi-lag perturbations induce different constraint structures.
#      - AR(1) vs. AR(2) perturbations yield qualitatively different decoupling
#        directions V2, and hence different look-ahead profiles.
#      - Other DGPs would induce their own specific perturbation pattern.
#
# 4. Three perturbation types were analysed:
#      i.  Delta-type (lag 0 only): structurally analogous to an ARMA(1,1)
#          process with a small MA(1) coefficient when delta is small, see 
#          exercise 2.
#      ii. AR(1)-type (all lags): the perturbation spreads across all lags
#          according to a slightly modified AR(1) parameter, see exercise 3.
#      iii.AR(2)-type (all lags, periodic): unlike perturbations (i) and (ii),
#          this perturbation differs from the original DGP not only in
#          magnitude but also in shape, introducing a more severe
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
# 6. When the perturbation is mildly misspecified (as in Exercise 3),
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









# Main Take-Aways

# -The AR(1) forecast problem is self-similar and one-dimensional and the PCS problem is impossible and infeasible.
# -Perturbating the DGP allows to expand the column-space of the PCS constraint system.
# -While the size of the perturbation is irrelevant, the type (single-lag, multiple lags) and shape 
# (AR(1) vs. AR(2) perturbation) are relevant.
# -We analyzed three kinds of perturbation:
# 1. delta-type at lag 0 (this is similar to the structure of an ARMA(1,1) with a very small MA(1)-term when delta is small)
# 2. AR(1)-type: the perturbation spreads over all lags according to a slightly modified AR(1) parameter.
# 3. AR(2)-type: we selected a periodic AR(2). In constrast to perturbations 1 and 2,  the perturbation here is sizeable not only 
#    in magnitude but also in shape.

# -In all considered cases the rank increased from 1 to two. The PCS solution lies in the space spanned 
#     by the original AR(1) and the perturbation vector (the first two eigenvectors of the constraint matrix corresponding to the two non-vanishing eigenvalues): either delta (e1), modified AR(1) or AR(2).
# -Changing lambda and beta allows to navigate in this space. Navigating through lambda and beta (instead of directly weighting V1 and V2) 
#  ensures optimality: not all linear combinations of V1 and V2 are also optimal predictors.
# Multiple perturbations could increase the rank of the constraint system to match the PCS constraints but 
# the effect on the target correlation CCF(h) could be deleterious.

















