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

# The AR(1) DGP represents the most challenging case for PCS. Its autocorrelation
# structure satisfies the Yule-Walker equations:
#
#   ACF(k) = a1 * ACF(k-1),
#
# which define a rank-one system that leaves virtually no room to adjust or
# reshape the CCF.
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
#───────────────────────────────────────────────────────────────────────────────

# ── RANK-ONE CONSTRAINT SYSTEM AND ITS CONSEQUENCES ───────────────────────────

# Due to self-similarity, the PCS constraint system has rank one for the AR(1)
# DGP. While it is possible to impose b' * (gamma_k - gamma_{k-1}) = beta for a
# single k, doing so with beta > 0 forces the sign of b in a direction that yields
# CCF(h) < 0 — a negative target correlation, which is unacceptable for a
# predictor of x_{t+h}. See Tutorial 14 (introduction) for further discussion.
#
# As a result, no degree of freedom remains to reshape the CCF away from the
# exponentially decaying path imposed by the AR(1) structure. For a non-AR(1)
# DGP, residual degrees of freedom would exist and could be exploited to at least
# partially address the PCS constraints, even if the constraint system is not
# exactly feasible.
#
# Rather than solving the PCS constraint system exactly — which would consume
# too many degrees of freedom when feasible — we adopt the penalized criterion
# from Wildi (2026), Equation 46 (Appendix E), with the solution given by
# Equation 49. The penalty weight nu > 0 (renamed lambda in these tutorials)
# balances target correlation maximization against constraint deviation. It acts
# as a soft regularization hyperparameter, not as a Lagrange multiplier.
#───────────────────────────────────────────────────────────────────────────────

# ── PERTURBATION STRATEGY: EXPANDING THE RANK ─────────────────────────────────

# To overcome the rank-one limitation of the AR(1) DGP, we introduce infinitesimal
# and imperceptible departures from its rigid structure. These perturbations expand
# the constraint system from rank 1 to rank 2 (or higher). For simplicity, only
# rank-two perturbations are considered here.
#
# The perturbation magnitude is controlled by a scaling parameter delta > 0.
# Its choice must balance two competing objectives:
#   (i)  Small enough to remain imperceptible and to preserve clean geometric
#        interpretability.
#   (ii) Large enough to avoid numerical precision issues.
#
# In practice, delta ~ 10^{-5} is recommended. For theoretical purposes, delta
# is treated as negligibly small, which simplifies the geometric analysis.
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
# The solution is constrained to a specific locus within the plane spanned by
# V1 and V2.
#
# When delta ≈ 0, V1 ≈ gamma_0, and the PCS predictor can be interpreted as a
# linear combination of:
#   - The classical MSE predictor direction gamma_h (proportional to gamma_0), and
#   - The fully decoupling vector V2.
#
# Full decoupling (b ≈ V2) is a theoretically valid strategy for generating
# look-ahead behaviour, but it is often too extreme in practice, potentially
# leading to sign inversion and uninterpretable predictions. The weight lambda2
# on V2 — in combination with lambfa1, both governed by the PCS hyperparameters 
# lambda and beta — enables controlled, partial decoupling: the predictor 
# departs from gamma_0 toward full decoupling in a graduated and controllable  
# manner.
#───────────────────────────────────────────────────────────────────────────────

# ── DFP VS. PERTURBATION-BASED DECOUPLING ─────────────────────────────────────

# When delta ≈ 0, V1 ≈ gamma_0 so that V2 is orthogonal to gamma_0 and induces
# full decoupling from the present. The PCS hyperparameters modulate the predictor
# between two extremes:
#   - No look-ahead:      b ≈ V1 ≈ gamma_0 ≈ gamma_h (replicates the MSE predictor).
#   - Maximum look-ahead: b ≈ V2             (full decoupling from the present).
#
# A key distinction from the original DFP approach:
#   - In original DFP, decoupling arises intrinsically from the DGP structure,
#     provided the data-generating process is sufficiently flexible to support it.
#   - Here, decoupling is exogenously imposed via the perturbation vector, which
#     introduces an artificial direction orthogonal to the AR(1) subspace.
#
# The common element across all perturbation-based PCS predictors in this tutorial
# is V1 ≈ gamma_0 ≈ gamma_h, the shared MSE predictor direction (valid when
# delta ≈ 0). The distinguishing element is V2, the full decoupling vector, which
# depends on the specific perturbation chosen.
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
# becomes highly sensitive to small changes in the hyperparameters. It is therefore
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
# sensistive to small changes in the hyperparameters which eventually facilitates 
# the search for interesting look ahead alternatives.
#───────────────────────────────────────────────────────────────────────────────

# ── EXTENSION TO DFP ──────────────────────────────────────────────────────────
#
# The perturbation framework developed above for PCS extends naturally to the DFP
# approach. Like PCS, DFP operates by manipulating the CCF of the predictor, but
# its constraints are generally less restrictive — particularly when compared to
# Type III PCS at longer horizons (h > 1).
#
# Consequently, the rank-expanding perturbation strategy can be applied to DFP
# in exactly the same way: by introducing a small perturbation delta to the AR(1)
# DGP, the rank-one constraint system is expanded to rank two, unlocking the
# decoupling direction V2 and enabling controlled look-ahead behaviour under the
# DFP criterion as well.
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
#             The perturbation is based on an AR(2) specification, introducing a
#             qualitatively different lag structure. As with Example 3, the
#             decoupling vector V2 is determined by the AR(2) perturbation and
#             shapes the look-ahead profile of the PCS predictor accordingly.
#

################################################################################
# Main Take-Away:

# The AR(1) structure renders it impossible to shift the CCF peak to the right.
# More precisely, no linear predictor can alter the exponentially decaying
# profile of the CCF at positive lags. In this sense, the AR(1) forecast problem 
# is THE HARDEST LOOK AHEAD FORECAST PROBLEM.

# How to Address Look-Ahead Behaviour When the CCF Peak Cannot Be Shifted?
#
# Although the right tail of the CCF is entirely determined by the AR(1) 
# structure and is therefore immutable, the left tail remains accessible to
# manipulation via the choice of predictor. The following observations,
# carried over from Exercise 3.5, elaborate on this point:
#
# Key observations:
#   - As beta (the slope hyper parameter) increases, the empirical CCF becomes 
#     increasingly right-skewed, reflecting a growing lead of the PCS predictor 
#     relative to the MSE predictor.
#
#   - The right tail of the CCF (lag > 0) always follows the AR(1) decay:
#     b' * gamma_h ∝ a1^h, since gamma_h = a1^h * gamma_0. This is a structural
#     consequence of the Yule-Walker equations and holds for any linear predictor
#     b. No non-zero predictor can alter this decay shape.
#
#   - Consequently, shifting the CCF peak strictly to the right of lag 0 is
#     impossible under the AR(1) DGP (see Exercise 1).
#
#   - Only the left tail of the CCF is amenable to modification. Whereas the MSE
#     predictor yields a symmetric CCF, the PCS predictor becomes progressively 
#     more asymmetric as beta increases. Effective look-ahead behaviour is thus 
#     achieved by skewing the CCF rightward — that is, by down-weighting the 
#     contribution of negative lags.
#
# Summary:
# When the right tail of the CCF cannot be modified — as is structurally
# the case for the AR(1) DGP — attention shifts to the left tail. By
# introducing suitable perturbations to the original DGP (which may be
# made arbitrarily small), the left tail can be shaped to induce look-ahead
# behaviour. This offers a principled resolution to an otherwise impossible
# problem: rather than attempting to shift the CCF peak directly, one
# instead recovers effective lead behaviour by redistributing CCF mass
# away from negative lags.
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
# Under strong regularisation, the PCS predictor is a linear combination of
# V[,1] and V[,2], with weights governed by beta and lambda.
#
# As beta varies, the predictor traces a path between the two extremes:
#   -V[,2]  (strongly negative beta)  and  +V[,2]  (strongly positive beta),
# passing through -V[,1] at approximately beta = 0.000000269 (lambda1 = 0).

# Fix a strong regularisation weight.
lambda <- 5000000

# Grid of beta values spanning the two extreme directions.
beta_vec <- c(-1, -0.1, 0, 0.0000001, 0.0000002, 0.00000025,
              0.000000269, 0.0000003, 0.0000005, 0.00001, 10)

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
                              ", beta =", round(beta_vec, 8))


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
# decay profile for all lags k > 0. The perturbation provides a single degree
# of freedom — a fine-tuning adjustment at lag 0 only — without altering the
# structure of the predictor at any other lag.
#
# This limited flexibility is the key limitation of the single-lag perturbation:
# it expands the rank from 1 to 2, but the additional degree of freedom is
# confined entirely to lag 0. As a result, the look-ahead behaviour induced by
# V2 (Panel 4) is relatively modest compared to multi-lag perturbations, which
# reshape the predictor profile across all lags and thereby generate a richer
# decoupling effect, see exercises 3 and 4 below.
#
# Under strong regularisation, the PCS predictors transition smoothly from -V2
# to +V2 as beta increases. The two extremes, ±V2, reflect configurations where
# the constraint is driven entirely by the decoupling direction V2, with V1
# (the gamma_0 component) receiving zero weight. V2 is the primary
# enabler of the PCS constraints and is therefore emphasized when lambda is 
# large (as is the case here): for sufficiently large positive or negative beta, 
# the predictor collapses onto ±V2 and the contribution of V1 vanishes.
#
# Between these two extremes, the predictor traces a continuum of optimal linear
# combinations of V1 and V2, with the weights determined by the
# interplay between the target correlation (governed by V1) and the constraint
# satisfaction (governed by V2) for the specified values of lambda and beta.

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
#   by the constraints.
#
# Panel 4 — CCF against V2:
#   Isolates the full decoupling contribution introduced by the perturbation.
#   This component captures the idiosyncratic look-ahead effect that arises
#   exclusively from the extraneous perturbation. Unlike the V1 component, it
#   varies with the hyperparameters and depends directly on the type and
#   direction of the perturbation chosen. Different perturbations yield different
#   V2 vectors and therefore different look-ahead profiles in this panel.


# ─────────────────────────────────────────────────────────────────────
# 2.5 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Same as Exercise 2.3 but with a medium-sized lambda:
# Medium regularization
lambda<-5
# Tipping points: the two extremes are -V[,1] and +V[,1]
beta_vec<-c(0,0.086,0.0874,0.08745,0.08746,0.08747,0.0875,0.088,0.09)

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}

filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,4))


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
# component — represented by V1 ≈ gamma_0 — dominates, and the solution space
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

# Grid of beta values spanning the transition between the two extremes.
beta_vec <- c(-1, 0, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45,
              0.455, 0.46, 0.47, 0.5, 5) / lambda

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
                                ", beta =", round(beta_vec*lambda, 8)))


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
# The PCS design exhibits sensitivity to the choice of beta: small perturbations
# within the interval [0.3/lambda, 0.5/lambda] primarily drive the transition from
# -V2 to +V2. This sensitivity reflects the near-singularity of the design as delta
# approaches zero — a fundamental trade-off inherent to the interpretability of the
# parameterization. Specifically, as delta -> 0, V1 converges to gamma0 (the direction
# associated with the AR(1) process), while V2 aligns with the fully decoupling
# direction. For larger delta, the design becomes less singular (less sensitive 
# to beta) but also less interpretable.

#
# ── Interpretation of CCF: Full-Lag AR(1) Perturbation ────────────────────────
#
# In contrast to Exercise 2, the CCF in Panel 4 (CCF against V2) now exhibits
# genuine look-ahead behaviour: as beta increases, the peak of the CCF shifts
# progressively to the right, toward the target horizon h. This right-shifting
# of the CCF peak reflects the increasing weight placed on the full decoupling
# direction V2, which — unlike the lag-0 perturbation in Exercise 2 — reshapes
# the predictor across all lags and thereby induces a meaningful lead in the
# correlation structure.

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
#   - Larger beta values (columns >= 10) produce increasingly leading predictors.
#   - As the degree of lead increases, predictors tend toward sign inversion,
#     reflecting the fundamental difficulty of the AR(1) forecasting problem.
#   - We select the leading predictors as well as the MSE benchmark predictor.
#   - All series are standardized to simplify visual inspection.
select_pcs<-10:ncol(y_out_mat)
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

select_pcs<-10:13
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
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])





# ─────────────────────────────────────────────────────────────────────
# 3.6 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Lambda is fixed at a moderate regularization strength. Beta is then
# varied across a grid of values. The two extreme values of beta
# correspond to the first eigenvector V[,1] with opposite signs
# (-V[,1] and +V[,1]), with intermediate solutions of the form
# -V[,2] + lambda1 * V[,1] (up to optimal scaling), where lambda1 depends 
# continuously on beta.

# Medium regularization
lambda<-5

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

beta_vec<-c(2,2.055,2.057,2.058,2.059,2.0591,2.0592,2.0593,2.0594,2.0595,2.06,2.1)/lambda


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
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))


# ─────────────────────────────────────────────────────────────────────
# 3.7 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V1 and +V1, passing through -V2 at an intermediate tipping point.

# The CCF in the fourth panel illustrates look ahead behaviour: the CCF peak 
# is shifted rightwards along the fully decoupled V2 direction.

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
#   - Smaller beta values produce lagging predictors (relative to the MSE).
#   - Larger beta values (columns >= 10) produce increasingly leading predictors.
#   - As the degree of lead increases, predictors tend toward sign inversion,
#     reflecting the fundamental difficulty of the AR(1) forecasting problem.
#   - We select the leading predictors as well as the MSE benchmark predictor.
#   - The leading predictors shift the CCF peak to the right in the 4-th panel 
#     of the previous CCF plot (along the full decoupling direction V2).
#   - All series are standardized to simplify visual inspection.
select_pcs<-c(4:7)
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
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])




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

# Beta grid spanning the transition between the two boundary solutions
# -V[,2] and +V[,2], passing through -V[,1] at the tipping point
beta_vec <- c(-10, -1, 0, 1, 1.2, 1.25, 1.27, 1.3,
              1.35, 1.4, 1.5, 2, 10) / lambda

Delta <- 1:h

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
                                ", beta =", round(beta_vec, 8)))


# ─────────────────────────────────────────────────────────────────────
# 4.4 Plots
# ─────────────────────────────────────────────────────────────────────

# The PCS predictor transitions smoothly between the two boundary solutions
# -V2 and +V2, passing through -V1 at an intermediate tipping point.
#
# Note: V1 appears with an inverted sign because the monotonically increasing
# CCF required by the constraints cannot be achieved without reversing the
# sign of the DGP direction encoded in V1. The very large lambda selected
# here amplifies this effect, driving the solution toward the sign-inverted
# direction as the constraint penalty dominates the optimisation objective.
#
# However, in contrast to Exercise 3, the CCF against xi (second panel)
# is either near zero or negative throughout. This indicates that placing
# excessive weight on the perturbed AR(2) constraints via large lambda
# induces misspecification: the predictor loses meaningful correlation
# with the target. A more balanced strategy is to employ small to
# moderate values of lambda, so that target correlation remains a
# relevant and influential component of the optimisation objective.

colo<-plot_func()


# ─────────────────────────────────────────────────────────────────────
# 4.5 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

# Since the above PCS predictors are misspecified under strong regularization,
# forecast comparisons are omitted for this configuration.




# ─────────────────────────────────────────────────────────────────────
# 4.6 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a medium regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# first eigenvector V[,1] with combinations V[,2]+lambda1*V[,1] in between, where 
# lambda1 depends on beta.



# Medium regularization
lambda<-5

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]

beta_vec<-c(0.43,0.4371,0.4372,0.43725,0.43727,0.4373,0.43733,0.43735,0.4374,0.4375,0.438)/lambda


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
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))



# ─────────────────────────────────────────────────────────────────────
# 4.7 Plots
# ─────────────────────────────────────────────────────────────────────


colo<-plot_func()

# Note
# -The CCF is evaluated against the true AR(1) DGP, i.e., xi:

# b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi%*%xi)).

# -Consider that the following slight modification 
#     b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   would be fixed since xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   is constant (not dependent on i if xi is the AR(1) DGP).
# -However, xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) is proportional to a^i i.e. decreases exponentially.

# Conclusions:
# 1. The observed decrease of the CCF is only due to the scaling effect in xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) 
#     and corresponds to a^i: all CCF's in the right panel decay with a^i.
# 2. It is not possible to have a locally increasing CCF except through sign inversion (impossibility and infeasibility)
# 3. In the original AR(2)-case (Tutorial 13) the peak of the CCF could be shifted because xi corresponded to the AR(2),i.e., one could rely on phase effect.
#     But here xi is AR(1): no phase effect. As a result, even the AR(2)-perturbation is unable to shift the peak.


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


anf<-150
enf<-500

# Select the relevant PCS: For increasing beta the predictors are increasingly left-shifted.
# For increasing beta the predictros appear to change sign.
# Very difficult forecast problem.
select_pcs<-c(2:5)
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


# For increasing beta, the CCF is increasingly skewed
mplot_ccf<-scale(na.exclude(y_out_mat[,select_vec]))
colnames(mplot_ccf)<-colnames(y_out_mat)[select_vec]


# Note: the right tail of the ccf always corresponds to the AR(1).
# This is because b' * gama_h \propto a1^h because gammah=a1^h*gamma0
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])



# Under misspecification (AR(2) perturbation) emphasizing the perturbated constraints
# too heavily (through large lambda) might lead to unusable predictors.


# Main Take-Aways

# -The AR(1) forecast problem is self-similar and one-dimensional and the PCS problem is impossible and infeasible.
# -Perturbating the DGP allows to expand the column-space of the PCS constraint system.
# -While the size of the perturbation is irrelevant, the shape is relevant.
# -We analyzed three sorts of perturbation:
# 1. delta-type at lag 0 (this is similar to the structure of an ARMA(1,1) with a very small MA(1)-term when delta is small)
# 2. AR(1)-type: the perturbation spreads over all lags according to a slightly modified AR(1) parameter.
# 3. AR(2)-type: we selected a periodic AR(2). In constrast to perturbations 1 and 2,  the perturbation here is sizeable not only 
#    in magnitude but also in shape.

# -In all considered cases the rank increased from 1 to two. The PCS solution lies in the space spanned 
#     by the original AR(1) and the perturbation vector (the first two eigenvectors of the constraint matrix corresponding to the two non-vanishing eigenvalues): either delta (e1), modified AR(1) or AR(2).
# -Changing lambda and beta allows to navigate in these spaces. Alternatively, one could just 
#  rely on classic linear weighting of the two eigenvectors.
# Multiple perturbations could increase the rank of the constraint system to match the PCS constraints but 
# the effect on the target correlation CCF(h) could be deleterious.

















