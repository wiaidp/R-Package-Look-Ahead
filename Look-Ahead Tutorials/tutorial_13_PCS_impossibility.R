
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS: IMPOSSIBILITY AND INFEASIBILITY
# ════════════════════════════════════════════════════════════════════

# The DFP and PCS look-ahead approaches impose constraints on the cross-
# correlation function (CCF) of the resulting predictor with lagged (k < 0),
# coincident (k = 0), or leading targets x_{t+k} (k > 0), while maximizing
# the CCF at the forecast horizon k = h, i.e., CCF(h) = Cor(x-hat_t, x_{t+h}).
#
# Specifically:
#   - The DFP controls CCF(0) while maximizing CCF(h).
#   - The PCS ideally shifts the peak of the CCF toward lag k = h while
#     maximizing the peak height at that lag.
#
# In difficult forecasting problems where the MSE-optimal predictor is "stuck
# at the present" (i.e., its CCF peaks at k = 0), two complementary strategies
# can unlock look-ahead behavior:
#   - Decoupling the predictor from the nowcast by penalizing CCF(0), as in
#     the DFP approach.
#   - Shifting the CCF peak away from k = 0 toward k = h, as targeted by the
#     PCS approach, when feasible.
#
# However, the constraints imposed on the CCF may conflict with the internal
# structure of the data-generating process (DGP). In such cases, the degrees
# of freedom available for maximizing the objective — the target correlation
# at the prespecified horizon h — may shrink, leaving little potential to
# achieve effective look-ahead behavior. In some cases, the conflict is severe
# enough that no feasible solution exists. This can occur in two ways:
#
#   Case A) The constraints cannot all be satisfied simultaneously
#           (the constraint system is overdetermined).
#   Case B) The constraints can be satisfied, but the implied target
#           correlation CCF(h) is non-positive.
#
# We distinguish three problem categories, defined as follows:
#
#   - Impossibility: No predictor exists — under any approach — that achieves
#     a strictly positive CCF peak at lag k = h. This is an intrinsic property
#     of the DGP, independent of the method used. See below for details.
#
#   - Infeasibility: No valid PCS solution exists within the proposed framework
#     (Types I, II, and III; see below), but this does not imply that the
#     problem is impossible. A solution may exist outside the proposed PCS
#     framework.
#
#   - Feasibility: A valid PCS solution exists. However, feasibility does not
#     guarantee that the resulting predictor achieves a CCF peak at k = h
#     (in the sense of a global or local maximum), nor that the peak height
#     is maximized at that lag. Notably, even an impossible problem may admit
#     a feasible PCS solution — see, e.g., Exercise 1.
#
# This tutorial analyzes a simple impossible example within the framework of
# Tutorial 9, based on an ARMA(1,1) process fitted to the monthly PAYEMS
# employment (and business cycle) indicator (see Examples 1 and 2 below).
#
# We begin with a brief summary of the main PCS predictor typology, organized
# by constraint structure and solution space. See Tutorial 10 for general
# background on PCS.


# ── PCS PREDICTOR TYPOLOGY ────────────────────────────────────────────────────

#   TYPE I — Monotonically Increasing CCF over {0, …, h}  [Most Restrictive]
#
#       The CCF must be strictly increasing across the full lag interval:
#           CCF(k-1) < CCF(k)  for all k = 1, …, h.
#       See Wildi (2026), Section 3.2 and Appendix E.
#       This condition is generally not exactly achievable (see Exercise 1).
#       The principal PCS optimization function PCS_func() enforces it as
#       closely as possible via regularization. Smaller regularization weights
#       allow greater flexibility, but may yield a CCF that does not peak at
#       k = h, even in cases where such a peak would otherwise be feasible.
#
#   TYPE II — Positive Local Slope at the Target Lag  [Weaker than Type I]
#
#       The CCF must be increasing over the final step only:
#           CCF(h-1) < CCF(h).
#       See Wildi (2026), Section 3.2.
#       In cases where the DGP imposes additional structure (e.g., via the
#       Yule-Walker equations of an AR(p) process), Types I and II may become
#       equivalent and may be equally feasible or infeasible.
#
#   TYPE III — Positive Average Slope from Lag 0 to Lag h  [Weaker than Type I]
#
#       The CCF must be increasing on average from k = 0 to k = h:
#           CCF(0) < CCF(h).
#       See Wildi (2026), Section 3.2.


# ── CONSTRAINT SUMMARY ───────────────────────────────────────────────────────
#
#   Type I:   CCF(k) > CCF(k-1)  for k = 1, …, h  (h constraints)
#   Type II:  CCF(h) > CCF(h-1)                    (1 constraint)
#   Type III: CCF(h) > CCF(0)                      (1 constraint)
#
# Types II and III are necessary but not sufficient conditions for a global
# CCF maximum at lag k = h. Type I is neither necessary (strict monotonicity
# may be overly restrictive) nor sufficient for a global maximum. Nevertheless,
# in many applications all three constraint types are effective in the sense
# that the resulting PCS predictor exhibits useful look-ahead behavior, even
# when the CCF peak does not fall exactly at k = h.
#
# Among the three types, Type I is the most stringent: it imposes the largest
# number of constraints (one per lag from k = 1 to k = h). While this
# increases the likelihood of attaining a CCF peak at k = h, it simultaneously
# reduces the degrees of freedom available for optimizing the target correlation
# at horizon h, and locks the CCF into a rigid monotonic profile. Consequently,
# Type I carries the highest risk of infeasibility, with that risk growing with
# h and depending strongly on the structure of the data-generating process (DGP).
#
# Note, however, that infeasibility under Type I can often be mitigated by
# choosing a moderate regularization weight. This provides sufficient control
# over the desired CCF peak shape while preserving enough flexibility to avoid
# over-constraining the path to the peak (see below).


# ── WHEN AND WHY CAN A PROBLEM BE INFEASIBLE? ────────────────────────────────
#
# The Type I PCS solution takes the form (see Wildi 2026, Appendix D):
#
#   b = gamma_h + sum_{k=1}^{h} lambda_k * (gamma_k - gamma_{k-1})
#
# where:
#   - b          is the filter coefficient vector that maximizes the target
#                correlation at forecast horizon h,
#   - gamma_h    is the MSE h-step-ahead predictor coefficient vector,
#   - gamma_k    is the MSE k-step-ahead predictor coefficient vector,
#   - lambda_k   are Lagrange multipliers (regularization weights) chosen to
#                enforce a monotonically increasing CCF from lag k = 0 to k = h.
#
# Since CCF(k) = b' * gamma_k / (||b|| * ||xi||), where xi is the DGP (weights of 
# the Wold decomposition)  the monotonicity condition CCF(i) > CCF(i-1) requires:
#
#   b' * (gamma_i - gamma_{i-1}) > 0,   i = 1, …, h.
#
# Our solution implements this condition in one of two forms:
#
#   (aa)  b' * (gamma_i - gamma_{i-1}) = beta
#   (ab)  b' * (gamma_i - gamma_{i-1}) = beta * ||gamma_i - gamma_{i-1}||
#
# The two cases are selected by setting scaled_constraints = FALSE (case aa)
# or scaled_constraints = TRUE (case ab).
#
# In both cases, beta is a prescribed common CCF increment that enforces
# the monotonically increasing profile if  beta > 0:
#   - Case (aa) assumes a fixed, uniform slope across all lags.
#   - Case (ab) scales the increment by ||gamma_i - gamma_{i-1}||, making the
#     effective increment beta_i := beta * ||gamma_i - gamma_{i-1}|| lag-dependent,
#     thereby accounting for the varying magnitude of successive predictor differences.
#
# Note on the implied CCF slope:
#   - In case (aa): CCF(i) - CCF(i-1) = beta / (||b|| * ||xi||), a constant
#     slope, provided the problem is feasible.
#   - In case (ab): the slope also depends on ||gamma_i - gamma_{i-1}|| and
#     therefore varies across lags.
#
# Substituting the expression for b into the constraint equations yields a
# system of h linear equations in h unknowns (lambda_1, …, lambda_h):
#
#   (gamma_h + sum_{k=1}^{h} lambda_k * (gamma_k - gamma_{k-1}))'
#       * (gamma_i - gamma_{i-1}) = beta,   i = 1, …, h,
#
# for case (aa), and analogously for case (ab).
#
# Infeasibility arises in two distinct ways:
#
#   Case A) The constraint vectors (gamma_i - gamma_{i-1}), i = 1, …, h, are
#           linearly dependent and the right-hand side h-dimensional vector 
#           (beta, …, beta)' does not lie in their column span. The linear 
#           system has no solution.
#
#   Case B) The linear system has a solution, but the implied target correlation
#           CCF(h) is non-positive. Since a predictor that is negatively
#           correlated with x_{t+h} is inadmissible, the problem is declared
#           infeasible in this case as well.
#
# A problem is therefore feasible if and only if:
#   (i)  A solution to the constraint system exists (Case A does not occur), and
#   (ii) The implied target correlation CCF(h) is strictly positive (Case B does
#        not occur).


# ── ILLUSTRATIVE EXAMPLES OF INFEASIBLE/IMPOSSIBLE PROBLEMS ───────────────────
#
# Example 1: AR(1) DGP with positive autoregressive coefficient a1 > 0.
#
#   For any AR(1) process with a1 > 0, the h-step-ahead MSE predictor
#   coefficient vectors satisfy:
#
#       gamma_h ∝ gamma_{h+k}   for all h >= 0, k >= 0.
#
#   It follows that all difference vectors (gamma_k - gamma_{k-1}),
#   k = 1, …, h, are proportional to one another, so the constraint matrix
#   has column rank one.
#
#   Under the scaled constraint system (case ab):
#
#       b' * (gamma_k - gamma_{k-1}) = beta * ||gamma_k - gamma_{k-1}||,
#                                                          k = 1, …, h,
#
#   the right-hand side vector
#
#       beta * (||gamma_1 - gamma_0||, …, ||gamma_h - gamma_{h-1}||)'
#
#   lies in the column space of the constraint matrix, so the system is
#   solvable. However, the sign of beta determines the sign of b and thus 
#   the sign of the target correlation CCF(h) at k=h. The peak of the CCF is always 
#   located at k=0, it cannot be shifted, and a monotonically increasing CCF is 
#   only possible by inverting signs. The problem is impossible (Case B infeasibility).
#
#   Under the unscaled constraint system (case aa):
#
#       b' * (gamma_k - gamma_{k-1}) = beta,   k = 1, …, h,
#
#   the right-hand side vector (beta, …, beta)' does not lie in the
#   rank-one column space when h > 1 (more than one constraint) and beta ≠ 0. 
#   The system therefore has no solution (Case A infeasibility).  
#
#
# Example 2: ARMA(1,1) DGP with positive autoregressive coefficient a1 > 0.
#
#   For h > 0 (strictly larger zero), the same proportionality as 
#   in Example 1 holds:
#
#       gamma_h ∝ gamma_{h+k}   for all h > 0, k >= 0.
#
#   As a consequence:
#
#       b' * (gamma_k - gamma_{k-1}) = 0   for k = 2, …, h,
#
#   meaning the constraint system cannot enforce a strictly increasing CCF
#   beyond lag k = 1. The CCF peak therefore cannot be located at k > 1,
#   and the problem is impossible and infeasible under Type I and Type II 
#   constraints for any h > 1.
#
#   However, unlike the AR(1) case, the MA component introduces an asymmetry
#   between k = 0 and k = 1. Specifically, if a1>0 and the MA coefficient b1 < 0,
#   then CCF(1) > CCF(0), and the CCF peak is located at k = 1. In this
#   configuration, a peak at k>0, namely k=1, is possible and feasible 
#   (but not at k>1). 
#   Interestingly, the Type III constraints (requiring CCF(h) > CCF(0) and
#   CCF(h) > 0) may remain feasible even for h > 1 in this example, see 
#   exercise 1. Nevertheless, because the peak is fixed at k = 1 and cannot be 
#   moved to larger lags, the problem remains impossible in the sense that the 
#   desired look-ahead horizon cannot be achieved for h > 1. This demonstrates 
#   feasibility of an impossible problem. See Exercise 1 for a worked example.


# ── FEASIBILITY AND ARMA(p,q) STRUCTURE ──────────────────────────────────────
#
# For an ARMA(p,q) process, the effective dimension of the PCS constraint
# space is at most p + q. This is because for lags k > q, the autocovariances
# R(k) = gamma_0' * gamma_k satisfy a p-dimensional linear recurrence
# (the Yule-Walker equations), which renders all higher-lag constraint vectors
# linearly dependent on the first p + q constraint vectors (see Exercise 2).
#
# The feasibility outcome therefore depends critically on the number of
# linearly independent constraints imposed relative to the effective
# dimension (rank) p + q:
#
#   Case 1 — Fewer than p + q independent constraints:
#     Residual degrees of freedom remain available to the optimizer. The
#     target correlation CCF(h) is ALWAYS POSITIVE and increases as fewer
#     constraints are imposed, giving the optimizer greater flexibility to
#     track the target. However, this comes at the potential cost of reduced
#     look-ahead effectiveness: there is no guarantee that the CCF peaks at
#     k = h, nor that the peak magnitude is maximized at that lag.
#
#   Case 2 — Exactly p + q independent constraints:
#     All available degrees of freedom are consumed by the constraints. The
#     target correlation CCF(h) is then fully determined by the constraint
#     system and may be non-positive, in which case the problem is infeasible
#     (Case B). If CCF(h) > 0, the problem is feasible, but again there is
#     no guarantee that the CCF peak occurs at k = h or that the peak height
#     is maximized at that lag.
#
#   Case 3 — More than p + q independent constraints:
#     The constraint system is overdetermined. Feasibility depends on whether
#     the right-hand side vector (beta, …, beta)' (in case aa) lies in the 
#     column space spanned by the constraint vectors (gamma_i - gamma_{i-1}):
#
#       * If it does not, no solution exists and the problem is infeasible
#         (Case A). See exercise 2 for a worked example (which is based 
#         on the constraint system of case aa).
#       * If it does, all constraints are simultaneously satisfiable. The
#         problem is then feasible if and only if the implied target
#         correlation CCF(h) is strictly positive; otherwise it is
#         infeasible (Case B). See exercise 3 for a worked example (which is based 
#         on the constraint system of case ab).


# ── ADDRESSING INFEASIBILITY VIA REGULARIZATION ──────────────────────────────

# Infeasible problems can be addressed via regularization, which penalizes
# departures from the constraints. When the problem is truly infeasible, these
# deviations do not vanish as the regularization weight grows, since the
# constraints cannot be satisfied regardless of how strongly they are enforced, 
# see exercise 4.
#
# Assigning a moderate (rather than arbitrarily large) regularization weight
# preserves flexibility, unfreezes degrees of freedom, and allows the optimizer
# to maximize tracking accuracy at horizon h (i.e., target correlation /
# minimum MSE). This flexibility also accommodates a wider variety of CCF
# shapes, including non-linear or non-monotonic profiles, that would otherwise
# be excluded by overly rigid constraints — ultimately improving the chances of
# locating the CCF peak at the forecast horizon k = h and maximizing its height.
#
# For example, a Type I formulation with a moderate regularization weight may
# successfully recover a PCS solution whose CCF peaks at k = h with a positive
# target correlation, precisely because the relaxation from a rigid (strictly 
# monotonic prespecified path: cases aa or ab above) CCF profile allows the 
# optimizer to explore a richer solution space, see exercise 5.

# ── POSSIBLE YET INFEASIBLE ──────────────────────────────────────────────────

# A  PCS problem may be possible yet infeasible in the following
# sense: a predictor exists whose cross-correlation function (CCF) peaks at
# k = h with a positive target correlation, and yet none of the three
# proposed constraint types (I, II, III) successfully identifies this solution.
# The reasons can be as follows:
#
#   - Type I constraints may be overly restrictive: they impose a specific
#     structural pattern on the CCF across the full lag interval k = 0, ..., h
#     (e.g., monotonically increasing, cases aa and ab), which can exclude 
#     valid solutions when a large regularization weight is applied.
#
#   - Type II constraints may be insufficiently restrictive: they require only
#     a local increase from lag k = h-1 to k = h, which may fail to enforce
#     the intended "peak at h" condition more broadly.
#
#   - Type III constraints may also be insufficiently restrictive: they require
#     only a positive average increase from k = 0 to k = h, which again may
#     fail to capture the "peak at h" solution in the general case, see 
#     exercise 1.
#
# For difficult forecast problems, we generally recommend PCS Type I (stronger structural
# control over the CCF profile) paired with a moderate regularization weight
# (greater flexibility), which permits controlled departures from a strictly
# predetermined monotonic CCF profile. This relaxation frees up degrees of
# freedom that can then be directed toward maximizing the target correlation
# CCF(h), ensuring that the look-ahead design achieves effective tracking of
# the target at horizon h while maintaining firm control over the desired
# peak-at-h profile. Fine-tuning of the regularization weight may be required
# to balance the inherent trade-off between structural control and
# optimization flexibility.
#
# However, when the CCF peak migrates naturally towards k = h under the
# simpler Type II or Type III constraints, the latter are generally preferred.
# By imposing less extraneous structure on the CCF profile, Types II and III
# leave more degrees of freedom available for optimization, making the target
# correlation CCF(h) more amenable to effective maximization and reducing the
# risk of misspecification-induced distortions.



# ── EXAMPLES OVERVIEW ─────────────────────────────────────────────────────────
#
# Example 1 — Impossible but Feasible (PCS Type III)
#
#   The problem is feasible in the sense that the Type III constraint is
#   satisfied: CCF(h) > CCF(0) and the implied target correlation CCF(h) is
#   strictly positive. However, the CCF peak occurs at k = 1 rather than at
#   the requested target horizon k = h = 12, which is structurally
#   unachievable for this DGP. The look-ahead objective is therefore not met,
#   and the problem is classified as impossible.
#
#   More strikingly, the resulting filter simultaneously degrades the
#   signal-to-noise ratio (i.e., amplifies noise) and introduces additional
#   lag — a doubly adverse outcome. This pathological behaviour arises because
#   the Type III constraint is imposed under structural conditions of the DGP
#   that leave insufficient degrees of freedom to address the forecasting
#   problem in any meaningful way: the constraint is satisfied, but only at
#   the cost of a filter that is actively detrimental.
#
#
# Example 2 — Impossible and Infeasible: Case A
#             (PCS Type I, unscaled constraint system, case aa)
#
#   Under case (aa), PCS Type I requires a linearly increasing CCF profile
#   from k = 0 to k = h when beta > 0. This profile is structurally
#   unachievable for the present DGP, and the problem falls under Case A
#   (an overdetermined constraint system with no solution):
#
#   - When b1 ≠ 0, the vector gamma_0 is linearly independent of gamma_h for
#     all h > 0, but gamma_h and gamma_{h+k} are linearly dependent for all
#     h > 0 and k >= 0.
#   - Consequently, the h constraint vectors (gamma_i - gamma_{i-1}),
#     i = 1, …, h, span a column space of effective dimension 2, reflecting
#     the ARMA(1,1) structure of the DGP — far smaller than the nominal
#     dimension h = 12 imposed in the example (one-year ahead forecast).
#   - The target right-hand side vector (beta, …, beta)' of dimension h = 12
#     does not lie in this 2-dimensional column space, so the constraint
#     system has no solution.
#   - Consequently, no filter coefficient vector b exists that simultaneously
#     satisfies all h = 12 monotonicity constraints, and the problem is
#     infeasible (Case A).
#
#
# Example 3 — Impossible and Infeasible: Case B
#             (PCS Type I, scaled constraint system, case ab)
#
#   Under case (ab), the scaled constraint system takes the form:
#
#       b' * (gamma_i - gamma_{i-1}) = beta * ||gamma_i - gamma_{i-1}||,
#                                                            i = 1, …, h.
#
#   For the considered ARMA(1,1) DGP and the one-year ahead forecast horizon 
#   h=12, the right-hand side vector
#
#       beta * (||gamma_1 - gamma_0||, …, ||gamma_h - gamma_{h-1}||)'
#
#   lies in the column space of the constraint matrix, so the system is
#   solvable. However, the implied target correlation CCF(h) is non-positive,
#   rendering the problem infeasible under Case B.
#
#   The solution to Exercise 3 illuminates the underlying mechanism:
#
#   - The constraints are satisfied, and the resulting CCF is monotonically
#     increasing over k = 0, 1, …, h, as required by the constraint system.
#   - However, the CCF begins at a strongly negative value at k = 0,
#     indicating a sign reversal in the predictor.
#   - Although the CCF increases monotonically, it remains negative at k = h,
#     so the target correlation is non-positive (Case B infeasibility).
#   - Furthermore, the CCF continues to increase beyond k = h, confirming
#     that no peak occurs at the desired horizon.
#   - So the constraints are formally met, i.e., CCF(k) > CCF(k-1) for 
#     k = 1, ..., h, but CCF(h) < 0 and the predictor is unusable.
#
#
# Examples 4–7 — An Easier (Possible) Forecast Problem Based on a periodic 
#                AR(2) DGP
#
#   These examples share a common periodic AR(2) DGP for which look-ahead
#   forecasting is genuinely achievable, and collectively illustrate how the
#   choice of constraint type and regularisation strength affects the quality
#   and feasibility of the resulting predictor.
#
#   - Example 4 (PCS Type I, large regularisation weight):
#     Highlights Type A infeasibility. The constraints impose a linearly
#     increasing CCF profile, which is structurally incompatible with the
#     AR(2) DGP. The heavy regularisation weight enforces the misspecified
#     constraints too rigidly, causing the resulting predictor to change sign
#     and become practically unusable.
#
#   - Example 5 (PCS Type I, moderate regularisation weight):
#     Explores the same framework with a reduced regularisation weight,
#     allowing greater flexibility in accommodating the misspecified constraint
#     system. Although the problem remains infeasible, the relaxed constraints
#     permit the predictor to exhibit meaningful look-ahead behaviour, at the
#     moderate cost of a slightly sub-optimal target correlation.
#
#   - Example 6 (PCS Type II, feasible reformulation):
#     Proposes a correctly specified PCS Type II constraint that is compatible
#     with the AR(2) DGP. The resulting predictor exhibits clear look-ahead
#     behaviour and, because the constraint is no longer misspecified, the
#     target correlation is maximised — in direct contrast to Example 5.
#
#   - Example 7 (PCS Type III, feasible reformulation):
#     Similar to example 6 but the alternative PCS Type III approach is used. 

# ═════════════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ═════════════════════════════════════════════════════════════════════════════





# ── INITIALISATION ───────────────────────────────────────────────────
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
# EXERCISE 1 — Impossible but Feasible (PCS Type III)
# ════════════════════════════════════════════════════════════════════

# We adopt the framework from Tutorial 9: an ARMA(1,1) model fitted to the
# monthly PAYEMS employment indicator. PCS Type III is applied with forecast
# horizon h = 12, imposing the constraint CCF(12) > CCF(0).
#
# The problem is feasible in the following sense:
#   - The Type III constraint is satisfied: CCF(12) > CCF(0).
#   - The implied target correlation is positive: CCF(12) > 0.
#
# However, the problem is impossible: the CCF peak cannot be located at
# k = h = 12 for this DGP, regardless of the approach used. The Type III
# constraint is satisfied, but only superficially — the resulting predictor
# does not achieve the intended look-ahead objective.


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

# Fit an ARMA(1,1) model: a parsimonious specification with adequate diagnostics.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)
a1 <- arima.obj$coef[1:ar_order]
b1 <- arima.obj$coef[ar_order + 1:ma_order]

# --- Wold Decomposition (MA-infinity representation) ---
# Compute the infinite-order MA coefficients (impulse response weights)
# of the fitted ARMA model. The filter length L ensures that the
# coefficients have decayed sufficiently close to zero by lag L.
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

# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: slowly decaying impulse response (post-1990)")

# The theoretical ACF implied by the Wold decomposition matches the
# empirical ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),
        main = "Model-based ACF", ylab = "", xlab = "Lag")

# For k > 0 the ACF satisfies the recurrence ACF(k+1) = a1 * ACF(k), so
# the DGP imposes a rigid linear structure on the autocorrelation function.
#
# For h > 0 and k >= 0 this implies gamma_h ∝ gamma_{h+k}: the MSE predictor
# coefficient vectors are mutually proportional for all horizons h > 0.
#
# Consequently, the column space of the constraint system has rank 2:
# (gamma_0 - gamma_1) and (gamma_1 - gamma_2) ∝ gamma_1 are the only
# linearly independent directions. Because the Type III PCS enforces only a
# single constraint, the problem remains feasible. Unfortunately, the CCF
# cannot peak at h = 12 — this is structurally impossible for the
# ARMA(1,1) DGP — so the problem is classified as impossible but feasible.


# ─────────────────────────────────────────────────────────────────────
# 1.3 MSE Benchmark
# ─────────────────────────────────────────────────────────────────────
# One-year-ahead forecast horizon.
h <- 12

# Truncate the Wold coefficients to length L to obtain the nowcast
# filter (gamma_0).
gamma0 <- xi[1:L]

# h-step-ahead MSE predictor (gamma_h):
# shift the Wold coefficients forward by h positions.
gammah <- xi[h + 1:L]


# ─────────────────────────────────────────────────────────────────────
# 1.4 PCS Type III Framework
# ─────────────────────────────────────────────────────────────────────
# The Type III PCS imposes decoupling of the predictor from the
# difference filter (gamma_0 - gamma_h). Note that for h = 1 the
# Type III and Type I PCS coincide; for h = 12 they differ.

gamma_constraint <- gamma0 - gammah
gamma_target     <- gammah
max_lag          <- 0

ts.plot(gamma_constraint, main = "PCS: gamma_constraint")

# The Type III problem can be solved via PCS or DFP. In the DFP solution
# the predictor is decoupled from gamma_constraint (rather than from gamma_0).
# The label "DFP" (Decoupling from Present) therefore refers to the solution
# methodology — i.e., the use of a decoupling device — rather than to the
# specific direction from which decoupling occurs.

# We use the unitary DFP so that the constraint parameter alpha is directly
# interpretable as the correlation between b and gamma_constraint:
#   - Positive alpha  →  decreasing CCF profile: CCF(0) > CCF(h)
#   - Negative alpha  →  increasing CCF profile: CCF(h) > CCF(0)
alpha_vec <- c(0.8, 0.6, 0.3, 0, -0.1)


cor_vec_mat <- b_mat <- NULL
for (i in 1:length(alpha_vec))
{
  alpha <- alpha_vec[i]
  # Solve the quadratic problem in lambda and compute the unit-length DFP.
  b_obj <- unitary_DFP_func(gamma_constraint, gamma_target, alpha)
  b     <- b_obj$b0
  b_mat <- cbind(b_mat, b)
  # Compute the CCF of the resulting PCS predictor at all relevant lags.
  cor_vec_mat <- cbind(cor_vec_mat,
                       compute_acf_at_lags_zero_delta_func(
                         max_lag, h, b_mat[, i], xi)$cor_vec)
}



# ─────────────────────────────────────────────────────────────────────
# 1.5 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: Verify unit length ---
# Each column of b_mat should have squared norm equal to 1.
apply(b_mat^2, 2, sum)


# --- Check 2: Verify that the PCS constraint is met ---
# Compute the correlation of each predictor with gamma_constraint.
# Since b'b = 1 (unit length), no additional scaling is required.
correlation_0 <- t(b_mat) %*% gamma_constraint /
  as.double(sqrt(gamma_constraint %*% gamma_constraint))
# All differences should be (numerically) zero: the constraints are satisfied.
correlation_0 - alpha_vec


# --- Check 3: Sign/orientation preservation ---
# If the sum of filter weights is strictly positive, the DFP does not
# reverse the sign (direction) of a trend signal.
apply(b_mat, 2, sum)


# --- Check 4: Positive target correlation ---
# All target correlations should be positive, confirming feasibility.
t(b_mat) %*% gammah / as.double(sqrt(gammah %*% gammah))


# --- Check 5: Minimum MSE ---

# MSE of the unitary (not yet optimally scaled) PCS predictors.
apply((b_mat - gammah)^2, 2, sum)

# Compute the MSE-optimal scaling factor for each predictor.
optimal_mse_scaling <- as.vector(
  t(b_mat) %*% gammah / apply(b_mat^2, 2, sum))

# Apply the optimal scaling.
b_mat_mse <- t(t(b_mat) * optimal_mse_scaling)

# MSE of the optimally scaled PCS predictors.
# Note: these values assume a standardised white-noise input (unit innovation
# variance); the ARMA(1,1) innovation variance sigma^2 is not applied here.
apply((b_mat_mse - gammah)^2, 2, sum)


# Assemble all relevant predictors for downstream analysis.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast", "MSE", paste("PCS ", alpha_vec, sep = ""))


# ─────────────────────────────────────────────────────────────────────
# 1.6 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo     <- c("black", "green", rainbow(ncol(filter_mat) - 2))

# ── Left panel: filter coefficient profiles ───────────────────────────
# Display original nowcast and MSE as well as unit-length PCS filter coefficients.
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lwd=ifelse(colnames(mplot)[i]=="MSE",2,1),lty=ifelse(colnames(mplot)[i]=="MSE",2,1))
}

abline(v = 1 + h, lty = 2)   # vertical marker at target horizon h
abline(h = 0)                 # zero-correlation reference line

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()

# Outcome:
# For the negative constraint parameter alpha = -0.1, the resulting PCS
# predictor satisfies the look-ahead condition CCF(h) - CCF(0) > 0:
ccf_mat["CCF at lead: 12", ncol(ccf_mat)] - ccf_mat["CCF at lead: 0", ncol(ccf_mat)]
# Since all target correlations are strictly positive, the problem is
# feasible across all designs. However, it remains impossible: the CCF
# peak does not occur at the requested horizon k = h = 12, which is
# structurally unachievable for this ARMA(1,1) DGP.
# For all designs with a positive constraint parameter alpha > 0, the
# look-ahead condition is reversed: CCF(h) - CCF(0) < 0.


# ─────────────────────────────────────────────────────────────────────
# 1.7 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────

#----------------------------------------------------------------------
# 1.7.1 Apply Predictors to Data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the innovations eps_t
# from the Wold decomposition). They are therefore applied to the model
# residuals rather than to the raw series. The AR from is derived in exercise 
# 1.9 below.
x_filt <- arima.obj$residuals

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


#----------------------------------------------------------------------
# 1.7.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))

# Full-sample overview of all predictor outputs.
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(filter_mat) - 2)),
        lwd = c(1, 2, rep(1, ncol(filter_mat) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Excerpt: dot-com recession period (observations 120–170).
ts.plot(y_out_mat[120:170, ],
        main = "Predictor Outputs: Dot-com Recession", col = colo,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(filter_mat) - 2)),
        lwd = c(1, 2, rep(1, ncol(filter_mat) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Excerpt: global financial crisis period (observations 200–250).
ts.plot(y_out_mat[200:250, ],
        main = "Predictor Outputs: Financial Crisis", col = colo,
        xlab = "", ylab = "",
        lty = c(2, 2, rep(1, ncol(filter_mat) - 2)),
        lwd = c(1, 2, rep(1, ncol(filter_mat) - 2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# As in Exercise 2 of Tutorial 9, the PCS predictors visibly LAG the MSE
# predictor rather than leading it. The problem cannot be addressed without 
# misspecifying the predictor, see Tutorial 14 for alternative possible problem 
# formulations.  


# ─────────────────────────────────────────────────────────────────────
# 1.8 Amplitude and Time-Shift Functions
# ─────────────────────────────────────────────────────────────────────

K      <- 600    # number of frequency grid points
plot_T <- FALSE  # suppress internal plotting; custom plot constructed below

amp_mat <- shift_mat <- NULL
for (i in 1:ncol(filter_mat))
{
  as_obj    <- amp_shift_func(K, filter_mat[, i], plot_T)
  amp_mat   <- cbind(amp_mat,   as_obj$amp)
  shift_mat <- cbind(shift_mat, as_obj$shift)
}
colnames(amp_mat) <- colnames(shift_mat) <- colnames(filter_mat)


# Plot amplitude and time-shift functions across frequencies [0, π].
par(mfrow = c(1, 2))

lty_vec <- c(2, 2, rep(1, ncol(filter_mat) - 2))

# --- Amplitude functions ---
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
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# --- Time-shift functions ---
mplot <- shift_mat
# Suppress extreme negative values in the last column to maintain a
# readable vertical scale.
mplot[which(mplot[, ncol(mplot)] < (-2)), ncol(mplot)] <- NA

plot(mplot[, 1], type = "l", axes = FALSE, lty = lty_vec[1],
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-Shift Functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lty = lty_vec[i])
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Amplitude functions:
#   - Imposing a stronger decoupling constraint (more negative alpha) reduces
#     low-frequency amplitude and increases noise leakage at higher frequencies,
#     thereby deteriorating the signal-to-noise ratio.
#
# Time-shift functions:
#   - Remarkably, the deterioration in signal-to-noise ratio does not translate
#     into a smaller (i.e., more lead-like) time-shift. This contrasts with
#     Exercise 2 of Tutorial 9, where the worse signal-to-noise ratio was
#     accompanied by a smaller shift at frequency zero (a lead) but a larger
#     lag at business-cycle frequencies.
#   - In the present case the Type III PCS worsens both characteristics
#     simultaneously: it is not on the efficient amplitude-time-shift frontier.
#
# Explanation: to satisfy CCF(h) > CCF(0) under the constraints imposed by
# this DGP, the predictor must reduce the weight on the most recent innovation,
# which mechanically increases the lag rather than inducing a lead.


# ─────────────────────────────────────────────────────────────────────
# 1.9 AR Form: AR Inversion of the ARMA(1,1) Model
# ─────────────────────────────────────────────────────────────────────

# Compute the AR inversion filter (the inverse of the MA polynomial),
# which transforms the Wold (MA-infinity) representation back into
# an AR representation.
ar_inv <- -ARMAtoMA(ar = -arima.obj$coef[ar_order + 1:ma_order],
                    ma = -arima.obj$coef[1:ar_order],
                    lag.max = L)
# Assemble the full AR filter (including the leading coefficient of 1).
theta <- c(1, -ar_inv)

# Verify correctness via a known identity:
# convolving the AR inversion filter with the Wold (MA) coefficients must
# yield the identity filter, i.e., [1, 0, 0, ...].
conv_two_filt_func(xi, theta)$conv[1:10]

# Visualise the AR inversion filter.
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR Inversion Filter")

# The first weight is always 1 (the coefficient on the current observation x_t).
# The remaining weights decay geometrically:
theta[2:L] / theta[1:(L - 1)]
# The first ratio equals a1 + b1 (up to sign):
arima.obj$coef[ar_order + 1:ma_order] + arima.obj$coef[1:ar_order]
# Subsequent ratios match b1 (up to sign), confirming geometric decay:
arima.obj$coef[ar_order + 1:ma_order]

# Having verified the identity, we convolve the AR operator with each PCS
# predictor (in MA form) to obtain the corresponding AR-form representation.


# ─────────────────────────────────────────────────────────────────────
# 1.10 Convolution of the AR Operator with the Predictors
# ─────────────────────────────────────────────────────────────────────

# Verification: convolving theta with gamma_0 should recover the identity
# filter. Residual deviations from [1, 0, 0, ...] vanish as L increases,
# reflecting the finite truncation of both the MA and AR representations.
conv_two_filt_func(theta, gamma0)$conv[1:10]

# Convolve the AR operator with each predictor in filter_mat to obtain
# the AR-form coefficient vectors.
filter_mat_ar <- NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar <- cbind(filter_mat_ar,
                         conv_two_filt_func(theta, filter_mat[, i])$conv)
colnames(filter_mat_ar) <- colnames(filter_mat)

# Verification: the first column (nowcast gamma_0) should be the identity.
filter_mat_ar[1:10, 1]


# ─────────────────────────────────────────────────────────────────────
# 1.11 Analysis and Plot of Predictors in AR Form
# ─────────────────────────────────────────────────────────────────────

colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- 10
par(mfrow = c(1, 1))
# Plot the first `first_lags` AR coefficients for each predictor.
ts.plot(
  filter_mat_ar[1:first_lags, ],
  col  = colo,
  main = "DFP (MSE-Optimal) Predictors in AR Form",
  lty  = c(2, 2, rep(1, ncol(filter_mat) - 2)),
  lwd  = c(1, 2, rep(1, ncol(filter_mat) - 2)))
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)

# As the constraint parameter alpha decreases (stronger decoupling), the
# AR-form weight assigned to the current observation x_t diminishes, while
# the weights on lagged observations x_{t-k}, k > 0, increase correspondingly.
# Although this redistribution of weights satisfies the decoupling constraint
# — producing an increasing CCF(h) - CCF(0) — it comes at a cost: the
# predictor becomes progressively more lagging and more noise-contaminated,
# as confirmed by the deteriorating amplitude and time-shift functions
# documented in Section 1.8.

# ─────────────────────────────────────────────────────────────────────
# 1.12 Equivalent PCS Type III Formulation via PCS_func()
# ─────────────────────────────────────────────────────────────────────
# As an alternative to the DFP methodology, the Type III decoupling
# constraint can be imposed directly via PCS_func(), which solves the
# constrained optimisation problem for a given target slope beta.

# Target slope values for the single Type III constraint:
#   b' * (gamma_h - gamma_{h-1}) = beta
beta_vec <- c(0.8, 0.6, 0.3, 0, -0.1)

# Single constraint imposed between k = 0 and k = h: CCF(h)-CC(F)>0.
Delta <- c(0,h)

# Provide the Wold decomposition as input to the optimiser.
gamma_pcs <- xi

# A very large regularisation weight ensures the constraint is imposed
# with negligible slack (i.e., enforced almost exactly).
lambda <- 10^10

# For Type III we set Type_III=T. 
# The default value is F (when the Boolean is omitted in the call to PCS_func()) 
Type_III <- T

# Loop over beta values and compute the corresponding PCS predictor.
b_mat <- NULL
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type III predictor for the current beta.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,Type_III)
  b        <- PCS_obj$b
  d_delta  <- PCS_obj$d_delta
  b_mat    <- cbind(b_mat, b)
  
  # Constraint verification: for a feasible system, the residual
  # |d_delta' * b + beta| should approach zero as lambda -> Inf.
  print(abs(d_delta %*% b + beta))
}


# ─────────────────────────────────────────────────────────────────────
# 1.13 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo     <- c("black", "green", rainbow(ncol(filter_mat) - 2))

# ── Left panel: filter coefficient profiles ───────────────────────────
# Display original nowcast and MSE as well as unit-length PCS filter coefficients.
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(0,min(mplot)), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lwd=ifelse(colnames(mplot)[i]=="MSE",2,1),lty=ifelse(colnames(mplot)[i]=="MSE",2,1))
}

abline(v = 1 + h, lty = 2)   # vertical marker at target horizon h
abline(h = 0)                 # zero-correlation reference line

axis(1, at     = 1:nrow(mplot),
     labels = -1 + 1:nrow(mplot))
axis(2)
box()

# Outcome:
# For the negative constraint parameter beta = -0.1, the resulting PCS
# predictor satisfies the look-ahead condition CCF(h) - CCF(0) > 0:
ccf_mat["CCF at lead: 12", ncol(ccf_mat)] - ccf_mat["CCF at lead: 0", ncol(ccf_mat)]
# Note that beta is proportional (but not identical) to the slope.

# Here we see CCF(h) - CCF (0) for all predictors: the difference vanishes when beta=0.
ccf_mat["CCF at lead: 12",] - ccf_mat["CCF at lead: 0", ]



# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: Impossible and Infeasible, Case A (PCS Type I)
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────

# This exercise uses the same empirical framework as Exercise 1, but applies
# PCS Type I constraints instead of Type III.
#
# The problem is infeasible under Case A: the constraint system has no
# solution. Specifically, the h = 12 constraint vectors
# (gamma_i - gamma_{i-1}), i = 1, …, h, span a column space of effective
# dimension 2, reflecting the ARMA(1,1) structure of the DGP. The
# right-hand side vector (beta, …, beta)' of dimension h = 12 does not lie
# in this 2-dimensional column space, so no solution to the constraint
# system exists and the problem is infeasible (Case A).

# ─────────────────────────────────────────────────────────────────────
# 2.1 DGP Structural Constraints on the PCS (and DFP) Solution Space
# ─────────────────────────────────────────────────────────────────────
# Assemble the matrix of h-step MSE predictor coefficient vectors
# (gamma_i for i = 0, 1, ..., L-1) to examine the rank of the
# constraint system implied by the ARMA(1,1) DGP.
gamma_mat <- gamma0
for (i in 1:(L - 1))
{
  gamma_i   <- xi[i + 1:L]
  gamma_mat <- cbind(gamma_mat, gamma_i)
}

eigenvalues <- eigen(gamma_mat)$values

# Effective rank of the constraint system:
# count the number of eigenvalues that exceed a numerical-zero threshold.
which(abs(eigenvalues) > 10^{-10})

# The effective rank two is smaller than the number of constraints (h = 12).
# Consequently, the h-dimensional right-hand-side vector (beta, ..., beta)'
# may lie outside the column space of the constraint matrix — which is
# indeed the case here, rendering the Type I PCS system infeasible.

# Note on Exercise 1 (Type III PCS):
# Type III imposes a single constraint of the form b' * (gamma_h - gamma_0) = beta,
# which can always be solved exactly. With the effective rank equal to 2, one
# degree of freedom remains available for optimisation. This is the degree of
# freedom exploited in Exercise 1 — though with limited success, since the
# resulting predictor is lagging and noisy.


# ─────────────────────────────────────────────────────────────────────
# 2.2 PCS Type I: Parameter Setup
# ─────────────────────────────────────────────────────────────────────
# One-year-ahead forecast horizon.
h <- 12

# Grid of target slope values for the CCF. A positive beta requires the CCF
# to increase linearly from k = 0 to k = h (provided the problem is feasible
# and lambda is sufficiently large). Negative and zero values are included as
# reference cases to illustrate how the CCF profile responds to the slope target.
beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)

# scaled_constraints = FALSE selects the unscaled constraint system (case aa):
# the slope is fixed at beta for all lags, regardless of the magnitude of the
# constraint vectors.
scaled_constraints <- FALSE

# Type I imposes a positive slope at every lag in Delta (here 1 to h),
# enforcing a monotonically increasing CCF over the full interval {0, ..., h}.
# This is the most restrictive of the three PCS types.
Delta <- 1:h

# A very large regularisation weight drives the solution toward exact
# satisfaction of all h slope constraints simultaneously, when feasible. In practice,
# this level of regularisation is often more restrictive than necessary
# and may reduce the target correlation unduly (see Exercises 3.5 and 4).
lambda <- 10^10

# Provide the Wold decomposition as input to the optimiser.
gamma_pcs <- xi


# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat              <- NULL   # filter coefficients, one column per beta value
Type_III <- FALSE

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type I predictor; both calls below are equivalent,
  # the second making the default Boolean arguments explicit.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: for a feasible system the residual
  # |d_delta' * b + beta| should approach zero as lambda -> Inf.
  # Here the residuals do not vanish with increasing lambda, confirming
  # that the system is infeasible: the vector (beta, ..., beta)' does not
  # lie in the column space of the constraint matrix.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 2.4 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Verified in the optimisation loop above: the constraints cannot be
# satisfied exactly, confirming infeasibility.

# --- Check 2: Positive target covariance ---
# When beta is positive (increasing CCF slope), the target covariance
# turns negative, indicating sign reversal of the predictor — a hallmark
# of the infeasible, impossible regime.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 2.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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


# Outcome:
#
# Filter coefficients:
#   - Imposing a linearly increasing CCF over k = 0, ..., h inverts the sign
#     of the predictor coefficients, rendering the predictor unusable.
#
# Population CCFs:
#   - With an inverted predictor, the CCFs do increase from k = 0 to k = h,
#     but remain entirely negative and do not peak at k = h.
#
# Note: for h = 1, PCS Types I and II are feasible because gamma_0 and
# gamma_1 are not collinear, so the constraint system has a solution.
#
# Note: when the problem is feasible and lambda is large, the CCF increases
# linearly from k = 0 to k = h with a constant slope. Here, because the
# problem is both infeasible and impossible, no constant-slope solution exists
# and the constraint residuals do not vanish as lambda -> Inf.



# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: Impossible and Infeasible — Case B (PCS Type I)
# ════════════════════════════════════════════════════════════════════
#
# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────
#
# This exercise uses the same empirical framework as Exercises 1 and 2,
# but applies the scaled constraint system (case ab) instead of the
# unscaled system (case aa) used in Exercise 2.
#
# In contrast to Exercise 2, the scaled constraint system is now solvable.
# However, the implied target correlation CCF(h) remains non-positive,
# so the problem is both impossible and infeasible (Case B). This stands
# in contrast to Exercise 1, where the impossible problem was nonetheless
# feasible — albeit with a predictor that delivered poor and actively
# detrimental performance.
#
# Why is the scaled constraint system solvable?
#
#   - The effective column rank of the constraint matrix remains 2,
#     reflecting the ARMA(1,1) structure of the DGP.
#   - However, switching from case (aa) to case (ab) replaces the fixed
#     right-hand-side vector (beta, ..., beta)' — which does not lie in
#     the 2-dimensional column space — with the scaled vector
#
#         beta * (||gamma_1 - gamma_0||, ..., ||gamma_h - gamma_{h-1}||)',
#
#     which does lie in the column space of the constraint matrix.
#   - Consequently, the scaled constraint system has an exact solution.
#     Nevertheless, the implied target correlation CCF(h) is non-positive,
#     and the problem remains infeasible (Case B).


# ─────────────────────────────────────────────────────────────────────
# 3.1 PCS Type I: Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# One-year-ahead forecast horizon.
h <- 12

# Grid of target slope values for the CCF. A positive beta requires the CCF
# to increase from k = 0 to k = h. In contrast to Exercise 2, the slope is
# not fixed at beta but varies across lags as beta * ||gamma_k - gamma_{k-1}||,
# k = 1, ..., h. This variable-slope formulation is compatible with the
# ARMA(1,1) structure, so all constraints can be satisfied exactly.
# Note: beta is still a scalar input; the scaling is controlled by setting
# scaled_constraints = TRUE below (cf. Exercise 2, where it was FALSE).
beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)

# scaled_constraints = TRUE selects the scaled constraint system (case ab):
# the slope at each lag k is beta * ||gamma_k - gamma_{k-1}|| rather than
# the fixed value beta used in case (aa).
scaled_constraints <- TRUE

# Type I imposes a monotonicity constraint at every lag in Delta (here 1 to h),
# enforcing an increasing CCF over the full interval {0, ..., h}. This is the
# most restrictive of the three PCS types (I, II, and III).
Delta <- 1:h

# A large regularisation weight drives the solution toward exact constraint
# satisfaction. Because the scaled system lies within the column space of the
# constraint matrix, the residuals can be made arbitrarily small by increasing
# lambda (subject to numerical precision).
lambda <- 10^9

# Provide the Wold decomposition as input to the optimiser.
gamma_pcs <- xi


# ─────────────────────────────────────────────────────────────────────
# 3.2 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat                <- NULL    # filter coefficients, one column per beta value
Type_III <- FALSE

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type I predictor under the scaled constraint system.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: in contrast to Exercise 2, the constraint
  # residuals can be made arbitrarily small by increasing lambda (assuming
  # sufficient numerical precision), confirming that the system is solvable.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 3.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Validated in the optimisation loop above. Unlike Exercise 2, the constraint
# residuals diminish toward zero as lambda increases, confirming feasibility
# of the scaled constraint system (subject to numerical precision limits).

# --- Check 2: Positive target covariance ---
# The problem is infeasible (Case B): for positive beta, the target
# covariance is non-positive, indicating sign reversal of the predictor.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))

# ─────────────────────────────────────────────────────────────────────
# 3.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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


# Note: because scaled_constraints = TRUE, the slope imposed at each lag k
# is beta_k = beta * ||gamma_k - gamma_{k-1}|| rather than the fixed value
# beta used in Exercise 2. Consequently, the CCF slope is not constant
# across lags.

# Verification: each row of d_delta has unit length (the constraint vectors
# are normalised under the scaled formulation).
apply(d_delta^2, 1, sum)

# The slope magnitude |CCF(k) - CCF(k-1)| is largest at k = 1 and decreases
# monotonically with k. This follows directly from the fact that the norms
# of the successive difference vectors are themselves decreasing:
#   ||gamma_1 - gamma_0|| > ||gamma_2 - gamma_1|| > ...
# so the scaled slope beta_k = beta * ||gamma_k - gamma_{k-1}|| is largest
# at k = 1 and tapers off at longer lags.



# ════════════════════════════════════════════════════════════════════
# Exercise 4: Possible AR(2) but Infeasible — PCS Type I (Case A)
# ════════════════════════════════════════════════════════════════════
#
# Exercises 4–6 examine an easier, *possible* forecast problem based on
# an AR(2) DGP. Possibility means that there exists a predictor whose
# CCF peaks at horizon h with a strictly positive value — in contrast to
# the ARMA(1,1) setting of Exercises 1–3, where no such predictor exists.
# Despite the problem being possible, the specific constraint formulation
# adopted in Exercise 4 (PCS Type I, unscaled) renders it infeasible.


# ─────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises
# the empirical framework (process specification, filter length, forecast
# horizon, and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# 4.1 Specify the AR(2) Model
# ─────────────────────────────────────────────────────────────────────
# Fit an ARMA(2,2) model to PAYEMS and retain only the AR(2) component;
# the MA(2) part is discarded for this set of exercises.
ar_order <- 2
ma_order <- 2

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)

# Extract the AR(2) coefficients. The fitted AR(2) is periodic (complex
# conjugate roots), producing a damped-cycle impulse response.
a1 <- arima.obj$coef[1]
a2 <- arima.obj$coef[2]
  
# Compute the Wold (MA-infinity) representation from the AR(2) component only.
# The MA(2) part of the fitted ARMA(2,2) model is deliberately discarded:
# the goal here is not a business-cycle application but rather to use the
# AR(2) structure as a controlled and transparent setting in which to
# illustrate and compare the various PCS constraint types (I, II, and III).
xi <- c(1, ARMAtoMA(ar = c(a1, a2), lag.max = length(x)))

# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition: AR(2) Damped Cycle")


# ─────────────────────────────────────────────────────────────────────
# 4.2 MSE Predictors and Constraint System Rank
# ─────────────────────────────────────────────────────────────────────
h      <- 12
gamma0 <- xi[1:L]
gammah <- xi[h + 1:L]

# Assemble the matrix of MSE predictor coefficient vectors and examine
# the effective rank of the constraint system.
gamma_mat <- gamma0
for (i in 1:(L - 1))
{
  gamma_i   <- xi[i + 1:L]
  gamma_mat <- cbind(gamma_mat, gamma_i)
}

eigenvalues <- eigen(gamma_mat)$values

# Count the number of numerically non-zero eigenvalues: rank of the 
# constraint system.
which(abs(eigenvalues) > 10^{-10})

# The effective rank is 2. More generally, for an AR(p) the rank of the
# constraint system equals p (a consequence of the Yule-Walker structure).
# Because the rank (2) is smaller than the number of constraints (h = 12),
# the fixed right-hand-side vector (beta, ..., beta)' may lie outside the
# 2-dimensional column space — which is the case here, making the unscaled
# Type I constraint system infeasible (Case A).


# ─────────────────────────────────────────────────────────────────────
# 4.3 PCS Type I: Parameter Setup
# ─────────────────────────────────────────────────────────────────────
# Grid of target slope values for the CCF. A positive beta requires the
# CCF to increase linearly from k = 0 to k = h (case aa). Negative and zero 
# values are included as reference cases to illustrate the effect of the slope
# target on the CCF profile and peak location.
beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)

# scaled_constraints = FALSE selects the unscaled constraint system (case aa):
# the slope is fixed by beta (proportional to beta) for all lags.
scaled_constraints <- FALSE

# Type I imposes a positive slope at every lag in Delta (here 1 to h),
# enforcing a monotonically increasing CCF over the full interval {0, ..., h}.
# This is the most restrictive of the three PCS types (I, II, and III).
Delta <- 1:h

# A very large regularisation weight drives the solution toward exact
# constraint satisfaction. However, because the system is infeasible (Case A),
# the constraint residuals remain sizeable regardless of lambda.
lambda <- 10^10

# Provide the Wold decomposition as input to the optimiser.
gamma_pcs <- xi


# ─────────────────────────────────────────────────────────────────────
# 4.4 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat                <- NULL    # filter coefficients, one column per beta value
Type_III <- FALSE               # Default setting for types I) or II) PCS

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type I predictor; both calls are equivalent, the second
  # making the default Boolean arguments explicit.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: for a feasible system the residuals should
  # approach zero as lambda -> Inf. Here they remain sizeable, confirming
  # infeasibility (Case A): the vector (beta, ..., beta)' does not lie in
  # the 2-dimensional column space of the constraint matrix.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 4.5 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Validated in the optimisation loop above: the constraint residuals do not
# vanish with increasing lambda, confirming Case A infeasibility.

# --- Check 2: Positive target covariance ---
# For positive beta (positive CCF slope), the target covariance is
# negative, indicating sign reversal — consistent with Case A infeasibility.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))

# ─────────────────────────────────────────────────────────────────────
# 4.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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

# Outcome:
#
# Filter coefficients:
#   - Increasing beta renders the predictors increasingly negative and unusable at 
#     horizon h=12 — an instance of the sign-reversal pathology discussed in the
#     overview (the constraint system is misspecified and a large lambda imposes 
#     this misspecification).
#
# Population CCFs:
#   - The positive beta CCFS overlap; the positive beta CCFs overlap too.
#   - With positive beta (positive slope), the CCFs do increase from k = 0 
#     to k = h, but remain largely negative and do not peak at k = h. 
#     The look-ahead objective is therefore not achieved in any meaningful sense.
#
# Note: for h = 1, PCS Types I and II are both feasible for this AR(2) DGP,
# because gamma_0 and gamma_1 are not collinear and the single constraint
# vector lies within the column space of the constraint matrix.
#
# Note: when the problem is feasible and lambda is large (strong regularization), 
# the slope of the CCF is fixed, i.e., the CCF increases linearly from k=0 to k=h. 
# Here, the problem is infeasible and impossible and the slope cannot be constant.



# ════════════════════════════════════════════════════════════════════
# Exercise 5: Same as Exercise 4 but with a Moderate Regularisation Weight
# ════════════════════════════════════════════════════════════════════
#
# ─────────────────────────────────────────────────────────────────────
# Note: Exercises 1 and 4 must be run before this exercise, as they
# initialise the empirical framework (process specification, filter
# length, forecast horizon, and Wold coefficient vector) required here.
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# 5.1 Moderate Regularisation Weight
# ─────────────────────────────────────────────────────────────────────
# In Exercise 4 the very large lambda enforced the misspecified constraints
# so rigidly that the predictor changed sign and became unusable. Here we
# reduce lambda to a moderate value, allowing the optimiser greater
# flexibility to trade off constraint satisfaction (and hence avoid 
# misspecification) against MSE performance.
# As a result, the deviations from constraints are larger, the predictor
# retains the correct sign and exhibits meaningful look-ahead behaviour.
lambda <- 1

# Retain the same beta grid as in Exercise 4.
beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)


# ─────────────────────────────────────────────────────────────────────
# 5.2 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat                <- NULL    # filter coefficients, one column per beta value
Type_III <- FALSE

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type I predictor; both calls are equivalent, the second
  # making the default Boolean arguments explicit.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: the residuals are larger than in Exercise 4
  # because lambda is moderate and the misspecified constraints are only
  # partially enforced rather than driven toward zero.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 5.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Validated in the optimisation loop above: the constraint residuals do not
# vanish, confirming that the system remains infeasible (Case A). The
# moderate lambda prevents the sign reversal observed in Exercise 4, at the
# cost of imperfect constraint satisfaction.

# --- Check 2: Positive target covariance ---
# In contrast to Exercise 4, all target covariances are now positive,
# indicating that the predictors retain the correct sign and orientation.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 5.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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

# Outcome:
# In contrast to Exercise 4, the CCF peak shifts progressively rightward as
# beta increases, which is the desired look-ahead behaviour.
#
# Two important observations:
#
#   1. The MSE predictor itself already exhibits look-ahead behaviour for
#      this AR(2) DGP, owing to the periodic (damped-cycle) structure of
#      the impulse response.
#
#   2. The key distinction between the MSE and PCS approaches is one of
#      anchoring: the PCS predictor generates look-ahead behaviour while
#      remaining anchored at the intended forecast horizon h, because it
#      directly maximises CCF(h) subject to the imposed constraints. By
#      contrast, a MSE predictor targeting horizon h_tilde >  h
#      achieves look-ahead by maximising CCF(h_tilde), progressively losing
#      control over the intended horizon h as h_tilde diverges from h.
#
# Note: the degree of rightward shift in the CCF peak can be tuned by
# adjusting beta and/or lambda; further exploration of this trade-off
# is left as an extension.


# ════════════════════════════════════════════════════════════════════
# Exercise 6: PCS Type II (Making the Possible Problem Feasible, Part One) 
# ════════════════════════════════════════════════════════════════════
#
# ─────────────────────────────────────────────────────────────────────
# Note: Exercises 1 and 4 must be run before this exercise, as they
# initialise the empirical framework (process specification, filter
# length, forecast horizon, and Wold coefficient vector) required here.
# ─────────────────────────────────────────────────────────────────────

# Exercises 4 and 5 imposed misspecified infeasible constraints through 
# Type III) PCS. Here we adopt the simpler PCS Type II) which renders 
# the single constraint feasible and not misspecified.

# ─────────────────────────────────────────────────────────────────────
# 6.1 Type II PCS Setup
# ─────────────────────────────────────────────────────────────────────
# Type II imposes a single slope constraint at the target horizon only:
#   b' * (gamma_h - gamma_{h-1}) = beta,
# i.e., CCF(h) > CCF(h-1) when beta > 0.
# A single constraint can always be satisfied exactly by the rank-two
# constraint system of the AR(2) DGP, so a very large lambda is appropriate.
Delta <- h

# Very large regularisation weight: the single constraint is enforced
# with negligible slack.
lambda <- 10^10

# A fine grid of small beta values is used to trace the effect of the
# slope constraint on the CCF peak location.
beta_vec <- c(-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3) / 50


# ─────────────────────────────────────────────────────────────────────
# 6.2 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat                <- NULL    # filter coefficients, one column per beta value
Type_III <- FALSE

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type II predictor; both calls are equivalent, the
  # second making the default Boolean arguments explicit.
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: because the single constraint lies within the
  # rank-2 column space of the AR(2) constraint matrix, the residual can
  # be made arbitrarily small by increasing lambda (assuming unlimited 
  # numerical precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 6.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Validated in the optimisation loop above: the single constraint is
# satisfied, confirming feasibility of the Type II formulation.

# --- Check 2: Positive target covariance ---
# All target covariances are positive, confirming feasibility. Note that
# for the largest beta values (strongest look-ahead), the target covariance
# approaches zero, indicating that the predictor is approaching full
# decoupling from x_{t+h}.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 6.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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



# Outcome:
# The Type II PCS controls the rightward shift of the CCF peak through a
# single constraint on CCF(h) - CCF(h-1):
#   - beta = 0:  the CCF peak shifts toward k = h.
#   - beta > 0:  the peak moves to horizons k >= h (stronger look-ahead).
#   - beta < 0:  the peak moves to horizons k <= h (weaker look-ahead).
#
# This single constraint, together with the internal periodic structure of
# the AR(2) DGP, is sufficient to operate an effective and well-controlled
# rightward shift. Crucially, no artificial extraneous structure is imposed
# beyond the slope requirement CCF(h) - CCF(h-1) ∝ beta. As a result, the
# target correlation CCF(h) is maximised among all predictors satisfying
# the same constraint — a property not shared by the Type I formulation
# of Exercises 4 and 5.


# ════════════════════════════════════════════════════════════════════
# Exercise 7: PCS Type III (Making the Possible Problem Feasible, Part Two) 
#
# ════════════════════════════════════════════════════════════════════
#
# ─────────────────────────────────────────────────────────────────────
# Note: Exercises 1 and 4 must be run before this exercise, as they
# initialise the empirical framework (process specification, filter
# length, forecast horizon, and Wold coefficient vector) required here.
# ─────────────────────────────────────────────────────────────────────

# Exercises 4 and 5 imposed misspecified and infeasible constraints via
# PCS Type I. Exercise 6 adopted the simpler PCS Type II, which rendered
# the single constraint both feasible and correctly specified. The present
# exercise turns to PCS Type III, with a different single and feasible 
# constraint.
#
# The two constraint types differ in how they address the CCF profile:
#
#   - Type II controls CCF(h) - CCF(h-1): setting beta = 0 is sufficient
#     to produce a flat CCF at k = h, which is enough to locate the peak
#     at the target horizon.
#
#   - Type III controls CCF(h) - CCF(0): setting beta = 0 is not sufficient
#     to shift the peak to k = h, because the constraint only anchors the
#     CCF relative to its value at lag zero, leaving the intermediate profile
#     unconstrained. A sufficiently large positive beta is required to elevate
#     CCF(h) above the natural peak and thereby displace it to k = h.

# ─────────────────────────────────────────────────────────────────────
# 7.1 Type III PCS Setup
# ─────────────────────────────────────────────────────────────────────
# Type III imposes a single slope constraint:
#
#   b' * (gamma_h - gamma_0) = beta,
#
# which enforces CCF(h) > CCF(0) whenever beta > 0.

Delta <- c(0,h)
# Type III setting:
Type_III <- T

# Since the constraint matrix of an AR(2) DGP has effective column rank 2,
# a single constraint is always exactly satisfiable regardless of the choice
# of beta. Consequently, a large regularization weight lambda is appropriate
# here: it drives the solution firmly towards satisfying the constraint
# without any risk of infeasibility.
# Very large regularisation weight: the single constraint is enforced
# with negligible slack.
lambda <- 10^10

# A coarser grid is used and fairly large positive values are required to 
# shift the peak of the CCF towards k=h: 
beta_vec <- c(-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3) *2



# ─────────────────────────────────────────────────────────────────────
# 7.2 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).


b_mat                <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute the PCS Type II predictor; both calls are equivalent, the
  PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda,
                      Type_III, scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint verification: because the single constraint lies within the
  # rank-2 column space of the AR(2) constraint matrix, the residual can
  # be made arbitrarily small by increasing lambda (assuming unlimited 
  # numerical precision).
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 7.3 Routine Checks
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: PCS slope constraints ---
# Validated in the optimisation loop above: the single constraint is
# satisfied, confirming feasibility of the Type II formulation.

# --- Check 2: Positive target covariance ---
# All target covariances are positive, confirming feasibility. Note that
# for the largest beta values (strongest look-ahead), the target covariance
# approaches zero, indicating that the predictor is approaching full
# decoupling from x_{t+h}.
t(b_mat) %*% gammah

# Assemble all filters (nowcast, MSE benchmark, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah, b_mat)
colnames(filter_mat) <- c("Nowcast",
                          paste0("MSE(", h, ")"),
                          paste0("PCS lambda=", lambda,
                                 ", beta=", round(beta_vec, 2)))


# ─────────────────────────────────────────────────────────────────────
# 7.4 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- filter_mat
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
mplot <- ccf_mat

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


# Outcome:
#
# The result is qualitatively similar to Exercise 6. The key difference is
# that beta must be sufficiently large to shift the CCF peak towards k = h,
# whereas in Exercise 6 setting beta = 0 was sufficient to achieve this.
#
# This distinction arises from the different constraint types used:
#
#   - Type II (Exercise 6) enforces CCF(h) > CCF(h-1), so the condition
#     CCF(h) - CCF(h-1) = 0 is already sufficient to produce a flat —
#     and therefore peaking — CCF at k = h. The required value beta = 0
#     is known a priori and requires no tuning.
#
#   - Type III (this exercise) enforces CCF(h) > CCF(0), so beta must be
#     large enough to ensure that CCF(h) exceeds CCF(0) by a sufficient
#     margin to displace the peak from its natural location to k = h.
#     Unlike Type II, the required value of beta is not known a priori:
#     it must either be derived analytically through involved calculations
#     or determined empirically, making Type III considerably harder to
#     tune in practice.
#
# In summary, Type II offers a simpler and more transparent route to peak
# displacement: setting beta = 0 is both necessary and sufficient, and the
# required value is known a priori without any tuning.
#
# Type III can achieve the same outcome but requires careful selection of a
# strictly positive beta, the appropriate value of which depends on the DGP
# and the target horizon h, and is not known a priori.
#
# Both Type II and Type III are feasible and correctly specified in this
# exercise, and both are efficient in the sense that they maximize the
# peak height CCF(h) ate k=h.
#
# This stands in contrast to Type I, which is infeasible and misspecified
# for this DGP. The inherent misspecification can be partially mitigated
# by relaxing the regularization weight, which reduces the effective influence
# of the constraints and grants the optimizer greater freedom to work around
# the misspecification. However, the misspecification cannot be fully absorbed,
# and consequently the target correlation CCF(h) is not maximized — unlike
# the outcomes achieved under Type II and Type III.
#
# It should be noted, however, that Type I can be more effective in
# difficult forecasting problems where controlling a single CCF difference —
# either CCF(h) - CCF(h-1) (Type II) or CCF(h) - CCF(0) (Type III) — is
# insufficient to reliably locate the CCF peak at the target horizon k = h.
# By enforcing a monotonically increasing CCF profile across all lags from
# k = 0 to k = h, Type I imposes a much stronger structural requirement on
# the filter, which can be decisive when the DGP offers little natural
# support for peak displacement to the desired horizon (as is the case for 
# the above periodic AR(2) DGP).



# ─────────────────────────────────────────────────────────────────────
# Main Takeaways
# ─────────────────────────────────────────────────────────────────────
# Imposing fewer and more targeted constraints allows the PCS to better
# optimise CCF(h) — one of the primary objectives of h-step-ahead prediction.
#
# Comparing PCS Types across Exercises:
#
#   - Type II (Exercise 6, AR(2)): a single constraint at k = h is sufficient
#     to achieve a well-controlled CCF peak shift while maximising CCF(h).
#     This makes Type II the preferred choice when the DGP structure supports it.
#
#   - Type I with moderate lambda (Exercise 5, AR(2)): although the constraint
#     system is misspecified and infeasible, reducing lambda allows the optimiser
#     sufficient flexibility to recover meaningful look-ahead behaviour at the
#     cost of imperfect constraint satisfaction and a sub-optimal CCF(h).
#
#   - Type III: more effective than Type II in difficult forecast settings where
#     a single constraint on CCF(h) - CCF(h-1) is insufficient to force a peak
#     shift. In such cases, increasing the number of constraints and exploring
#     a range of regularisation weights is recommended.
#
# ── PRACTICAL RECOMMENDATION ──────────────────────────────────────────────────
#
# If maximizing the target correlation CCF(h) is of high priority:
#
#   Begin with Type II (setting beta = 0) or Type III (setting beta > 0 and
#   increasing it gradually until the CCF peak migrates, eventually to k = h), using the
#   minimal number of constraints. If the resulting look-ahead behavior is
#   adequate, no further action is needed. Otherwise, progressively increase
#   the structural control by switching to Type I, and tune lambda and/or beta
#   until a satisfactory trade-off between look-ahead performance and target
#   correlation CCF(h) is achieved.
#
# If small losses in CCF(h) are admissible:
#
#   Rely on PCS Type I with a suitably moderate regularization weight lambda
#   and a sufficiently large beta > 0 to displace the CCF peak towards k = h
#   where feasible. The regularization weight should remain moderate to avoid
#   over-constraining the filter and to preserve sufficient flexibility for
#   meaningful optimization of the target correlation. 

# ── TECHNICAL NOTES ───────────────────────────────────────────────────────────
#
# 1. Interaction between beta and lambda:
#
#    The parameters beta and lambda interact in a compensatory fashion: an
#    excessively large beta — which imposes an exaggerated and potentially
#    misspecified positive CCF slope — can be partially mitigated by reducing
#    lambda. The precise nature of this interaction is given in equation (49)
#    of Wildi (2026): the product beta * lambda operates on the numerator (so
#    the two parameters partially offset each other), while lambda alone enters
#    the denominator through the matrix M, which depends on lambda. Note that
#    Wildi (2026) uses the notation nu in place of lambda. In practice, some
#    experimentation may be required to identify a good — if not optimal —
#    pairing of beta and lambda.
#
# 2. Infeasibility and Misspecification:
#
#    When the rank of the constraint matrix is smaller than the number of
#    imposed constraints, the system may be infeasible (Case A). This occurs
#    when the right-hand side vector — either (beta, …, beta)' in the fixed-
#    slope case (aa), or (b_1, …, b_h)' in the varying-slope case (ab) —
#    does not lie in the column space of the constraint matrix. This mismatch
#    may be interpreted as misspecification: the constraints impose a structure
#    on the CCF that is incompatible with the data-generating process.
#
#    In such settings, the regularization weight lambda should not be set too
#    large. An excessively large lambda amplifies the influence of the
#    misspecified constraints, which can propagate into the filter coefficients
#    and produce actively detrimental predictors — for example, through sign
#    inversion, where the filter becomes negatively correlated with the target
#    at the desired horizon, resulting in CCF(h) < 0, which is unacceptable.








