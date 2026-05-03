
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS: INFEASIBILITY
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

#   Type I:   CCF(k) > CCF(k-1)  for k = 1, …, h  (h constraints)
#   Type II:  CCF(h) > CCF(h-1)                    (1 constraint)
#   Type III: CCF(h) > CCF(0)                      (1 constraint)
#
# Types II and III are necessary but not sufficient conditions for a global
# CCF maximum at lag k = h. Type I is neither necessary (monotonicity may be
# overly restrictive) nor sufficient. Nevertheless, in many applications all
# three constraint types are effective in the sense that the resulting PCS
# predictor exhibits useful look-ahead behavior, even when the CCF peak does
# not fall exactly at k = h.
#
# Among the three types, Type I is the most stringent: it imposes the largest
# number of constraints (one per lag from k = 1 to k = h), which increases the
# likelihood of attaining a CCF peak at k = h, but simultaneously reduces the
# degrees of freedom available for optimizing the target correlation at horizon
# h, and locks the CCF into a rigid monotonic profile. As a result, Type I
# carries the highest risk of infeasibility, with that risk increasing with h
# and depending strongly on the structure of the DGP.
#
# Note, however, that infeasibility under Type I can often be mitigated by
# selecting a moderate regularization weight, which provides sufficient control
# over the desired peak shape while preserving enough flexibility to avoid
# over-constraining the path to the peak, see below.


# ── WHEN AND WHY CAN A PROBLEM BE INFEASIBLE? ────────────────────────────────

# The Type I PCS solution takes the form (see Wildi 2026, Appendix D):
#
#   b = gamma_h + sum_{k=1}^{h} lambda_k * (tilde_gamma_k - tilde_gamma_{k-1})
#
# where:
#   - b              is the filter coefficient vector maximizing the target
#                    correlation at horizon h,
#   - gamma_h        is the MSE h-step-ahead predictor coefficient vector,
#   - tilde_gamma_k  is the k-step-ahead MSE predictor coefficient vector,
#                    normalized to unit length: ||tilde_gamma_k|| = 1,
#   - lambda_k       are regularization weights chosen to enforce a
#                    monotonically increasing CCF from k = 0 to k = h.
#
# Since CCF(k) = b' * gamma_k / (||b|| * ||gamma_k||), the monotonicity
# condition CCF(k) > CCF(k-1) requires:
#
#   b' * tilde_gamma_k  >  b' * tilde_gamma_{k-1}
#
# This leads to the constraint system:
#
#   b' * (tilde_gamma_i - tilde_gamma_{i-1}) > 0,   i = 1, …, h,
#
# Specifically, our solution implements:
#
#   b' * (tilde_gamma_i - tilde_gamma_{i-1}) = beta,   i = 1, …, h,
#
# where beta > 0 is a prescribed (common) CCF increment enforcing the
# monotonically increasing profile. In principle, beta could be made
# lag-dependent (beta_i), but no natural or principled choice exists in
# general, so a common value is used for simplicity.
#
# Substituting the expression for b into the constraint system yields h
# linear equations in h unknowns (lambda_1, …, lambda_h):
#
#   (gamma_h + sum_{k=1}^{h} lambda_k * (tilde_gamma_k - tilde_gamma_{k-1}))' *
#       (tilde_gamma_i - tilde_gamma_{i-1}) = beta,   i = 1, …, h.
#
# Infeasibility arises in two ways, corresponding to Cases A and B above:
#
#   Case A) The constraint vectors (tilde_gamma_k - tilde_gamma_{k-1}) are
#           linearly dependent and the target vector (beta, …, beta) does not
#           lie in their column span. The system has no solution.
#
#   Case B) The system has a solution, but the implied target correlation
#           CCF(h) is non-positive. Since a predictor negatively correlated
#           with x_{t+h} is inadmissible, the problem is declared infeasible.
#
# A problem is therefore feasible if and only if:
#   (i)  A solution to the constraint system exists, and
#   (ii) The implied target correlation CCF(h) is strictly positive.


# ── ILLUSTRATIVE EXAMPLES OF IMPOSSIBLE PROBLEMS ─────────────────────────────

# Example 1: AR(1) DGP with positive autoregressive coefficient a1.
#
#   For any AR(1) process with a1 > 0, the h-step-ahead MSE predictor
#   coefficients satisfy:
#       gamma_h \propto gamma_{h+k}  for all h >= 0, k >= 0,
#   so that tilde_gamma_h = tilde_gamma_{h+k} for all k.
#
#   It follows that:
#       b' * (tilde_gamma_k - tilde_gamma_{k-1}) = 0  for all k = 1, …, h,
#   which means the CCF increment beta cannot be made strictly positive.
#   Consequently, the CCF cannot be made monotonically increasing from
#   k = h to k = h+1 for any h>=0. The problem is impossible and infeasible.
#
# Example 2: ARMA(1,1) DGP with positive autoregressive coefficient a1.
#
#   For lags h > 0 (strictly larger zero), the same proportionality holds:
#       gamma_h \propto gamma_{h+k}  for all h > 0, k >= 0,
#   so that tilde_gamma_h = tilde_gamma_{h+k} for h > 0.
#
#   As in Example 1, this implies:
#       b' * (tilde_gamma_k - tilde_gamma_{k-1}) = 0  for k = 1, …, h,
#   and the CCF cannot be made increasing from k = h to k = h+1.
#
#   However, unlike the AR(1) case, the MA component introduces an asymmetry
#   at k = 0: if the MA coefficient b1 < 0, the CCF may still increase from
#   k = 0 to k = 1. This means that the problem, while impossible under
#   Types I and II, may remain feasible under Type III when h > 0 and b1 < 0.
#   See Exercise 1 for a worked example.
#

# ── FEASIBILITY AND ARMA(p,q) STRUCTURE ──────────────────────────────────────

# For an ARMA(p,q) process, the effective dimension of the PCS constraint
# space is at most p + q. This is because for lags k > q, the autocovariances
# R(k) = gamma_0' * gamma_k satisfy a p-dimensional linear recurrence
# (the Yule-Walker equations), which renders higher-lag constraint vectors
# linearly dependent on lower-lag ones (see exercise 2).
#
# The feasibility outcome therefore depends on the number of linearly
# independent constraints imposed relative to the effective dimension p + q:
#
#   - Fewer than p+q independent constraints:
#     Residual degrees of freedom remain. The target correlation is always
#     positive and increases as fewer constraints are imposed, giving the
#     optimizer greater room to track the target — though possibly at the
#     cost of reduced look-ahead effectiveness. As in the case below, there
#     is no guarantee that the CCF peaks at k = h or that the peak height
#     is maximized at that lag.
#
#   - Exactly p+q independent constraints:
#     All available degrees of freedom are consumed. The target correlation
#     is fully determined by the constraints and may be non-positive, in
#     which case the problem is infeasible (Case B). If positive, the problem
#     is feasible, but there is no guarantee that the CCF peaks at k = h or
#     that the peak height is maximized at that lag.
#
#   - More than p+q independent constraints:
#     The constraint system is overdetermined. Feasibility depends on whether
#     the target vector (beta, …, beta) lies in the column space spanned by
#     the constraint vectors:
#       * If it does not, no solution exists and the problem is infeasible
#         (Case A).
#       * If it does, the constraints are simultaneously satisfiable. The
#         problem is then feasible if the implied target correlation CCF(h)
#         is strictly positive, and infeasible otherwise (Case B).

# ── ADDRESSING INFEASIBILITY VIA REGULARIZATION ──────────────────────────────

# Infeasible problems can be addressed via regularization, which penalizes
# departures from the constraints. When the problem is truly infeasible, these
# deviations do not vanish as the regularization weight grows, since the
# constraints cannot be satisfied regardless of how strongly they are enforced.
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
# target correlation, precisely because the relaxation from a rigid (strictly monotonic 
# prespecified path) CCF profile allows the optimizer to explore a richer solution space.

# ── POSSIBLE YET INFEASIBLE ──────────────────────────────────────────────────

# A  PCS problem may be possible yet infeasible in the following
# sense: a predictor exists whose cross-correlation function (CCF) peaks at
# k = h with a positive target correlation, and yet none of the three
# proposed constraint types (I, II, III) successfully identifies this solution.
# The reasons are as follows:
#
#   - Type I constraints may be overly restrictive: they impose a specific
#     structural pattern on the CCF across the full lag interval k = 0, ..., h
#     (e.g., monotonically increasing), which can exclude valid solutions when
#     a large regularization weight is applied.
#
#   - Type II constraints may be insufficiently restrictive: they require only
#     a local increase from lag k = h-1 to k = h, which may fail to enforce
#     the intended "peak at h" condition more broadly.
#
#   - Type III constraints may also be insufficiently restrictive: they require
#     only a positive average increase from k = 0 to k = h, which again may
#     fail to capture the "peak at h" solution in the general case.
#
# For this reason, we generally recommend Type I (more control) paired with a 
# moderate regularization weight (more flexibility) that permits controlled 
# departures from a predetermined strict CCF profile. This 
# relaxation frees up degrees of freedom that can then be directed toward 
# maximizing the objective function, ensuring that the look-ahead design 
# achieves optimal tracking of the target at horizon h while controlling more 
# firmly for the requested `peak at h' profile.

# ── EXAMPLES OVERVIEW ─────────────────────────────────────────────────────────

# Example 1 — Impossible but Feasible (PCS Type III)
#
#   The problem is feasible in the sense that CCF(h) > CCF(0), as required by
#   the Type III constraint, and the implied target correlation is positive.
#   However, the CCF peak occurs at k = 1 rather than at the requested target
#   horizon k = h = 12, which is unachievable for this DGP. The look-ahead
#   objective is therefore not met.
#
#   More strikingly, the resulting filter simultaneously degrades the
#   signal-to-noise ratio (i.e., amplifies noise) and introduces lag — a
#   doubly adverse outcome. This pathological behavior arises from imposing
#   the Type III constraint under structural conditions of the DGP that leave
#   insufficient degrees of freedom to address the forecasting problem
#   in any meaningful way.
#
# Example 2 — Impossible and Infeasible: Case A (PCS Type I, Default Settings)
#
#   Under default settings, PCS Type I imposes a prespecified monotonically 
#   increasing CCF profile from k = 0 to k = h. This is unachievable for the present DGP,
#   and the problem falls under Case A (overdetermined constraint system):
#
#   - The h = 12 constraint vectors (tilde_gamma_i - tilde_gamma_{i-1}) span
#     a column space of effective dimension 2 (reflecting the ARMA(1,1)
#     structure of the DGP), which is far smaller than h = 12.
#   - The target vector (beta, …, beta) of dimension h = 12 does not lie in
#     this 2-dimensional column space, so the constraint system has no
#     solution.
#   - Consequently, no filter coefficient vector b exists that simultaneously
#     satisfies all h = 12 monotonicity constraints.
#
# Example 3 — Impossible and Infeasible: Case B (PCS Type I, Non-Default Settings)
#
#   Under a particular non-default configuration, PCS Type I imposes a
#   monotonically increasing CCF profile from k = 0 to k = h that is
#   structurally achievable for this DGP. Nevertheless, the problem falls
#   under Case B, as the implied target correlation is non-positive:
#
#   - The h = 12 constraint vectors (tilde_gamma_i - tilde_gamma_{i-1}) again
#     span a 2-dimensional column space (ARMA(1,1) structure). Under this
#     non-default configuration, however, the target vector (beta, …, beta)
#     of dimension h = 12 does lie within this column space, so all h = 12
#     monotonicity constraints can be satisfied exactly.
#   - Despite the system being solvable, the target correlation CCF(h) implied
#     by this solution is negative, confirming that a strictly positive CCF
#     peak at k = h is unachievable — consistent with the impossibility of
#     the problem.

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────────────
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

# Fit an ARMA(1,1) model: parsimonious specification with adequate diagnostics.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)
a1<-arima.obj$coef[1:ar_order]
b1<-arima.obj$coef[ar_order+1:ma_order]

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

# For k > 0, the ACF satisfies the recurrence ACF(k+1) = a1 * ACF(k), meaning
# the DGP imposes a rigid linear structure on the autocorrelation function.
#
# For h > 0 and k >= 0, the MSE predictor coefficient vectors gamma_h and
# gamma_{h+k} are proportional, so their normalized counterparts satisfy:
#   tilde_gamma_h = tilde_gamma_{h+k}.
#
# Consequently, all normalized predictors at lags h, h+1, h+2, … are
# identical, and the constraint vectors (tilde_gamma_i - tilde_gamma_{i-1})
# collapse to zero for i > 1. The effective dimension of the constraint space
# is therefore at most 2: tilde_gamma_0 and tilde_gamma_1 are the only
# linearly independent directions, provided b1 != 0.



# ─────────────────────────────────────────────────────────────────────
# 1.3 MSE Benchmark
# ─────────────────────────────────────────────────────────────────────
# We consider a one-year ahead forecast horizon
h      <- 12       

# Truncate the Wold coefficients to length L to obtain the nowcast
# filter (gamma0).
gamma0 <- gamma[1:L]

# h-step-ahead MSE predictor (gammah):
# Shift gamma forward by h positions:
gammah <- gamma[h + 1:L]     



# ─────────────────────────────────────────────────────────────────────
# 1.4 PCS Type III Framework
# ─────────────────────────────────────────────────────────────────────
# The type III PCS  imposes decoupling of the predictor from (tilde_gamma_0-tilde_gamma_h).
# If h=1 then type III and I PCS coincide. However, here h=12, and therefore both 
# types differ.


gamma_constraint<-gamma0-gammah
#gamma_constraint<-gammah-gamma0
gamma_target<-gammah
max_lag<-0

ts.plot(gamma_constraint,main="PCS: gamma_constraint")

# Shifting the peak of the CCF from lag=0 to lead=-1 is obtained by 
# imposing at least full decoupling. We here consider different intermediate 
# values for the constraint parameter beta

# Note: we use the unitary DFP so that the constraint parameter reflect a 
# correlation
beta_vec<-c(0.8,0.6,0.3,0,-0.1)


cor_vec_mat<-b_mat<-lambda1_vec<-lambda2_vec<-tau_vec<-NULL
for (i in 1:length(beta_vec))
{ 
  beta<-beta_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b_obj<-unitary_DFP_func(gamma_constraint,gamma_target,beta)
  b<-b_obj$b0
  b_mat<-cbind(b_mat,b)
  lambda1_vec<-c(lambda1_vec,b_obj$lambda1)
  lambda2_vec<-c(lambda2_vec,b_obj$lambda2)
# Compute shift at frequency zero  
  tau_vec<-c(tau_vec, sum((0:(L-1)) * b) / sum(b))
  
  
  # Compute CCF of PCS predictors  
  cor_vec_mat<-cbind(cor_vec_mat,compute_acf_at_lags_zero_delta_func(max_lag,h,b_mat[,i],gamma0)$cor_vec)
}

# ─────────────────────────────────────────────────────────────────────
# 1.5 Routine Checks
# ─────────────────────────────────────────────────────────────────────


# --- Check 1: verify unity length ---

apply(b_mat^2,2,sum)


# --- Check 2: verify that the PCS conntraint is met ---

# Compute the correlation with gamma_constraint. In the DFP (previous tutorials)
# gamma_constraint = gamma0 is the nowcast. Here, gamma_constraint = gamma0 - gammah
# Note that b%*%b=1 (unit length) so that we do not need to scale with b%*%b to obtain the correlation     
correlation_0<-t(b_mat)%*%gamma_constraint/as.double(sqrt(gamma_constraint%*%gamma_constraint))
# This difference should vanish
correlation_0-beta_vec


# CHECK 3 — Sign/orientation preservation: If the sum of filter weights 
# is strictly positive, the DFP does not reverse
# the direction (sign) of a trend signal.
apply(b_mat, 2, sum)

# CHECK 4 — Positive Target correlation

t(b_mat)%*%gammah/as.double(sqrt(gammah%*%gammah))

# Check 5 - Minimum MSE ----

# MSE of unitary (not optimally scaled) PCS
apply((b_mat-gammah)^2,2,sum)

# Compute optimal MSE scaling
optimal_mse_scaling<-as.vector(t(b_mat)%*%gammah/apply(b_mat^2,2,sum))

# Rescale PCS:
b_mat_mse<-t(t(b_mat)*(optimal_mse_scaling))

# The optimally scaled PCS is obtained by minimizing the Mean Squared Error (MSE).
# Note: The MSE values computed here assume a standardized white noise input,
# meaning the innovation variance (sigma^2 from the ARMA(1,1) model) is ignored.
apply((b_mat_mse-gammah)^2,2,sum)

# We now assemble all relevant predictors, skipping the fully decoupled design
# which is unusable in this example.
filter_mat<-cbind(gamma0,gammah,b_mat)
colnames(filter_mat)<-c("Nowcast","MSE",paste("PCS ",beta_vec,sep=""))



# ─────────────────────────────────────────────────────────────────────
# 1.6 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo     <- c("black", "green", rainbow(ncol(filter_mat) - 2))

# ── Left panel: filter coefficient profiles ───────────────────────────
# Display original nowcast and MSE as well as unit-length DFP filter coefficients.
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
max_lag <- 20
mplot  <-ccf_mat <- NULL

for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot, compute_ccf_func(filter_mat[, i], gamma0)[L - 1 + 1:max_lag])
colnames(mplot) <- colnames(filter_mat)
rownames(mplot)<-paste("CCF at lead: ",-1+1:max_lag,sep="")
ccf_mat<-mplot

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

#  PROBLEM IMPOSSIBLE BUT FEASIBLE.

# Outcome:
# Filter coefficients:
# -Complying with a positive CCF slope is only possible when flipping the sign of the predictor.
#   This is because of the structural constraints imposed by the data generating process
#   gamma_h is proportional to gamma_htilde whenever h,htilde>=1. 
#   There are no degrees of freedom left for optimization.
# -The sign flip explians the negative target correlation when beta<0.

# CCF
# - A positive slope can be enforced when flipping the sign of the predictor.
# - As a result, the target correlation is negative,


# ─────────────────────────────────────────────────────────────────────
# 1.7 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.7.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals

if (F)
{
# Simulated data ARMA(1,1): empirical CCFs converge to expected values.
  len<-1000000
  set.seed(462)
  eps<-x<-x_filt<-rnorm(len)
  for (i in 2:len)
    x[i]<-a1*x[i-1]+eps[i]+b1*eps[i-1]
}

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 1.7.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# As in exercise 2, Tutorial 9, the PCS predictors actually LAG the MSE predictor.



# ─────────────────────────────────────────────────────────────────────
# 1.8 Amplitude and Time-Shifts
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
# Scale amplitudes for better visual inspection.
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

# Amplitude functions:
#   - Enforcing a smaller beta reduces low-frequency content (amplitude shrinks at lower frequencies)
#     and increases noise leakage (amplitudes grow at higher frequencies).
#   - The signal to noise ratio is negatively affected.
#
# Time shifts:
#   - Remarkably, the worse signal to noise ratio does not imply a lower time-shift in this example.
#   - This result can be constrasted with exercise 2, tutorial 9, where the worse signal-to-noise ratio
#     obtained a smaller shift at frequency zero (a lead) but a larger lag at the relevant business-cycle frequencies.
#   - So in way the type III PCS obtained worsened both characteristics in this case (it is not on the efficient ATS frontier).

# Explanation: to obtain CCF(h)>CCF(0) under the implied DGP constraints, the predictor
# reduces weight on the most recent innovation which increases the lag.

# ─────────────────────────────────────────────────────────────────────
# 1.9 AR Form: AR Inversion of ARMA(1,1)
# ─────────────────────────────────────────────────────────────────────

# AR inversion:

ar_inv <- -ARMAtoMA(ar = -arima.obj$coef[ar_order + 1:ma_order], 
                    ma = -arima.obj$coef[1:ar_order], lag.max = L)
# AR-filter
theta<-c(1,-ar_inv)

# Verify the approach via a known identity:
# Convolving the AR inversion with the Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros). 
conv_two_filt_func(xi, theta)$conv[1:10]


# Visualise theta: 
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR inversion")

# The first weight is always 1 (the weight assigned to x_t)
# The other weights are decaying: the decay is very regular
theta[2:L]/theta[1:(L-1)]
# First element matches a1+b1 (up to sign)
arima.obj$coef[ar_order + 1:ma_order]+arima.obj$coef[1:ar_order]
# After that, the exponential decay matches b1 (up to sign)
arima.obj$coef[ar_order + 1:ma_order]

# Having confirmed the identity, we now convolve the AR operator with
# the MSE and DFP predictors (in MA form) to obtain their AR form equivalents.


# ─────────────────────────────────────────────────────────────────────
# 1.10 Convolution of the AR inversion with the Predictors
# ─────────────────────────────────────────────────────────────────────

# a. MSE predictor: convolve the AR operator with the predictors.

# Check: convolution of theta with gamma0 should be the identity: smaller 
# deviations become vanishing with increasing L (length of finite MA and AR inversions)
conv_two_filt_func(theta, gamma0)$conv[1:10]

# b. DFP predictors
filter_mat_ar<-NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar<-cbind(filter_mat_ar,conv_two_filt_func(theta, filter_mat[,i])$conv)

colnames(filter_mat_ar)<-colnames(filter_mat)

# Check: the first column (corresponding to the nowcast gamma0) should be the identity.
filter_mat_ar[1:10,1]

# ─────────────────────────────────────────────────────────────────────
# 1.11 Analysis and Plot of DFP Predictors in AR Form
# ─────────────────────────────────────────────────────────────────────
#

# Assign colors
colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- 10
par(mfrow=c(1,1))
# Plot the first first_lags AR coefficients of each predictor.
ts.plot(
  filter_mat_ar[1:first_lags, ],
  col  = colo,
  main = "DFP (MSE-Optimal) Predictors in AR Form",
  lty=c(2,2,rep(1,ncol(filter_mat)-2)),lwd=c(1,2,rep(1,ncol(filter_mat)-2)))
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)







# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: Impossible and Infeasible, Case A (PCS Type I)
# ════════════════════════════════════════════════════════════════════




# The same empirical framework as Exercise 1 but using type I) PCS
# The problem is infeasible because the constraint vectors (gamma_k - gamma_{k-1}) are not linearly
#     independent and the right-hand side vector (beta, …, beta) does not lie in the column
#     space spanned by the constraint vectors (Case A of Infeasibility).





# ─────────────────────────────────────────────────────────────────────
# 2.1 DGP Structural Constraints on PCS (and DFP) Solution Space
# ─────────────────────────────────────────────────────────────────────

# For h>=1 gammah<-a1^(h+0:L)
gamma_mat<-gamma0
for (i in 1:(L-1))
{
  gamma_i<-xi[i+1:L]
  gamma_mat<-cbind(gamma_mat,gamma_i)
}  

eigenvalues<-eigen(gamma_mat)$values

eigenvalues

# Rank of constraint system
which(abs(eigenvalues)>10^{-10})

# The rank is smaller than the number of constraints and therefore the right hand constraint 
# vector (beta,...,beta) can lie outside the column space (which is the case here).


# Note about exercise (Type III PCS):
# Type III imposes a single constraint of the type b' * (gamma_h-gamma_{0}). This can be solved 
# exactly. Of the available 2 degrees of freedom, one degree of freedom is left for 
# optimization: this is the degree of freedom exploited in exercise 1 above (without much success, since the resulting 
# PCS is lagging at business-cycle frequencies).

# Note about Type I or II with h>1:
# gammah and gamma_{h-1} are linearly dependent and therefore the room spanned 
# by gammah and (gamma_h-gamma_{h-1}) is one-dimensional. Hence the solution 
# of the PCS must be along gammah, either in the same or opposite direction.
# There is no room left for optimization.

# ─────────────────────────────────────────────────────────────────────
# 2.2 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)

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
lambda <- 10^10


# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(Delta, xi, L, beta, lambda)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Here, increasing lambda does not decrease arbitrarily the deviations: 
  # the problem is not feasible (the constraints cannot be solved exactly).
  # The vector (beta,...,beta) is not in the column space of the constraint matrix.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 2.4 Routine Checks
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

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah

# Infeasibility: the constraints with negative slope beta<0 can be enforced, 
# but the target correlation turns negative.

# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,  b_mat)
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
# Filter coefficients:
# -Complying with a positive CCF slope is only possible when flipping the sign of the predictor.
#   This is because of the structural constraints imposed by the data generating process
#   gamma_h is proportional to gamma_htilde whenever h,htilde>=1. 
#   There are no degrees of freedom left for optimization.
# -The sign flip explains the negative target correlation when beta<0.

# CCF
# - A positive slope can be enforced when flipping the sign of the predictor.
# - As a result, the target correlation is negative, too.

# Note: for h=1 PCS type I and II are feasible because gamma0 and gamma1 are not collinear. But the solution is lagging.
# For h>1 neither type I nor II are feasible, due to collinearity of gammah for h>=1.
# Type III is feasible irrespetive of h because gamma0 and gammah are not collinear. But the filter is useless (poor signal noise ratio AND higher lag)


# Main Take-Aways
# Type III PCS is feasible but useless (lagging)
# Type I and II PCS are infeasible: enforcing the constraints leads to a negative target correlation.

# Note: shifting the peak from horizon h>=1 to h+k, k>0, is not meaningful because 
# gamma_h and gamma_{h+k} are collinear: having the peak located at h or h+k `does the same',
# i.e. the peak location is irrelevant.









# ════════════════════════════════════════════════════════════════════
# 3. The constraints can be met (but the target correlation is still negative)
# ════════════════════════════════════════════════════════════════════
# Same as exercise 2 but (beta, …, beta) lies in the column
#     space spanned by the constraint vectors. However, the target correlation is negative (Case B of Infeasibility).


# ─────────────────────────────────────────────────────────────────────
# 3.1 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.2, -0.1, 0, 0.1, 0.2, 0.3)

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
# 3.2 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value
initialize_with_null=F
scaled_constraints=T
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(Delta, xi, L, beta, lambda,initialize_with_null,scaled_constraints)
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))

# ─────────────────────────────────────────────────────────────────────
# 3.3 Routine Checks
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

# ── Check 3: Positive target covariance ──────────────────────────────
# Confirms that each PCS predictor has a positive inner product with the
# h-step-ahead MSE predictor, i.e., a positive target correlation at lag h.
t(b_mat) %*% gammah


# Assemble all filters (nowcast, MSE references, and PCS variants) into a
# single matrix for joint plotting and comparison.
filter_mat <- cbind(gamma0, gammah,  b_mat)
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






# ════════════════════════════════════════════════════════════════════
# Exercise 4: Relax Strong Regularization
# ════════════════════════════════════════════════════════════════════












