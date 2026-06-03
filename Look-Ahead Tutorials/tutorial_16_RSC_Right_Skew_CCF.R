# ══════════════════════════════════════════════════════════════════════════════
# TUTORIAL 16 — RSC : RIGHT SKEWING THE CCF: 
#               APPLICATION TO THE HARDEST LOOK AHEAD FORECAST PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TUTORIAL 16 — DFP II: DECOUPLE FROM PAST 
# ══════════════════════════════════════════════════════════════════════════════



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
#   CCF(k) = (b' %*% gamma_k) / (||b|| * ||gamma_0||) = a1^k * CCF(0).
#
# For a1 > 0, the CCF decays monotonically and exponentially from its peak at
# k=0. This pattern is rigidly enforced by the DGP on every predictor b via the
# Yule-Walker equations. Consequently, reshaping the CCF according to any of the
# PCS constraint types above is generally impossible (unless one merely replicates
# the original AR(1) profile).
#
# However, for negative lags k=-1,-2,...
#
#   CCF(k) = (b[-k+(1:L)]' %*% gamma_0) / (||b|| * ||gamma_0||) 
#
# This expression depends on the predictor b and the (negative) lag k and hence
# can be controlled to some extent by b, irrespective of gamma_0 (assuming gamma_h \neq 0).

#───────────────────────────────────────────────────────────────────────────────

# ── DFP II: DECOUPLE FROM PAST  ───────────────────────────────────────────────

# Main Ideas

# 1. Decouple from past addresses the CCF at negative lags k=-1,-2,...
#
#   CCF(k) = (b[-k+(1:L)]' %*% gamma_0) / (||b|| * ||gamma_0||) 
#
#   This expression depends on the predictor b and the (negative) lag k and hence
#   can be controlled to some extent by b. 

# 2. Specifically, we aim at controlling 

#     \sum_{k=-l_start}^{l_end} CCF(k) \propto   
#     \sum_{k=-l_start}^{l_end} b[-k+(1:L)]' %*% gamma_0 =
#     b' %*% Sigma %*% gamma_0

#   Where Sigma is the integrator operator of order l_end-l_start, see examples below.  

# 3. Sigma %*% gamma_0 increases the rank of the constraint system: even if xi is 
#    rank one (AR(1)-process), so that (gamma_0, gamma_h) are linearly dependent for 
#    all h, the integrator of order l_end - l_start augments the system's rank, i.e. 
#    Sigma %*% gamma_0 and gamma_h are linearly independent (assuming gamma_h \neq 0).

# 4. DFP II Criterion: Given linear independence, the criterion
#
#     max b' %*% gamma_h
#     b' %*% Sigma %*% gamma_0 = alpha_0
#
#     is feasible, assuming gamma_h \notpropto Sigma %*% gamma_0 (linear dependence 
#     would require either periodicity or a very unlikely DGP structure).
#     Note that if xi is AR(1) then clearly gamma_h and Sigma %*% gamma_0 are linearly independent.

# 5.  Controlling  \sum_{k=-l_start}^{l_end} CCF(k)  by making is smaller decouples 
#     the predictor from the past of the series: decoupling from the paste pushes 
#     into the future.


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
# One approach to affect the left tail are perturbation based methods, see Tutorial 14. 
# Here we propose an alternative more direct and fundamental approach, which does 
# not rely on an extraneous perturbation profile. 



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
#     or inverting the contribution of negative lags, i.e., by decoupling from the past.

mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[5])
ccf(mplot_ccf[,1],mplot_ccf[,6],main=colnames(mplot_ccf)[6])
ccf(mplot_ccf[,1],mplot_ccf[,7],main=colnames(mplot_ccf)[7])


# Main Take-Away from Predictor Outputs
# - Increasing the slope (moving from blue to violet tones) is not the same as
#   classic mean reversion, where the predictor tends to `naively' return to the mean
#   when offset. In AR(1) multi-step-ahead (non-standardised) MSE predictors, the
#   tendency is to pull back toward zero irrespective of the history (as seen in tentacle plots).
#   - After standardization, the MSE predictors overlap and are non-informative (juest replicate the latest data point).
#
# - After standardization (emphasising target correlation rather than MSE),
#   the more forward-looking perturbated designs do not simply mean-revert. Instead, they
#   appear left-shifted and can, in some cases, enforce departures from the mean
#   (e.g., during stronger up- or down-swings).
#   This behavior arises from decoupling the predictor from the recent past:
#   moving in the direction opposite to recent history. Since excessive decoupling can
#   even lead to sign inversion, it remains important to maximize the target
#   correlation as in DFP or PCS.





# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2:  DFP II (Right-Skewing the CCF). 
#   The Role of the Integrator (Sigma)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


# Remark: the optimization criterion relies on a `decouple from paste' constraint. 
# The resulting problem can be solved in the original DFP framework. However, 
# in contrast to the present decoupling, based on gamma_0, the `decouple from paste`
# problem relies on decoupling from Sigma %*% gamma_0. 

# ─────────────────────────────────────────────────────────────────────
# 2.1 Set-Up DFP II 
# ─────────────────────────────────────────────────────────────────────

# Lag support: 
# - Consider left tail of CCF from k = -l_start to k = l_end.
# - Find DFP II predictor such CCF(k) is pulled down: right-skewed CCF.
# To illustrate and understand the role of the integrator we here select 
# particular (not necessarily practically relevant) l_start and 
# l_end

l_start<-4
l_end<-8
if (l_start>=l_end)
{
  print("l_start must be smaller than l_end")
  l_start<-l_end+1
}
# Integrator Sigma
Sigma<-matrix(rep(0,L^2),ncol=L)
for (i in (l_start+1):l_end)
  Sigma[i,(1+l_start):i]<-1
if (l_end<L)
{
  for (i in (l_end+1):L) #i<-l_end+1
    Sigma[i,((i-l_end+l_start)+1):i]<-1
}
gamma_constraint<-(Sigma%*%c(rep(0,l_start),gamma0[1:(L-l_start)]))



par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector for controlling the right-skweness of the CCF")
abline(h = 0)



# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor gammah.
# Purpose: mse_coup serves as a natural upper bound for the DFP constraint,
# i.e., any effective decoupling should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, each strictly smaller than mse_coup.
# Smaller (more negative) values enforce a progressively stronger right skewing 
# of  the CCF (decoupling from the paste).  
# Note: since the predictors are not normalized (||b|| != 1), the
# rule is not exact — alpha0 < mse_coup does not guarantee stronger decoupling
# of b from gamma_constraint — but it serves as a useful practical proxy.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.5,-1,-2,-3,-4)

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
  
  # Compute MSE-DFP II predictor with modified constraint vector
#  b <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
#  b <- regularized_dfp_func(gamma_constraint, gammah, alpha0,lambda)$b
  
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
                          paste0("DFP II ", round(alpha0_vec, 2)))



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
max_lag<-l_end
ccf_mat<-NULL
for (i in 1:ncol(filter_mat))
  ccf_mat<-cbind(ccf_mat,compute_acf_at_lags_zero_delta_func(
    max_lag, h, filter_mat[,i], xi)$cor_vec)
rownames(ccf_mat)<--max_lag - 1 + 1:nrow(ccf_mat)
colnames(ccf_mat)<-colnames(filter_mat)
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

# Cross-check the decoupling constraint:
# 1. Select CCF in decoupling range l_start to l_end: note that CCF starts at 
#    max_lag=l_end above (first entry is l_end)
ccf_decouple<-matrix(ccf_mat[1+1:(l_end-l_start),],nrow=(l_end-l_start))
colnames(ccf_decouple)<-colnames(ccf_mat)
# 2. Retain only DFP designs (remove nowcast and benchmarks)
ccf_decouple_check<-ccf_decouple[,which("DFP"==substr(colnames(ccf_decouple),1,3))[1]:ncol(ccf_decouple),drop=F]
# 3. Apply integrator column wise and compare with alpha_0 scaled by lengths of b and xi: the differences should vanish:
#     Note: we use xi instead of gamma0 when computing the CCF. Therefore we must scale with length of xi (not gamma0)
# The differences should vanish (note: there's a 0/0 singularity when alpha0=0: just ignore)
apply(ccf_decouple_check,2,sum)-(alpha0_vec/(sqrt(apply(b_mat^2,2,sum))*sqrt(sum(xi^2))))


# ── Outcomes ─────────────────────────────────────────────────────────
# Left panel (filter coefficients):
#   - For lags <= l_start, the predictors are unaffected and follow the original 
#     AR(1) pattern. 
#   - Decreasing alpha0 accelerates the decay of the predictor weights for k > l_start, which 
#     turn negative for sufficiently small alpha0. 
#   - l_start controls for decay of the coefficients at the start: for lags >= l_start, 
#     coefficients decay faster.
#   - l_end controls the steepness of the decay: 
#       - For l_start=0 and l_end = 1 the constraint vector is AR(1) and thus the 
#         system is infeasible (constraint and target are collinear)
#       - For l_start = 0 and l_end = 2 the constraint imposes a single lag (discontinous)
#         decay: this case includes the IDENTITY filter (which is used as a benchmark: dark green line)
#         as special case.
#   - For increasing l_end, the decay of predictor weights operates in the interval [l_start, l_end] 
#     i.e., the decay is longer, more gradual and smoother.
#
# Right panel (CCFs):
#   - The MSE predictor maximizes the CCF at the forecast horizon h=12.
#   - Enforcing right-skewness by decoupling from the paste works as intended: as alpha0 decreases, the 
#     left tail of the CCF drops markedly while the right tail is optimized 
#     for maximal CCF at the forecast horizon h=12. 
#   - Duality: no other linear predictor can increase decoupling from paste (skewness) 
#     for given target correlation (efficient frontier).
#   - l_start and l_end control the lag interval over which the asymmetry of the CCF is imposed : 
#       - The CCF skewness is obtained by pulling down the CCF on average (through the integrator Sigma) 
#         in the interval (-l_start):(-l_end). 





# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3:  DFP II and the Identity as Special Case
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Note: Exercise 1 must be run before this exercise, as it initialises the
# empirical framework (process specification, filter length, forecast horizon,
# and MA coefficient vector) required by all subsequent exercises.
# ─────────────────────────────────────────────────────────────────────────────


# We here select l_start<-1 and l_end<-2:
#  - The resulting Sigma integrator simplifies to a simple backshift by one time 
#     unit, i.e., we decouple the predictor from the lagged AR(1).
#  - Full decoupling is obtained by the identity (up to MSE optimal scaling), i.e., 
#     the identity is a special case of the DFP II.
#  -Explanation: the lagged DGP is epsilon_{t-1} + a1 * epsilon_{t-2} + a1^2 * epsilon_{t-3} + ...
#    The DFP II predictor b = epsilon_t, assigning full weight to epsilon_t, is thus 
#    fully decoupled (orthogonal).
#  -Assigning full weight to epsilon_t signifies a left-shift over the MSE predictor 
#     (looking ahead), but the predictor is very noisy.

# ─────────────────────────────────────────────────────────────────────
# 3.1 Set-Up DFP II 
# ─────────────────────────────────────────────────────────────────────

# Lag support: 
# - Consider left tail of CCF from k = -l_start to k = l_end.
# - Find DFP II predictor such CCF(k) is pulled down: right-skewed CCF.
l_start<-1
l_end<-2
if (l_start>l_end)
{
  print("l_start must be smaller equal l_end")
  l_start<-l_end
}


# Integrator Sigma
Sigma<-matrix(rep(0,L^2),ncol=L)
for (i in (l_start+1):l_end)
  Sigma[i,(1+l_start):i]<-1
if (l_end<L)
{
  for (i in (l_end+1):L) #i<-l_end+1
    Sigma[i,((i-l_end+l_start)+1):i]<-1
}



gamma_constraint<-(Sigma%*%c(rep(0,l_start),gamma0[1:(L-l_start)]))


# The decoupling vector is the lagged DGP:
par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector for controlling the right-skweness of the CCF")
abline(h = 0)





# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor gammah.
# Purpose: mse_coup serves as a natural upper bound for the DFP constraint,
# i.e., any effective decoupling should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, each strictly smaller than mse_coup.
# Smaller (more negative) values enforce a progressively stronger right skewing 
# of  the CCF (decoupling from the paste).  
# Note: since the predictors are not normalized (||b|| != 1), the
# rule is not exact — alpha0 < mse_coup does not guarantee stronger decoupling
# of b from gamma_constraint — but it serves as a useful practical proxy.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.1,-0.2,-0.3,-0.5)

# Display alpha0_vec: the last (negative) entry indicates that the DFP
# constraint enforces stronger decoupling than the MSE predictor gammah,
# which should shift the CCF peak to the right from k = 0 to k = h = 1.
alpha0_vec


# ─────────────────────────────────────────────────────────────────────
# 3.2 Decoupling over the alpha0-Grid
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
  
  # Compute MSE-DFP II predictor with modified constraint vector
  #  b <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  #  b <- regularized_dfp_func(gamma_constraint, gammah, alpha0,lambda)$b
  
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
# 3.3 Routine Checks
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
                          paste0("DFP II ", round(alpha0_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 3.4 Plots and Performance Summary
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
max_lag<-l_end
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
#   - The lag 0 weight is fixed (same as MSE) since l_start = 1 > 0. 
#   - Decreasing alpha0 successively downweights the AR(1) profile at lags >= 1 which  
#     turns negative for sufficiently small alpha0. 
#   - For alpha0 = 0 (cyan) the identity is obtained.
#
# Right panel (CCFs):
#   - All, except the last predictor retain a positive target correlation. 
#   - Decoupling is enforced at lag k = -1: the latter can turn negative while 
#     keeping the target correlation into positive territory.



# ─────────────────────────────────────────────────────────────────────
# 3.5 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 3.5.1 Forecast Comparison ────────────────────────────────────────


# Generate a long white-noise series for a reliable empirical evaluation
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

scale(y_out_mat)[,1:2]

colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))


# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 390
enf <- 430
anf <- 500
enf <- 600

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   - As alpha0 decreases, the predictors become noisy and strongly mean-reverting:
#     the predictors are unable to track episodic swings away from the center line. 

# ── 3.5.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(3, 2))

select_vec<-c(2,4,7,9,11,13)

for ( i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main = paste(colnames(y_out_mat)[i],sep=""))
}  

# Compute target correlations and smoothness
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
  smooth<-c(smooth,sqrt(mean(diff(diff(scale(y_out[,i])))^2)))
names(smooth)<-colnames(y_out_mat)
# The predictors are becoming increasingly noisy with decreasing alpha_0
smooth


# Left-shift of trough: 
#  -Curve: average distance between zero-crossings
#  -Trough in curve: lag at which zero-crossings of DFP II and MSE(12) are closest 
#    i.e. at which series are aligned.
# -lead (negative) or lag (positive) of DFP II compared to MSE(12): at trough at -k 
#   indicates a lead of k time units of DFP II over MSE(12).
# -Increased skewness (smaller alpha0) generates a left-shift (larger lead) of DFP II

max_lead   <- 12
par(mfrow = c(3, 2))
plot_vec<-NULL
for (i in select_vec)
{
  xy_mat <- cbind(y_out[,i],y_out[,"MSE(12)"])
  colnames(xy_mat)<-c(colnames(y_out)[i],"MSE(12)")
  tau<-compute_min_tau_func(xy_mat, max_lead)
  tau$min_tau_plot
}



if (F)
{
  
  omega<-pi/20
  x<-cos((1:1000)*omega)
  y<-cos((1+1:1000)*omega)
  
  xy_mat<-cbind(x,y)
  max_lead<-40
  vicinity=4
  last_crossing_or_closest_crossing=F
  outlier_limit=max_lead
  tau<-compute_min_tau_func(xy_mat, max_lead,vicinity,last_crossing_or_closest_crossing,outlier_limit)
  
}


# Time-Shifts

K<-600
plot_T<-F
shift_mat<-NULL
for (i in select_vec)
{
  
  b1<- filter_mat[,i]
  shift_mat<-cbind(shift_mat,amp_shift_func(K,b1, plot_T)$shift)   # time-shift for lagged filter (b1)
  
}
colnames(shift_mat)<-colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat
colnames(mplot) <- colnames(shift_mat)

plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: Whole frequency interval",
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

# From 0 to pi/6 
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "In interval [0,pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# From 0 to pi/6 and trimmed to [-10,10]
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Trimmed to -10,10",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()







# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4:  DFP II (Right-Skewing the CCF). 
#   A Special Case
# ══════════════════════════════════════════════════════════════════════════════



# ─────────────────────────────────────────────────────────────────────
# 4.1 Set-Up DFP II 
# ─────────────────────────────────────────────────────────────────────

# Lag support: 
# - Consider left tail of CCF from k = -l_start to k = l_end.
# - Find DFP II predictor such CCF(k) is pulled down: right-skewed CCF.
l_start<-0
l_end<-h
if (l_start>l_end)
{
  print("l_start must be smaller equal l_end")
  l_start<-l_end
}


# Integrator Sigma
Sigma<-matrix(rep(0,L^2),ncol=L)
for (i in (l_start+1):l_end)
  Sigma[i,(1+l_start):i]<-1
if (l_end<L)
{
  for (i in (l_end+1):L) #i<-l_end+1
    Sigma[i,((i-l_end+l_start)+1):i]<-1
}



gamma_constraint<-(Sigma%*%c(rep(0,l_start),gamma0[1:(L-l_start)]))



par(mfrow=c(1,1))
ts.plot(gamma_constraint,
        main = expression(gamma[constraint] == Sigma * gamma[0]),
        xlab = "Lag", ylab = "",
        sub = "Algebraic constraint vector for controlling the right-skweness of the CCF")
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
# - While the concepts and ideas are reminiscent, and the link intriguing, 
#   there is also a notable difference:
#   in Tutorial 12, Exercise 1.4, the target is the convolved (integrated from 
#   original monthly to yearly-growth) DGP, whereas here the target remains the 
#   original (un-convolved) AR(1) process and the constraint vector is built to 
#   affect the CCF at NEGATIVE lags (the exponentially decaying profile of the CCF 
#   at positive lags is immutable for the present AR(1): infeasible forecast problem).





# Baseline coupling: inner product of gammah with gamma_constraint under the
# unconstrained MSE predictor gammah.
# Purpose: mse_coup serves as a natural upper bound for the DFP constraint,
# i.e., any effective decoupling should enforce a coupling strictly below this.
mse_coup <- as.double(gammah %*% gamma_constraint)

# Sequence of decoupling levels alpha0, each strictly smaller than mse_coup.
# Smaller (more negative) values enforce a progressively stronger right skewing 
# of  the CCF (decoupling from the paste).  
# Note: since the predictors are not normalized (||b|| != 1), the
# rule is not exact — alpha0 < mse_coup does not guarantee stronger decoupling
# of b from gamma_constraint — but it serves as a useful practical proxy.
alpha0_vec <- c(mse_coup / 1.5^(1:5), 0, -0.5,-1,-2,-3,-4)

# Display alpha0_vec: the last (negative) entry indicates that the DFP
# constraint enforces stronger decoupling than the MSE predictor gammah,
# which should shift the CCF peak to the right from k = 0 to k = h = 1.
alpha0_vec


# ─────────────────────────────────────────────────────────────────────
# 4.2 Decoupling over the alpha0-Grid
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
  
  # Compute MSE-DFP II predictor with modified constraint vector
  #  b <- compute_mse_dfp(alpha0, gamma_constraint, gammah)$b0
  b <- mse_dfp_from_alpha0_func(gamma_constraint, gammah, alpha0)$b
  #  b <- regularized_dfp_func(gamma_constraint, gammah, alpha0,lambda)$b
  
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
# 4.3 Routine Checks
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
                          paste0("DFP II ", round(alpha0_vec, 2)))



# ─────────────────────────────────────────────────────────────────────
# 4.4 Plots and Performance Summary
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
max_lag<-l_end
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
#   - Decreasing alpha0 accelerates the decay of the predictor weights, which 
#     turn negative for sufficiently small alpha0. 
#   - l_start controls for decay of the coefficients at the start: for lags >= l_start, 
#     coefficients decay faster.
#   - l_end controls the steepness of the decay: 
#       - For l_start=0 and l_end = 1 the constraint vector is AR(1) and thus the 
#         system is infeasible (constraint and target are collinear)
#       - For l_start = 0 and l_end = 2 the constraint imposes a single lag (discontinous)
#         decay: this case includes the IDENTITY filter (which is used as a benchmark: dark green line)
#         as special case.
#   - For increasing l_end, the decay of predictor weights operates in the interval [l_start, l_end] 
#     i.e., the decay is longer, more gradual and smoother.
#
# Right panel (CCFs):
#   - The MSE predictor maximizes the CCF at the forecast horizon h=12.
#   - Enforcing righ-skewness works as intended: as alpha0 decreases, the 
#     left tail of the CCF drops markedly while the right tail is optimized 
#     for maximal CCF at the forecast horizon h=12. 
#   - No other linear predictor can increase skewness (as implied by the constraint) 
#     for given target correlation (efficient frontier).
#   - l_end controls the lag interval over which the asymmetry of the CCF is imposed : 
#       - For l_start=0 and l_end = 1 the constraint vector is AR(1) and thus the 
#         system is infeasible (constraint and target are collinear)
#       - The CCF asymmetry is obtained by pulling down the CCF on average (through the integrator Sigma) 
#         in the interval [l_start, l_end]. 



# ─────────────────────────────────────────────────────────────────────
# 4.5 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 4.5.1 Forecast Comparison ────────────────────────────────────────


# Generate a long white-noise series for a reliable empirical evaluation
set.seed(1)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes the linear filter sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

scale(y_out_mat)[,1:2]

colo <- c("black", "green","darkgreen", rainbow(ncol(b_mat)))


# Plot a short excerpt to visually compare the temporal alignment of each predictor
anf <- 390
enf <- 430
anf <- 500
enf <- 600

par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor outputs (standardised): excerpt",
        col = colo, xlab = "Time", ylab = "")
abline(h = 0)
for (i in 1:ncol(filter_mat))
  mtext(colnames(filter_mat)[i], col = colo[i], line = -i)

# Outcome:
#   - As alpha0 decreases, the predictor output shifts progressively to the left 
#     (looks further ahead) relative to the MSE predictor. This visual lead is confirmed quantitatively by the
#     empirical CCFs below.
#   - Strong skewing (small alpha0) generally lags the identity, but the latter 
#       - Is a special of skewing, when l_start = 0 and l_end = 2
#       - Is much noisier; see below.

# ── 4.5.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 1.5 is
# reproduced in finite-sample data.

par(mfrow = c(3, 2))

select_vec<-c(2,4,7,9,11,13)

for ( i in select_vec)
{
  ccf(na.exclude(y_out_mat[, 1]),
      na.exclude(y_out_mat[, i]),
      lag.max = h, plot = TRUE,
      main = paste(colnames(y_out_mat)[i],sep=""))
}  

# Compute target correlations and smoothness
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
  smooth<-c(smooth,sqrt(mean(diff(diff(scale(y_out[,i])))^2)))
names(smooth)<-colnames(y_out_mat)
smooth


# Left-shift of trough: 
#  -Curve: average distance between zero-crossings
#  -Trough in curve: lag at which zero-crossings of DFP II and MSE(12) are closest 
#    i.e. at which series are aligned.
# -lead (negative) or lag (positive) of DFP II compared to MSE(12): at trough at -k 
#   indicates a lead of k time units of DFP II over MSE(12).
# -Increased skewness (smaller alpha0) generates a left-shift (larger lead) of DFP II

max_lead   <- 8
par(mfrow = c(3, 2))
plot_vec<-NULL
for (i in select_vec)
{
  xy_mat <- cbind(y_out[,i],y_out[,"MSE(12)"])
  colnames(xy_mat)<-c(colnames(y_out)[i],"MSE(12)")
  tau<-compute_min_tau_func(xy_mat, max_lead)
  tau$min_tau_plot
}



if (F)
{
  
  omega<-pi/20
  x<-cos((1:1000)*omega)
  y<-cos((1+1:1000)*omega)
  
  xy_mat<-cbind(x,y)
  max_lead<-40
  vicinity=4
  last_crossing_or_closest_crossing=F
  outlier_limit=max_lead
  tau<-compute_min_tau_func(xy_mat, max_lead,vicinity,last_crossing_or_closest_crossing,outlier_limit)
  
}


# Time-Shifts

K<-600
plot_T<-F
shift_mat<-NULL
for (i in select_vec)
{
  
  b1<- filter_mat[,i]
  shift_mat<-cbind(shift_mat,amp_shift_func(K,b1, plot_T)$shift)   # time-shift for lagged filter (b1)
  
}
colnames(shift_mat)<-colnames(filter_mat)[select_vec]

par(mfrow = c(2, 2))
colo  <- rainbow(ncol(shift_mat))
mplot <- shift_mat
colnames(mplot) <- colnames(shift_mat)

plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: Whole frequency interval",
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

# From 0 to pi/6 
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "In interval [0,pi/6]",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()

# From 0 to pi/6 and trimmed to [-10,10]
mplot<-shift_mat[1:K/6,]
plot(mplot[,1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Trimmed to -10,10",
     ylim = c(-10, 10), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/36, 2*pi/36, 3*pi/36, 4*pi/36, 5*pi/36, pi/6))
axis(2)
box()











# ════════════════════════════════════════════════════════════════════
# EXERCISE 5:  Applying DFP II to MA(9)
# ════════════════════════════════════════════════════════════════════






b_h <- regularized_dfp_func(gamma_constraint, gammah, alpha0,lambda)$b


