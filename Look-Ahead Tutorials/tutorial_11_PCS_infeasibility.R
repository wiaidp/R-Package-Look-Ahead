# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS: INFEASIBILITY
# ════════════════════════════════════════════════════════════════════

# The DFP and PCS look-ahead approaches impose constraints on the cross-
# correlation function (CCF) of the resulting predictors.
#
# These constraints may conflict with the internal structure of the data-
# generating process (DGP). In such cases, the degrees of freedom available
# for maximizing the objective — namely, the target correlation at a
# prespecified horizon h — may shrink, leaving little potential to effectively
# look ahead. In some cases the conflict is so severe that a feasible solution
# does not exist, meaning that:
#   a) Some constraints cannot be satisfied simultaneously
#      (the system is overdetermined), or
#   b) All constraints can be met exactly, but the implied target correlation
#      is negative.
#
# Note: A PCS problem may be declared infeasible in the above sense even when
# a PCS predictor exists whose CCF peaks at k = h with a positive target
# correlation. This can occur, for instance, when the CCF increases toward k = h
# in a non-linear or non-monotonic fashion that is not captured by any of the
# three constraint types:
#   - Type I may be overly restrictive, as it requires a strictly (linearly)
#     increasing CCF over the full interval k = 0, …, h, when imposing a very 
#     large regularization weight.
#   - Types II and III may be insufficiently restrictive, requiring only a
#     local increase from k = h-1 to k = h, or a positive average increase
#     from k = 0 to k = h, respectively.
#
# For this reason, we generally recommend Type I, but paired with a moderate
# regularization weight that permits controlled departures from the strictly
# linear CCF increase. This relaxation frees up degrees of freedom that can
# then be directed toward maximizing the objective function, ensuring that the
# look-ahead design achieves optimal tracking of the target at horizon h.
#
#
# This tutorial analyzes a simple infeasible example within the framework of
# Tutorial 9, based on an ARMA(1,1) process fitted to the monthly PAYEMS
# employment (and business cycle) indicator, see example 1 and 2 below.
#
# We also address solutions to infeasibility by proposing variants (see 
# exercises 3-5) that will be explored and refined later.
#
# We begin with a brief summary of the main PCS predictor typology, organized
# by the underlying constraint structure and solution space. See Tutorial 10
# for general background on PCS.

# ── PCS PREDICTOR TYPOLOGY ────────────────────────────────────────────────────

#   TYPE I — Monotonically Increasing CCF over {0, …, h}  [Most Restrictive]
#
#       The CCF must be strictly increasing across the full lag interval, i.e.,
#           CCF(k-1) < CCF(k)  for all k = 1, …, h.
#       See Wildi (2026), Section 3.2 and Appendix E.
#       This condition is generally not exactly feasible (see Exercise 1).
#       The principal PCS optimization function PCS_shift_func() enforces it
#       as closely as possible via regularization.
#
#   TYPE II — Positive Local Slope at the Target Lag  [Weaker than Type I]
#
#       The CCF must be increasing over the final step only, i.e.,
#           CCF(h-1) < CCF(h).
#       See Wildi (2026), Section 3.2.
#       In cases where additional structure is imposed by the DGP (e.g., via
#       the Yule-Walker equations of an AR(p) process), Types I and II may
#       become equivalent and may be equally feasible or infeasible.
#
#   TYPE III — Positive Average Slope from Lag 0 to Lag h  [Weaker than Type I]
#
#       The CCF must be increasing on average from k = 0 to k = h, i.e.,
#           CCF(0) < CCF(h).

# Summary of constraints when feasible:
#   Type I:   CCF(k) > CCF(k-1)  for k = 1, …, h  (h constraints)
#   Type II:  CCF(h) > CCF(h-1)                    (1 constraint)
#   Type III: CCF(h) > CCF(0)                      (1 constraint)
#
# Each condition is necessary but not sufficient for attaining a global maximum
# of the CCF at lag k = h. Nevertheless, even when the CCF peak does not fall
# exactly at k = h, the resulting PCS predictor generally exhibits look-ahead
# behavior, provided the problem is feasible.
#
# Among the three types, Type I is the most stringent: it imposes the largest
# number of constraints (one per lag from k = 1 to k = h), which maximizes the
# chances of achieving a CCF peak at k = h, but simultaneously leaves the fewest
# degrees of freedom for optimizing the objective (i.e., the target correlation
# at forecast horizon h). As a result, Type I is also the most likely to be
# infeasible, with the risk of infeasibility increasing with h and depending
# strongly on the structure of the DGP.
#
# Infeasible problems can be addressed via regularization, which penalizes
# departures from the constraints. When the problem is truly infeasible, these
# deviations do not vanish as the regularization weight grows, since the
# constraints cannot be satisfied regardless of how strongly they are enforced.
# Assigning a moderate (rather than arbitrarily large) regularization weight
# preserves flexibility, unfreezes degrees of freedom, and allows the optimizer
# to maximize tracking accuracy at horizon h (i.e., target correlation /
# minimum MSE).

# ── EXAMPLES OVERVIEW ─────────────────────────────────────────────────────────

# Example 1 — PCS Type III: Feasible but Ineffective.
#   The problem is feasible in the sense that CCF(h) > CCF(0), as required by
#   the Type III constraint, and the target correlation is positive. However, 
#   the CCF peak occurs at k = 1 rather than at the target horizon k = h = 12, 
#   so the look-ahead objective is not achieved.
#   More strikingly, the resulting filter simultaneously worsens the signal-to-noise
#   ratio (i.e., increases noise) and introduces lag — a doubly adverse outcome.
#   This pathological behavior arises from imposing the Type III constraint
#   under adverse structural conditions of the DGP, which leave insufficient
#   degrees of freedom to address the problem.


# Example 2 — PCS Type I with Strong Regularization: Infeasible
#   Imposing a monotonically (linearly) increasing CCF from k = 0 to k = h
#   is infeasible for this DGP:
#   - The constraints can be satisfied exactly, but the resulting target
#     correlation is negative.
#   - Ten out of the 12 imposed constraints are redundant, due to the
#     ARMA(1,1) structure: the ACF decays exponentially for lags beyond q = 1
#     (Yule-Walker equations apply for k > q = 1).
#   - The nowcast coefficient gamma_0 (Wold decomposition) is not collinear
#     with the one-step-ahead MSE predictor gamma_1, but gamma_1 and gamma_h
#     are collinear for any h > 0.
#   - Consequently, only two degrees of freedom are available to solve what
#     reduces to a regular 2-dimensional linear constraint system, yielding a
#     unique solution with no room for further optimization.

# Examples 3–5 — Restoring Feasibility via Three Different Approaches

# Example 3 — PCS Type I with Perturbation (builds on Example 2)
#   The framework from Example 2 is slightly perturbed so that the solution
#   space expands from dimension 2 to dimension L (= 50), providing greater
#   freedom to optimize the target correlation.
#   The problem becomes feasible and the CCF increases monotonically,
#   peaking at k = 12. However, because the perturbations are not optimized,
#   the predictor is prone to trend reversals and is not straightforwardly
#   interpretable.

# Example 4 — PCS Type I with Yearly Growth Target (builds on Example 3)
#   The target is changed from monthly growth to yearly growth.
#   The solution space expands to dimension 12. Three settings are explored:
#
#   Setting 4.1 — Constraint space of dimension 12:
#     Exactly determined; the system can be solved exactly but leaves no
#     degrees of freedom for target correlation maximization. The resulting
#     target correlation is negative.
#
#   Setting 4.2 — Constraint space of dimension 11:
#     Leaves one degree of freedom for target maximization. The problem is
#     feasible and the PCS exhibits look-ahead behavior.
#
#   Setting 4.3 — Constraint space of dimension 13 (overdetermined) with
#     moderate lambda:
#     An approximately feasible solution exists. The CCF peak is shifted
#     (though not exactly to h = 12), the target correlation is positive,
#     and the PCS exhibits look-ahead behavior.

# Example 5 — PCS Type III with Yearly Growth Target (builds on Example 4)
#   The constraint space is 2-dimensional and therefore underdetermined:
#   the problem is feasible and the predictor exhibits look-ahead behavior.
#   However, the weaker Type III constraint controls only CCF(h) > CCF(0),
#   which in this example does not guarantee that the CCF peak is relocated
#   to k = h.

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
# EXERCISE 1: PCS Type III)
# Non-Standard Case: Feasible but Useless (no Look Ahead)
# ════════════════════════════════════════════════════════════════════

# We rely on the framework in Tutorial 9: ARMA(1,1) applied to PAYEMS.


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

# A slowly and monotonically decaying ACF pattern suggests that the MSE
# predictor will be 'stuck at the present'; see Tutorial 1.

# Optionally target a smoothed version of x rather than x itself.
if (FALSE) {
  # Acausal moving average over the preceding and following year
  # (symmetric filter of length 23).
  L_target    <- 12 * 2 - 1
  gamma_target <- rep(1 / L_target, L_target)
  gamma        <- conv_two_filt_func(xi, gamma_target)
} else {
  # Default: target the raw differenced series directly.
  gamma <- xi
}



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
# The type III PCS  imposed decoupling of the predictor from (gamma0-gammah).
# If h=1 then type III and I PCS coincide. However, here h=5, and therefore both 
# types differ at least pro forma.


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

# Outcome:
# Filter coefficients:
# -Complying with a positive CCF slope is only possible when flipping the sign of the predictor.$
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
# EXERCISE 2: PCS Type I)
# Infeasibilty
# ════════════════════════════════════════════════════════════════════

# The same empirical framework as Exercise 1 but using type I) PCS
# The problem is infeasible because the type I system contraints are overspecified:
# The DGP is subject to structural constraints which contradict the type I) CCF constraints.



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

# Maximal rank of an equation system involving the CCF in PCS
which(abs(eigenvalues)>10^{-10})

# Type III PCS (see exercise 1):
# In this case imposing a constraint of the type b' * (gamma_h-gamma_{0}) steals 
# one of the available 2 degrees of freedom: only one degree of freedom is left for 
# optimization: this is the degree of freedom exploited in exercise 1 above (without much success, since the resulting 
# PCS is lagging (non-standatd-case???))

# Type I or II with h>1:
# The problem is worse if one does not include the nowcast gamma0, since then 
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
lambda <- 100000


# ─────────────────────────────────────────────────────────────────────
# 2.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_shift_func(Delta, xi, L, beta, lambda)
  
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
                     max_lag, h, filter_mat[, i], gamma0)$cor_vec)
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
# EXERCISE 3: Re-Installing Feasibility (Part A)
# ════════════════════════════════════════════════════════════════════

# We rely on the previous exercise but slightly alter the MSE predictors to recover full rank space for the PCS predictor.
# Note: this exercise is for illsutration only. The solution is not useful 
#  in terms of `look ahead' business-cycle indicator but only to suggest a way 
# out of the singularity. A more systematic approach based on this idea is 
# proposed in Tutorial ???.



# ─────────────────────────────────────────────────────────────────────
# 3.1 Breaking-Down Structural Singularity 
# ─────────────────────────────────────────────────────────────────────


truncate_at<-L
xi_truncate<-c(xi[1:truncate_at],rep(0,length(xi)-(truncate_at)))

par(mfrow=c(1,1))
ts.plot(cbind(xi,xi_truncate)[1:(2*L),],col=c("black","red"),lty=c(1,2))
mtext("Xi",line=-1)
mtext("Truncated Xi",line=-2,col="red")

gamma_mat<-gamma_mat_truncate<-gamma0
for (i in 1:(L-1))
{
  gamma_i<-xi[i+1:L]
  gamma_mat<-cbind(gamma_mat,gamma_i)
  gamma_truncate_i<-xi_truncate[i+1:L]
  gamma_mat_truncate<-cbind(gamma_mat_truncate,gamma_truncate_i)
}  

# Only two dimensions in original system
which(abs(eigen(gamma_mat)$values)>10^(-12))
# L dimensions in truncated system
which(abs(eigen(gamma_mat_truncate)$values)>10^(-12))

# Define gamma0 and gammah based on truncation
gamma0<-xi_truncate[1:L]
gammah<-xi_truncate[h+1:L]



# ─────────────────────────────────────────────────────────────────────
# 3.2 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.2, -0.1, 0, 0.02, 0.04)

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
lambda <- 5


# ─────────────────────────────────────────────────────────────────────
# 3.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_shift_func(Delta, xi_truncate, L, beta, lambda)
  
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
# 3.5 Plots and Performance Summary
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
                     max_lag, h, filter_mat[, i], gamma0)$cor_vec)
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

# ─────────────────────────────────────────────────────────────────────
# 3.6 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 3.6.1 Apply Predictors to artificial ARMA(1,1) data
#----------------------------------------------------------------------

# Simulated data ARMA(1,1): empirical CCFs converge to expected values.
len<-1000000
set.seed(462)
eps<-x_sim<-x_filt<-rnorm(len)

x_filt<-eps

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

# Scale to facilitate visual inspection
y_out_mat<-scale(y_out_mat,center=F,scale=T)

#----------------------------------------------------------------------
# 3.6.2 Plot
#----------------------------------------------------------------------

anf<-600
enf<-700

# Select only the non-trend reverting designs
mplot<-y_out_mat[anf:enf,]
colnames(mplot)<-colnames(y_out_mat)#[1:4]


par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)

# Outcome:
# All PCS, and in particular the last three trend inverting designs, are 
# difficult to interpret.
# Resolving the singularity by truncating the MSE predictors makes the PCS problem 
# feasible: the CCF increases from k=0 to k=h=12 and the target correlation at k=h 
# remains positive (in contrast to exercise 1).
# But the predictors do not provide useful look ahead content.

# The following empirical CCF illustrates the difficulty of interpreting the strongest PCS design 

# Empirical CCF of strongest PCS
par(mfrow = c(1, 1))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 40, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor.
                  Monotonically Increasing from k=0 to k=12.
                  But strongest peaks at lags -1,-2"))

# The PCS constraints are met: the CCF increases from k=0 to k=h=12.
# But the CCF is very difficult to interpret.
# The outcome suggests overfitting: the PCS exploits whatever degrees of freedom 
# are provided in the perturbated full-rank space based on truncated MSE predictors.





# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: Re-Install Feasibility (Part B)
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 4.1 Framework
# ─────────────────────────────────────────────────────────────────────

# Note: exercise 1 must be run first to initialize the general framework
# for working with the differenced log-PAYEMS series.

# The differenced log-PAYEMS series is fairly noisy, with pronounced
# downturns during recession episodes.
ts.plot(x)

# To reduce noise we apply a lowpass filter. Possible choices include:
#   - Classic trend filters, e.g., the HP filter (see tutorial 10)
#   - Ideal lowpass filter
#   - AR(1) smoother
#   - Moving average (MA)
#
# For simplicity, we use an equally-weighted MA(12) for the following reasons:
#   - The DFP approach is agnostic to this choice; results generalize to
#     other target filter designs.
#   - An equally-weighted MA(12) applied to differenced log-PAYEMS corresponds
#     directly to yearly growth, making the target readily interpretable.
#   - Averaging over a full year reduces noise and amplifies the relevant
#     business-cycle dynamics.

# Define the equally-weighted yearly MA target filter
gamma_target <- rep(1/12, 12)

# ─────────────────────────────────────────────────────────────────────
# 4.2 Model Fit
# ─────────────────────────────────────────────────────────────────────

# Reuse the ARMA model estimated in exercise 1.
# Key difference relative to exercise 1: the target is now yearly growth
# (MA(12) of differenced log), rather than the raw differenced log series.

if (T) {
  # Convolve the Wold decomposition with the MA(12) target filter to obtain
  # the Wold coefficients of the yearly growth target process.
  gamma <- conv_two_filt_func(xi, gamma_target)$conv
} else {
  # Alternative: target the raw differenced log series directly (no smoothing).
  gamma <- xi
}

# Visualize the Wold coefficients for both representations.
par(mfrow = c(2, 1))
ts.plot(xi,    main = "Wold Decomposition: Monthly Growth (Post-1990)")
ts.plot(gamma, main = "Wold Decomposition: Yearly Growth (Post-1990)")

# Set MSE nowcast and h=12-step ahead predictors:
gamma0<-gamma[1:L]
gammah<-gamma[h+1:L]
# Note: The h=12-step ahead MSE predictor is AR(1) with a1 determined by the ARMA(1,1).

# ─────────────────────────────────────────────────────────────────────
# 4.3 DGP Structural Constraints on PCS (and DFP) Solution Space
# ─────────────────────────────────────────────────────────────────────

# For h>=1 gammah<-a1^(h+0:L)
gamma_mat<-gamma[1:L]
for (i in 1:(L-1))
{
  gamma_i<-gamma[i+1:L]
  gamma_mat<-cbind(gamma_mat,gamma_i)
}  

eigenvalues<-eigen(gamma_mat)$values

# Maximal rank of an equation system involving the CCF in PCS
length(which(abs(eigenvalues)>10^{-10}))-1

# Type III PCS (see exercise 1):
# In this case imposing a constraint of the type b' * (gamma_h-gamma_{0}) steals 
# one of the available 2 degrees of freedom: only one degree of freedom is left for 
# optimization: this is the degree of freedom exploited in exercise 1 above (without much success, since the resulting 
# PCS is lagging (non-standatd-case???))

# Type I or II with h>1:
# The problem is worse if one does not include the nowcast gamma0, since then 
# gammah and gamma_{h-1} are linearly dependent and therefore the room spanned 
# by gammah and (gamma_h-gamma_{h-1}) is one-dimensional. Hence the solution 
# of the PCS must be along gammah, either in the same or opposite direction.
# There is no room left for optimization.

# ─────────────────────────────────────────────────────────────────────
# 4.4 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.1, 0, 0.1, 0.2, 0.3)/2



# Using yearly differences increases the rank of the constraint system.

# However, when length(Delta) >12 the system is still overdetermined: the PCS equation system is infeasible.
# If length(Delta)=12 the equations system is regular and can be solved with a unique solution. 
# Hence no degree of freedom is left for optimiaztion and the target correlation could become negative (as is the case here).
# In both cases the PCS is infeasible.
# If length(Delta)<12 the equation system is underdetermined and henece there are degrees of freedom
# left for optimization which implies that the target correlation is positive.

# Proceeding
# Selecting Delta=1:11  and lambda arbitrarily large allows for positive target correlation
# Selecting Delta=2:12  and lambda arbitrarily large does not always allow for positive target correlation
# because the rank of the system without lag 0 is reduced.
# Obviously, selecting Delta=1:12 and lambda arbitrarily large does not always allow for positive target correlation
# Selecting Delta=1:12 and lambda moderate allows for positive target correlation and increased flexibility 
#   in addressing the constraints: preferred solution.
# We here select 


# This is infeasible for positive slopes (the CCF peak cannot be shifted)
Delta <- 2:12
lambda<-1000000

# This is still infeasible for the selected positive slopes: lambda is too large and the target correlations are negative
Delta <- 2:12
lambda<-10

# This is feasible: there is one degree of freedom left for maximizing the target correlation which is hence positive
Delta <- 1:11
lambda<-1000000



# This is feasible: lambda is sufficiently small to allow maximization of the target correlation, addressing the 
# PCS constraints in a more flexible way
Delta <- 1:12
lambda<-1





if (length(Delta)>=length(which(abs(eigenvalues)>10^{-10}))-1)
{
  print("PCS system has no degrees of freedom left for optimization")
  print("Select regularization weight lambda not too large or reduce length of Delta (number of constraints")
}




# ─────────────────────────────────────────────────────────────────────
# 4.5 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_shift_func(Delta, gamma, L, beta, lambda)
  
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
# 4.6 Routine Checks
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
# 4.7 Plots and Performance Summary
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
                     max_lag, h, filter_mat[, i], gamma0)$cor_vec)
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

# Predictor weights:
# -The classic MSE(12) is AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# -If lambda is moderate (i.e. lambda=1) the peak of the CCF is shifted rightwards but 
#   it is not located at k=h=12. Nevertheless, the PCS has look ahead behaviour which is 
#   is the main purpose of PCS.


# ─────────────────────────────────────────────────────────────────────
# 4.8 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 4.8.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 4.8.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)



# Empirical CCF of strongest PCS
par(mfrow = c(1, 1))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 20, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor.
                  Monotonically Increasing from k=0 to k=12.
                  But strongest peaks at lags -1,-2"))




# ════════════════════════════════════════════════════════════════════
# EXERCISE 5: As Exercise 4 but Type III) PCS based on PCS_shift_func()
# ════════════════════════════════════════════════════════════════════

# New feature: we can address a positive average growth between k=0 and k=h=12.
# This generalizes exercise 1 based on unitary DFP, since we can use PCS_shift_func()
# which allows fine-tuning of the constraint (in contrast to DFP in exercise 1).

# Since we impose a single aggregate constraint (between k=0 and k=h) the PCS constraints 
# are underdetermined and allow for maximization of the target correlation for arbitrarily large lambda. 
# Delta: from k=0 to k=12
Delta <- c(0,12)
# New feature: we must inform PCS_shift_func below that it uses the span k=0 to k=h=12 as specified in Delta
initialize_with_null<-T

lambda<-1000000



if (length(Delta)>=length(which(abs(eigenvalues)>10^{-10}))-1)
{
  print("PCS system has no degrees of freedom left for optimization")
  print("Select regularization weight lambda not too large or reduce length of Delta (number of constraints")
}

beta_vec <- c(-0.1, 0, 0.1, 0.2, 0.5)



# ─────────────────────────────────────────────────────────────────────
# 5.5 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value
for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor. We supply the additional initialize_with_null whose default value is F (when omitted in the previous exercises)
  PCS_obj <- PCS_shift_func(Delta, gamma, L, beta, lambda,initialize_with_null)
  
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
# 5.6 Routine Checks
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
# 5.7 Plots and Performance Summary
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
                     max_lag, h, filter_mat[, i], gamma0)$cor_vec)
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

# Predictor weights:
# -The classic MSE(12) is AR(1).
# -The PCS account for the DGP structure at lags smaller than 12
# -Increasing the slope assigns increasing weight to lag 0 (intuitively appealing). 

# CCF:
# - We address the mean-growth between k=0 and k=12: beta>0 means that CCF(h)-CCF(0)>0.
# - This does not imply that the CCF peak is shifted to k=h.
# - But we can still gain look ahead behaviour


# ─────────────────────────────────────────────────────────────────────
# 5.8 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 5.8.1 Apply Predictors to data
#----------------------------------------------------------------------
# All filters are defined in MA form (as applied to the einnovations eps_t 
# in the Wold decomposition). Therefore we apply the filters to model residuals.
x_filt   <- arima.obj$residuals


y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)

#----------------------------------------------------------------------
# 5.8.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
mplot<-scale(y_out_mat,center=F,scale=T)
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(mplot,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,rep(1,ncol(mplot)-2)),lwd=c(1,2,rep(1,ncol(mplot)-2)))
abline(h = 0)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],col=colo[i],line=-i)



# Empirical CCF of strongest PCS
par(mfrow = c(1, 1))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 20, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor.
                  Monotonically Increasing from k=0 to k=12.
                  But strongest peaks at lags -1,-2"))


