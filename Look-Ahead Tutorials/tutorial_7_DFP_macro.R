# ════════════════════════════════════════════════════════════════════
# TUTORIAL 7 — APPLICATION OF DFP TO MONTHLY PAYEMS —
# ════════════════════════════════════════════════════════════════════

# A brief overview on the DFP is provided in tutorial_3_introduction.r

# We consider application of the interpretable MSE-DFP to PAYEMS, an 
# important monthly US business-cycle indicator.
# Interpretability of MSE-DFP is provided by applying the DFP time-shift 
# constraint introduced in tutorial 5. The constraint specifies the constraint
# in terms of lead of the DFP over the MSE predictor.
#

# ─────────────────────────────────────────────────────────────────────
# MOTIVATION:
# ─────────────────────────────────────────────────────────────────────
# Without a unit-length constraint on the filter (unitary DFP, tutorial 3), 
# the parameter alpha0 in the
# MSE-DFP formulation lacks a straightforward interpretation: its effect on the
# predictor cannot easily be expressed in terms of an observable or intuitive
# quantity (unlike the correlation-based interpretation available for the
# unitary DFP Tutorial 3). Moreover, as discussed in tutorial 4, it is unclear 
# how the necessary decoupling from the present x_t also addresses 
# look-ahead behaviour in terms of a lead over the MSE benchmark.
#
# To remedy this, we re-parameterise the DFP constraint by linking alpha0 to
# the time-shift (phase delay) at frequency zero — the trend frequency —
# as introduced in Tutorial 2.
#
# Concretely, the re-parameterised DFP predictor is designed so that its
# output LEADS the MSE predictor by a pre-specified number of time steps, tau,
# at frequency zero. This gives alpha0 a clear and interpretable meaning:
#
#   tau = 0  →  no lead relative to the MSE predictor (recovers MSE)
#   tau > 0  →  the DFP output anticipates the MSE predictor by tau periods
#               at the trend frequency
#
# The question is: does this lead at frequency zero extend to business-cycle 
# movements and to which extent?
#----------------------------------------------------------------------

# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines
# Provides DFP_compute_lambda_alpha0_func() and related solvers
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, CCF helpers)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED using the alfred library (no API key required).
install.packages("alfred")
library(alfred)



# ════════════════════════════════════════════════════════════════════
# Exercise 1: DFP PAYEMS Settings
# ════════════════════════════════════════════════════════════════════


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
# Log transformation addresses non-stationarity in the variance as the level 
# of the series evolves. 
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y)<-index(PAYEMS["1990::2019"])
plot(y,main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = F,
     xlab = "", ylab = "")
axis(1, at = 1:length(y),
     labels = names(y))
axis(2)
box()

# We consider (stationary) first differences of the log-series:
# The log stabilizes the variance.
# The difference stabilizes the level.
x<-diff(y)

# diff-log PAYEMS is fairly noisy with some strong downturns on recession 
# episodes:
ts.plot(x)
# The dependence structure is similar to above AR(3): slowly monotonically 
# decaying ACF: 
acf(x)

# ─────────────────────────────────────────────────────────────────────
# 1.2 Model Fit
# ─────────────────────────────────────────────────────────────────────

L <- 50   # filter length (number of MA coefficients retained)

# ARMA(1,1): simple model with OK diagnostics 
ar_order<-2
ma_order<-2

arima.obj<-arima(x,order=c(ar_order,0,ma_order))

tsdiag(arima.obj)

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 100 was chosen to ensure that
# the coefficients decay sufficiently close to zero by lag L.
xi <- c(1, ARMAtoMA(
  ar      = arima.obj$coef[1:ar_order],
  ma      = arima.obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))

# Visualise xi: the slow decay confirms the longer-memory character of the
# post-1990 log-returns relative to the full post-WWII sample.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")

# The theoretical ACF based on the Wold-decomposition (ARMA-model) matches the 
# above empirical ACF.
ts.plot(ARMAacf(ar=0,ma=xi,lag.max=L))

# A slowly monotonically decaying pattern suggest that the MSe predictor will be 
# stuck at present, see tutorial 1.

# Target the series (x) or a smoothed version of the series
if (F)
{
# Mean over last and next year: acausal filter of length 23  
  L_target<-12*2-1
  gamma_target<-rep(1/L_target,L_target)
  gamma<-conv_two_filt_func(xi,gamma_target)

} else
{
  gamma<-xi
}

# ─────────────────────────────────────────────────────────────────────
# 1.3 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# Two forecast horizons are considered:
#   h      — the primary (short) horizon used for the MSE-DFP predictor
#   htilde — a longer horizon used as an additional reference

h <- 12    # primary forecast horizon (one year ahead)
# Longer forecast horizon used as a long-term benchmark
htilde <- 2*h

# Nowcast:
# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah):
# shift gamma by h positions and pad with zeros at the end.
# The trailing zeros reflect the fact that MA coefficients beyond index L
# are negligible for a finite-length filter of length L.
gammah <- c(gamma[h + (1:(L - h))], rep(0, h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead_vec <- c(-2,-4,-6,-8,-10)


# ─────────────────────────────────────────────────────────────────────
# 1.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh <- sum((0:(L-1)) * gammah) / sum(gammah)

# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
for (i in 1:length(lead_vec))
{
# Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)

# Extract the components returned by the function
  tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
  tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
  lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
  b       <- dfp_obj$b        # raw DFP filter coefficients
  
  alpha0 <- as.double(t(gamma0) %*% b)
  

# Normalise b to unit length to obtain the unitary DFP filter
  b_tau <- b / as.double(sqrt(b %*% b))
  b_mat<-cbind(b_mat,b_tau)
  lambda_vec<-c(lambda_vec,lambda0)
  alpha_vec<-c(alpha_vec,alpha0)
}

# ─────────────────────────────────────────────────────────────────────
# 1.5 Validation
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved leads matches the specified leads ---
tau_vec<-NULL
for (i in 1:length(lead_vec))
{
  taub <- sum((0:(L-1)) * b_mat[,i]) / sum(b_mat[,i])  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2
  tau_vec<-c(tau_vec,taub)
# Actual lead of the DFP over the MSE predictor at frequency zero
  lead_dfp_mse <- taub - tauh

# This difference should be (numerically) zero
  print(lead_dfp_mse - lead_vec[i])
}



# ─────────────────────────────────────────────────────────────────────
# 1.6 Compute Complete Decoupling for Additional Reference
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled DFP corresponds to alpha0 = 0, i.e. the DFP filter
# is orthogonal to gamma0. This serves as a reference benchmark alongside the
# time-shift DFP computed above.

alpha0_cd <- 0  # complete decoupling: <gamma0, b_cd> = 0

# Compute the completely decoupled DFP filter via Proposition 1
dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
lambda_cd <- dfp_obj$lambda
b_cd      <- dfp_obj$b

# Normalise to unit length so that b_cd is comparable to b_opt
scale <- as.double(1 / sqrt(t(b_cd) %*% b_cd))
b_cd  <- scale * b_cd

filter_mat<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 1.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# Note on phase reversal:
# In contrast to tutorial 5 (exercise 1.9), the fully decoupled DFP does not 
# invert orientation of the trend (or change sign of the mean): the sum of its 
# filter coefficients is positive.
sum(b_cd)

# We can compute the time-shift at frequency zero:
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

# Lead over MSE: smaller (larger lead) than the above lead_vec
tau_cd-tauh


# ─────────────────────────────────────────────────────────────────────
# 1.8 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (positive = right-shift or lag when applied to linear trend)
#   lambda    — DFP regularisation weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value)
#
# Rows correspond to:
#   MSE(h)       — h-step MSE predictor
#   MSE(htilde)  — long-horizon MSE predictor (reference)
#   DFP-shifted  — time-shift DFP with specified lead
#   DFP full dec.— completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 3+length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)",  "lambda", "alpha0")
rownames(mat_perf) <- c(paste("MSE(", h,      ")", sep = ""),
                        paste("MSE(", htilde, ")", sep = ""),
                        paste("DFP-shifted by ",lead_vec,sep=""),
                        "DFP fully decoupled")

# MSE h-step: time-shift and unit-normalised gain; lambda and alpha0 not applicable
mat_perf[1, 1] <- tauh

# Long-horizon MSE: same metrics for htilde
mat_perf[2, 1] <- tauhtilde
# Time-shift DFP and completely decoupled DFP: all metrics available
mat_perf[3:nrow(mat_perf),1]<-c(tau_vec,tau_cd)
mat_perf[3:nrow(mat_perf),2]<-c(lambda_vec,lambda_cd)
mat_perf[3:nrow(mat_perf),3]<-c(alpha_vec,alpha0_cd)


# Note: NAs indicate that the corresponding entries are left blank (not meaningful)
mat_perf

# Discussion:
# 1. tau(0)-column:
#     -The classic h=12 steps ahead MSE predictor has a time-shift of ~13.01
#       Interpretation: when the filter (predictor) is applied to a linear trend, 
#       then the trend will be right-shifted (delayed) by 13 time points.
#     -Increasing the forecast horizon to h=24 (MSE(24) predictor) 
#       reduces the lag: the forecast horizon can be used to some extent to address 
#       timeliness (look ahead) but the effect remains limited overall.
#     -DFP-shifted have time-shifts that differ by lead_vec from MSE(12).
#     -DFP fully decoupled: the time-shift is also well defined since the fully-decoupled DFP
#       does not invert the trend direction, see exercise 1.5 (to be contrasted with 
#       tutorial 5, exercise 1.9 (where the fully decoupled DFP inverts trend direction).
#       Note: the forecast horizon, h=12, is larger which puts some distance between 
#       lag 0 (decoupling) and forecast horizon h=12 (maximization of target correlation). 
#       In a way, the forecast problem is less conflicting (less complex) here.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 4, exercise 1.6. More negative lambda indicate stronger rotation 
#       in the figure of tutorial 4, exercise 1.6.
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 1.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 3.9)
par(mfrow=c(1,2))

colo <- c("black", "green", rainbow(ncol(filter_mat)-2))

# Collect all four filters into a matrix (no scaling applied)
mplot<-filter_mat
col_names <- colnames(filter_mat)
colnames(mplot) <- col_names

# Diagnostic: sum of squared coefficients per filter (proxy for filter energy)
apply(mplot^2, 2, sum)

# --- Top-left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters and add colour-coded labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], type = "l")
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the MSE h-step filter on top to ensure visibility
lines(mplot[, 2], col = colo[2])

axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()

# --- Top-right panel: cross-correlation functions (CCF) ---
# Compute the CCF between each predictor and the AR(3) process at lags
# surrounding lag 0 and the h-step-ahead lag
max_lag<-20
mplot<-NULL
for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot,compute_ccf_func(filter_mat[,i], gamma0)[L-1+1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[ i])
}
abline(v =  1 + h, lty = 2)
abline(h = 0)

axis(1, at = 1:nrow(mplot),
     labels = -1+1:nrow(mplot))
axis(2)
box()

# Discussion:
# CCF (right panel)
#   -The MSE predictor (green) maximizes the CCF at the target horizon h=12.
#    However, the MSE predictor correlates very strongly with x_t (lag 0).
#   -The tau-shifted DFP correlate less strongly with x_t with increasing lead in lead_vec.
#     As a consequence, the correlation with the target at h=12 decreases.
#     But the DFP minimizes this loss at h=12.
#   -Finally, the fully decoupled (red) has a vanishing CCF at lag 0. As 
#     a consequence, the target correlation at h=12 is pulled down even more strongly, 
#     though it is still maximal subject to the imposed full decoupling.

# ─────────────────────────────────────────────────────────────────────
# 1.10 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 1.10.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.


# All filters are defined in MA form (as applied to the einnovations eps_t in the Wold decomposition)
# Therefore we apply the filters to model residuals.
# Note: example 2.4 in tutorial 5 applied the MA form to x_t instead, which is not optimal.
x_filt   <- arima.obj$residuals

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 1.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)



ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
# The DFP designs (blue and red) tend to lie to the right of the MSE predictor
#   (green) especially at longer swings above or below the zero (mean) line.
# Short term high-frequency noise cannot be anticipated.
# The time-shifted DFP (blue) leads the MSE predictor on a linear trend, by design of the constraint.
# We shall see below how this materializes as useful look ahead feature for other series/applications.



# ─────────────────────────────────────────────────────────────────────
# 1.11 Amplitude and Time-Shifts
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
mplot <- amp_mat

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude functions",
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


mplot <- shift_mat
mplot[which(mplot[,ncol(mplot)]<(-2)),ncol(mplot)]<-NA

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
#mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  #  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Outcome:
# 1. Amplitude:
#   -All filters are lowpass. Only the fully-decoupled (red) does not have a monotonically decaying amplitude.
#   -The MSE amplitude (green) is close to the process (black) illustrating the `stuck to present' problem.
#   -The DFP amplitudes are smaller at lower frequencies and larger at higher frequencies: ATS trilemma.
#     -They generate more high-frequency noise relative to low frequency signal.
# 2. Time shifts
#   -MSE (green) and process (black) are close: `stuck at present problem'.
#   -The DFP (blue) time-shift is exactly tau=-2 smaller than MSE at frequency zero.
#   -The frequency-domain shifts at frequency zero confirm the time-domain time-shifts computed in the performance table of 
#    exercise 1.10: the MSE predictor lags a linear time trend by 4 time units.
#   -The lead of DFP (blue) over MSE (green) is fairly stable in a vicinity of frequency zero: 
#    this confirms the lead at business-cycles frequencies in exercise 2.2 and 2.3.
#   -The fully-decoupled DFP (red) is anticipative (negative shift for lower frequencies).











# ════════════════════════════════════════════════════════════════════
# Exercise 2: Same as Exercise 1 but a Diferent Model
# ════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────
# 2.1 Load the Data
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
# Log transformation addresses non-stationarity in the variance as the level 
# of the series evolves. 
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y)<-index(PAYEMS["1990::2019"])
plot(y,main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = F,
     xlab = "", ylab = "")
axis(1, at = 1:length(y),
     labels = names(y))
axis(2)
box()

# We consider (stationary) first differences of the log-series:
# The log stabilizes the variance.
# The difference stabilizes the level.
x<-diff(y)

# diff-log PAYEMS is fairly noisy with some strong downturns on recession 
# episodes:
ts.plot(x)
# The dependence structure is similar to above AR(3): slowly monotonically 
# decaying ACF: 
acf(x)

# ─────────────────────────────────────────────────────────────────────
# 2.2 Model Fit
# ─────────────────────────────────────────────────────────────────────

L <- 50   # filter length (number of MA coefficients retained)

# ARMA(1,1): simple model with OK diagnostics 
ar_order<-1
ma_order<-1

arima.obj<-arima(x,order=c(ar_order,0,ma_order))

tsdiag(arima.obj)

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 100 was chosen to ensure that
# the coefficients decay sufficiently close to zero by lag L.
xi <- c(1, ARMAtoMA(
  ar      = arima.obj$coef[1:ar_order],
  ma      = arima.obj$coef[ar_order + 1:ma_order],
  lag.max = L - 1
))

# Visualise xi: the slow decay confirms the longer-memory character of the
# post-1990 log-returns relative to the full post-WWII sample.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition xi: Slowly Decaying Impulse Response (Post-1990)")

# The theoretical ACF based on the Wold-decomposition (ARMA-model) matches the 
# above empirical ACF.
ts.plot(ARMAacf(ar=0,ma=xi,lag.max=L))

# A slowly monotonically decaying pattern suggest that the MSe predictor will be 
# stuck at present, see tutorial 1.

# Target the series (x) or a smoothed version of the series
if (F)
{
  # Mean over last and next year: acausal filter of length 23  
  L_target<-12*2-1
  gamma_target<-rep(1/L_target,L_target)
  gamma<-conv_two_filt_func(xi,gamma_target)
  
} else
{
  gamma<-xi
}

# ─────────────────────────────────────────────────────────────────────
# 2.3 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# Two forecast horizons are considered:
#   h      — the primary (short) horizon used for the MSE-DFP predictor
#   htilde — a longer horizon used as an additional reference

h <- 12    # primary forecast horizon (one year ahead)
# Longer forecast horizon used as a long-term benchmark
htilde <- 2*h

# Nowcast:
# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah):
# shift gamma by h positions and pad with zeros at the end.
# The trailing zeros reflect the fact that MA coefficients beyond index L
# are negligible for a finite-length filter of length L.
gammah <- c(gamma[h + (1:(L - h))], rep(0, h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L - htilde))], rep(0, htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead_vec <- c(-2,-4,-6,-8,-10)


# ─────────────────────────────────────────────────────────────────────
# 2.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh <- sum((0:(L-1)) * gammah) / sum(gammah)

# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
for (i in 1:length(lead_vec))
{
  # Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  # Extract the components returned by the function
  tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
  tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
  lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
  b       <- dfp_obj$b        # raw DFP filter coefficients
  
  alpha0 <- as.double(t(gamma0) %*% b)
  
  
  # Normalise b to unit length to obtain the unitary DFP filter
  b_tau <- b / as.double(sqrt(b %*% b))
  b_mat<-cbind(b_mat,b_tau)
  lambda_vec<-c(lambda_vec,lambda0)
  alpha_vec<-c(alpha_vec,alpha0)
}

# ─────────────────────────────────────────────────────────────────────
# 2.5 Validation
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved leads matches the specified leads ---
tau_vec<-NULL
for (i in 1:length(lead_vec))
{
  taub <- sum((0:(L-1)) * b_mat[,i]) / sum(b_mat[,i])  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2
  tau_vec<-c(tau_vec,taub)
  # Actual lead of the DFP over the MSE predictor at frequency zero
  lead_dfp_mse <- taub - tauh
  
  # This difference should be (numerically) zero
  print(lead_dfp_mse - lead_vec[i])
}



# ─────────────────────────────────────────────────────────────────────
# 2.6 Compute Complete Decoupling for Additional Reference
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled DFP corresponds to alpha0 = 0, i.e. the DFP filter
# is orthogonal to gamma0. This serves as a reference benchmark alongside the
# time-shift DFP computed above.

alpha0_cd <- 0  # complete decoupling: <gamma0, b_cd> = 0

# Compute the completely decoupled DFP filter via Proposition 1
dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
lambda_cd <- dfp_obj$lambda
b_cd      <- dfp_obj$b

# Normalise to unit length so that b_cd is comparable to b_opt
scale <- as.double(1 / sqrt(t(b_cd) %*% b_cd))
b_cd  <- scale * b_cd

filter_mat<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 2.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# Note on phase reversal:
# In contrast to tutorial 5 (exercise 1.9), the fully decoupled DFP does not 
# invert orientation of the trend (or change sign of the mean): the sum of its 
# filter coefficients is positive.
sum(b_cd)

# We can compute the time-shift at frequency zero:
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

# Lead over MSE: smaller (larger lead) than the above lead_vec
tau_cd-tauh


# ─────────────────────────────────────────────────────────────────────
# 2.8 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (positive = right-shift or lag when applied to linear trend)
#   lambda    — DFP regularisation weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value)
#
# Rows correspond to:
#   MSE(h)       — h-step MSE predictor
#   MSE(htilde)  — long-horizon MSE predictor (reference)
#   DFP-shifted  — time-shift DFP with specified lead
#   DFP full dec.— completely decoupled DFP (alpha0 = 0)

mat_perf <- matrix(nrow = 3+length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)",  "lambda", "alpha0")
rownames(mat_perf) <- c(paste("MSE(", h,      ")", sep = ""),
                        paste("MSE(", htilde, ")", sep = ""),
                        paste("DFP-shifted by ",lead_vec,sep=""),
                        "DFP fully decoupled")

# MSE h-step: time-shift and unit-normalised gain; lambda and alpha0 not applicable
mat_perf[1, 1] <- tauh

# Long-horizon MSE: same metrics for htilde
mat_perf[2, 1] <- tauhtilde
# Time-shift DFP and completely decoupled DFP: all metrics available
mat_perf[3:nrow(mat_perf),1]<-c(tau_vec,tau_cd)
mat_perf[3:nrow(mat_perf),2]<-c(lambda_vec,lambda_cd)
mat_perf[3:nrow(mat_perf),3]<-c(alpha_vec,alpha0_cd)


# Note: NAs indicate that the corresponding entries are left blank (not meaningful)
mat_perf

# Discussion:
# 1. tau(0)-column:
#     -The classic h=12 steps ahead MSE predictor has a time-shift of ~13.01
#       Interpretation: when the filter (predictor) is applied to a linear trend, 
#       then the trend will be right-shifted (delayed) by 13 time points.
#     -Increasing the forecast horizon to h=24 (MSE(24) predictor) 
#       reduces the lag: the forecast horizon can be used to some extent to address 
#       timeliness (look ahead) but the effect remains limited overall.
#     -DFP-shifted have time-shifts that differ by lead_vec from MSE(12).
#     -DFP fully decoupled: the time-shift is also well defined since the fully-decoupled DFP
#       does not invert the trend direction, see exercise 1.5 (to be contrasted with 
#       tutorial 5, exercise 1.9 (where the fully decoupled DFP inverts trend direction).
#       Note: the forecast horizon, h=12, is larger which puts some distance between 
#       lag 0 (decoupling) and forecast horizon h=12 (maximization of target correlation). 
#       In a way, the forecast problem is less conflicting (less complex) here.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 4, exercise 1.6. More negative lambda indicate stronger rotation 
#       in the figure of tutorial 4, exercise 1.6.
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 2.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 3.9)
par(mfrow=c(1,2))

colo <- c("black", "green", rainbow(ncol(filter_mat)-2))

# Collect all four filters into a matrix (no scaling applied)
mplot<-filter_mat
col_names <- colnames(filter_mat)
colnames(mplot) <- col_names

# Diagnostic: sum of squared coefficients per filter (proxy for filter energy)
apply(mplot^2, 2, sum)

# --- Top-left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters and add colour-coded labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], type = "l")
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}

# Redraw the MSE h-step filter on top to ensure visibility
lines(mplot[, 2], col = colo[2])

axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()

# --- Top-right panel: cross-correlation functions (CCF) ---
# Compute the CCF between each predictor and the AR(3) process at lags
# surrounding lag 0 and the h-step-ahead lag
max_lag<-20
mplot<-NULL
for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot,compute_ccf_func(filter_mat[,i], gamma0)[L-1+1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[ i])
}
abline(v =  1 + h, lty = 2)
abline(h = 0)

axis(1, at = 1:nrow(mplot),
     labels = -1+1:nrow(mplot))
axis(2)
box()

# Discussion:
# CCF (right panel)
#   -The MSE predictor (green) maximizes the CCF at the target horizon h=12.
#    However, the MSE predictor correlates very strongly with x_t (lag 0).
#   -The tau-shifted DFP correlate less strongly with x_t with increasing lead in lead_vec.
#     As a consequence, the correlation with the target at h=12 decreases.
#     But the DFP minimizes this loss at h=12.
#   -Finally, the fully decoupled (red) has a vanishing CCF at lag 0. As 
#     a consequence, the target correlation at h=12 is pulled down even more strongly, 
#     though it is still maximal subject to the imposed full decoupling.

# ─────────────────────────────────────────────────────────────────────
# 2.10 Compare Predictors
# ─────────────────────────────────────────────────────────────────────
#----------------------------------------------------------------------
# 2.10.1 Apply Predictors to data
#----------------------------------------------------------------------
# Assemble the filter matrix, normalising gamma0 and gammah to unit L2-norm
# so that all four filters are on a comparable amplitude scale.
# Note: b_cd remains phase-reversing at frequency zero even after normalisation.


# All filters are defined in MA form (as applied to the einnovations eps_t in the Wold decomposition)
# Therefore we apply the filters to model residuals.
# Note: example 2.4 in tutorial 5 applied the MA form to x_t instead, which is not optimal.
x_filt   <- arima.obj$residuals

y_out_mat<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat<-cbind(y_out_mat,filter(x_filt,filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 2.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)



ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
# The DFP designs (blue and red) tend to lie to the right of the MSE predictor
#   (green) especially at longer swings above or below the zero (mean) line.
# Short term high-frequency noise cannot be anticipated.
# The time-shifted DFP (blue) leads the MSE predictor on a linear trend, by design of the constraint.
# We shall see below how this materializes as useful look ahead feature for other series/applications.



# ─────────────────────────────────────────────────────────────────────
# 2.11 Amplitude and Time-Shifts
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
mplot <- amp_mat

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitude functions",
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


mplot <- shift_mat
mplot[which(mplot[,ncol(mplot)]<(-2)),ncol(mplot)]<-NA

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
#mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  #  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Outcome:
# 1. Amplitude:
#   -All filters are lowpass. Only the fully-decoupled (red) does not have a monotonically decaying amplitude.
#   -The MSE amplitude (green) is close to the process (black) illustrating the `stuck to present' problem.
#   -The DFP amplitudes are smaller at lower frequencies and larger at higher frequencies: ATS trilemma.
#     -They generate more high-frequency noise relative to low frequency signal.
# 2. Time shifts
#   -MSE (green) and process (black) are close: `stuck at present problem'.
#   -The DFP (blue) time-shift is exactly tau=-2 smaller than MSE at frequency zero.
#   -The frequency-domain shifts at frequency zero confirm the time-domain time-shifts computed in the performance table of 
#    exercise 1.10: the MSE predictor lags a linear time trend by 4 time units.
#   -The lead of DFP (blue) over MSE (green) is fairly stable in a vicinity of frequency zero: 
#    this confirms the lead at business-cycles frequencies in exercise 2.2 and 2.3.
#   -The fully-decoupled DFP (red) is anticipative (negative shift for lower frequencies).



