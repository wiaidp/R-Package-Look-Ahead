
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 9 — DFP TRICKY SPECIAL CASES
# ════════════════════════════════════════════════════════════════════

# A brief overview of the DFP is provided in tutorial_3_DFP_overview.r.

# We consider the application to PAYEMS in tutorial 8. The only differences are
# 1. A different (simpler) ARMA(1,1) model
# 2. A slightly different computation of gammah


# ── INITIALISATION ───────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: DFP PAYEMS SETTINGS
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
ar_order<-1
ma_order<-1

arima.obj<-arima(x,order=c(ar_order,0,ma_order))

tsdiag(arima.obj)

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 100 was chosen to ensure that
# the coefficients decay sufficiently close to zero by lag L.
if (ma_order>0)
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = arima.obj$coef[ar_order + 1:ma_order],
    lag.max = length(x)))
} else
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = 0,
    lag.max = length(x)
  ))
  
}


xi[1:(L-1)]/xi[2:L]

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

#?????????????
# The first is doable and leads to `interesting' solution that induces a 
# a lead at omega=0 but a lag at business-cycle frequencies
gammah <- c(gamma[h + (1:(L-h))],rep(0,h))
# This one generates a strange (negative) DFP...???
gammah <- gamma[h + 1:L]

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- gamma[htilde + 1:L]

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead_vec <- c(-2,-4,-6,-8,-10,-20,-100,-1000000)
lead_vec <- -2^(0:6)


# ─────────────────────────────────────────────────────────────────────
# 1.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh <- sum((0:(L-1)) * gammah) / sum(gammah)
tau0 <- sum((0:(L-1)) * gamma0) / sum(gamma0)



if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
}


# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
# Normalize DFP to unit-length (or not)
# Normalization eases visual inspection below.
unit_length<-T
for (i in 1:length(lead_vec))#i<-1
{
  # Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0,gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    
    # Extract the components returned by the function
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    alpha0 <- as.double(t(gamma0) %*% b)
    
    if (unit_length)
    {
      # Normalise b to unit length to obtain the unitary DFP filter
      b_tau <- b / as.double(sqrt(b %*% b))
    } else
    {
      b_tau<-b
    }
    b_mat<-cbind(b_mat,b_tau)
    lambda_vec<-c(lambda_vec,lambda0)
    alpha_vec<-c(alpha_vec,alpha0)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}


# We briefly check that the solution in the non-standard case outperforms 
# the standard solution (assuming tauh<tau0) in terms of target correlation
# Select a lead
k<-1
tau<-lead_vec[k]
# Formula for lambda0: standard case
lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
# DFP predictor: sign on gammah is positive!
b<-gammah+lambda0*gamma0
# Check time-shift constraint
taub <- sum((0:(L-1)) * b) / sum(b)  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2
# Actual lead of the DFP over the MSE predictor at frequency zero
lead_dfp_mse <- taub - tauh
# This difference should be (numerically) zero
print(lead_dfp_mse - tau)
# Target correlation is negative
b%*%gammah/sqrt(b%*%b*gammah%*%gammah)

# Correction in non-standard case
lambda0<--lambda0
# DFP predictor: sign on gammah is negative!
b<--gammah+lambda0*gamma0
# Target correlation is negative
b%*%gammah/sqrt(b%*%b*gammah%*%gammah)
# Check time-shift constraint
taub <- sum((0:(L-1)) * b) / sum(b)  # frequency-zero shift of the DFP filter, see tutorial 2, exercise 3.3.2
# Actual lead of the DFP over the MSE predictor at frequency zero
lead_dfp_mse <- taub - tauh
# This difference should be (numerically) zero
print(lead_dfp_mse - tau)





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


if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
  print("Supply the sign inverted -gammah to MSE-DFP: minimize target correlation")
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, -gammah, alpha0_cd)
  
} else
{
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
  
}

lambda_cd <- dfp_obj$lambda
b      <- dfp_obj$b
if (unit_length)
{
  # Normalise b to unit length to obtain the unitary DFP filter
  b_cd <- b / as.double(sqrt(b %*% b))
} else
{
  b_cd<-b
}




filter_mat<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 1.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# Note on phase reversal:
# The fully decoupled DFP inverts orientation of the trend (or change sign 
# of the mean): the sum of its filter coefficients is negative.
sum(b_cd)

# In addition, the target covariance is negative
b_cd%*%gammah


# We can compute the time-shift at frequency zero.
# But since trend orientation is inverted, this number corresponds to the time-shift
# of the sign inverted DFP: -b_cd
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

if (F)
{
  # We could  consider the fully decoupled DFP in the standard case, which would 
  # apply when tauh<tau0.
  # Use gammah in the DFP call (instead of -gammah)
  b<-mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)$b
  
  if (unit_length)
  {
    # Normalise b to unit length to obtain the unitary DFP filter
    b_cd <- b / as.double(sqrt(b %*% b))
  } else
  {
    b_cd<-b
  }
  
  # Check full decoupling
  t(b_cd) %*% gamma0
  # No trend reversion: the sum of its filter coefficients is positive.
  sum(b_cd)
  # In addition, the target correlation is (marginally)positive
  b_cd%*%gammah
  
  # But since tauh>tau0, the solution to the standard case has tau_cd>tauh: the 
  # fully decoupled DFP is lagging
  tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
  tau_cd
}

# To summarize:
# In this example (ARMA(1,1)), enforcing full decoupling in the non-standard case
# is asking too much from the predictor: the DFP reverts trend direction and worse, 
# its target correlation is negative






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
#       tutorial 6, exercise 1.9 (where the fully decoupled DFP inverts trend direction).
#       Note: the forecast horizon, h=12, is larger which puts some distance between 
#       lag 0 (decoupling) and forecast horizon h=12 (maximization of target correlation). 
#       In a way, the forecast problem is less conflicting (less complex) here.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 5, exercise 1.6. More negative lambda indicate stronger rotation 
#       in the figure of tutorial 5, exercise 1.6.
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 1.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 1.9)
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
# Note: example 2.4 in tutorial 6 applied the MA form to x_t instead, which is not optimal.
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


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:



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


# Discussion

# Need a more refined design which addresses aggregate lead (not only at frequency 0): PCS



###################################################################################
###################################################################################
###################################################################################
# ════════════════════════════════════════════════════════════════════
# Exercise 2: Same as Exercise 1 but a Different Model
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
if (ma_order>0)
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = arima.obj$coef[ar_order + 1:ma_order],
    lag.max = length(x)))
} else
{
  xi <- c(1, ARMAtoMA(
    ar      = arima.obj$coef[1:ar_order],
    ma      = 0,
    lag.max = length(x)
  ))
  
}


xi[1:(L-1)]/xi[2:L]

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

#?????????????
# The first is doable and leads to `interesting' solution that induces a 
# a lead at omega=0 but a lag at business-cycle frequencies
gammah <- c(gamma[h + (1:(L-h))],rep(0,h))

# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- c(gamma[htilde + (1:(L-htilde))],rep(0,htilde))

# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency)
lead_vec <- c(-2,-4,-6,-8,-10,-20,-100,-1000000)
lead_vec <- -2^(0:6)


# ─────────────────────────────────────────────────────────────────────
# 2.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shift of the long-horizon MSE filter (reference)
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tauh <- sum((0:(L-1)) * gammah) / sum(gammah)

# In contrast to exercise 3 above, tauh>tau0: standard case
if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
}


# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
# Normalize DFP to unit-length (or not)
# Normalization eases visual inspection below.
unit_length<-T
for (i in 1:length(lead_vec))#i<-1
{
  # Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0,gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    
    # Extract the components returned by the function
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    alpha0 <- as.double(t(gamma0) %*% b)
    
    if (unit_length)
    {
      # Normalise b to unit length to obtain the unitary DFP filter
      b_tau <- b / as.double(sqrt(b %*% b))
    } else
    {
      b_tau<-b
    }
    b_mat<-cbind(b_mat,b_tau)
    lambda_vec<-c(lambda_vec,lambda0)
    alpha_vec<-c(alpha_vec,alpha0)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
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


if (tauh>tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
  print("Supply the sign inverted -gammah to MSE-DFP: minimize target correlation")
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, -gammah, alpha0_cd)
  
} else
{
  dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
  
}

lambda_cd <- dfp_obj$lambda
b      <- dfp_obj$b
if (unit_length)
{
  # Normalise b to unit length to obtain the unitary DFP filter
  b_cd <- b / as.double(sqrt(b %*% b))
} else
{
  b_cd<-b
}




filter_mat<-cbind(gamma0,gammah,b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast","MSE",paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 2.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# In contrast to previous example, the fully decoupled DFP does not invert trend direction!
sum(b_cd)



# We can compute the time-shift at frequency zero. 
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

# In contrast to exercise 3 above, the fully decoupled design is OK here.
# Also, full decoupling is less extreme than time-shift DFP with large leads




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
#       tutorial 6, exercise 1.9 (where the fully decoupled DFP inverts trend direction).
#       Note: the forecast horizon, h=12, is larger which puts some distance between 
#       lag 0 (decoupling) and forecast horizon h=12 (maximization of target correlation). 
#       In a way, the forecast problem is less conflicting (less complex) here.
# 3. lambda: 
#     -The estimated lambda in the DFP: a negative lambda implies that gammah lies between b and gamm0, 
#       see tutorial 5, exercise 1.6. More negative lambda indicate stronger rotation 
#       in the figure of tutorial 5, exercise 1.6.
# 4. alpha0:
#     -The MSE-DFP constraint parameter. It cannot be interpreted as a correlation (except when it is vanishing).
#     -Reformulating the DFP constraint in terms of tau (instead of alpha0) increases interpretability.
#     -The fully decoupled DFP leads to a vanishing alpha0



# ─────────────────────────────────────────────────────────────────────
# 2.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two plots in the top row (filter coefficients, CCF),
# and a third plot spanning the full bottom row (predictor outputs, Section 2.9)
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
# Note: example 2.4 in tutorial 6 applied the MA form to x_t instead, which is not optimal.
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


# Dotcom recession
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
# We observe a lag!!!!!!!!!!!!!!!!!!!!!
# How is that possible??


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


# Discussion

# Need a more refined design which addresses aggregate lead (not only at frequency 0): PCS



