# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS: NON-STANDARD CASE
# ════════════════════════════════════════════════════════════════════

# Rely on framework of Tutorial 9. See Tutorial 10 for background on PCS.

# We distinguish three PCS types, see Tutorial 10

#   TYPE I)  
#       Monotonically increasing CCF over {0, …, h} (most restrictive):
#       The CCF must be strictly increasing across the full interval, i.e.,
#       CCF(k-1) < CCF(k) for all k = 1, …, h. See Wildi (2026), Section 3.2 
#       and Appendix E. This condition is generally not exactly feasible 
#       (see Exercise ???); The principal PCS optimization function 
#       PCS_shift_func() enforces it as closely as possible via regularisation.
#
#   TYPE II) 
#       Local positive slope at the target lag (weaker than I):
#       The CCF must be increasing over the final step only, i.e.,
#       CCF(h-1) < CCF(h). See Wildi (2026), Section 3.2.
#       In some cases where additional structure is imposed by the data-
#       generating process (e.g., from the Yule-Walker equations of an AR(p)),
#       conditions I) and II) may become equivalent.
#
#   TYPE III) 
#       Positive average slope from lag 0 to lag h (weaker than I):
#       The CCF must be increasing on average from k = 0 to k = h, i.e.,
#       CCF(0) < CCF(h).

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────
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

tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)
tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)

# Diagnose whether we are in the non-standard case:
# If tauh > tau0, the MSE predictor lags the nowcast at frequency zero,
# which triggers the sign-inversion logic described in Wildi (2026), Appendix A.
if (tauh > tau0)
{
  print("Non-standard case: the MSE predictor lags the nowcast at frequency zero")
}



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
col_names <- colnames(filter_mat)

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
colnames(mplot) <- col_names
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
# -The PCS type III coefficients somehow resemble the DFP predictors in Tutorial 9, exercise 2.8.
#   -There is an important difference though: here, tauh>tau0 (non-standard case) wheres in tutorial 
#     9 exercise 2, the MSE predictor gammah was modified so that tauh<tau0 (artificially reinstalling the standard case).
#   -It is therefore surprising that the PCS partly recovers the profile of a solution based on inverted premises.  

# - The CCFs in the right panel peak at lead=1: the peak has not been shifted to h=12.
# - However, enforcing a negative average slope (beta<0) in the PCS constraint implies that CCF(0)-CCF(12) decreases 
#   and finally becomes negative (violet line), as confirmed below: 

ccf_mat["CCF at lead: 0",]-ccf_mat[paste("CCF at lead: ",h,sep=""),]

# -The difference could be made more negative by imposing a more negative beta in the constraint.



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
colnames(y_out_mat) <- col_names

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





###################################################################################################
###################################################################################################


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
# 3.3 PCS Optimisation over the Slope Grid
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

# ── Outcomes ─────────────────────────────────────────────────────────
#
# Left panel (filter coefficients):
#   - Unlike the MSE predictor, the PCS filters assign non-zero weight up to
#     the farthest lag k = q.
#   - As the slope parameter beta increases, weight is progressively shifted
#     away from recent observations toward older lags. This is counter-intuitive
#     but is a direct algebraic consequence of enforcing the CCF slope constraint.
#   - The most extreme PCS designs exhibit negative weights on older lags,
#     which causes trend inversion — a direct cost of aggressive look-ahead.
#
# Right panel (CCFs):
#   - The MSE predictors maximise the CCF at their respective horizons h and
#     h_tilde, but the peak remains at k = 0 in both cases (stuck at present).
#   - Very large lambda forces the CCF to follow a strictly linear path from
#     k = 0 to k = h = 5, regardless of whether this is necessary (assuming 
#     the problem is feasible).
#   - A positive slope parameter beta relocates the CCF peak toward h = 5,
#     as intended by the PCS design.
#   - Both the linear CCF constraint and a large positive beta are more
#     restrictive than necessary for look-ahead purposes, and both reduce
#     the achievable target correlation at k = h (the lowest peak value 
#     is attained by the largest beta PCS, violet line).
#   - The loss in target correlation at lag h is minimised subject to the
#     imposed constraints; however, the constraints themselves are overly
#     tight in this example, making the trade-off unnecessarily costly.

# Technical note:
#   As the regularisation weight lambda increases, the CCF slope at each
#   constrained lag converges to beta / (b' * b), i.e., the target slope
#   beta divided by the squared norm of the filter vector b. The dependence
#   on ||b||^2 arises naturally from the unconstrained structure of the
#   optimisation: removing it would require imposing a unit-norm constraint
#   on b, which complicates the geometry and typically leads to multiple
#   solutions (cf. the unitary DFP in Tutorial 4). Crucially, this
#   normalisation does not affect the look-ahead properties of the predictor,
#   since the CCF peak location is invariant to a common positive scaling
#   of b.


# ─────────────────────────────────────────────────────────────────────
# 3.6 Applying the Filters to Simulated Data
# ─────────────────────────────────────────────────────────────────────

# ── 3.6.1 Forecast Comparison ────────────────────────────────────────

# Generate a long white-noise series for reliable empirical evaluation.
set.seed(17)
len <- 10000
eps <- rnorm(len)

# Apply each filter to eps via causal (one-sided) convolution.
# filter(..., sides = 1) computes sum_{k=0}^{L-1} b_k * eps_{t-k}.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat,
                     filter(eps, filter_mat[, i], sides = 1))
colnames(y_out_mat) <- colnames(filter_mat)

colo <- c("black", "green", "darkgreen", rainbow(ncol(b_mat)))

# Plot a short excerpt to visually compare the temporal alignment of each
# predictor output relative to the target series.
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
#   As beta increases, the predictor output shifts progressively to the left
#   (further ahead in time) relative to the MSE predictor. This visual lead
#   is confirmed quantitatively by the empirical CCFs below.

# ── 3.6.2 Empirical CCF Comparison ───────────────────────────────────
# Compute empirical CCFs between the nowcast (x_t) and each predictor to
# confirm that the population peak shift observed in Section 3.5 is
# reproduced in finite-sample data.

par(mfrow = c(2, 2))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 2]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", h, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, 3]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: MSE(", htilde, "): Peak at k = 0 (no peak shift)"))

ccf(na.exclude(y_out_mat[, 1]),
    na.exclude(y_out_mat[, ncol(y_out_mat)]),
    lag.max = 10, plot = TRUE,
    main = paste0("CCF: strongest PCS predictor\n",
                  "Peak shifted from k = 0 to k = h = ", h))

# Outcome:
#   The empirical CCF confirms the population-level peak shift toward h = 5,
#   consistent with the population results in Section 3.5. However, because
#   lambda is very large and the slope beta is also unnecessarily large, the
#   imposed linear CCF constraint is overly tight: the predictor is forced to
#   satisfy restrictive slope requirements across all lags in {0, …, h},
#   reducing the achievable target correlation at h more than necessary.
#   Exercise 4 addresses this by holding beta fixed to a smaller value and 
#   varying the regularisation weight across a range of values, demonstrating 
#   how a lighter design can recover target correlation while preserving the
#   essential peak-shifting behaviour.

















