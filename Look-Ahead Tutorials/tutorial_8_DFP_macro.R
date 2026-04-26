# ════════════════════════════════════════════════════════════════════
# TUTORIAL 8 — APPLICATION OF THE DFP TO MONTHLY PAYEMS
# ════════════════════════════════════════════════════════════════════

# A brief overview of the DFP is provided in tutorial_3_DFP_overview.r.

# We consider an application of the interpretable MSE-DFP introduced in 
# Tutorial 6 to PAYEMS, an important monthly US business-cycle indicator. 
# Interpretability is achieved by applying the DFP time-shift constraint 
# introduced in Tutorial 6, which specifies the constraint of the DFP 
# directly in terms of a pre-specified lead at frequency zero, i.e. a
# corresponding left-shift of a linear trend by the predictor when compared 
# to the classic MSE predictor.

# ─────────────────────────────────────────────────────────────────────
# MOTIVATION
# ─────────────────────────────────────────────────────────────────────
# Without a unit-length constraint on the filter (unitary DFP, Tutorial 4),
# the parameter alpha0 in the MSE-DFP formulation represents a scale-dependent 
# covariance whose values are more difficult to interpret than the correlation 
# meaning in the unitary DFP (tutorial 4). Moreover, as discussed in Tutorial 5, 
# it is unclear how decoupling from the present value x_t translates into a 
# quantifiable lead over the MSE benchmark.
#
# To remedy this, we re-parameterise the DFP constraint by linking alpha0
# to the time-shift (phase delay) at frequency zero — the trend frequency —
# as introduced in Tutorial 2, see tutorial 6 for background.
#
# Concretely, the re-parameterised DFP predictor is designed so that its
# output leads the MSE predictor by a pre-specified number of time steps,
# tau, at frequency zero. This gives alpha0 a clear and interpretable
# meaning:
#
#   tau = 0  →  no lead relative to the MSE predictor (recovers MSE)
#   tau > 0  →  the DFP output anticipates the MSE predictor by tau
#               periods at the trend frequency
#
# The key question is: does this lead at frequency zero extend to
# business-cycle frequencies, and if so, to what extent?

#----------------------------------------------------------------------
# Notes on the chosen time series:
#   - PAYEMS is an important monthly US employment indicator, widely used
#     in business-cycle analysis.
#   - The series is non-stationary; predictors are therefore applied to the
#     stationary first differences of the log-transformed series.
#   - Tracking and forecasting the growth rate is a practically relevant
#     objective in real-time business-cycle analysis. In particular, leading
#     at zero-crossings of the differenced series corresponds to anticipating
#     turning points (local maxima and minima) of the original log-level series.
#----------------------------------------------------------------------
# Notes on the chosen model:
#
#   - Data generating process:
#       This tutorial uses an ARMA(2,2) model as the data generating process.
#
#   - Business-cycle periodicity:
#       The AR(2) component exhibits periodicity with a cycle length of
#       approximately 6 years, consistent with typical business-cycle
#       frequencies.
#
#   - Natural decoupling via periodicity:
#       As a direct consequence of this periodicity, the MSE predictor
#       naturally decouples from the concurrent observation x_t in this
#       example — without any additional constraints. Therefore, decoupling 
#       by the DFP is not strictly required in this framework.
#
#   - Motivation for using the DFP:
#       Although explicit decoupling via the DFP is not strictly necessary
#       in this example — since the MSE predictor already decouples naturally —
#       applying the DFP here is nonetheless instructive, as it underscores 
#       interpretability.
#
#   - Upcoming comparison:
#       A simpler, aperiodic ARMA(1,1) model will be examined in tutorial 9,
#       providing a useful contrast to the periodic AR(2,2) setting.

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
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
# EXERCISE 1: DFP APPLIED TO MONTHLY PAYEMS (EMPLOYMENT BCA INDICATOR)
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
# The log transformation stabilises the variance as the level of the
# series grows over time.
y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y) <- index(PAYEMS["1990::2019"])

plot(y, main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = FALSE,
     xlab = "", ylab = "")
axis(1, at = 1:length(y), labels = names(y))
axis(2)
box()

# Compute stationary first differences of the log-series:
#   - The log transformation stabilises the variance.
#   - The difference removes the stochastic trend (stabilises the level).
x <- diff(y)

# The differenced log-PAYEMS series is fairly noisy, with pronounced
# downturns during recession episodes.
ts.plot(x)

# The ACF decays slowly and monotonically, consistent with an AR-type
# dependence structure — this suggests the MSE predictor will be
# "stuck at the present" (see Tutorial 1).
acf(x,main="Slowly monotonically decaying empirical ACF")



# ─────────────────────────────────────────────────────────────────────
# 1.2 Model Fit: ARMA(2,2) with Periodic AR(2) Component
# ─────────────────────────────────────────────────────────────────────

L <- 50   # Filter length (number of MA coefficients retained).

# Fit an ARMA(2,2) model: a parsimonious specification with fairly good 
# diagnostics (an alternative ARMA model specification is analysed in 
# exercise 3 below).
ar_order <- 2
ma_order <- 2

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))

tsdiag(arima.obj)

# The AR(2) part is periodic with periodicity:
a1<-arima.obj$coef[1]
a2<-arima.obj$coef[2]
b1<-arima.obj$coef[3]
b2<-arima.obj$coef[4]
cos_theta <- a1 / (2 * sqrt(-a2))
theta     <- acos(cos_theta)
# --- Period ---
period <- 2 * pi / theta
# --- Period in years ---
period/12

# --- Wold Decomposition (MA-Infinity Representation) ---
# Compute the infinite-order MA coefficients (impulse response weights) of
# the fitted ARMA model. The filter length L = 50 was chosen to ensure that
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

# Visualise the Wold coefficients: the slow decay confirms the longer-memory
# character of the post-1990 log-differences relative to the full
# post-WWII sample.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold Decomposition: Slowly Decaying Impulse Response (Post-1990)")

# The theoretical ACF implied by the Wold decomposition closely matches
# the empirical ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),main="Slowly decaying Model-ACF (damped cycle)")

# The slowly decaying (damped cycle) ACF pattern means that the MSE predictor
# will NOT be stuck at the present; see Tutorial 1.

# The target weights are set equal to the Wold decomposition coefficients,
# meaning the predictor targets the original data (monthly log-differences
# of PAYEMS). A more complex target is considered in Exercise 3.
gamma <- xi


# ─────────────────────────────────────────────────────────────────────
# 1.3 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# Two forecast horizons are considered:
#   h      — the primary one-year horizon used for the MSE-DFP predictor
#   htilde — a longer 2-year horizon used as an additional reference

h <- 12    # primary forecast horizon (one year ahead)
# Longer forecast horizon used as a longer-term benchmark
htilde <- 2*h

# Nowcast:
# Truncate the MA expansion to length L for the nowcast/MSE filter (gamma0)
gamma0 <- gamma[1:L]

# h-step-ahead cross-correlation vector (gammah): shift gamma by h positions.
gammah <- gamma[h + 1:L]

# Check that transfer functions are positive at frequency zero.
# Otherwise the time-shift decoupling DFP should not be used.
if (sum(gammah)<0|sum(gamma0)<0)
{
  print("#####################################################")
  print("Time shifts at frequency zero are not well-defined")
  print("Nowcast or MSE forecast do not preserve trend orientation")
  print("#####################################################")
}  


# Check: gamma0 and gammah must be non-collinear; if they are, the DFP
# optimisation is degenerate and cannot be solved without additional interventions.
if (abs(abs(gamma0 %*% gammah) - sqrt(sum(gamma0^2) * sum(gammah^2))) < 1e-15)
{
  print("Warning: gammah and gamma0 are nearly collinear — the DFP predictor will not be computed.")
  return()
}


# Analogous cross-correlation vector for the longer horizon htilde
gammahtilde <- gamma[htilde + 1:L]

# Increasing the forecast horizon shifts the cyclical pattern in the predictor
# progressively to the left:

ts.plot(scale(cbind(gamma0, gammah, gammahtilde)), col = c("black", "green", "orange"))
mtext("Wold decomposition",col="black",line=-1)
mtext(paste("MSE(",h,") predictor",sep=""),col="green",line=-2)
mtext(paste("MSE(",htilde,") predictor",sep=""),col="orange",line=-3)


#   - Decoupling via horizon selection:
#       Arbitrarily strong decoupling from the concurrent observation x_t
#       can be achieved simply by increasing the forecast horizon h,
#       without imposing any explicit decoupling constraint.
#
#   - Out-of-phase predictor at h ~ 36:
#       At approximately h = 36 (roughly half the cycle period of ~72 months),
#       the MSE predictor becomes out-of-phase with x_t. At this horizon,
#       the correlation between the predictor and x_t is negative,
#       reflecting the anti-phase relationship between the predictor
#       and the concurrent observation.


# Desired lead of the DFP predictor over the MSE predictor at frequency zero
# (negative value = the DFP output leads by |lead| time steps at the zero 
# trend frequency). We analyse a sequence of increasing leads at frequency zero.
lead_vec <- c(-2^(0:5),-200)

# Note on DFP behavior under large leads:
#
#   - Constraint at frequency zero:
#       The DFP is required to comply with the specified time-shift
#       at frequency zero (i.e., the zero-frequency constraint), while
#       simultaneously maximizing the correlation with x_{t+h} at the
#       target horizon h = 1 year ahead.
#       See tutorial 6 for theoretical background on this constraint.
#
#   - Instructive stress test:
#       It is informative to observe how the DFP manages
#       the relatively large leads imposed via lead_vec — balancing the
#       zero-frequency time-shift requirement against optimal tracking
#       of the target x_{t+h} (conflicting requirements).

lead_vec

# ─────────────────────────────────────────────────────────────────────
# 1.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute the frequency-zero time-shifts, see tutorial 2.
tau0 <- sum((0:(L - 1)) * gamma0) / sum(gamma0)
tauh <- sum((0:(L - 1)) * gammah) / sum(gammah)

# Note:
# The 24-steps-ahead MSE predictor does not preserve trend orientation: the
# sum of its coefficients is negative (i.e., the transfer function at
# frequency zero is negative), as confirmed below.
sum(gammahtilde)

# Illustration
trend<-1:100
mse24<-NULL
for (i in L:100)
  mse24[i]<-gammahtilde%*%trend[i:(i-L+1)]
# The direction of the predicted trend is inverted:
ts.plot(mse24)

# Consequently, defining a time-shift at frequency zero is not meaningful
# for gammahtilde. The formula can nonetheless be evaluated:
sum((0:(L - 1)) * gammahtilde) / sum(gammahtilde)

# In this case, the formula computes the time-shift of the sign-inverted
# filter -gammahtilde at frequency zero; see Section 4.1, Wildi (2024).
# Since the time-shift is not defined for gammahtilde itself, it is set to NA:
tauhtilde <- NA


# Verify the lead of the MSE predictor over the nowcast at frequency zero.
#   - In the standard case, tauh - tau0 < 0, i.e., the MSE predictor leads
#     the nowcast (a linear trend is left-shifted by gammah).
#   - In the non-standard case, tauh > tau0, i.e., the predictor lags the
#     nowcast; this (rather unusual) case is addressed in Tutorial 9.

tauh - tau0


# Call the dedicated function to compute the DFP filter for a specified lead
# (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)

b_mat<-lambda_vec<-alpha_vec<-NULL
for (i in 1:length(lead_vec))
{
# Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj))
  {

# Extract the components returned by the function
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    alpha0 <- as.double(t(gamma0) %*% b)
  

# Normalise b to unit length (simplifies comparisons in plots below)
    b_tau <- b / as.double(sqrt(b %*% b))
    b_mat<-cbind(b_mat,b_tau)
    lambda_vec<-c(lambda_vec,lambda0)
    alpha_vec<-c(alpha_vec,alpha0)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}
# ─────────────────────────────────────────────────────────────────────
# 1.5 Validation
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved leads match the specified leads ---
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

# --- Check 2: verify that the DFP preserve trend orientation ---
# The sums of the coefficients are positive (the transfer functions at 
# frequency zero are positive, see Wildi 2026, section 4.1). 
apply(b_mat,2,sum)
# Technical note: when imposing the DFP constraint in terms of left-shift at 
# frequency zero, the DFP cannot invert trend orientation, i.e., the above sums 
# must be positive.


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

# Assemble all predictors: Scale now- and MSE -forecasts to unit-length (as DFP)
filter_mat<-cbind(gamma0/as.double(sqrt(gamma0%*%gamma0))
                  ,gammah/as.double(sqrt(gammah%*%gammah)),
                  gammahtilde/as.double(sqrt(gammahtilde%*%gammahtilde)),
                  b_mat,b_cd)
colnames(filter_mat)<-c("Nowcast",paste("MSE(",h,")",sep=""),
          paste("MSE(",2*h,")",sep=""),paste("DFP ",lead_vec,sep=""),"DFP FD")


# ─────────────────────────────────────────────────────────────────────
# 1.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0
t(b_cd) %*% gamma0

# Note on phase reversal:
# As in Tutorial 6 (Exercise 1.9), the fully decoupled DFP inverts the
# orientation of the trend (i.e., changes the sign of the mean): the sum
# of its filter coefficients is negative:
sum(b_cd)

# Note: in Exercise 3 below, which addresses a more interesting forecast target,
# the fully decoupled design preserves trend orientation: the sum of filter 
# coefficients will be positive.

# Despite this inversion, the usual time-shift formula (evaluated at frequency
# zero) can still be applied. Note, however, that the result corresponds to
# the time-shift of the sign-inverted filter -b_cd:
#   - b_cd inverts the direction of the trend (see plots below).
#   - Hence -b_cd preserves the trend direction.
#   - The classical time-shift formula computes the shift of -b_cd;
#     see Section 4.1, Wildi (2026).
tau_cd <- sum((0:(L - 1)) * b_cd) / sum(b_cd)
tau_cd

# As for the 2-two year ahead MSE, the time-shift of b_cd is set to NA since 
# b_cd does not preserve trend orientation.
tau_cd<-NA



# ─────────────────────────────────────────────────────────────────────
# 1.8 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise four key performance metrics for each predictor:
#   tau(0)    — frequency-zero time-shift (positive = right-shift or lag when applied to linear trend)
#   lambda    — DFP weight on gamma0
#   alpha0    — inner product <gamma0, b> (DFP constraint value: covariance of DFP and nowcast)
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
# Time-shift DFP and completely decoupled DFP: 
mat_perf[3:nrow(mat_perf),1]<-c(tau_vec,NA)
mat_perf[3:nrow(mat_perf),2]<-c(lambda_vec,lambda_cd)
mat_perf[3:nrow(mat_perf),3]<-c(alpha_vec,alpha0_cd)


# Note: NAs indicate that the corresponding entries are left blank (not meaningful)
mat_perf

# Discussion:
#
# 1. tau(0) column — time-shift at frequency zero:
#
#     - MSE(12) predictor (h = 12):
#         The classic 12-steps-ahead MSE predictor exhibits a time-shift of
#         approximately 1.4 at frequency zero.
#         Interpretation: when this filter is applied to a linear trend, the
#         trend component is delayed (right-shifted) by 1.4 time points.
#         See tutorial 6 for a detailed explanation of this interpretation.
#
#     - MSE(24) predictor (h = 24):
#         Increasing the forecast horizon to h = 24 produces a predictor that
#         inverts both the trend and the mean — an undesirable property in
#         most forecasting applications (except when the process is strongly 
#         periodic).
#
#     - Tau-shifted DFPs:
#         The time-shifts at frequency zero of the DFP variants differ from 
#         that of MSE(12) by exactly the values specified in lead_vec, 
#         by construction.
#
# 2. lambda — DFP weight on gamma0: b = gammah + lambda*gamma0
#
#     - The estimated lambda governs the rotation between the MSE predictor
#       and the decoupling direction.
#     - In the standard case, a negative lambda implies that gammah lies 
#       between b and gamma0 in the common plane.
#     - More negative values of lambda correspond to a stronger rotation,
#       as illustrated in tutorial 5, exercise 1.6.
#
# 3. alpha0 — MSE-DFP constraint parameter:
#
#     - alpha0 is the raw constraint parameter of the MSE-DFP formulation.
#       It should not be interpreted as a correlation coefficient,
#       except in the special case where it equals zero. 
#       Note: for the unitary DFP presented in tutorial 4, alpha0 IS a correlation.
#     - Reformulating the DFP constraint in terms of tau (the time-shift)
#       rather than alpha0 substantially improves interpretability, see tutorial 6.
#     - By construction, the fully decoupled DFP corresponds to a
#       vanishing alpha0 (i.e., alpha0 = 0). In this case, however, the fully 
#       decoupled DFP inverts trend orientation, which is undesirable.

# Note:
#   The fully decoupled design cannot be recovered by imposing a suitable
#   time-shift constraint in this exercise. Full decoupling is a strictly
#   stronger requirement than imposing an infinite lead at frequency zero.
#   This contrasts with Exercise 3 below, where the fully decoupled design
#   is commensurate with imposing a finite lead at frequency zero.


# ─────────────────────────────────────────────────────────────────────
# 1.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

# Layout: two panels: filter coefficients (left) and CCF (right),
par(mfrow=c(1,2))

colo <- c("black", "green", rainbow(ncol(filter_mat)-2))

# Collect all four filters into a matrix (no scaling applied)
mplot<-filter_mat
col_names <- colnames(filter_mat)
colnames(mplot) <- col_names

# Diagnostic: sum of squared coefficients per filter (proxy for filter energy)
apply(mplot^2, 2, sum)
lty_vec<-lwd_vec<-c(2,2,2,rep(1,ncol(filter_mat)-3))

# --- Left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",lty=lty_vec[1],lwd=lwd_vec[1],
     xlab = "Lags", ylab = "",
     col  = colo[1], 
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)
# Overlay remaining filters and add colour-coded labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], type = "l",lty=lty_vec[i],lwd=lwd_vec[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()

# --- Right panel: cross-correlation functions (CCF) ---
# Compute the CCF between each predictor and the AR(3) process at lags
# surrounding lag 0 and the h-step-ahead lag
max_lag<-20
mplot<-NULL
for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot,compute_ccf_func(filter_mat[,i], gamma0)[L-1+1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",lty=lty_vec[1],lwd=lwd_vec[1],
     xlab = "", ylab = "",
     col  = colo[1], 
     ylim = c(min(mplot), max(mplot)))
# Overlay CCFs for DFP-shifted and fully decoupled DFP
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[ i],lty=lty_vec[i],lwd=lwd_vec[i])
}
abline(v =  1 + h, lty = 2)
abline(h = 0)

axis(1, at = 1:nrow(mplot),
     labels = -1+1:nrow(mplot))
axis(2)
box()


# Discussion:
# CCF (right panel)
#
#   - MSE 12-steps-ahead predictor (green, dashed):
#       Maximizes the CCF at the target horizon h = 12, by design.
#       However, it remains strongly correlated with x_t at lag 0,
#       indicating minimal decoupling from the concurrent observation.
#
#   - MSE 24-steps-ahead predictor (MSE(24): red, dashed):
#       Substantially decoupled from x_t — unlike in previous tutorials,
#       increasing the forecast horizon in this example *increases* decoupling.
#       However, this predictor maximizes the CCF at h = 24, not at the
#       intended target horizon h = 12.
#       For an equivalent level of decoupling, the DFP with shift -200
#       (violet) achieves the same degree of decoupling while maximizing
#       the CCF at h = 12 — no other linear predictor can improve the
#       target correlation under the same decoupling constraint.
#
#   - Tau-shifted DFPs (varying leads in lead_vec):
#       As the lead increases, these predictors become progressively less
#       correlated with x_t at lag 0. Consequently, the achievable CCF
#       at the target horizon h = 12 also decreases. Nevertheless, at
#       each given level of decoupling, the DFP minimizes this loss —
#       i.e., it retains as much target correlation at h = 12 as is
#       theoretically possible under the imposed constraint.
#
#   - Fully decoupled predictor (red):
#       By construction, the CCF at lag 0 vanishes entirely, reflecting
#       complete decoupling from x_t. This imposes the strongest reduction
#       in the achievable target correlation at h = 12. Nevertheless, the
#       predictor remains optimal in the sense that it maximizes the CCF
#       at h = 12 subject to the full decoupling constraint — no other
#       fully decoupled linear predictor can achieve a higher target correlation.
#
#   - Interestingly, imposing a large lead at frequency zero (-200, violet)
#     still allows the predictor to track x_{t+h} effectively at the intended
#     one-year horizon in this example.
#     An explanation will be provided in the frequency-domain analysis
#     (see exercise 1.12): the time-shift at frequency zero changes very
#     steeply for omega > 0, meaning that the large imposed lead at frequency
#     zero spills over to business-cycle frequencies only in substantially
#     reduced form, enabling optimal tracking of x_{t+h} (subject to the 
#     imposed constraint). 

# Coefficients (left panel):
#   Increasing the lead at frequency zero has three main effects on the
#   predictor coefficients:
#
#     1. Phase advancement (left-shift):
#          The minimum of the right half-cycle shifts to the left, reflecting
#          an increased lead. However, this phase advancement is less aggressive
#          than that of MSE(24) (red, shaded), which exhibits the most 
#          pronounced phase shift among all predictors shown.
#
#     2. Strengthened negative half-cycle:
#          The negative portion of the cyclical pattern in the coefficients
#          becomes more pronounced, contributing to the larger time-shift
#          at frequency zero through increased band-pass behavior (see amplitude 
#          functions below).
#
#     3. Reduced weight on the innovation epsilon_t:
#          In the MA representation of the predictor, the weight assigned
#          to the most recent innovation epsilon_t decreases as the lead grows.
#
#   Explanation of the DFP strategy:
#     These three effects reflect the DFP balancing competing objectives:
#       - To remain tightly connected to x_{t+12} (the target), the phase
#         advancement must be kept moderate — avoiding an overly large shift.
#       - To disconnect from x_t (the concurrent observation), the weight
#         on epsilon_t is reduced.
#       - The large time-shift at frequency zero is then achieved primarily
#         through enhanced band-pass behavior, i.e., a stronger negative
#         half-cycle in the filter coefficients.
#
#   Note: The DFP adopts a markedly different `strategy' to reconcile  
#   mutually conflicting requirements in tutorial 9,
#   where the underlying ARMA(1,1) model is aperiodic and the periodic
#   structure exploited here is absent.




# ─────────────────────────────────────────────────────────────────────
# 1.10 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────

#----------------------------------------------------------------------
# 1.10.1 Apply Predictors to Data
#----------------------------------------------------------------------

# All predictors are defined in MA form, i.e., as linear filters applied
# to the innovations eps_t from the Wold decomposition of the process.
# Accordingly, the filters are applied to the model residuals, which
# serve as empirical proxies for the innovations eps_t.
# Note: the AR form of the predictors is derived separately in exercise 2.

# Extract model residuals (proxy for Wold innovations eps_t)
x_filt <- arima.obj$residuals

# Apply each filter (predictor) to the residuals sequentially,
# collecting the filtered output into a matrix of forecast series.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))

# Assign predictor names to columns for identification in subsequent plots
colnames(y_out_mat) <- col_names


#----------------------------------------------------------------------
# 1.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# The entire time span
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,2,rep(1,ncol(filter_mat)-3)),lwd=c(2,2,2,rep(1,ncol(filter_mat)-3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)

# Dotcom crisis
ts.plot(y_out_mat[120:170,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,2,rep(1,ncol(filter_mat)-3)),lwd=c(2,2,2,rep(1,ncol(filter_mat)-3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Financial crisis
ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",
        lty=c(2,2,2,rep(1,ncol(filter_mat)-3)),lwd=c(2,2,2,rep(1,ncol(filter_mat)-3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)


# Discussion:
#
#   - Effect of increasing the lead (time-shift at frequency zero):
#       Increasing the lead progressively left-shifts the DFP predictors
#       in the time domain, reflecting the earlier anticipation of the
#       target x_{t+h}.
#
#   - MSE(24) — largest left-shift, but at a cost:
#       The MSE(24) predictor exhibits the strongest left-shift among all
#       predictors shown. However, this shift is a mechanical consequence
#       of increasing the forecast horizon: as h grows, the phase advancement
#       eventually leads to a complete phase reversal (sign inversion of the
#       predicted cycle), which is no longer meaningful or interpretable in
#       economic terms.
#
#   - DFP predictors — controlled left-shift with preserved target link:
#       In contrast, the DFP predictors achieve a targeted left-shift while
#       maintaining the strongest possible correlation with x_{t+12} at the
#       intended (practically relevant) one-year forecast horizon. As shown in 
#       exercise 1.9 above, the phase-shift in the filter coefficients 
#       (left panel) is deliberately kept less pronounced than that of 
#       MSE(24), in order to preserve a tight link with the target x_{t+12}.
#
#   - Verification:
#       This key property of the DFP — maximizing the correlation with x_{t+12}
#       under the imposed decoupling constraint — is verified empirically in
#       the simulation exercise that follows. Specifically, we evaluate and
#       compare MSE(24) and the DFP with lead -200, which exhibit nearly
#       identical decoupling from x_t (as confirmed by the CCF plot in
#       exercise 1.9 above), but differ in their ability to track x_{t+12}
#       at the intended forecast horizon.


# ─────────────────────────────────────────────────────────────────────
# 1.11 Simulation Exercise
# ─────────────────────────────────────────────────────────────────────
# Purpose: verify empirically that, for a comparable level of decoupling
# from x_t, the DFP with lead -200 outperforms MSE(24) in tracking the
# target x_{t+12} at the intended one-year forecast horizon.
# ─────────────────────────────────────────────────────────────────────

#----------------------------------------------------------------------
# 1.11.1 Apply Predictors to Simulated Data
#----------------------------------------------------------------------

# Simulate a long realization of the AR(2,2) process to obtain
# reliable empirical correlation estimates.
len_sim <- 1000000
set.seed(37)
x <- eps <- rnorm(len_sim)

# Generate the ARMA(2,2) process recursively from the innovations eps.
for (i in 3:len_sim)
  x[i] <- a1*x[i-1] + a2*x[i-2] + eps[i] + b1*eps[i-1] + b2*eps[i-2]

# Select the two filters to be compared:
#   - MSE(24): MSE-optimal predictor at h = 24 (strongly decoupled from x_t,
#              but targets h = 24 rather than h = 12).
#   - DFP -200: DFP with time-shift -200 at frequency zero (similarly decoupled
#               from x_t, but designed to maximize tracking at h = 12).
filter_sim <- cbind(filter_mat[, "MSE(24)"], filter_mat[, "DFP -200"])
colnames(filter_sim) <- c("MSE(24)", "DFP -200")

# Apply each filter to the simulated innovations eps,
# collecting the filtered outputs into a matrix of forecast series.
y_out_mat_sim <- NULL
for (i in 1:ncol(filter_sim))
  y_out_mat_sim <- cbind(y_out_mat_sim, filter(eps, filter_sim[, i], side = 1))

# Assign predictor names to columns for identification in output.
colnames(y_out_mat_sim) <- colnames(filter_sim)

#----------------------------------------------------------------------
# 1.11.2 CCF — Decoupling and Tracking Performance
#----------------------------------------------------------------------

# Compute two key performance metrics for each predictor:
#   Row 1 — Decoupling: correlation with x_t at lag 0 (lower is better).
#   Row 2 — Tracking:   correlation with x_{t+h} at the target horizon
#                        h = 12 (higher is better).
perf_mat <- c(cor(x[L:len_sim], y_out_mat_sim[L:len_sim, 1]),
              cor(x[L:len_sim], y_out_mat_sim[L:len_sim, 2]))

perf_mat <- rbind(perf_mat,
                  c(cor(x[(L+h):len_sim], y_out_mat_sim[L:(len_sim-h), 1]),
                    cor(x[(L+h):len_sim], y_out_mat_sim[L:(len_sim-h), 2])))

colnames(perf_mat) <- c("MSE(24)", "DFP -200")
rownames(perf_mat) <- c("Decoupling at lag 0",
                        paste("Tracking at forecast horizon h =", h))

perf_mat

# Discussion:
#   - For a comparable level of decoupling from x_t (row 1), the DFP with
#     lead -200 achieves strictly higher tracking of x_{t+12} at h = 12
#     (row 2) than MSE(24) — confirming the theoretical optimality of the DFP.
#     No other predictor can improve tracking of x_{t+12} under the imposed 
#     decoupling.
#
#   - Increasing the forecast horizon of the MSE predictor causes the predictor
#     to become progressively out-of-phase with x_{t+12}: the predicted cycle
#     shifts further and further to the left, creating a superficial appearance
#     of anticipation when predictors are viewed as a sequence of increasing
#     horizons. However, this apparent lead is a mechanical phase artifact
#     rather than genuine forecasting skill — and once the phase reversal
#     (sign inversion) sets in, the resulting predictor is uninterpretable
#     in meaningful terms (at least in the absence of strong periodicity in the 
#     data).
#
#   - THE DFP PRESERVES INTERPRETABILITY by keeping the focus firmly on the
#     intended forecast horizon h: it looks ahead of the classic MSE(h) design
#     — achieving greater decoupling from x_t — while ensuring that the
#     predictor remains anchored to x_{t+h} as the primary optimization target.

# ─────────────────────────────────────────────────────────────────────
# 1.12 Amplitude and Time-Shifts
# ─────────────────────────────────────────────────────────────────────

K      <- 600      # number of frequency grid points
plot_T <- FALSE    # suppress internal plotting; we build a custom plot below
amp_mat<-shift_mat1<-shift_mat2<-NULL
for (i in 1:ncol(filter_mat))
{
  as_obj <- amp_shift_func(K, filter_mat[,i], plot_T)   # time-shift for lagged filter (b1)
  amp_mat<-cbind(amp_mat,as_obj$amp)
  shift<-as_obj$shift
  # Ignore too negative shifts  
  shift[which(shift<(-200))]<-NA
  shift_mat1<-cbind(shift_mat1,shift)
  # Ignore too negative shifts  
  shift[which(shift<(-3))]<-NA
  shift_mat2<-cbind(shift_mat2,shift)
}
colnames(amp_mat)<-colnames(shift_mat1)<-colnames(shift_mat2)<-colnames(filter_mat)


# Plot amplitude functions across frequencies [0, π]
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

# Plot time-shift functions (phase divided by frequency) across frequencies [0, π]
mplot <- shift_mat1
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


# Outcome:
#
#   Amplitude functions:
#     - Increasing the lead (time-shift at frequency zero) progressively
#       morphs the filter from a lowpass design toward a bandpass-like design.
#       This reflects the DFP's strategy of amplifying business-cycle frequencies
#       to achieve the large imposed lead while preserving target correlation
#       at h = 12 (as discussed in exercise 1.9).
#
#   Time-shift functions:
#     - The time-shifts at frequency zero match the imposed leads exactly,
#       confirming that the zero-frequency constraint is satisfied by construction.
#     - The imposed leads change steeply as a function of frequency for omega > 0:
#       the extreme leads at frequency zero do not spill over to business-cycle
#       frequencies. This is a critical property — if the large lead were to
#       propagate to business-cycle frequencies, the target correlation at
#       h = 12 would not be maximal (subject to the time-shift constraint).
#
#   - A more detailed analysis of the time-shift behavior is obtained in the
#     following plot, where extreme leads at lower frequencies are removed
#     to allow closer inspection of the shift at business-cycle frequencies.




par(mfrow=c(1,1))
mplot <- shift_mat2

plot(mplot[, 1], type = "l", axes = FALSE,lty=lty_vec[1],
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift functions",
     ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))), col = colo[1])
#mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i],lty=lty_vec[i])
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# The time-shifts of the DFP variants propagate in a controlled manner
# across business-cycle frequencies:
#
#   - For frequencies below the crossing point near omega = pi/12 (i.e., at
#     lower frequencies, including the trend), increasing the lead at
#     omega = 0 corresponds to a progressively smaller (but controlled) 
#     time-shift.
#
#   - For frequencies above the crossing point near omega = pi/12 (i.e., at
#     higher, shorter-cycle frequencies), the ordering reverses: predictors
#     with larger imposed leads actually exhibit *larger* time-shifts.
#     This effect is visible in exercise 1.10.2, where some high-lead DFP
#     designs appear to lag slightly at noisy short-term dynamics.
#
#   - MSE(24) exhibits the smallest time-shift across business-cycle
#     frequencies among all predictors shown. However, this comes at the
#     cost of a phase reversal at trend frequencies — the shift already
#     inverts the trend direction (see exercise 1.4), rendering the predictor
#     difficult to interpret in economic terms, as discussed previously.
#
#   - The fully decoupled DFP also inverts the trend direction, highlighting
#     that full decoupling represents an extreme DFP design in this example. 
#     While useful for exploring the theoretical limits of the approach and 
#     understanding the decoupling-tracking trade-off, it is generally too 
#     extreme to be practically recommended as a forecasting tool.


# The next exercise replicates the above analysis in the more natural AR form
# of the predictors: rather than applying the MA-form filters to the model
# residuals (Wold innovations, see exercise 1.10), the equivalent AR-form 
# filters are applied directly to the original observed data x_t. This 
# representation is more natural and practical, as it expresses each predictor 
# as a weighted sum of past observations rather than past innovations.

# ─────────────────────────────────────────────────────────────────────
# Exercise 2. AR Form
# ─────────────────────────────────────────────────────────────────────

y   <- as.double(log(PAYEMS["1990::2019"]))
len <- length(y)
names(y) <- index(PAYEMS["1990::2019"])

plot(y, main = "Log(PAYEMS): 1990–2019",
     type = "l", axes = FALSE,
     xlab = "", ylab = "")
axis(1, at = 1:length(y), labels = names(y))
axis(2)
box()

# Compute stationary first differences of the log-series:
#   - The log transformation stabilises the variance.
#   - The difference removes the stochastic trend (stabilises the level).
x <- diff(y)

# The differenced log-PAYEMS series is fairly noisy, with pronounced
# downturns during recession episodes.
ts.plot(x,main="Log-differences PAYEMS")

# We now convert the MA-form predictors to their AR equivalents by
# convolving each predictor with the AR(3) operator. A similar proceeding
# applies to the unitary DFP in tutorial 4.

# --------------------------------------------------------------------------
# 2.1 AR inversion
# --------------------------------------------------------------------------

# MA inversion: Wold decomposition
xi <- c(1, ARMAtoMA(
  ar      = arima.obj$coef[1:ar_order],
  ma      = arima.obj$coef[ar_order + 1:ma_order],
  lag.max = length(x)
))

# AR inversion
ar_inv <- -ARMAtoMA(ar = -arima.obj$coef[ar_order + 1:ma_order], ma = -arima.obj$coef[1:ar_order], lag.max = max_lag)
# AR-filter
theta<-c(1,-ar_inv)


# Verify the approach via a known identity:
# Convolving the AR inversion with the Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros). 
conv_two_filt_func(xi, theta)$conv[1:10]


# Visualise theta: 
par(mfrow = c(1, 1))
ts.plot(theta, main = "AR inversion (Post-1990)")


# Having confirmed the identity, we now convolve the AR operator with
# the MSE and DFP predictors (in MA form) to obtain their AR equivalents.

# --------------------------------------------------------------------------
# 2.2 Convolution of the AR inversion with the Predictors
# --------------------------------------------------------------------------

# a. MSE predictor: convolve the AR operator with the predictors.

# Check: convolution of theta with gamma0 should be the identity: smaller 
# deviations become vanishing with increasing L (length of finite MA and AR inversions)
conv_two_filt_func(theta, gamma0)$conv

# b. DFP predictors
filter_mat_ar<-NULL
for (i in 1:ncol(filter_mat))
  filter_mat_ar<-cbind(filter_mat_ar,conv_two_filt_func(theta, filter_mat[,i])$conv)

colnames(filter_mat_ar)<-colnames(filter_mat)

# Check: the first column (the nowcast) should be the identity.
# Note however that we scaled gamma0 to unit length: so the first weight 
# generally differs from one. 
filter_mat_ar[1:10,1]

# --------------------------------------------------------------------------
# 2.3 Analysis and Plot of DFP Predictors in AR Form
# --------------------------------------------------------------------------
# Key structural property of the DFP predictor in AR form:
#
# Only the FIRST AR coefficient varies across DFP designs; all higher-order
# AR coefficients are identical regardless of the chosen lambda.
#
# Theoretical justification (Wildi 2026, Section 3.1, Equation 19):
#   The DFP filter is defined as:
#     b = gammah + lambda * gamma0
#   When b is converted to AR form by inverting gamma0, the term
#   lambda * gamma0 maps to lambda * identity (a pure scalar shift).
#   Because gammah is fixed and convolution is linear, this scalar shift
#   affects only the first AR coefficient, leaving all higher-order
#   coefficients unchanged across DFP designs.

# Assign colors
colo <- c("black", "green", rainbow(ncol(filter_mat_ar) - 2))

first_lags <- 10

# Plot the first first_lags AR coefficients of each predictor.
ts.plot(
  filter_mat_ar[1:first_lags, ],
  col  = colo,
  main = "DFP Predictors in AR Form"
)
for (i in 1:ncol(filter_mat_ar))
  mtext(colnames(filter_mat_ar)[i], col = colo[i], line = -i)

# Important note on scaling:
#   In principle, only the first AR coefficient differs across DFP designs
#   (see exercise 2.3, tutorial 5). However, here the predictors have been
#   scaled to unit length (normalized), which distributes the difference
#   across all coefficients and obscures the simple structural invariance
#   in the plot above.
#   Without normalization, and for filter length L sufficiently large,
#   only the first AR coefficient vary across DFP designs —
#   all remaining coefficients are identical, as shown in the following plot.

b_mat_unscaled<-NULL
for (i in 1:length(lead_vec))
{
  # Lead over MSE at frequency zero  
  lead<-lead_vec[i]
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj))
  {
    b       <- dfp_obj$b        # raw DFP filter coefficients
    b_mat_unscaled<-cbind(b_mat_unscaled,b)
  } else
  {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}
filter_mat_ar_unscaled<-NULL
for (i in 1:ncol(b_mat_unscaled))
  filter_mat_ar_unscaled<-cbind(filter_mat_ar_unscaled,conv_two_filt_func(theta, b_mat_unscaled[,i])$conv)

ts.plot(filter_mat_ar_unscaled,col=rainbow(ncol(filter_mat_ar_unscaled)),main="Only the first weight is affected")


# ─────────────────────────────────────────────────────────────────────
# 2.4 Show Equivalence of MA and AR Forms
# ─────────────────────────────────────────────────────────────────────

#----------------------------------------------------------------------
# 2.4.1  Apply Predictors to data
#----------------------------------------------------------------------


# All filters are defined in AR form (as applied to x_t)
x_filt   <- x

y_out_mat_ar<-NULL
for (i in 1:ncol(filter_mat))
  y_out_mat_ar<-cbind(y_out_mat_ar,filter(x_filt,filter_mat_ar[, i], side = 1))
colnames(y_out_mat_ar) <- col_names

# Check:
# Compare predictors in MA form (y_out_mat) and in AR form (y_out_mat_ar)
# 1. They are virtually identical up to an offset (the mean) which is ignored 
# by the MA form.
# 2. The small deviations after scaling can be made vanishingly small by increasing L.
# Select any of the columns in y_out_mat and y_out_mat and compare standardized series.
# Standardization removes the mean offset.
k<-4
# k cannot be larger than the number of columns of y_out_mat
k<-min(k,ncol(y_out_mat))

ts.plot(scale(cbind(y_out_mat[,k],y_out_mat_ar[,k])),main="MA- and AR-forms of DFP overlap")



#----------------------------------------------------------------------
# 2.4.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))
# Plot a representative excerpt (obs. 300–350) to compare predictor outputs visually
ts.plot(y_out_mat_ar,
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(y_out_mat_ar)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat_ar))
  mtext(colnames(y_out_mat_ar)[i],col=colo[i],line=-i)



ts.plot(y_out_mat[200:250,],
        main = "Predictor Outputs", col = colo, xlab = "", ylab = "",lty=c(2,2,rep(1,ncol(filter_mat)-2)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i],col=colo[i],line=-i)



# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: FORECASTING SMOOTH GROWTH
# ════════════════════════════════════════════════════════════════════
# This exercise extends the DFP framework from exercise 1 to a smoothed
# growth target: instead of targeting the raw differenced log-PAYEMS
# series, we target yearly growth — defined as an equally-weighted
# 12-month moving average of the differenced log series. This reduces
# noise and improves interpretability of the forecast objective.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 3.1 Framework
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
# 3.2 Model Fit
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

# ─────────────────────────────────────────────────────────────────────
# 3.3 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# Settings are identical to exercise 1; repeated here for self-containedness.

h      <- 12     # primary forecast horizon: one year ahead
htilde <- 2 * h  # longer benchmark horizon: two years ahead

# Nowcast filter (gamma0): truncate the Wold expansion of yearly grwoth 
# to length L.
gamma0 <- gamma[1:L]

# h-steps-ahead MSE filter (gammah): shift the Wold expansion by h positions.
gammah <- gamma[h + 1:L]

# Check that the transfer functions at frequency zero are positive.
# If either sum is negative, the predictor inverts the trend direction,
# and the time-shift-based DFP decoupling is not well-defined.
if (sum(gammah) < 0 | sum(gamma0) < 0) {
  print("#####################################################")
  print("Time shifts at frequency zero are not well-defined")
  print("Nowcast or MSE forecast do not preserve trend orientation")
  print("#####################################################")
}

# Wold coefficients for the longer benchmark horizon htilde.
gammahtilde <- gamma[htilde + 1:L]
# Note: in contrast to exercise 1 above, gammahtilde is not inverting trend 
# direction: the sum of coefficients is larger zero:
sum(gammahtilde)

# Grid of desired leads of the DFP over the MSE(12) predictor at frequency zero.
# Negative values indicate that the DFP leads the MSE predictor (anticipates 
# earlier). We use the same as in exercise 1 (but discard the extreme lead).
lead_vec <- -2^(0:5)

# ─────────────────────────────────────────────────────────────────────
# 3.4 Run Time-Shift MSE-DFP
# ─────────────────────────────────────────────────────────────────────

# Compute frequency-zero time-shifts of the reference filters as benchmarks.
tauhtilde <- sum((0:(L-1)) * gammahtilde) / sum(gammahtilde)
tau0      <- sum((0:(L-1)) * gamma0)      / sum(gamma0)
tauh      <- sum((0:(L-1)) * gammah)      / sum(gammah)

# Verify standard case: the MSE predictor leads the nowcast at frequency zero
# (tauh < tau0). The non-standard case (tauh >= tau0) is analysed in tutorial 9.
tauh - tau0

# Compute the DFP filter for each specified lead in lead_vec.
# Uses mse_dfp_from_tau_func(), which implements Theorem 2 (Wildi 2026).
b_mat <- lambda_vec <- alpha_vec <- NULL

for (i in 1:length(lead_vec)) {
  
  lead <- lead_vec[i]  # desired lead of DFP over MSE at frequency zero
  
  dfp_obj <- mse_dfp_from_tau_func(gamma0, gammah, lead)
  
  if (!is.null(dfp_obj)) {
    
    # Extract output components from the DFP function
    tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
    tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
    lambda0 <- dfp_obj$lambda0  # DFP regularization weight on gamma0
    b       <- dfp_obj$b        # raw DFP filter coefficients
    
    # Compute the decoupling inner product <gamma0, b>
    alpha0 <- as.double(t(gamma0) %*% b)
    
    # Normalize b to unit length for comparability across designs
    b_tau  <- b / as.double(sqrt(b %*% b))
    b_mat  <- cbind(b_mat, b_tau)
    lambda_vec <- c(lambda_vec, lambda0)
    alpha_vec  <- c(alpha_vec, alpha0)
    
  } else {
    print("Warning: gammah and gamma0 are nearly collinear")
  }
}

# ─────────────────────────────────────────────────────────────────────
# 3.5 Validation
# ─────────────────────────────────────────────────────────────────────

# --- Check 1: verify that the achieved leads match the specified leads ---
# For each DFP filter, compute the actual frequency-zero time-shift and
# compare it against the intended lead. The difference should be zero
# (up to numerical precision).
tau_vec <- NULL

for (i in 1:length(lead_vec)) {
  
  # Frequency-zero time-shift of the i-th DFP filter (see tutorial 2, exercise 3.3.2)
  taub <- sum((0:(L-1)) * b_mat[, i]) / sum(b_mat[, i])
  tau_vec <- c(tau_vec, taub)
  
  # Actual lead of the DFP over the MSE predictor at frequency zero
  lead_dfp_mse <- taub - tauh
  
  # Print the discrepancy — should be numerically zero
  print(lead_dfp_mse - lead_vec[i])
}


# --- Check 2: verify that the DFP preserve trend orientation ---
# The sums of the coefficients are positive (the transfer functions at 
# frequency zero are positive, see Wildi 2026, section 4.1). 
apply(b_mat,2,sum)

# ─────────────────────────────────────────────────────────────────────
# 3.6 Compute Complete Decoupling for Additional Reference
# ─────────────────────────────────────────────────────────────────────
# The completely decoupled (FD) DFP is defined by alpha0 = 0, i.e., the
# DFP filter is orthogonal to gamma0 (zero inner product with the nowcast).
# This represents the theoretical maximum decoupling and serves as an
# extreme-case reference benchmark.

alpha0_cd <- 0  # complete decoupling constraint: <gamma0, b_cd> = 0

# Compute the completely decoupled DFP filter via Proposition 1 (Wildi 2026)
dfp_obj  <- mse_dfp_from_alpha0_func(gamma0, gammah, alpha0_cd)
lambda_cd <- dfp_obj$lambda
b_cd      <- dfp_obj$b

# Normalize to unit length for comparability with the tau-shifted DFP filters
scale <- as.double(1 / sqrt(t(b_cd) %*% b_cd))
b_cd  <- scale * b_cd

# Assemble all predictors into a single filter matrix.
# All filters are normalized to unit length for consistent comparison.
filter_mat <- cbind(
  gamma0      / as.double(sqrt(gamma0      %*% gamma0)),       # Nowcast
  gammah      / as.double(sqrt(gammah      %*% gammah)),       # MSE(h)
  gammahtilde / as.double(sqrt(gammahtilde %*% gammahtilde)),  # MSE(2h)
  b_mat,                                                        # Tau-shifted DFPs
  b_cd                                                          # Fully decoupled DFP
)

colnames(filter_mat) <- c(
  "Nowcast",
  paste("MSE(", h,    ")", sep = ""),
  paste("MSE(", 2*h,  ")", sep = ""),
  paste("DFP ", lead_vec,  sep = ""),
  "DFP FD"
)


# ─────────────────────────────────────────────────────────────────────
# 3.7 Checks: Complete Decoupling DFP
# ─────────────────────────────────────────────────────────────────────

# Verify orthogonality: <b_cd, gamma0> should be (numerically) zero,
# confirming that the completely decoupled filter is orthogonal to gamma0.
t(b_cd) %*% gamma0

# Key difference relative to exercise 1:
# The fully decoupled DFP here preserves trend orientation — the sum of
# coefficients is positive, meaning the filter does not invert the trend.
sum(b_cd)

# Compute the frequency-zero time-shift of the fully decoupled DFP.
# This is well-defined precisely because the filter preserves trend orientation.
tau_cd <- sum((0:(L-1)) * b_cd) / sum(b_cd)
tau_cd

# Lead of the fully decoupled DFP over the MSE(h) predictor at frequency zero.
tau_cd - tauh

# ─────────────────────────────────────────────────────────────────────
# 3.8 Performance Table
# ─────────────────────────────────────────────────────────────────────
# Summarise three key performance metrics for each predictor:
#   tau(0)  — frequency-zero time-shift: positive values indicate a lag
#             (right-shift) when the filter is applied to a linear trend.
#   lambda  — DFP weight on gamma0 (not applicable for MSE).
#   alpha0  — inner product <gamma0, b>: the DFP constraint value
#             (not interpretable as a correlation, except when zero).

mat_perf <- matrix(nrow = 3 + length(lead_vec), ncol = 3)

colnames(mat_perf) <- c("tau(0)", "lambda", "alpha0")
rownames(mat_perf) <- c(
  paste("MSE(", h,      ")", sep = ""),
  paste("MSE(", htilde, ")", sep = ""),
  paste("DFP-shifted by ", lead_vec, sep = ""),
  "DFP fully decoupled"
)

# MSE(h): record frequency-zero time-shift; lambda and alpha0 are not applicable.
mat_perf[1, 1] <- tauh

# MSE(htilde): record frequency-zero time-shift for the long-horizon benchmark.
mat_perf[2, 1] <- tauhtilde

# Tau-shifted DFPs and fully decoupled DFP: populate all three metrics.
# Note: the time-shift is also well-defined for the fully decoupled DFP here
# (in contrast to exercise 1, where the fully decoupled DFP inverted the trend).
mat_perf[3:nrow(mat_perf), 1] <- c(tau_vec, tau_cd)
mat_perf[3:nrow(mat_perf), 2] <- c(lambda_vec, lambda_cd)
mat_perf[3:nrow(mat_perf), 3] <- c(alpha_vec, alpha0_cd)

# Note: NA entries indicate metrics that are not meaningful for the
# corresponding predictor (e.g., lambda and alpha0 for MSE predictors).
mat_perf

# Discussion:
# The general structure of results mirrors exercise 1; key novelties are:
#  -Both MSE(24) and fully-decoupled DFP preserve trend orientation. Their 
#   time-shifts at frequency zero are therefore well defined and interpretable. 


# ─────────────────────────────────────────────────────────────────────
# 3.9 Plot Predictor Filters
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))

colo <- c("black", "green", rainbow(ncol(filter_mat) - 2))

# Collect all filters into a matrix for plotting (unit-normalized)
mplot    <- filter_mat
col_names <- colnames(filter_mat)
colnames(mplot) <- col_names

# Diagnostic: sum of squared coefficients per filter (filter energy; should be ~1)
apply(mplot^2, 2, sum)

# Line type and width: dashed/thick for reference predictors, solid/thin for DFPs
lty_vec <- lwd_vec <- c(2, 2, 2, rep(1, ncol(filter_mat) - 3))

# --- Left panel: filter coefficient profiles ---
plot(mplot[, 1],
     main = "Scaled Predictors", axes = F, type = "l",
     lty  = lty_vec[1][1],
     xlab = "Lags", ylab = "",
     col  = colo[1],
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

# Overlay remaining filters with color-coded labels
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], type = "l", lty = lty_vec[i], lwd = lwd_vec[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
axis(1, at = c(0, (1:(nrow(mplot)/10)) * 10),
     labels = c(0, (1:(nrow(mplot)/10)) * 10))
axis(2)
box()

# --- Right panel: cross-correlation functions (CCF) ---
# Compute the CCF between each predictor and the nowcast (gamma0),
# evaluated at lags surrounding lag 0 and the target horizon h.
max_lag <- 20
mplot   <- NULL
for (i in 1:ncol(filter_mat))
  mplot <- cbind(mplot, compute_ccf_func(filter_mat[, i], gamma0)[L - 1 + 1:max_lag])
colnames(mplot) <- col_names

plot(mplot[, 1],
     main = "CCF", axes = F, type = "l",
     lty  = lty_vec[1], lwd = lwd_vec[1],
     xlab = "", ylab = "",
     col  = colo[1],
     ylim = c(min(mplot), max(mplot)))

# Overlay CCFs for all predictors
for (i in 1:ncol(mplot)) {
  lines(mplot[, i], col = colo[i], lty = lty_vec[i], lwd = lwd_vec[i])
}
abline(v = 1 + h, lty = 2)  # vertical line marking the target horizon h
abline(h = 0)

axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# Discussion:
# The general structure of results mirrors exercise 1; key novelties are:
#
#   - Decoupling is less costly in terms of target correlation:
#       Strong decoupling at lag 0 weighs less heavily on the CCF at the
#       target horizon h = 12 compared to exercise 1. Even the fully
#       decoupled DFP retains a target correlation above 0.5 at h = 12,
#       indicating that the smoother yearly growth target is more amenable
#       to decoupling without a severe loss in forecasting performance.
#
#   - The MA(12) smoothing dampens the AR(2) periodicity:
#       The equally-weighted MA(12) target filter attenuates the business-cycle
#       periodicity of the underlying AR(2) process. As a consequence, MSE(24)
#       is no longer subject to phase-hunting behavior — unlike in exercise 1,
#       where the strong periodicity caused MSE(24) to lock onto the wrong
#       phase. In this smoother setting, the performance loss of MSE(24) at h = 12
#       relative to the optimal DFP designs (for identical decoupling of x_t)
#       is marginal. This stands in contrast to exercises 1.9 and 1.11, where
#       MSE(24) suffered a loss in target tracking at h = 12
#       compared to the DFP with equivalent decoupling.

# ─────────────────────────────────────────────────────────────────────
# 3.10 Compare Forecasts
# ─────────────────────────────────────────────────────────────────────

#----------------------------------------------------------------------
# 3.10.1 Apply Predictors to Data
#----------------------------------------------------------------------

# All predictors are defined in MA form, applied to the Wold innovations
# eps_t. Accordingly, filters are applied to the ARIMA model residuals,
# which serve as empirical proxies for eps_t.
# Note: example 2.4 in tutorial 6 incorrectly applied the MA form to x_t
# directly — applying to residuals, as done here, is the correct approach.
x_filt <- arima.obj$residuals

# Apply each filter sequentially and collect outputs into a matrix.
y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- col_names

#----------------------------------------------------------------------
# 3.10.2 Plot
#----------------------------------------------------------------------

par(mfrow = c(1, 1))

# Full sample: overview of all predictor outputs
ts.plot(y_out_mat,
        main = "Predictor Outputs", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)),
        lwd  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Zoom in on the Dot-com recession episode (approx. obs. 120–170)
ts.plot(y_out_mat[120:170, ],
        main = "Predictor Outputs — Dot-com Recession", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)),
        lwd  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Zoom in on the Global Financial Crisis episode (approx. obs. 200–250)
ts.plot(y_out_mat[200:250, ],
        main = "Predictor Outputs — Financial Crisis", col = colo,
        xlab = "", ylab = "",
        lty  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)),
        lwd  = c(2, 2, 2, rep(1, ncol(filter_mat) - 3)))
abline(h = 0)
for (i in 1:ncol(y_out_mat))
  mtext(colnames(y_out_mat)[i], col = colo[i], line = -i)

# Discussion:
# The general structure of results mirrors exercise 1; key novelties are:
#
#   - Reduced phase reversal risk:
#       The MSE predictor is less prone to phase reversal in this exercise
#       because the equally-weighted MA(12) target filter dampens the
#       influence of the AR(2) periodicity. The smoother target effectively
#       reduces the dominance of the business-cycle frequency, making the
#       MSE predictor more stable across a wider range of forecast horizons.
#
#   - Higher-lead DFPs now anticipate MSE(24):
#       Unlike in exercise 1, the DFP variants with larger imposed leads
#       now visibly lead MSE(24) in the time domain. Tracking the target 
#       does not (less) refrain from leading in this exercise.


# ─────────────────────────────────────────────────────────────────────
# 3.11 Amplitude and Time-Shifts
# ─────────────────────────────────────────────────────────────────────

K      <- 600    # number of frequency grid points for spectral evaluation
plot_T <- FALSE  # suppress internal plotting; custom plot is built below

# Compute amplitude and time-shift functions for each predictor filter
amp_mat <- shift_mat <- NULL
for (i in 1:ncol(filter_mat)) {
  as_obj <- amp_shift_func(K, filter_mat[, i], plot_T)
  amp_mat <- cbind(amp_mat, as_obj$amp)
  shift   <- as_obj$shift
  # Suppress extreme negative shifts for visual clarity
  shift[which(shift < (-3))] <- NA
  shift_mat <- cbind(shift_mat, shift)
}
colnames(amp_mat) <- colnames(shift_mat) <- colnames(filter_mat)

# Plot amplitude and time-shift functions across frequencies [0, pi]
par(mfrow = c(1, 2))

# Line types: dashed for reference predictors, solid for DFP variants
lty_vec <- c(2, 2, rep(1, ncol(filter_mat) - 2))

# --- Left panel: amplitude functions ---
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

# --- Right panel: time-shift functions ---
mplot <- shift_mat

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

# Outcome:
# See the corresponding discussion in exercise 1.12. 
# Main differences:
# -All filters preserve trend orientation.
# -The `faster' DFP have smaller time-shifts than MSE(24) at the relevant 
#   business-cycles frequencies (leading).


# ═════════════════════════════════════════════════════════════════════
# MAIN TAKE AWAYS
# ═════════════════════════════════════════════════════════════════════
#
# Exercises 1 and 2 — Monthly Growth:
#
# Stylized facts:
#   - The differenced log-PAYEMS series is noisy, with pronounced downturns
#     during recession episodes.
#   - The AR(2) component of the model is periodic, which means the MSE
#     predictor is not stuck at present: the periodicity allows it to look
#     forward by exploiting the cyclical phase structure of the process.
#   - In this context, the look-ahead advantage of the DFP over the MSE
#     predictor is less of a priority than in the aperiodic case — the MSE
#     predictor already anticipates to some degree. Nevertheless, the
#     comparison remains instructive, as it reveals potential gains from
#     the DFP approach even in a setting where the MSE predictor is
#     naturally forward-looking.
#
# Phase-hunting vs. controlled tracking:
#   - MSE(24) and the fully decoupled DFP do not preserve trend orientation,
#     which can cause severe distortions when the series has a non-zero mean
#     (the sign of the mean is effectively inverted). Full decoupling is
#     therefore considered too extreme a design choice in this setting.
#   - An important advantage of the time-shift DFP is that the tau-based
#     constraint guarantees preservation of linear trend orientation 
#     (or mean sign) by design — a property not shared by MSE(24) or the 
#     fully decoupled DFP.
#   - MSE(24) is prone to phase-hunting: by targeting too wide a horizon, it
#     locks onto a phase that may be difficult to interpret economically, 
#     assuming the data (phenomenon) is not strongly periodic. In
#     contrast, the DFP maintains maximal tracking ability at the intended
#     forecast horizon h = 12 while simultaneously looking ahead — offering
#     a principled and interpretable alternative in the context of mutually 
#     conflicting design goals.
#
# Complexity:
#   - The shape of the optimal filter coefficients (predictor weights) reflects
#     the complexity arising from mutually conflicting design goals: timeliness,
#     decoupling, and target tracking pull the filter in different directions,
#     resulting in coefficient profiles that deviate markedly from classic
#     filter designs. However, simulation results confirm pertinence of the 
#     approach with respect to the specified design goals.
#
# Time-shift tau:
#
# 1. Effect on forecasts:
#   - Increasing the lead (tau) of the time-shift DFP produces a progressive
#     left-shift of the DFP predictor in the time domain.
#       - Even large leads imposed at frequency zero spill over to adjacent
#         frequencies only in highly attenuated form, steeply diminishing as
#         omega increases — as confirmed by the frequency-domain time-shift plots.
#       - This attenuation is a direct consequence of the DFP's objective to
#         maximize tracking at h = 12: propagating a large lead into the
#         business-cycle band would conflict with this goal by shifting the
#         predictor away from the intended forecast horizon.
#       - The observed anticipation at recession turning points is consistent
#         with the attenuated time-shift values in the business-cycle band.
#       - Requiring more aggressive look-ahead behavior would risk phase
#         reversal, conflicting with the target horizon and undermining
#         interpretability.
#   - The DFP predictor remains anchored at the intended, practically relevant
#     one-year forecast horizon: look-ahead behavior is achieved without
#     resorting to obvious phase-hunting.
#
# 2. Effect on amplitude and time-shift functions:
#   - As the lead at frequency zero increases, the filter progressively morphs
#     from a lowpass design (MSE) toward a bandpass design (fully decoupled
#     DFP), in line with the ATS trilemma: timeliness, accuracy, and smoothness
#     cannot all be improved simultaneously.
#   - While bandpass behavior might be acceptable — or even desirable — in a
#     pure phase-hunting design, the DFP actively resists this morphing: it
#     attempts to preserve lowpass behavior as much as possible while still
#     satisfying the imposed zero-frequency time-shift constraint. The result
#     is a controlled compromise between anticipation and signal fidelity.
#   - The time-shift functions start at the imposed lead value at frequency
#     zero and then rapidly converge toward values consistent with one-year-
#     ahead forecasting at business-cycle frequencies. This rapid convergence
#     reflects the DFP's targeted and controlled propagation of the imposed
#     lead across the frequency domain — confining the large lead to frequency
#     zero while preserving the intended forecast horizon h = 12 at
#     business-cycle frequencies.

#
# Exercise 3 — Yearly Growth (Smooth Target):
#
#   - Exercise 3 generalizes exercises 1 and 2 by targeting a generic signal
#     derived from x_t — here an equally-weighted MA(12) corresponding to
#     yearly growth — rather than the raw differenced log series.
#   - The smoother target reduces noise and renders the effect of the DFP
#     more pronounced and interpretable in both the time and frequency domains.
#     It also simplifies the forecast problem by reducing the tension between
#     the competing design goals of timeliness, decoupling, and target tracking,
#     making the overall design less conflictual.
#   - The shape of the filter coefficients is more regular and closer to
#     classic lowpass filter designs, directly reflecting the reduced conflict
#     between design objectives when the target is smooth: the filter no longer
#     needs to simultaneously accommodate aggressive anticipation and noisy
#     short-term dynamics.
#   - In contrast to exercises 1 and 2, the fully decoupled DFP preserves
#     trend orientation in this smoother setting. The fully decoupled DFP can 
#     therefore serve as a consistent, if extreme, predictor design in this 
#     context.
#
# Coming next:
#   - Tutorial 9 analyzes non-standard configurations and other challenging
#     DFP settings, including the aperiodic ARMA(1,1) model where the
#     periodic structure exploited here is absent.
#   - Tutorial 10 introduces the Peak Correlation Shifting (PCS) approach.
