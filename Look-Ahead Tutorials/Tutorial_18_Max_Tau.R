# This is still Work in progress!!!



# ══════════════════════════════════════════════════════════════════════════
# NEW MATERIAL: September 2026
# Introduction of the Max-Tau Predictor
# ══════════════════════════════════════════════════════════════════════════
#
# ══════════════════════════════════════════════════════════════════════════
# NEW MATERIAL: September 2026
# Introduction of the Max-Tau Predictor
# ══════════════════════════════════════════════════════════════════════════
#
# CONCEPT
# -------
# Max-Tau differs from DFP in a fundamental way:
#
#   - DFP decouples from the *nowcast* and defines a CONDITIONAL efficient
#     frontier between Target Correlation (TC) and lead, i.e. conditional on
#     that decoupling constraint.
#
#   - Max-Tau decouples from *new vectors which maximise lead
#     unconditionally*. This means: no other linear predictor of the same
#     filter length can outperform Max-Tau in BOTH target correlation (TC)
#     and lead at the reference frequency. Max-Tau therefore defines an
#     UNCONDITIONAL efficient TC/lead frontier.
#
#
# LOOK-AHEAD AND SIGN INVERSION
# ------------------------------
# Max-Tau always outperforms DFP in terms of TC/lead ON THE FRONTIER.
# However, MSE-DFP can generate look-ahead behaviour that is out of reach
# for both Max-Tau and time-shift DFP — at a cost:
#
#   - Such extreme look-ahead designs require increasing SIGN REVERSION,
#     starting with trend and mean inversion. Formally, the transfer
#     function becomes negative at the reference frequency omega_0 = 0,
#     i.e. Gamma(0) < 0 (see exercise 1).
#   - TC is also substantially reduced in such extreme DFP designs
#     (see examples below).
#
# In contrast, time-shift DFP and Max-Tau CANNOT invert signs: the
# orientation of the trend and the sign of the mean are always preserved.
#
#
# NOVELTY: GENERALISING THE REFERENCE FREQUENCY
# -----------------------------------------------
# Until now, DFP has addressed the reference frequency omega_0 = 0 (trend)
# only. We extend Max-Tau here to address omega_0 > 0 as well — e.g.
# Max-Tau can maximise lead at BUSINESS-CYCLE frequencies, not just at
# the trend frequency.
#
#
# CAVEAT: OVERFITTING
# --------------------
# Because Max-Tau maximises lead at a single reference frequency, it
# becomes prone to overfitting as filter length L increases (DFP does not
# share this problem to the same extent). Two remedies are illustrated in
# this and subsequent tutorials:
#   (1) controlling curvature of the filter, or
#   (2) extending the lead criterion to multiple frequencies.
#
# Reference: see the accompanying paper in the "Papers" folder.
#
# ══════════════════════════════════════════════════════════════════════════
#
# Reference: see the accompanying paper in the "Papers" folder.
#
# ══════════════════════════════════════════════════════════════════════════


# ── 1. INITIALISATION ───────────────────────────────────────────────────────

rm(list = ls())

# --- Load core algorithms (packaged as functions; treated here as black boxes) ---

# DFP optimisation routines: provides DFP_compute_lambda_alpha0_func() and
# the mse-based DFP solver mse_dfp_from_alpha0_func() used below for comparison.
source(file.path(getwd(), "R", "DFP.r"))

# Max-Tau optimisation routines (new for this tutorial).
source(file.path(getwd(), "R", "Max_Tau.r"))

# HP filter utilities (construction of HP trend/gap filters, MSE weights, etc.)
source(file.path(getwd(), "R utility functions", "HP_JBCY_functions.r"))

# General DFP/PCS utility functions (amplitude, time-shift, and CCF helpers,
# e.g. compute_acf_at_lags_zero_delta_func()).
source(file.path(getwd(), "R utility functions", "DFP_PCS_utility_functions.r"))

# --- Required packages ---
library(xts)
library(mFilter)

# 'alfred' allows direct FRED data retrieval without an API key.
# Install once if not already available.
if (!requireNamespace("alfred", quietly = TRUE)) install.packages("alfred")
library(alfred)


# ── 2. DATA: US REAL GDP ─────────────────────────────────────────────────────

# Toggle reload_data to TRUE to fetch fresh data from FRED (overwrites local
# copy); FALSE loads the previously saved series (recommended for reproducible
# tutorial runs).
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDP"))
} else {
  load(file.path(getwd(), "Data", "GDP"))
}

# Quick sanity checks
head(GDPC1)
tail(GDPC1)
is.xts(GDPC1)

# --- Restrict sample and switch to plain numeric vector ---
#
# NOTE: xts objects carry implicit index-handling conventions that can
# interfere with downstream computations (e.g. filter() applied to an xts
# object may silently reverse the time axis). We therefore keep a plain
# numeric vector (y) for all computations, and an xts copy (y_xts) purely
# for plotting.

start_year <- 1992
end_year   <- 2024
sample_range <- paste(start_year, end_year, sep = "/")

y_xts <- log(GDPC1[sample_range])          # for plotting only
y     <- as.double(y_xts)                  # for computation
len   <- length(y)


# ── 3. EXPLORATORY PLOTS ─────────────────────────────────────────────────────

par(mfrow = c(2, 2))
plot(GDPC1,                  main = "US Real GDP (levels)")
plot(y_xts,                  main = "Log GDP")
plot(diff(y_xts),            main = "Log-differences of GDP")
acf(na.exclude(diff(y_xts)), main = "ACF of log-differences")

# The data, retrieved from FRED (https://fred.stlouisfed.org/), are displayed in the above 
# Figure, where log-differences (bottom left) cover the last three recession episodes from 
# 1992 to 2024. HP is applied to log-differences to track trend-growth, whose sign indicates 
# below/above-average growth (Wildi, 2024). While this application deviates from business-cycle 
# orthodoxy—where the HP bandpass is typically applied to the original non-stationary GDP—
# we select this framework because it emphasizes lead/lag issues more clearly and because 
# tracking growth through the lowpass addresses spurious cycles inherent to the classic 
# approach (Wildi, 2024) and earlier Tutorials.

# ── 4. TARGET SPECIFICATION: HP TREND OF DIFF-LOG GDP ───────────────────────

# Forecast horizon (in periods): we select a two-quarters ahead horizon as in Wildi (2026)
h <- 2

# HP smoothing parameter (standard quarterly setting)
lambda_hp <- 1600

# Filter length: L must be odd so the symmetric (two-sided) filter is
# centred at position (L-1)/2 + 1.
L <- 51

# --- Two-sided HP benchmark filter (double length) ---
# We compute the two-sided filter at length 2*(L-1)+1 so that we can later
# compare a one-sided (real-time) filter of length L against the right tail
# of the corresponding two-sided (symmetric) filter.
HP_obj <- HP_target_mse_modified_gap(2 * (L - 1) + 1, lambda_hp)

HP_two          <- HP_obj$target
hp_gap          <- HP_obj$hp_gap[1:L]
modified_hp_gap <- HP_obj$modified_hp_gap[1:L]

# Concurrent (one-sided) HP trend filter, assuming an I(2) process
hp_trend_long <- HP_obj$hp_trend
hp_trend      <- hp_trend_long[1:L]

# MSE-optimal one-sided filter for the bi-infinite HP trend, assuming
# white-noise innovations (white noise is ensured by ACF in above plot: bottom right panel)
hp_mse_long <- HP_obj$hp_mse
hp_mse      <- hp_mse_long[1:L]


##############################################################################
# EXERCISE 1: MSE-DFP
##############################################################################

#-------------------------------------------------------------------------------
# ── 1.1. DFP SETTINGS: TARGET AND CONSTRAINT VECTORS ──────────────────────────
#-------------------------------------------------------------------------------

# Forecast horizon (delta) for the target
delta <- h

# Start lag for the cross-correlation function (CCF); max_lag = 0 restricts
# attention to the right tail only (non-negative lags/leads).
max_lag <- 0

# Target: MSE-optimal predictor of the HP trend at horizon h = delta
gammah <- gamma_target <- hp_trend_long[h + 1:L]

# Constraint: nowcast (horizon 0) predictor of the HP trend
gamma0 <- gamma_constraint <- hp_trend_long[1:L]

# --- Sanity checks on gamma_target ---
ts.plot(gamma_target, main = "Target vector: HP trend weights at horizon h")

sum(hp_trend_long)                             # should sum to 1
sqrt(t(hp_trend_long) %*% hp_trend_long)       # filter norm (std-dev proxy)


#-------------------------------------------------------------------------------
# ── 1.2. MSE-DFP BENCHMARK: FAMILY OF SOLUTIONS INDEXED BY alpha0 ─────────────────
#-------------------------------------------------------------------------------
#
# Before introducing Max-Tau, we replicate the classic MSE-DFP benchmark
# across a grid of alpha0 values. alpha0 controls the degree of decoupling
# from the nowcast constraint: smaller alpha0 permits more decoupling
# (and hence more lead).
#
# NOTE ON UPCOMING EXERCISES:
#   - Exercise 2: time-shift DFP, where alpha0 = alpha(tau) is derived
#     directly from the desired lead tau at the reference frequency
#     omega_0 = 0.
#   - Exercise 3: Max-Tau.
#
# CAUTION:
#   MSE-DFP can generate look-ahead behaviour that is out of reach for
#   both time-shift DFP and Max-Tau — but at the cost of SIGN INVERSION.
#   In this example, that will occur when alpha0 is close to 0 (i.e.
#   near-complete decoupling from the nowcast constraint).

alpha0_vec <- c(0.1, 0.05, 0.02, 0.017, 0.005, 0)

# Containers for results across the alpha0 grid
lambda               <- NULL   # Lagrange multipliers from the DFP solve
b0_mat               <- NULL   # DFP filter coefficients (one column per alpha0)
cor_vec_mse_la_mat   <- NULL   # CCF of each DFP filter vs. the MSE target

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # --- Core DFP solve: quadratic-in-lambda problem, then unit-length filter ---
  b0_obj <- mse_dfp_from_alpha0_func(gamma_constraint, gamma_target, alpha0)
  b      <- b0_obj$b
  
  b0_mat <- cbind(b0_mat, b)
  lambda <- c(lambda, b0_obj$lambda)
  
  # --- Cross-correlation of the resulting DFP predictor with the MSE target ---
  cor_vec_mse_la_mat <- cbind(
    cor_vec_mse_la_mat,
    compute_acf_at_lags_zero_delta_func(max_lag, h, b0_mat[, i], hp_trend)$cor_vec
  )
}

colnames(b0_mat)             <- alpha0_vec
colnames(cor_vec_mse_la_mat) <- alpha0_vec

# CCF of the MSE (HP trend) target itself, for reference/comparison
cor_vec_t_hp_trend <- compute_acf_at_lags_zero_delta_func(
  max_lag, h, gammah, hp_trend
)$cor_vec



#-------------------------------------------------------------------------------
# ── 1.3 SUMMARY STATISTICS ACROSS THE alpha0 GRID ────────────────────────────
#-------------------------------------------------------------------------------

b_alpha0                  <- b0_mat
cor_vec_mse_la_mat_alpha0 <- cor_vec_mse_la_mat

# Gamma(0): sum of filter coefficients. Gamma(0)<0 implies trend and mean inversion.
Gamma0_alpha0 <- apply(b_alpha0, 2, sum)
# The last two filters are subject to inversion:
Gamma0_alpha0

# Time-shift (lead/lag) at frequency zero, for each alpha0
tau_alpha0 <- -as.vector(t(b_alpha0) %*% (0:(L - 1)) / apply(b_alpha0, 2, sum))

# Time-shift is not properly defined when Gamma(0) < 0 (it corresponds to the shift 
# of the sign inverted predictor). Therefore we insert NA.
tau_alpha0[which(Gamma0_alpha0 < 0)] <- NA

# --- Corresponding statistics for the plain MSE benchmark filter ---
mse_b <- c(
  t(gammah) %*% gamma0 / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. nowcast
  t(gammah) %*% gammah / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. itself
  sum(gammah),                                                              # Gamma(0)
  -as.double(t(gammah) %*% (0:(L - 1)) / sum(gammah))                       # tau at freq 0
)

# --- Combine into a single comparison table: MSE benchmark vs. DFP(alpha0) grid ---
table_alpha0 <- cbind(
  mse_b,
  rbind(
    cor_vec_mse_la_mat_alpha0[1, ],      # TC at lag 0 (nowcast)
    cor_vec_mse_la_mat_alpha0[h + 1, ],  # TC at lag h (target horizon)
    Gamma0_alpha0,                       # Gamma(0)
    tau_alpha0                           # time-shift (lead) at frequency 0
  )
)
rownames(table_alpha0)<-c("Correlation with nowcast: CCF(0)","TC: CCF(h)","Gamma(0)","Shift at omega0=0 (lead when positive)")

# DISCUSSION OF RESULTS MSE-DFP: THE TC/LEAD DILEMMA AND THE (CONDITIONAL) EFFICIENT FRONTIER
#
# Ideally, a predictor would simultaneously maximise:
#   - Target Correlation (TC) (first row in below table), measured as the CCF at 
#     horizon h, CCF(h), and
#   - Lead (tau) (last row in column).
#
# Both cannot be achieved at once — this is the DILEMMA. The best attainable
# outcome is a predictor lying ON THE EFFICIENT FRONTIER between TC and lead.
# MSE-DFP sits on the efficient frontier of all predictors SUBJECT TO
# nowcast decoupling. Here is how this manifests in the table (table_alpha0):
#
#   1. STRONGER DECOUPLING (smaller entries in row 2, CCF(0)) spills over
#      into smaller TC (row 1, CCF(h)). MSE-DFP minimises this undesirable
#      loss of TC: no other linear predictor can achieve both a smaller
#      CCF(0) AND a larger CCF(h) simultaneously — this is precisely what
#      makes it "efficient".
#
#   2. Stronger decoupling can be linked BIJECTIVELY (via a strictly
#      monotonic function) to the lead (tau) at the reference frequency
#      omega_0 = 0 — see Corollary 2 in Wildi (2026). This link requires
#      Gamma(0) > 0 (row 3 in the table).
#
#   3. Therefore, BY COROLLARY 2, the dilemma between tau (row 4, lead) and
#      TC (row 1) holds whenever Gamma(0) > 0: pushing lead higher
#      necessarily costs some TC, and vice versa.
#
#   4. HOWEVER, the link between CCF(0) and tau does NOT guarantee that tau
#      is maximised across ALL linear predictors of the same length — it is
#      not. The efficient frontier described above (points 1-3) is only
#      efficient CONDITIONAL ON nowcast decoupling.
#
#      Max-Tau resolves this geometric limitation: it maximises tau
#      UNCONDITIONALLY, among ALL linear predictors of the same length —
#      not merely among those that decouple from the nowcast.
#
table_alpha0


# ── NOTE ON SIGN INVERSION ───────────────────────────────────────────────────
#
# The last two columns of `table_alpha0` (alpha0 close to zero) generate
# Gamma(0) < 0.
#
# WHAT THIS MEANS:
#   The sign of a non-vanishing mean will be INVERTED by the filter. WHY?
#
#   Let x_t = mu > 0 be a constant series. Apply the filter
#   b = (b_0, ..., b_{L-1})' to x_t:
#
#       (mu, ..., mu) %*% b = sum_{k=0}^{L-1} mu * b_k = mu * Gamma(0)
#
#   If Gamma(0) < 0, the sign of x_t = mu is REVERSED after filtering.
#
#   This does NOT mean the predictor is meaningless: as long as TC > 0
#   (which is the case here — see row 1 of `table_alpha0`), the predictor
#   still correlates POSITIVELY with the target, i.e. it remains
#   effectively informative about the target's dynamics.
#
# WHAT IT DOES MEAN IN PRACTICE:
#   Sign inversion of the mean signifies that the data should be CENTERED
#   before applying the predictor; otherwise the resulting level will carry
#   the wrong sign. Once the forecast is computed on centered data, the
#   correct mean can simply be added back to the centered forecast.
#
# WHY THIS IS NOT A SERIOUS LIMITATION:
#   Static mean or scale adjustments are trivial operations. They do NOT
#   affect the dynamic look-ahead capabilities of the predictor — i.e. the
#   TC/lead trade-off discussed above is entirely unaffected by whether the
#   data was centered before or after filtering. Both TC and Tau are insensitive 
#   to shift and scaling.

#
# This sets up the comparison in the next section: Max-Tau will match or
# dominate every point on the MSE-DFP frontier above in terms of the
# TC/lead trade-off.





########################################################################################
# EXERCISE 2: TIME-SHIFT DFP
########################################################################################

# As seen in `table_alpha0`, MSE-DFP can generate SIGN INVERSION
# (Gamma(0) < 0) when alpha0 is pushed close to zero.
#
# Time-shift DFP AVOIDS sign inversion by construction: instead of
# specifying the decoupling parameter alpha0 directly, we specify the
# desired LEAD tau at the reference frequency omega_0 = 0, and infer the
# corresponding alpha0 = alpha(tau) via the bijective (strictly monotonic)
# relationship established in Proposition 3 and Corollary 2, Wildi (2026).
#
# Since alpha(tau) is derived from this relationship, it is guaranteed to
# stay within the range that preserves Gamma(0) > 0 — hence no sign
# inversion.
#
# Apart from this reparameterisation (tau -> alpha0, rather than choosing
# alpha0 directly), Exercise 2 is otherwise IDENTICAL to Exercise 1.


#-------------------------------------------------------------------------------
# ── 2.1 COMPUTE TIME-SHIFT DFP ────────────────────────────
#-------------------------------------------------------------------------------


# Grid of desired leads (tau) at the reference frequency omega_0 = 0: the last entry 
# is `large`, representing an infinite lead. 
# Note: Tau>0 is the lead of DFP over the MSE benchmark: the constraint addresses the 
# DIFFERENCE between the shift of DFP and MSE.
tau_vec<-c(1,2,6,100000)

b0_mat<-cor_vec_mse_la_mat<-NULL
for (i in 1:length(tau_vec))#i<-1
{
# Note: the sign convention in mse_dfp_from_tau_func is that leads correspond to negative numbers.   
  lead<--tau_vec[i]
  # Call the dedicated function to compute the DFP filter for a specified lead
  # (see dfp_from_tau_func for the derivation based on Proposition 3, Wildi 2026)
  dfp_obj <- mse_dfp_from_tau_func(gamma_constraint, gamma_target, lead)
  
  # Extract the components returned by the function
  tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
  tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
  lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
  b       <- dfp_obj$b        # raw DFP filter coefficients
  # Normalise b to unit length (corresponds to unitary DFP)
  #  b_unit <- b / as.double(sqrt(b %*% b))
  b0_mat  <-cbind(b0_mat,b)
  # Compute target correlation with respect to MSE gamma_target
  cor_vec_mse_la_mat<-cbind(cor_vec_mse_la_mat,compute_acf_at_lags_zero_delta_func(max_lag,h,b,hp_trend)$cor_vec)
  
}
colnames(b0_mat)<-colnames(cor_vec_mse_la_mat)<-paste("Shift ",-tau_vec,sep="")


b_tau<-b0_mat
cor_vec_mse_la_mat_tau<-cor_vec_mse_la_mat

#-------------------------------------------------------------------------------
# ── 2.2 VARIOUS CHECKS ────────────────────────────
#-------------------------------------------------------------------------------

# Compute Gamma(0)
Gamma0_tau<-apply(b_tau,2,sum)
# Check: all positive (as Tau\to\infty, Gamma(0)\to 0, see Wildi 2026)
Gamma0_tau

# Compute time-shifts at frequency zero
tau_tau<--as.vector(t(b_tau)%*%(0:(L-1))/apply(b_tau,2,sum))
# Check: lead over MSE matches imposed constraint
# a. Compute shift of MSE
tauh<--as.double(gammah%*%(0:(L-1))/sum(gammah))
# b. Compute lead of DFP over MSE
tau<-(tau_tau-tauh)
# c. Lead matches constraint: difference vanishes
tau-tau_vec

#-------------------------------------------------------------------------------
# ── 2.3 SUMMARY STATISTICS ────────────────────────────
#-------------------------------------------------------------------------------

# Compute Performances
# Compute decoupling parameter, i.e., CCF(0) (up to scaling)
alpha_tau<-as.vector(t(b_tau)%*%gamma0)
# Compute corresponding performances for MSE benchmark.
mse_b<-c(t(gammah)%*%gamma0/sqrt(t(gammah)%*%gammah*t(gamma0)%*%gamma0),t(gammah) %*% gammah / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0),-as.double(t(gammah)%*%(0:(L-1))/sum(gammah)),t(gammah)%*%gamma0)

table_tau<-cbind(mse_b,rbind(cor_vec_mse_la_mat_tau[1,],cor_vec_mse_la_mat_tau[h+1,],tau_tau,alpha_tau))

rownames(table_tau)<-c("Correlation with nowcast: CCF(0)","TC: CCF(h)","Shift at omega0=0 (lead when positive)","alpha0")

# Similar to table_alpha0 in Exercise 1 though we now remove the Gamma(0)-row (always positive) 
# and add an alpha0 row: the decoupling parameter alpha0=alpha0(Tau) is now derived from lead Tau over MSE predictor 
table_tau


#-------------------------------------------------------------------------------
# ── 2.4 PLOTS ────────────────────────────
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 2.4.1 Plot 1: Filter coefficients and CCF
#-------------------------------------------------------------------------------

par(mfrow=c(2,2))

#-------
# I) MSE-DFP based on alpha0
# I.1) Predictor weights

filt_mat_alpha0<-cbind(gamma_target,b_alpha0)
colnames(filt_mat_alpha0)<-c("MSE",paste("\u03B1","=",alpha0_vec,sep=""))
colo<-c("green",rainbow(ncol(b_alpha0)))

# Scale filter coefficients:
mplot<-scale(filt_mat_alpha0,center=F,scale=T)

layout<-plot(mplot[,1],main="Predictors: original MSE-DFP",axes=F,type="l",xlab="Lag",ylab="",col=colo[1],lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(paste(colnames(mplot)[i]),line=-i,col=colo[i])  
}
abline(h=0)
axis(1,at=c(1,1:(nrow(mplot)/10)*10),labels=c(1,1:(nrow(mplot)/10)*10)-c(1,rep(0,5)))
axis(2)
box()


# I.2) CCF
mplot<-cbind(cor_vec_t_hp_trend,cor_vec_mse_la_mat_alpha0)[1:20,]
colnames(mplot)<-c("MSE",paste("MSE-DFP: ","\u03B1","=",alpha0_vec,sep=""))

layout<-plot(mplot[,1],main="CCFs",axes=F,type="l",xlab="Lag",ylab="",col=colo[1],ylim=c(min(mplot),max(mplot)),lty=2,lwd=3)
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
  #  mtext(bquote(lambda==.(lambda_vec[i-1])),col=colo[i],line=-i)
}
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+delta,lty=2)
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-max_lag-1+1:(nrow(mplot)))
axis(2)
box()

# II) Time-shift DFP 
# II.1) Predictor weights

filt_mat_tau<-cbind(gammah,b_tau)
colo<-c("green",rainbow(ncol(b_tau)))
tau_vec[length(tau_vec)]<--Inf

colnames(filt_mat_tau)<-c("MSE",paste("\u03C4","=",tau_vec,sep=""))

# Scale filter coefficients:
mplot<-scale(filt_mat_tau,center=F,scale=T)

layout<-plot(mplot[,1],main="Predictors: time-shift DFP",axes=F,type="l",xlab="Lag",ylab="",col=colo[1],lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],line=-i,col=colo[i])  
}
abline(h=0)
axis(1,at=c(1,1:(nrow(mplot)/10)*10),labels=c(1,1:(nrow(mplot)/10)*10)-c(1,rep(0,5)))
axis(2)
box()

# II.2) Time-shift
K<-600
plot_T<-F
amp<-shift<-NULL
for (i in 1:ncol(filt_mat_tau))
{
  trf_obj<-amp_shift_func(K,filt_mat_tau[,i],plot_T)
  shift<-cbind(shift,-trf_obj$shift)
  amp<-cbind(amp,trf_obj$amp)
}
colnames(amp)<-colnames(shift)<-colnames(filt_mat_tau)
mplot<-shift[1:(K/10),]

plot(mplot[,1],ylim=c(min(mplot),9),axes=F,col=colo[1],type="l",xlab="Frequency",ylab="",main="Time-Shift Lower Frequencies",lty=2,lwd=2)
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=c(1,K/20,K/10),labels=c("0","pi/20","pi/10"))
axis(2)
box()




#-------------------------------------------------------------------------------
# Plot 2.4.2: Predictors at Crises
#-------------------------------------------------------------------------------


diff_log_GDP<-diff(log(GDPC1[paste(as.integer(start_year-L/4),"/",end_year,sep="")]))
len<-length(diff_log_GDP)

y_mat_tau<-NULL
for (i in 1:ncol(filt_mat_tau))
  y_mat_tau<-cbind(y_mat_tau,filter(diff_log_GDP,filt_mat_tau[,i],side=1))
colnames(y_mat_tau)<-colnames(filt_mat_tau)
y_mat_alpha0<-NULL
for (i in 1:ncol(filt_mat_alpha0))
  y_mat_alpha0<-cbind(y_mat_alpha0,filter(diff_log_GDP,filt_mat_alpha0[,i],side=1))
colnames(y_mat_alpha0)<-colnames(filt_mat_alpha0)
y_mat_tau_max<-NULL




par(mfrow=c(2,2))


# COVID great lockdown
if (F)
{
  anf<-158
  enf<-172
  mplot<-scale(y_mat[anf:enf,])
  
  plot(mplot[,1],col=colo[1],main="Great lockdown", axes=F,type="l",xlab="",ylab="",lwd=1,ylim=c(min(mplot),max(mplot)))
  #mtext(colnames(mplot)[1],col=colo[1],line=-1)
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    #  mtext(bquote(beta==.(beta_vec[i-1])),col=colo[i],line=-i)
  }
  
  abline(h=0)
  label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat[,3]))+1):length(GDPC1)])
  axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
  axis(2)
  box()
}

#------------
# I). Designs based MSE-DFP
colo<-c("green",rainbow(ncol(b_alpha0)))

# I.1) Dotcom Recession
anf<-87
enf<-105
mplot<-scale(y_mat_alpha0)[anf:enf,]
colnames(mplot)<-colnames(y_mat_alpha0)


plot(mplot[,1],col=colo[1],main="Dotcom: based on decoupling", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_alpha0[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

# I.2) Great Financial Crisis
anf<-108
enf<-134

mplot<-scale(y_mat_alpha0)[anf:enf,]

plot(mplot[,1],col=colo[1],main="Great recession", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(bquote(beta==.(beta_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_alpha0[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()


# II) Time-shift DFP

# II,1) Dotcom

colo<-c("green",rainbow(ncol(b_tau)))

anf<-87
enf<-105
mplot<-scale(y_mat_tau)[anf:enf,]
colnames(mplot)<-colnames(y_mat_tau)


plot(mplot[,1],col=colo[1],main=paste("Dotcom: based on time-shift"), axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_tau[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

# Great Fincial Crisis
anf<-108
enf<-134

mplot<-scale(y_mat_tau)[anf:enf,]

plot(mplot[,1],col=colo[1],main="Great recession", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(bquote(beta==.(beta_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_tau[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

# Outcome (copy-paste from paper)
# While statistical evidence for classic GDP forecasts at horizons $h>2$ may remain 
# weak (Heinisch et al., 2026), the authors argue that forecasts may remain significant 
# even beyond one year after lowpass filtering. Our example aligns with these findings: 
# stronger look-ahead advances the predictor such that zero-crossings—transitions 
# between growth and contraction—and recession dips can be anticipated relative to the 
# MSE benchmark. The time-shift DFP enables interpretable fine-tuning by linking 
# decoupling to lead while preserving signs and trend orientation, avoiding distortions 
# during prolonged up- or down-cycle phases and remaining anchored at the target horizon. 
# While stronger MSE-DFP decoupling can further improve look-ahead, fully decoupled designs 
# (CCF$(0)=0$) are extreme and serve mainly for illustration.



########################################################################################
# EXERCISE 3 : MAX-TAU
########################################################################################

# Design: replicate TC of time-shift DFP and compare shifts/leads of Max-Tau vs. time-shift DFP.
# Max-Tau outperforms the latter at the reference frequency.
# We use two reference frequencies: omega0=0 (trend) and omega0=pi/20 (10 year cycle).


#-------------------------------------------------------------------------------
# 3.1 COMPUTE TC TO MATCH TIME-SHIFT DFP
#-------------------------------------------------------------------------------

# Compute TC of time-shift DFP: these are used as constraints for Max-Tau.
# For identical TC, Max-Tau must generate a larger lead at the reference frequency.
# Novelty: in addition to the trend frequency omega0=0, we also provide results for the
# business-cycle frequency omega0=pi/20, corresponding to a periodicity of
# 2*pi/omega0=40 quarters, i.e. ten years.

# Note: the DFP above relies on TC normalized with 1/|gamma0| (1/|hp_trend|). However,
# Max-Tau normalizes with 1/|gammah| (1/gamma_target) for alphah in its constraint. So we have to
# recompute the TC with the new re-scaling to obtain the correct Max-Tau matching the DFP.
target_correlation_vec<-NULL
# Scaling is required because alphah has the meaning of a correlation
for (i in 1:ncol(b_tau))
{
  if (T)
  {
    # Correlation with MSE gammah instead of gamma0 (correct)
    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*gamma_target%*%gamma_target))
    #    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*hp_trend%*%hp_trend))
  } else
  {
    # Correlation with nowcast (incorrect)
    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*hp_trend%*%hp_trend))
  }
}

#-------------------------------------------------------------------------------
# 3.2 COMPUTE MAX-TAU: OPTIMIZED FOR omega0=pi/20 (10 YEAR CYCLE)
#-------------------------------------------------------------------------------
# 2. Compute Max-Tau
gamma_h<-gamma_target
# Ten year periodicity
n_freq<-20
omega0<-pi/n_freq
# Impose maximal lead at omega0 with respect to MSE target gamma_h (phase_excess<-T) or with respect to identity (phase_excess<-F), see Corollaries 3 and 4 in Wildi (2026).
# If gamma_h is lagging at omega0, then phase_excess<-F will generate a larger lead when feasible.
# Note that the maximal phase lead is restricted to pi/2 (n_freq/4 in time units) with respect to gamma_h or identity, ensuring strict positivity (like the original time-shift DFP).
# Effect of phase_excess:
# -Maximizing the lead relative to identity or gammah is determined by a slightly different decoupling vector, see Corollary 4.
# -Only difference: the upper boundary of pi/2 is taken with respect to identity or gammah.
# -As long as both leads (with respect to identity or gammah) are below pi/2, the solution is exactly the same.
phase_excess<-F
# Notes:
# When feasible, Max-Tau generates the maximal lead, namely n_freq/2.
# For given L, the imposed TC might restrict Tau being smaller than n_freq/2.
# However, for increasing L the filter exploits the additional degrees of freedom so that Tau will increase.
# Ultimately, for arbitrarily large L, the upper limit Tau=n_freq/2 is achieved for any TC (overfitting).
# Overfitting will be addressed below (Exercise 4).

b_tau_max<-max_tau_vec<-max_tau_excess_vec<-NULL
for (i in 1:length(target_correlation_vec))#i<-1
{
  target_correlation<-target_correlation_vec[i]

  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,omega0,phase_excess)
  
  # Append predictor weights
  b_tau_max<-cbind(b_tau_max,max_tau_obj$b_opt)
  # Append shifts (lead/lag) referenced against identity
  max_tau_vec<-c(max_tau_vec,max_tau_obj$tau_max)
}
# Take minus sign to match sign convention in paper (in code it is assumed that transferfunction is based on exp(+i*omega) whereas in paper we assume exp(-1.i*omega)).
# A lead means a positive number (of the sign-inverted tau):
max_tau_vec
# Compare Tau of Max-Tau with Tau of MSE.
# Note: when omega0>0 the formula for Tau is: Tau=Phi(omega0)/omega0
tauh10<-(Arg(gamma_h%*%exp(-1.i*omega0*(0:(L-1))))/omega0)
tauh10
# Max-Tau exceeds tauh10 (efficient frontier)

# Note: if tauh<0 then the MSE has a lag (compared to the identity). Therefore, setting phase_excess<-F will generate a larger lead, limited only by pi/2 with respect to identity (instead of pi/2 with respect to the lagging MSE).

b_tau_max_omega<-b_tau_max
filt_mat_tau_max_omega<-cbind(gamma_target,b_tau_max_omega)


#-------------------------------------------------------------------------------
# 3.3 CHECKS
#-------------------------------------------------------------------------------
# a. Solution is linear combination of gamma_target, \mathbf{1} and \mathbf{k}, i.e., the residual in the following regression vanishes
summary(lm(b_tau_max_omega[,i]~cbind(gamma_target,cos(omega0*(0:(L-1))),sin(omega0*(0:(L-1))))-1))

# b. TC constraint
target_correlation_check<-NULL
# Scaling is required because alpha_h has the meaning of a correlation
for (i in 1:ncol(b_tau))
  target_correlation_check<-c(target_correlation_check,b_tau_max_omega[,i]%*%gamma_target/sqrt(b_tau_max_omega[,i]%*%b_tau_max_omega[,i]*gamma_target%*%gamma_target))
# Check: should vanish
# Note: CCF(h) computed for the table differs because the reference is hp_trend (the nowcast), not gamma_target (the MSE forecast), in computing the CCF. This is just a scaling effect.
target_correlation_check-target_correlation_vec

# c. Check unit length constraint: should vanish
diag(t(b_tau_max_omega)%*%b_tau_max_omega)-1


#-------------------------------------------------------------------------------
# 3.4 MAX-TAU FOR TREND FREQUENCY ZERO
#-------------------------------------------------------------------------------

omega0<-0
# Note: for omega0=0 the Boolean phase_excess==T/F has no effect because Tau is not bounded.
b_tau_max<-max_tau_vec<-max_tau_excess_vec<-NULL
for (i in 1:length(target_correlation_vec))#i<-1
{
  target_correlation<-target_correlation_vec[i]
  
  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,omega0,phase_excess)

  b_tau_max<-cbind(b_tau_max,max_tau_obj$b_opt)
  max_tau_vec<-c(max_tau_vec,max_tau_obj$tau_max)
}
# Take minus sign to match sign convention in paper (in code it is assumed that transferfunction is based on exp(+i*omega) whereas in paper we assume exp(-1.i*omega)).
# A lead means a positive number (of the sign-inverted tau):
max_tau_vec
# Compute Tau of MSE
if (omega0==0)
{
  tauh0<--gamma_target%*%(0:(L-1))/sum(gamma_h)
} else
{
  tauh0<--Arg(sum(gamma_target*exp(1.i*omega0*(0:(L-1)))))/omega0
}
tauh0
# Outcome: max_tau_vec>tauh0 (efficient frontier)

b_tau_max_0<-b_tau_max

#-------------------------------------------------------------------------------
# 3.5 CHECKS
#-------------------------------------------------------------------------------
# a. Solution is linear combination of gamma_target, \mathbf{1} and \mathbf{k}:
# the regression residual vanishes.
summary(lm(b_tau_max_0[,i]~cbind(gamma_target,rep(1,L),(0:(L-1)))-1))

# b. TC constraint
target_correlation_check<-NULL
# Scaling is required because alpha_h has the meaning of a correlation
for (i in 1:ncol(b_tau))
  target_correlation_check<-c(target_correlation_check,b_tau_max_0[,i]%*%gamma_target/sqrt(b_tau_max_0[,i]%*%b_tau_max_0[,i]*gamma_target%*%gamma_target))
# Check: should vanish
# Note: CCF(h) computed for the table differs because the reference is hp_trend (the nowcast), not gamma_target (the MSE forecast), in computing the CCF. This is just a scaling effect.
target_correlation_check-target_correlation_vec

# c. Check unit length constraint: should vanish
diag(t(b_tau_max_0)%*%b_tau_max_0)-1

#-------------------------------------------------------------------------------
# 3.6 COMPUTE PERFORMANCES
#-------------------------------------------------------------------------------
# Compute time-shifts of DFP and Max-Tau designs at both reference frequencies:
# all filters share IDENTICAL target correlations (TC). Therefore, Max-Tau
# optimized for a given omega0 must outperform all other filters AT THAT
# FREQUENCY, i.e. its Tau must be the largest in the corresponding column.

# I) omega0=0
# -----------------------------------------------------------------------------
# For each TC level, compute Tau(0) (lead at the trend frequency) for:
#   - the time-shift DFP filter (b_tau)
#   - Max-Tau optimized for omega0=0     (b_tau_max_0)
#   - Max-Tau optimized for omega0=pi/20 (b_tau_max_omega)
tau_mat_0<-NULL
for (i in 1:ncol(b_tau))
{
  tau_mat_0<-rbind(tau_mat_0,-c(b_tau[,i]%*%(0:(L-1))/sum(b_tau[,i]), b_tau_max_0[,i]%*%(0:(L-1))/sum(b_tau_max_0[,i]), b_tau_max_omega[,i]%*%(0:(L-1))/sum(b_tau_max_omega[,i])))
}
# The last row corresponds to TC=1 (perfect replication of the MSE target),
# in which case Tau(0) of DFP and Max-Tau(0) both diverge to their
# theoretical maximum: set to Inf here for clarity in the table.
tau_mat_0[nrow(tau_mat_0),1:2]<-Inf
colnames(tau_mat_0)<-c("DFP","Max-Tau(0)","Max-Tau(pi/20)")
rownames(tau_mat_0)<-round(target_correlation_vec,3)
# RESULT: for given TC (rownames of the matrix), Max-Tau(0) (middle column) has the largest
# Tau(0) at omega0=0. No other linear predictor of length <= L can
# outperform Max-Tau(0) in BOTH TC and Tau(0) simultaneously — confirming
# the efficient-frontier property at the trend frequency.
tau_mat_0


# II) omega0=pi/20
# -----------------------------------------------------------------------------
# Same comparison, but now measuring the lead at the business-cycle
# frequency omega0=pi/20 (10-year cycle).
omega<-pi/n_freq
tau_mat_omega<-NULL
for (i in 1:ncol(b_tau))
{
  tau_mat_omega<-rbind(tau_mat_omega,-c(Arg(b_tau[,i]%*%exp(1.i*omega*(0:(L-1)))), Arg(b_tau_max_0[,i]%*%exp(1.i*omega*(0:(L-1)))),Arg(b_tau_max_omega[,i]%*%exp(1.i*omega*(0:(L-1)))))/omega) 
}
colnames(tau_mat_omega)<-c("DFP","Max-Tau(0)","Max-Tau(pi/20)")
rownames(tau_mat_omega)<-round(target_correlation_vec,3)
# RESULT: for given TC (rownames of the matrix), Max-Tau(pi/20) (rightmost column) has the
# largest Tau(pi/20) at omega0=pi/20 — confirming the efficient-frontier
# property at the business-cycle frequency.
tau_mat_omega



#-------------------------------------------------------------------------------
# 3.7 PLOTS
#-------------------------------------------------------------------------------

# I) PREDICTOR WEIGHTS AND TIME-SHIFTS
# -----------------------------------------------------------------------------

# Assemble filter matrices with descriptive column names showing the
# achieved lead (tau) for each TC level.
filt_mat_tau_max_0<-cbind(gamma_target,b_tau_max_0)
colnames(filt_mat_tau_max_0)<-c("MSE",paste("\u03C4","=",round(tau_mat_0[,2],2),sep=""))

colnames(filt_mat_tau_max_omega)<-c("MSE",paste("\u03C4","=",round(tau_mat_omega[,3],2),sep=""))


# Plot layout: filter coefficients (lag domain) and time-shift (frequency domain)
colo<-c("green",rainbow(ncol(b_tau_max_0)))

par(mfrow=c(2,2))


# I.1) omega0=0: FILTER COEFFICIENTS
# -----------------------------------------------------------------------------

# Scale filter coefficients for comparability across filters
mplot<-scale(filt_mat_tau_max_0,center=F,scale=T)

layout<-plot(mplot[,1],main="Predictors: Max-Tau(0)",axes=F,type="l",xlab="Lag",ylab="",col=colo[1],lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(paste("TC: ",round(target_correlation_vec[i-1],3),", ",colnames(mplot)[i],sep=""),line=-i,col=colo[i])  
}
abline(h=0)
axis(1,at=c(1,1:(nrow(mplot)/10)*10),labels=c(1,1:(nrow(mplot)/10)*10)-c(1,rep(0,5)))
axis(2)
box()

# I.2) omega0=0: TIME-SHIFT ACROSS FREQUENCIES
# -----------------------------------------------------------------------------
K<-600
plot_T<-F
amp<-shift<-NULL
for (i in 1:ncol(filt_mat_tau_max_0))
{
  trf_obj<-amp_shift_func(K,filt_mat_tau_max_0[,i],plot_T)
  shift<-cbind(shift,-trf_obj$shift)
  amp<-cbind(amp,trf_obj$amp)
}
colnames(amp)<-colnames(shift)<-colnames(filt_mat_tau_max_0)
mplot<-shift[1:(K/10),]

plot(mplot[,1],ylim=c(min(mplot),9),axes=F,col=colo[1],type="l",xlab="Frequency",ylab="",main="Time-Shift Lower Frequencies",lty=2,lwd=2)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=c(1,K/20,K/10),labels=c("0","pi/20","pi/10"))
axis(2)
box()


# II.2) omega0=pi/20: FILTER COEFFICIENTS
# -----------------------------------------------------------------------------

# Scale filter coefficients for comparability across filters
mplot<-scale(filt_mat_tau_max_omega,center=F,scale=T)

layout<-plot(mplot[,1],main="Max-Tau(pi/20)",axes=F,type="l",xlab="Lag",ylab="",col=colo[1],lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  mtext(colnames(mplot)[i],line=-i,col=colo[i])  
}
abline(h=0)
axis(1,at=c(1,1:(nrow(mplot)/10)*10),labels=c(1,1:(nrow(mplot)/10)*10)-c(1,rep(0,5)))
axis(2)
box()

# 3.2 omega0=pi/20: TIME-SHIFT ACROSS FREQUENCIES
# -----------------------------------------------------------------------------
K<-600
plot_T<-F
amp<-shift<-NULL
for (i in 1:ncol(filt_mat_tau_max_omega))
{
  trf_obj<-amp_shift_func(K,filt_mat_tau_max_omega[,i],plot_T)
  shift<-cbind(shift,-trf_obj$shift)
  amp<-cbind(amp,trf_obj$amp)
}
colnames(amp)<-colnames(shift)<-colnames(filt_mat_tau_max_omega)
mplot<-shift[1:(K/10),]

plot(mplot[,1],ylim=c(-5,9),axes=F,col=colo[1],type="l",xlab="Frequency",ylab="",main="Time-Shift Lower Frequencies",lty=2,lwd=2)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
abline(v=K/20)
axis(1,at=c(1,K/20,K/10),labels=c("0","pi/20","pi/10"))
axis(2)
box()

#-------------------------------------------------------------------------------
# OUTCOME
#-------------------------------------------------------------------------------
# A) Top two plots (omega0=0):
# - For decreasing TC:
#   a) the weight on the linear decoupling vector of Max-Tau increases, and the
#      predictor profile becomes "more linear" (violet line, top left panel).
#   b) the time-shift (lead) at frequency zero increases (top right panel).
#   c) Unfortunately, this lead does NOT extend to business-cycle frequencies.
#      Therefore, Max-Tau(0) is not well-suited as a leading indicator design
#      for business-cycle analysis.
#
# B) Bottom two plots (omega0=pi/20):
# - For decreasing TC:
#   a) the weight on the purely sinusoidal decoupling vector of Max-Tau increases,
#      and the predictor looks increasingly periodic (violet line, bottom left panel).
#   b) the time-shift increases at omega0=pi/20 (bottom right panel).
#   c) Max-Tau(pi/20) consistently outperforms the MSE target at business-cycle
#      frequencies, implying the predictor LEADS the nowcast for business-cycle
#      analysis.

# II) PREDICTORS AT RECESSIONS
# -----------------------------------------------------------------------------
# Apply each filter to the actual GDP growth series and visually inspect
# behaviour around two historical recession episodes: the Dotcom bust and
# the Great Recession (Financial Crisis).

y_mat_tau_max_omega<-NULL
for (i in 1:ncol(filt_mat_tau_max_omega))
  y_mat_tau_max_omega<-cbind(y_mat_tau_max_omega,filter(diff_log_GDP,filt_mat_tau_max_omega[,i],side=1))
colnames(y_mat_tau_max_omega)<-colnames(filt_mat_tau_max_omega)
y_mat_tau_max_0<-NULL
for (i in 1:ncol(filt_mat_tau_max_0))
  y_mat_tau_max_0<-cbind(y_mat_tau_max_0,filter(diff_log_GDP,filt_mat_tau_max_0[,i],side=1))
colnames(y_mat_tau_max_0)<-colnames(filt_mat_tau_max_0)

par(mfrow=c(2,2))


#------------
# II.1) omega0=0: RECESSION PLOTS
# -----------------------------------------------------------------------------
colo<-c("green",rainbow(ncol(b_tau_max_0)))

# Dotcom recession window
anf<-87
enf<-105
mplot<-scale(y_mat_tau_max_0)[anf:enf,]
colnames(mplot)<-colnames(y_mat_tau_max_0)


plot(mplot[,1],col=colo[1],main="Dotcom: Max-Tau(0)", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_alpha0[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

# Great Recession / Financial Crisis window
anf<-108
enf<-134

mplot<-scale(y_mat_tau_max_0)[anf:enf,]

plot(mplot[,1],col=colo[1],main="Great recession", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(bquote(beta==.(beta_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_alpha0[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()



# II.2) omega0=pi/20: RECESSION PLOTS
# -----------------------------------------------------------------------------

# Dotcom recession window
anf<-87
enf<-105
mplot<-scale(y_mat_tau_max_omega)[anf:enf,]
colnames(mplot)<-colnames(y_mat_tau_max_omega)


plot(mplot[,1],col=colo[1],main=paste("Dotcom: Max-Tau(pi/20)"), axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_tau[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

# Great Recession / Financial Crisis window
anf<-108
enf<-134

mplot<-scale(y_mat_tau_max_omega)[anf:enf,]

plot(mplot[,1],col=colo[1],main="Great recession", axes=F,type="l",xlab="",ylab="",lwd=3,lty=2,ylim=c(min(mplot),max(mplot)))
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(bquote(beta==.(beta_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat_tau[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()


#-------------------------------------------------------------------------------
# OUTCOME
#-------------------------------------------------------------------------------
# As predicted by the time-shift panels in the previous plot, Max-Tau(0) does
# NOT qualify as a leading business-cycle tool: the design leads at LOWER
# frequencies. Specifically, for components with duration exceeding 20 years,
# Max-Tau(0) would lead.
#
# In contrast, Max-Tau(pi/20) is leading (left-shifted) at BOTH recession
# episodes (Dotcom and Great Recession).
#
# Larger and/or different leads could be obtained by allowing for a smaller
# TC or by choosing an alternative omega0.
#
# Alternatively, one could control overfitting to enhance the lead further —
# see the Exercises below.


###############################################################################
# EXERCISE 4: REPLICATE DUAL MAX-TAU BY INVERTED PRIMAL MAX-TAU
####################################################################################







# Additional exercises not in paper:
# 1. Show equivalence of primal and dual on frontier (an d difference away from frontier)
# 2. Multi-frequency approach
# 3. Apply Curvature Max-Tau to BCA





