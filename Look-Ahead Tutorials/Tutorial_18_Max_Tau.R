# WORK IN PROGRESS (SEPTEMBER 2026)
# NEW MATERIAL: September 2026



# ══════════════════════════════════════════════════════════════════════════
# TUTORIAL 18: MAX-TAU PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
#
# CONCEPT
# -------
# Max-Tau differs from DFP in a fundamental way:
#
#   - DFP decouples from the *nowcast* and defines a CONDITIONAL (weak)
#     efficient frontier between Target Correlation (TC) and lead, i.e.
#     conditional on that decoupling constraint.
#
#   - Max-Tau decouples from *all linear predictors (of the same
#     or shorter length)*. This means: no other
#     linear predictor of at least this length can outperform Max-Tau
#     in BOTH target correlation (TC) and lead at the reference frequency.
#     Max-Tau therefore defines an UNCONDITIONAL (strong) efficient
#     TC/lead frontier.
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
#     i.e. Gamma(0) < 0 (see Exercise 1).
#   - TC is also substantially reduced in such extreme DFP designs
#     (see examples below).
#
# In contrast, time-shift DFP and Max-Tau CANNOT invert signs: the
# orientation of the trend and the sign of the mean are always preserved.
#
# The time-shift DFP can, however, lead to negative TC (see Section 4.4
# in the new [Sept-2026] paper) and needs specific control to avoid this
# (most) undesirable outcome. In contrast, Max-Tau always ensures a
# positive TC.
#
#
# NOVELTY: GENERALISING THE REFERENCE FREQUENCY
# -----------------------------------------------
# Until now, DFP has addressed the reference frequency omega_0 = 0 (trend)
# only. We extend Max-Tau here to address omega_0 > 0 as well — e.g.
# Max-Tau can maximise lead at BUSINESS-CYCLE frequencies, not just at
# the trend frequency (see Exercise 3).
#
#
# CAVEAT: OVERFITTING
# --------------------
# Because Max-Tau maximises lead at a single reference frequency, it
# becomes prone to overfitting as filter length L increases (DFP does not
# share this problem to the same extent). Two remedies are illustrated in
# this and subsequent tutorials:
#   (1) controlling curvature of the filter (Exercise 5), or
#   (2) extending the lead criterion to multiple frequencies (Exercise 6).
#
#
# PROS AND CONS
# --------------
# PROS:
#   - Max-Tau defines a strong (unconditional) efficient frontier against
#     ALL linear predictors.
#   - Max-Tau preserves the sign of mean/trend and ensures a positive TC.
#
# CONS:
#   - MSE-DFP can generate more extreme look-ahead when sign inversion (of, e.g.,
#     mean/trend) is permitted, at the cost of substantially reduced TC.
#   - Max-Tau is subject to overfitting at the reference frequency. This
#     can be addressed by additional regularisation (curvature, see
#     Exercise 5) or by extending Max-Tau from a single to multiple
#     reference frequencies (see Exercise 6).
#
# ══════════════════════════════════════════════════════════════════════════
#
# ── BACKGROUND / REFERENCE ────────────────────────────────────────────────

#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     A Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

#   Wildi, M. (2026b) (New Sept-2026)
#     Forecasting on the Accuracy-Timeliness Frontier:
#     Decoupling From Present and Max-Tau Predictors.
#     (not yet published, see "Papers" folder in project)
#
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# 1. INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

rm(list = ls())

# --- Load core algorithms (packaged as functions; treated here as black boxes) ---

# DFP optimisation routines: used below for comparison against Max-Tau.
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


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA: US REAL GDP
# ══════════════════════════════════════════════════════════════════════════

# Toggle reload_data to TRUE to fetch fresh data from FRED (overwrites the
# local copy); FALSE loads the previously saved series (recommended for
# reproducible tutorial runs).
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDP"))
} else {
  load(file.path(getwd(), "Data", "GDP"))
}

# --- Quick sanity checks ---
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

start_year   <- 1992
end_year     <- 2024
sample_range <- paste(start_year, end_year, sep = "/")

y_xts <- log(GDPC1[sample_range])   # for plotting only
y     <- as.double(y_xts)           # for computation
len   <- length(y)


# ══════════════════════════════════════════════════════════════════════════
# 3. EXPLORATORY PLOTS
# ══════════════════════════════════════════════════════════════════════════

par(mfrow = c(2, 2))
plot(GDPC1,                   main = "US Real GDP (levels)")
plot(y_xts,                   main = "Log GDP")
plot(diff(y_xts),             main = "Log-differences of GDP")
acf(na.exclude(diff(y_xts)),  main = "ACF of log-differences")

# The data, retrieved from FRED (https://fred.stlouisfed.org/), are displayed
# in the figure above, where log-differences (bottom left) cover the last
# three recession episodes from 1992 to 2024. HP is applied to log-differences
# to track trend-growth, whose sign indicates below/above-average growth
# (Wildi, 2024). While this application deviates from business-cycle
# orthodoxy -- where the HP bandpass is typically applied to the original
# non-stationary GDP -- we select this framework because it emphasizes
# lead/lag issues more clearly and because tracking growth through the
# lowpass addresses spurious cycles inherent to the classic approach
# (Wildi, 2024, and earlier Tutorials).


# ══════════════════════════════════════════════════════════════════════════
# 4. TARGET SPECIFICATION: HP TREND OF DIFF-LOG GDP
# ══════════════════════════════════════════════════════════════════════════

# Forecast horizon (in periods): a two-quarters-ahead horizon, as in
# Wildi (2026b).
h <- 2

# HP smoothing parameter (standard quarterly setting).
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

# Concurrent (one-sided) HP trend filter, assuming an I(2) process.
hp_trend_long <- HP_obj$hp_trend
hp_trend      <- hp_trend_long[1:L]

# MSE-optimal one-sided filter for the bi-infinite HP trend, assuming
# white-noise innovations (white noise is confirmed by the ACF plot above,
# bottom-right panel).
hp_mse_long <- HP_obj$hp_mse
hp_mse      <- hp_mse_long[1:L]








##############################################################################
# EXERCISE 1: MSE-DFP
##############################################################################

#===============================================================================
# 1.1 DFP SETTINGS: TARGET AND CONSTRAINT VECTORS
#===============================================================================

# ---- Forecast horizon and CCF range ------------------------------------------

# Forecast horizon (delta) for the target.
delta <- h

# Start lag for the cross-correlation function (CCF); max_lag = 0 restricts
# attention to the right tail only (non-negative lags/leads).
max_lag <- 0

# ---- Target and constraint vectors --------------------------------------------

# Target: MSE-optimal predictor of the HP trend at horizon h = delta.
gammah <- gamma_target <- hp_trend_long[h + 1:L]

# Constraint: nowcast (horizon 0) predictor of the HP trend.
gamma0 <- gamma_constraint <- hp_trend_long[1:L]

# ---- Sanity checks on gamma_target ---------------------------------------------
ts.plot(gamma_target, main = "Target vector: HP trend weights at horizon h")

sum(hp_trend_long)                        # should sum to 1
sqrt(t(hp_trend_long) %*% hp_trend_long)  # filter norm (std-dev proxy)


#===============================================================================
# 1.2 MSE-DFP BENCHMARK: FAMILY OF SOLUTIONS INDEXED BY alpha0
#===============================================================================
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
#   both time-shift DFP and Max-Tau -- but at the cost of SIGN INVERSION.
#   In this example, that occurs when alpha0 is close to 0 (i.e. near-
#   complete decoupling from the nowcast constraint).

alpha0_vec <- c(0.1, 0.05, 0.02, 0.017, 0.005, 0)

# ---- Containers for results across the alpha0 grid -----------------------------
lambda             <- NULL   # Lagrange multipliers from the DFP solve
b0_mat             <- NULL   # DFP filter coefficients (one column per alpha0)
cor_vec_mse_la_mat <- NULL   # CCF of each DFP filter vs. the MSE target

# ---- Loop over the alpha0 grid: solve DFP and evaluate its CCF -----------------
for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # Core DFP solve: quadratic-in-lambda problem, then unit-length filter.
  b0_obj <- mse_dfp_from_alpha0_func(gamma_constraint, gamma_target, alpha0)
  b      <- b0_obj$b
  
  b0_mat <- cbind(b0_mat, b)
  lambda <- c(lambda, b0_obj$lambda)
  
  # Cross-correlation of the resulting DFP predictor with the MSE target.
  cor_vec_mse_la_mat <- cbind(
    cor_vec_mse_la_mat,
    compute_acf_at_lags_zero_delta_func(max_lag, h, b0_mat[, i], hp_trend)$cor_vec
  )
}

colnames(b0_mat)             <- alpha0_vec
colnames(cor_vec_mse_la_mat) <- alpha0_vec

# ---- Reference CCF: the MSE (HP trend) target against itself -------------------
cor_vec_t_hp_trend <- compute_acf_at_lags_zero_delta_func(
  max_lag, h, gammah, hp_trend
)$cor_vec


#===============================================================================
# 1.3 SUMMARY STATISTICS ACROSS THE alpha0 GRID
#===============================================================================

b_alpha0                  <- b0_mat
cor_vec_mse_la_mat_alpha0 <- cor_vec_mse_la_mat

# ---- Gamma(0): sum of filter coefficients --------------------------------------
# Gamma(0) < 0 implies trend and mean inversion (see discussion below).
Gamma0_alpha0 <- apply(b_alpha0, 2, sum)
Gamma0_alpha0   # the last two filters are subject to inversion

# ---- Time-shift (lead/lag) at frequency zero, for each alpha0 ------------------
tau_alpha0 <- -as.vector(t(b_alpha0) %*% (0:(L - 1)) / apply(b_alpha0, 2, sum))

# Time-shift is not properly defined when Gamma(0) < 0 (it would correspond
# to the shift of the sign-inverted predictor). We therefore insert NA.
tau_alpha0[which(Gamma0_alpha0 < 0)] <- NA

# ---- Corresponding statistics for the plain MSE benchmark filter ---------------
mse_b <- c(
  t(gammah) %*% gamma0 / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. nowcast
  t(gammah) %*% gammah / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. itself
  sum(gammah),                                                              # Gamma(0)
  -as.double(t(gammah) %*% (0:(L - 1)) / sum(gammah))                       # tau at freq 0
)

# ---- Combine into a single comparison table: MSE benchmark vs. DFP(alpha0) grid ----
table_alpha0 <- cbind(
  mse_b,
  rbind(
    cor_vec_mse_la_mat_alpha0[1, ],      # TC at lag 0 (nowcast)
    cor_vec_mse_la_mat_alpha0[h + 1, ],  # TC at lag h (target horizon)
    Gamma0_alpha0,                       # Gamma(0)
    tau_alpha0                           # time-shift (lead) at frequency 0
  )
)
rownames(table_alpha0) <- c(
  "Correlation with nowcast: CCF(0)",
  "TC: CCF(h)",
  "Gamma(0)",
  "Shift at omega0=0 (lead when positive)"
)

table_alpha0


#-------------------------------------------------------------------------------
# DISCUSSION OF RESULTS: THE TC/LEAD DILEMMA AND THE (CONDITIONAL)
# EFFICIENT FRONTIER
#-------------------------------------------------------------------------------
#
# Ideally, a predictor would simultaneously maximise:
#   - Target Correlation (TC), measured as the CCF at horizon h, CCF(h)
#     (row 2 in table_alpha0), and
#   - Lead (tau) (row 4 in table_alpha0).
#
# Both cannot be achieved at once -- this is the DILEMMA. The best
# attainable outcome is a predictor lying ON THE EFFICIENT FRONTIER
# between TC and lead. MSE-DFP sits on the efficient frontier of all
# predictors SUBJECT TO nowcast decoupling. Here is how this manifests in
# the table:
#
#   1. STRONGER DECOUPLING (smaller entries in row 1, CCF(0)) spills over
#      into smaller TC (row 2, CCF(h)). MSE-DFP minimises this undesirable
#      loss of TC: no other linear predictor can achieve both a smaller
#      CCF(0) AND a larger CCF(h) simultaneously -- this is precisely what
#      makes it "efficient".
#
#   2. Stronger decoupling can be linked BIJECTIVELY (via a strictly
#      monotonic function) to the lead (tau) at the reference frequency
#      omega_0 = 0 -- see Corollary 2 in Wildi (2026b). This link requires
#      Gamma(0) > 0 (row 3 in the table).
#
#   3. Therefore, BY COROLLARY 2, the dilemma between tau (row 4, lead)
#      and TC (row 2) holds whenever Gamma(0) > 0: pushing lead higher
#      necessarily costs some TC, and vice versa.
#
#   4. HOWEVER, the link between CCF(0) and tau does NOT guarantee that
#      tau is maximised across ALL linear predictors of the same length --
#      it is not. The efficient frontier described above (points 1-3) is
#      only efficient CONDITIONAL ON nowcast decoupling.
#
#      Max-Tau resolves this geometric limitation: it maximises tau
#      UNCONDITIONALLY, among ALL linear predictors of the same length --
#      not merely among those that decouple from the nowcast.


#-------------------------------------------------------------------------------
# NOTE ON SIGN INVERSION
#-------------------------------------------------------------------------------
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
#   (which is the case here -- see row 2 of `table_alpha0`), the predictor
#   still correlates POSITIVELY with the target, i.e. it remains
#   effectively informative about the target's dynamics.
#
# WHAT IT MEANS IN PRACTICE:
#   Sign inversion of the mean signifies that the data should be CENTERED
#   before applying the predictor; otherwise the resulting level will
#   carry the wrong sign. Once the forecast is computed on centered data,
#   the correct mean can simply be added back to the centered forecast.
#
# WHY THIS IS NOT A SERIOUS LIMITATION:
#   Static mean or scale adjustments are trivial operations. They do NOT
#   affect the dynamic look-ahead capabilities of the predictor -- i.e.
#   the TC/lead trade-off discussed above is entirely unaffected by
#   whether the data was centered before or after filtering. Both TC and
#   tau are insensitive to shift and scaling.
#
#
# This sets up the comparison in the next exercise: Max-Tau will match or
# dominate every point on the MSE-DFP frontier above in terms of the
# TC/lead trade-off.


########################################################################################
# EXERCISE 2: TIME-SHIFT DFP
########################################################################################

#====================================================================================
# It is assumed that Exercise 1 has been run to initialize all settings!
#====================================================================================
#
# As seen in `table_alpha0`, MSE-DFP can generate SIGN INVERSION
# (Gamma(0) < 0) when alpha0 is pushed close to zero.
#
# Time-shift DFP AVOIDS sign inversion by construction: instead of
# specifying the decoupling parameter alpha0 directly, we specify the
# desired LEAD tau at the reference frequency omega_0 = 0, and infer the
# corresponding alpha0 = alpha(tau) via the bijective (strictly monotonic)
# relationship established in Proposition 3 and Corollary 2, Wildi (2026b).
#
# Since alpha(tau) is derived from this relationship, it is guaranteed to
# stay within the range that preserves Gamma(0) > 0 -- hence no sign
# inversion.
#
# Apart from this reparameterisation (tau -> alpha0, rather than choosing
# alpha0 directly), Exercise 2 is otherwise IDENTICAL to Exercise 1.


#===============================================================================
# 2.1 COMPUTE TIME-SHIFT DFP
#===============================================================================

# ---- Grid of desired leads (tau) at the reference frequency omega_0 = 0 -------
# The last entry is a `large` value, representing an (approximately) infinite
# lead.
#
# NOTE: tau > 0 is the lead of DFP OVER the MSE benchmark: the constraint
# addresses the DIFFERENCE between the shift of DFP and the shift of MSE.
tau_vec <- c(1, 2, 6, 100000)

# ---- Containers for results across the tau grid --------------------------------
b0_mat             <- NULL   # DFP filter coefficients (one column per tau)
cor_vec_mse_la_mat <- NULL   # CCF of each DFP filter vs. the MSE target

# ---- Loop over the tau grid: solve time-shift DFP and evaluate its CCF ---------
for (i in seq_along(tau_vec)) {
  
  # NOTE ON SIGN CONVENTION: mse_dfp_from_tau_func() expects leads to be
  # passed as NEGATIVE numbers (a "lead" is a negative "lag").
  lead <- -tau_vec[i]
  
  # Call the dedicated function to compute the DFP filter for a specified
  # lead (see mse_dfp_from_tau_func for the derivation, based on
  # Proposition 3, Wildi 2026b).
  dfp_obj <- mse_dfp_from_tau_func(gamma_constraint, gamma_target, lead)
  
  # Extract the components returned by the function.
  tau0    <- dfp_obj$tau0     # frequency-zero time-shift of gamma0
  tauh    <- dfp_obj$tauh     # frequency-zero time-shift of gammah
  lambda0 <- dfp_obj$lambda0  # DFP regularisation weight on gamma0
  b       <- dfp_obj$b        # raw DFP filter coefficients
  
  b0_mat <- cbind(b0_mat, b)
  
  # Cross-correlation of the resulting DFP predictor with the MSE target.
  cor_vec_mse_la_mat <- cbind(
    cor_vec_mse_la_mat,
    compute_acf_at_lags_zero_delta_func(max_lag, h, b, hp_trend)$cor_vec
  )
}

colnames(b0_mat) <- colnames(cor_vec_mse_la_mat) <- paste("Shift ", -tau_vec, sep = "")

b_tau                  <- b0_mat
cor_vec_mse_la_mat_tau <- cor_vec_mse_la_mat


#===============================================================================
# 2.2 VALIDATION CHECKS
#===============================================================================

# ---- Check (a): Gamma(0) should be positive for every tau ----------------------
# As tau -> infinity, Gamma(0) -> 0 (see Wildi, 2026b), but it should never
# turn negative -- confirming that time-shift DFP avoids sign inversion.
Gamma0_tau <- apply(b_tau, 2, sum)
cat("Gamma(0) across the tau grid (all should be positive):\n")
print(Gamma0_tau)

# ---- Check (b): the imposed lead constraint is met ------------------------------

# Time-shift of each DFP filter at frequency zero.
tau_tau <- -as.vector(t(b_tau) %*% (0:(L - 1)) / apply(b_tau, 2, sum))

# Time-shift of the MSE benchmark at frequency zero.
tauh <- -as.double(gammah %*% (0:(L - 1)) / sum(gammah))

# Lead of DFP over MSE: should match the originally imposed tau_vec.
tau <- tau_tau - tauh

cat("Residual: imposed lead vs. realised lead (should be ~0):\n")
print(tau - tau_vec)


#===============================================================================
# 2.3 SUMMARY STATISTICS
#===============================================================================

# ---- Decoupling parameter implied by each tau -----------------------------------
# alpha0 = alpha(tau) is the decoupling parameter, i.e. CCF(0) up to scaling
# (compare to alpha0 chosen directly in Exercise 1).
alpha_tau <- as.vector(t(b_tau) %*% gamma0)

# ---- Corresponding statistics for the plain MSE benchmark filter ----------------
mse_b <- c(
  t(gammah) %*% gamma0 / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. nowcast
  t(gammah) %*% gammah / sqrt(t(gammah) %*% gammah * t(gamma0) %*% gamma0), # TC vs. itself
  -as.double(t(gammah) %*% (0:(L - 1)) / sum(gammah)),                      # tau at freq 0
  t(gammah) %*% gamma0                                                      # alpha0 (MSE benchmark)
)

# ---- Combine into a single comparison table: MSE benchmark vs. DFP(tau) grid ----
table_tau <- cbind(
  mse_b,
  rbind(
    cor_vec_mse_la_mat_tau[1, ],      # TC at lag 0 (nowcast)
    cor_vec_mse_la_mat_tau[h + 1, ],  # TC at lag h (target horizon)
    tau_tau,                          # time-shift (lead) at frequency 0
    alpha_tau                         # decoupling parameter alpha0(tau)
  )
)
rownames(table_tau) <- c(
  "Correlation with nowcast: CCF(0)",
  "TC: CCF(h)",
  "Shift at omega0=0 (lead when positive)",
  "alpha0"
)

# Similar to table_alpha0 in Exercise 1, though we now:
#   - remove the Gamma(0) row (it is always positive here, by construction), and
#   - add an alpha0 row: the decoupling parameter alpha0 = alpha0(tau) is now
#     DERIVED from the desired lead tau over the MSE predictor, rather than
#     chosen directly.
table_tau


#===============================================================================
# 2.4 PLOTS
#===============================================================================

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

# Great Financial Crisis
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

# Outcome (excerpt from paper)
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
# EXERCISE 3: MAX-TAU
########################################################################################
#
# DESIGN OF THIS EXERCISE:
#   We replicate the TC (target correlation) of the time-shift DFP filters
#   from Exercise 2, and compare the resulting time-shifts / leads of
#   Max-Tau vs. time-shift DFP AT THE SAME TC. Since both families share
#   identical TC, any difference in lead is attributable purely to the
#   different optimization criterion -- and Max-Tau, by construction, must
#   dominate at its reference frequency.
#
#   We illustrate this at TWO reference frequencies:
#     - omega0 = 0       (trend)
#     - omega0 = pi/20   (10-year business-cycle periodicity)
#
#====================================================================================
# It is assumed that Exercises 1 and 2 have been run to initialize all settings!
#====================================================================================


#===============================================================================
# 3.1 COMPUTE TC TO MATCH TIME-SHIFT DFP
#===============================================================================
#
# NOTE ON SCALING:
#   The DFP filters above (Exercise 2) were evaluated using TC normalized by
#   1/|gamma0| (i.e. relative to hp_trend, the nowcast constraint). Max-Tau,
#   however, normalizes its TC constraint by 1/|gammah| (i.e. relative to
#   gamma_target, the MSE target) -- this is what gives alphah/target_correlation
#   its interpretation as an actual correlation in the Max-Tau formulation.
#
#   We must therefore
#   RECOMPUTE the TC of each time-shift DFP filter using the Max-Tau
#   convention (normalized against gamma_target) before using it as a
#   constraint for Max-Tau.

target_correlation_vec <- NULL

for (i in 1:ncol(b_tau)) {
  
  # Correct convention: correlation of the DFP filter with gamma_target
  # (the MSE target), NOT with hp_trend/gamma0 (the nowcast).
  target_correlation_vec <- c(
    target_correlation_vec,
    b_tau[, i] %*% gamma_target /
      sqrt(b_tau[, i] %*% b_tau[, i] * gamma_target %*% gamma_target)
  )
  
  # For reference, the (incorrect, mismatched-normalization) alternative
  # would instead correlate against hp_trend:
  #   b_tau[, i] %*% gamma_target /
  #     sqrt(b_tau[, i] %*% b_tau[, i] * hp_trend %*% hp_trend)
}


#===============================================================================
# 3.2 COMPUTE MAX-TAU: OPTIMIZED FOR omega0 = pi/20 (10-YEAR CYCLE)
#===============================================================================

gamma_h <- gamma_target

# ---- Reference frequency: 10-year periodicity -----------------------------------
n_freq <- 20
omega0 <- pi / n_freq

# ---- phase_excess switch --------------------------------------------------------
# Max-Tau can impose maximal lead at omega0 relative to EITHER:
#   - the MSE target gamma_h    (phase_excess <- TRUE), or
#   - the identity/reference    (phase_excess <- FALSE)
# See Corollaries 3 and 4 in Wildi (2026b).
#
# If gamma_h is itself LAGGING at omega0, then phase_excess <- FALSE will
# generate a LARGER lead (since the upper bound is then measured against
# the identity rather than against an already-lagging MSE target).
#
# In both cases, the maximal phase lead is capped at pi/2 (i.e. n_freq/4 in
# time units), ensuring strict positivity of Gamma(omega0) -- exactly as
# for the original time-shift DFP. As long as both candidate leads (vs.
# identity and vs. gamma_h) stay below pi/2, the two conventions coincide.
# See details in the paper (Corollary 3 vs. Corollary 4)
phase_excess <- FALSE

# ---- Overfitting caveat (addressed later, Exercise 4) ---------------------------
# When feasible, Max-Tau achieves the maximal lead n_freq/2. For a given
# filter length L, the imposed TC constraint may prevent reaching this
# maximum. However, as L increases, the extra degrees of freedom let the
# filter push Tau closer to n_freq/2 -- and in the limit of arbitrarily
# large L, the bound Tau = n_freq/2 becomes achievable for ANY TC. This is
# overfitting, and will be addressed in Exercise 4.

# ---- Loop over the TC grid (matched to time-shift DFP) --------------------------
b_tau_max          <- NULL
max_tau_vec        <- NULL
max_tau_excess_vec <- NULL

for (i in seq_along(target_correlation_vec)) {
  
  target_correlation <- target_correlation_vec[i]
  
  max_tau_obj <- max_tau_dual_func(gamma_target, target_correlation, omega0, phase_excess)
  
  b_tau_max   <- cbind(b_tau_max, max_tau_obj$b_opt)     # predictor weights
  max_tau_vec <- c(max_tau_vec, max_tau_obj$tau_max)      # lead/lag vs. identity
}

# NOTE ON SIGN: the code assumes a transfer function based on exp(+i*omega),
# whereas the paper assumes exp(-i*omega). We take the negative sign below
# so that a POSITIVE number consistently means a LEAD, matching the paper's
# convention.
cat("Max-Tau lead at omega0 = pi/20, across the matched-TC grid:\n")
print(max_tau_vec)

# ---- Compare against the lead of the MSE target itself --------------------------
# For omega0 > 0, the time-shift formula is Tau = Phi(omega0) / omega0,
# where Phi is the (unwrapped) phase of the transfer function.
tauh10 <- Arg(gamma_h %*% exp(-1i * omega0 * (0:(L - 1)))) / omega0
cat("Lead of the MSE target itself at omega0 = pi/20:\n")
print(tauh10)
# OUTCOME: Max-Tau's lead exceeds tauh10 -- confirming the efficient-frontier
# property at this frequency.
#
# NOTE: if tauh10 < 0, the MSE target itself LAGS the identity at omega0.
# In that case, phase_excess <- FALSE will generate an even larger lead,
# since it is then bounded by pi/2 relative to the identity, rather than by
# pi/2 relative to the (already lagging) MSE target.

b_tau_max_omega        <- b_tau_max
filt_mat_tau_max_omega <- cbind(gamma_target, b_tau_max_omega)


#===============================================================================
# 3.3 VALIDATION CHECKS (omega0 = pi/20)
#===============================================================================

# ---- Check (a): solution lies in the span of {gamma_target, cos, sin} ----------
# Theory predicts the Max-Tau solution is a linear combination of
# gamma_target, cos(omega0 * lag), and sin(omega0 * lag). The regression
# residual below should therefore vanish (R^2 = 1).
cat("Check (a): Max-Tau solution as linear combination of {gamma_target, cos, sin}\n")
print(summary(lm(
  b_tau_max_omega[, i] ~ cbind(gamma_target, cos(omega0 * (0:(L - 1))), sin(omega0 * (0:(L - 1)))) - 1
)))

# ---- Check (b): TC constraint is met --------------------------------------------
# Scaling is required because alpha_h (target_correlation) has the meaning
# of an actual correlation, hence the sqrt-normalization below.
target_correlation_check <- NULL
for (i in 1:ncol(b_tau)) {
  target_correlation_check <- c(
    target_correlation_check,
    b_tau_max_omega[, i] %*% gamma_target /
      sqrt(b_tau_max_omega[, i] %*% b_tau_max_omega[, i] * gamma_target %*% gamma_target)
  )
}
cat("Check (b): TC constraint residual (should be ~0):\n")
print(target_correlation_check - target_correlation_vec)
# NOTE: the CCF(h) reported later in the summary table differs slightly
# because there, the reference is hp_trend (the nowcast), not gamma_target
# (the MSE forecast) -- this is a pure scaling effect, not an inconsistency.

# ---- Check (c): unit-length constraint ------------------------------------------
cat("Check (c): unit-length constraint residual (should be ~0):\n")
print(diag(t(b_tau_max_omega) %*% b_tau_max_omega) - 1)


#===============================================================================
# 3.4 MAX-TAU FOR THE TREND FREQUENCY (omega0 = 0)
#===============================================================================

omega0 <- 0
# NOTE: at omega0 = 0, the phase_excess switch has NO effect, since Tau is
# not bounded by n_freq/2 at the trend frequency (the n_freq/2 cap only applies to
# strictly positive reference frequencies).

b_tau_max          <- NULL
max_tau_vec        <- NULL
max_tau_excess_vec <- NULL

for (i in seq_along(target_correlation_vec)) {
  
  target_correlation <- target_correlation_vec[i]
  
  max_tau_obj <- max_tau_dual_func(gamma_target, target_correlation, omega0, phase_excess)
  
  b_tau_max   <- cbind(b_tau_max, max_tau_obj$b_opt)
  max_tau_vec <- c(max_tau_vec, max_tau_obj$tau_max)
}
max_tau_vec0 <- max_tau_vec

# Sign convention, as above: negate so that positive = lead.
cat("Max-Tau lead at omega0 = 0 (trend), across the matched-TC grid:\n")
print(max_tau_vec)

# ---- Compare against the lead of the MSE target itself --------------------------
if (omega0 == 0) {
  tauh0 <- -gamma_target %*% (0:(L - 1)) / sum(gamma_h)
} else {
  tauh0 <- -Arg(sum(gamma_target * exp(1i * omega0 * (0:(L - 1))))) / omega0
}
cat("Lead of the MSE target itself at omega0 = 0:\n")
print(tauh0)
# OUTCOME: max_tau_vec > tauh0 -- confirming the efficient-frontier property
# at the trend frequency as well.

b_tau_max_0 <- b_tau_max


#===============================================================================
# 3.5 VALIDATION CHECKS (omega0 = 0)
#===============================================================================

# ---- Check (a): solution lies in the span of {gamma_target, 1, lag} ------------
# At omega0 = 0, cos(0*lag) = 1 and sin(0*lag) = 0, so the relevant spanning
# set degenerates to {gamma_target, constant, linear trend in lag}.
cat("Check (a): Max-Tau solution as linear combination of {gamma_target, 1, lag}\n")
print(summary(lm(
  b_tau_max_0[, i] ~ cbind(gamma_target, rep(1, L), (0:(L - 1))) - 1
)))

# ---- Check (b): TC constraint is met --------------------------------------------
target_correlation_check <- NULL
for (i in 1:ncol(b_tau)) {
  target_correlation_check <- c(
    target_correlation_check,
    b_tau_max_0[, i] %*% gamma_target /
      sqrt(b_tau_max_0[, i] %*% b_tau_max_0[, i] * gamma_target %*% gamma_target)
  )
}
cat("Check (b): TC constraint residual (should be ~0):\n")
print(target_correlation_check - target_correlation_vec)
# NOTE: as before, the CCF(h) reported in the summary table differs slightly
# because it is referenced against hp_trend rather than gamma_target -- a
# pure scaling effect.

# ---- Check (c): unit-length constraint ------------------------------------------
cat("Check (c): unit-length constraint residual (should be ~0):\n")
print(diag(t(b_tau_max_0) %*% b_tau_max_0) - 1)


#===============================================================================
# 3.6 PERFORMANCE COMPARISON: TIME-SHIFT DFP vs. MAX-TAU
#===============================================================================
#
# We now compute the time-shift (lead) of every predictor -- time-shift DFP
# and both Max-Tau variants -- AT BOTH reference frequencies. Since all
# filters at a given TC level share IDENTICAL target correlation, Max-Tau
# optimized for a given omega0 MUST outperform every other filter AT THAT
# FREQUENCY: this is the defining property of the unconditional efficient
# frontier.

# ---- I) Time-shift evaluated AT omega0 = 0 --------------------------------------
tau_mat_0 <- NULL
for (i in 1:ncol(b_tau)) {
  tau_mat_0 <- rbind(
    tau_mat_0,
    -c(
      b_tau[, i]           %*% (0:(L - 1)) / sum(b_tau[, i]),           # time-shift DFP
      b_tau_max_0[, i]     %*% (0:(L - 1)) / sum(b_tau_max_0[, i]),     # Max-Tau(0)
      b_tau_max_omega[, i] %*% (0:(L - 1)) / sum(b_tau_max_omega[, i])  # Max-Tau(pi/20)
    )
  )
}

# Insert Inf for very large Tau
tau_mat_0[nrow(tau_mat_0), 1:2] <- Inf

colnames(tau_mat_0) <- c("DFP", "Max-Tau(0)", "Max-Tau(pi/20)")
rownames(tau_mat_0) <- round(target_correlation_vec, 3)

cat("Time-shift comparison AT omega0 = 0 (rows = matched TC level):\n")
print(tau_mat_0)
# RESULT: for every TC level (rownames), Max-Tau(0) (middle column) attains the
# LARGEST Tau(0). No other linear predictor of length <= L can outperform
# Max-Tau(0) in BOTH TC and Tau(0) simultaneously at omega0 = 0 -- confirming
# the efficient-frontier property exactly at the frequency it was designed
# for.


# ---- II) Time-shift evaluated AT omega0 = pi/20 ---------------------------------
# Same comparison as above, but now measuring the lead AT the business-cycle
# frequency omega0 = pi/20 (10-year cycle), rather than at the trend (omega0
# = 0). Since omega0 > 0 here, we must use the general phase-based formula
# Tau(omega0) = -Arg(Filter(omega0)) / omega0, rather than the omega0 = 0
# shortcut used in tau_mat_0.
omega <- pi / n_freq

tau_mat_omega <- NULL
for (i in 1:ncol(b_tau)) {
  tau_mat_omega <- rbind(
    tau_mat_omega,
    -c(
      Arg(b_tau[, i]           %*% exp(1i * omega * (0:(L - 1)))),  # time-shift DFP
      Arg(b_tau_max_0[, i]     %*% exp(1i * omega * (0:(L - 1)))),  # Max-Tau(0)
      Arg(b_tau_max_omega[, i] %*% exp(1i * omega * (0:(L - 1))))   # Max-Tau(pi/20)
    ) / omega
  )
}

colnames(tau_mat_omega) <- c("DFP", "Max-Tau(0)", "Max-Tau(pi/20)")
rownames(tau_mat_omega) <- round(target_correlation_vec, 3)

cat("Time-shift comparison AT omega0 = pi/20 (rows = matched TC level):\n")
print(tau_mat_omega)
# RESULT: for every TC level (rownames), Max-Tau(pi/20) (rightmost column)
# attains the LARGEST Tau(pi/20). No other linear predictor of length <= L
# can outperform Max-Tau(pi/20) in BOTH TC and Tau(pi/20) simultaneously at
# omega0 = pi/20 -- confirming the efficient-frontier property exactly at
# the frequency it was designed for.
#
# TAKEN TOGETHER, tau_mat_0 and tau_mat_omega illustrate the CENTRAL claim
# of Max-Tau: there is no single predictor that is simultaneously optimal
# at every frequency. Max-Tau(0) dominates at omega0 = 0 but is beaten by
# Max-Tau(pi/20) at omega0 = pi/20, and vice versa -- each Max-Tau design
# defines its OWN efficient frontier, tailored to its target frequency,
# and strictly dominates the (frequency-agnostic) time-shift DFP filter at
# that frequency for the same TC.



#===============================================================================
# 3.7 PLOTS
#===============================================================================

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
# EXERCISE 4: REPLICATE DUAL MAX-TAU BY INVERTED PRIMAL MAX-TAU ON FRONTIER
###############################################################################
#
# TOPIC:
#   Max-Tau can be obtained via two different routes:
#     - the DUAL formulation (Appendix B in Wildi, 2026b) -- this is what
#       max_tau_dual_func() implements, and what we used throughout
#       Exercise 3;
#     - the INVERTED PRIMAL formulation (Theorem 2) -- solve for TC(Tau)
#       directly, invert it to get Tau(TC) from a target TC, then plug that Tau
#       back into the primal to recover the predictor.
#
#   We verify here that the two formulations COINCIDE exactly ON the
#   efficient frontier, but DIFFER away from it. This matters practically:
#   it tells us that once we know we're on the frontier, either formulation
#   is valid and interchangeable; off the frontier, they answer genuinely
#   different questions.
#
#====================================================================================
# It is assumed that Exercises 1-3 have been run to initialize all settings!
#====================================================================================


#===============================================================================
# 4.1 IDENTIFY WHICH DESIGNS LIE ON THE FRONTIER
#===============================================================================
#
# The frontier is characterized by TC values LARGER than a threshold U (see
# Theorem 2 for the exact definition/derivation of U). Designs with
# TC > U lie on the frontier; designs with TC <= U do not.

U <- as.double(sqrt(1 - sum(gammah)^2 / (gammah %*% gammah * L)))
cat("Frontier threshold U:\n")
print(U)

# Check which of our matched TC values (from Exercise 3) exceed U:
on_frontier <- target_correlation_vec > U
cat("Which TC values lie on the frontier (TC > U)?\n")
print(on_frontier)
# OUTCOME: in our case, only the first three imposed TC values are above U
# --> these three designs are ON the frontier; the fourth (largest-index)
#     TC value is NOT on the frontier (see paper for details/interpretation).


#===============================================================================
# 4.2 VERIFY: PRIMAL AND DUAL COINCIDE ON THE FRONTIER
#===============================================================================
# Select any TC value known to lie on the frontier (from Section 4.1).

i <- 1
target_correlation <- target_correlation_vec[i]

# ---- Step 1: invert TC(Tau) to recover Tau from the target TC (Theorem 2) -----
TC  <- tau_from_f(gammah, target_correlation)
tau <- max(TC$tau)

# This Tau should match max_tau_vec0[i], the maximized Tau obtained from the
# DUAL formulation in Exercise 3.2 -- i.e. the two routes to Max-Tau should
# agree numerically on the frontier.
cat("Check: dual Tau vs. inverted-primal Tau on the frontier (should be ~0):\n")
print(tau - max_tau_vec0[i])

# ---- Step 2: plug Tau into the primal to recover the Max-Tau predictor --------
tau_primal_obj <- max_tau_primal_func(gammah, tau)

# Check: the predictor obtained from the (inverted) primal should coincide
# with the predictor obtained from the dual (b_tau_max[, i], from Exercise 3.2).
cat("Check: max abs. difference between primal and dual predictors on the frontier (should be ~0):\n")
print(max(abs(tau_primal_obj$b - b_tau_max[, i])))


#===============================================================================
# 4.3 VERIFY: PRIMAL AND DUAL DIVERGE AWAY FROM THE FRONTIER
#===============================================================================
# Select the TC value known to lie OFF the frontier (from Section 4.1) --
# in our grid, this is the last one.

i <- 4
target_correlation <- target_correlation_vec[i]

# ---- Step 1: invert TC(Tau) to recover Tau from the target TC -----------------
TC  <- tau_from_f(gammah, target_correlation)
tau <- max(TC$tau)

# This Tau does NOT match max_tau_vec0[i] anymore. The inverted-primal value
# here is FINITE, whereas the corresponding dual value max_tau_vec0[i] is
# INFINITE (see Wildi, 2026b, for the underlying explanation of this
# divergence away from the frontier).
cat("Check: dual Tau vs. inverted-primal Tau OFF the frontier (should NOT be ~0):\n")
print(tau - max_tau_vec0[i])

# ---- Step 2: plug Tau into the primal to recover the predictor ----------------
tau_primal_obj <- max_tau_primal_func(gammah, tau)

# Check: the predictors from primal and dual should now DIFFER, since the
# two formulations are no longer equivalent off the frontier.
cat("Check: max abs. difference between primal and dual predictors OFF the frontier (should NOT be ~0):\n")
print(max(abs(tau_primal_obj$b - b_tau_max[, i])))


#===============================================================================
# SUMMARY
#===============================================================================
# The inverted-primal and dual Max-Tau formulations coincide ONLY on the
# efficient frontier. In practice, only the frontier is relevant: any design
# AWAY from the frontier is, by definition, sub-optimal -- it could always
# be outperformed in BOTH target correlation (TC) and Tau simultaneously
# (for instance, by a design ON the frontier). Consequently, the equivalence
# verified in Section 4.2 is the practically important case; the divergence
# in Section 4.3 mainly serves to sharpen the definition of "the frontier"
# itself.


#################################################################################
# EXERCISE 5: CONTROLLING OVERFITTING IN DUAL MAX-TAU THROUGH CURVATURE
#################################################################################
#
# TOPIC:
#   Exercise 3 showed that (dual) Max-Tau can achieve Tau = infinity whenever
#   TC < U -- i.e. whenever the imposed target correlation is too weak
#   relative to the filter length L to keep the design ON the efficient
#   frontier. This is a form of OVERFITTING: for increasing L, the filter can 
#   become arbitrarily concentrated (singular) at the reference frequency omega0,
#   chasing an ever larger (eventually infinite) lead.
#
#   Here we introduce a CURVATURE constraint at omega0 as a second lever
#   (besides TC and L) to control this singularity. This turns the original
#   two-way trade-off (TC vs. Tau) into a three-way TRILEMMA (TC vs. Tau vs.
#   Curvature), defining a higher-dimensional efficient frontier.
#
#====================================================================================
# It is assumed that Exercises 1-4 have been run to initialize all settings!
#====================================================================================


#===============================================================================
# 5.1 SET-UP AND (ORIGINAL) DUAL MAX-TAU -- WITHOUT CURVATURE CONTROL
#===============================================================================

target_correlation <- 0.9

# Default settings for this exercise: reference frequency at the trend, and
# lead measured against the identity (not against gammah).
omega        <- 0
phase_excess <- FALSE

dual_obj <- max_tau_dual_func(gammah, target_correlation, omega0, phase_excess)

b_dual <- dual_obj$b_opt
taub   <- dual_obj$tau_max
cat("Time-shift of (uncurved) dual Max-Tau:\n")
print(taub)

# Compare against the benchmark: the time-shift of the MSE target itself.
tauh <- -gammah %*% (0:(L - 1)) / (gammah %*% rep(1, L))
cat("Time-shift of the MSE target (benchmark):\n")
print(tauh)


#-------------------------------------------------------------------------------
# OUTCOME
#-------------------------------------------------------------------------------
# - Dual Max-Tau dominates by design at the reference frequency omega0 = 0:
#   taub > tauh.
#
# - Effect of increasing L: Max-Tau exploits the additional degrees of
#   freedom to push Tau -> infinity for a GIVEN TC. If L is "small" and TC
#   is "large", Tau = infinity is infeasible. But as L -> infinity, Max-Tau
#   can concentrate its effect arbitrarily tightly at omega0 = 0, so that
#   Tau = infinity ultimately becomes achievable for ANY TC < 1. In other
#   words: in the limit L -> infinity, TC and Tau are no longer in conflict
#   at all -- which is precisely the signature of overfitting.
#
# - An infinite Tau means Max-Tau no longer lies ON the efficient frontier.
#   This happens exactly when TC <= U (see Theorem 2).

# ---- Check: is our current example inside or outside the frontier? -----------
U <- as.double(sqrt(1 - sum(gammah)^2 / (gammah %*% gammah * L)))
cat("Frontier threshold U:\n")
print(U)

cat("Is our target_correlation below U (i.e., off the frontier -> Tau = Inf expected)?\n")
print(target_correlation < U)

# For any TC < U, Tau = infinity.
#
# In this example, L is relatively large (L = 51) while TC is only moderate
# (target_correlation = 0.9), so taub = infinity (in practice: a very large
# finite number) is indeed possible here.
#
# Decreasing L increases U; so decreasing L and/or increasing TC (until
# TC > U) would bring Max-Tau back onto the (lower-dimensional) frontier,
# with a finite Tau. ALTERNATIVELY -- and this is the point of this
# exercise -- one can control this singularity directly by constraining the
# CURVATURE of the filter at omega0, without touching L or TC. Once
# curvature is added, Tau, TC, and Curvature compete simultaneously at the
# optimum: a TRILEMMA, defining a higher-dimensional efficient frontier. In
# that setting, the new curvature-constrained dual Max-Tau (Section 5.2) can
# sit ON its (higher-dimensional) frontier, even while the original,
# uncurved dual Max-Tau (Section 5.1) does NOT sit on its own
# (lower-dimensional) frontier.


#===============================================================================
# 5.2 CURVATURE DUAL MAX-TAU
#===============================================================================

# ---- Decoupling vector for the curvature constraint ----------------------------
k2_vec <- (0:(L - 1))^2

# ---- Curvature of the (uncurved) dual Max-Tau filter, as a reference ---------
# NOTE: the second derivative of the transfer function at omega = 0 is
# (-i)^2 * b_dual %*% k2_vec = -b_dual %*% k2_vec (the leading minus sign
# comes from (-i)^2 = -1), hence the sign flip below.
curvature_max_tau <- as.double(-b_dual %*% k2_vec)
cat("Curvature of the uncurved dual Max-Tau (should be positive):\n")
print(curvature_max_tau)

# ---- Impose a (much) smaller curvature ------------------------------------------
# Setting e_val <- 0 forces the filter to be perfectly FLAT (zero curvature)
# at omega0 = 0 -- the strongest possible curvature constraint. Try, e.g.,
# e_val <- curvature_max_tau / 200 for a softer constraint.
e_val <- 0  # curvature_max_tau / 200

# A few diagnostic quantities for the UNCURVED filter, for later comparison:
cat("Uncurved filter: squared norm b'b:\n");                       print(b_dual %*% b_dual)
cat("Uncurved filter: correlation with gammah (i.e. TC):\n");      print(b_dual %*% gammah / sqrt(sum(gammah * gammah)))
cat("Uncurved filter: curvature b'k2:\n");                         print(b_dual %*% k2_vec)

# ---- Call the curvature-constrained dual Max-Tau solver -------------------------
dual_curvature_obj <- max_tau_dual_curvature(gammah, target_correlation, e_val)

b_dual_curvature <- as.double(dual_curvature_obj$b)

# ---- Time-shift of the curvature-constrained filter -----------------------------
tau_b_dual_curvature <- -b_dual_curvature %*% (0:(L - 1)) / (b_dual_curvature %*% rep(1, L))
cat("Time-shift of the curvature-constrained dual Max-Tau:\n")
print(tau_b_dual_curvature)
# This is no more infinity.

#===============================================================================
# 5.3 OUTPUTS AND VALIDATION CHECKS
#===============================================================================

# ---- Optimal f = Gamma(0) -------------------------------------------------------
# f > 0 means the filter does NOT fully remove trend signals -- in contrast
# to the overfitted (uncurved) Max-Tau, whose f = 0 (it DOES remove trends)
# precisely because tau = infinity there. Conceptually, the curvature dual
# is solved in two stacked steps: optimize Tau given f (inner step), then
# optimize over f (outer step) -- mirroring the inverted-primal-on-the-
# frontier logic from Exercise 4.
cat("Optimal f = Gamma(0) of the curvature-constrained filter:\n")
print(dual_curvature_obj$f)
# The curvature controlled Max-Tau does not remove trend or level information.

# ---- Max-Tau predictor weights ---------------------------------------------------
cat("Curvature-constrained Max-Tau predictor weights:\n")
print(dual_curvature_obj$b)

# ---- Minimum objective (= -Tau) --------------------------------------------------
# The solver minimizes b'k / f, which equals -Tau; so the minimum objective
# corresponds to Max-Tau SUBJECT TO both the TC constraint (as in the
# original dual) AND the additional curvature constraint.
#
#   - If the curvature constraint is binding, the resulting maximized Tau is
#     SMALLER than that of the (overfitted) uncurved dual.
#   - At the optimum, Curvature, TC, and Tau compete with one another --
#     the trilemma described in Section 5.1 -- defining a 3-dimensional
#     efficient frontier.
cat("Minimum objective (-Tau) under the curvature constraint:\n")
print(dual_curvature_obj$objective)

cat("Compare: -objective (curvature-constrained Tau) should be SMALLER than taub (uncurved Tau):\n")
print(taub)

# ---- Validation checks: all four should evaluate to (approximately) zero -------

# a) Gamma(0) equals f, by construction of the stacked optimization
cat("Check (a): Gamma(0) == f (should be ~0):\n")
print(sum(dual_curvature_obj$b) - dual_curvature_obj$f)

# b) Unit-length constraint
cat("Check (b): unit-length constraint (should be ~0):\n")
print(sum(dual_curvature_obj$b^2) - 1)

# c) TC constraint
cat("Check (c): TC constraint (should be ~0):\n")
print(sum(dual_curvature_obj$b * gammah) / sqrt(sum(gammah * gammah)) - target_correlation)

# d) Curvature constraint
cat("Check (d): curvature constraint (should be ~0):\n")
print(sum(dual_curvature_obj$b * k2_vec) - e_val)


#===============================================================================
# 5.4 PLOTS
#===============================================================================

# ---- Compute the shift (time-domain lead) as a function of frequency, for ------
# both the uncurved and the curvature-constrained filters.
K <- 600

obj_dual           <- amp_shift_func(K, b_dual,           FALSE)
shift_dual         <- obj_dual$shift

obj_dual_curvature <- amp_shift_func(K, b_dual_curvature, FALSE)
shift_dual_curvature <- obj_dual_curvature$shift

mplot <- -cbind(shift_dual, shift_dual_curvature)

par(mfrow = c(1, 1))
plot(mplot[1:(K / 10), 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "Shift",
     main = "Shift", ylim = c(-5, 5), col = "red")
lines(mplot[1:(K / 10), 2], col = "blue")
abline(h = 0)
axis(1, at = 1 + 0:6 * K / 60,
     labels = c("0", "pi/60", "2pi/60", "3pi/60", "4pi/60", "5pi/60", "pi/10"))
axis(2)
box()


#-------------------------------------------------------------------------------
# OUTCOME
#-------------------------------------------------------------------------------
# The time-shift of the curvature dual Max-Tau (blue) is SMALLER at the
# reference frequency omega0 = 0, where the uncurved dual (red) dominates by
# design (its shift is capped/truncated in the figure due to near-singular
# behavior) -- but the curvature dual is LARGER at most OTHER frequencies in
# [0, pi/10], where it now dominates the original, uncurved Max-Tau.
#
# As L increases, the uncurved dual Max-Tau (red) becomes increasingly
# singular/concentrated at omega0 = 0, whereas the curvature-constrained
# dual Max-Tau (blue) remains SMOOTH at omega0 = 0 by construction --
# illustrating how the curvature constraint directly controls overfitting.



#################################################################################
# EXERCISE 6: MULTI-FREQUENCY MAX-TAU
#################################################################################
#
# In Exercise 5 (dual Max-Tau), we maximized Tau at ONE trend (zero-) reference 
# frequency omega0=0. This works well at that frequency but says nothing about 
# Tau elsewhere, for example at business-cycle frequencies.
#
# The idea in this exercise: extend optimization to SEVERAL frequencies. 
# This renders overfitting more difficult by `eating up' degrees of 
# freedom at each single frequency.
#
# NOTE (work in progress):
#   - This approach cannot currently be applied at the zero frequency
#     (omega0 = 0). 
#
# PREREQUISITE:
#   Exercises 1-5 must have been run beforehand, since we reuse:
#     - gamma_target        (target filter coefficients)
#     - target_correlation  (target correlation constraint)
#     - L                   (filter length)
#     - b_dual, b_dual_curvature (predictor weights from earlier exercises,
#                                 used here for comparison)
#
#################################################################################


#===============================================================================
# 6.1 SET UP MULTI-FREQUENCY DUAL MAX-TAU AND COMPUTE THE PREDICTOR
#===============================================================================

# ---- Step 1: choose the reference frequencies -------------------------------
# We pick three business-cycle frequencies, corresponding to cycle durations
# of 10, 5 and 2.5 years respectively (recall: duration = 2*pi / omega).
#
#   omega = pi/20  ->  duration = 40 half-years = 10 years
#   omega = pi/10  ->  duration = 20 half-years = 5 years
#   omega = pi/5   ->  duration = 10 half-years = 2.5 years
#
# Technical restriction: omega = 0 is currently NOT supported by the
# underlying optimization (see note above).
omega_vec <- c(pi / 20, pi / 10, pi / 5)

# ---- Step 2: solve the multi-frequency dual Max-Tau problem -----------------
# max_tau_dual_mutiple_freq_func() extends the single-frequency dual Max-Tau
# solver: it finds ONE common f 
#
#     b %*% cos(omega_i * (0:(L-1))) == f     for every omega_i in omega_vec
#
# holds simultaneously across all chosen frequencies, subject to the usual
# TC and unit-length constraints on b. Note that 
#   b %*% cos(omega_i * (0:(L-1))) = Re(Gamma(omega_i))=f

dual_max_tau_mult_obj <- max_tau_dual_mutiple_freq_func(gamma_target,target_correlation,omega_vec)

# ---- Step 3: extract the results ---------------------------------------------
f             <- dual_max_tau_mult_obj$f_opt          # common optimal f in stacked dual
b_dual_mult   <- dual_max_tau_mult_obj$b_opt           # optimal predictor weights
min_objective <- dual_max_tau_mult_obj$min_objective   # value of the objective at optimum


#===============================================================================
# 6.2 VALIDATION CHECKS
#===============================================================================
#
# Before trusting the numerical solution, we verify that all constraints of
# the optimization problem are (numerically) satisfied. Each check below
# should return a value close to zero.

# ---- Check (a): target-correlation (TC) constraint ---------------------------
# The TC between b and gamma_target should match target_correlation.
tc_check <- sum(b_dual_mult * gamma_target) / sqrt(gamma_target %*% gamma_target) -
  target_correlation
cat("TC constraint residual (should be ~0):", tc_check, "\n")

# ---- Check (b): unit-length constraint ----------------------------------------
# The predictor weights must have unit norm: b %*% b == 1.
unit_length_check <- b_dual_mult %*% b_dual_mult - 1
cat("Unit-length constraint residual (should be ~0):", unit_length_check, "\n")

# ---- Check (c): time-shift constraints at each frequency ----------------------
# For every omega_i in omega_vec, the projection of b onto cos(omega_i * lag)
# must equal the common shift value f.
cat("Time-shift constraint residuals (should all be ~0):\n")
for (i in seq_along(omega_vec)) {
  shift_residual <- b_dual_mult %*% cos(omega_vec[i] * (0:(L - 1))) - f
  cat(sprintf("  omega = %.4f  ->  residual = %s\n", omega_vec[i], shift_residual))
}


#===============================================================================
# 6.3 PLOTS
#===============================================================================

# -------------------------------------------------------------------------
# I) Predictor weights
# -------------------------------------------------------------------------
# A quick look at the shape of the new multi-frequency predictor filter.
ts.plot(b_dual_mult,
        main = "Multi-Frequency Dual Max-Tau: Predictor Weights",
        xlab = "Lag",
        ylab = "Weight")


# -------------------------------------------------------------------------
# II) Time-shift comparison across the business-cycle band (+ low frequencies)
# -------------------------------------------------------------------------
# We compare the shift function of THREE predictors:
#   1. b_dual            - single-frequency dual Max-Tau (Exercise 5)
#   2. b_dual_curvature   - curvature-regularized dual Max-Tau (Exercise 5)
#   3. b_dual_mult        - the new multi-frequency dual Max-Tau (this exercise)
#
# amp_shift_func() computes, for a grid of K frequencies, the resulting
# amplitude and shift functions of a given filter. We only need the shift
# component here.

K <- 600   # number of frequency grid points for evaluation

# ---- Compute shift functions for all three predictors -----------------------
obj_dual             <- amp_shift_func(K, b_dual, FALSE)
shift_dual           <- obj_dual$shift

obj_dual_curvature   <- amp_shift_func(K, b_dual_curvature, FALSE)
shift_dual_curvature <- obj_dual_curvature$shift

obj_dual_mult        <- amp_shift_func(K, b_dual_mult, FALSE)
shift_dual_mult      <- obj_dual_mult$shift

# ---- Combine into a single matrix for plotting ------------------------------
# Note the sign flip (negative): this follows the convention used in earlier
# exercises, where a positive "shift" value corresponds to a lag (delay) in
# the plot.
mplot <- -cbind(shift_dual, shift_dual_curvature, shift_dual_mult)

# ---- Plot: restrict to the low-frequency range [0, pi/5] --------------------
# We only plot the first K/5 grid points, i.e. frequencies up to pi/5, since
# this covers the full business-cycle band of interest.
par(mfrow = c(1, 1))

plot(mplot[1:(K / 5), 1],
     type = "l", col = "red",
     axes = FALSE, ylim = c(-5, 5),
     xlab = "Frequency", ylab = "Shift",
     main = "Shift Comparison: Single- vs. Multi-Frequency Dual Max-Tau")
lines(mplot[1:(K / 5), 2], col = "blue")
lines(mplot[1:(K / 5), 3], col = "violet")
abline(h = 0)

# Custom x-axis labeled in units of pi, matching the frequency grid.
axis(1, at = 1 + 0:6 * K / 30,
     labels = c("0", "pi/30", "2pi/30", "3pi/30", "4pi/30", "5pi/30", "pi/5"))
axis(2)
box()

legend("topright",
       legend = c("Dual Max-Tau", "Dual Max-Tau (curvature)", "Dual Max-Tau (multi-freq)"),
       col    = c("red", "blue", "violet"),
       lty    = 1, bty = "n")

# -------------------------------------------------------------------------
# INTERPRETATION
# -------------------------------------------------------------------------
# The multi-frequency predictor (violet) matches the single-frequency
# predictors at omega = pi/20 and OUTPERFORMS them around omega = pi/10 
# and omega = pi/5.



#################################################################################
# EXERCISE 7 MAX-TAU AND THE SELF-SIMILAR AR(1)
#################################################################################
# In contrast to random-walk, AR(1) is subject to mean reversion which can be anticipated more 
# cleverly than through classic zero-shrinkage of MSE by DFP II or by Max-Tau.
