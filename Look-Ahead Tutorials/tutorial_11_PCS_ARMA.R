# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — PCS APPLIED TO ARMA
# ════════════════════════════════════════════════════════════════════

# Overview:
# This tutorial replicates the empirical framework of Tutorial 7 (DFP), 
# but applies the PCS instead of the DFP to obtain look ahead behaviour over
# the classic MSE predictor gamma_h.

# ════════════════════════════════════════════════════════════════════
# Main Take-Aways
# ════════════════════════════════════════════════════════════════════
#
#   1. DIFFICULT FORECAST PROBLEM
#      The ARMA(3,2) process studied here is inherently difficult to forecast.
#      Increasing the forecast horizon (MSE(h_tilde) vs. MSE(h)) does not
#      achieve meaningful look-ahead behaviour: the predictor remains strongly
#      coupled to x_t at lag 0.
#
#
#   2. EXPLOITING NON-UTILISED STRUCTURE IN THE DATA-GENERATING PROCESS
#      A direct comparison of predictor weights illustrates that the DFP
#      exploits structure in the data-generating process that is left unused
#      by the MSE predictor. As decoupling (Exercise 1) or lead (Exercise 2)
#      increases, the predictor (filter) weights become increasingly irregular,
#      reflecting structure in the data-generating process that is
#      otherwise masked by the single dominant AR root.
#
#   3. TREND AND LEVEL INVERSION
#      Aggressive decoupling — for example, pursuing full decoupling — may
#      induce undesirable side effects: inversion of the trend direction or
#      a sign change in constant levels (see Exercise 1). While such aggressive
#      look-ahead behaviour may not be achievable without these effects, trend
#      and level inversions are generally considered undesirable in typical
#      forecasting applications, as they undermine the explainability and
#      interpretability of the predictor output.
#
#   4. TIME-SHIFT DFP CONSTRAINT
#      Expressing the DFP constraint in terms of the zero-frequency lead (exercise 2)
#      provides a natural safeguard against trend and level inversion.
#      Although full decoupling may not always be achievable, the resulting
#      look-ahead dynamics are likely to be sufficient for many practical
#      forecasting applications, offering a favourable balance between
#      timeliness and interpretability.
#
#   5. ATS TRILEMMA
#      A stronger dfp lead generates a left-shift in the filter output but at 
#      the cost of increased noise. This is a direct consequence of the ATS 
#      trilemma in prediction. The MSE predictor represents a single fixed 
#      point on this tradeoff surface: it optimises accuracy alone, ignoring 
#      timeliness (lead) and smoothness (noise suppression) objectives. The 
#      DFP framework can replicate the MSE solution and, beyond that, navigate
#      along the efficient frontier defined by the Accuracy-Timeliness (A-T) 
#      tradeoff; see Wildi (2026), Sections 3.4 and 3.5.
#      Additional tutorials (MDFA and M-SSA) explore alternative aspects of this 
#      fundamental prediction tradeoff.
#
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════


# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)


# ════════════════════════════════════════════════════════════════════
# Exercise 1: MSE-DFP Applied to ARMA(3,2)
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 1.1 Process Specification
# ─────────────────────────────────────────────────────────────────────

# AR coefficients of the ARMA(3,2) process
ar1 <- 0.4
ar2 <- 0.3
ar3 <- 0.2

# MA coefficients of the ARMA(3,2) process
b1 <- 0.5
b2 <- 0.4

# Filter length (number of lags used in the predictor)
L <- 50

# Forecast horizon — kept the same as in Tutorial 6 for comparability
h <- 3

# Compute the Wold (MA-infinity) representation coefficients via ARMAtoMA();
# prepend 1 for the contemporaneous term (lag 0)
xi <- c(1, ARMAtoMA(
  ar      = c(ar1, ar2, ar3),
  ma      = c(b1, b2),
  lag.max = 1000
))

# Plot the Wold decomposition to inspect the impulse-response decay
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition")

# Examine the roots of the AR characteristic polynomial:
#   - Arg(...)/pi gives the cycle length in years (period of oscillation)
#   - abs(...)    gives the modulus; values < 1 confirm stationarity
1 / (Arg(polyroot(c(-ar3, -ar2, -ar1, 1))) / pi)
abs(polyroot(c(-ar3, -ar2, -ar1, 1)))

# The ACF is slowly monotonically decaying: a clear indication 
# of the "stuck at present" problem.
ts.plot(ARMAacf(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=20),main="ACF",
        ylab="",xlab="Lag")

# Work with the MA (Wold) form of the predictors throughout;
# note: the equivalent AR form is derived in exercise 1.4.
gamma <- xi

# Extract L-length coefficient vectors for the nowcast and the h-step forecast
gamma0  <- gamma[1:L]          # nowcast filter  (lag 0 to L-1)
gammah  <- gamma[h + (1:L)]    # h-step MSE forecast filter

# Define a longer reference horizon for comparison purposes (to illustrate 
# "stuck at present" problem).
htilde       <- 20
gammahtilde  <- gamma[htilde + (1:L)]   # htilde-step MSE forecast filter

# Maximum cross-correlation lag (0 = contemporaneous only)
max_lag <- 0

# ── Reference CCF for the h-step MSE predictor ────────────────────────
# Compute the cross-correlation of the h-step MSE predictor output with x_t;
# store for later comparison against DFP designs
cor_vec          <- compute_acf_at_lags_zero_delta_func(max_lag, h, gammah, gamma0)$cor_vec
cor_vec_mat_mse1  <- cor_vec
# Retain CCF at lag 0 (contemporaneous) and at lag h (target horizon)
cor_vec_mse1      <- c(cor_vec[1], cor_vec[1 + h])

# ── Reference CCF for the htilde-step MSE predictor ───────────────────
# Same computation for the longer-horizon MSE predictor
cor_vec          <- compute_acf_at_lags_zero_delta_func(max_lag, h, gammahtilde, gamma0)$cor_vec
cor_vec_mat_mse  <- cbind(cor_vec_mat_mse1, cor_vec)
cor_vec_mse      <- rbind(cor_vec_mse1, c(cor_vec[1], cor_vec[1 + h]))


#───────────────────────────────────────────────────────────────────────────────
# 1.2 PCS Setup: Strong Regulation
#───────────────────────────────────────────────────────────────────────────────

# Forecast horizon.
h <- 12

# Target: the original AR(1) Wold decomposition.
gamma_pcs <- xi

# Constrained lag set:
# Type I PCS imposes a non-negative slope at every lag in Delta, enforcing a
# monotonically increasing CCF (when beta > 0 and the problem is feasible) over
# the full interval {0, ..., h}. This is the most restrictive of the three PCS
# types (I, II, and III).
Delta <- 1:h

# Regularisation weight (penalty on constraint deviation): strong regularisation.
lambda <- 10000

# Constraint slope parameter (negative here to probe the impossible regime).
beta <- -0.0001

b_mat <- NULL

# Compute the Type I PCS predictor.
PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol

beta_vec<-PCS_obj$beta_vec


#───────────────────────────────────────────────────────────────────────────────
# 1.3 Run PCS
#───────────────────────────────────────────────────────────────────────────────

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h, Delta, xi, L, beta, lambda)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}



# Prepend the classical MSE predictor (gamma_0) as a reference.
filter_mat           <- cbind(gamma0, b_mat)
# Beta is scaled by lambda in the column names: to allow readability in plots.
colnames(filter_mat) <- c("MSE",
                          paste("lambda =", round(lambda, 2),
                                ", beta*lambda =", round(beta_vec*lambda, 8)))

# ─────────────────────────────────────────────────────────────────────
# 1.4 Plots
# ─────────────────────────────────────────────────────────────────────

colo <- rainbow(ncol(filter_mat))
par(mfrow = c(1, 2))

# Scale all filters to unit energy for visual comparability.
mplot <- scale(filter_mat, center = FALSE, scale = TRUE)

# Verify filter energies after scaling (should all equal 1).
apply(mplot^2, 2, sum)

# ── Panel 1: Scaled predictor coefficient profiles ────────────────────────────
plot(mplot[, 1],
     main = "Scaled Predictors", axes = FALSE, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

for (i in 2:ncol(mplot)) {
  lines(mplot[, i],
        col = colo[i],
        lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
        lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
lines(mplot[, 2], col = colo[2])   # Redraw second filter on top for visibility.

axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
     labels = c(0, (1:(nrow(mplot) / 10)) * 10))
axis(2)
box()

# ── Panel 2: CCF against xi (Wold decomposition) ──────────────────────────────
# For each predictor, compute the CCF against xi at lags 0, 1, ..., h.
# The dashed vertical line marks the target horizon h; the horizontal line
# marks zero correlation.
max_lag <- 0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat) <- colnames(filter_mat)
rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against xi", axes = FALSE, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i],
        col = colo[i],
        lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
        lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
}

abline(v = 1 + h, lty = 2)   # Vertical marker at target horizon h.
abline(h = 0)                 # Zero-correlation reference line.

axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ─────────────────────────────────────────────────────────────────────
# 1.5 Apply and Compare Predictors
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
#     reflecting the fundamental difficulty of the AR(1) forecasting problem.
#   - We select the leading predictors as well as the MSE benchmark predictor.
#   - All series are standardized to simplify visual inspection.
select_pcs<-2:5
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



mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,1],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])


#───────────────────────────────────────────────────────────────────────────────
# 1.6 Medium Regulation
#───────────────────────────────────────────────────────────────────────────────


# Regularisation weight (penalty on constraint deviation): strong regularisation.
lambda <- 0.1

Delta<-c(h-1,h)


# Constraint slope parameter (negative here to probe the impossible regime).
beta <- -0.0001

b_mat <- NULL

# Compute the Type I PCS predictor.
PCS_obj <- PCS_func(h, Delta, gamma_pcs, L, beta, lambda)

b         <- PCS_obj$b
d_delta   <- PCS_obj$d_delta
b_mat     <- cbind(b_mat, b)
M         <- PCS_obj$M
N         <- PCS_obj$N
gamma_sol <- PCS_obj$gamma_sol

beta_vec<-PCS_obj$beta_vec



#beta_vec<-beta_vec_pcs[1]+(1:10)*(beta_vec_pcs[2]-beta_vec_pcs[1])/10


#───────────────────────────────────────────────────────────────────────────────
# 1.3 Run PCS
#───────────────────────────────────────────────────────────────────────────────

Type_III=T
b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h, Delta, xi, L, beta, lambda,Type_III)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}



# Prepend the classical MSE predictor (gamma_0) as a reference.
filter_mat           <- cbind(gamma0, b_mat)
# Beta is scaled by lambda in the column names: to allow readability in plots.
colnames(filter_mat) <- c("MSE",
                          paste("lambda =", round(lambda, 2),
                                ", beta*lambda =", round(beta_vec*lambda, 8)))

# ─────────────────────────────────────────────────────────────────────
# 3.4 Plots
# ─────────────────────────────────────────────────────────────────────

colo <- rainbow(ncol(filter_mat))
par(mfrow = c(1, 2))

# Scale all filters to unit energy for visual comparability.
mplot <- scale(filter_mat, center = FALSE, scale = TRUE)

# Verify filter energies after scaling (should all equal 1).
apply(mplot^2, 2, sum)

# ── Panel 1: Scaled predictor coefficient profiles ────────────────────────────
plot(mplot[, 1],
     main = "Scaled Predictors", axes = FALSE, type = "l",
     xlab = "Lags", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(mplot), max(mplot)))
mtext(colnames(mplot)[1], col = colo[1], line = -1)

for (i in 2:ncol(mplot)) {
  lines(mplot[, i],
        col = colo[i],
        lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
        lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
lines(mplot[, 2], col = colo[2])   # Redraw second filter on top for visibility.

axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
     labels = c(0, (1:(nrow(mplot) / 10)) * 10))
axis(2)
box()

# ── Panel 2: CCF against xi (Wold decomposition) ──────────────────────────────
# For each predictor, compute the CCF against xi at lags 0, 1, ..., h.
# The dashed vertical line marks the target horizon h; the horizontal line
# marks zero correlation.
max_lag <- 0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat) <- colnames(filter_mat)
rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against xi", axes = FALSE, type = "l",
     xlab = "", ylab = "",
     col  = colo[1], lwd = 1,
     ylim = c(min(0, min(mplot)), max(mplot)))

for (i in 1:ncol(mplot)) {
  lines(mplot[, i],
        col = colo[i],
        lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
        lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
}

abline(v = 1 + h, lty = 2)   # Vertical marker at target horizon h.
abline(h = 0)                 # Zero-correlation reference line.

axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
axis(2)
box()

# ─────────────────────────────────────────────────────────────────────
# 3.5 Apply and Compare Predictors
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
#     reflecting the fundamental difficulty of the AR(1) forecasting problem.
#   - We select the leading predictors as well as the MSE benchmark predictor.
#   - All series are standardized to simplify visual inspection.
select_pcs<-2:ncol(filter_mat)
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



mplot_ccf           <- scale(na.exclude(y_out_mat[, select_vec]))
colnames(mplot_ccf) <- colnames(y_out_mat)[select_vec]
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,1],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[4])



