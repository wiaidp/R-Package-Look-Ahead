
# Address impossibility by changing target







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

h<-12

truncate_at<-L
xi_truncate<-c(xi[1:truncate_at],rep(0,length(xi)-(truncate_at)))

# Define gamma0 and gammah based on truncation
gamma0<-xi_truncate[1:L]
gammah<-xi_truncate[h+1:L]


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

# Moderately large regularisation weight: drives the solution toward 
# satisfaction of all h slope constraints simultaneously, producing a CCF
# that increases almost linearly from k = 0 to k = h with slope
# beta / (b' * b). 
lambda <- 40


# ─────────────────────────────────────────────────────────────────────
# 3.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <- NULL    # filter coefficients, one column per beta value

for (i in seq_along(beta_vec)) {
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj <- PCS_func(h, Delta, xi_truncate, L, beta, lambda)
  
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
                     max_lag, h, filter_mat[, i], xi_truncate)$cor_vec)
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
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda)
  
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
# EXERCISE 5: As Exercise 4 but Type III) PCS based on PCS_func()
# ════════════════════════════════════════════════════════════════════

# New feature: we can address a positive average growth between k=0 and k=h=12.
# This generalizes exercise 1 based on unitary DFP, since we can use PCS_func()
# which allows fine-tuning of the constraint (in contrast to DFP in exercise 1).

# Since we impose a single aggregate constraint (between k=0 and k=h) the PCS constraints 
# are underdetermined and allow for maximization of the target correlation for arbitrarily large lambda. 
# Delta: from k=0 to k=12
Delta <- c(0,12)
# New feature: we must inform PCS_func below that it uses the span k=0 to k=h=12 as specified in Delta
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
  PCS_obj <- PCS_func(h, Delta, gamma, L, beta, lambda,initialize_with_null)
  
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
# All filters are defined in MA form (as applied to the innovations eps_t 
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



# ════════════════════════════════════════════════════════════════════
# EXERCISE 6: As Exercise 3 but Perturbation at smaller lags
# ════════════════════════════════════════════════════════════════════

# Same as exercise 2 but we 



# ─────────────────────────────────────────────────────────────────────
# 6.1 Breaking-Down Structural Singularity 
# ─────────────────────────────────────────────────────────────────────

# Define gamma0 and gammah based on xi: these are used for verifying positiveness of target correlation
# or compute the CCF
gamma0<-xi[1:L]
gammah<-xi[h+1:L]
# Target: original process
gamma_pcs<-xi



# ─────────────────────────────────────────────────────────────────────
# 6.2 PCS Type I): Parameter Setup
# ─────────────────────────────────────────────────────────────────────

# Grid of target slope values to be imposed on the CCF. A positive beta
# implies that the CCF increases regularly from k = 0 to k = h, provided:
#   1) The optimisation problem is feasible, and
#   2) The regularisation weight lambda is sufficiently large to drive the
#      solution close to exact constraint satisfaction.
# Negative or zero values of beta are also included as reference cases to
# illustrate how the CCF profile and peak location respond to the slope target.

h<-12

beta_vec <- c(-0.2, -0.1, -0.01, 0.02, 0.04)
beta_vec <- c(-0.3,-0.2, -0.1, -0.01)

# Constrained lag set: Type I) imposes a positive slope at every lag in Delta, 
# here from 1 to h, enforcing a monotonically increasing CCF (if beta is 
# positive and the problem is feasible) over the full interval
# {0, …, h}. This is the most restrictive of the three PCS types (I, II and III).
Delta <- 1:h
Delta <- 1:2

delta<--10^(-5)
delta<-1


# Case 1
# No perturbation but scaled constraints: feasible but target cor < 0
delta<-0
scaled_constraints<-T

# Case 2
# No perturbation unscaled constraints: unfeasible 
delta<-0
scaled_constraints<-F

# Cases 3&4
# With perturbation: always feasible but target cor can be <0 (note that if delta is small, then lambda must be very large)
delta<-0.00001
scaled_constraints<-F
scaled_constraints<-T




perturbation_delta_mat<-NULL
for (i in 1:length(Delta))
  perturbation_delta_mat<-rbind(perturbation_delta_mat,c(delta,i))
perturbation_delta_mat<-matrix(perturbation_delta_mat,ncol=2)

# Moderately large regularisation weight: drives the solution toward 
# satisfaction of all h slope constraints simultaneously, producing a CCF
# that increases almost linearly from k = 0 to k = h with slope
# beta / (b' * b). 
lambda <- 5000000000


# ─────────────────────────────────────────────────────────────────────
# 6.3 PCS Optimisation over the Slope Grid
# ─────────────────────────────────────────────────────────────────────
# For each beta in beta_vec, compute the regularised PCS predictor using
# criterion (46) from Appendix D of Wildi (2026).

b_mat <-  NULL    # filter coefficients, one column per beta value
initialize_with_null<- F
for (i in seq_along(beta_vec)) { #i<-1
  
  beta <- beta_vec[i]
  
  # Compute PCS Type I) predictor.
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,initialize_with_null,perturbation_delta_mat,scaled_constraints)
  
  
  b       <- PCS_obj$b
  d_delta <- PCS_obj$d_delta
  b_mat   <- cbind(b_mat, b)
  M<-PCS_obj$M
  gamma_sol=PCS_obj$gamma_sol
  
  
  # Constraint check: for a feasible system, the deviation of each slope
  # constraint from its target beta should shrink to zero as lambda -> Inf.
  # Each printed value is the residual for one of the h = 5 constraints.
  # Large lambda means small deviations provided the problem is feasible.
  print(abs(d_delta %*% b + beta))
}

# M does not depend on beta
eigenM<-eigen(M)
# The first two eigenvalues are different from zero
eigenM$values
# Here we see the corresponding eigenvectors
V<-eigenM$vectors
ts.plot(V[,1:2])

V[,1]%*%gamma_sol
V[,2]%*%gamma_sol
k<-5

# Check diagonalization formula
max(abs(M-V%*%diag(eigenM$values)%*%t(V)))
# Inverse
max(abs(solve(M)-V%*%diag(1/eigenM$values)%*%t(V)))
# So solve(M)=V%*%diag(1/eigenM$values)%*%t(V)
# Now solve(M) is applied to gamma_sol i.e. b=solve(M)%*%gamma_sol

# Most weight is assigned to first two eigenvectors: negative and positive weights
diag(1/eigenM$values)%*%t(V)%*%gamma_sol

b<-V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol
ts.plot(b)

# First two eigenvectors
ts.plot(V[,1:2])


colnames(b_mat) <- paste0("lambda=", lambda, ", beta=", round(beta_vec, 3))


# ─────────────────────────────────────────────────────────────────────
# 6.4 Routine Checks
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
# 6.5 Plots and Performance Summary
# ─────────────────────────────────────────────────────────────────────

par(mfrow = c(1, 2))
colo <- c("black", "green", rainbow(ncol(b_mat)))

# ── Left panel: filter coefficients ──────────────────────────────────
mplot <- scale(filter_mat,scale=T,center=F)
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
                     max_lag, h, filter_mat[, i],xi)$cor_vec)
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



