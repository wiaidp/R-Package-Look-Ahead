
# Most difficult: self-similar AR(1)



# ── EXAMPLES OVERVIEW ─────────────────────────────────────────────────────────

# Example 1 — Impossibility 


# Example 2 — perturbtaion: looks lik ARMA(1,1)

# Examples 3– - AR(1) perturbation: cyclicality in a completely acacyclical framework

# Example 4 - AR(2) perturbation

# Example 5 - changing the target



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

# Fit an ARMA(1,1) model: a parsimonious specification with adequate diagnostics.
ar_order <- 1
ma_order <- 1

arima.obj <- arima(x, order = c(ar_order, 0, ma_order))
tsdiag(arima.obj)
a1 <- arima.obj$coef[1:ar_order]
b1 <- arima.obj$coef[ar_order + 1:ma_order]

# --- Wold Decomposition (MA-infinity representation) ---
# Compute the infinite-order MA coefficients (impulse response weights)
# of the fitted ARMA model. The filter length L ensures that the
# coefficients have decayed sufficiently close to zero by lag L.
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

# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: slowly decaying impulse response (post-1990)")

# The theoretical ACF implied by the Wold decomposition matches the
# empirical ACF computed above.
ts.plot(ARMAacf(ar = 0, ma = xi, lag.max = L),
        main = "Model-based ACF", ylab = "", xlab = "Lag")

# For k > 0 the ACF satisfies the recurrence ACF(k+1) = a1 * ACF(k), so
# the DGP imposes a rigid linear structure on the autocorrelation function.
#
# For h > 0 and k >= 0 this implies gamma_h ∝ gamma_{h+k}: the MSE predictor
# coefficient vectors are mutually proportional for all horizons h > 0.
#
# Consequently, the column space of the constraint system has rank 2:
# (gamma_0 - gamma_1) and (gamma_1 - gamma_2) ∝ gamma_1 are the only
# linearly independent directions. Because the Type III PCS enforces only a
# single constraint, the problem remains feasible. Unfortunately, the CCF
# cannot peak at h = 12 — this is structurally impossible for the
# ARMA(1,1) DGP — so the problem is classified as impossible but feasible.



# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: Rank-One 
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 1.1 AR(1)
# ─────────────────────────────────────────────────────────────────────

L <- 50   # filter length (number of MA coefficients retained)

# Fit an ARMA(1,1) model: a parsimonious specification with adequate diagnostics.
a1 <- 0.9

xi <- c(1, ARMAtoMA(ar= a1, ma=0,lag.max = 1000))


# Visualise the Wold coefficients.
par(mfrow = c(1, 1))
ts.plot(xi, main = "Wold decomposition: slowly decaying impulse response (post-1990)")

ts.plot(ARMAacf(ar = a1, lag.max = L),
        main = "ACF", ylab = "", xlab = "Lag")

# The ACF satisfies the recurrence ACF(k+1) = a1 * ACF(k).
# The rank is one: gamma_h is proportional to gamma_{h+k} for any h,k>=0.


# ─────────────────────────────────────────────────────────────────────
# 1.2 PCS
# ─────────────────────────────────────────────────────────────────────

# Forecast horizon
h<-12

# Target: original process
gamma_pcs<-xi

# Constrained lag set: Type I) imposes a positive slope at every lag in Delta, 
# here from 1 to h, enforcing a monotonically increasing CCF (if beta is 
# positive and the problem is feasible) over the full interval
# {0, …, h}. This is the most restrictive of the three PCS types (I, II and III).
Delta <- 1:h


# Regularization weight
lambda <- 10000





beta <--0.0001

# Compute PCS Type I) predictor.
PCS_obj<-PCS_func(h,Delta, gamma_pcs, L, beta, lambda)




b       <- PCS_obj$b
d_delta <- PCS_obj$d_delta
b_mat   <- cbind(b_mat, b)
M<-PCS_obj$M
N<-PCS_obj$N
gamma_sol=PCS_obj$gamma_sol

# ─────────────────────────────────────────────────────────────────────
# 1.3 Background: Some Linear Algebra
# ─────────────────────────────────────────────────────────────────────
# The closed-form formula for PCS is: b <- solve(M) %*% gamma_sol
# M depends on lambda but not on beta.
# gamma_sol depends on lambda and beta.

# Check: difference vanishes:
max(abs(b-solve(M) %*% gamma_sol))

# M is symmetric and can be diagonalized
# Check:
eigenM<-eigen(M)
V<-eigenM$vectors
# Check diagonalization formula: difference should vanish
max(abs(M-V%*%diag(eigenM$values)%*%t(V)))
# Inverse
max(abs(solve(M)-V%*%diag(1/eigenM$values)%*%t(V)))
# So solve(M)=V%*%diag(1/eigenM$values)%*%t(V)
# Now solve(M) is applied to gamma_sol.
# gamma_sol is the weighted linear combination of gamma_h and the constraints.
# In the AR(1) case gamma_h and the constraints are all linear dependent (Rank One)
ts.plot(gamma_sol)
# gamma_sol is again AR(1) with exponential a1 decay:
gamma_sol[2:L]/gamma_sol[1:(L-1)]

# M=I+lambda*N where N=sum_{k=1}^h (gamma_k-gamma_{k-1}) (gamma_k-gamma_{k-1})'
# Check: difference vanishes:
max(abs(M-diag(rep(1,L))-lambda*N))
# Since (gamma_k-gamma_{k-1}) are AR(1) for all k, the L*L matrix N has rank one:
eigenN<-eigen(N)
# Only one eigenvalue larger than 10^-10
which(abs(eigenN$values)>10^(-10))
# Lets have a look at its eigenvector:
ts.plot(N[,1], main="Eigenvector of non-vanishing eigenvalue of N")
# Some basic results:
# -Eigenvalues of M=I+lambda*N are 1+lambda*n_i where n_i are eigenvalues of N.
# -Eigenvalues of M^{-1} are 1/(1+lamba*n_i).
# -Eigenvectors of M are the same as eigenvectors of N.
# -Rank(N)=1, Rank(M)=L
# Note: the following does not vanish because the orderings of the eigenvectors are different
max(abs(V-eigenN$vectors))
eigenM$values
ts.plot(V[,1],main="First eigenvector of M")
# V is orthogonal and gamma_sol is proportional to V[,1]. Therefore V[,k]%*%gamma_sol=0
# Check:
k<-2
# Vanishes if k>1
V[,k]%*%gamma_sol
# Equivalently (since V is symmetric)
t(V)[k,]%*%gamma_sol
# Only the first element in the following vector is different from zero:
t(V)%*%gamma_sol
# Same here
g<-(diag(1/eigenM$values)%*%t(V)%*%gamma_sol)
g

# By the above b=solve(M)%*%gamma_sol and solve(M) = V%*%diag(1/eigenM$values)%*%t(V)
# Since g:=diag(1/eigenM$values)%*%t(V)%*%gamma_sol has only the first element that does not vanish we infer 
# that V%*%g = g[1] * V[,1]
# Check:
abs(max(V%*%g-g[1]*V[,1]))

# We conclude that b=V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol must be proportional to V[,1], 
# the first eigenvetor of M, which is AR(1).
# Check:

b<-V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol
ts.plot(b)
# the PCS predictor is AR(1):
b[2:L]/b[1:(L-1)]
# This holds irrespective of lambda.

# ─────────────────────────────────────────────────────────────────────
# 1.4 Impossibility
# ─────────────────────────────────────────────────────────────────────

# It is impossible to shift the peak of the CCF to k>0 in the AR(1) case.
# The process has rank one, all MSE predictors gamma_h are proportional to gamma_0.
# Therefore t(V)%*%gamma_sol is vanishing except at its first entry; similarly for g.
# Therefore b = V%*%g is proportional to V[,1] which is again AR(1).
# This holds irrespective of lambda>0.

# Without increasing the rank of the problem, shifting the CCF peak is impossible.




# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: Increasing the Rank: a Perturbation Based Approach
# ════════════════════════════════════════════════════════════════════

# Introduce a single perturbation delta at lag 0
# This expands the rank from 1 to two provided gamma0 enters the constraints 
# (otherwise the perturbation is ignored). 

# ─────────────────────────────────────────────────────────────────────
# 2.1 
# ─────────────────────────────────────────────────────────────────────

# Forecast horizon
h<-12

# Introduce a single perturbation delta at lag 0
# First unit vector
e1<-c(1,rep(0,length(xi)-1))
delta<-0.0001
perturbation_vec<-delta*e1
xi_perturbated<-xi+perturbation_vec


# Target: original process
gamma_pcs_perturbated<-xi_perturbated

gamma0_perturbated<-xi_perturbated[1:L]
gamma1_perturbated<-xi_perturbated[1+1:L]

gamma1_perturbated[2:L]/gamma1_perturbated[1:(L-1)]
gamma2_perturbated<-xi_perturbated[2+1:L]

# The first two constraints of the PCS system are given by
# b' * (gamma_1-gamma_0) =beta
# b' * (gamma_2-gamma_1) =beta
# We here compute the first two delta_i=(gamma_i-gamma_{i-1}) based on 
# the perturbated system:
delta1<-gamma0_perturbated-gamma1_perturbated
delta2<-gamma1_perturbated-gamma2_perturbated

# For i>2 delta_i is proportional to delta_{i-1} since the perturbation affects only the first lag of xi (at k=0)

# delta1 and delta2 are spanned by gamma0 (the original AR(1)) and perturbation_vec
# Check:
gamma0<-xi[1:L]
# delta1 depends on gamma0 and perturbation_vec 
summary(lm(delta1~gamma0+perturbation_vec[1:L]-1))
# delta2 depends only on gamma0 (the perturbation affects only the first lag)
summary(lm(delta2~gamma0-1))


# Constrained lag set: Type I) imposes a positive slope at every lag in Delta, 
# here from 1 to h, enforcing a monotonically increasing CCF (if beta is 
# positive and the problem is feasible) over the full interval
# {0, …, h}. This is the most restrictive of the three PCS types (I, II and III).
Delta <- 1:h


# Regularization weight
lambda <- 10000000





beta <--0.0001

# Compute PCS Type I) predictor.
PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)




b       <- PCS_obj$b
d_delta <- PCS_obj$d_delta
b_mat   <- cbind(b_mat, b)
M<-PCS_obj$M
N<-PCS_obj$N
gamma_sol=PCS_obj$gamma_sol


# ─────────────────────────────────────────────────────────────────────
# 2.2 Rank Two
# ─────────────────────────────────────────────────────────────────────
# The closed-form formula for PCS is: b <- solve(M) %*% gamma_sol
# In contrast to exercise 1, gamma_sol is not perfectly AR(1) anymore
gamma_sol[2:L]/gamma_sol[1:(L-1)]

# gamma_sol is the linear combination of gamma0 (the original non-perturbated AR(1)) and perturbation vector
# Perfect fit: gamma0 and perturbation_vec are significant and the residual vanishes.
# Note: the size of delta affects the significance.
summary(lm(gamma_sol~perturbation_vec[1:L]+gamma0-1))


eigenM<-eigen(M)
V<-eigenM$vectors
eigenN<-eigen(N)


# In contrast to exercise 1, the rank is two: two eigenvalues larger than 10^-10
which(abs(eigenN$values)>10^(-10))

# The columns space of N is two-dimensional: it is spanned by delta1 and delta2 computed above.


# Lets have a look at the two eigenvectors:
par(mfrow=c(1,1))
ts.plot(eigenN$vectors[,1:2], main="Eigenvectors of non-vanishing eigenvalues of N",lty=1:2)

# The first two elements in the following vector are different from zero:
t(V)%*%gamma_sol
# Same here
g<-(diag(1/eigenM$values)%*%t(V)%*%gamma_sol)
g

# By the above b=solve(M)%*%gamma_sol and solve(M) = V%*%diag(1/eigenM$values)%*%t(V)
# Since g:=diag(1/eigenM$values)%*%t(V)%*%gamma_sol has only the first two elements that do not vanish we infer 
# that V%*%g = g[1] * V[,1] + g[2] * V[,2]
# Check:
abs(max(V%*%g-g[1]*V[,1]-g[2]*V[,2]))

# We conclude that b=V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol must be a linear combination 
# of V[,1], V[,2] or, equivalently, a linear combination of gamma 0 and perturbation_vec
# Check: 
b<-V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol
ts.plot(b)
# the PCS predictor is not AR(1) anymore:
b[2:L]/b[1:(L-1)]
# Verify the decomposition of b into gamma0 and perturbation_vec: perfect fit (up to numerical precision)
summary(lm(b~gamma0+perturbation_vec[1:L]-1))

# ─────────────────────────────────────────────────────────────────────
# 2.3 Playing with Rank Two System: Strong Regularization
# ─────────────────────────────────────────────────────────────────────
# The PCS predictor is a linear combination of gamma0 and perturbation_vec and 
# the weights of the linear combination can be tuned by beta and lambda.


# We fix lambda to a very strong regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# second eigenvector V[,2] with combinations -V[,1]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.
# For beta~0.000000269 one obtains -V[,1], i.e., lambda1=0.



# Very strong regularization
lambda<-5000000

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]
beta_vec<-c(-1,-0.1,0,0.0000001,0.0000002,0.00000025,0.000000269,0.0000003,0.0000005, 0.00001)

Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{

  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)

  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8))


# ─────────────────────────────────────────────────────────────────────
# 2.4 Plots
# ─────────────────────────────────────────────────────────────────────


colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against xi", axes = F, type = "l",
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




max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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


# CCF against V1 and V2:
# b is a linear combination of both.
# b is also linear combination of gamma0 and perturbation
# V[,1:2] orthogonal
# If V[,i]=gamma0 then original CCF against gamma0 measures the gamma0-effect only and discards perturbation.
# Here V[,1] = - gamma0 so that CCF is same as against gamma0 but sign inverted.
# Decompose CCF additively into effect of V[,1] (gamma0) and V[,2] (perturbation+gamma0)
# This is better than CCF against perturbation because perturbation and gamma0 are not orthogonal: cannot decompose CCF additively.
# We can see the PCS effect on CCF of V[,2]: a shift by one to the right.

# ─────────────────────────────────────────────────────────────────────
# 2.5 Playing with Rank Two System: Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# Medium regularization
lambda<-5
# Tipping points: the two extremes are -V[,1] and +V[,1]
beta_vec<-c(0,0.086,0.0874,0.08745,0.08746,0.08747,0.0875,0.088,0.09)

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_func(h,Delta, gamma_pcs_perturbated, L, beta, lambda)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}

filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,4))


# ─────────────────────────────────────────────────────────────────────
# 2.6 Plots
# ─────────────────────────────────────────────────────────────────────



colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

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

max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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

# Note: 
# 1. Instead of using lambda and beta to parameterize the PCS predictors, the above perturbated PCS predictors could 
# be obtained by gamma0+lambda_1*perturbation_vec[1:L] and -gamma0+lambda_1*perturbation_vec[1:L]
# where lambda1 is a real (positive or negative) number.
# 2. The results do not depend on the size delta of the perturbation in the sense that the same 
#   solution space is obtained irrespective of delta. Of course, lambda and beta (or lambda1) must 
#   be recalibrated, but the space remains the same.



# Note that we could increase the rank by adding e2, e3,....
# However, this would not affect the CCF against the true gamma0 (AR(1)).


# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: ALTERNATIVE AR(1) PERTURBATION
# ════════════════════════════════════════════════════════════════════

# Introduce a single perturbation affecting all lags.
# This expands the rank from 1 to two for any constraints (in conztrast to exercise 2, 
# where gamma0 must be part of the constraints) 


# ─────────────────────────────────────────────────────────────────────
# 3.1 
# ─────────────────────────────────────────────────────────────────────

# Construct the MSE predictors gamma_i used for deriving delta_i=gamma_i-gamma_{i-1} 

gamma_all <- xi
# --- Build the shifted covariance matrix 'gammah_mat' ---
# Each row contains the MSE predictor coefficients (gamma_all) shifted by
# a specific lead value drawn from 'Delta'. 
# We start with Delta[1] - 1 because we compute differences: gamma_h-gamma_{h-1}
# and therefore we need gamma_{Delta[1] - 1} to define the first difference.
gammah_mat <- gamma_all[Delta[1] - 1 + 1:L]/sqrt(sum(gamma_all^2) ) 
if (length(Delta) > 0)
{
  for (i in 1:length(Delta))
  {
    gammah_mat <- rbind(gammah_mat,
                        gamma_all[Delta[i] + 1:L]/sqrt(sum(gamma_all^2) ) )
  }
}

# Perturbate a1
delta<-0.0001
a1_perturbate<-a1+delta

xi_a1_perturbate <- c(1, ARMAtoMA(ar= a1_perturbate, ma=0,lag.max = 1000))

gamma_all_a1_perturbate <- xi_a1_perturbate


ts.plot(xi-xi_a1_perturbate)

gammah_mat_perturbate<- gammah_mat

gammah_mat_perturbate[1,]<-gamma_all_a1_perturbate[1:L]/sqrt(sum(gamma_all_a1_perturbate^2)) 


lambda<-10000

PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  



b       <- PCS_obj$b
d_delta <- PCS_obj$d_delta
b_mat   <- cbind(b_mat, b)
M<-PCS_obj$M
N<-PCS_obj$N
gamma_sol=PCS_obj$gamma_sol

# ─────────────────────────────────────────────────────────────────────
# 3.2 Background: Some Linear Algebra
# ─────────────────────────────────────────────────────────────────────
# The closed-form formula for PCS is: b <- solve(M) %*% gamma_sol
# M depends on lambda but not on beta.
# gamma_sol depends on lambda and beta.

ts.plot(gamma_sol)
# gamma_sol is not AR(1): the decay is not exponential with fixed a1:
gamma_sol[2:L]/gamma_sol[1:(L-1)]

eigenM<-eigen(M)
V<-eigenM$vectors


# M=I+lambda*N where N=sum_{k=1}^h (gamma_k-gamma_{k-1}) (gamma_k-gamma_{k-1})'
# Check: difference vanishes:
max(abs(M-diag(rep(1,L))-lambda*N))
# N does not have rank one but two
eigenN<-eigen(N)
# Only two eigenvalue larger than 10^-10
which(abs(eigenN$values)>10^(-10))
# Lets have a look at the two eigenvectors of the non-vanishing eigenvalues:
# Note: 
#   The eigenvectors depend on the constraint-matrix only: they are independent of lambda or beta
#   The eigenvalues depend on lambda, but not on beta
#   gamma_sol depends on lambda*beta
par(mfrow=c(1,1))
ts.plot(eigenN$vectors[,1:2], main="Eigenvectors of non-vanishing eigenvalues of N")
# Some basic results:
# -Eigenvalues of M=I+lambda*N are 1+lambda*n_i where n_i are eigenvalues of N.
# -Eigenvalues of M^{-1} are 1/(1+lambda*n_i).
# -Eigenvectors of M are the same as eigenvectors of N.
# -Rank(N)=2, Rank(M)=L
par(mfrow=c(1,1))
ts.plot(V[,1:2],main="First two eigenvectors of M")
# V is orthogonal, gamma_sol is in the column space of the first two eigenvectors V[,1:2]. 
# Therefore V[,k]%*%gamma_sol=0 if k>2.
# Check:
# Only the first element in the following vector is different from zero:
t(V)%*%gamma_sol
# Same here
g<-(diag(1/eigenM$values)%*%t(V)%*%gamma_sol)
g

# By the above b=solve(M)%*%gamma_sol and solve(M) = V%*%diag(1/eigenM$values)%*%t(V)
# Since g:=diag(1/eigenM$values)%*%t(V)%*%gamma_sol has only the first element that does not vanish we infer 
# that V%*%g = g[1] * V[,1] + g[2] * V[,2]
# Check:
abs(max(V%*%g-g[1]*V[,1]-g[2]*V[,2]))

# We conclude that b=V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol lies in the space spanned by V[,1] and V[,2] 
# or xi[1:L] and xi_a1_perturbate[1:L].
# The PCS predictor is a linear combination of V[,1] and V[,2]

b<-V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol
ts.plot(b)
# the PCS predictor is not AR(1) in general (though it could be as a special case):
b[2:L]/b[1:(L-1)]
# This holds irrespective of lambda.

# ─────────────────────────────────────────────────────────────────────
# 3.3 Play the Expanded Rank-Game: Strong Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a very strong regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# second eigenvector V[,2] with combinations -V[,1]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.
# For beta~0.000000269 one obtains -V[,1], i.e., lambda1=0.



# Very strong regularization
lambda<-5000000

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]
beta_vec<-c(-1,0,0.0000001,0.0000002,0.00000025,0.00000026,0.000000265,0.000000266,0.000000267,0.000000269,0.0000003,0.0000005, 0.00001)

Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8))


# ─────────────────────────────────────────────────────────────────────
# 3.4 Plots
# ─────────────────────────────────────────────────────────────────────


colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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





# As in exercise 2.3, the PCS predictor is a linear combination of gamma0 (AR(1)) and 
# gamma0_perturbated or, alternatively, of V[,1] and V[,2].
# In contrast to exercise the rank is two.
# In contrast to exercise 2, the perturbation does not affect lag 0 only, but also 
#   all other lags.
# The result is a seemingl richer structure of the solution space, allowing, among others, 
# a sort of cyclical pattern in a purely aperiodic framework (with monotonically decaying weights).



# ─────────────────────────────────────────────────────────────────────
# 3.5 Medium Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a very strong regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# second eigenvector V[,2] with combinations -V[,1]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.
# For beta~0.000000269 one obtains -V[,1], i.e., lambda1=0.



# Medium regularization
lambda<-5

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]
beta_vec<-c(0,0.086,0.0872,0.0874,0.08745,0.08746,0.08747,0.0875,0.0877,0.088,0.09)


Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8))


# ─────────────────────────────────────────────────────────────────────
# 3.6 Plots
# ─────────────────────────────────────────────────────────────────────


colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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




# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: ALTERNATIVE AR(2) PERTURBATION
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 4.1 
# ─────────────────────────────────────────────────────────────────────

# Construct the MSE predictors gamma_i used for deriving delta_i=gamma_i-gamma_{i-1} 

gamma_all <- xi
# --- Build the shifted covariance matrix 'gammah_mat' ---
# Each row contains the MSE predictor coefficients (gamma_all) shifted by
# a specific lead value drawn from 'Delta'. 
# We start with Delta[1] - 1 because we compute differences: gamma_h-gamma_{h-1}
# and therefore we need gamma_{Delta[1] - 1} to define the first difference.
gammah_mat <- gamma_all[Delta[1] - 1 + 1:L]/sqrt(sum(gamma_all^2) ) 
if (length(Delta) > 0)
{
  for (i in 1:length(Delta))
  {
    gammah_mat <- rbind(gammah_mat,
                        gamma_all[Delta[i] + 1:L]/sqrt(sum(gamma_all^2) ) )
  }
}

# Specify a periodic AR(2)
a1_ar2<-1.81381 
a2_ar2<--0.8291025 
xi_ar2_all <- c(1, ARMAtoMA(ar= c(a1_ar2,a2_ar2), ma=0,lag.max = 2000))
k_start<-20
k_start<-0
xi_ar2<-xi_ar2_all[k_start+1:1001]

gamma_all_ar2 <- xi_ar2

par(mfrow=c(1,1))
ts.plot(cbind(xi,xi_ar2),col=c("black","red"))

gammah_mat_perturbate_ar2<- gammah_mat

gammah_mat_perturbate_ar2[1,]<-gamma_all_ar2[1:L]/sqrt(sum(gamma_all_ar2^2)) 




PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)




b       <- PCS_obj$b
d_delta <- PCS_obj$d_delta
b_mat   <- cbind(b_mat, b)
M<-PCS_obj$M
N<-PCS_obj$N
gamma_sol=PCS_obj$gamma_sol

# ─────────────────────────────────────────────────────────────────────
# 4.2 Background: Some Linear Algebra
# ─────────────────────────────────────────────────────────────────────
# The closed-form formula for PCS is: b <- solve(M) %*% gamma_sol
# M depends on lambda but not on beta.
# gamma_sol depends on lambda and beta.

ts.plot(gamma_sol)
# gamma_sol is not AR(1): the decay is not exponential with fixed a1:
gamma_sol[2:L]/gamma_sol[1:(L-1)]

eigenM<-eigen(M)
V<-eigenM$vectors


# M=I+lambda*N where N=sum_{k=1}^h (gamma_k-gamma_{k-1}) (gamma_k-gamma_{k-1})'
# Check: difference vanishes:
max(abs(M-diag(rep(1,L))-lambda*N))
# N does not have rank one but two
eigenN<-eigen(N)
# Only two eigenvalues larger than 10^-10
which(abs(eigenN$values)>10^(-10))
# Lets have a look at the two eigenvectors of the non-vanishing eigenvalues:
par(mfrow=c(1,1))
ts.plot(eigenN$vectors[,1:2], main="Eigenvectors of non-vanishing eigenvalues of N")
# Some basic results:
# -Eigenvalues of M=I+lambda*N are 1+lambda*n_i where n_i are eigenvalues of N.
# -Eigenvalues of M^{-1} are 1/(1+lamba*n_i).
# -Eigenvectors of M are the same as eigenvectors of N.
# -Rank(N)=2, Rank(M)=L
ts.plot(V[,1:2],main="First two eigenvectors of M",lty=1:2)
# V is orthogonal, gamma_sol is in the column space of the first two eigenvectors V[,1:2]. 
# Therefore V[,k]%*%gamma_sol=0 if k>2.
# Check:
# Only the first element in the following vector is different from zero:
t(V)%*%gamma_sol
# Same here
g<-(diag(1/eigenM$values)%*%t(V)%*%gamma_sol)
g

# By the above b=solve(M)%*%gamma_sol and solve(M) = V%*%diag(1/eigenM$values)%*%t(V)
# Since g:=diag(1/eigenM$values)%*%t(V)%*%gamma_sol has only the first element that does not vanish we infer 
# that V%*%g = g[1] * V[,1] + g[2] * V[,2]
# Check:
abs(max(V%*%g-g[1]*V[,1]-g[2]*V[,2]))

# We conclude that b=V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol lies in the space spanned by V[,1] and V[,2] 
# or xi[1:L] and xi_a1_perturbate[1:L].
# The PCS predictor is a linear combination of V[,1] and V[,2]

b<-V%*%diag(1/eigenM$values)%*%t(V)%*%gamma_sol
ts.plot(b)
# the PCS predictor is not AR(1) in general (though it could be as a special case):
b[2:L]/b[1:(L-1)]
# This holds irrespective of lambda.

# ─────────────────────────────────────────────────────────────────────
# 4.3 Play the Expanded Rank-Game: strong regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a very strong regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# second eigenvector V[,2] with combinations -V[,1]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.
# For beta~0.000000269 one obtains -V[,1], i.e., lambda1=0.



# Very strong regularization
lambda<-5000000

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]
beta_vec<-c(-1,-0.1,0,0.00000005,0.00000008,0.00000009,0.0000001,0.00000013,0.0000002,0.000003,0.1)

Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8))


# ─────────────────────────────────────────────────────────────────────
# 4.4 Plots
# ─────────────────────────────────────────────────────────────────────



colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against AR(1)", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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



# Note
# -The CCF is evaluated against the true AR(1) DGP, i.e., xi:

# b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi%*%xi)).

# -Consider that the following slight modification 
#     b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   would be fixed since xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   is constant (not dependent on i if xi is the AR(1) DGP).
# -However, xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) is proportional to a^i i.e. decreases exponentially.

# Conclusions:
# 1. The observed decrease of the CCF is only due to the scaling effect in xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) 
#     and corresponds to a^i: all CCF's in the right panel decay with a^i.
# 2. It is not possible to have a locally increasing CCF except through sign inversion (impossibility and infeasibility)
# 3. In the original AR(2)-case (Tutorial 13) the peak of the CCF could be shifted because xi corresponded to the AR(2),i.e., one could rely on phase effect.
#     But here xi is AR(1): no phase effect. As a result, even the AR(2)-perturbation is unable to shift the peak.





# Against the AR(2) benchmark the PCS shifts the peak forward as intended.


# ─────────────────────────────────────────────────────────────────────
# 4.5 Play the Expanded Rank-Game: medium Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a very strong regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# second eigenvector V[,2] with combinations -V[,1]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.
# For beta~0.000000269 one obtains -V[,1], i.e., lambda1=0.



# Very strong regularization
lambda<-5

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]
beta_vec<-c(-1,0,0.0000001,0.0000002,0.00000025,0.00000026,0.000000265,0.000000266,0.000000267,0.000000269,0.0000003,0.0000005, 0.00001)

beta_vec<-c(0,0.086,0.0874,0.08745,0.08746,0.08747,0.0875,0.088,0.09)

beta_vec<-c(-1,-0.2,0,0.05,0.07,0.09)


Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-b_mat
colnames(filter_mat)<-paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8))



# ─────────────────────────────────────────────────────────────────────
# 4.6 Plots
# ─────────────────────────────────────────────────────────────────────


colo<-rainbow(ncol(filter_mat))
par(mfrow=c(2,2))
mplot <- scale(filter_mat,center=F,scale=T)
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
max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], xi)$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against AR(1)", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,1])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V1", axes = F, type = "l",
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


max_lag<-0
ccf_mat <- NULL
for (i in 1:ncol(filter_mat))
  ccf_mat <- cbind(ccf_mat,
                   compute_acf_at_lags_zero_delta_func(
                     max_lag, h, filter_mat[, i], V[,2])$cor_vec)
colnames(ccf_mat)<-colnames(filter_mat)
rownames(ccf_mat)<-paste("CCF at lead: ",-max_lag-1+1:nrow(ccf_mat),sep="")

mplot <- ccf_mat

plot(mplot[, 1],
     main = "CCF against V2", axes = F, type = "l",
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

# Note
# -The CCF is evaluated against the true AR(1) DGP, i.e., xi:

# b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi%*%xi)).

# -Consider that the following slight modification 
#     b[1:min(L,L_gamma-i)]%*%xi[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   would be fixed since xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi[i+(1:min(L,L_gamma-i))]%*%xi[i+(1:min(L,L_gamma-i))]))
#   is constant (not dependent on i if xi is the AR(1) DGP).
# -However, xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) is proportional to a^i i.e. decreases exponentially.

# Conclusions:
# 1. The observed decrease of the CCF is only due to the scaling effect in xi[i+(1:min(L,L_gamma-i))]/(sqrt(xi%*%xi)) 
#     and corresponds to a^i: all CCF's in the right panel decay with a^i.
# 2. It is not possible to have a locally increasing CCF except through sign inversion (impossibility and infeasibility)
# 3. In the original AR(2)-case (Tutorial 13) the peak of the CCF could be shifted because xi corresponded to the AR(2),i.e., one could rely on phase effect.
#     But here xi is AR(1): no phase effect. As a result, even the AR(2)-perturbation is unable to shift the peak.






# Main Take-Aways

# -The AR(1) forecast problem is self-similar and one-dimensional and the PCS problem is impossible and infeasible.
# -Perturbating the DGP allows to expand the column-space of the PCS constraint system.
# -While the size of the perturbation is irrelevant, the shape is relevant.
# -We analyzed three sorts of perturbation:
# 1. delta-type at lag 0 (this is similar to the structure of an ARMA(1,1) with a very small MA(1)-term when delta is small)
# 2. AR(1)-type: the perturbation spreads over all lags according to a slightly modified AR(1) parameter.
# 3. AR(2)-type: we selected a periodic AR(2). In constrast to perturbations 1 and 2,  the perturbation here is sizeable not only 
#    in magnitude but also in shape.

# -In all considered cases the rank increased from 1 to two. The PCS solution lies in the space spanned 
#     by the original AR(1) and the perturbation vector (the first two eigenvectors of the constraint matrix corresponding to the two non-vanishing eigenvalues): either delta (e1), modified AR(1) or AR(2).
# -Changing lambda and beta allows to navigate in these spaces. Alternatively, one could just 
#  rely on classic linear weighting of the two eigenvectors.
# Multiple perturbations could increase the rank of the constraint system to match the PCS constraints but 
# the effect on the target correlation CCF(h) could be deleterious.



















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


