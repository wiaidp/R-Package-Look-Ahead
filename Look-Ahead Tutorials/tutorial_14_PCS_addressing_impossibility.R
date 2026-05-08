
# Most difficult: self-similar AR(1)


# If delta is small V[,1] \propto gamma0
# V[,2] orthogonal to V[,1] and hence gamma0: therefore V[,2] fully decouples from gamma0
# In AR(1) decoupling is impossible becuase gamma_h\propto gamma_h: decoupling is realized through perturbation.
# Perturbation allows to generate an artificial  fully decoupled predictor. This does not lie in gamma0, gamma_h space as classic decoupling.
# However, as shown in DFP, full decoupling generates a lead.
# The perturbated PCS lie in the room spanned by gamma0 and fully perturbation decoupling (but PCS is only a subspace of that room/plane)
# Assigning full weight to gamma0 replicates gammah (since gamma0 \pripto gammah)
# Assiging full weight to perturbation maximzes look ahead. However, target correlation is not under explicit control in this case.


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
b_mat<-NULL
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

# Shifting the CCF peak is impossible because b' * gamma_h = a1^h b' * gamma0 
# irrespective of b. 

# The identity would be faster (looking ahead) but the identity implies CCF(k)=0 for k>0.
# The constraints make this choice impossible since |CCF(1)-CCF(0)| would 
# be very large when compared to |CCF(k)-CCF(k-1)|, k>0, so that no fixed beta 
# exists such the constraints are small. However, selecting beta_1 large and beta_k small for k>1 would enable the identity.
# In other words: the constraints (beta) should account for the type of perturbation.



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
beta_vec<-c(-1,-0.1,0,0.0000001,0.0000002,0.00000025,0.000000269,0.0000003,0.0000005, 0.00001,10)

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

# perturbate gamma0 only and only slightly:
# gammah_mat_perturbate[1,]<-(gammah_mat[1,]+delta*gamma_all_a1_perturbate[1:L])/sqrt(sum(gamma_all_a1_perturbate^2)) 
# In this case V1 \approx gamma0 (the main direction with the largest eigenvalue is gamma0)
# V2 is orthogonal to V1 \approx gamma0.


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
delta<-0.001
a1_perturbate<-a1+delta

xi_a1_perturbate <- c(1, ARMAtoMA(ar= a1_perturbate, ma=0,lag.max = 1000))

gamma_all_a1_perturbate <- xi_a1_perturbate


ts.plot(xi-xi_a1_perturbate)

gammah_mat_perturbate<- gammah_mat

# Perturbate gamma0
gammah_mat_perturbate[1,]<-gamma_all_a1_perturbate[1:L]/sqrt(sum(gamma_all_a1_perturbate^2))

gammah_mat_perturbate[1,]<-(gammah_mat[1,]+delta*gamma_all_a1_perturbate[1:L])/sqrt(sum(gamma_all_a1_perturbate^2)) 
                                                        

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
V[2:L,1]/V[1:(L-1),1]
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

beta_vec<-c(-1,0,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.455,0.46,0.47,0.5,5)/5000000

Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))


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
# 3.5 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

len<-10000
set.seed(534)

x_filt <- rnorm(len)

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)



anf<-150
enf<-500
# Select the relevant PCS: 
# -Smaller beta imply a lag
# -larger beta (columns >= 10) are increasingly leading.
# -The more they lead the more the predictor appears to invert the sign; 
# -Very difficult forecast problem.
select_pcs<-10:ncol(y_out_mat)
select_vec<-c(1,select_pcs)
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


# -Very difficult forecast problem.

# Here we show look ahead predictors that do not emphasize sign inversion:
# Select a narrower time span to magnify the look ahead effect.
# Note: the look-ahead effect mainly works on longer swings. Short-term 
# random spikes cannot be predicted. 
# This particular `long swing' look ahead behaviour might be relevant in the 
# context of business cycle analysis, where crises are typically determined by 
# longer and stronger downturns, i.e., negative swings.
anf<-280
enf<-400

select_pcs<-10:12
select_vec<-c(1,select_pcs)
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



# For increasing beta, the CCF is increasingly skewed
mplot_ccf<-scale(na.exclude(y_out_mat[,select_vec]))
colnames(mplot_ccf)<-colnames(y_out_mat)[select_vec]


# Note: the right tail (to the right of lag 0) of the ccf always corresponds to the AR(1).
# This is because b' * gama_h \propto a1^h because gammah=a1^h*gamma0
# It is impossible to shift the peak of the CCF to the right of zero in the AR(1) case (with a1>0).
# Nevertheless, look ahead behaviour is obtained by skewing the CCF. 
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])





# ─────────────────────────────────────────────────────────────────────
# 3.6 Medium Regularization
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

beta_vec<-c(2,2.055,2.057,2.058,2.059,2.0591,2.0592,2.0593,2.0594,2.0595,2.06,2.1)/lambda


Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))


# ─────────────────────────────────────────────────────────────────────
# 3.7 Plots
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

# ─────────────────────────────────────────────────────────────────────
# 3.8 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

len<-10000
set.seed(534)

x_filt <- rnorm(len)

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


anf<-150
enf<-500

# Select the relevant PCS: For increasing beta the predictors are increasingly left-shifted.
# For increasing beta the predictros appear to change sign.
# Very difficult forecast problem.
select_pcs<-c(4:7)
select_vec<-c(1,select_pcs)
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


# For increasing beta, the CCF is increasingly skewed
mplot_ccf<-scale(na.exclude(y_out_mat[,select_vec]))
colnames(mplot_ccf)<-colnames(y_out_mat)[select_vec]


# Note: the right tail of the ccf always corresponds to the AR(1).
# This is because b' * gama_h \propto a1^h because gammah=a1^h*gamma0
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])




# ════════════════════════════════════════════════════════════════════
# EXERCISE 4: ALTERNATIVE AR(2) PERTURBATION
# ════════════════════════════════════════════════════════════════════

# perturbate gamma0 only and only slightly:
# gammah_mat_perturbate_ar2[1,]<-gammah_mat[1,]+delta*gamma_all_ar2[1:L]/sqrt(sum(gamma_all_ar2^2)) 
# In this case V1 \approx gamma0 (the main direction with the largest eigenvalue is gamma0)
# V2 is orthogonal to V1 \approx gamma0.


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

delta<-0.0001

gammah_mat_perturbate_ar2[1,]<-gammah_mat[1,]+delta*gamma_all_ar2[1:L]/sqrt(sum(gamma_all_ar2^2)) 




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
ts.plot(eigenN$vectors[,1:2], main="Eigenvectors of non-vanishing eigenvalues of N",lty=1:2)
V[2:L,1]/V[1:(L-1),1]
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



# Very strong regularization
lambda<-5000000

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]

beta_vec<-c(-10,-1,0,1,1.2,1.25,1.27,1.3,1.35,1.4,1.5,2,10)/lambda

Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))


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
# 4.5 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

len<-10000
set.seed(534)

x_filt <- rnorm(len)

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


anf<-150
enf<-500

# Select the relevant PCS: For increasing beta the predictors are increasingly left-shifted.
# For increasing beta the predictros appear to change sign.
# Very difficult forecast problem.
select_pcs<-c(1:5)
select_vec<-c(1,select_pcs)
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


# For increasing beta, the CCF is increasingly skewed
mplot_ccf<-scale(na.exclude(y_out_mat[,select_vec]))
colnames(mplot_ccf)<-colnames(y_out_mat)[select_vec]


# Note: the right tail of the ccf always corresponds to the AR(1).
# This is because b' * gama_h \propto a1^h because gammah=a1^h*gamma0
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])








# ─────────────────────────────────────────────────────────────────────
# 4.6 Play the Expanded Rank-Game: medium Regularization
# ─────────────────────────────────────────────────────────────────────

# We fix lambda to a medium regularization
# We then vary beta: the two extreme beta values correspond to plus and minus the 
# first eigenvector V[,1] with combinations V[,2]+lambda1*V[,2] in between, where 
# lambda1 depends on beta.



# Medium regularization
lambda<-5

# Tipping points: the two extremes are -V[,2] and +V[,2]
# For beta=0.000000269 one obtains -V[,1]

beta_vec<-c(0.43,0.4371,0.4372,0.43725,0.43727,0.4373,0.43733,0.43735,0.4374,0.4375,0.438)/lambda


Delta<-1:h

b_mat<-NULL
for (i in 1:length(beta_vec))
{
  
  beta<-beta_vec[i]
  PCS_obj<-PCS_perturbation_func(h,Delta, gamma_pcs, L, beta, lambda,gammah_mat_perturbate_ar2)
  
  b       <- PCS_obj$b
  b_mat<-cbind(b_mat,b)
}
filter_mat<-cbind(gamma0,b_mat)
colnames(filter_mat)<-c("MSE",paste("lambda=",round(lambda,2),", beta=",round(beta_vec,8)))



# ─────────────────────────────────────────────────────────────────────
# 4.7 Plots
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


# ─────────────────────────────────────────────────────────────────────
# 4.8 Apply and Compare Predictors
# ─────────────────────────────────────────────────────────────────────

len<-10000
set.seed(534)

x_filt <- rnorm(len)

y_out_mat <- NULL
for (i in 1:ncol(filter_mat))
  y_out_mat <- cbind(y_out_mat, filter(x_filt, filter_mat[, i], side = 1))
colnames(y_out_mat) <- colnames(filter_mat)


anf<-150
enf<-500

# Select the relevant PCS: For increasing beta the predictors are increasingly left-shifted.
# For increasing beta the predictros appear to change sign.
# Very difficult forecast problem.
select_pcs<-c(2:5)
select_vec<-c(1,select_pcs)
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


# For increasing beta, the CCF is increasingly skewed
mplot_ccf<-scale(na.exclude(y_out_mat[,select_vec]))
colnames(mplot_ccf)<-colnames(y_out_mat)[select_vec]


# Note: the right tail of the ccf always corresponds to the AR(1).
# This is because b' * gama_h \propto a1^h because gammah=a1^h*gamma0
par(mfrow=c(2,2))
ccf(mplot_ccf[,1],mplot_ccf[,2],main=colnames(mplot_ccf)[1])
ccf(mplot_ccf[,1],mplot_ccf[,3],main=colnames(mplot_ccf)[2])
ccf(mplot_ccf[,1],mplot_ccf[,4],main=colnames(mplot_ccf)[3])
ccf(mplot_ccf[,1],mplot_ccf[,5],main=colnames(mplot_ccf)[4])





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

















