# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1 — MSE: THE LOOK AHEAD PROBLEM
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial presents simple forecast problem, illustrating that the 
# The classic mean-squared error (MSE) multi-step ahead forecast does not 
# look ahead or anticipate data dynamics.
#
# Theoretical background:
#   Wildi, M. (2026). Forecasting on the Accuracy-Timeliness Frontier: 
#   Two Novel `Look Ahead' Predictors. 
#   https://doi.org/10.48550/arXiv.2602.23087


# ─────────────────────────────────────────────────────────────────

rm(list=ls())

# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R utility functions/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R utility functions/HP_JBCY_functions.r",sep=""))


#============================================================
# EXercise 1: MSE Forecast
#============================================================
# Data-Generating Process (DGP):
#   x_t = \sum_{k=0}^9 0.9^kε_{t-k}  (MA(9))
#============================================================

# 1.1 Data
# We generate a MA(9) based on a truncated MA-inversion of an AR(1),
# i.e., b_k=ar1^k, k=0,...,9

q<-9
a1<-0.9
b<-a1^(0:9)
par(mfrow=c(1,1))
ts.plot(b,main="MA(9) coefficients",xlab="Lag")
# generate data
len<-100
set.seed(231)
eps<-rnorm(len+q+1)
# Initialize data and MSE predictor
x<-rep(NA,len+q+1)
for (i in (q+1):(len+q+1))
{
  x[i]<-b%*%eps[i:(i-q)]
}

ts.plot(x,main="A realization of the MA(9)")
# Dependence
acf(na.exclude(x))


# 1.2 MSE predictor
# Forecast horizon
h<-5
if (h>L_target)
  print("L_target must be larger than h. Otherwise best forecast is zero!!!!!")

# Note that in applications, typically, epsilon_t are unobserved and can be recovered 
# through AR-inversion of the MA process. In this example, however, epsilon_t are known.
# We here interpret the predictor as a filter applied to epsilon_t. We seek optimal 
# filter (predictor) weights such that the filter output (predictor) is closest possible to 
# x_{t+h} in a MSE sense.
# To derive the MSE predictor of x_{t+h} note that
# x_{t+h}=ε_{t+h}+b_1ε_{t+h-1}+...+b_{h-1}ε_{t+1}+b_hε_{t}+...+b_qε_{t+h-q}
# The h-step ahead MSE predictor is obtained by replacing future ε_{t+k}, k>0, by zero (their best MSE forecast)
# x_{MSE,th}=b_hε_{t}+...+b_qε_{t+h-q}
# Thus the optimal MSE predictor has weights 
b_MSE<-b[h+1:L]
# Compute the MSE predictor
xhat<-rep(NA,len+L_target)
for (i in L_target:(len+L_target))
{
  xhat[i]<-b_MSE%*%eps[i:(i-L+1)]
}


# Plot forward-shifted x (the forecast target) and the MSE predictor
colo<-c("black","green")

par(mfrow=c(2,1))

mplot<-na.exclude(cbind(c(x[(h+1):(len-h)],rep(NA,h)),xhat))
plot(mplot[,1],main=paste("x_t shifted forward ",h," steps (target) and optimal 5-step ahead MSE forecast (green)",sep=""),axes=F,type="l",xlab="Time",ylab="",col=colo[1],lwd=1)
lines(mplot[,2],col=colo[2])
mtext("Target",line=-1)
mtext("MSE predictor",col="green",line=-2)
axis(1,at=1:nrow(mplot),labels=1:nrow(mplot))
axis(2)
box()

# Plot x (without shift) and the MSE predictor
mplot<-na.exclude(cbind(x,xhat))
plot(mplot[,1],main="Original x_t and optimal 5-step ahead forecast (green)",axes=F,type="l",xlab="Time",ylab="",col=colo[1],lwd=1)
lines(mplot[,2],col=colo[2])
mtext("Original data: MA(9) without shift",line=-1)
mtext("MSE predictor",col="green",line=-2)
axis(1,at=1:nrow(mplot),labels=1:nrow(mplot))
axis(2)
box()


# 1.3 Cross correlation function, see Wildi 2026a 
# We now compute the correlation of the MSE predictor with x_{t+k} for -4<=k<=9
# Ideally, the MSE predictor correlates strongly with x_{t+h} (target correlation)
# One can show, indeed, that the MSE predictor maximizes the correlation with x_{t+h}:
# no other predictor can increase the MSE target correlation.

# Lags in computation of cor
max_lag<-5

# Compute CCF
cor_vec_lead<-cor_vec_lag<-NULL
# Leads: 0 up to h
for (i in 0:h)#i<-1
  cor_vec_lead<-c(cor_vec_lead,b[1:(min(L+i,L_target)-i)]%*%gammak[(i+1):min(L+i,L_target)]/(sqrt(b%*%b)*sqrt(gammak%*%gammak)))
# Leads: h+1,...,L_target (after L_target the best forecast is zero)
for (i in 1:(L-1))#i<-1
  cor_vec_lead<-c(cor_vec_lead,b[1:(L-i)]%*%gammak[(h+i)+1:(L-i)]/(sqrt(b%*%b)*sqrt(gammak%*%gammak)))
# Lags
for (i in 1:(max_lag-1))#i<-1
  cor_vec_lag<-c(cor_vec_lag,b[(i+1):L]%*%gammak[1:(L-i)]/(sqrt(b%*%b)*sqrt(gammak%*%gammak)))
cor_vec<-c(cor_vec_lag[length(cor_vec_lag):1],cor_vec_lead)

par(mfrow=c(1,1))
colo<-c("green","green")
plot(cor_vec,main="CCF at various leads and lags",axes=F,type="l",xlab="Leads (positive) and lags (negative numbers)",ylab="correlation",col=colo[1],lwd=1)
abline(v=max_lag)
abline(v=max_lag+h,col="green")
axis(1,at=1:length(cor_vec),labels=-(max_lag)+1:length(cor_vec))
axis(2)
box()



# ============================================================
# DISCUSSION: MSE FILTER & FORECAST TRILEMMA
# ============================================================
# The classical MSE filter optimizes predictive accuracy alone,
# ignoring smoothness and timeliness. As the forecast horizon
# (delta) increases:
#   → MSE increases        (bottom row of output table)
#   → Holding time drops   (filter becomes noisier)
#   → Delay/shift reduces  (signal appears more timely)
#
# These trade-offs illustrate the core tension known as the
# FORECAST TRILEMMA: accuracy, smoothness, and timeliness
# cannot all be simultaneously improved.
# ============================================================


# ============================================================
# SSA 
# ============================================================
# SSA extends the classical MSE framework by explicitly
# controlling smoothness via the holding time (ht) parameter.
#
# Two equivalent formulations:
#   - Primal form : Minimize MSE subject to a fixed ht constraint
#   - Dual form   : Maximize ht subject to a fixed MSE constraint
#
# Together, these trace out an EFFICIENT FRONTIER in the
# (ht, MSE) space — analogous to Markowitz in portfolio theory —
# allowing practitioners to navigate the smoothness-accuracy
# trade-off in a principled way.
# ============================================================



# ============================================================
# NOTE ON TIMELINESS CONTROL
# ============================================================
# SSA controls Timeliness INDIRECTLY via the forecast horizon 
# parameter (delta).
#
# More direct and specialized approaches are available:
#   → Look-Ahead DFP and PCS 
#   → These trace out an efficient frontier between MSE and timeliness
#   → A corresponding tutorial is in preparation. 
#
# ============================================================

