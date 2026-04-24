
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 7 — DECOUPLE FROM PRESENT (DFP) PREDICTOR
# PART 4: Exploiting Hidden Structure
# ════════════════════════════════════════════════════════════════════

# This Tutorial applies the procedure of Tutorial 6 to an ARMA process. 
# The ARMA process is `difficult' to forecast in the sense 
# It applies the DFP based on a time-shift constraint in frequency zero.

#

#----------------------------------------------------------------------

# ── INITIALISATION ────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines
# Provides DFP_compute_lambda_alpha0_func() and related solvers
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, CCF helpers)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)




# ════════════════════════════════════════════════════════════════════
# Exercise 1: ARMA(3,2)
# ════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────
# 1.1 Process specification
# ─────────────────────────────────────────────────────────────────────

# AR-coefficients
ar1<-0.4
ar2<-0.3
ar3<-0.2
# Ma coefficients
b1<-0.5
b2<-0.4

L<-50
# Same forecast horizon as in tutorial 6
h<-3

xi <- c(1, ARMAtoMA(
  ar      = c(ar1,ar2,ar3),
  ma      = c(b1,b2),
  lag.max = 100
))


par(mfrow=c(1,1))
ts.plot(xi,main="Wold decomposition")


# Roots of characteristic polynomial
1/(Arg(polyroot(c(-ar3,-ar2,-ar1,1)))/pi)
abs(polyroot(c(-ar3,-ar2,-ar1,1)))


# MA inversion: we rely on MA form of predictors (the AR form is provided below)
gamma<-xi
# L-length now- and MSE forecast
gamma0<-gamma[1:L]
gammah<-gamma[h+(1:L)]
# For reference we also use a long forecast horizon MSE predictor
htilde<-20
gammahtilde<-gamma[htilde+(1:L)]

# Compute CCF of h-step MSE predictor: for reference against DFP
cor_vec<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec
cor_vec_mat_mse<-cor_vec
cor_vec_mse<-c(cor_vec[1],cor_vec[1+h])
# Compute CCF of htilde-step MSE predictor: for reference against DFP
cor_vec<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammahtilde,gamma0)$cor_vec
cor_vec_mat_mse<-cbind(cor_vec_mat_mse,cor_vec)
cor_vec_mse<-rbind(cor_vec_mse,c(cor_vec[1],cor_vec[1+h]))




# ─────────────────────────────────────────────────────────────────────
# 1.2 MSE-DFP 
# ─────────────────────────────────────────────────────────────────────

# Compute covariance of MSE gammah and nowcast gamma0 
alpha0_mse<-as.double(gammah%*%gamma0)

# Specify alpha0 in DFP constraint: alpha0 is a scale dependent covariance (not a correlation)
# We select alpha0 as percentage of alpha0_mse: from 70% to 0% (the last is the fully decoupled design)
alpha0_vec<-round(c(0.7,0.45,0.22,0.1,0)*alpha0_mse,2)

# Compute MSE-DFP for the alpha0-sequence
max_lag<-0
b_mat<-b_mat_unscaled<-a_mat<-lambda_vec2<-cor_vec_mat_1<-cor_vec_1<-NULL

for (i in 1:length(alpha0_vec))#i<-1
{
  alpha0<-alpha0_vec[i]
  # Function for deriving b0  
  dfp_obj<-mse_dfp_from_alpha0_func(gamma0,gammah,alpha0)
  
  b0<-dfp_obj$b
  lambda0<-dfp_obj$lambda
  b_mat<-cbind(b_mat,b0)
# Compute CCF  
  cor_vec<-compute_acf_at_lags_zero_delta_func(max_lag,h,b_mat[,ncol(b_mat)],gamma0)$cor_vec
  cor_vec_mat_1<-cbind(cor_vec_mat_1,cor_vec)
# Extract CCF at lead 0 and h  
  cor_vec_1<-rbind(cor_vec_1,c(cor_vec[1],cor_vec[1+h]))
}

# Assemble MSE and DFP predictors and CCFs
colnames(b_mat)<-paste("alph0=",alpha0_vec,sep="")
filter_mat<-cbind(gammah,gammahtilde,b_mat)
colnames(filter_mat)<-c(paste("MSE(",h,")",sep=""),paste("MSE(",htilde,")",sep=""),colnames(b_mat))

cor_vec_2<-rbind(cor_vec_mse,cor_vec_1)
colnames(cor_vec_2)<-c("Lag 0",paste("h=",h,sep=""))
rownames(cor_vec_2)<-colnames(filter_mat)

cor_vec_mat<-cbind(cor_vec_mat_mse,cor_vec_mat_1)
colnames(cor_vec_mat)<-colnames(filter_mat)

# ─────────────────────────────────────────────────────────────────────
# 1.3 Routine Checks: 
# ─────────────────────────────────────────────────────────────────────

# 1. Check DFP constraint: should vanish
# Note: neither b_mat nor gamma0 are scaled to unit length and therefore alpha0_vec is not a correlation
t(b_mat)%*%gamma0-alpha0_vec

# 2. Preserve signs
# All designs except fully decoupled DFP preserve trend orientation:
# When the sum of filter coefficients is positive, the direction of the predicted 
# linear trend is not changed, see section 4.1 Wildi (2026) 

apply(b_mat,2,sum)

par(mfrow=c(2,1))
# Illustration
trend<-1:100
forecast_trend<-NULL
# Select fully decoupled DFP
b<-b_mat[,ncol(b_mat)]
for (i in L:100)
  forecast_trend[i]<-b%*%trend[i:(i-L+1)]
# The direction of the predicted trend is inverted:
ts.plot(forecast_trend,main="Forecasted trend: fully decoupled DFP inverts orientation",ylab="")

# Select any other (non fully decoupled) DFP
k<-3
# Ensure admissibility: k>0 and k<ncol(b_mat)
k<-min(ncol(b_mat)-1,k)
k<-max(1,k)
b<-b_mat[,k]
for (i in L:100)
  forecast_trend[i]<-b%*%trend[i:(i-L+1)]
# The direction of the predicted trend is NOT inverted:
ts.plot(forecast_trend,main="NON fully decoupled DFP does NOT invert orientation",ylab="")

# Note: the fully decoupled DFP also inverts the sign of a (non-zero) mean level:
par(mfrow=c(2,1))
# Illustration
mu<-rep(1,100)
forecast_mu<-NULL
# Select fully decoupled DFP
b<-b_mat[,ncol(b_mat)]
for (i in L:100)
  forecast_mu[i]<-b%*%mu[i:(i-L+1)]
# The direction of the predicted trend is inverted:
ts.plot(forecast_mu,main="Forecasted mu: fully decoupled DFP changes sign",ylab="",ylim=c(1.1*forecast_mu[100],0))
abline(h=0,lty=2)
# Select any other (non fully decoupled) DFP
k<-3
# Ensure admissibility: k>0 and k<ncol(b_mat)
k<-min(ncol(b_mat)-1,k)
k<-max(1,k)
b<-b_mat[,k]
for (i in L:100)
  forecast_mu[i]<-b%*%mu[i:(i-L+1)]
# The direction of the predicted trend is NOT inverted:
ts.plot(forecast_mu,main="NON fully decoupled DFP does NOT change sign",ylab="",ylim=c(0,1.1*forecast_mu[100]))
abline(h=0,lty=2)



# Discussion
# Maintaining trend direction or the sign of the mean is often (though not always) 
# a meaningful criterion.
# Counter-example: if the process is periodic with periodicity per, then the predictor becomes 
# out-of-phase at forecast horizon h=per/2. In this case the predictor changes signs.




# ─────────────────────────────────────────────────────────────────────
# 1.4 AR Form
# ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------
# 1.4.1 Compute AR Inversion
# ---------------------------------------------------------------------

# AR-inversion of DFP
# 1. Compute AR-inversion of ARMA
ar_inv <- -ARMAtoMA(ar = -c(b1,b2), ma = -c(ar1,ar2,ar3), lag.max = L-1)
# AR-filter
theta<-c(1,-ar_inv)
# Verify the approach via a known identity:
# Convolving the AR inversion with the Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros). 
conv_two_filt_func(xi, theta)$conv[1:10]




# 2. Compute AR weights of MSE and DFP predictors
# a. MSE
filt2<-gammah
ar_mse_arma32<-conv_two_filt_func(theta,filt2)$conv
# b. DFP
ar_dfp_arma32_mat<-NULL
for (i in 1:length(alpha0_vec))
{
  # Use original (unscaled) MSE-DFP
  filt2<-b_mat[,i]
  ar_dfp_arma32_mat<-cbind(ar_dfp_arma32_mat,conv_two_filt_func(theta,filt2)$conv)
  
}

# ---------------------------------------------------------------------
# 1.4.2 Plot: Compare MA and AR Forms
# ---------------------------------------------------------------------

par(mfrow=c(1,1))
ts.plot(cbind(ar_mse_arma32,ar_dfp_arma32_mat)[1:10,],col=rainbow(length(alpha0_vec)+1),main="MSE and DFP predictors AR-inverted")
# Redraw MSE in standard green color
lines(ar_mse_arma32[1:10],lwd=1,col="green")

# Check: apply filters to MA and AR representations
set.seed(1)
len<-1000
x<-eps<-rnorm(len)
for (i in 4:len)
{
  x[i]<-ar1*x[i-1]+ar2*x[i-2]+ar3*x[i-3]+eps[i]+b1*eps[i-1]+b2*eps[i-2]
}

y_dfp_ar32<-y_dfp_ma32<-rep(NA,len)
# Select DFP design
k<-2
for (i in L:len)
{
  y_dfp_ma32[i]<-b_mat[,k]%*%eps[i:(i-L+1)]
  y_dfp_ar32[i]<-ar_dfp_arma32_mat[1:L,k]%*%x[i:(i-L+1)]
}
# Both series are identical up to negligible finite MA/AR inversion errors
ts.plot(cbind(y_dfp_ma32,y_dfp_ar32)[1:200,])
# Maximal error is negligible (due to finite length MA/AR inversions)
max(na.exclude(abs(y_dfp_ma32-y_dfp_ar32)[1:200]))


# ─────────────────────────────────────────────────────────────────────
# 1.5 DFP Predictors: Weights and CCF
# ─────────────────────────────────────────────────────────────────────
# Plot


par(mfrow=c(1,2))


colo<-c("green","darkgreen","brown","orange","blue","violet","red")

#mplot3<-cbind(gammah,b_mat)


mplot<-filter_mat

ts.plot(mplot,main="ARMA(3,2)",col=colo,xlab="",ylab="")
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=colo[i])
abline(h=0)


mplot<-cor_vec_mat[1:22,]*as.double(sqrt(gamma0%*%gamma0)/sqrt(gamma%*%gamma))

plot(mplot[,1],axes=F,type="l",xlab="",ylab="",main="CCF",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i])
}
abline(h=0)
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
axis(1,at=1:nrow(mplot),labels=-max_lag-1+1:(nrow(mplot)))
axis(2)
box()


# Outcome:
# CCF: 
#   -Difficult estimation problem: alpha0 must be decreased heavily to affect CCF at lag 0 
# see first column in cor_vec
round(cor_vec_2,4)
#   -As alpha0 decreases, the CCF at lag 0 decreases (stronger decoupling).
#   -This loss spills over to the forecast horizon h=3 but the DFP minimizes this loss.
# Coefficients:
#   -As decoupling increases (smaller alpha0) the original smooth pattern of 
#    the MSE predictor (green) becomes increasingly unsmooth and ragged.
#   -Increased look ahead behaviour of the DFP emphasizes features of the 
#     data generating process that are hidden by the MSE predictor.

# ─────────────────────────────────────────────────────────────────────
# 1.6 Verification: 
# Empirical Performances Converge Towards Expected (True) Values
# ─────────────────────────────────────────────────────────────────────

len<-100000
set.seed(932)

x<-eps<-rnorm(len)

for (i in 4:len)
  x[i]<-ar1*x[i-1]+ar2*x[i-2]+ar3*x[i-3]+eps[i]+b1*eps[i-1]+b2*eps[i-2]


y_out_mat<-NULL
perf_mat<-matrix(ncol=ncol(filter_mat),nrow=2)
colnames(perf_mat)<-colnames(filter_mat)
rownames(perf_mat)<-c("Lag 0",paste("h=",h,sep=""))
for (i in 1:ncol(filter_mat))
{  
  y<-filter(eps,filter_mat[,i],side=1)
  y_out_mat<-cbind(y_out_mat,y)
  perf_mat[1,i]<-cor(y[L:len],x[L:len])
  perf_mat[2,i]<-cor(y[L:(len-h)],x[(h+L):len])
}
colnames(y_out_mat)<-colnames(filter_mat)

# Empirical CCF: remove MSE to align with cor_vec_2
t(perf_mat)
# Compare with true (expected) CCF of DFP
cor_vec_2

anf<-650
enf<-750

# Compare data left-shifted by h=3 with MSE(3) and MSE(20)
select_filters<-1:2
mplot<-cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,]
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])
# The main effect is scaling: MSE(20) has a smaller variance than MSE(3)

# To better evaluate the `look ahead' effect of the MSE(20) over MSE(3) we now standardize the series 
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])
# Outcome: increasing the forecast horizon in MSE does not allow to look ahead.
# The forecast problem is `difficult'.


# We now add the DFP designs. all DFP except fully decoupled (the latter inverts 
# trend orientation)

# Select all filters except fully decoupled
select_filters<-1:(ncol(y_out_mat)-1)
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Target, MSE and DFP Predictors")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])


# Let's apply a magnifying glass at a `turning point':
# Note that the data (black line) is left shifted by h=3
anf<-710
enf<-720
# Select all filters except fully decoupled
select_filters<-1:(ncol(y_out_mat)-1)
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),select_filters])[anf:enf,])
colnames(mplot)<-c("Data",colnames(y_out_mat)[select_filters])
par(mfrow=c(1,1))
coli<-c("black",colo)

ts.plot(mplot,col=coli,main="Magnifying glass")
lines(mplot[,2],col="green",lty=2,lwd=2)
for (i in 1:ncol(mplot))
  mtext(colnames(mplot)[i],line=-i,col=coli[i])





# ════════════════════════════════════════════════════════════════════
# Main Take Aways
# ════════════════════════════════════════════════════════════════════
# The 





















