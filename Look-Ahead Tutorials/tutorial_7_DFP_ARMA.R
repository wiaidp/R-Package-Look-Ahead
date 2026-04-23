
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 7 — DECOUPLE FROM PRESENT (DFP) PREDICTOR
# PART 4: Exploiting Hidden Structure
# ════════════════════════════════════════════════════════════════════

# This Tutorial applies the procedure of Tutorial 6 to an ARMA process. 
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


# MA inversion
gamma<-xi
# L-length now- and MSE forecast
gamma0<-gamma[1:L]
gammah<-gamma[h+(1:L)]

# ─────────────────────────────────────────────────────────────────────
# 1.2 MSE-DFP 
# ─────────────────────────────────────────────────────────────────────

# Specify alpha0 ij DFP constraint: alpha0 is a scale dependent covariance (not a correlation)
alpha0_vec<-c(0.9,0.45,0.22,0.1,0)

# Compute MSE-DFP for the alpha0-sequence
max_lag<-0
cor_vec_mat<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec
b_mat<-b_mat_unscaled<-a_mat<-lambda_vec2<-NULL
cor_vec_2<-matrix(ncol=2,nrow=length(alpha0_vec))

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
  cor_vec_mat<-cbind(cor_vec_mat,cor_vec)
# Extract CCF at lead 0 and h  
  cor_vec_2[i,1]<-cor_vec[1]
  cor_vec_2[i,2]<-cor_vec[1+h]
}
colnames(cor_vec_2)<-c("Lag 0",paste("h=",h,sep=""))
rownames(cor_vec_2)<-colnames(b_mat)<-paste("alph0=",alpha0_vec,sep="")

# Check DFP constraint: should vanish
# Note: neither b_mat nor gamma0 are scaled to unit length and therefore alpha0_vec is not a correlation
t(b_mat)%*%gamma0-alpha0_vec
# Alternative check DFP constraint: should vanish
# Note: since cor_vec2 is the CCF we have to scale the covariance alpha0_vec by inverse lengths of gamma0 and b
cor_vec_2[,1]-alpha0_vec/sqrt(diag((t(b_mat)%*%b_mat))*as.double(t(gamma0)%*%gamma0))



mplot3<-scale(cbind(gammah,b_mat),center=F,scale=T)/sqrt(L-1)
#mplot3<-cbind(gammah,b_mat)

mplot4<-cor_vec_mat[1:22,]*as.double(sqrt(gamma0%*%gamma0)/sqrt(gamma%*%gamma))
colnames(mplot4)<-c("MSE",colnames(b_mat))

# ─────────────────────────────────────────────────────────────────────
# 1.3 AR Inversion
# ─────────────────────────────────────────────────────────────────────
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
ts.plot(cbind(ar_mse_arma32,ar_dfp_arma32_mat)[1:L,],col=rainbow(length(alpha0_vec)+1),main="MSE and DFP predictors AR-inverted")
lines(ar_mse_arma32,lwd=2)

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
# 1.4 DFP Predictors: Weights and CCF
# ─────────────────────────────────────────────────────────────────────
# Plot


par(mfrow=c(1,2))

#ts.plot(gammah1,main=paste("MSE first process: h=",h,sep=""),col="green",xlab="",ylab="")

#ts.plot(gammah,main="Second process",col="green",xlab="",ylab="")


colo<-c("green","brown","orange","blue","violet","red")



ts.plot(mplot3,main="ARMA(3,2)",col=colo,xlab="",ylab="")
mtext(expression(paste("DFP ",alpha[0],"=0.9 ")),line=-2,col=colo[2])
mtext(expression(paste("    ",alpha[0],"=0.45 ")),line=-3,col=colo[3])
mtext(expression(paste("    ",alpha[0],"=0.22 ")),line=-4,col=colo[4])
mtext(expression(paste("    ",alpha[0],"=0.1 ")),line=-5,col=colo[5])
mtext(expression(paste("    ",alpha[0],"=0 ")),line=-6,col=colo[6])
abline(h=0)


plot(mplot4[,1],main="",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot4),max(mplot4)))
for (i in 2:ncol(mplot4))
{  
  lines(mplot4[,i],col=colo[i])
}
abline(h=0)
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9")),line=-2,col=colo[2])
#mtext(expression(paste("DFP ",alpha[0],"=0.45")),line=-3,col=colo[3])
#mtext(expression(paste("DFP ",alpha[0],"=0.22")),line=-4,col=colo[4])
#mtext(expression(paste("DFP ",alpha[0],"=0.1")),line=-5,col=colo[5])
#mtext(expression(paste("DFP ",alpha[0],"=0")),line=-6,col=colo[6])
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
axis(1,at=1:nrow(mplot4),labels=-max_lag-1+1:(nrow(mplot4)))
axis(2)
box()


# Outcome:
# CCF: 
#   -As alpha0 decreases, the CCF at lag 0 decreases (stronger decoupling).
#   -This loss spills over to the forecast horizon h=3 but the DFP minimizes this loss.
# Coefficients:
#   -As decoupling increases (smaller alpha0) the original smooth pattern of 
#    the MSE predictor (green) becomes increasingly unsmooth and ragged.
#   -Increased look ahead behaviour of the DFP emphasizes features of the 
#     data generating process that are hidden by the MSE predictor.
round(cor_vec_2,2)

# ─────────────────────────────────────────────────────────────────────
# 1.5 Verification: 
# Empirical Performances Converge Towards Expected (True) Values
# ─────────────────────────────────────────────────────────────────────

len<-100000
set.seed(932)

x<-eps<-rnorm(len)

for (i in 4:len)
  x[i]<-ar1*x[i-1]+ar2*x[i-2]+ar3*x[i-3]+eps[i]+b1*eps[i-1]+b2*eps[i-2]

filter_mat<-cbind(gammah,b_mat)
colnames(filter_mat)<-c("MSE",colnames(b_mat))

y_out_mat<-NULL
perf_mat<-matrix(ncol=ncol(filter_mat),nrow=2)
colnames(perf_mat)<-colnames(filter_mat)
rownames(perf_mat)<-c("Lag 0",paste("h=",h,sep=""))
for (i in 1:ncol(b_mat))
{  
  y<-filter(eps,filter_mat[,i],side=1)
  y_out_mat<-cbind(y_out_mat,y)
  perf_mat[1,i]<-cor(y[L:len],x[L:len])
  perf_mat[2,i]<-cor(y[L:(len-h)],x[(h+L):len])
}

# Empirical CCF
t(perf_mat)
# Compare with true (expected) CCF of DFP
cor_vec_2

# Lead
anf<-600
enf<-900

length(x[(h+1):len])
nrow(y_out_mat[1:(len-h),])

# 
mplot<-scale(cbind(x[(h+1):len],y_out_mat[1:(len-h),c(1,3)])[anf:enf,])
par(mfrow=c(1,1))
colo<-c("black","green",rainbow(ncol(b_mat)))
ts.plot(mplot,col=colo,main="Target, MSE and DFP Predictors")




# ════════════════════════════════════════════════════════════════════
# Main Take Aways
# ════════════════════════════════════════════════════════════════════
# The 





















