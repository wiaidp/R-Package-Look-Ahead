# New Material : September 2026
# Introduction of Max-Tau Predictor
# Main difference to DFP: Max-Tau decouples from new vectors which maximize lead unconditionally: 
# no other linear predictor of the same length can outperform Max-Tau in terms of target correlation (TC) and 
# lead at the reference frequency.

# Max-Tau defines an unconditional efficient frontier between Accuracy (TC) and Timeliness (time-shift lead at reference frequency)

# New paper: See Papers folder.



# ── INITIALISATION ────────────────────────────────────────────────────────────
rm(list = ls())

# Load DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load Max-Tau optimisation routines.
source(paste(getwd(), "/R/Max_Tau.r", sep = ""))


# Load HP utilities
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)
library(mFilter)

# Install and load the alfred package for direct FRED data access (no API key required).
install.packages("alfred")
library(alfred)


# ── DATA ──────────────────────────────────────────────────────────────────────
# Toggle reload_data to TRUE to fetch fresh data from FRED and overwrite the
# locally saved file; set to FALSE to load the previously saved copy.
reload_data <- FALSE

if (reload_data) {
  GDPC1 <- get_fred_series("GDPC1", series_name = "GDP")
  GDPC1 <- as.xts(GDPC1)
  save(GDPC1, file = file.path(getwd(), "Data", "GDP"))
} else {
  load(file = file.path(getwd(), "Data", "GDP"))
}

head(GDPC1)
tail(GDPC1)
is.xts(GDPC1)

# Convert to a plain numeric vector.
# xts objects carry implicit index-handling conventions that can interfere with
# downstream computations (e.g., applying a filter to an xts object may silently
# reverse the time axis). Working with plain doubles avoids these pitfalls.
start_year <- 1992
end_year   <- 2024

y     <- as.double(log(GDPC1[paste(start_year, "/", end_year, sep = "")]))
y_xts <-           log(GDPC1[paste(start_year, "/", end_year, sep = "")])
len   <- length(y)



# ── EXPLORATORY PLOTS ─────────────────────────────────────────────────────────
par(mfrow = c(2, 2))
plot(GDPC1,                          main = "US Real GDP (levels)")
plot(y_xts,                          main = "Log GDP")
plot(diff(y_xts),                    main = "Log-differences of GDP")
acf(na.exclude(diff(y_xts)),         main = "ACF of log-differences")




# Specify target: HP trend applied to diff-log GDP
h<-2
# HP
lambda_hp<-1600
# L is an odd integer such that the symmetric filter is centered at (L-1)/2+1  
L<-51


# Here we compute only the two-sided filter for double length 2*(L-1)+1
# This is used when comparing one-sided to right tail of two-sided
HP_obj<-HP_target_mse_modified_gap(2*(L-1)+1,lambda_hp)
HP_two=HP_obj$target
hp_gap=HP_obj$hp_gap[1:L]
modified_hp_gap=HP_obj$modified_hp_gap[1:L]
# Concurrent HP assuming I(2)-process
hp_trend_long=HP_obj$hp_trend
hp_trend=hp_trend_long[1:L]
# MSE estimate of bi-infinite HP assuming white noise
hp_mse_long=HP_obj$hp_mse
hp_mse<-hp_mse_long[1:L]


# Show that HP-gap applied to returns is the same as original gap applied to levels
hp_gap<-c(1-hp_trend[1],-hp_trend[2:L])
# Apply HP-gap and HP-gap transformed to log and lago-returns
eps<-y#-mean(y)
len<-length(eps)
y_gaph<-filter(log(GDPC1),hp_gap,side=1)
y_gap<-y_gaph[-1]
y_gap_modified<-filter(eps,modified_hp_gap,side=1)

#----------------------------------
# DFP general settings
# Forecast horizon
h<-delta<-h
# Start for CCF (start at lag k i.e. only positive legas (right tail))
max_lag<-0

# Specify gamma at forecast horizon sup_vec_target=delta and at lead/lag sup_vec_constraint
# Target: MSE predictor
gamma_target<-gammah<-hp_trend_long[h+1:L]
# Constraint: nowcast
gamma_constraint<-gamma0<-hp_trend_long[1:L]
# Some checks
ts.plot(gamma_target)
# Should add to one
sum(hp_trend_long)
# STD
sqrt(t(hp_trend_long)%*%hp_trend_long)


#-------------------------------------
# DFP based on alpha0

alpha0_vec<-c(0.1,0.05,0.02,0.017,0.005,0)



lambda<-lambda1<-lambda2<-NULL
b0_mat<-NULL
cor_vec_mse_la_mat<-NULL

for (i in 1:length(alpha0_vec))#i<-1
{ 
  alpha0<-alpha0_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b0_obj<-mse_dfp_from_alpha0_func(gamma_constraint, gamma_target, alpha0)
  b<-b0_obj$b
  b0_mat<-cbind(b0_mat,b)
  lambda<-c(lambda,b0_obj$lambda)
  
  # Compute CCF of PCS predictors with respect to MSE gamma_target  
  cor_vec_mse_la_mat<-cbind(cor_vec_mse_la_mat,compute_acf_at_lags_zero_delta_func(max_lag,h,b0_mat[,i],hp_trend)$cor_vec)
  
}
colnames(b0_mat)<-colnames(cor_vec_mse_la_mat)<-alpha0_vec

cor_vec_t_hp_trend<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,hp_trend)$cor_vec


b_alpha0<-b0_mat
cor_vec_mse_la_mat_alpha0<-cor_vec_mse_la_mat
# Compute Gamma(0)
Gamma0_alpha0<-apply(b_alpha0,2,sum)
# Compute time-shifts at frequency zero
tau_alpha0<--as.vector(t(b_alpha0)%*%(0:(L-1))/apply(b_alpha0,2,sum))
tau_alpha0[which(Gamma0_alpha0<0)]<-NA
# Compute all statistics for MSE benchmark also
mse_b<-c(t(gammah)%*%gamma0/sqrt(t(gammah)%*%gammah*t(gamma0)%*%gamma0),t(gammah)%*%gammah/sqrt(t(gammah)%*%gammah*t(gamma0)%*%gamma0),sum(gammah),-as.double(t(gammah)%*%(0:(L-1))/sum(gammah)))


table_alpha0<-cbind(mse_b,rbind(cor_vec_mse_la_mat_alpha0[1,],cor_vec_mse_la_mat_alpha0[h+1,],Gamma0_alpha0,tau_alpha0))
dim(table_alpha0)


########################################################################################
# Exercise 2: Time-Shift DFP
########################################################################################

#-------------------------------------
# Time-shift DFP (based on tau)


tau_vec<--c(1,2,6,100000)
b0_mat<-cor_vec_mse_la_mat<-NULL
for (i in 1:length(tau_vec))#i<-1
{
  lead<-tau_vec[i]
  # Call the dedicated function to compute the DFP filter for a specified lead
  # (see dfp_from_tau_func for the derivation based on Theorem 2, Wildi 2026)
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
colnames(b0_mat)<-colnames(cor_vec_mse_la_mat)<-paste("Shift ",tau_vec,sep="")

b_tau<-b0_mat
cor_vec_mse_la_mat_tau<-cor_vec_mse_la_mat

# Compute Gamma(0)
Gamma0_tau<-apply(b_tau,2,sum)
# Compute time-shifts at frequency zero
tau_tau<--as.vector(t(b_tau)%*%(0:(L-1))/apply(b_tau,2,sum))
tau_tau[which(Gamma0_tau<0)]<-NA
alpha_tau<-as.vector(t(b_tau)%*%gamma0)
mse_b<-c(t(gammah)%*%gamma0/sqrt(t(gammah)%*%gammah*t(gamma0)%*%gamma0),1,-as.double(t(gammah)%*%(0:(L-1))/sum(gammah)),t(gammah)%*%gamma0)

table_tau<-cbind(mse_b,rbind(cor_vec_mse_la_mat_tau[1,],cor_vec_mse_la_mat_tau[h+1,],tau_tau,alpha_tau))
dim(table_tau)

b_tau[,4]%*%gamma_target/sqrt(b_tau[,4]%*%b_tau[,4]*gamma_target%*%gamma_target)


# Plots
par(mfrow=c(2,2))
ts.plot(b_tau,col=rainbow(ncol(cor_vec_mse_la_mat_tau)),main="b0, Tau")
ts.plot(cor_vec_mse_la_mat_tau,col=rainbow(ncol(cor_vec_mse_la_mat_tau)),main="CCF, Tau")
ts.plot(scale(b_alpha0,center=F,scale=T),col=rainbow(ncol(cor_vec_mse_la_mat_alpha0)),main="b0 scaled, alpha0")
ts.plot(cor_vec_mse_la_mat_alpha0,col=rainbow(ncol(cor_vec_mse_la_mat_alpha0)),main="CCF, alpha0")


#---------------------------------
# Checks
# Expressions should vanish
t(b_alpha0)%*%gamma_constraint-alpha0_vec
t(b_tau)%*%(0:(L-1))/apply(b_tau,2,sum)-tauh-tau_vec



#----------------------------------
# Plots
# Compute CCF of HP-trend



# Plots: filter coefficients and CCF

par(mfrow=c(2,2))

#-------
# 1. DFP based on alpha0
# 1.1 b

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


# 1.2 CCF
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

# 2 Time-shift DFP 
# 

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

# 2.2 Time-shift
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
# 1. Designs based on decoupling alpha0
colo<-c("green",rainbow(ncol(b_alpha0)))

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





colnames(table_alpha0) <- c("MSE","DFP~$\\alpha_0=0.1$", alpha0_vec[2:length(alpha0_vec)])
rownames(table_alpha0) <- c("CCF(0)", "CCF(h=2)","$\\Gamma(0)$","$\\tau$")
xt <- round(table_alpha0,4)
xt

  
colnames(table_tau) <- c("MSE","DFP~Lead~=~$1$", -tau_vec[2:(length(tau_vec)-1)],"$\\infty$")
rownames(table_tau) <- c("CCF(0)", "CCF(h=2)","$\\tau$","$\\alpha_0(\\tau)$")
table_tau[3,5]<-Inf
xt <- round(table_tau,3)
xt


#######################################################################################
# Exercise 3
#######################################################################################
# Max-Tau
# Design: replicate target correlation of time-shift DFP and compare shifts of Max-Tau vs. time-shift DFP
# Max-Tau should outperform the latter at the reference frequencies (omega0=0 and omega0=pi/20)
  
  
# 1. Compute target correlations of time-shift DFP: these are used as constraints for max-tau
# Note: the DFP above relies on TC normalized with 1/|gamma0| (1/|hp_trend|). However, the Max-Tau normalizes with 1/|gammah| (1/gamma_target). So we have to recompute the TC with the new re-scaling to obtain the correct Max-Tau which match the DFP.
target_correlation_vec<-NULL
# Scaling is required because alpha_h has the meaning of a correlation
for (i in 1:ncol(b_tau))
{
  if (T)
  {
    # Correlation with MSE gammah instead of gamma0 (the rescaling is necessary for Max-Tau)    
    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*gamma_target%*%gamma_target))
    #    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*hp_trend%*%hp_trend))
  } else
  {
    # Correlation with nowcast  
    target_correlation_vec<-c(target_correlation_vec,b_tau[,i]%*%gamma_target/sqrt(b_tau[,i]%*%b_tau[,i]*hp_trend%*%hp_trend))
  }
}

#---------------------------
# 2. Compute Max-Tau: omega0=pi/20: 10 year cycle
gamma_h<-gamma_target
# Ten year periodicity
n_freq<-20
omega0<-pi/n_freq
# Impose maximal lead at omega0 with respect to MSE target gamma_h (phase_excess<-T) or with respect to identity (phase_excess<-F). If gamma_h is lagging at omega0, then phase_excess<-F will generate a larger lead when feasible.
# Note that the maximal phase lead is restricted to pi/2 (n_freq/4 in time units) with respect to gamma_h or identity, ensuring strict positivity (like the original time-shift DFP).
# Effect of phase_excess:
# -Maximizing the lead relative to zero or gammah is the same optimization problem.
# -Only difference: the upper boundary of pi/2 with respect to 0 or gammah
# -As long as both leads (with respect to 0 or gammah) are below pi/2, the solution is exactly the same.
phase_excess<-F
# Notes: 
# 1. When feasible, Max-Tau generates the maximal lead namely n_freq/2
# 2. Maximizing the imaginary part of the transferfunction (for given real part) implies that 
#     b_opt will have a periodic component
# 3. However, The imaginary part vanishes at frequency zero, which generally conflicts with 
#     with the target correlation when the target is a lowpass (not vanishing at frequency zero).

#gamma_target<--gamma_target
b_tau_max<-max_tau_vec<-max_tau_excess_vec<-NULL
for (i in 1:length(target_correlation_vec))#i<-1
{
  target_correlation<-target_correlation_vec[i]
  #  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,epsilon)
  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,omega0,phase_excess)
  #  design_optimal_filter(gamma_target, target_correlation,omega0=0) 
  
  #  max_tau_obj<-optimize_filter( omega0, gamma_h, target_correlation, epsilon)
  b_tau_max<-cbind(b_tau_max,max_tau_obj$b_opt)
  max_tau_vec<-c(max_tau_vec,max_tau_obj$tau_max)
  max_tau_excess_vec<-c(max_tau_excess_vec,max_tau_obj$tau_max_excess)
}#
# Take minus sign to match sign convention in paper (in code it is assumed that transferfunction is based on exp(+i*omega) whereas in paper we assume exp(-1.i*omega)).
# A lead means a positive number (of the sign inverted tau):
max_tau_vec
max_tau_excess_vec
# The difference between max_tau_excess_vec and max_tau_vec is tau_h, the time shift of the MSE:
tauh<--(Arg(gamma_h%*%exp(1.i*omega0*(0:(L-1))))/omega0)
tauh
# Note: if tauh<0 then the MSE has a lag (compared to the identity). Therefore, setting phase_excess<-F will generate a larger lead, limited only by pi/2 with respect to identity (instead of pi/2 with respect to lagging MSE).

b_tau_max_omega<-b_tau_max
filt_mat_tau_max_omega<-cbind(gamma_target,b_tau_max_omega)




#-------------------
# Checks: 
# a. Solution is linear combination of gamma_target, \mathbf{1} and \mathbf{k}
summary(lm(b_tau_max_omega[,i]~cbind(gamma_target,cos(omega0*(0:(L-1))),sin(omega0*(0:(L-1))))-1))

# b. Target correlation constraint
target_correlation_check<-NULL
# Scaling is required because alpha_h has the meaning of a correlation
for (i in 1:ncol(b_tau))
  target_correlation_check<-c(target_correlation_check,b_tau_max_omega[,i]%*%gamma_target/sqrt(b_tau_max_omega[,i]%*%b_tau_max_omega[,i]*gamma_target%*%gamma_target))
# Check: should vanish
# Note: CCF(h) computed for the table differs because the reference is hp_trend (the nowcast), not gamma_target (the MSE forecast), in computing the CCF. This is just a scaling effect.
target_correlation_check-target_correlation_vec

# c. Check unit length constraint: should vanish
diag(t(b_tau_max_omega)%*%b_tau_max_omega)-1


#----------------------------
# 3. Max-Tau optimized at trend-frequency zero
omega0<-0
#phase_excess<-F
b_tau_max<-max_tau_vec<-max_tau_excess_vec<-NULL
for (i in 1:length(target_correlation_vec))#i<-1
{
  target_correlation<-target_correlation_vec[i]
  #  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,epsilon)
  max_tau_obj<-max_tau_dual_func(gamma_target, target_correlation,omega0,phase_excess)
  #  design_optimal_filter(gamma_target, target_correlation,omega0=0) 
  
  #  max_tau_obj<-optimize_filter( omega0, gamma_h, target_correlation, epsilon)
  b_tau_max<-cbind(b_tau_max,max_tau_obj$b_opt)
  max_tau_vec<-c(max_tau_vec,max_tau_obj$tau_max)
  max_tau_excess_vec<-c(max_tau_excess_vec,max_tau_obj$tau_max_excess)
}#
# Take minus sign to match sign convention in paper (in code it is assumed that transferfunction is based on exp(+i*omega) whereas in paper we assume exp(-1.i*omega)).
# A lead means a positive number (of the sign inverted tau):
max_tau_vec
max_tau_excess_vec
# The difference between max_tau_excess_vec and max_tau_vec is tau_h, the time shift of the MSE:
if (omega0==0)
{
  tauh0<--gamma_target%*%(0:(L-1))/sum(gamma_h)
} else
{
  tauh0<--Arg(sum(gamma_target*exp(1.i*omega0*(0:(L-1)))))/omega0
}
tauh0
# Note: if tauh<0 then the MSE has a lag (compared to the identity). Therefore, setting phase_excess<-F will generate a larger lead, limited only by pi/2 with respect to identity (instead of pi/2 with respect to lagging MSE).
b_tau_max_0<-b_tau_max

#-------------------
# Checks: 
# a. Solution is linear combination of gamma_target, \mathbf{1} and \mathbf{k}
summary(lm(b_tau_max_0[,i]~cbind(gamma_target,rep(1,L),(0:(L-1)))-1))

# b. Target correlation constraint
target_correlation_check<-NULL
# Scaling is required because alpha_h has the meaning of a correlation
for (i in 1:ncol(b_tau))
  target_correlation_check<-c(target_correlation_check,b_tau_max_0[,i]%*%gamma_target/sqrt(b_tau_max_0[,i]%*%b_tau_max_0[,i]*gamma_target%*%gamma_target))
# Check: should vanish
# Note: CCF(h) computed for the table differs because the reference is hp_trend (the nowcast), not gamma_target (the MSE forecast), in computing the CCF. This is just a scaling effect.
target_correlation_check-target_correlation_vec

# c. Check unit length constraint: should vanish
diag(t(b_tau_max_0)%*%b_tau_max_0)-1

#-----------------------------------------
# 4. Compute time-shifts of DFP and Max-Tau designs at both frequencies: all have identical target correlations

# a. omega0=0

tau_mat_0<-NULL
for (i in 1:ncol(b_tau))
{
  tau_mat_0<-rbind(tau_mat_0,-c(b_tau[,i]%*%(0:(L-1))/sum(b_tau[,i]), b_tau_max_0[,i]%*%(0:(L-1))/sum(b_tau_max_0[,i]), b_tau_max_omega[,i]%*%(0:(L-1))/sum(b_tau_max_omega[,i])))
}
tau_mat_0[nrow(tau_mat_0),1:2]<-Inf
colnames(tau_mat_0)<-c("DFP","Max-Tau(0)","Max-Tau(pi/20)")
rownames(tau_mat_0)<-round(target_correlation_vec,3)
tau_mat_0

# b. omega0=pi/20
omega<-pi/n_freq
tau_mat_omega<-NULL
for (i in 1:ncol(b_tau))
{
  tau_mat_omega<-rbind(tau_mat_omega,-c(Arg(b_tau[,i]%*%exp(1.i*omega*(0:(L-1)))), Arg(b_tau_max_0[,i]%*%exp(1.i*omega*(0:(L-1)))),Arg(b_tau_max_omega[,i]%*%exp(1.i*omega*(0:(L-1)))))/omega) 
}
colnames(tau_mat_omega)<-c("DFP","Max-Tau(0)","Max-Tau(pi/20)")
rownames(tau_mat_omega)<-round(target_correlation_vec,3)
tau_mat_omega

filt_mat_tau_max_0<-cbind(gamma_target,b_tau_max_0)
colnames(filt_mat_tau_max_0)<-c("MSE",paste("\u03C4","=",round(tau_mat_0[,2],2),sep=""))

colnames(filt_mat_tau_max_omega)<-c("MSE",paste("\u03C4","=",round(tau_mat_omega[,3],2),sep=""))


#----------------------------------
# Plots
# Compute CCF of HP-trend



# Plots: filter coefficients and CCF

colo<-c("green",rainbow(ncol(b_tau_max_0)))

par(mfrow=c(2,2))



# 1 DFP based on Max-Tau(0)
# 2.1 b_tau 


# Scale filter coefficients:
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

# 2.2 Time-shift
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


# 2 Max-tau based on omega0=pi/20

# Scale filter coefficients:
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

# 3.2 Time-shift
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
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
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
# 1. Designs based on decoupling alpha0
colo<-c("green",rainbow(ncol(b_tau_max_0)))

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

anf<-108
enf<-134

mplot<-scale(y_mat_tau_max_0)[anf:enf,]

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



# 2. Based on time-shift DFP


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

anf<-108
enf<-134

mplot<-scale(y_mat_tau_max_omega)[anf:enf,]

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




tab2 <- tau_mat_omega
tab1 <- tau_mat_0

colnames(tab1)[ncol(tab1)] <- colnames(tab2)[ncol(tab2)] <- "Max-Tau($\\pi/$20)"

tab1_df <- as.data.frame(" " = rownames(tab1), tab1, 
                         check.names = FALSE, stringsAsFactors = FALSE)
xt1 <- tab1_df
tab2_df <- as.data.frame(" " = rownames(tab2), tab2, 
                         check.names = FALSE, stringsAsFactors = FALSE)
xt2 <- tab2_df

# Here we introduce target correlation based on normalization with gammah (gamma_target).
# This the correct in this context. But the time-shift DFP has its CCF normalizing with respect to the nowcast gamma0 (hp_trend). So we refer to the latter. The solution is not affected by 
# the scaling.
# a) Normalize with respect to gammah
xt1 <- cbind(TC = rownames(tab1), as.data.frame(tab1, check.names = FALSE))
# b) Normalize with respect to gamma0 (relying on previous DFP results)
xt1 <- cbind(TC = table_tau[2,2:ncol(table_tau)], as.data.frame(tab1, check.names = FALSE))

xt1 <- cbind(TC = table_tau[2,2:ncol(table_tau)], as.data.frame(tab1, check.names = FALSE))

xt1


####################################################################################
# Additional exercises not in paper:
# 1. Show equivalence of primal and dual on frontier (an d difference away from frontier)
# 2. Multi-frequency approach
# 3. Apply Curvature Max-Tau to BCA





