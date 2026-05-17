# ════════════════════════════════════════════════════════════════════
# TUTORIAL 11 — BUSINESS CYCLE ANALYSIS AND LEADING INDICATOR DESIGN
# ════════════════════════════════════════════════════════════════════

# ── MACRO INDICATOR DESIGN ───────────────────────────────────────────
#
# This section follows the Leading Indicator Design (LID) framework
# introduced in Wildi (2026), Section 3.5.
#
# Let x_t be a stationary indicator of interest — for example, the first
# difference of a non-stationary macro series such as industrial production,
# employment, income, or GDP. The target signal is defined as:
#
#   Phi' * X_t  (AR form)   ≡   gamma' * Epsilon_t  (MA form)
#
# Where Phi and gamma are vectors of length L and X_t=(x_t,...,X-{t-L+1}), 
# Epsilon_t=(epsilon_t,...,epsilon_{t-L+1}).

# Typical targets Phi (or gamma) include trend-, cycle-, or
# seasonal adjustment filters and  Phi' * X_t represents signal-growth: trend-, 
# cycle-, or seasonally adjusted growth. 

# Stationary signal-growth is often more relevant to analysts and decision 
# makers than non-stationary signal-level.

# Throughout this tutorial we work in the MA form. Here, gamma is the 
# convolution of the Wold decomposition xi of x_t with the filter Phi:
#
#   gamma = Phi ∘ xi,   where ∘ denotes convolution.
#
# Let gamma_k denote the minimum mean-squared-error (MSE) predictor of the
# signal at horizon k.

# ── LEADING INDICATOR DESIGN (LID) ─────────────────────────────────────────
#
# The optimization problem is:
#
#   Minimize  (b - gamma)' (b - gamma)          [MSE objective]
#   subject to  b' * (gamma_h - gamma_{h-1}) = beta   [lead constraint]
#
# See section 3.3, Wildi (2026). This problem is related (though not identical) 
# to the Type II PCS approach introduced in Tutorial 12. 

# The objective minimizes the distance from the causal (nowcast) filter
# gamma; no explicit forecasting step is involved. The hyperparameters
# h > 0 and beta >= 0 in the constraint jointly govern the profile of the 
# cross-correlation function (CCF) at the specified lead:
#
#   - beta > 0 : the CCF peak cannot be located in h-1.
#   - beta = 0 : the CCF is constrained to be flat between lags h-1 and h.
# 
# Under some circumstances, these constraints can determine an effective shift 
# of the CCF at k >= h, see examples below.
#
# ── HP TREND: BUSINESS-CYCLE ANALYSIS ─────────────────────────
#
# When the HP trend (Phi) is applied to the first differences (X_t) of the data, 
# the resulting indicator estimates current growth (drift):
#
#   Positive values → economic expansion
#   Negative values → economic contraction / recession
#
# This constitutes a straightforward form of business-cycle analysis. 
#
# - HP-trend tracks the level of the first differences (trend-growth), 
#   and `cyclical' up- and down-turns are triggered by changes in the 
#   underlying data growth-rate due to transitions between economic phases 
#   of expansion and contraction.  
# - These dynamics are endogenous to the data, not the filter.
# - Applying a trend filter to first differences tracks effective growth, 
#   thereby mitigating the problem of spurious cycle of conventional 
#   business-cycle designs, see Wildi (2014).

# Background is provided in the M-SSA Tutorial Series (Tutorial 2 of M-SSA).  

# ── SIMPLE VS. CHALLENGING FORECAST PROBLEMS ─────────────────────────
#
# In general, imposing a flat CCF at lag h (beta = 0) does not guarantee
# that the global CCF peak occurs exactly at lag h. However, for the
# present business-cycle application — which combines the HP filter with
# the Type II PCS design — the problem is relatively well-conditioned:
# the CCF peak is naturally shifted to h = 0 as a direct consequence of
# the DGP-implied filter gamma (HP convolved with xi).
#
# In more demanding forecasting settings (covered in Tutorials 12–15),
# such a peak shift may not arise automatically and may require more
# elaborate constraint specifications or alternative PCS designs.

# ═════════════════════════════════════════════════════════════════════
# ── REFERENCES ───────────────────────────────────────────────────────
#
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     arXiv preprint. https://doi.org/10.48550/arXiv.2602.23087
#
# ═════════════════════════════════════════════════════════════════════



# ── INITIALISATION ─────────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the PCS optimisation routines.
source(paste(getwd(), "/R/PCS.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/HP_JBCY_functions.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

library(mFilter)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
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

# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application 
# more challenging (as an example, applying a filter to a xts-object reverts time).
# We here skip the pandemic: outliers affect the design 

end_year<-2024
start_year<-1992
y<-as.double(log(GDPC1[paste(start_year,"/",end_year,sep="")]))
y_xts<-log(GDPC1[paste(start_year,"/",end_year,sep="")])
len<-length(y)
use_adjusted_hamilton<-F
use_new_i2_adjustment<-F


# Plot
par(mfrow=c(2,2))
plot(GDPC1,main="US GDP")
plot(y_xts,main="Log-GDP")
plot(diff(y_xts),main="Diff-log")
acf(na.exclude(diff(y_xts)),main="ACF of log-diff")


# The LID design relies on a single constraint and can be operationalized either 
# in DFP form or in PCS form. Exercise 1 presents the DFP implementation and exercise 2 
# presents the PCS implementation.


# ─────────────────────────────────────────────────────────────────────
# Example 1. PCS Leading Indicator Design: Rely on DFP Optimization
# ─────────────────────────────────────────────────────────────────────
# Application of DFP to quarterly GDP data

# ─────────────────────────────────────────────────────────────────────
# 1.1 HP Set-Up
# ─────────────────────────────────────────────────────────────────────


# HP setting for quarterly data: lambda = 1600
lambda_hp<-1600
# L is an odd integer such that the symmetric filter is centered at (L-1)/2+1  
L<-51
# One year ahead forecast horizon
h<-delta<-4
# Setting for computing the CCF: has no effect on predictor.
max_lag<-0


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


# ─────────────────────────────────────────────────────────────────────
# 1.2 DFP Set-Up
# ─────────────────────────────────────────────────────────────────────

beta_vec<-c(0.8,0.6,0.4,0.2,0)


# Specify gamma at forecast horizon sup_vec_target=delta and at lead/lag sup_vec_constraint
# Classic h-step ahead predictor
gamma_target<-hp_trend_long[h+1:L]
# Nowcast: the design is a leading indicator PCS
gamma_target<-hp_trend_long[1:L]
# Some checks
ts.plot(gamma_target)
# Should add to one
sum(hp_trend_long)
# STD
sqrt(t(hp_trend_long)%*%hp_trend_long)

# 2. Want flat CCF at h
gamma_constraint<-hp_trend_long[delta-1+1:L]-hp_trend_long[delta+1:L]
ts.plot(gamma_constraint)

# ─────────────────────────────────────────────────────────────────────
# 1.3 Run DFP
# ─────────────────────────────────────────────────────────────────────

cor_vec_mse_la_mat<-NULL
b0_mat<-matrix(ncol=length(beta_vec),nrow=L)
lambda1_vec<-lambda2_vec<-NULL
for (i in 1:length(beta_vec))
{ 
  beta<-beta_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b0_obj<-unitary_DFP_func(gamma_constraint,gamma_target,beta)
  
  b0_mat[,i]<-b0_obj$b0
  lambda1_vec<-c(lambda1_vec,b0_obj$lambda1)
  lambda2_vec<-c(lambda2_vec,b0_obj$lambda2)
  
  # Compute CCF of PCS predictors  
  cor_vec_mse_la_mat<-cbind(cor_vec_mse_la_mat,compute_acf_at_lags_zero_delta_func(max_lag,h,b0_mat[,i],hp_trend)$cor_vec)
}

#?????????????????????????????????????????????????
# MSE-PCS is     MSE-PCS = gammah + lambda * (gamma_{h-1} - gammah)
# Check that this is indeed MSE...
# Problem:
# The weight on gammah should be 1.
# On the other hand the MSE predictor is given by projecting gammah orthogonally to plan 
# spanned by PCS constraint, i.e. above formula.

# Which argument is correct? The following piece shows that MSE-PCS as in paper is correct
#???????????????????????????????????????????????????????
# Note: select either gamma_target<-hp_trend_long[h+1:L]
# or gamma_target<-hp_trend_long[1:L] as targets above.

if (F)
{
  # Verify which PCS is MSE optimal
  k<-ncol(b0_mat)
  k<-4
  b<-b0_mat[,k]
  lambda1<-lambda1_vec[k]
  lambda2<-lambda2_vec[k]
  # 1. MSE-PCS as in paper: gammah+lambda(gamma_{h-1}-gammah) (divide above b0 by lambda1)
  b<-b/lambda1
  mean((b-gamma_target)^2)
  # 1. MSE-PCS with overall weight 1 on gammah: gammah+lambda(gamma_{h-1}-gammah)
  b<-b0_mat[,k]
  b<-b/(lambda1-lambda2)
  mean((b-gamma_target)^2)
  
}



colnames(b0_mat)<-colnames(cor_vec_mse_la_mat)<-beta_vec
# Check unit length:
apply(b0_mat^2,2,sum)

# ─────────────────────────────────────────────────────────────────────
# 1.4 Routine Checks
# ─────────────────────────────────────────────────────────────────────


ts.plot(b0_mat,col=rainbow(ncol(cor_vec_mse_la_mat)))
ts.plot(cor_vec_mse_la_mat,col=rainbow(ncol(cor_vec_mse_la_mat)))
# Check 1: should be one on diagonal (unit length)
diag(t(b0_mat)%*%b0_mat)
# Check 2: should vanish
t(b0_mat)%*%gamma_constraint/as.double(sqrt(t(gamma_constraint)%*%gamma_constraint))-beta_vec

# CCF of MSE nowcast
cor_vec_mse<-compute_acf_at_lags_zero_delta_func(max_lag,h,b0_mat[,1],hp_mse)$cor_vec

# Compute CCF of HP-trend
cor_vec_t_hp_trend<-compute_acf_at_lags_zero_delta_func(max_lag,h,hp_trend,hp_trend)$cor_vec
ts.plot(cor_vec_t_hp_trend)
abline(v=max_lag)
abline(v=max_lag+h)



# ─────────────────────────────────────────────────────────────────────
# 1.5 Plots
# ─────────────────────────────────────────────────────────────────────

# Plots: filter coefficients and CCF


layout(matrix(c(1,2,3,3), 2, 2, byrow = T)) 

# Scale filter coefficients:
mplot<-cbind(gamma_target/as.double(sqrt(t(gamma_target)%*%gamma_target)),b0_mat)
colnames(mplot)<-c("MSE",paste("PCS: ",expression(beta),"=",beta_vec[1],sep=""),paste("PCS: ",expression(beta),"=",beta_vec[2:length(beta_vec)],sep=""))
colo<-c("green","orange","blue","red","black","violet")
#layout<-plot(mplot[,1],main=expression(paste("MSE and PCS for various ",beta)),axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
layout<-plot(mplot[,1],main="Predictors",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  if (i==2)
  {
    mtext(bquote(beta[4]==.(beta_vec[i-1])),col=colo[i],line=-i)
  } else
  {
    mtext(bquote(beta[4]==.(beta_vec[i-1])),col=colo[i],line=-i)
  } 
  
}
abline(h=0)
axis(1,at=c(1,1:(nrow(mplot)/10)*10),labels=c(1,1:(nrow(mplot)/10)*10)-c(1,rep(0,5)))
axis(2)
box()


mplot<-cbind(cor_vec_t_hp_trend,cor_vec_mse_la_mat)[1:20,]
colnames(mplot)<-c("HP concurrent",paste("PCS: ",expression(beta),"=",beta_vec,sep=""))
layout<-plot(mplot[,1],main="CCFs",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
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


# ─────────────────────────────────────────────────────────────────────
# 1.6 Generate Leading Indicators
# ─────────────────────────────────────────────────────────────────────



# Compute filter outputs: provide a slightly longer series because HP is longer than Hamilton

eps<-diff(log(GDPC1[paste(as.integer(start_year-L/4),"/",end_year,sep="")]))
len<-length(eps)


filt_mat<-cbind(gamma_target,b0_mat)
y_mat<-NULL
for (i in 1:ncol(filt_mat))
  y_mat<-cbind(y_mat,filter(eps,filt_mat[,i],side=1))
colnames(y_mat)<-c("HP",paste("PCS: ",expression(beta),"=",beta_vec,sep=""))
# Select subsample  
anf<-1
# Full length: length of series minus filter length L minus forecast horizon delta   
enf<-nrow(y_mat)
anf<-1
# Full length: length of series minus filter length L minus forecast horizon delta   
enf<-nrow(y_mat)



anf<-1
enf<-nrow(y_mat)
mplot<-y_mat[anf:enf,]


first_series<-scale(na.exclude(mplot[,1]))
layout<-plot(first_series,col=colo[1],main="Standardized forecasts", axes=F,type="l",xlab="",ylab="",lwd=1,ylim=c(min(scale(na.exclude(mplot))),max(scale(na.exclude(mplot)))))
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(scale(na.exclude(mplot[,i]),scale=T,center=T),col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  #  mtext(bquote(lambda==.(lambda_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat[,3]))+1):length(GDPC1)])
#axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[(length(label_vec)+1-length(na.exclude(mplot[,3]))):length(label_vec)])
axis(1,at=12*(1:(length(na.exclude(mplot[,3]))/12)),labels=label_vec[12*((length(label_vec)+1-length(na.exclude(mplot[,3]))):(length(label_vec)/12))])
axis(2)
box()







par(mfrow=c(1,3))
colo<-c("orange","green","blue","red","violet")

anf<-80
enf<-105
mplot<-y_mat[anf:enf,1:3]
mplot<-scale(y_mat[anf:enf,])

plot(mplot[,1],col=colo[1],main="Dotcom", axes=F,type="l",xlab="",ylab="",lwd=1,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  lines(mplot[,i],col=colo[i])
  #  mtext(colnames(y_mat)[i],col=colo[i],line=-i)
  mtext(bquote(beta[4]==.(beta_vec[i-1])),col=colo[i],line=-i)
}

abline(h=0)
label_vec<-(as.character(index(GDPC1))[(length(GDPC1)-length(na.exclude(y_mat[,3]))+1):length(GDPC1)])
axis(1,at=1:length(na.exclude(mplot[,3])),labels=label_vec[-L+anf:enf])
axis(2)
box()

anf<-108
enf<-132
mplot<-y_mat[anf:enf,]

mplot<-scale(y_mat[anf:enf,])

plot(mplot[,1],col=colo[1],main="Great recession", axes=F,type="l",xlab="",ylab="",lwd=1,ylim=c(min(mplot),max(mplot)))
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























#?????????????????????????????????????????????????
# MSE-PCS is     MSE-PCS = gammah + lambda * (gamma_{h-1} - gammah)
# Check that this is indeed MSE...
# Problem:
# The weight on gammah should be 1.
# On the other hand the MSE predictor is given by projecting gammah orthogonally to plan 
# spanned by PCS constraint, i.e. above formula.

# Which argument is correct?
#???????????????????????????????????????????????????????


#???????????????????????????????????????????????????????
# 2. Interpretability: DFP vs. PCS (Tutorial 4):
#    Even with the frequency-zero re-parameterisation of Exercise 3, the DFP
#    concept remains less directly interpretable than the Peak Correlation 
#    Shifting (PCS) predictor introduced in Tutorial 4, which, in its simplest 
#    form, is defined as:

#      MSE-PCS = gammah + lambda * (gamma_{h-1} - gammah)

#    In PCS, the look-ahead modification weighted by lambda addresses the lead of the predictor 
#    in AGGREGATE, i.e., not only at the trend-frequency omega=zero.  

# Moreover, PCS does not interpolate between gamma_{h-1} and gammah but instead 
# extrapolate, because lambda<0 whenn looking ahead.
#    Note also that because (gamma_{h-1} - gammah) is not proportional to
#    gamma0, AR-inversion no longer yields an identity convolution, so both
#    the MA and AR forms of the PCS predictor involve multiple coefficients
#    varying across designs — more complex than the DFP, but more interpretable.

# Finally, the above simplest form might not suffice to shift the CCF peak: then 
# the more complex PCS is required, i.e., PCS is inherently more complex than 
# just decoupling at present: PCS controls monotonicity of CCF from present to h,
# while DFP only controls present.
#?????????????????????????????????????????????????????????











