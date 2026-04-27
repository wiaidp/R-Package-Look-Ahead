# Leading indicator PCS

# Aggreagte lead instead of lead at frewuency zero (overfittting, DFP)
# PCS relies also on decoupling but not from xt. Instead it decouples from hammah-gamma_{h-1}



# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

####################################################
# R code for solving phase excess theta as a function of betah


# Define lengths of MSE predictors and angle thetah between them
# We use the same gammah, gammahm1 as in above plot
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
betah<--0.1

lh<-sqrt(sum(gammah^2))
lhm1<-sqrt(sum(gammahm1^2))
thetah<-atan2(gammah[2],gammah[1])- atan2(gammahm1[2],gammahm1[1])


a<-lh-cos(thetah)*lhm1
b<-sin(thetah)*lhm1
c<--betah



solve_acos_bsin_eq <- function(a, b, c) {
  R <- sqrt(a^2 + b^2)
  if (R == 0) {
    if (c == 0) return(list(status = "infinite solutions (all theta)", theta = NULL, phi = NA, R = 0))
    return(list(status = "no solution", theta = NULL, phi = NA, R = 0))
  }
  phi <- atan2(b, a)                # phase shift
  x <- c / R
  # Clamp for numerical safety
  x <- max(min(x, 1), -1)
  if (abs(c) > R + .Machine$double.eps^0.5) {
    return(list(status = "no real solution (|c| > R)", theta = NULL, phi = phi, R = R))
  }
  if (abs(abs(x) - 1) < 1e-14) {
    # Single solution modulo 2π
    theta <- if (x > 0) phi else (phi + pi)
    theta <- atan2(sin(theta), cos(theta))  # wrap to (-pi, pi]
    return(list(status = "one solution modulo 2π", theta = theta, phi = phi, R = R))
  }
  alpha <- acos(x)
  theta1 <- phi + alpha
  theta2 <- phi - alpha
  # Wrap to (-pi, pi]
  wrap <- function(t) atan2(sin(t), cos(t))
  theta <- sort(c(wrap(theta1), wrap(theta2)))
  list(status = "two solutions modulo 2π", theta = theta, phi = phi, R = R)
}

# Find theta for given a,b,c
solve_acos_bsin_eq(a, b, c )

#########################################################
























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



# Load data from FRED using the alfred library (no API key required).
install.packages("alfred")
library(alfred)
library(xts)
library(mFilter)

source(paste(getwd(), "/R/DFP.r", sep = ""))


# Load tau-statistic (measures lead/lag performance)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used in JBCY (requires mFilter)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

source(paste(getwd(),"/R utility functions/HP_JBCY_functions.r",sep=""))






# Vector components (edit these)
# Vector components (edit these)
vx <- 2
vy <- 2
vx <- 2
vy <- 2

# Specify gammahm1 and gammah
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
# Specify lambda0
lambda0<-0.3
# Lengths
l0<-sqrt(sum(gammahm1^2))
lh<-sqrt(sum(gammah^2))

# Angle between gammah and gammahm1: gammah is above (larger angle)
theta_h <- atan2(gammah[2], gammah[1])-atan2(gammahm1[2], gammahm1[1])

# Set up plot limits with some padding
x_min<-0
x_max<-1.5
y_min<--0.2
y_max<-1.2
lim <- 1.2 * max(1, abs(c(vx, vy))+0.5)
plot(NA, xlim = c(x_min,x_max+0.3), ylim = c(y_min, y_max),xlab = "", ylab = "", axes = TRUE,asp=1)

#     asp = 1, xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gammahm1
arrows(0, 0,gammahm1[1],gammahm1[2], length = 0.12, lwd=1, col = "black")
text(gammahm1[1]+0.1,gammahm1[2], labels = expression(gamma[h-1]), col = "black", cex = 1.2)
# gammah
arrows(0, 0,gammah[1],gammah[2], length = 0.12, lwd=1, col = "black")
text(gammah[1]+0.1,gammah[2], labels = expression(gamma[h]), col = "black", cex = 1.2)
# Insert unit length b1: first solution corresponding to beta=0
b0<-c(0.7,1.05)
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.1,b0[2]/lb0, labels = expression(b[1]), col = "red", cex = 1.2)
ls<-2
segments(0,0,ls*b0[1],ls*b0[2],  lwd = 1,lty=2, col = "red")

# Orthogonal projection of gammah onto b0
x1<--0.27
theta_gammah<-atan(b0[2]/b0[1])
segments(gammah[1],gammah[2],gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah), lwd = 1,lty=2, col = "red")
#text(gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah)+0.03, labels =expression(b[0]*gamma[h])) 
# Orthogonal projection of gammahm1 onto b0
x1<--1
theta_gammahm1<-atan(b0[2]/b0[1])
segments(gammahm1[1],gammahm1[2],gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1), lwd = 1,lty=2, col = "red")
text(gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[1]*gamma[h-1]),col="red") 
text(gammahm1[1]+x1*sin(theta_gammahm1)+0.1,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels ="=",col="red") 
text(gammahm1[1]+x1*sin(theta_gammahm1)+0.18,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[1]*gamma[h]),col="red") 


#text(gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[0]*gamma[h-1]-b[0]*gamma[h]=0)) 

# Angle between gammah and gammahm1
theta_h <- atan2(gammah[2], gammah[1])-atan2(gammahm1[2], gammahm1[1])
# Draw the angle 
r <- 0.4 * lh  # arc radius
th_seq <- atan2(gammahm1[2], gammahm1[1])+seq(0, theta_h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)
text(1.15 * r * cos(th_seq[length(th_seq)]), 1.15 * r * sin(th_seq[length(th_seq)]-0.1),
     labels = expression(theta[hh-1]), col = "black", cex = 1.2)

# Angle between gammah and b1
theta_hm1 <- atan2(b0[2], b0[1])-atan2(gammah[2], gammah[1])
# Draw the angle theta_h (between gammah and b0)
r <- 0.3 * lh  # arc radius
thm1_seq <- atan2(gammah[2], gammah[1])+seq(0, theta_hm1, length.out = 100)
lines(r * cos(thm1_seq), r * sin(thm1_seq), col = "red", lwd=1)
text(1.15 * r * cos(thm1_seq[length(thm1_seq)])+0.05, 1.15 * r * sin(thm1_seq[length(thm1_seq)])-0.05,
     labels = expression(theta[hb1]), col = "red", cex = 1.2)

# Unit circle
theta <- atan2(b0[2], b0[1])-atan2(gammahm1[2], gammahm1[1])
r <- 1  # arc radius
th_seq <- seq(-0.2, pi/2+0.2, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1,lty=2)


# Insert second unit length b2: with beta<0
b0<-c(0.4,1.05)
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "blue")
text(b0[1]/lb0-0.1,b0[2]/lb0-0.05, labels = expression(b[2]), col = "blue", cex = 1.2)
ls<-2
segments(0,0,ls*b0[1],ls*b0[2],  lwd = 1,lty=2, col = "blue")

# Orthogonal projection of gammah onto b0
x1<--0.83
theta_gammah<-atan(b0[2]/b0[1])
segments(gammah[1],gammah[2],gammah[1]+x1*sin(theta_gammah),gammah[2]-x1*cos(theta_gammah), lwd = 1,lty=2, col = "blue")
text(gammah[1]+x1*sin(theta_gammah)-0.05,gammah[2]-x1*cos(theta_gammah)+0.03, labels =expression(b[2]*gamma[h]),col="blue") 
# Orthogonal projection of gammahm1 onto b0
x1<--1.29
theta_gammahm1<-atan(b0[2]/b0[1])
segments(gammahm1[1],gammahm1[2],gammahm1[1]+x1*sin(theta_gammahm1),gammahm1[2]-x1*cos(theta_gammahm1), lwd = 1,lty=2, col = "blue")
text(gammahm1[1]+x1*sin(theta_gammahm1)-0.08,gammahm1[2]-x1*cos(theta_gammahm1)+0.03, labels =expression(b[2]*gamma[h-1]),col="blue") 

# Angle between gammah and b2
theta_hm1 <- atan2(b0[2], b0[1])-atan2(gammah[2], gammah[1])
# Draw the angle theta_h (between gammah and b0)
r <- 0.2 * lh  # arc radius
thm1_seq <- atan2(gammah[2], gammah[1])+seq(0, theta_hm1, length.out = 100)
lines(r * cos(thm1_seq), r * sin(thm1_seq), col = "blue", lwd=1)
text(1.15 * r * cos(thm1_seq[length(thm1_seq)])+0.08, 1.15 * r * sin(thm1_seq[length(thm1_seq)])-0.02,
     labels = expression(theta[hb2]), col = "blue", cex = 1.2)

















# R code for solving phase excess theta as a function of betah


# Define lengths of MSE predictors and angle thetah between them
# We use the same gammah, gammahm1 as in above plot
gammahm1<-c(3,1)*0.44*1.2
gammah<-c(1.5,1)*0.66*1.2
betah<--0.1

lh<-sqrt(sum(gammah^2))
lhm1<-sqrt(sum(gammahm1^2))
thetah<-atan2(gammah[2],gammah[1])- atan2(gammahm1[2],gammahm1[1])


a<-lh-cos(thetah)*lhm1
b<-sin(thetah)*lhm1
c<--betah



solve_acos_bsin_eq <- function(a, b, c) {
  R <- sqrt(a^2 + b^2)
  if (R == 0) {
    if (c == 0) return(list(status = "infinite solutions (all theta)", theta = NULL, phi = NA, R = 0))
    return(list(status = "no solution", theta = NULL, phi = NA, R = 0))
  }
  phi <- atan2(b, a)                # phase shift
  x <- c / R
  # Clamp for numerical safety
  x <- max(min(x, 1), -1)
  if (abs(c) > R + .Machine$double.eps^0.5) {
    return(list(status = "no real solution (|c| > R)", theta = NULL, phi = phi, R = R))
  }
  if (abs(abs(x) - 1) < 1e-14) {
    # Single solution modulo 2π
    theta <- if (x > 0) phi else (phi + pi)
    theta <- atan2(sin(theta), cos(theta))  # wrap to (-pi, pi]
    return(list(status = "one solution modulo 2π", theta = theta, phi = phi, R = R))
  }
  alpha <- acos(x)
  theta1 <- phi + alpha
  theta2 <- phi - alpha
  # Wrap to (-pi, pi]
  wrap <- function(t) atan2(sin(t), cos(t))
  theta <- sort(c(wrap(theta1), wrap(theta2)))
  list(status = "two solutions modulo 2π", theta = theta, phi = phi, R = R)
}

# Find theta for given a,b,c
solve_acos_bsin_eq(a, b, c )


###############################################################################
# Example BCA

# Application of DFP to quarterly GDP data

# Source data directly from FRED
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

# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application of SSA 
#     more cumbersome, counter-intuitive, unpredictable and hazardous (try applying a filter to a xts-object...).
# We here skip the pandemic: outliers affect HF-regression.
# Effects of the pandemic are analyzed in our last example. 

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


lambda_hp<-1600
# L is an odd integer such that the symmetric filter is centered at (L-1)/2+1  
L<-51
h<-delta<-4
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
cor_vec_mse_la_mat<-NULL
b0_mat<-matrix(ncol=length(beta_vec),nrow=L)
lambda1_vec<-lambda2_vec<-NULL
for (i in 1:length(beta_vec))
{ 
  beta<-beta_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b0_obj<-DFP_compute_lambda_alpha0_func(gamma_constraint,gamma_target,h,L,beta)
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










