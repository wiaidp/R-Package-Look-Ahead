



# ════════════════════════════════════════════════════════════════════
# Exercise 4: ARMA(3,2)
# ════════════════════════════════════════════════════════════════════
# Second process: ARMA(3,2)
# AR-coefficients
ar1<-0.4
ar2<-0.3
ar3<-0.2
b1<-0.5
b2<-0.4

ts.plot(ARMAacf(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=100))


# Roots of characteristic polynomial
1/(Arg(polyroot(c(-ar3,-ar2,-ar1,1)))/pi)
abs(polyroot(c(-ar3,-ar2,-ar1,1)))

# MA inversion
# Compute long sequence: need more values than L for MSE forecasts below
gamma<-c(1,ARMAtoMA(ar=c(ar1,ar2,ar3),ma=c(b1,b2),lag.max=1000))
ts.plot(gamma[1:20])
# L-length now- and MSE forecast
gamma0<-gamma[1:L]
gammah<-gamma[h+(1:L)]


# Compute MSE-DFP
cor_vec_mat<-compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec
alpha0_vec<-c(0.9,0.45,0.22,0.1,0)
#alpha0_vec<-5*c(0.9,0.45,0.22,0.1,0)
b_mat<-b_mat_unscaled<-a_mat<-lambda_vec2<-NULL
cor_vec_2<-matrix(ncol=2,nrow=length(alpha0_vec))

for (i in 1:length(alpha0_vec))#i<-1
{
  alpha0<-alpha0_vec[i]
  # Function for deriving b0  
  b0<-compute_mse_dfp(alpha0,gamma0,gammah)$b0
  # This is the same as 
  lambda<-as.double((alpha0-t(gamma0)%*%gammah)/(t(gamma0)%*%gamma0))
  b0_simpler<-gammah+lambda*gamma0
  max(abs(b0-b0_simpler))
  b_mat_unscaled<-cbind(b_mat_unscaled,b0)
  
  lambda_vec2<-lambda
  # MSE scaling is not used anymore (closed-form solution is already MSE optimal)  
  if (F)
  {
    scale<-as.double(t(b0)%*%gammah/(t(b0)%*%b0))
    b0<-scale*b0
  }
  # Compute minimum phase DFP  
  if (use_min_phase)
  {
    # Compute roots      
    roots<-polyroot(b0)
    # Invert unstable roots      
    roots[which(abs(roots)>1)]<-1/roots[which(abs(roots)>1)]
    # Compute coefficients from roots: minimum phase version of DFP      
    b_min_phase<-Re(rev(poly_from_roots(roots)))
    # Check lag-one ACF: the same
    b0[1:(length(b0)-1)]%*%b0[2:length(b0)]/t(b0)%*%b0-b_min_phase[1:(length(b_min_phase)-1)]%*%b_min_phase[2:length(b_min_phase)]/b_min_phase%*%b_min_phase
    b0<-b_min_phase
  }
  # Use either solution
  b_mat<-cbind(b_mat,b0)
  # AR-inversion (problem: b0 is not always invertible)  
  a_mat<-cbind(a_mat,-ARMAtoMA(ar=-b0[2:L]/b0[1],lag.max=L))
  
  # Compute CCF  
  cor_vec<-compute_acf_at_lags_zero_delta_func(max_lag,h,b_mat[,ncol(b_mat)],gamma0)$cor_vec
  cor_vec_mat<-cbind(cor_vec_mat,cor_vec)
  # Extract CCF at lead 0 and h  
  cor_vec_2[i,1]<-cor_vec[1]
  cor_vec_2[i,2]<-cor_vec[1+h]
}

# Check DFP constraint: should vanish
# Note: neither b_mat nor gamma0 are scaled to unit length and therefore alpha0_vec is not a correlation
t(b_mat)%*%gamma0-alpha0_vec
# Alternative check DFP constraint: should vanish
# Note: since cor_vec is the CCF we have to scale alpha0_vec by inverse lengths of gamma0 and b
cor_vec_1[,1]-alpha0_vec/sqrt(diag((t(b_mat)%*%b_mat))*as.double(t(gamma0)%*%gamma0))
# Check AR-inversion: the max absolute deviations should vanish (all designs are non-invertible and therefore one observes numerical cancellation effects, i.e., absolute deviations do not vanish exactly)
for (i in  1:length(alpha0_vec))#i<-1
{
  b0<-b_mat[,i]
  ar_vec<-a_mat[,i]
  print(max(abs(c(1,ARMAtoMA(ar=ar_vec,lag.max=L-1))-b0/b0[1])))
}



mplot3<-scale(cbind(gammah,b_mat),center=F,scale=T)/sqrt(L-1)
#mplot3<-cbind(gammah,b_mat)

mplot4<-cor_vec_mat[1:22,]*as.double(sqrt(gamma0%*%gamma0)/sqrt(gamma%*%gamma))

#--------------
# AR-inversion of DFP: we rely on possibility B described above
# Solution B: compute DFP AR-weights by convolution of original AR weights with DFP (MA inversion)
# 1. Compute AR-inversion of ARMA
ar_inv<-c(1,ARMAtoMA(ar=-c(b1,b2),ma=-c(ar1,ar2,ar3),lag.max=L))
# 1.1 check: the following should give an identity
filt1<-ar_inv
filt2<-gamma
conv_two_filt_func(filt1,filt2)$conv[1:10]

# 2. Compute AR weights of MSE and DFP predictors
# a. MSE
filt2<-gammah
ar_mse_arma32<-conv_two_filt_func(filt1,filt2)$conv
# b. DFP
ar_dfp_arma32_mat<-NULL
for (i in 1:length(alpha0_vec))
{
  # Use original (unscaled) MSE-DFP
  filt2<-b_mat_unscaled[,i]
  ar_dfp_arma32_mat<-cbind(ar_dfp_arma32_mat,conv_two_filt_func(filt1,filt2)$conv)
  
}
ts.plot(cbind(ar_mse_arma32,ar_dfp_arma32_mat)[1:L,],col=rainbow(ncol(a_mat)),main="Method B: MSE and DFP predictors AR-inverted")
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
  y_dfp_ma32[i]<-b_mat_unscaled[,k]%*%eps[i:(i-L+1)]
  y_dfp_ar32[i]<-ar_dfp_arma32_mat[1:L,k]%*%x[i:(i-L+1)]
}
# Both series are identical up to negligible finite MA/AR inversion errors
ts.plot(cbind(y_dfp_ma32,y_dfp_ar32)[1:200,])
# Maximal error is negligible (due to finite length MA/AR inversions)
max(na.exclude(abs(y_dfp_ma32-y_dfp_ar32)[1:200]))


#-------------------------------------------------
# Plot


par(mfrow=c(2,2))

#ts.plot(gammah1,main=paste("MSE first process: h=",h,sep=""),col="green",xlab="",ylab="")

#ts.plot(gammah,main="Second process",col="green",xlab="",ylab="")


colo<-c("green","brown","orange","blue","violet","red")
# Scale filters
ts.plot(mplot1,main="Predictors: AR(3)",col=colo,xlab="",ylab="")
mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9, ",rho,"=",0.92)),line=-2,col=colo[2])
#mtext(expression(paste("    ",alpha[0],"=0.45, ",rho,"=",0.76)),line=-3,col=colo[3])
#mtext(expression(paste("    ",alpha[0],"=0.22, ",rho,"=",0.51)),line=-4,col=colo[4])
#mtext(expression(paste("    ",alpha[0],"=0.1, ",rho,"=",0.26)),line=-5,col=colo[5])
#mtext(expression(paste("    ",alpha[0],"=0, ",rho,"=",0.0)),line=-6,col=colo[6])
mtext(expression(paste("DFP ",alpha[0],"=0.9 ")),line=-2,col=colo[2])
mtext(expression(paste("    ",alpha[0],"=0.45 ")),line=-3,col=colo[3])
mtext(expression(paste("    ",alpha[0],"=0.22 ")),line=-4,col=colo[4])
mtext(expression(paste("    ",alpha[0],"=0.1 ")),line=-5,col=colo[5])
mtext(expression(paste("    ",alpha[0],"=0 ")),line=-6,col=colo[6])
abline(h=0)

cor_vec_1


ts.plot(mplot3,main="ARMA(3,2)",col=colo,xlab="",ylab="")
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("    ",alpha[0],"=0.9, ",rho,"=",0.99)),line=-2,col=colo[2])
#mtext(expression(paste("    ",alpha[0],"=0.45, ",rho,"=",0.96)),line=-3,col=colo[3])
#mtext(expression(paste("    ",alpha[0],"=0.22, ",rho,"=",0.84)),line=-4,col=colo[4])
#mtext(expression(paste("    ",alpha[0],"=0.1, ",rho,"=",0.58)),line=-5,col=colo[5])
#mtext(expression(paste("    ",alpha[0],"=0, ",rho,"=",0.0)),line=-6,col=colo[6])
abline(h=0)

plot(mplot2[,1],main="CCF",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot2),max(mplot2)))
for (i in 2:ncol(mplot2))
{  
  lines(mplot2[,i],col=colo[i])
}
#mtext("MSE",line=-1,col=colo[1])
#mtext(expression(paste("DFP ",alpha[0],"=0.9")),line=-2,col=colo[2])
#mtext(expression(paste("DFP ",alpha[0],"=0.45")),line=-3,col=colo[3])
#mtext(expression(paste("DFP ",alpha[0],"=0.22")),line=-4,col=colo[4])
#mtext(expression(paste("DFP ",alpha[0],"=0.1")),line=-5,col=colo[5])
#mtext(expression(paste("DFP ",alpha[0],"=0")),line=-6,col=colo[6])
abline(v=max_lag+1,lty=1)
abline(v=max_lag+1+h,lty=2)
abline(h=0)
axis(1,at=1:nrow(mplot2),labels=-max_lag-1+1:(nrow(mplot2)))
axis(2)
box()

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


mat_cor_vec<-round(cbind(cor_vec_1,cor_vec_2),2)

# Must use a special proceeding when utilizing greek letters in column or rownames

rownames(mat_cor_vec) <- c("$\\alpha_0=0.9$","$\\alpha_0=0.45$","$\\alpha_0=0.22$","$\\alpha_0=0.1$","$\\alpha_0=0$")
colnames(mat_cor_vec) <- c(" $\\textrm{Process 1: CCF }\\delta=0$","$\\delta=5$","$\\textrm{Process 2: } \\delta=0$","$\\delta=5$")

mat_cor_vec


if (F)
{
  tbl <- xtable(mat_cor_vec)
  caption(tbl)<-paste("CCFs of the DFP predictors for the first process (columns 1–2) and the second process (columns 3–4) evaluated at $\\delta=0$ (columns 1 and 3) and at $\\delta=h=5$ (columns 2 and 4), shown for multiple values of the decoupling parameter $\\alpha_0$. Note that $\\alpha_0$ generally differs from the CCF at $\\delta=0$ except in the case of complete decoupling (columns 1 and 3, last row).")
  label(tbl)<-"ar3_decoupling"
  
  print.xtable(tbl, sanitize.text.function = function(x) x)
}


















# ════════════════════════════════════════════════════════════════════
# Exercise 5: Complete decoupling and limit to look ahead
# ════════════════════════════════════════════════════════════════════
# Intuitively difficult since latest observation most important.



# ════════════════════════════════════════════════════════════════════
# Exercise 6: Leading indicator DFP
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























#######################################################################################################

# Example: DFP for a given tau
# Example in the paper but no plot




max_lag<-2
L<-50
h<-3
# First AR(3)
lambda1<-0.3
lambda2<-0.8
lambda3<-0.2
ar1<-ar11<-lambda1+lambda2+lambda3
ar2<-ar21<--lambda1*lambda2-lambda1*lambda3-lambda2*lambda3
ar3<-ar31<-lambda1*lambda2*lambda3

# Compute long sequence: need more values than L for MSE forecasts below
gamma<-ARMAtoMA(ar=c(ar1,ar2,ar3),lag.max=1000)

ts.plot(gamma[1:L])

gamma0<-gamma[1:L]
# MSE: last entries are vanishing (we could also insert the longer MA-expansion but this would not be the MSe estimate in the finite length MA case)
gammah<-c(gamma[h+(1:(L-h))],rep(0,h))

# Select lead over MSE
lead<--1

if (F)
{
  # Old code
  # Compute shifts at frequency zero
  tau0<-sum((0:(L-1))*gamma0)/sum(gamma0)
  tauh<-sum((0:(L-1))*gammah)/sum(gammah)
  
  # MSE is slightly leading
  tau0
  tauh
  tau<-lead
  # Formula for lambda0
  lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
  # Compute b
  b<-gammah+lambda0*gamma0
  
}

# Compute shifts at frequency zero

dfp_obj<-mse_dfp_from_tau_func(gamma0,gammah,lead)
tau0=dfp_obj$tau0
tauh=dfp_obj$tauh
lambda0=dfp_obj$lambda0
b=dfp_obj$b
# Unitary DFP
b_opt<-b/as.double(sqrt(b%*%b))

# Check lead
taub<-sum((0:(L-1))*b_opt)/sum(b_opt)
# Should equal lead (or tau): this is an exact result
taub-tauh

# Compute alpha0
alpha0<-compute_alpha_0_func(gamma0,gammah,lambda0)$alpha0
# Replicate DFP predictor with DFP criterion in MSE_LA_closed_form_rank_two_func
criterion_number<-1
# Select large lambda
lambda<-1000
val_vec_target<-1
# Need to scale alpha0 since the optimization routine assumes scaled gamma0,gammah
val_vec_constraint<-alpha0/as.double(sqrt(gamma0%*%gamma0))

MSE_LA_obj<-MSE_LA_closed_form_rank_two_func(criterion_number,h,lambda,gammah,gamma0,val_vec_target,val_vec_constraint,L)

# Check: ratio should be nearly one (up to negligible errors due to roots of quartic equation)  
b_opt/MSE_LA_obj$b


#--------------------------------------------------------
# Plots

par(mfrow=c(3,2))

colo<-c("black","green","blue")

mplot<-scale(cbind(gamma0,gammah,b_opt),center=F,scale=F)#/sqrt((L-1))
colnames(mplot)<-c("Nowcast",paste("MSE ",h,"-step"),"DFP")
apply(mplot^2,2,sum)
plot(mplot[,1],main="Filters",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
lines(mplot[,2],col=colo[2])
axis(1,at=1:nrow(mplot),labels=0:(nrow(mplot)-1))
axis(2)
box()

mplot<-cbind(compute_acf_at_lags_zero_delta_func(max_lag,h,gammah,gamma0)$cor_vec,compute_acf_at_lags_zero_delta_func(max_lag,h,b_opt,gamma0)$cor_vec)




K<-600
mplot<-scale(cbind(gamma0,gammah,b_opt),center=F,scale=F)#/sqrt((L-1))
apply(mplot^2,2,sum)
colnames(mplot)<-c("Nowcast",paste("MSE ",h,"-step"),"DFP")
shift_mat<-amp_mat<-matrix(ncol=ncol(mplot),nrow=K+1)
colnames(shift_mat)<-colnames(amp_mat)<-colnames(mplot)
for (i in 1:ncol(mplot))
{  
  filt_obj<-amp_shift_func(K,mplot[,i],F)
  shift_mat[,i]<-apply(cbind(rep(0,K+1),filt_obj$shift),1,max)
  amp_mat[,i]<-filt_obj$amp
}  


mplot<-amp_mat
plot(mplot[,1],ylim=c(0,max(mplot)),axes=F,col=colo[1],type="l",xlab="Frequency",ylab="",main="Amplitude")
#mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  #  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()

# Positive numbers signify left shift
mplot<-cbind(shift_mat[,1]-shift_mat[,1],shift_mat[,2]-shift_mat[,1],shift_mat[,3]-shift_mat[,1])
plot(mplot[,1],axes=F,col=colo[1],type="l",xlab="",ylab="",main="Leads over nowcast",ylim=c(min(mplot),max(mplot)))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{  
  lines(mplot[,i],col=colo[i],type="l")
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}  
abline(h=0)
axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
axis(2)
box()


# 1. Linear trend
len<-10000
x<-1:len
# Scale all filters to unit-length
filter_mat<-cbind(gamma0/mean(gamma0),gammah/mean(gammah),b_opt/mean(b_opt))/L
apply(filter_mat,2,sum)
y_out_mat<-filter(x,filter_mat[,1],side=1)
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,2],side=1))
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,3],side=1))
colnames(y_out_mat)<-c("Process=nowcast",paste("MSE ",h,"-step",sep=""),"DFP")
colo<-c("black","green","blue")

anf<-100
enf<-110
ts.plot(y_out_mat[anf:enf,]-(anf-tau0-1),col=colo,main="Linear trend",xlab="",ylab="")
#mtext("Nowcast",line=-1,col=colo[1])
#mtext("MSE",line=-2,col=colo[2])
#mtext("DFP",line=-3,col=colo[3])
abline(h=4)

set.seed(345)

x<-rnorm(len)
y_out_mat<-filter(x,filter_mat[,1],side=1)
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,2],side=1))
y_out_mat<-cbind(y_out_mat,filter(x,filter_mat[,3],side=1))
colnames(y_out_mat)<-c("Process=nowcast",paste("MSE ",h,"-step",sep=""),"DFP")

ts.plot(scale(y_out_mat[290:320,],center=F,scale=T),main="White noise",col=colo,xlab="",ylab="")
abline(h=0)

#######################################################################################################




# Example DFP applied to MA(9)
# It relies on quadratic DFP in lambda1,lambda2: function DFP_compute_lambda_alpha0_func above.



# We use the solution to the first unitary DFP criterion: quadratic in lambda
# Advantage: alpha0 in decoupling constraint is lag-zero CCF (theta)

# Design
h<-5
L<-10
ar1<-0.9
ar2<-0.
# Use c(1,ARMAtoMA(ar=c(ar1,ar2),lag.max=L)) since the weight 1 of epsilon_t is omitted
gamma<-c(1,ARMAtoMA(ar=c(ar1,ar2),lag.max=L-1))
# Forecast horizon
delta<-h

# Compute MSE forecast and nowcast (the latter is the DGP since x_t is causal)
gamma0<-gamma
gammah<-c(gamma0[(h+1):L],rep(0,h))

# CCF of MSE predictor at delta=0
ccf_mse0<-as.double(t(gammah)%*%gamma0/sqrt(t(gamma0)%*%gamma0*t(gammah)%*%gammah))

# Compute alpha0 in decoupling constraint: this is also the lag zero CCF (or theta)
# Impose mild decoupling and complete decoupling
alpha0_vec<-c(ccf_mse0/2,0)
# Compute DFP predictors
b0_mat<-matrix(nrow=L,ncol=length(alpha0_vec))
lambda1<-lambda2<-NULL
for (i in 1:length(alpha0_vec))
{ 
  alpha0<-alpha0_vec[i]
  # Compute quadratic in lambda and then unit length DFP  
  b0_obj<-DFP_compute_lambda_alpha0_func(gamma0,gammah,h,L,alpha0)
  b0_mat[,i]<-b0_obj$b0
  lambda1<-c(lambda1,b0_obj$lambda1)
  lambda2<-c(lambda2,b0_obj$lambda2)
  
}
colnames(b0_mat)<-c("DFP mild","DFP complete")

ts.plot(b0_mat)
# Check: should be one on diagonal (unit length)
diag(t(b0_mat)%*%b0_mat)


#----------------------------------------------
# Apply filters to data
# generate filtered series
len1<-100000

set.seed(4)
eps<-rnorm(len1)
mat_out<-matrix(nrow=len1,ncol=3)
z<-mse<-fast_for<-rep(NA,len)
for (i in L:len1)
{
  # DGP  
  z[i]<-gamma0%*%eps[i:(i-L+1)]
  # MSE and DFP predictors  
  mat_out[i,1]<-gammah%*%eps[i:(i-L+1)]
  mat_out[i,2]<-b0_mat[,1]%*%eps[i:(i-L+1)]
  mat_out[i,3]<-b0_mat[,2]%*%eps[i:(i-L+1)]
}
colnames(mat_out)<-c("mse",colnames(b0_mat))
colo<-c("black","green","royalblue")
ts.plot(scale(mat_out[500:min(1000,len1),],scale=T,center=F),col=colo)
for (i in 1:ncol(mat_out))
  mtext(colnames(mat_out)[i],col=colo[i],line=-i)

#-----------------------
max_lag<-0

# Simon's proposal: MSE-predictor of x_{t+h}-x_t
#gammah<-gammah-gamma0


# Compute CCFs of predictors
# MSE
gamma1<-gammah
# Add zeroes to avoid NAs
gamma_ref<-c(gamma0,rep(0,100))
ccf_mat<-compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec
# DFP mild
# Add zerors to avoid NAs
gamma1<-b0_mat[,1]
ccf_mat<-cbind(ccf_mat,compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec)
# DFP mild
gamma1<-b0_mat[,2]
ccf_mat<-cbind(ccf_mat,compute_ccf_func(gamma1,gamma_ref,h,max_lag,L)$cor_vec)

colnames(ccf_mat)<-c("MSE",colnames(b0_mat))


#----------------------------------------
# Generate plot and table


colo<-c("green","red","royalblue")

layout(matrix(c(1,2,3,3), 2, 2, byrow = T)) 


mplot<-cbind(gammah,b0_mat)
plot(mplot[,1],main="Predictor",axes=F,type="l",xlab="Lags",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-ccf_mat
plot(mplot[,1],main="CCF",axes=F,type="l",xlab="Leads",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(v=max_lag+1,col="royalblue")
abline(v=max_lag+1+h)
abline(h=0)
text(5,0.7,"MSE",col=colo[1])
text(5,0.5,"Partial decoupling",col=colo[2])
text(5,0.1,"Complete decoupling",col=colo[3])

axis(1,at=1:nrow(mplot),labels=-(max_lag+1)+1:nrow(mplot))
axis(2)
box()

anf<-900
anf<-1300
anf<-1500
anf<-1800
anf<-3200
mplot<-scale(mat_out[anf:min(anf+80,len1),],scale=T,center=F)

plot(mplot[,1],main="Forecast",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
abline(h=0)
axis(1,at=(1:(nrow(mplot)/10))*10,labels=(1:(nrow(mplot)/10))*10)
axis(2)
box()




mplot<-cbind(gammah,b0_mat)

# Performances: CCFs and lead at frequency zero
mat_cor_vec<-ccf_mat[c(max_lag+1,max_lag+1+h),]
colnames(mat_cor_vec)<-c("MSE","DFP weak decoupling","DFP complete decoupling")
rownames(mat_cor_vec)<-c("CCF at lag=0",paste("CCF at h=",h,sep=""))
mat_cor_vec
# Impose a positive sign of zero
mat_cor_vec[1,3]<-abs(mat_cor_vec[1,3])


# Time shifts at omega=0: DFP with complete decoupling is not meaningful because of phase reversal at zero
tauh<-(1:(L-1))%*%gammah[2:L]/sum(gammah)
tau_pd<-(1:(L-1))%*%b0_mat[2:L,1]/sum(b0_mat[,1])
tau_cd<-(1:(L-1))%*%b0_mat[2:L,2]/sum(b0_mat[,2])



mat_cor_vec<-rbind(mat_cor_vec,c(0,2.12,4.03))
mat_cor_vec[3,]<-round(mat_cor_vec[3,])

rownames(mat_cor_vec)[3]<-"Relative lead over MSE"
mat_cor_vec


#######################################################################################################


# Application of MSE-DFP to AR(3) and ARMA






