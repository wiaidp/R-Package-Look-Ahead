# ════════════════════════════════════════════════════════════════════
# TUTORIAL 2 — DFP PREDICTOR -
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────────────────
# Tutorial 1 demonstrates that the classical Mean Squared Error (MSE)
# multi-step-ahead predictor can become "stuck" at the current time point:
# rather than targeting x_{t+h}, it correlates most strongly with x_t,
# where h > 0 denotes the forecast horizon. This phenomenon motivates the
# introduction of the so-called Decouple-From-Present (DFP) criterion,
# a novel approach designed to explicitly enforce look-ahead behavior in
# the predictor.
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════
# Theoretical background:
#   Wildi, M. (2026). Forecasting on the Accuracy–Timeliness Frontier:
#   Two Novel "Look-Ahead" Predictors.
#   https://doi.org/10.48550/arXiv.2602.23087

# ════════════════════════════════════════════════════════════════════


rm(list = ls())

# Load tau-statistic (measures lead/lag performance)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used in JBCY (requires mFilter)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))




# This piece of code calculates lambda2 and lambda1 for unit DFP predictor b0=lambda1*gammah+lambda2*gamma0
# 1. Set up design
# First example such that gamma0, gammah are not orthogonal
set.seed(1)
L<-20
h<-4
alpha0<-0.5
gamma0<-rnorm(L)
gammah<-c(gamma0[(h+1):L],rep(0,h))

if (F)
{
  # Second example with orthogonal gamma0, gammah
  h<-4
  gamma0<-(-1)^(1:L)
  gammah<-c(rep(1,L-h),rep(0,h))
  # Check orthogonality  
  t(gamma0)%*%gammah
}


DFP_compute_lambda_alpha0_func<-function(gamma0,gammah,h,L,alpha0)
{
  # First case: gamma0 and gammah are not orthogonal  
  if (abs( t(gamma0)%*%gammah)>1.e-10)
  {
    a<-(t(gamma0)%*%gamma0)^2*t(gammah%*%gammah)/(t(gamma0%*%gammah))^2-t(gamma0%*%gamma0)
    b<-2*(alpha0*sqrt(t(gamma0%*%gamma0)))*(1-t(gamma0)%*%gamma0*t(gammah)%*%gammah/(t(gamma0)%*%gammah)^2)
    c<-(alpha0^2*t(gamma0%*%gamma0))*t(gammah)%*%gammah/(t(gamma0)%*%gammah)^2-1
    
    # Compute the two roots for lambda2 and select the one maximizing the objective
    lambda21<-as.double((-b+sqrt(b^2-4*a*c))/(2*a))
    # Compute lambda1
    lambda11<-as.double((alpha0*sqrt(t(gamma0%*%gamma0))-lambda21*t(gamma0)%*%gamma0)/t(gamma0%*%gammah))
    # Compute predictor
    b01<-lambda11*gammah+lambda21*gamma0
    
    lambda22<-as.double((-b-sqrt(b^2-4*a*c))/(2*a))
    lambda12<-as.double((alpha0*sqrt(t(gamma0%*%gamma0))-lambda22*t(gamma0)%*%gamma0)/t(gamma0%*%gammah))
    # Compute predictor
    b02<-lambda12*gammah+lambda22*gamma0
    # Select the solution that maximizes objective    
    if (t(b02)%*%gammah>t(b01)%*%gammah)
    {
      which_sol<-"negative sign"
      b0<-b02
      lambda2<-lambda22
      lambda1<-lambda12
    } else
    {
      which_sol<-"positive sign"
      b0<-b01
      lambda2<-lambda21
      lambda1<-lambda11
    }
  } else
  {
    # Second case: gamma0 and gammah are  orthogonal  
    lambda21<-as.double(alpha0/sqrt(t(gamma0)%*%gamma0))
    # First solution with positive lambda1    
    lambda11<-as.double(sqrt((1-alpha0^2)/t(gammah)%*%gammah))
    b01<-lambda11*gammah+lambda21*gamma0
    # Second solution with negative lambda11
    lambda22<-lambda21
    # First solution with positive lambda1    
    lambda12<--as.double(sqrt((1-alpha0^2)/t(gammah)%*%gammah))
    b02<-lambda12*gammah+lambda22*gamma0
    # Select the solution that maximizes objective    
    if (t(b02)%*%gammah>t(b01)%*%gammah)
    {
      b0<-b02
      lambda2<-lambda22
      lambda1<-lambda12
    } else
    {
      b0<-b01
      lambda2<-lambda21
      lambda1<-lambda11
    }
    
    
  }
  
  return(list(b0=b0,lambda1=lambda1,lambda2=lambda2,which_sol=which_sol))
}

b0_obj<-DFP_compute_lambda_alpha0_func(gamma0,gammah,h,L,alpha0)

b0<-b0_obj$b0
lambda1<-b0_obj$lambda1
lambda2<-b0_obj$lambda2
which_sol<-b0_obj$which_sol


# Specify coefficients of quadratic in lambda2

# Checks:
# 1. Length constraint: should vanish
t(b0)%*%b0-1
# 2. Decoupling constraint: should vanish
t(gamma0)%*%b0/sqrt(t(gamma0%*%gamma0))-alpha0























# Vector components (edit these)
vx <- 3
vy <- 2


# Specify gamma0 and gammah
gamma0<-c(3,0.5)*3/3.5
gammah<-c(1.5,1)*2/1.5
# Specify lambda0
lambda0<-0.3
# Lengths
l0<-sqrt(sum(gamma0^2))
lh<-sqrt(sum(gammah^2))

# Angle between gammah and gamma0: gammah is above (larger angle)
theta_h <- atan2(gammah[2], gammah[1])-atan2(gamma0[2], gamma0[1])

# Set up plot limits with some padding
x_min<--0.5
x_max<-3
y_min<--0.5
y_max<-1.5
lim <- 1.2 * max(1, abs(c(vx, vy))+0.5)
plot(NA, xlim = c(x_min,x_max), ylim = c(y_min, y_max),
     asp = 1, xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gamma0
arrows(0, 0,gamma0[1],gamma0[2], length = 0.12, lwd=1, col = "black")
text(gamma0[1]+0.1,gamma0[2], labels = expression(gamma[0]), col = "black", cex = 1.2)
# gammah
arrows(0, 0,gammah[1],gammah[2], length = 0.12, lwd=1, col = "black")
text(gammah[1]+0.1,gammah[2], labels = expression(gamma[h]), col = "black", cex = 1.2)
# Insert unit length b0
b0<-c(gammah[1]-lambda0*gamma0[1],gammah[2]-lambda0*gamma0[2])
lb0<-sqrt(sum(b0^2))
arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.5,b0[2]/lb0+0.1, labels = expression(b==lambda[1]*gamma[h]+lambda[2]*gamma[0]), col = "red", cex = 1.2)
segments(0,0,1.5*(gammah[1]-lambda0*gamma0[1]),1.5*(gammah[2]-lambda0*gamma0[2]),  lwd = 1,lty=2, col = "red")

text(b0[1]/lb0,b0[2]/lb0+0.5, "Intersection of cone with plane", col = "red", cex = 1)

# Draw the angle theta_h (between gammah and gamma0)
r <- 0.25 * lh  # arc radius
th_seq <- atan2(gamma0[2], gamma0[1])+seq(0, theta_h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0[2], gamma0[1])+theta_h / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(theta[0*h]), col = "black", cex = 1.2)
# Draw the angle theta (between b0 and gamma0)
theta <- atan2(b0[2], b0[1])-atan2(gamma0[2], gamma0[1])

r <- 1  # arc radius
th_seq <- atan2(gamma0[2], gamma0[1])+seq(0, theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "blue", lwd=1)

th_mid <-  atan2(gamma0[2], gamma0[1])+theta / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(+theta[0*b]), col = "blue", cex = 1.2)

# Draw arrow from gammah to b0
final_b0<-c(r * cos(th_seq[length(th_seq)]), r * sin(th_seq[length(th_seq)]))
lamb<-0.176
from_gammah<-(final_b0+lamb*gamma0)
arrows(from_gammah[1],from_gammah[2],final_b0[1],final_b0[2], length = 0.12, lwd=1, col = "black")
text(from_gammah[1],from_gammah[2]+0.1, labels = expression(lambda[1]*gamma[h]), col = "black", cex = 1.2)
# Draw second intersection of cone at -theta
angle_from_x_axis<--(th_seq[length(th_seq)]-2*atan2(gamma0[2],gamma0[1]))
segments(0,0,1.5*cos(angle_from_x_axis),1.5*sin(angle_from_x_axis),  lwd = 1,lty=2, col = "red")

# Draw the angle -theta 
theta <- atan2(b0[2], b0[1])-atan2(gamma0[2], gamma0[1])

r <- 1 # arc radius
th_seq <- atan2(gamma0[2], gamma0[1])+seq(0, -theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "blue", lwd=1,lty=2)

th_mid <-  atan2(gamma0[2], gamma0[1])-theta / 2
text(1.15 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(-theta[0*b]), col = "blue", cex = 1.2)

text(1.15 * r * cos(th_mid)+0.9, 1.15 * r * sin(th_mid)-0.4,"Intersection of cone with plane", col = "red", cex = 1)

text(2.,0.05,"Unit sphere (intersection with plane)", col = "blue", cex = 1)
















# Specify gamma0 and gammah
gamma0<-c(3,0.5)*3.5/4
gammah<-c(1.5,1)*2/1.5

# Specify lambda0
lambda0<-0.5
# Lengths
l0<-sqrt(sum(gamma0^2))
lh<-sqrt(sum(gammah^2))

# Angle between gammah and gamma0: gammah is above (larger angle)
beta0h <- atan2(gammah[2], gammah[1])-atan2(gamma0[2], gamma0[1])
# Angle between gammah and b
beta <- atan2(gammah[2]-lambda0*gamma0[2], gammah[1]-lambda0*gamma0[1])-atan2(gammah[2], gammah[1])

# Set up plot limits with some padding
x_min<-min(0,min(c(gamma0[1],gammah[1]))-1)
x_max<-max(c(gamma0[1],gammah[1]))+0.5
y_min<-min(c(gamma0[2],gammah[2]))-1
y_min<-0
y_max<-max(c(gamma0[2],gammah[2]))+1
y_max<-max(c(gamma0[2],gammah[2]))+0.2
plot(NA, xlim = c(x_min,x_max), ylim = c(y_min, y_max),
     asp=1.5,xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gamma0
arrows(0, 0,gamma0[1],gamma0[2], length = 0.12, lwd=1, col = "black")
text(gamma0[1]+0.1,gamma0[2], labels = expression(gamma[0]), col = "black", cex = 1.2)
# gammah
arrows(0, 0,gammah[1],gammah[2], length = 0.12, lwd=1, col = "red")
text(gammah[1]+0.1,gammah[2], labels = expression(gamma[h]), col = "black", cex = 1.2)
# gammah-lambda0*gamma0
arrows(gammah[1],gammah[2],gammah[1]-lambda0*gamma0[1],gammah[2]-lambda0*gamma0[2], length = 0.12, lwd=1, col = "red")
#  text(gammah[1]-lambda0*gamma0[1],gammah[2]-lambda0*gamma0[2]+0.2, labels = expression(tilde(b)==gamma[h]+lambda[1]*gamma[0]), col = "red", cex = 1.2)
text(gammah[1]-lambda0*gamma0[1]-0.4,gammah[2]-lambda0*gamma0[2], labels = expression(b==gamma[h]+lambda*gamma[0]), col = "black", cex = 1.2)
text(gammah[1]-lambda0*gamma0[1]+0.4,gammah[2]-lambda0*gamma0[2]+0.15, labels = expression(b==~"|"~lambda*gamma[0]~"|"), col = "red", cex = 1)

expression("E" *  "|" ~ Y)

# Insert unit length b0
b0<-c(gammah[1]-lambda0*gamma0[1],gammah[2]-lambda0*gamma0[2])
lb0<-sqrt(sum(b0^2))
#  arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
arrows(0,0,b0[1],b0[2], length = 0.12, lwd=1, col = "red")
#  text(b0[1]/lb0-0.1,b0[2]/lb0+0.1, labels = expression(b), col = "red", cex = 1.2)
text(b0[1]/lb0-0.3,b0[2]/lb0-0.2, labels = expression(c==~"|"~gamma[h]+lambda*gamma[0]~"|"), col = "red", cex = 1)
#  segments(0,0,1.5*(gammah[1]-lambda0*gamma0[1]),1.5*(gammah[2]-lambda0*gamma0[2]),  lwd = 1,lty=2, col = "red")

# Draw the angle beta0h (between gammah and gamma0)
r <- 0.3   # arc radius
th_seq <- atan2(gamma0[2], gamma0[1])+seq(0, beta0h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0[2], gamma0[1])+beta0h / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(theta[0*h]), col = "black", cex = 1.2)

# Draw the angle beta between gammah and b 
r <- 0.35   # arc radius
th_seq <- -beta0h+atan2(gammah[2]-lambda0*gamma0[2], gammah[1]-lambda0*gamma0[1])+seq(0, beta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(beta), col = "red", cex = 1.2)


# Draw the angle theta (between b0 and gamma0)
theta <- atan2(b0[2], b0[1])-atan2(gamma0[2], gamma0[1])
r <- 0.5  # arc radius
th_seq <- atan2(gamma0[2], gamma0[1])+seq(0, theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0[2], gamma0[1])+theta / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid)-0.1,
     labels = expression(theta[0*b]), col = "black", cex = 1.2)

# Add side naming for side a  
text(1.2 * r * cos(th_mid)+0.5, 1.15 * r * sin(th_mid)+0.3,
     labels = expression(a==~"|"~gamma[h]~"|"), col = "red", cex = 1)

# Draw the angle gamma from the apex gammah
r <- 0.35   # arc radius
th_seq <- seq(pi+0.15, pi+0.6, length.out = 100)
lines(gammah[1]+r * cos(th_seq), gammah[2]+r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah[1]+1.4 * r * cos(th_mid)+0.15, gammah[2]+1.15 * r * sin(th_mid),
     labels = expression(gamma==theta[0*h]), col = "red", cex = 1.2)

# Draw the angle alpha from the apex b
r <- 0.1   # arc radius
th_seq <- seq(-pi/2-0.4, pi/8-0.2, length.out = 100)
lines(gammah[1]-lambda0*gamma0[1]+r * cos(th_seq), gammah[2]-lambda0*gamma0[2]+r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah[1]-lambda0*gamma0[1]+1.4 * r * cos(th_mid)+0.05, gammah[2]-lambda0*gamma0[2]+1.15 * r * sin(th_mid),
     labels = expression(alpha), col = "red", cex = 1.2)














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

# Compute shifts at frequency zero
tau0<-sum((0:(L-1))*gamma0)/sum(gamma0)
tauh<-sum((0:(L-1))*gammah)/sum(gammah)

# MSE is slightly leading
tau0
tauh
# Select lead over MSE
lead<--1
tau<-lead
# Formula for lambda0
lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
# Compute b
b<-gammah+lambda0*gamma0
# Unitary DFP
b_opt<-b/as.double(sqrt(b%*%b))

# Check lead
taub<-sum((0:(L-1))*b_opt)/sum(b_opt)
# Should equal lead (or tau): this is an exact result
taub-tauh

# Compute alpha0
alpha0<-as.double(t(gamma0)%*%(gammah+lambda0*gamma0)/sqrt(t(gammah+lambda0*gamma0)%*%(gammah+lambda0*gamma0)))
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



# We use the solution to the first unit-length DFP criterion: quadratic in lambda
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

