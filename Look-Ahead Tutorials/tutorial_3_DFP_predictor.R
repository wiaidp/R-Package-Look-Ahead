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





















