
# As shown in tutorial 1, exercise 1.3, the h-step ahead MSE predictor tends to 
# be stuck at present time in difficult forecast problems, in the sense 
# that its CCF peaks at lag 0. The Peak Correlation Shifting approach 
# tries to design the h-step ahed predictor so that the peak of the CCF is shifted 
# towards h. While this shift is not always feasible or practically implementable, 
# attenuating the original peak at lag 0 can be already effective, as this is the strategy of the 
# DFP predictor (decoupling from present). 

# We now discuss two necessary (but not sufficient) conditions for the shift to occur from k=0 to k=h 
# 1. Factually, shifting the peak from zero to h>0 means that the slope must be positive, 
#   i.e., CCF(k) is increasing for k=0,...,h, see Wildi (section 3.2), Appendix E.
# 2. The slope from k=h-1 to k=h should be positive, see Wildi (2026), section 3.2.

# The first condition is more stringent since it imposes a monotonic increase 
# of the CCF over an interval. In general, this criterion is not feasible exactly, 
# but we enforce behaviour as well (and strong) as possible: function PCS_shift_func().

# The second condition can be obtained by applying the DFP optimization under a  
# suitable modification of the decoupling constraint. 

# Main ideas: 
# The CCF at lag k is given by 


# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ──────────────────────────────────────────
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════


# ── INITIALISATION ───────────────────────────────────────────────────
rm(list = ls())

# Load the DFP optimisation routines.
# Provides DFP_compute_lambda_alpha0_func() and related solvers.
source(paste(getwd(), "/R/DFP.r", sep = ""))

# Load the tau-statistic utility: measures lead/lag at zero crossings.
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load general DFP/PCS utility functions (amplitude, time-shift, and CCF helpers).
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(xts)

# Load data from FRED via the alfred package (no API key required).
install.packages("alfred")
library(alfred)


# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: PCS Applied to MA(9)
# ════════════════════════════════════════════════════════════════════

h<-5

Delta<-1:h
lambda<-10000
beta<--0.1








# ─────────────────────────────────────────────────────────────────────
# Exercise 2 Geometry of the PCS Predictor
# ─────────────────────────────────────────────────────────────────────

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

#########################################################








