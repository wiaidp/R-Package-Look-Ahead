# ════════════════════════════════════════════════════════════════════
# TUTORIAL 5 — DECOUPLE FROM PRESENT (DFP) PREDICTOR
# PART 2: MSE-DFP
# ════════════════════════════════════════════════════════════════════

# A brief overview is provided in tutorial_3_introduction.r

# ── TWO OPTIMISATION FORMS ────────────────────────────────────────────
# DFP can be formulated in two equivalent but complementary ways:
#
#   Form 1 — UNITARY DFP  (Equation 2 in Wildi 2026)
#     A quadratic (squared) optimisation problem. The constraint
#     hyperparameter has a direct, intuitive interpretation: it corresponds
#     to a prescribed correlation between y_t(h) and x_t (the degree of
#     decoupling from the present).
#
#   Form 2 — MSE-DFP  (Equation 9 in Wildi 2026)
#     A linear optimisation problem obtained by relaxing the unit-length 
#     constraint in a MSE formulation of the problem.  Computationally 
#     simpler, but the constraint hyperparameter is less directly 
#     interpretable in isolation.
#
#----------------------------------------------------------------------
# This tutorial discusses the second form: the MSE-DFP.
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


# ════════════════════════════════════════════════════════════════════
# CONCEPTUAL BACKGROUND
# ════════════════════════════════════════════════════════════════════
# Let x_t be a stationary time series with a convergent (square-summable)
# Wold decomposition:
#
#   x_t = sum_{k=0}^{inf} gamma_k * epsilon_{t-k}
#
# It is natural to express the DFP predictor in terms of the INNOVATIONS
# epsilon_t rather than in terms of x_t directly.  Exercise 2 will extend 
# this to the observable x_t representation.
#
# Two important notes on the MA / Wold representation:
#   (1) In practice, x_t is observed and the innovations are LATENT;
#       they must be recovered by applying an AR-inversion filter to x_t
#       (requiring stationarity of the AR or ARMA for a convergent Wold decomposition).
#       Here epsilon_t is simulated directly, so no inversion is needed.
#   (2) The DFP framework does NOT require the MA sequence to be invertible
#       (minimum-phase). The MA representation can be non-invertible.
# ════════════════════════════════════════════════════════════════════




# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: Introduction to MSE-DFP 
# ════════════════════════════════════════════════════════════════════
# This exercise introduces three novelties relative to Tutorial 3:
#
# A) THE MSE-DFP CRITERION (Equation 9, Wildi 2026)
#    Like the unit-length DFP, the MSE-DFP predictor lies in the plane
#    spanned by gamma0 (nowcast) and gammah (MSE predictor), but with
#    the weight on gammah fixed at one:
#
#      b0 = gammah + lambda * gamma0
#
#    Only lambda remains to be determined, making the optimisation LINEAR
#    with a unique closed-form solution (Proposition 1, Wildi 2026).
#    By emphasising the MSE objective (rather than target correlation as
#    in Equation 2), the unit-norm constraint can be dropped entirely.
#
#    Advantages over unitary DFP in tutorial 3:
#      - Reduces to a linear problem with a single globally optimal
#        solution.
#      - The predictor is naturally scaled to minimise MSE rather than
#        constrained to unit length.
#    Disadvantage:
#      - The hyperparameter alpha0 can no longer be interpreted directly
#        as the correlation between the DFP predictor and the nowcast.
#        Its value depends on the scale of gamma0 (and hence gammah).
#      - Except for alpha0=0 (complete decoupling) this might challenge 
#        sensible a priori selection of the DFP constraint parameter.
#      - See however exercise 4 below for an alternative selection criterion.
#
# B) AR FORM OF THE PREDICTORS
#    In addition to the MA form, we derive the AR form of each predictor.
#    The AR form is the natural representation when the predictor is
#    applied to the observed series x_t rather than to the innovation
#    sequence eps_t. We verify that both forms yield identical outputs.
#
# C) A MORE PERSISTENT DGP: AR(3)
#    Exercise 1 used an MA(9) with a finite Wold decomposition. Here we
#    switch to an AR(3), whose Wold decomposition is infinite and must be
#    truncated at length L for practical computation. This exercises the
#    MA-inversion step and tests DFP on a richer, AR-class of processes.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# 1.1 Data-Generating Process: AR(3) 
# ─────────────────────────────────────────────────────────────────────
# Construct an AR(3) by specifying its three characteristic roots,
# then recover the AR parameters from Vieta's formulas.
# All roots are real and inside the unit circle → stationary process.

# Note: 
# This process represents a challenging forecast problem, as its ACF decays
# slowly and monotonically. In such cases, the classical MSE predictor is
# typically trapped at the present, unable to anticipate future movements, see 
# tutorial 1.

lambda1 <- 0.3
lambda2 <- 0.8
lambda3 <- 0.2

# AR coefficients derived from roots via Vieta's formulas
ar1 <- ar11 <-  lambda1 + lambda2 + lambda3
ar2 <- ar21 <- -(lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3)
ar3 <- ar31 <-  lambda1 * lambda2 * lambda3

# Verify: roots of the characteristic polynomial should recover lambda1/2/3
polyroot(c(-ar3, -ar2, -ar1, 1))

# Wold decomposition: infinite MA representation of the AR(3)
# The ACF-based plot gives a visual check of the decay rate
par(mfrow = c(1, 1))
ts.plot(ARMAacf(ar = c(ar1, ar2, ar3), lag.max = 100),
        main = "Wold decomposition of AR(3) — MA coefficients",
        xlab = "Lag", ylab = expression(gamma[k]))

# Compute the Wold coefficients (truncated MA representation)
# We need more than L coefficients to form MSE forecasts at horizon h
gamma <- c(1, ARMAtoMA(ar = c(ar1, ar2, ar3), lag.max = 1000))

# Sanity check: inverting gamma back to AR should recover ar1, ar2, ar3
# (first three entries match; all subsequent entries are numerically zero)
ts.plot(-ARMAtoMA(ar = -gamma[2:length(gamma)], lag.max = 40),
        main = "AR inversion check — first three entries should match ar1/ar2/ar3")


# ─────────────────────────────────────────────────────────────────────
# 1.2 DFP Settings
# ─────────────────────────────────────────────────────────────────────
# As in tutorial 3, predictors are first derived in MA (innovation) form.
# Exercise 2 will translate these to the observable AR (data) form.

h <- 5    # forecast horizon
L <- 50   # filter length (truncation of the infinite Wold decomposition)

# Nowcast and MSE predictor filters (truncated to length L)
gamma0 <- gamma01 <- gamma[1:L]         # nowcast: Wold coefficients at lags 0,...,L-1
gammah <- gammah1 <- gamma[h + (1:L)]   # MSE predictor: Wold coefficients shifted by h

# Assumption: gamma0 and gammah are not collinear: the ratio of coefficients
# is not constant
gammah/gamma0

# Plot both filters for reference
colo  <- c("green", "black")
mplot <- cbind(gammah, gamma0)
colnames(mplot) <- c(paste0("MSE predictor (h=", h, ")"), "Nowcast")

par(mfrow = c(1, 1))
plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Lag", ylab = "Coefficient",
     main = "MA-form filter coefficients: MSE predictor vs. nowcast",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
abline(h = 0)
axis(1, at = 1:nrow(mplot), labels = rownames(mplot))
axis(2); box()

# Compute the population CCF of the MSE predictor as a baseline reference
max_lag    <- 0
cor_vec_mat <- compute_ccf_func(gammah, gamma0)[L:(2 * L - 1)]

# Compute DFP constraint value of MSE predictor for reference:
# To enforce decoupling, alpha0 should be smaller than this value
alpha0_MSE<-as.double(gammah%*%gamma0)
alpha0_MSE


# Grid of alpha0 values to sweep the AT frontier
# Note: alpha0 is NOT a correlation here (no unit-norm constraint);
# it is the raw inner product b0 %*% gamma0, controlling decoupling strength
alpha0_vec <- c(alpha0_MSE/1.1,alpha0_MSE/(1.5^(1:6)),0)
  

# ─────────────────────────────────────────────────────────────────────
# 1.3 MSE-DFP: Sweep the AT Frontier 
# ─────────────────────────────────────────────────────────────────────
# For each alpha0, compute the MSE-DFP predictor via Proposition 1
# (Wildi 2026): b0 = gammah + lambda * gamma0, where
#   lambda = (alpha0 - gamma0 %*% gammah) / (gamma0 %*% gamma0)
# This closed-form solution minimises the MSE subject to the decoupling
# constraint b0 %*% gamma0 = alpha0.

b_mat      <- NULL          # stores filter coefficients for each alpha0
lambda_vec1 <- NULL         # stores lambda values
cor_vec_mat <- NULL         # stores CCF 
cor_vec_1  <- matrix(ncol = 2, nrow = length(alpha0_vec))  # CCF at lags 0 and h

for (i in seq_along(alpha0_vec)) {
  
  alpha0 <- alpha0_vec[i]
  
  # ── Compute MSE-DFP via utility function ─────────────────────────
  b0 <- compute_mse_dfp(alpha0, gamma0, gammah)$b0
  
  # ── Alternative closed-form derivation (Proposition 1) ───────────
  # lambda scales gamma0 to enforce the decoupling constraint exactly
  lambda          <- as.double((alpha0 - t(gamma0) %*% gammah) / (t(gamma0) %*% gamma0))
  b0_alternative  <- gammah + lambda * gamma0
  # Verify both derivations agree (should be zero)
  max(abs(b0 - b0_alternative))
  
  b_mat       <- cbind(b_mat, b0)
  lambda_vec1 <- c(lambda_vec1,lambda)
  
  # ── Compute population CCF for this predictor ────────────────────
  cor_vec <- compute_acf_at_lags_zero_delta_func(max_lag, h,
                                                 as.vector(b0), gamma0)$cor_vec
  cor_vec_mat      <- cbind(cor_vec_mat, cor_vec)
  cor_vec_1[i, 1]  <- cor_vec[1]       # CCF at lag 0  (coupling with present)
  cor_vec_1[i, 2]  <- cor_vec[1 + h]   # CCF at lag h  (coupling with target)
}

colnames(b_mat)<- colnames(cor_vec_mat) <- names(lambda_vec1)  <- paste0("alpha0=", round(alpha0_vec,3))
colnames(cor_vec_1) <- c("Lag 0", "Lag h")

# ── Verification checks ───────────────────────────────────────────────
# Check 1: DFP constraint b0 %*% gamma0 = alpha0 should hold exactly
# (residuals should be zero for all alpha0)
t(b_mat) %*% gamma0 - alpha0_vec

# Check 2: Equivalent check via the normalised CCF
# Since cor_vec is the CCF, alpha0 must be scaled by the norms of b0 and gamma0
cor_vec_1[, 1] - alpha0_vec / sqrt(diag(t(b_mat) %*% b_mat) *
                                     as.double(t(gamma0) %*% gamma0))



# ─────────────────────────────────────────────────────────────────────
# 1.4 Plots and Performances
# ─────────────────────────────────────────────────────────────────────

# Set up a 1×2 panel layout for side-by-side plots
par(mfrow = c(1, 2))
# Define a colour palette for up to 6 predictors
colo <- c("green", rainbow(ncol(b_mat)))
# Assemble MSE and DFP
mplot <- cbind(gammah, b_mat)

ts.plot(mplot,main="Unit-scaled predictors: AR(3)",col=colo,xlab="",ylab="")
mtext("MSE",line=-1,col=colo[1])
for (i in 1:ncol(b_mat))
  mtext(paste("DFP: alpha0=",round(alpha0_vec[i],3),sep=""),line=-(i+1),col=colo[i+1])
abline(h=0)

ccf_mse<-compute_acf_at_lags_zero_delta_func(max_lag, h,
                                             as.vector(gammah), gamma0)$cor_vec
mplot<-cbind(ccf_mse,cor_vec_mat)[1:22,]

plot(mplot[,1],main="",axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(mplot),max(mplot)))
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


# Round and display the correlation matrix for inspection
mat_cor_vec <- round(cor_vec_1, 2)
mat_cor_vec


# ─────────────────────────────────────────────────────────────────────
# 1.5 Apply Predictors (Filters) to Data
# ─────────────────────────────────────────────────────────────────────

# Fix the random seed for reproducibility
# (set.seed(45) overridden below; set.seed(17) is the active seed)
set.seed(45)
set.seed(17)
len <- 10000

# Generate a white-noise (standard normal) input series of length `len`
eps <- rnorm(len)

# Generate AR(3)
x<-xhat<-eps
for (i in 4:len)
{
# AR(3)  
  x[i]<-ar1*x[i-1]+ar2*x[i-2]+ar3*x[i-3]+eps[i]
# MSE predictor
  xhat[i]<-gammah[1:min(i,L)]%*%eps[i:max(1,i-L+1)]
}

# Apply each DFP/MSE filter column in b_mat to eps (MA form) using one-sided (causal)
# convolution, and collect all filtered outputs as columns of y_out_mat.
# Each column of b_mat corresponds to one predictor (MSE or a DFP variant).
y_out_mat <- NULL
for (i in 1:ncol(b_mat))
  y_out_mat <- cbind(y_out_mat, filter(eps, b_mat[, i], side = 1))

# Select a short time span for visualization
anf<-350
enf<-415
# Reset to single-panel layout and plot a representative excerpt (obs. 300–350)
# of the filtered outputs to visually compare predictor behaviours
par(mfrow = c(1, 1))
ts.plot(scale(y_out_mat[anf:enf, ]),
        main = "Predictor Outputs", col = colo[2:length(colo)], xlab = "", ylab = "")
abline(h = 0)
lines(scale(x[anf:enf]),lty=2,lwd=2)
lines(scale(xhat[anf:enf]),col="green",lty=2,lwd=2)
mtext("AR(3)",line=-1)
mtext("MSE",line=-2,col="green")
for (i in 1:length(alpha0_vec))
  mtext(paste("alpha0=", round(alpha0_vec[i],3), sep = ""), col = colo[i+1], line = -i-2)

# --- Interpretation of the Plot ---
#
# The plot illustrates the practical effect of the DFP parameter alpha0 on 
# predictor behaviour:
#
#  - Moderate decoupling (intermediate alpha0): the predictor anticipates
#    mean reversion over mid-term dynamics. This is visible as sustained
#    intervals where the predictor leads the process (black line) across
#    the zero line — i.e., it signals turning points before they occur.
#
#  - Strong decoupling (small alpha0): the predictor aggressively
#    anticipates maxima, minima, and zero-crossings of the process.
#    However, two costs emerge simultaneously:
#
#      (i)  Amplitude loss — the predictor becomes anchored near the
#           mean during sustained swings, losing the ability to track
#           the true amplitude of the process.
#
#      (ii) Increased noise — the predictor output becomes noisier.
#
#    Together, these affect the cross-correlation (CCF)
#    between the predictor and the target at forecast horizon h.
#
#  - Note: alpha0 can be interpreted as a covariance (up to a factor of 
#    sigma^2, the innovation variance). The scale-dependence of the covariance
#    makes the raw value of alpha0 difficult to interpret in isolation.
#
#    In the example above, all candidate values of alpha0 were chosen to
#    be smaller than the covariance between the MSE predictor and the
#    nowcast, given by gammah %*% gamma0. This quantity serves as a
#    natural reference point: enforcing alpha0 < gammah %*% gamma0
#    is precisely what constitutes decoupling of the MSE-DFP in practice.


# --- Remark on Complete Decoupling (alpha0 = 0) ---
#
# Complete decoupling enforces zero correlation between the predictor
# output and x_t (the most recent observation). Since x_t is
# conventionally regarded as highly informative, one might ask: does
# imposing zero correlation conflict with its importance?
#
# A complete answer is deferred to Exercise 2 below, which examines
# the AR-form representation of the DFP predictor. Here, we reason
# about the weight assigned to the most recent innovation epsilon_t
# in the MA-form representation (i.e., the filter applied directly
# to the innovation sequence epsilon_{t-k}, k = 0, 1, 2, ...).
#
# In the MA form, zero correlation (complete decoupling from x_t)
# does NOT imply that the filter weight assigned to epsilon_t is
# zero. Rather, the decoupling constraint reshapes the overall filter
# so that the predictor output is structurally orthogonal to x_t —
# while epsilon_t itself may still receive a substantial weight.
# Orthogonality is a global property of the filter as a whole, not
# a local constraint on any single coefficient.
#
# As the plot below confirms, epsilon_t receives the LARGEST filter
# weight among all lags in the completely decoupled DFP (alpha0 = 0).
# Decoupling is therefore achieved through the collective interplay
# of all filter coefficients, not by suppressing the weight on
# epsilon_t alone.
#
# The complementary AR-form perspective is examined in Exercise 2
# below: progressively strengthening the decoupling (decreasing
# alpha0) deflates the weight assigned to x_t, yet x_t retains its
# dominant importance throughout.


ts.plot(b_mat[, ncol(b_mat)],
        main = "Complete decoupling: the largest weight is assigned to x_t",
        xlab = "Lag",
        ylab = "Filter coefficient")

# Caveat: complete decoupling (alpha0 = 0) is an extreme DFP
# configuration. It pushes look-ahead behaviour to the boundary of
# what is statistically consistent and practically interpretable, and
# should therefore be treated as a limiting reference case rather than
# a recommended operational setting.


# ─────────────────────────────────────────────────────────────────────
# 1.6 Geometry of the MSE-DFP Predictor
# ─────────────────────────────────────────────────────────────────────
# The following figure (reproduced from Wildi 2026) illustrates the geometry of
# the MSE-DFP solution in the plane spanned by gamma0 (nowcast filter)
# and gammah (h-step MSE predictor).
#
# Assumption: gamma0 and gammah are linearly independent (not collinear);
# see Wildi (2026) for the degenerate case.
#
# --- Geometric simplification relative to the unitary DFP ---
#
# The MSE-DFP is geometrically simpler than the unitary DFP of tutorial 3:
#
#   Unitary DFP (Tutorial 3):
#     Two constraints are active simultaneously — the decoupling constraint
#     AND the unit-length constraint. The DFP lies on the intersection of the 
#     plane spanned by gammah and gamm0, a cone with axis gamma0 (nowcast), 
#     and the unit sphere, see exercise 1.10. 
#
#   MSE-DFP (this exercise):
#     Dropping the unit-length constraint removes the quadratic term entirely.
#     Only the linear decoupling constraint remains, defining an affine
#     hyperplane in R^L. The feasible set is flat (a hyperplane), and the
#     optimisation reduces to a standard orthogonal projection problem.
#
# --- Optimality and closed-form solution ---
#
# By the least-squares optimality principle, the MSE-DFP solution b0 is the
# orthogonal projection of gammah onto the affine hyperplane defined by:
#
#   gamma0' * b = alpha0      (decoupling constraint)
#
# This projection has the closed-form expression (Proposition 1, Wildi 2026):
#
#   b0 = gammah + lambda * gamma0
#
# where the weight on gammah is exactly one (gammah is the unconstrained
# optimum), and lambda is the unique scalar satisfying:
#
#   gamma0' * b0(lambda) = alpha0
#
# Because this equation is linear in lambda, the solution is unique and
# given in closed form.
#
# --- Intuition for the sign of lambda ---
#
# Intuition: decoupling b0 from gamma0 requires a negative weight on
# gamma0 (lambda < 0). We subtract the nowcast direction from the MSE
# predictor, thereby removing its 'present-anchoring' component and
# retaining only the forward-looking portion.
#
# Note on dimensionality:
# For illustration purposes, gamma0 and gammah in the figure below are drawn
# as vectors in a two-dimensional plane. In general, both vectors live in R^L,
# where L is the filter length. This is without loss of generality.


par(mfrow=c(1,1))
# Specify gamma0 and gammah.

gamma0_plot<-c(3,0.5)*3.5/4
gammah_plot<-c(1.5,1)*2/1.5

# Specify lambda0
lambda0<-0.5
# Lengths
l0<-sqrt(sum(gamma0_plot^2))
lh<-sqrt(sum(gammah_plot^2))

# Angle between gammah_plot and gamma0_plot: gammah_plot is above (larger angle)
beta0h <- atan2(gammah_plot[2], gammah_plot[1])-atan2(gamma0_plot[2], 
                                                      gamma0_plot[1])
# Angle between gammah_plot and b
beta <- atan2(gammah_plot[2]-lambda0*gamma0_plot[2], gammah_plot[1]-
                lambda0*gamma0_plot[1])-atan2(gammah_plot[2], gammah_plot[1])

# Set up plot limits with some padding
x_min<-min(0,min(c(gamma0_plot[1],gammah_plot[1]))-1)
x_max<-max(c(gamma0_plot[1],gammah_plot[1]))+0.5
y_min<-min(c(gamma0_plot[2],gammah_plot[2]))-1
y_min<-0
y_max<-max(c(gamma0_plot[2],gammah_plot[2]))+1
y_max<-max(c(gamma0_plot[2],gammah_plot[2]))+0.2
plot(NA, xlim = c(x_min,x_max), ylim = c(y_min, y_max),
     asp=1.5,xlab = "", ylab = "", axes = TRUE)
# Axes
abline(h = 0, v = 0, col = "gray85")
# gamma0_plot
arrows(0, 0,gamma0_plot[1],gamma0_plot[2], length = 0.12, lwd=1, col = "black")
text(gamma0_plot[1]+0.1,gamma0_plot[2], labels = expression(gamma[0]), 
     col = "black", cex = 1.2)
# gammah_plot
arrows(0, 0,gammah_plot[1],gammah_plot[2], length = 0.12, lwd=1, col = "red")
text(gammah_plot[1]+0.1,gammah_plot[2], labels = expression(gamma[h]), 
     col = "black", cex = 1.2)
# gammah_plot-lambda0*gamma0_plot
arrows(gammah_plot[1],gammah_plot[2],gammah_plot[1]-lambda0*gamma0_plot[1],
       gammah_plot[2]-lambda0*gamma0_plot[2], length = 0.12, lwd=1, col = "red")
text(gammah_plot[1]-lambda0*gamma0_plot[1]-0.4,gammah_plot[2]-
       lambda0*gamma0_plot[2], labels = expression(b==gamma[h]+
                                                     lambda*gamma[0]), col = "black", cex = 1.2)
text(gammah_plot[1]-lambda0*gamma0_plot[1]+0.4,gammah_plot[2]-
       lambda0*gamma0_plot[2]+0.15, labels = 
       expression(b==~"|"~lambda*gamma[0]~"|"), col = "red", cex = 1)

expression("E" *  "|" ~ Y)

# Insert unit length b0
b0<-c(gammah_plot[1]-lambda0*gamma0_plot[1],gammah_plot[2]-
        lambda0*gamma0_plot[2])
lb0<-sqrt(sum(b0^2))
#  arrows(0,0,b0[1]/lb0,b0[2]/lb0, length = 0.12, lwd=1, col = "red")
arrows(0,0,b0[1],b0[2], length = 0.12, lwd=1, col = "red")
text(b0[1]/lb0-0.3,b0[2]/lb0-0.2, labels = expression(c==~"|"~gamma[h]+
                                                        lambda*gamma[0]~"|"), col = "red", cex = 1)

# Draw the angle beta0h (between gammah_plot and gamma0_plot)
r <- 0.3   # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, beta0h, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+beta0h / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(theta[0*h]), col = "black", cex = 1.2)

# Draw the angle beta between gammah_plot and b 
r <- 0.35   # arc radius
th_seq <- -beta0h+atan2(gammah_plot[2]-lambda0*gamma0_plot[2], 
                        gammah_plot[1]-lambda0*gamma0_plot[1])+seq(0, beta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid),
     labels = expression(beta), col = "red", cex = 1.2)


# Draw the angle theta (between b0 and gamma0_plot)
theta <- atan2(b0[2], b0[1])-atan2(gamma0_plot[2], gamma0_plot[1])
r <- 0.5  # arc radius
th_seq <- atan2(gamma0_plot[2], gamma0_plot[1])+seq(0, theta, length.out = 100)
lines(r * cos(th_seq), r * sin(th_seq), col = "black", lwd=1)

th_mid <-  atan2(gamma0_plot[2], gamma0_plot[1])+theta / 2
text(1.4 * r * cos(th_mid), 1.15 * r * sin(th_mid)-0.1,
     labels = expression(theta[0*b]), col = "black", cex = 1.2)

# Add side naming for side a  
text(1.2 * r * cos(th_mid)+0.5, 1.15 * r * sin(th_mid)+0.3,
     labels = expression(a==~"|"~gamma[h]~"|"), col = "red", cex = 1)

# Draw the angle gamma from the apex gammah_plot
r <- 0.35   # arc radius
th_seq <- seq(pi+0.15, pi+0.6, length.out = 100)
lines(gammah_plot[1]+r * cos(th_seq), gammah_plot[2]+r * sin(th_seq), 
      col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah_plot[1]+1.4 * r * cos(th_mid)+0.15, gammah_plot[2]+
       1.15 * r * sin(th_mid),
     labels = expression(gamma==theta[0*h]), col = "red", cex = 1.2)

# Draw the angle alpha from the apex b
r <- 0.1   # arc radius
th_seq <- seq(-pi/2-0.4, pi/8-0.2, length.out = 100)
lines(gammah_plot[1]-lambda0*gamma0_plot[1]+r * cos(th_seq), gammah_plot[2]-
        lambda0*gamma0_plot[2]+r * sin(th_seq), col = "red", lwd=1)

th_mid <-  th_seq[50]
text(gammah_plot[1]-lambda0*gamma0_plot[1]+1.4 * r * cos(th_mid)+0.05, 
     gammah_plot[2]-lambda0*gamma0_plot[2]+1.15 * r * sin(th_mid),
     labels = expression(alpha), col = "red", cex = 1.2)


# Note on the geometry of the DFP solution:
#
# The figure illustrates that the DFP predictor b lies on the side of gammah
# (the MSE predictor) that is OPPOSITE to gamma0 (the nowcast). Equivalently,
# gammah lies between gamma0 and b, so the angle between gamma0 and b exceeds
# the angle between gamma0 and gammah:
#
#   theta_{0b} > theta_{0h}
#
# This condition is termed (positive) "phase excess" in Wildi (2026). In the
# standard case (typical applications), phase excess is sufficient to guarantee
# that b leads gammah at the trend frequency (frequency zero):
#
#   - Proposition 2 establishes the phase-excess condition formally.
#   - Theorem 2 derives the lead tau > 0 from a set of basic assumptions
#     that include theta_{0b} > theta_{0h} as a key premise.
#
# Intuitively: if gammah already leads gamma0 at frequency zero,  the DFP 
# predictor b accentuates this existing lead by rotating further away from 
# gamma0 and beyond gammah, i.e. it does 'more of the same'. In doing so, 
# b amplifies the phase advance that gammah has relative to gamma0, thereby 
# acquiring an additional lead relative to gammah itself.
#
# However, in some non-standard cases, gammah does not lead gamma0 at frequency 
# zero. This is discussed in Wildi (2026, Appendix A). In such a case, a
# lead by the DFP predictor is obtained when either
#
#   (i)  b lies between gamma0 and gammah, or
#
#   (ii) b lies on the opposite side of gamma0 relative to gammah
#        (gamma0 lies between gammah and b)
#
# Counterintuitively, in the non-standard case the objective function is
# inverted in sign: rather than maximising the correlation with gammah
# (the MSE predictor, as in the standard case), the DFP minimises it.
# This highlights that designing predictors with genuine look-ahead
# behaviour is more challenging and subtle than it may first appear.





# ─────────────────────────────────────────────────────────────────────
# Exercise 2 AR Form
# ─────────────────────────────────────────────────────────────────────
# We now convert the MA-form predictors to their AR equivalents by
# convolving each predictor with the AR(3) operator. A similar proceeding
# applies to the unitary DFP in tutorial 3.

# --------------------------------------------------------------------------
# 2.1 Validation of the Convolution Approach
# --------------------------------------------------------------------------
# Specify the predictor matrix: MSE filter and DFP filters.
filter_mat          <- cbind(gammah, b_mat)
colnames(filter_mat) <- c("MSE", colnames(b_mat))

# Verify the approach via a known identity:
# Convolving the AR(3) operator with its Wold (MA) decomposition must
# yield the identity filter (i.e., the convolution output is 1 followed
# by zeros).
filt1 <- c(1, -ar1, -ar2, -ar3)  # AR(3) operator
filt2 <- gamma                    # Wold (MA) decomposition of the AR(3)
conv_two_filt_func(filt1, filt2)$conv[1:10]

# Having confirmed the identity, we now convolve the AR(3) operator with
# the MSE and DFP predictors (in MA form) to obtain their AR equivalents.

# --------------------------------------------------------------------------
# 2.2 Convolution of the AR(3) Operator with the Predictors
# --------------------------------------------------------------------------

# a. MSE predictor: convolve AR(3) operator with the MSE filter.
filt1      <- c(1, -ar1, -ar2, -ar3)
filt2      <- gammah
ar_mse_ar3 <- conv_two_filt_func(filt1, filt2)$conv

# b. DFP predictors: convolve AR(3) operator with each DFP filter.
ar_dfp_ar3_mat <- NULL
for (i in 1:ncol(filter_mat)) {
  # Use the original (unscaled) DFP predictor.
  filt2          <- filter_mat[, i]
  ar_dfp_ar3_mat <- cbind(
    ar_dfp_ar3_mat,
    conv_two_filt_func(filt1, filt2)$conv
  )
}
colnames(ar_dfp_ar3_mat)<-colnames(filter_mat)

# --------------------------------------------------------------------------
# 2.3 Analysis and Plot of DFP Predictors in AR Form
# --------------------------------------------------------------------------
# Key structural property of the DFP predictor in AR form:
#
# Only the FIRST AR coefficient varies across DFP designs; all higher-order
# AR coefficients are identical regardless of the chosen lambda.
#
# Theoretical justification (Wildi 2026, Section 3.1, Equation 19):
#   The DFP filter is defined as:
#     b = gammah + lambda * gamma0
#   When b is converted to AR form by inverting gamma0, the term
#   lambda * gamma0 maps to lambda * identity (a pure scalar shift).
#   Because gammah is fixed and convolution is linear, this scalar shift
#   affects only the first AR coefficient, leaving all higher-order
#   coefficients unchanged across DFP designs.

# Assign colours: green for the baseline (MSE), rainbow for DFP variants
colo <- c("green", rainbow(ncol(filter_mat) - 1))

# Plot the first 5 AR coefficients of each DFP predictor to highlight
# the structural invariance: only the first coefficient differs across designs
ts.plot(
  ar_dfp_ar3_mat[1:5, ],
  col  = colo,
  main = "Method B: DFP Predictors in AR Form"
)

# Add colour-coded in-plot labels for each predictor
for (i in 1:ncol(ar_dfp_ar3_mat))
  mtext(colnames(ar_dfp_ar3_mat)[i], col = colo[i], line = -i)


# --------------------------------------------------------------------------
# 2.4 Verification: Comparing MA and AR Forms
# --------------------------------------------------------------------------
# We verify that both forms produce numerically identical outputs when applied 
# to their respective inputs:
#   - AR form applied to observed data x.
#   - MA form applied to white noise innovations eps.

# Illustration: the convolution of an AR-form DFP filter with its MA-form
# counterpart does NOT yield the identity.
# Select any of the filters in filter_mat:
k     <- 4
# k cannot be larger than column dimension of filter_mat
k<-min(k,ncol(filter_mat))

# Simulate an AR(3) process to verify output equivalence.
set.seed(1)
len <- 1000
x   <- eps <- rnorm(len)
for (i in 4:len) {
  x[i] <- ar1 * x[i-1] + ar2 * x[i-2] + ar3 * x[i-3] + eps[i]
}

y_dfp_ma <- y_dfp_ar <- rep(NA, len)

# Apply the MA form to the innovations eps.
y_dfp_ma<-filter(eps,filter_mat[, k])
# Apply the AR form to the observed series x.
y_dfp_ar<-filter(x,ar_dfp_ar3_mat[, k])

# Both outputs should be numerically identical up to negligible errors
# arising from finite-length MA/AR truncation.
ts.plot(cbind(y_dfp_ma, y_dfp_ar)[1:200, ],main="AR- and MA-forms overlap exactly")

# Confirm: the maximum absolute difference is negligible.
max(na.exclude(abs(y_dfp_ma - y_dfp_ar)[1:200]))
# Note on approximation accuracy:
# Any residual discrepancy arises because the filters gamma0 and gammah are
# truncated to a finite length L. As L increases, the difference between the 
# MA and AR implementations vanishes asymptotically.

# ─────────────────────────────────────────────────────────────────────
# --- Interpretation and Remarks ---
# ─────────────────────────────────────────────────────────────────────
#
# 1. Structural simplicity of the DFP in AR form:
#    The entire DFP effect is absorbed into a single AR
#    coefficient — the weight on the most recent observation x_t. This makes
#    the AR form compact and seemingly easy to interpret.
#
#    Interpretability — Part I (AR-form perspective):
#      - Reducing the weight on x_t lowers the correlation between the
#        predictor output and x_t, directly controlling the degree of
#        decoupling.
#      - Complete decoupling (alpha0 = 0) corresponds to the design where
#        this correlation is driven to zero. Importantly, zero correlation
#        does NOT imply that the weight on x_t vanishes — it is the
#        interplay of all filter coefficients that achieves orthogonality.
#
# 2. The DFP as a Partial 'Black Box':
#    Despite the clean AR-form structure, the DFP constraint does not directly
#    reveal its look-ahead behaviour. While alpha0 controls the degree of
#    decoupling, it is not immediately obvious how decoupling translates into
#    an interpretable lead of the predictor (over the MSE benchmark). In fact, 
#    one of the two solutions of the unitary DFP (tutorial 3) is 
#    effectively lagging.
#
#    To make this precise: a useful predictor must satisfy two conditions:
#      - NECESSARY condition:  the predictor must be decoupled from the
#                              current data x_t (low correlation with x_t),
#                              so that it is not merely echoing the present, 
#                              see tutorial 1.
#      - SUFFICIENT condition: the predictor must genuinely anticipate future
#                              values, i.e. it must lead the MSE predictor by a
#                              meaningful and quantifiable amount.
#
#    The DFP constraint directly addresses the necessary condition (decoupling)
#    via alpha0, but leaves the sufficient condition (actual lead behaviour)
#    implicit. It is not guaranteed that a given alpha0 produces a predictor
#    that leads the target by any specific or desirable amount. 
#
#    Tutorial 5 partly resolves this gap by re-parameterising alpha0 in terms
#    of a pre-specified lead at the trend frequency (frequency zero), thereby
#    making the sufficient condition explicit — at least for the trend component.
#
# 3. Interpretability — Part II: DFP vs. PCS (Tutorial ???):
#    Even with the frequency-zero re-parameterisation of Tutorial 5, the DFP
#    concept remains less directly `look ahead' interpretable than the Peak 
#    Correlation Shifting (PCS) predictor introduced in Tutorial ????. The key 
#    distinction is the scope of the `lead' constraint:
#
#      - PCS addresses the lead of the predictor in an AGGREGATE sense,
#        not only at the trend frequency omega = 0. This makes the PCS 
#        constraint directly meaningful in terms of the predictor's overall 
#        timing behaviour (the `sufficiency' part).
#
#      - As a consequence of this specific `look ahead' constraint, the PCS 
#        effect generally spreads across all AR lags rather than being
#        confined to the first coefficient as in the above DFP. The filter 
#        (predictor) structure is therefore more complex.
#
#    In summary, the two look ahead approaches present an 
#    interpretability-complexity trade-off:
#
#      DFP — simpler filter/predictor structure (only the first AR coefficient 
#            is affected); seemingly easy to interpret (decrease weight on x_t),
#            but the resulting look-ahead effect is less straightforward to 
#            control.
#
#      PCS — more complex filter/predictor structure (all coefficients of the 
#            AR form are affected), but the look-ahead effect is directly 
#            interpretable as a classic (aggregate/overall) lead. 



