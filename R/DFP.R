# ════════════════════════════════════════════════════════════════════
# DFP (Decoupling From Present) — Core Functions
# ════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────
# unitary_DFP_func
# ────────────────────────────────────────────────────────────────────
# Computes the unit-norm DFP predictor, i.e., the filter b of length 1
# that maximizes the target correlation b' * gamma_h subject to the
# decoupling constraint b' * gamma_0 = alpha0, where ||b|| = 1.
#
# Arguments:
#   gamma0  : Nowcast MSE predictor coefficients (Wold coefficients at lag 0).
#   gammah  : h-step-ahead MSE predictor coefficients (Wold coefficients at lag h).
#   alpha0  : Target correlation with the present (decoupling level).
#             Must satisfy |alpha0| <= 1 (it is a correlation, not a covariance).
#
# Returns a list with:
#   b0         : DFP filter coefficient vector.
#   lambda1    : Lagrange multiplier associated with the gammah component.
#   lambda2    : Lagrange multiplier associated with the gamma0 component.
#   which_sol  : Character string indicating which root was selected
#                ("positive sign" or "negative sign").
# ────────────────────────────────────────────────────────────────────

unitary_DFP_func <- function(gamma0, gammah, alpha0)
{
  # Guard: if gamma0 and gammah are nearly collinear, the DFP problem is
  # degenerate (the constraint and objective are essentially the same vector)
  # and no meaningful solution exists.
  if (abs(abs(gamma0 %*% gammah) - sqrt(sum(gamma0^2) * sum(gammah^2))) < 1e-10)
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  # Guard: alpha0 must be a valid correlation coefficient.
  if (abs(alpha0) > 1)
  {
    print("|alpha0| must be smaller than one: it is a correlation!")
    return()
  }
  
  # ── Case 1: gamma0 and gammah are NOT orthogonal ───────────────────
  # The DFP predictor lies in the span of {gamma0, gammah}, so we write
  # b = lambda1 * gammah + lambda2 * gamma0 and solve for the Lagrange
  # multipliers by substituting into the unit-norm and decoupling constraints.
  # This yields a quadratic equation in lambda2; both roots are computed and
  # the one maximizing the target correlation b' * gammah is retained.
  if (abs(t(gamma0) %*% gammah) > 1e-10)
  {
    # Coefficients of the quadratic equation in lambda2
    a <- (t(gamma0) %*% gamma0)^2 * t(gammah %*% gammah) /
      (t(gamma0 %*% gammah))^2 - t(gamma0 %*% gamma0)
    b <- 2 * (alpha0 * sqrt(t(gamma0 %*% gamma0))) *
      (1 - t(gamma0) %*% gamma0 * t(gammah) %*% gammah /
         (t(gamma0) %*% gammah)^2)
    c <- (alpha0^2 * t(gamma0 %*% gamma0)) *
      t(gammah) %*% gammah / (t(gamma0) %*% gammah)^2 - 1
    
    # Root 1: positive discriminant branch
    lambda21 <- as.double((-b + sqrt(b^2 - 4 * a * c)) / (2 * a))
    # Recover lambda1 from the decoupling constraint b' * gamma0 = alpha0
    lambda11 <- as.double((alpha0 * sqrt(t(gamma0 %*% gamma0)) -
                             lambda21 * t(gamma0) %*% gamma0) /
                            t(gamma0 %*% gammah))
    b01 <- lambda11 * gammah + lambda21 * gamma0
    
    # Root 2: negative discriminant branch
    lambda22 <- as.double((-b - sqrt(b^2 - 4 * a * c)) / (2 * a))
    lambda12 <- as.double((alpha0 * sqrt(t(gamma0 %*% gamma0)) -
                             lambda22 * t(gamma0) %*% gamma0) /
                            t(gamma0 %*% gammah))
    b02 <- lambda12 * gammah + lambda22 * gamma0
    
    # Select the root that yields the higher target correlation b' * gammah
    if (t(b02) %*% gammah > t(b01) %*% gammah)
    {
      which_sol <- "negative sign"
      b0      <- b02
      lambda2 <- lambda22
      lambda1 <- lambda12
    } else {
      which_sol <- "positive sign"
      b0      <- b01
      lambda2 <- lambda21
      lambda1 <- lambda11
    }
    
  } else {
    
    # ── Case 2: gamma0 and gammah ARE orthogonal ───────────────────
    # Orthogonality decouples the two constraints: lambda2 is determined
    # solely by the decoupling condition, and lambda1 is determined solely
    # by the unit-norm condition. Two sign choices for lambda1 remain;
    # the one maximizing b' * gammah is selected.
    
    lambda21 <- as.double(alpha0 / sqrt(t(gamma0) %*% gamma0))
    
    # Positive lambda1 branch
    lambda11 <- as.double(sqrt((1 - alpha0^2) / t(gammah) %*% gammah))
    b01 <- lambda11 * gammah + lambda21 * gamma0
    
    # Negative lambda1 branch
    lambda22 <- lambda21
    lambda12 <- -as.double(sqrt((1 - alpha0^2) / t(gammah) %*% gammah))
    b02 <- lambda12 * gammah + lambda22 * gamma0
    
    # Select the root that yields the higher target correlation b' * gammah
    if (t(b02) %*% gammah > t(b01) %*% gammah)
    {
      b0      <- b02
      lambda2 <- lambda22
      lambda1 <- lambda12
    } else {
      b0      <- b01
      lambda2 <- lambda21
      lambda1 <- lambda11
    }
  }
  
  return(list(b0 = b0, lambda1 = lambda1, lambda2 = lambda2, which_sol = which_sol))
}

# ────────────────────────────────────────────────────────────────────
# compute_alpha_0_func
# ────────────────────────────────────────────────────────────────────
# Computes the decoupling correlation alpha0 = b' * gamma0 / ||b||
# implied by a given Lagrange multiplier lambda0.
#
# This is the inverse operation of mse_dfp_from_alpha0_func: given lambda0,
# it recovers alpha0.
#
# Arguments:
#   gamma0  : Nowcast MSE predictor coefficients.
#   gammah  : h-step-ahead MSE predictor coefficients.
#   lambda0 : Lagrange multiplier defining the DFP filter via b = gammah + lambda0 * gamma0.
#
# Returns a list with:
#   alpha0 : DFP constraint
# ────────────────────────────────────────────────────────────────────

compute_alpha_0_func <- function(gamma0, gammah, lambda0)
{
  # Reconstruct b and compute alpha0
  b      <- gammah + lambda0 * gamma0
  alpha0 <- as.double(t(gamma0) %*% b / sqrt(t(b) %*% b))
  
  return(list(alpha0 = alpha0))
}






# ────────────────────────────────────────────────────────────────────
# compute_mse_dfp
# ────────────────────────────────────────────────────────────────────
# Computes the MSE-optimal DFP predictor via a constrained least-squares
# projection. The predictor b maximizes b' * gammah (equivalently, minimizes
# the MSE of the filtered series) subject to the linear decoupling constraint
# b' * gamma0 = alpha0. No unit-norm constraint is imposed here.
#
# The solution is obtained by projecting gammah onto the affine subspace
# { b : b' * gamma0 = alpha0 } using the basis matrix B, whose columns
# span the orthogonal complement of gamma0.
#
# Arguments:
#   alpha0  : Target correlation/covariance with the present (decoupling level).
#   gamma0  : Nowcast MSE predictor coefficients.
#   gammah  : h-step-ahead MSE predictor coefficients.
#   plot_T  : Logical; if TRUE, plots the resulting filter coefficients.
#
# Returns a list with:
#   b0 : MSE-optimal DFP filter coefficient vector.
# ────────────────────────────────────────────────────────────────────

compute_mse_dfp <- function(alpha0, gamma0, gammah, plot_T = FALSE)
{
  # Guard: collinear gamma0 and gammah render the DFP problem degenerate.
  if (abs(abs(gamma0 %*% gammah) - sqrt(sum(gamma0^2) * sum(gammah^2))) < 1e-10)
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  L <- length(gamma0)
  
  # B: (L x L-1) basis matrix whose columns span the null space of gamma0',
  # i.e., all vectors orthogonal to gamma0. Any b satisfying b' * gamma0 = alpha0
  # can be written as b = alpha0_vec + B * beta for some free vector beta.
  B <- rbind(-gamma0[2:L] / gamma0[1], diag(rep(1, L - 1)))
  
  # Particular solution satisfying the constraint b' * gamma0 = alpha0 exactly.
  alpha0_vec <- c(alpha0 / gamma0[1], rep(0, L - 1))
  
  # Project gammah onto the column space of B to find the free component beta
  # that minimises ||b - gammah||, i.e., brings b as close to gammah as possible
  # within the feasible affine subspace.
  b_free <- solve(t(B) %*% B) %*% t(B) %*% (gammah - alpha0_vec)
  
  # Reconstruct the full DFP filter
  b0 <- alpha0_vec + B %*% b_free
  
  if (plot_T)
    ts.plot(b0)
  
  # Verification (should be zero up to numerical precision):
  # t(b0) %*% gamma0 - alpha0
  
  return(list(b0 = b0))
}


# ────────────────────────────────────────────────────────────────────
# mse_dfp_from_tau_func
# ────────────────────────────────────────────────────────────────────
# Computes the MSE-optimal DFP predictor using a time-shift (group-delay)
# formulation at frequency zero, as an alternative parameterisation to the
# correlation-based approach of mse_dfp_from_alpha0_func.
#
# The decoupling level alpha0 is implicitly determined by requiring that the
# group delay of b at frequency zero equals a user-specified target 'lead'.
#
# Arguments:
#   gamma0  : Nowcast MSE predictor coefficients.
#   gammah  : h-step-ahead MSE predictor coefficients.
#   lead    : Desired group delay of the DFP filter at frequency zero.
#
# Returns a list with:
#   tau0        : Group delay of gamma0 at frequency zero.
#   tauh        : Group delay of gammah at frequency zero.
#   lambda0     : Lagrange multiplier (see Proposition 1, Wildi 2026).
#   b           : MSE-optimal DFP filter (rescaled for MSE optimality where needed).
#   b_unscaled  : Raw filter before any MSE rescaling.
# ────────────────────────────────────────────────────────────────────

mse_dfp_from_tau_func <- function(gamma0, gammah, lead)
{
  # Guard: collinear predictors render the DFP problem degenerate.
  if (abs(abs(gamma0 %*% gammah) - sqrt(sum(gamma0^2) * sum(gammah^2))) < 1e-15)
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  # Guard: the group-delay formulation requires a well-defined group delay at
  # frequency zero, which exists only if the DC gain (sum of coefficients) is
  # non-zero. A zero sum means the filter eliminates or reverses the trend,
  # making the time-shift interpretation meaningless.
  if (abs(sum(gamma0)) < 1e-10)
  {
    print("gamma0 eliminates the trend or reverses its direction: the time-shift formulation at frequency zero is not meaningful.")
    print("Use a correlation-based DFP function instead, e.g., mse_dfp_from_alpha0_func or compute_mse_dfp.")
    return()
  }
  if (abs(sum(gammah)) < 1e-10)
  {
    print("gammah eliminates the trend or reverses its direction: the time-shift formulation at frequency zero is not meaningful.")
    print("Use a correlation-based DFP function instead, e.g., mse_dfp_from_alpha0_func or compute_mse_dfp.")
    return()
  }
  
  # Compute group delays at frequency zero for gamma0 and gammah.
  # For a causal filter with coefficients g_0, …, g_{L-1}, the group delay at
  # frequency zero is: tau = sum_{k=0}^{L-1} k * g_k / sum_{k=0}^{L-1} g_k
  tau0 <- sum((0:(L - 1)) * gamma0) / sum(gamma0)
  tauh <- sum((0:(L - 1)) * gammah) / sum(gammah)
  
  tau <- lead
  
  # ── Standard case: b is NOT proportional to gamma0 ────────────────
  # The Lagrange multiplier lambda0 is derived from the group-delay constraint
  # (see Wildi 2026, Proposition 1). A near-zero denominator signals that the
  # solution would collapse to b ∝ gamma0, handled separately below.
  if (abs(tau + tauh - tau0) < 1e-10)
  {
    # ── Singular case: b is aligned with gamma0 ──────────────────────
    # The group-delay constraint forces b to be proportional to gamma0.
    # We rescale to restore MSE optimality (i.e., to project onto gammah).
    print("Formula for lambda0 is near singularity: b is aligned with gamma0.")
    b          <- gamma0
    b_unscaled <- b
    b          <- b * as.double(gammah %*% b / (b %*% b))
    
  } else {
    
    # Lagrange multiplier ensuring the group delay of b equals 'lead'
    lambda0 <- -(tau * sum(gammah)) / ((tau + tauh - tau0) * sum(gamma0))
    
    # Standard DFP filter: b = gammah + lambda0 * gamma0
    b          <- gammah + lambda0 * gamma0
    b_unscaled <- b
    
    # ── Non-standard case: tauh > tau0 ────────────────────────────────
    # When the h-step-ahead predictor lags the nowcast at frequency zero,
    # the standard formula may place b on the wrong side of gamma0 relative
    # to gammah, effectively minimising rather than maximising the target
    # correlation. A sign correction is applied based on the sign of the
    # denominator (tau + tauh - tau0).
    if (tauh > tau0)
    {
      print("Non-standard case: tauh > tau0.")
      print("The MSE predictor lags the nowcast at frequency zero.")
      print("The DFP solution must be inverted: b lies on the side of gamma0 opposite to gammah.")
      print("Equivalently, the target correlation is minimised rather than maximised.")
      print("The sign of gammah is therefore inverted in this branch.")
      
      if ((tau + tauh - tau0) > 0)
      {
        # b lies between gamma0 and gammah — standard formula applies
        lambda0    <- lambda0
        b          <- gammah + lambda0 * gamma0
        b_unscaled <- b
      } else {
        # gamma0 lies between b and gammah — sign inversion required
        lambda0    <- -lambda0
        b          <- -gammah + lambda0 * gamma0
        b_unscaled <- b
      }
      
      # Rescale to recover MSE optimality (the norm of b is otherwise arbitrary
      # in the non-standard branch)
      b <- b * as.double(gammah %*% b / (b %*% b))
    }
  }
  
  # Warn if the resulting filter has a negative target correlation
  if (b %*% gammah < 0)
    print("Warning: the target correlation b' * gammah is negative.")
  
  return(list(tau0 = tau0, tauh = tauh, lambda0 = lambda0,
              b = b, b_unscaled = b_unscaled))
}


# ────────────────────────────────────────────────────────────────────
# mse_dfp_from_alpha0_func
# ────────────────────────────────────────────────────────────────────
# Note: this is an alternative (simpler) variant of compute_mse_dfp(): 
# It does not involve a matrix inversion.
# Computes the MSE-optimal DFP predictor given an explicit decoupling
# level alpha0 = b' * gamma0 (a target covariance with the present).
#
# The closed-form solution follows directly from Proposition 1 of
# Wildi (2026): b = gammah + lambda * gamma0, where lambda is chosen
# so that b' * gamma0 = alpha0.
#
# Arguments:
#   gamma0  : Nowcast MSE predictor coefficients.
#   gammah  : h-step-ahead MSE predictor coefficients.
#   alpha0  : Target covariance with the present (decoupling level).
#
# Returns a list with:
#   lambda : Lagrange multiplier.
#   b      : MSE-optimal DFP filter coefficient vector.
# ────────────────────────────────────────────────────────────────────

mse_dfp_from_alpha0_func <- function(gamma0, gammah, alpha0)
{
  # Guard: collinear predictors render the DFP problem degenerate.
  if (abs(abs(gamma0 %*% gammah) - sqrt(sum(gamma0^2) * sum(gammah^2))) < 1e-10)
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  # Closed-form Lagrange multiplier from Proposition 1, Wildi (2026):
  #   b' * gamma0 = alpha0  =>  (gammah + lambda * gamma0)' * gamma0 = alpha0
  #   =>  lambda = (alpha0 - gammah' * gamma0) / (gamma0' * gamma0)
  lambda <- as.double((alpha0 - t(gamma0) %*% gammah) / (t(gamma0) %*% gamma0))
  b      <- gammah + lambda * gamma0
  
  return(list(lambda = lambda, b = b))
}



















