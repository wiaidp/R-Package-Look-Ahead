# Implements the PCS (Phase Change Shift) criterion defined in Appendix D,
# Equation 46, Wildi (2026).
# Computes a feasible generalized PC shift even when the underlying system is singular.
#
# Objective:
#   Seeks a monotonically increasing Cross-Correlation Function (CCF) over the
#   set of leads specified in 'Delta'. Rather than imposing an exact constraint
#   (which may be infeasible in general), the regularization parameter 'lambda'
#   penalizes deviations from monotonicity, steering the solution toward the
#   best attainable behaviour.
#
# Arguments:
#   Delta  : Integer vector of leads over which the CCF is required to be increasing.
#   beta   : Target slope. A positive value shifts the CCF peak to the right
#            (i.e., toward higher leads).
#   xi     : Wold decomposition coefficients of the target process.
#   L      : Filter length (number of coefficients to be estimated).
#   lambda : Regularization (penalty) parameter controlling the strength of the
#            monotonicity constraint. Larger values enforce the constraint more
#            strictly; set to zero to recover the unconstrained solution.
#
# Note on scale:
#   The function operates on covariances rather than correlations. This
#   distinction is immaterial in practice: since the goal is merely to achieve
#   a zero or mildly positive slope, the scale of the CCF does not affect the
#   validity of the result. Working with correlations would impose a unit-norm
#   constraint on b, introducing additional complexity and potentially yielding
#   multiple solutions.


PCS_shift_func <- function(Delta, xi, L, beta, lambda)
{
# MSE h-step predictor  
  gammah<-xi[h+1:L]
  
  # Flip the sign of beta to align the internal convention with the paper's
  # definition: a positive beta in the function interface corresponds to a
  # rightward peak shift (toward higher leads), which requires a negative
  # internal slope.
  slope <- -beta
  
  gamma_all <- xi
  
  # --- Build the shifted covariance matrix 'gammah_mat' ---
  # Each row contains the MSE predictor coefficients (gamma_all) shifted by
  # a specific lead value drawn from 'Delta'. 
  gammah_mat <- gamma_all[Delta[1] - 1 + 1:L] 
  if (length(Delta) > 0)
  {
    for (i in 1:length(Delta))
      gammah_mat <- rbind(gammah_mat,
                          gamma_all[Delta[i] + 1:L])
  }
  
  # --- Compute consecutive difference vectors ('d_delta') ---
  # Each row of d_delta is the difference between two consecutive rows of
  # gammah_mat:
  #
  #   d_delta[i, ] = gammah_mat[i, ] - gammah_mat[i + 1, ]
  #
  # These differences encode the pairwise monotonicity constraints: requiring
  #
  #   b' * d_delta[i, ] = beta > 0
  #
  # forces the CCF to increase by beta from lead Delta[i] to lead Delta[i+1].
  #
  # Why normalise the rows of gammah_mat before differencing?
  # ──────────────────────────────────────────────────────────
  # The rows of gammah_mat are normalised to unit length before the differences
  # are formed. This ensures that the constraint b' * d_delta[i, ] = beta
  # imposes a *uniform* slope of beta across all consecutive lead pairs,
  # regardless of the raw norms of the underlying MSE predictors.
  #
  # Without normalisation, the differences d_delta[i, ] would have unequal
  # norms across i, so the same value of beta would correspond to different
  # effective slopes for different lead pairs — making the constraint
  # inhomogeneous and difficult to interpret.
  #
  # Asymptotic interpretation (large lambda):
  # ─────────────────────────────────────────
  # When the system is feasible and lambda -> Inf, the regularised solution
  # converges to the exact constrained solution, and the slope of the CCF
  # between consecutive leads is:
  #
  #   CCF(Delta[i+1]) - CCF(Delta[i])  =  b' * d_delta[i, ]  =  beta
  #
  # Since CCF(k) = b' * gamma_k / ||b||  (using normalised gamma_k), the
  # slope in terms of the normalised CCF is:
  #
  #   [CCF(Delta[i+1]) - CCF(Delta[i])] / ||b||  =  beta / ||b||
  #
  # The norm ||b|| is determined by the MSE optimisation and is not controlled
  # directly; however, because it is constant across all lead pairs, the
  # relative slopes are preserved and the monotonicity ordering is unaffected.
  d_delta <- (gammah_mat[1, ] - gammah_mat[2, ])/(sqrt(sum((gammah_mat[1, ] - gammah_mat[2, ])^2)))
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      d_delta <- rbind(d_delta, (gammah_mat[i, ] - gammah_mat[i + 1, ])/sqrt(sum((gammah_mat[i, ] - gammah_mat[i + 1, ])^2)))
  }
  
  if (F)
  {
    # --- Rank diagnostic (disabled; enable for debugging) ---
    # For an AR(p) process, d_delta %*% t(d_delta) has rank at most p, because
    # the Yule-Walker equations confine the covariance vectors to a p-dimensional
    # subspace. Consequently, only p eigenvalues of d_delta %*% t(d_delta) are
    # non-zero. When length(Delta) >= p, the monotonicity constraints may be
    # mutually inconsistent, rendering the system infeasible. Inspect the
    # eigenvalues below to assess the effective rank of the constraint system.
    eigen(d_delta %*% t(d_delta))$values
    # In the presence of zero eigenvalues the exact constraints are infeasible,
    # but the regularized solution below still yields the best attainable
    # compromise.
  }
  
  # --- Compute the PCS filter coefficients ---
  
  # a. Assemble the regularized normal-equation matrix:
  #       M = I + lambda * sum_i ( d_delta[i,] %o% d_delta[i,] )
  #    The identity term ensures M is non-singular even when d_delta is rank-
  #    deficient, while the outer-product terms penalize deviations from the
  #    monotonicity target.
  M <- diag(rep(1, L)) + lambda * d_delta[1, ] %*% t(d_delta[1, ])
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      M <- M + lambda * d_delta[i, ] %*% t(d_delta[i, ])
  }
  
  # b. Assemble the right-hand side vector:
  #       gamma_sol = gamma_h + lambda * slope * sum_i d_delta[i,]
  #    The second term encodes the desired target slope into the linear system:
  #    in the limit lambda -> Inf, b' * d_delta[i,] -> slope for every i,
  #    provided the system is feasible.
  gamma_sol <- gammah_mat[h, ] + lambda * slope * d_delta[1, ]
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      gamma_sol <- gamma_sol + lambda * slope * d_delta[i, ]
  }
  
  # c. Solve M %*% b = gamma_sol for the PCS filter coefficient vector b.
  #    Because M is symmetric positive definite (by construction), the solution
  #    is unique.
  b <- solve(M) %*% gamma_sol
  
  # --- Feasibility check ---
  # Evaluates the residual | b' * d_delta[i,] - slope | for each constraint i.
  # For a feasible system these residuals converge to zero as lambda -> Inf.
  # Residuals that remain persistently non-zero as lambda grows indicate an
  # infeasible constraint system (e.g., caused by rank deficiency of d_delta,
  # as discussed in the rank diagnostic above).
  abs(d_delta %*% b - slope)
  
  b_mse<-b*as.double(t(b)%*%gammah/(t(b)%*%b))
  
  return(list(b = b, d_delta = d_delta,b_mse=b_mse))
  
}


