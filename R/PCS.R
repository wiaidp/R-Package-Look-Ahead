# Implements the PCS (Phase Change Shift) criterion defined in Appendix D,
# Equation 46, Wildi 2026.
# Computes a feasible generalized PC shift even in the case of singular systems.
#
# Objective:
#   Seeks a monotonically increasing Cross-Correlation Function (CCF) over the
#   set of leads specified in 'Delta'. Rather than imposing an exact constraint
#   (which may be infeasible), the regularization parameter 'lambda' penalizes
#   deviations, steering the solution toward the best attainable behaviour.
#
# Arguments:
#   Delta  : Integer vector of leads over which the CCF should be increasing.
#   beta   : Target slope. A positive value shifts the CCF peak to the right
#            (i.e., toward higher leads).
#   xi     : Wold decomposition coefficients of the process.
#   lambda : Regularization (penalty) parameter controlling the strength of the
#            monotonicity constraint.
#
# Note:
#   The function operates on covariances rather than correlations. This
#   distinction is largely immaterial in practice: since the goal is merely
#   to achieve a zero or mildly positive slope, the scale of the CCF does
#   not affect the validity of the result. Using correlations instead would
#   impose a unit-norm constraint on b, introducing additional complexity
#   and potentially multiple solutions.


PCS_shift_func <- function(Delta, xi, L, beta, lambda)
{
  
  # The sign of beta is flipped here to align the internal convention with the
  # paper's definition: a positive beta in the interface corresponds to a
  # rightward peak shift (toward higher leads).
  beta <- -beta
  
  gamma_all <- xi
  
  # --- Build the shifted covariance matrix 'gammah_mat' ---
  # Each row contains the covariance vector gamma shifted by a specific lead
  # in Delta. The first row corresponds to the baseline lead Delta<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a> - 1,
  # and subsequent rows correspond to leads Delta<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a>, Delta<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[2]</a>, ..., Delta[end].
  gammah_mat <- gamma_all[Delta[1] - 1 + 1:L]
  if (length(Delta) > 0)
  {
    for (i in 1:length(Delta))
      gammah_mat <- rbind(gammah_mat, gamma_all[Delta[i] + 1:L])
  }
  
  # --- Compute the difference vectors d_delta (denoted d_delta in the paper) ---
  # Each row of d_delta is the difference between consecutive rows of gammah_mat,
  # encoding the monotonicity constraint for the corresponding lead pair.
  d_delta <- gammah_mat[1, ] - gammah_mat[2, ]
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      d_delta <- rbind(d_delta, gammah_mat[i, ] - gammah_mat[i + 1, ])
  }
  
  # --- Rank diagnostic ---
  # For an AR(p) process, d_delta %*% t(d_delta) has rank at most p due to the
  # Yule-Walker equations: only p eigenvalues are non-zero. If p <= length(Delta)
  # (i.e., the number of constraints meets or exceeds the model order), the
  # monotonicity constraints may not admit a feasible solution. Inspect the
  # eigenvalues below to assess the effective rank.
  eigen(d_delta %*% t(d_delta))$values
  
  # --- Compute the PCS predictor ---
  
  # a. Assemble the regularized matrix M = I + lambda * sum(d_delta[i,] %o% d_delta[i,])
  M <- diag(rep(1, L)) + lambda * d_delta[1, ] %*% t(d_delta[1, ])
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      M <- M + lambda * d_delta[i, ] %*% t(d_delta[i, ])
  }
  
  # b. Assemble the target vector gamma_sol = gamma_h + lambda * beta * sum(d_delta[i,])
  #    This encodes the desired slope beta into the right-hand side of the linear system.
  gamma_sol <- gammah_mat[h, ] + lambda * beta * d_delta[1, ]
  if (length(Delta) > 1)
  {
    for (i in 2:length(Delta))
      gamma_sol <- gamma_sol + lambda * beta * d_delta[i, ]
  }
  
  # c. Solve the linear system M %*% b = gamma_sol for the PCS filter coefficients b.
  b <- solve(M) %*% gamma_sol
  
  # --- Feasibility check ---
  # Computes the residual |d_delta %*% b - beta| for each constraint.
  # For a feasible system,  residuals should converge to zero as lambda -> Inf.
  # Persistent non-zero residuals indicate an infeasible constraint system
  # (e.g., when the number of constraints exceeds the effective rank of d_delta,
  # as in the AR example above).
  abs(d_delta %*% b - beta)
  
  return(list(b = b, d_delta = d_delta))
  
}














# Implements the PCS (Phase Change Shift) criterion defined in Appendix D, 
# Equation 46, Wildi 2026.
# Computes a feasible generalized PC shift even in the case of singular systems.
#
# Objective:
#   Seeks a monotonically increasing Cross-Correlation Function (CCF) over the
#   set of leads specified in 'Delta'. Rather than imposing an exact constraint
#   (which may be infeasible), the regularization parameter 'lambda' penalizes
#   deviations, steering the solution toward the best attainable behaviour.
#
# Arguments:
#   Delta  : Integer vector of leads over which the CCF should be increasing.
#   beta   : Target slope. A positive value shifts the CCF peak to the right
#            (i.e., toward higher leads).
#   xi     : Wold decomposition coefficients of the process.
#   lambda : Regularization (penalty) parameter controlling the strength of the
#            monotonicity constraint.
#
# Note:
#   The function operates on covariances rather than correlations. This
#   distinction is largely immaterial in practice: since the goal is merely
#   to achieve a zero or mildly positive slope, the scale of the CCF does
#   not affect the validity of the result. Using correlations instead would
#   impose a unit-norm constraint on b, introducing additional complexity
#   and potentially multiple solutions.



PCS_shift_func<-function(Delta,xi,L,beta,lambda)
{
  
# Invert sign of slope or invert sign on gammah_mat[i,]-gammah_mat[i+1,] below. 
  beta<--beta
  
  gamma_all<-xi
  
  
# Stack gammas shifted by increasing lag in matrix
  gammah_mat<-gamma_all[Delta[1]-1+1:L]
  if (length(Delta)>0)
  {
    for (i in 1:length(Delta))
      gammah_mat<-rbind(gammah_mat,gamma_all[Delta[i]+1:L])
  }  
  
  # Specify d_delta in paper
  d_delta<-gammah_mat[1,]-gammah_mat[2,]
  if (length(Delta)>1)
  {
    for (i in 2:(length(Delta)))
      d_delta<-rbind(d_delta,gammah_mat[i,]-gammah_mat[i+1,])
  }
  
  dim(d_delta)
  # For an AR(p), the matrix gammas_delta%*%t(gammas_delta) has rank p: only p non-vanishing eigenvalues
  #   This is because of the Yule-Walker equations
  # If p<=length(Delta) (number of constraints) the problem does not admit a feasible solution
  # Check the rank by looking at the non-vanishing eigenevalues
  eigen(d_delta%*%t(d_delta))$values
  
  
  #------------------
  # Compute PCS predictor
  
  # a. Compute M
  M<-diag(rep(1,L))+lambda*d_delta[1,]%*%t(d_delta[1,])
  if (length(Delta)>1)
  {
    for (i in 2:(length(Delta)))
      M<-M+lambda*d_delta[i,]%*%t(d_delta[i,])
  }  
  
  # Compute target vector
  gamma_sol<-gammah_mat[h,]+lambda*beta*d_delta[1,]
  if (length(Delta)>1)
  {
    for (i in 2:(length(Delta)))
      gamma_sol<-gamma_sol+lambda*beta*d_delta[i,]
  }  
  
  # PCS predictor
  b<-solve(M)%*%gamma_sol
  
  #---------------
  # Check: for large lambda and a feasible system the constraints should be (nearly) met (the absolute differences should converge to zero as lambda\to\infty).
  # For a non-feasible system (check AR example above), the absolute differences do not converge to zero as lambda\to\infty 
  
  abs(d_delta%*%b-beta)
  
  return(list(b=b,d_delta=d_delta))
  
}