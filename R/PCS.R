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
#   gamma_pcs : MA form of PCS problem structure (in classic forecasting this is the Wold decomposition).
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


########################################################################################
PCS_func <- function(h,Delta, gamma_pcs, L, beta, lambda,Type_III=F,scaled_constraints=F)
{
  # MSE h-step predictor  
  gammah<-gamma_pcs[h+1:L]
  
  # Flip the sign of beta to align the internal convention with the paper's
  # definition: a positive beta in the function interface corresponds to a
  # rightward peak shift (toward higher leads), which requires a negative
  # internal slope.
  slope <- -beta
  
  gamma_all <- gamma_pcs
  # --- Build the shifted covariance matrix 'gammah_mat' ---
  # Each row contains the MSE predictor coefficients (gamma_all) shifted by
  # a specific lead value drawn from 'Delta'. 
  # We start with Delta[1] - 1 because we compute differences: gamma_h-gamma_{h-1}
  # and therefore we need gamma_{Delta[1] - 1} to define the first difference.
  if (Type_III)
  {
    gammah_mat<-NULL
  } else
  {
    if (Delta[1]<1)
    {
      print("Delta[1] must be larger or equal 1")
      return()
    }
    gammah_mat <- gamma_all[Delta[1] - 1 + 1:L]/sqrt(sum(gamma_all^2) ) 
  }
  if (length(Delta) > 0)
  {
    for (i in 1:length(Delta))
    {
      if (Delta[i] + 1<1)
      {
        print("Delta[i] + 1<1")
        print("The index is outside gamma_pcs")
        return()
      }
      if (Delta[i] + L>length(gamma_all))
      {
        print("Delta[i] + L>length(gamma_all)")
        print("The index is outside gamma_pcs")
        return()
      }
      
      gammah_mat <- rbind(gammah_mat,
                          gamma_all[Delta[i] + 1:L]/sqrt(sum(gamma_all^2) ) )
    }
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
  # When scaled_constraints==T then we use unit-scaled d_delta[i, ]
  # This implies that b' * d_delta[i, ]  =  beta does not depend on changing scales of gamma_h-gamma_{h-1) in delta (and the scale of b is fixed).
  #  -In an AR(1) case the system is then still feasible (whereas if scaled_constraints==F, the system is not feasible anymore)  
  
  if (scaled_constraints)
  {
    d_delta <- (gammah_mat[1, ] - gammah_mat[2, ])/sqrt(sum((gammah_mat[1, ] - gammah_mat[2, ])^2))
  } else
  {
    d_delta <- (gammah_mat[1, ] - gammah_mat[2, ])
  } 
    
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
    {
      if (scaled_constraints)
      {
        d_delta <- rbind(d_delta, (gammah_mat[i, ] - gammah_mat[i + 1, ])/sqrt(sum((gammah_mat[i, ] - gammah_mat[i + 1, ])^2)))
      } else
      {
        d_delta <- rbind(d_delta, (gammah_mat[i, ] - gammah_mat[i + 1, ]))
      } 
    }
  }
  d_delta<-matrix(d_delta,nrow=length(Delta)-ifelse(Type_III,1,0) )
  
  # For a full-rank PCS system, the `squared' constraint matrix should be strictly positive definite  
  min_eigen<-min(eigen(d_delta%*%t(d_delta))$values)
  max_eigen<-max(eigen(d_delta%*%t(d_delta))$values)
  if (min_eigen/max_eigen<10^{-12})
    print("PCS constraints eventually singular: problem is potentially infeasible")
  
  
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
  if (F)
  {
    M <- diag(rep(1, L)) + lambda * d_delta[1, ] %*% t(d_delta[1, ])
    if (length(Delta) > 1&!Type_III)
    {
      for (i in 2:length(Delta))
        M <- M + lambda * d_delta[i, ] %*% t(d_delta[i, ])
    }
  }
  N <-d_delta[1, ] %*% t(d_delta[1, ])
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
      N <- N + d_delta[i, ] %*% t(d_delta[i, ])
  }
  M <- diag(rep(1, L))+lambda * N
  
  # b. Assemble the right-hand side vector:
  #       gamma_sol = gamma_h + lambda * slope * sum_i d_delta[i,]
  #    The second term encodes the desired target slope into the linear system:
  #    in the limit lambda -> Inf, b' * d_delta[i,] -> slope for every i,
  #    provided the system is feasible.
  gamma_sol <- gamma_pcs[h+1:L] + lambda * slope * d_delta[1, ]
  if (length(Delta) > 1&!Type_III)
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
  t(b)%*%gammah
  
  b_mse<-b*as.double(t(b)%*%gammah/(t(b)%*%b))
  
  return(list(b = b, d_delta = d_delta,b_mse=b_mse,M=M,N=N,gamma_sol=gamma_sol))
  
}


# gamma_pcs is the original unperturbated matrix with MSE predictors: this is used to specify the target h-step ahead MSE predictor
# gammah_mat_perturbated is the matrix of perturbated MSE predictors: this is used to build the constraints
PCS_perturbation_func <- function(h,Delta, gamma_pcs, L, beta,lambda,gammah_mat_perturbated, Type_III=F,scaled_constraints=F)
{
  

  # MSE h-step predictor  
  gammah<-gamma_pcs[h+1:L]
  
  # Flip the sign of beta to align the internal convention with the paper's
  # definition: a positive beta in the function interface corresponds to a
  # rightward peak shift (toward higher leads), which requires a negative
  # internal slope.
  slope <- -beta
  
  # --- Compute consecutive difference vectors ('d_delta') ---
  # Each row of d_delta is the difference between two consecutive rows of
  # gammah_mat_perturbated:
  #
  #   d_delta[i, ] = gammah_mat_perturbated[i, ] - gammah_mat_perturbated[i + 1, ]
  #
  # These differences encode the pairwise monotonicity constraints: requiring
  #
  #   b' * d_delta[i, ] = beta > 0
  #
  # forces the CCF to increase by beta from lead Delta[i] to lead Delta[i+1].
  #
  # When scaled_constraints==T then we use unit-scaled d_delta[i, ]
  # This implies that b' * d_delta[i, ]  =  beta does not depend on changing scales of gamma_h-gamma_{h-1) in delta (and the scale of b is fixed).
  #  -In an AR(1) case the system is then still feasible (whereas if scaled_constraints==F, the system is not feasible anymore)  
  
  if (scaled_constraints)
  {
    d_delta <- (gammah_mat_perturbated[1, ] - gammah_mat_perturbated[2, ])/(sqrt(sum((gammah_mat_perturbated[1, ] - gammah_mat_perturbated[2, ])^2)))
  }
  {
    d_delta <- (gammah_mat_perturbated[1, ] - gammah_mat_perturbated[2, ])
  }
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
    {
      if (scaled_constraints)
      {
        d_delta <- rbind(d_delta, (gammah_mat_perturbated[i, ] - gammah_mat_perturbated[i + 1, ])/sqrt(sum((gammah_mat_perturbated[i, ] - gammah_mat_perturbated[i + 1, ])^2)))
      } else
      {
        d_delta <- rbind(d_delta, (gammah_mat_perturbated[i, ] - gammah_mat_perturbated[i + 1, ]))
      } 
    }
  }
  
  d_delta<-matrix(d_delta,nrow=length(Delta)-ifelse(Type_III,1,0) )
  
  # For a full-rank PCS system, the `squared' constraint matrix should be strictly positive definite  
  min_eigen<-min(eigen(d_delta%*%t(d_delta))$values)
  max_eigen<-max(eigen(d_delta%*%t(d_delta))$values)
  if (min_eigen/max_eigen<10^{-12})
    print("PCS constraints eventually singular: problem is potentially infeasible")
  
  
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
  if (F)
  {
    M <- diag(rep(1, L)) + lambda * d_delta[1, ] %*% t(d_delta[1, ])
    if (length(Delta) > 1&!Type_III)
    {
      for (i in 2:length(Delta))
        M <- M + lambda * d_delta[i, ] %*% t(d_delta[i, ])
    }
  }
  N <-d_delta[1, ] %*% t(d_delta[1, ])
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
      N <- N + d_delta[i, ] %*% t(d_delta[i, ])
  }
  M <- diag(rep(1, L))+lambda * N
  
  # b. Assemble the right-hand side vector:
  #       gamma_sol = gamma_h + lambda * slope * sum_i d_delta[i,]
  #    The second term encodes the desired target slope into the linear system:
  #    in the limit lambda -> Inf, b' * d_delta[i,] -> slope for every i,
  #    provided the system is feasible.
  gamma_sol <- gammah + lambda * slope * d_delta[1, ]
  if (length(Delta) > 1&!Type_III)
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
  
  return(list(b = b, d_delta = d_delta,b_mse=b_mse,gamma_sol=gamma_sol,M=M,N=N,gamma_sol=gamma_sol))
  
}










###########################################################################################################################
# Older designs

PCS_delta_perturbation_func <- function(h,Delta, gamma_pcs, L, beta, lambda,Type_III=F,perturbation_delta_mat=NULL,scaled_constraints=F)
{
  dim_row<-length(Delta)+ifelse(Type_III,1,0)
  
  if (!is.null(perturbation_delta_mat))
  {
    if (!is.matrix(perturbation_delta_mat))
    {
      print("perturbation_delta_mat must be a matrix with two columns")
      return()
    }
    if (dim(perturbation_delta_mat)[2]<2)
    {
      print("perturbation_delta_mat needs two columns: the perturbation (first column) and the lag (second column) at which the perturbation is applied.")
      print("no perturbation is currently applied")
      perturbation_delta_mat<-matrix(rep(0,2*dim_row,ncol=2))
    } else
    {
      perturbation_delta_mat<-matrix(perturbation_delta_mat[,1:2],ncol=2)
      if (dim(perturbation_delta_mat)[1]<dim_row)
      {
        print("dim(perturbation_delta_mat)[1]<dim_row: no perturbation is applied to the missing lags")
        perturbation_delta_mat<-rbind(perturbation_delta_mat,matrix(rep(0,2*(dim_row-dim(perturbation_delta_mat)[1])),ncol=2))
      }
    }
  } else
  {
    perturbation_delta_mat<-matrix(rep(0,2*dim_row,ncol=2))
  }
  if (max(perturbation_delta_mat[,2])>L-1)
  {
    print("max(perturbation_delta_mat[,2])>L: the maximal lag cannot exceed L-1")
    return()
  }
  if (min(perturbation_delta_mat[,2])<1)
  {
    print("min(perturbation_delta_mat[,2])<0: the minimal lag cannot be smaller than 0")
    return()
  }
  # MSE h-step predictor  
  gammah<-gamma_pcs[h+1:L]
  
  # Flip the sign of beta to align the internal convention with the paper's
  # definition: a positive beta in the function interface corresponds to a
  # rightward peak shift (toward higher leads), which requires a negative
  # internal slope.
  slope <- -beta
  
  gamma_all <- gamma_pcs
  
  # --- Build the shifted covariance matrix 'gammah_mat' ---
  # Each row contains the MSE predictor coefficients (gamma_all) shifted by
  # a specific lead value drawn from 'Delta'. 
  # We start with Delta[1] - 1 because we compute differences: gamma_h-gamma_{h-1}
  # and therefore we need gamma_{Delta[1] - 1} to define the first difference.
  if (Type_III)
  {
    gammah_mat<-NULL
  } else
  {
    if (Delta[1]<1)
    {
      print("Delta[1] must be larger or equal 1")
      return()
    }
    gamma_vec<-gamma_all[Delta[1] - 1 + 1:L]
# Apply perturbation (0 if not specified): the lag is  perturbation_delta_mat[1,2]+1; the perturbation is perturbation_delta_mat[1,1]
    gamma_vec[perturbation_delta_mat[1,2]+1]<-gamma_vec[perturbation_delta_mat[1,2]+1]+perturbation_delta_mat[1,1] 
    gamma_vec<-gamma_vec/sqrt(sum(gamma_all^2) )
    gammah_mat <- gamma_vec
    
  }
  if (length(Delta) > 0)
  {
    for (i in 1:length(Delta))
    {
      if (Delta[i] + 1<1)
      {
        print("Delta[i] + 1<1")
        print("The index is outside gamma_pcs")
        return()
      }
      if (Delta[i] + L>length(gamma_all))
      {
        print("Delta[i] + L>length(gamma_all)")
        print("The index is outside gamma_pcs")
        return()
      }
      
      gamma_vec<-gamma_all[Delta[i] + 1:L]
      # Apply perturbation (0 if not specified): the lag is  perturbation_delta_mat[1,2]+1; the perturbation is perturbation_delta_mat[1,1]
      gamma_vec[perturbation_delta_mat[i+ifelse(Type_III,1,0),2]+1]<-gamma_vec[perturbation_delta_mat[i+ifelse(Type_III,1,0),2]+1]+perturbation_delta_mat[i+ifelse(Type_III,1,0),1] 
      gamma_vec<-gamma_vec/sqrt(sum(gamma_all^2) )
      gammah_mat <- rbind(gammah_mat,gamma_vec)
    }
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
  # When scaled_constraints==T then we use unit-scaled d_delta[i, ]
  # This implies that b' * d_delta[i, ]  =  beta does not depend on changing scales of gamma_h-gamma_{h-1) in delta (and the scale of b is fixed).
  #  -In an AR(1) case the system is then still feasible (whereas if scaled_constraints==F, the system is not feasible anymore)  
  
  # If gamma_k are all normalized to unit length then 
  # Since CCF(k) = b' * gamma_k / ||b||, the
  # slope in terms of the CCF is:
  #
  #   CCF(Delta[i+1]) - CCF(Delta[i])  = b' * (gamma_{i+1}- gamma_i)/ ||b||=  beta / ||b||
  #
  # However, if gamma_i are of different lengths, then the link between the slope of the CCF 
  # and beta is not fixed anymore (depends on i)
  if (scaled_constraints)
  {
    d_delta <- (gammah_mat[1, ] - gammah_mat[2, ])/(sqrt(sum((gammah_mat[1, ] - gammah_mat[2, ])^2)))
  }
  {
    d_delta <- (gammah_mat[1, ] - gammah_mat[2, ])
  }
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
    {
      if (scaled_constraints)
      {
        d_delta <- rbind(d_delta, (gammah_mat[i, ] - gammah_mat[i + 1, ])/sqrt(sum((gammah_mat[i, ] - gammah_mat[i + 1, ])^2)))
      } else
      {
        d_delta <- rbind(d_delta, (gammah_mat[i, ] - gammah_mat[i + 1, ]))
      } 
    }
  }
  
  d_delta<-matrix(d_delta,nrow=length(Delta)-ifelse(Type_III,1,0) )
  
  # For a full-rank PCS system, the `squared' constraint matrix should be strictly positive definite  
  min_eigen<-min(eigen(d_delta%*%t(d_delta))$values)
  max_eigen<-max(eigen(d_delta%*%t(d_delta))$values)
  if (min_eigen/max_eigen<10^{-12})
    print("PCS constraints eventually singular: problem is potentially infeasible")
  
  
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
  if (length(Delta) > 1&!Type_III)
  {
    for (i in 2:length(Delta))
      M <- M + lambda * d_delta[i, ] %*% t(d_delta[i, ])
  }
  
  # b. Assemble the right-hand side vector:
  #       gamma_sol = gamma_h + lambda * slope * sum_i d_delta[i,]
  #    The second term encodes the desired target slope into the linear system:
  #    in the limit lambda -> Inf, b' * d_delta[i,] -> slope for every i,
  #    provided the system is feasible.
  gamma_sol <- gammah_mat[length(Delta), ] + lambda * slope * d_delta[1, ]
  if (length(Delta) > 1&!Type_III)
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
  
  return(list(b = b, d_delta = d_delta,b_mse=b_mse,M=M,gamma_sol=gamma_sol))
  
}


