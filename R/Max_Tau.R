##############################################################################################
# These function compute the Max-Tau predictor
# Two parts: 1. Dual optimization problem; 2. Primal optimization problem
##############################################################################################



##############################################################################################
# 1. DUAL OPTIMIZATION: single frequency omega0
##############################################################################################

##############################################################################################
# 1.1 Dual Without curvature constraint
##############################################################################################


# Computes the Max-Tau predictor
# gamma_h: target in MA form (typically MSe predictor or nowcast but could be arbitrary filter, not necessarily MSE optimal)
# target_correlation: desired correlation with gamma_h (between -1 and +1)
# omega0: reference frequency for computing the lead (typically zero).
# phase_excess: if T then the lead is tau := tau_b-tau_h (a lead with respect to gamma_h); otherwise tau=tau_b (in absolute terms or with respect to identity).
max_tau_dual_func <- function(gamma_target, target_correlation,omega0=0,phase_excess=F) {
  
  if (omega0<0|omega0>pi)
  {
    print("omega0 belongs in the interval [0,pi]")
    return()
  }
  
  if (target_correlation<0)
  {
    print("Target correlation must be positive")
    return()
  }
  if (target_correlation>1)
  {
    print("Target correlation must be smaller 1")
    return()
  }
  gamma_target<-as.vector(gamma_target)
  if (t(gamma_target)%*%gamma_target<10^{-20})
  {
    print("gamma_target must differ from zero")
    return()
  }
  L<-length(gamma_target)
  
  # Rescale target correlation  
  alpha_h<-as.double(target_correlation*sqrt((gamma_target)%*%gamma_target))
  
  # Compute weights in objective function and time-shift constraint  
  weight_obj<-compute_weights_max_tau_func(omega0,phase_excess,gamma_target)
  
  if (is.null(weight_obj))
    return()
  
  objective_weights=weight_obj$objective_weights
  constraint_weights=weight_obj$constraint_weights
  
  
  C <- cbind(gamma_target, constraint_weights)
  
  # Compute structural matrices (Least Squares and Projections)
  # (C^T C)^-1
  CTC_inv <- solve(t(C) %*% C)
  
  # Base vectors for b_LS(f) = u + f * v
  u <- C %*% (CTC_inv %*% c(alpha_h, 0))
  v <- C %*% (CTC_inv %*% c(0, 1))
  
  # Projection matrix onto the null space of C^T: P_C_perp = I - C (C^T C)^-1 C^T
  I <- diag(L)
  P_C <- C %*% CTC_inv %*% t(C)
  P_C_perp <- I - P_C
  
  # Compute scalars for the quadratic equation
  K_perp <- as.numeric(t(objective_weights) %*% P_C_perp %*% objective_weights)
  
  # Determine feasible region: where the constraints intersect:
  
  feasible_obj<-determine_feasible_region_max_tau(u,v,objective_weights)
  
  if (is.null(feasible_obj))
    return()
  
  admissible_range=feasible_obj$admissible_range
  A=feasible_obj$A
  B=feasible_obj$B
  D=feasible_obj$D
  E=feasible_obj$E
  
  # Determine minimum of objective in feasible range  
  min_obj<-determine_minimum_func(A,B,D,E,K_perp,admissible_range)
  
  if (is.null(min_obj))
    return()
  f_min=min_obj$f_min

  b_LS <- u + f_min * v
  # Compute nu: note that for x_min within the admissibility range, the value under the square root is positive but due to numerical imprecision it might become `minus zero': we therefore rely on the absolute value to avoid errors.    
  nu<-as.double(sqrt(((1-t(b_LS)%*%b_LS))/K_perp))
  
  # Final filter weights
  b_opt <- b_LS - nu * (P_C_perp %*% objective_weights)#objective_weights[2:L]-objective_weights[1:(L-1)]
  
  if (F)
  {
# Check time-shift constraint: difference should vanish    
    sum(b_opt*constraint_weights)-f_min
  }
  
  # Shift at reference frequency
  if (omega0==0)
  {
    tau_max<--as.double(t(b_opt)%*%(0:(L-1))/sum(b_opt))
    if (phase_excess)
    {
      tau_max_excess<-tau_max-sum(-gamma_target*(0:(L-1)))/sum(gamma_target)
    } else
    {
      tau_max_excess<-tau_max
    }
    
  } else
  {
    # Lead: based on general formula with objective_weights and constraint_weights which are valid irrespective of phase_excess   
    tau_max_excess<--atan(t(b_opt)%*%objective_weights/t(b_opt)%*%constraint_weights)/omega0
    if (phase_excess)
    {
      # Phase_excess==T: Absolute lead is phase excess + lead or lag of MSE target
      tau_max<-tau_max_excess-Arg(sum(gamma_target*exp(1.i*omega0*(0:(L-1)))))/omega0
    } else
    {
      # Phase_excess==F: Absolute lead is identical with phase excess
      tau_max<-tau_max_excess
    }
  }
  # Return results as a list
  return(list(
    b_opt = as.vector(b_opt),
    f_opt = f_min,
    tau_max=tau_max,
    tau_max_excess=tau_max_excess
  ))
}




# Computes the weights of predictor in objective function and (time-shift) constraint.
compute_weights_max_tau_func<-function(omega0,phase_excess,gamma_target)
{
  L<-length(gamma_target)
  if (omega0>0)
  {
    # Phase excess over benchmark MSE or absolute (i.e., over identity)    
    if (phase_excess)
    {
      # Lead over MSE benchmark
      # Sign convention as in paper: exp(-1.i*omega0*(0:(L-1)))
      Gamma_h_omega0=sum(gamma_target*exp(-1.i*omega0*(0:(L-1))))
      if (abs(Im(Gamma_h_omega0))>0&abs(Re(Gamma_h_omega0))>0)
      {
        constraint_weights <- cos(omega0*(0:(L-1)))*Re(1/Gamma_h_omega0)+sin(omega0*(0:(L-1)))*Im(1/Gamma_h_omega0)
    # We take minus objective since we want to minimze    
        objective_weights<-sin(omega0*(0:(L-1)))*Re(1/Gamma_h_omega0)-cos(omega0*(0:(L-1)))*Im(1/Gamma_h_omega0)
      }
      # If the transfer function of the target is a positive number we use the same objective_weights as below (absolute: lead over identity)      
      if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)>0)
      {
        constraint_weights <- cos(omega0*(0:(L-1)))
        objective_weights<-sin(omega0*(0:(L-1)))
      }
      # If the transfer function of the target is a negative number we shift by pi (invert the sign).       
      if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)<0)
      {
        constraint_weights <- -cos(omega0*(0:(L-1)))
        objective_weights<--sin(omega0*(0:(L-1)))
      }
      if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)==0)
      {
        print("Transfer function of target vanishes at omega_0: use another omega_0 or set phase_excess==F")
      }
      # If  the transfer function of the target is purely imaginary: shift by pi/2 i.e. use cos (this does not depend on the sign)    
      if (Re(Gamma_h_omega0)==0)
      {
        objective_weights<-cos(omega0*(0:(L-1)))
        print("Transfer function of target is purely imaginary at omega0. This case has not been analyzed yet. Change omega0 or set phase_excess==F")
        return()
      }
      
    } else
    {
      # Absolute: lead over identity  
      constraint_weights <- cos(omega0*(0:(L-1)))
      objective_weights <- sin(omega0*(0:(L-1)))
    }
  } else
  {
    if (phase_excess)
    {
      # Relative lead over MSE: Tau=Tau_b-Tau_h
      # Notes: 
      # 1. we minimize objective i.e. sign of objective is inverted. 
      # 2. Sign convention as in paper: a) Gamma=sum gamm_k*exp(-ikomega) (minus sign) and b) Gamma=A(omega)*exp(i*Phi(omega)) (positive sign)
      constraint_weights<-rep(1,L)*sum(gamma_target)/sum(gamma_target)^2
      objective_weights<-(-rep(1,L)*sum(gamma_target*(0:(L-1)))+(0:(L-1))*sum(gamma_target))/sum(gamma_target)^2
    } else
    {
      # Absolute lead (over identity): Tau=Tau_b-0  
      constraint_weights <- rep(1,L)
      objective_weights<-(0:(L-1))
    }
    
  }
  return(list(objective_weights=objective_weights,constraint_weights=constraint_weights))
}


# This function is a check for the formula of objective and constraint when omega=0 and phase_excess==T
# The function is not used in deriving the solution (it serves as a validation test only)
check_objective_constraint_func<-function()
{
  set.seed(97)
  L<-20
  b<-rnorm(L)
  gamma_target<-rnorm(L)
  
# Check phase excess when omega=0
  
# Select omega close to 0  
  omega<-.0001
# Sign convention as in paper  
  Gammab<-sum(b*exp(-1.i*(0:(L-1))*omega))
  Gammah<-sum(gamma_target*exp(-1.i*(0:(L-1))*omega))
# Objective: (Note: its sign must be inverted)
  (Re(Gammab)*Im(1/Gammah)+Im(Gammab)*Re(1/Gammah))/omega
  (sum(b)*sum(gamma_target*(0:(L-1)))-sum(b*(0:(L-1)))*sum(gamma_target))/sum(gamma_target)^2
# Constraint:
  Re(Gammab)*Re(1/Gammah)
  sum(b)*sum(gamma_target)/sum(gamma_target)^2
  
}


# Determines the feasibility region: intersection of the constraints.
determine_feasible_region_max_tau<-function(u,v,objective_weights)
{
  # Want to minimize objective function
  # V(f):=t(k)%*%u/f+t(k)%*%v-\frac{sqrt(K_perp)}{f}*sqrt(1-t(u)%*%u-2*f*t(u)%*%v-f^2t(v)%*%v)
  # This rewriten as 
  # W(x)=t(k)%*%u*x+t(k)%*%v-sqrt(K_perp)*sqrt(1-t(u)%*%u*x^2-2*t(u)%*%v*x-t(v)%*%v)
  # =D*x+t(k)%*%v-sqrt(K_perp)*sqrt(P_1(x)), with P_1(x)=A*x^2+B*x+E
  A<-1-as.double(t(u)%*%u)
  B<-(-2*as.double(t(u)%*%v))
  E<-(-as.double(t(v)%*%v))
  # D is the linear term in W(x) (the constant is irrelevant for minimization)  
  D<-t(objective_weights)%*%u
  
  # Note that P_1(x) should be implicitely positive because 1-b_LS'b_LS >= 0 (positive by construction due to length constraint). However, P_1(x) is not always positive for all x. We must therefore ensure to select x>=0 (must be positive for time-shift constraint) in the admissible range. For that we first check that B^2-4*A*C>0. If not no zeroes exist and since P_1(0)=E<0 we conclude P_1(x)<0 for all x and hence no solution exists.
  if (B^2-4*A*E<0)
  {
    print("The constraints do not intersect. Select another target_correlation or a larger L")
    return()
  }
  # Distinguish quadratic and linear cases:  
  if (abs(A)>0)
  {
    # P_1 is quadratic    
    # Next we determine the range of admissibility. Note that P_1(0)=E<0. If A>0 then P_1(x) is convex and P_1(x)>0 for all large x. If A<0 then P_1(x) is concave and P_1(x) < 0 for large x. Since B^2-4*A*C > 0 (checked above) zeroes must exist but they could be negative. If all zeroes are negative then no solution exists since then P_1(0)=E<0 implies P_1(x)<0 for all x>=0 (constraints do not have an intersection).
    z1<-(-B+sqrt(B^2-4*A*E))/(2*A)
    z2<-(-B-sqrt(B^2-4*A*E))/(2*A)
    if (z1<0&z2<0)
    {
      print("All zeroes of P_1 are negative: the constraints do not intersect. Select another target_correlation or a larger L")
      return()
    }
    # Since z1<0&z2<0 is excluded by the above check, we have at least one positive root of P_1(x) and since P_1(0)=E<0 we infer that there exist x>0 such that P_1(x)>0. We distinguish two cases:
    # i) A>0: P_1(x) convex. Since P_1(0)<0 we infer that z1=min(z1,z2)<0 and z2=max(z1,z2)>0. Hence for all x>=z2 we must have P_1(x)>= 0 (noting that x >= 0).
    if (A>0)
      admissible_range<-c(max(z1,z2),Inf)
    # ii) A<0: P_1(x) concave. Note that we have already excluded the case of  no roots above. 
    # Since P_1(0)<0 we infer that either z1 and z 2 <0  or z1 and z2 > 0 due to concavity. The former case (both negative) has already been checked above (rejected). 
    # Hence z1=min(z1,z2)>0 and z2=max(z1,z2)>0 and for all z1<=x<=z2 we must have P_1(x)>= 0
    if (A<0)
      admissible_range<-c(min(z1,z2),max(z1,z2))
  } else
  {
    # Quadratic P_1 degenerates to linear function: check that function is not constant:    
    if (abs(B)>0)
    {
      # Not constant function.  Zero:      
      z1<--E/B
      # Determine admissible range of positive values of P_1(x)
      if (B<0)
      {
        # Monotonically decreasing P_1    
        if (z1>0)
        {
          admissible_range<-c(0,z1)
        } else
        {
          print("Degenerate linear case without solution: select another target correlation or a larger L")
          return()
        }
      } else
      {
        # Increasing P_1: all x to the right of z1 are admissible but they must be positive.   
        admissible_range<-c(max(0,z1),Inf)
      }
    } else
    {
      print("P_1 quadratic polynomial degenerates to a constant")
      print("This case has not been analyzed yet")
      return()
    }
  }
  return(list(admissible_range=admissible_range,A=A,B=B,D=D,E=E))
}



# Computes minimum within feasibiltiy region.
determine_minimum_func<-function(A,B,D,E,K_perp,admissible_range)
{
  # Now that we have determined the feasible region where P_1(x)>=0 so that sqrt(P_1(x)) and thus the objective W(x) are well defined we look for the minimum of W(x) in this region. For this purpose we compute the derivative \dot{W}(x):
  # \dot{W}(x)= D-sqrt(P_perp)*\frac{2*A*x+B}{2*sqrt(P_1(x))}
  # We also compute the second order derivative of the objective W(x), see proof:
  # \dot{\dot{W}}(x)=\frac{B^2-4*A*E}{4*P_1(x)^(3/2)} 
  # This is interesting because in the admissible range B^2-4*A*E > 0 and hence \dot{\dot{W}}(x) > 0 implies that W(x) is convex in the admissible range (it might become non-convexe outside). 
  
  # We now distinguish two cases: I) The first derivative of the objective has a zero in the admissible range and II) no zero in the admissible range. 
  # Case I): Convexity of W(x) implies that the minimal value of the objective is attained at the zero of its derivative. Let's compute the zero: \dot{W}(x)=0 gives after rearranging:
  # D*sqrt(P_1(x))=\sqrt(P_perp)*(A*x+B). Since the left side has sign equal to sign(D), we require sign(A*x+B)=sign(D) at the admissible zero. 
  # Compute the zero(es) of the first derivative: this leads to a new squared function P_2(x) with coefficients:
  a<-as.double(4*A*(D^2-K_perp*A))
  b<-as.double(4*B*(D^2-K_perp*A))
  c<-as.double(4*D^2*E-K_perp*B^2)
  
  if (b^2-4*a*c>0)
  {
    # Compute the roots of the quadratic: to select the correct root we impose sign(2*A*root+B)==sign(D). Otherwise, the result of squaring is not correct.
    eps1<-(-b-sqrt(b^2-4*a*c))/(2*a)
    eps2<-(-b+sqrt(b^2-4*a*c))/(2*a)
    # Check which of the two roots is admissible and effectively a zero of the first order derivative.
    # Note that since W is convex in the admissible region it can only have one zero in its derivative in this region: so the solution must be either z1, or z2 or none.
    if (sign(2*A*eps1+B)==sign(D))
    {
      # It is eps1: in which case we do not have to check for eps2 which cannot be a second different root of the derivative in the admissibility region due to convexity.      
      z<-eps1
    } else
    {
      # If not eps1 it could be eps2      
      if (sign(2*A*eps2+B)==sign(D))
      {
        z<-eps2
      } else
      {
        z<-NULL
      }
    } 

# Must check that root is in admissible region: otherwise set z=NULL    
    if (!is.null(z))
      if (z<=admissible_range[1]|z>=admissible_range[2])  
        z<-NULL
  } else
  {
    # No roots: this is the same case as root is not in admissible region, above.    
    z<-NULL
  }
  # If a root exists and is admissible, then the minimum is taken there since W is convex in admissible region. In all other cases 
  # the minimum is taken at the boundary of the admissible region. Which boundary depends on the sign of the derivative.
  if (!is.null(z))
  {
    # Case I): the first order derivative vanishes in the admissible range and by convexity this must be the minimum of the objective function
    x_min<-z
  } else
  {
    # Case II): the first order derivative of the objective does not vanish (at all or in the admissible range). If the derivative at any single admissible point is positive then it must be positive in the entire admissible region (but not necessarily everywhere) and the minimum of the objective function is taken at the left boundary of admissibility (otherwise at the right one). Note that we can not use the boundary values of admissible_range to verify the sign of the derivative because the square root vanishes and thus the ratio is not well-defined. But we can use any point strictly within the admissibility region since then the square root will not vanish.
    # We first check that the admissible range is not a single point:    
    if (admissible_range[1]<admissible_range[2])
    {
      if (admissible_range[2]<Inf) 
      {
        point_within_range<-admissible_range[1]+(admissible_range[2]-admissible_range[1])/2
      } else
      {
        # Since the upper bound is infinity we can add any positive number to admissible_range[1]: here 10        
        point_within_range<-admissible_range[1]+10
      }
      # Derivative at point_within_range positive?      
      if (D-sqrt(K_perp)*(2*A*point_within_range+B)/(2*sqrt(A*point_within_range^2+B*point_within_range+E))>0)
      {
        # If yes: minimum of objective is attained at lower bound of admissibility region        
        x_min<-admissible_range[1]
      } else
      {
        # Otherwise at upper bound        
        x_min<-admissible_range[2]
      }
    } else
    {
      # If the admissible range is a single point then we can use either one      
      x_min<-admissible_range[1]
      # This is the same...      
      x_min<-admissible_range[2]
    }
  }  # 
  # Note that x_min cannot be zero since f=1/x_min=infty would conflict with the unit scaling (zero is not an admissible value for x). Therefore we can invert:  
  if (x_min<Inf)
  {  
    f_min<-1/x_min  
  } else
  {
    # A small positive number rather than zero because of numerical precision (this gives a safety margin for computing the square root in the expression for nu below)    
    # Note that this number is small relative to the unit-scaling t(b)%*%b=1, i.e., it is small in absolute terms.    
    f_min<-10^(-12)
  }
  return(list(f_min=f_min))
}




##############################################################################################
# 1.2 Dual With curvature constraint: additional scalar e_val for curvature constraint e_val = b%*%(0:(L--1))^2
##############################################################################################

max_tau_dual_curvature <- function(gammah, target_correlation, e_val) {
  L <- length(gammah)
  alpha_h<-target_correlation*sqrt(sum(gammah*gammah))
  k <- 0:(L - 1)
  k2 <- k^2
  ones <- rep(1, L)
  
  # 1. Setup constraint matrix V and projections
  V <- cbind(ones, gammah, k2)
  VtV_inv <- solve(t(V) %*% V)
  
  c0 <- c(0, alpha_h, e_val)
  c1 <- c(1, 0, 0)
  
  w <- VtV_inv %*% t(V) %*% k
  w0 <- sum(c0 * w)
  w1 <- sum(c1 * w)
  
  Proj_V_k <- V %*% w
  k_perp <- k - Proj_V_k
  K_sq <- sum(k_perp^2)
  K <- sqrt(K_sq)
  
  # 2. Compute scalar constants
  C_val <- sum(c0 * (VtV_inv %*% c0))
  B_val <- 2 * sum(c1 * (VtV_inv %*% c0))
  A_val <- sum(c1 * (VtV_inv %*% c1))
  
  best_f <- NA
  best_obj <- Inf
  best_b <- rep(NA, L)
  
  # Function to evaluate and update the best candidate
  evaluate_candidate <- function(f_cand, z_norm) {
    c_vec <- c0 + f_cand * c1
    v_f <- V %*% VtV_inv %*% c_vec
    
    # z is anti-parallel to k_perp
    z <- -k_perp * z_norm / K
    b_cand <- v_f + z
    
    obj <- sum(b_cand * k) / f_cand
    
    # Update global best if valid
    if (is.finite(obj) && obj < best_obj) {
      best_obj <<- obj
      best_f <<- f_cand
      best_b <<- b_cand
    }
  }
  
  # --- STEP A: Check the Boundary (where ||z|| = 0) ---
  # A*f^2 + B*f + (C - 1) = 0
  bound_disc <- B_val^2 - 4 * A_val * (C_val - 1)
  if (bound_disc >= 0) {
    f_bound1 <- (-B_val + sqrt(bound_disc)) / (2 * A_val)
    f_bound2 <- (-B_val - sqrt(bound_disc)) / (2 * A_val)
    # At boundary, z_norm is exactly 0
    evaluate_candidate(f_bound1, 0)
    evaluate_candidate(f_bound2, 0)
  }
  
  # --- STEP B: Check the Interior (where derivative = 0) ---
  Q2 <- K_sq * (B_val^2) / 4 + (w0^2) * A_val
  Q1 <- (w0^2) * B_val - K_sq * B_val * (1 - C_val)
  Q0 <- K_sq * (1 - C_val)^2 - (w0^2) * (1 - C_val)
  
  interior_disc <- Q1^2 - 4 * Q2 * Q0
  if (interior_disc >= 0) {
    f_roots <- c((-Q1 + sqrt(interior_disc)) / (2 * Q2), 
                 (-Q1 - sqrt(interior_disc)) / (2 * Q2))
    
    for (f in f_roots) {
      z_norm_sq <- 1 - C_val - B_val * f - A_val * f^2
      
      if (z_norm_sq >= -1e-10) {
        z_norm_sq <- max(0, z_norm_sq)
        
        # Verify it's not an extraneous root from squaring
        lhs <- K * (1 - C_val - (B_val / 2) * f)
        rhs <- w0 * sqrt(z_norm_sq)
        
        if (abs(lhs - rhs) < 1e-7) {
          evaluate_candidate(f, sqrt(z_norm_sq))
        }
      }
    }
  }
  
  if (is.infinite(best_obj)) {
    stop("No valid roots found. The constraints do not intersect the unit sphere.")
  }
  
  return(list(f = best_f, b = best_b, objective = best_obj))
}




##############################################################################################
# 1.3 Dual with multiple frequencies: omega is a vector with several frequencies.
# Frequency zero cannot be addressed currently: the function stops with an error message.
# If omega is a single frequency larger zero, the function replicates max_tau_dual_func above.
##############################################################################################

max_tau_dual_mutiple_freq_func <- function(gamma_target, target_correlation, omega) {
  
  if (abs(target_correlation)>1)
  {
    print("Target correlation must be smaller one in absolute value")
    return()
  }
  if (0%in%omega)
  {
    print("Currently the code cannot handle zero frequency!")
    return()
  }
  L <- length(gamma_target)
  I <- length(omega)
  # Rescale target correlation  
  alphah<-as.double(target_correlation*sqrt((gamma_target)%*%gamma_target))
  
  
  # Create vectors
  S <- rep(0, L)
  C <- matrix(0, nrow = L, ncol = I + 1)
  C[, 1] <- gamma_target
  
  for (i in 1:I) {
    t <- 0:(L-1)
    ci <- cos(omega[i] * t)
    si <- sin(omega[i] * t)
    C[, i + 1] <- ci
    S <- S + si
  }
  
  # Matrices and Projections
  CTC_inv <- solve(t(C) %*% C)
  v0 <- c(alphah, rep(0, I))
  v1 <- c(0, rep(1, I))
  
  b0_0 <- C %*% (CTC_inv %*% v0)
  b0_1 <- C %*% (CTC_inv %*% v1)
  
  # Feasibility quadratic: ||b0_0 + f*b0_1||^2 <= 1
  # A*f^2 + 2*B*f + C0 <= 1
  A <- sum(b0_1^2)
  B <- sum(b0_0 * b0_1)
  C0 <- sum(b0_0^2)
  
  disc <- B^2 - A * (C0 - 1)
  if (disc < 0) stop("No feasible f exists.")
  
  f_min <- (-B - sqrt(disc)) / A
  f_max <- (-B + sqrt(disc)) / A
  
  # Ensure f > 0
  f_min <- max(f_min, 1e-9)
  if (f_min > f_max) stop("No feasible f > 0 exists.")
  
  # Projection of S
  P_S <- S - C %*% (CTC_inv %*% (t(C) %*% S))
  norm_PS <- sqrt(sum(P_S^2))
  
  # Objective function
  obj_fun <- function(f) {
    b0_f <- b0_0 + f * b0_1
    norm_b0_sq <- sum(b0_f^2)
    
    # Handle slight numerical inaccuracies at boundaries
    if (norm_b0_sq > 1) norm_b0_sq <- 1 
    
    N_f <- sum(b0_f * S) - norm_PS * sqrt(1 - norm_b0_sq)
    return(N_f / f)
  }
  
  # Optimize over f
  opt_res <- optimize(obj_fun, interval = c(f_min, f_max))
  f_opt <- opt_res$minimum
  min_obj <- opt_res$objective
  
  # Check boundaries
  obj_min_bound <- obj_fun(f_min)
  obj_max_bound <- obj_fun(f_max)
  
  f_final <- f_opt
  if (obj_min_bound < min_obj) {
    f_final <- f_min
    min_obj <- obj_min_bound
  }
  if (obj_max_bound < min_obj) {
    f_final <- f_max
    min_obj <- obj_max_bound
  }
  
  # Reconstruct optimal b
  b0_final <- b0_0 + f_final * b0_1
  d_final <- - P_S / norm_PS * sqrt(1 - sum(b0_final^2))
  b_final <- b0_final + d_final
  
  list(
    f_optimal = f_final,
    b_optimal = b_final,
    objective = min_obj,
    feasible_range = c(f_min, f_max)
  )
}






##############################################################################################
# 3. PRIMAL OPTIMIZATION
##############################################################################################

# ============================================================
# Solve: max b'gammah  s.t.  b'(E + tau*e) = 0,  b'b = 1
#
# Closed-form solution:
#   v = E + tau*e
#   b* = (gammah - proj_v(gammah)) / ||gammah - proj_v(gammah)||
#   f(tau) = ||gammah - proj_v(gammah)||
#          = sqrt( gammah'gammah - (gammah'v)^2 / (v'v) )
# ============================================================

max_tau_primal_func <- function(gammah, tau) {
  L <- length(gammah)
  e <- rep(1, L)
  E <- 0:(L - 1)
  v <- E + tau * e
  
  # Projection coefficient of gammah onto v
  vv   <- as.numeric(t(v) %*% v)
  gv   <- as.numeric(t(gammah) %*% v)
  proj <- (gv / vv) * v
  
  # Residual (component of gammah orthogonal to v)
  resid <- gammah - proj
  
  # Optimal b (normalized residual) and objective value
  b_star <- resid / sqrt(sum(resid^2))
  f_val  <- sqrt(sum(resid^2))          # = sqrt(gammah'gammah - gv^2/vv)
# This is the same as:  
  f_val<-as.double(b_star%*%gammah)
  target_correlation<-f_val/sqrt(as.double(gammah%*%gammah))
  list(b = b_star, objective = target_correlation, tau = tau)
}


# ------------------------------------------------------------
# Direct closed-form for maximzed objective function f(tau) only (faster, vectorized)
# ------------------------------------------------------------
f_tau <- function(gammah, tau) {
  L <- length(gammah)
  e <- rep(1, L)
  E <- 0:(L - 1)
  
  a <- as.numeric(t(gammah) %*% E)   # gammah'E
  c <- as.numeric(t(gammah) %*% e)   # gammah'e
  S0 <- as.numeric(t(e) %*% e)       # = L
  S2 <- as.numeric(t(E) %*% e)       # sum of E
  S1 <- as.numeric(t(E) %*% E)       # sum of E^2
  gg <- as.numeric(t(gammah) %*% gammah)
  
  Q <- S0 * tau^2 + 2 * S2 * tau + S1
  g <- (a + tau * c)^2 / Q
  
# Normalize by sqrt(as.double(gammah%*%gammah)) to obtain target correlation  
  target_correlation<-sqrt(pmax(gg - g, 0))/sqrt(as.double(gammah%*%gammah))   # guard against tiny negative rounding
  target_correlation
}


# This function is the inverse of f_tau: it computes tau from the target correlation: 
# The efficient frontier is obtained for target correlation values between 1 and the asymptote (as tau to infty).
# For this range of values of the target correlation, the inversion has two roots tau1 and tau2 and max(tau1,tau2) is 
# determines the efficient frontier.
tau_from_f <- function(gammah, target_correlation, tol = 1e-10) {
  f0<-target_correlation*sqrt(as.double(gammah%*%gammah))
  L <- length(gammah)
  e <- rep(1, L)
  E <- 0:(L - 1)
  
  a  <- as.numeric(t(gammah) %*% E)
  c  <- as.numeric(t(gammah) %*% e)
  S0 <- L
  S2 <- L * (L - 1) / 2
  S1 <- (L - 1) * L * (2 * L - 1) / 6
  gg <- as.numeric(t(gammah) %*% gammah)   # ||gammah||^2
  
  # Feasibility check: f0 must be attainable
  if (f0 < 0) stop("f0 must be non-negative (it's a norm).")
  if (f0^2 > gg + tol) {
    warning("f0 exceeds ||gammah||, the global maximum of f(tau). No solution exists.")
    return(list(tau = numeric(0), f0 = f0, feasible = FALSE))
  }
  
  K <- gg - f0^2   # = (a+tau*c)^2 / Q(tau)   -- must be >= 0
  
  # Quadratic coefficients:  A*tau^2 + B*tau + C = 0
  A <- c^2 - K * S0
  B <- 2 * a * c - 2 * K * S2
  C <- a^2 - K * S1
  
  roots <- numeric(0)
  
  if (abs(A) < tol) {
    # Linear (or degenerate) case
    if (abs(B) < tol) {
      if (abs(C) < tol) {
        warning("Equation degenerates to 0=0: f0 matches identically (constant case).")
        return(list(tau = NA, f0 = f0, feasible = TRUE, note = "infinite solutions"))
      } else {
        return(list(tau = numeric(0), f0 = f0, feasible = FALSE))
      }
    }
    roots <- -C / B
  } else {
    disc <- B^2 - 4 * A * C
    if (disc < -tol) {
      # No real roots (shouldn't happen if f0 is feasible, but guard anyway)
      return(list(tau = numeric(0), f0 = f0, feasible = FALSE))
    }
    disc <- max(disc, 0)  # clip tiny negative rounding error
    roots <- c((-B + sqrt(disc)) / (2 * A),
               (-B - sqrt(disc)) / (2 * A))
  }
  
  # Verify each root numerically (guards against spurious algebraic roots
  # introduced by squaring, and removes duplicates / roots where Q(tau)=0)
  Q <- function(t) S0 * t^2 + 2 * S2 * t + S1
  valid <- sapply(roots, function(t) {
    qt <- Q(t)
    if (qt <= tol) return(FALSE)
    fval <- sqrt(max(gg - (a + t * c)^2 / qt, 0))
    abs(fval - f0) < 1e-6 * max(1, abs(f0))
  })
  
  roots <- sort(unique(roots[valid]))
  
  list(tau = roots, f0 = f0, feasible = length(roots) > 0)
}




# ------------------------------------------------------------
# Critical points tau1, tau2: roots of derivative of objective function (from the analytic derivation)
# ------------------------------------------------------------
critical_taus <- function(gammah) {
  L <- length(gammah)
  e <- rep(1, L)
  E <- 0:(L - 1)
  
  a <- as.numeric(t(gammah) %*% E)
  c <- as.numeric(t(gammah) %*% e)
  S0 <- L
  S2 <- L * (L - 1) / 2
  S1 <- (L - 1) * L * (2 * L - 1) / 6
  
  tau2 <- if (abs(c) > 1e-12) -a / c else NA
  denom <- c * S2 - a * S0
  tau1 <- if (abs(denom) > 1e-12) (a * S2 - c * S1) / denom else NA
  
  # Degenerate case: c == 0 -> single critical point at vertex of Q(tau)
  tau_v <- -S2 / S0
  
  list(tau1 = tau1, tau2 = tau2, tau_vertex = tau_v)
}






#################################################################################################################
#################################################################################################################
# Old code


if (F)
{
  # This is older material, ignoring the extension to arbitrary omega0 and/or the phase excess over the MSe predictor
  
  max_tau_func<- function(gamma_h, target_correlation,epsilon,omega0=0) {
    
    if (omega0<0|omega0>pi)
    {
      print("omega0 belongs in the interval [0,pi]")
      return()
    }
    if (epsilon<0)
    {
      print("epsilon must be positive: the sign is changed")
      epsilon<-abs(epsilon)
    }
    
    if (target_correlation<0)
    {
      print("Target correlation must be positive")
      return()
    }
    if (target_correlation>1)
    {
      print("Target correlation must be smaller 1")
      return()
    }
    gamma_h<-as.vector(gamma_h)
    L<-length(gamma_h)
    
    # Rescale target correlation  
    alpha_h<-as.double(target_correlation*sqrt((gamma_h)%*%gamma_h))
    
    # 1. Define base vectors and matrices
    omega_vec <- cos(omega0*(0:(L-1)))
    if (omega0>0)
    {
      k_vec <- sin(omega0*(0:(L-1)))
    } else
    {
      k_vec<-0:(L-1)
    }
    C <- cbind(gamma_h, omega_vec)
    
    # 2. Compute structural matrices (Least Squares and Projections)
    # (C^T C)^-1
    CTC_inv <- solve(t(C) %*% C)
    
    # Base vectors for b_LS(f) = u + f * v
    u <- C %*% (CTC_inv %*% c(alpha_h, 0))
    v <- C %*% (CTC_inv %*% c(0, 1))
    
    # Projection matrix onto the null space of C^T: P_C_perp = I - C (C^T C)^-1 C^T
    I <- diag(L)
    P_C <- C %*% CTC_inv %*% t(C)
    P_C_perp <- I - P_C
    
    # 3. Compute scalars for the quadratic equation
    K_perp <- as.numeric(t(k_vec) %*% P_C_perp %*% k_vec)
    
    b_LS <- u + epsilon * v
    
    
    # Calculate nu
    #  if (as.numeric(t(b_LS) %*% b_LS)>1)
    #  {
    print("target correlation and epsilon do not match: the solution space is empty")
    # Solve equation for epsilon such that system is feasible
    a<-as.double(t(v)%*%v)
    b<-2*as.double(t(u)%*%v)
    c<-as.double(t(u)%*%u)-1
    eps_1<-(-b+sqrt(b^2-4*a*c))/(2*a)
    eps_2<-(-b-sqrt(b^2-4*a*c))/(2*a)
    # Select minimal positive solution
    min_eps<-max(max(0,eps_1),max(0,eps_2))
    print(paste("Minimal epsilon for given target correlation is ",round(min_eps,4),sep=""))
    print(paste("This minimal epsilon is used instead of the original epsilon=",round(epsilon,4),paste=""))
    # Add a small positive margin so that number under square root remains positive
    epsilon1<-min_eps+10^{-12}
    b_LS <- u + epsilon1 * v
    nu<-0
    #  } else
    #  {
    #    epsilon1<-epsilon
    #    nu <- sqrt((1 - as.numeric(t(b_LS) %*% b_LS)) / K_perp)
    #  }
    # Final filter weights
    b_opt <- b_LS - nu * (P_C_perp %*% k_vec)
    # Shift at reference frequency
    if (omega0==0)
    {
      tau_max<-as.double(t(b_opt)%*%(0:(L-1))/sum(b_opt))
    } else
    {
      tau_max<-atan(t(b_opt)%*%sin(omega0*(0:(L-1)))/t(b_opt)%*%cos(omega0*(0:(L-1))))/omega0
    }
    # Return results as a list
    return(list(
      b_opt = as.vector(b_opt),
      f_opt = epsilon1,
      tau_max=tau_max
    ))
  }
  
  
  
  
  max_tau_func <- function(gamma_h, target_correlation,epsilon,omega0=0,phase_excess=F) {
    
    if (omega0<0|omega0>pi)
    {
      print("omega0 belongs in the interval [0,pi]")
      return()
    }
    if (epsilon<0)
    {
      print("epsilon must be positive: the sign is changed")
      epsilon<-abs(epsilon)
    }
    
    if (target_correlation<0)
    {
      print("Target correlation must be positive")
      return()
    }
    if (target_correlation>1)
    {
      print("Target correlation must be smaller 1")
      return()
    }
    gamma_h<-as.vector(gamma_h)
    if (t(gamma_h)%*%gamma_h<10^{-20})
    {
      print("gamma_h must differ from zero")
      return()
    }
    L<-length(gamma_h)
    
    # Rescale target correlation  
    alpha_h<-as.double(target_correlation*sqrt((gamma_h)%*%gamma_h))
    
    # 1. Define base vectors and matrices
    if (omega0>0)
    {
      # Phase excess over benchmark MSE or absolute (i.e., over identity)    
      if (phase_excess)
      {
        # Lead over MSE benchmark
        Gamma_h_omega0=sum(gamma_h*exp(1.i*omega0*(0:(L-1))))
        if (abs(Im(Gamma_h_omega0))>0&abs(Re(Gamma_h_omega0))>0)
        {
          omega_vec <- cos(omega0*(0:(L-1)))*1/Re(Gamma_h_omega0)-sin(omega0*(0:(L-1)))*1/Im(Gamma_h_omega0)
          k_vec<-sin(omega0*(0:(L-1)))*1/Re(Gamma_h_omega0)+cos(omega0*(0:(L-1)))*1/Im(Gamma_h_omega0)
        }
        # If the transfer function of the target is a positive number we use the same k_vec as below (absolute: lead over identity)      
        if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)>0)
        {
          omega_vec <- cos(omega0*(0:(L-1)))
          k_vec<-sin(omega0*(0:(L-1)))
        }
        # If the transfer function of the target is a negative number we shift by pi (invert the sign).       
        if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)<0)
        {
          omega_vec <- -cos(omega0*(0:(L-1)))
          k_vec<--sin(omega0*(0:(L-1)))
        }
        if (Im(Gamma_h_omega0)==0&Re(Gamma_h_omega0)==0)
        {
          print("Transfer function of target vanishes at omega_0: use another omega_0 or set phase_excess==F")
        }
        # If  the transfer function of the target is purely imaginary: shift by pi/2 i.e. use cos (this does not depend on the sign)    
        if (Re(Gamma_h_omega0)==0)
        {
          k_vec<-cos(omega0*(0:(L-1)))
          print("Transfer function of target is purely imaginary at omega0. This case has not been analyzed yet. Change omega0 or set phase_excess==F")
          return()
        }
        
      } else
      {
        # Absolute: lead over identity  
        omega_vec <- cos(omega0*(0:(L-1)))
        k_vec <- sin(omega0*(0:(L-1)))
      }
    } else
    {
      omega_vec <- rep(1,L)
      k_vec<-0:(L-1)
    }
    C <- cbind(gamma_h, omega_vec)
    
    # 2. Compute structural matrices (Least Squares and Projections)
    # (C^T C)^-1
    CTC_inv <- solve(t(C) %*% C)
    
    # Base vectors for b_LS(f) = u + f * v
    u <- C %*% (CTC_inv %*% c(alpha_h, 0))
    v <- C %*% (CTC_inv %*% c(0, 1))
    
    # Projection matrix onto the null space of C^T: P_C_perp = I - C (C^T C)^-1 C^T
    I <- diag(L)
    P_C <- C %*% CTC_inv %*% t(C)
    P_C_perp <- I - P_C
    
    # 3. Compute scalars for the quadratic equation
    K_perp <- as.numeric(t(k_vec) %*% P_C_perp %*% k_vec)
    
    # Want to minimize objective function
    # V(f):=t(k)%*%u/f+t(k)%*%v-\frac{sqrt(K_perp)}{f}*sqrt(1-t(u)%*%u-2*f*t(u)%*%v-f^2t(v)%*%v)
    # This rewriten as 
    # W(x)=t(k)%*%u*x+t(k)%*%v-sqrt(K_perp)*sqrt(1-t(u)%*%u*x^2-2*t(u)%*%v*x-t(v)%*%v)
    # =D*x+t(k)%*%v-sqrt(K_perp)*sqrt(P_1(x)), with P_1(x)=A*x^2+B*x+E
    A<-1-as.double(t(u)%*%u)
    B<-(-2*as.double(t(u)%*%v))
    E<-(-as.double(t(v)%*%v))
    # D is the linear term in W(x) (the constant is irrelevant for minimization)  
    D<-t(k_vec)%*%u
    
    
    # Note that P_1(x) should be implicitely positive because 1-b_LS'b_LS >= 0 (positive by construction due to length constraint). However, P_1(x) is not always positive for all x. We must therefore ensure to select x>=0 (must be positive for time-shift constraint) in the admissible range. For that we first check that B^2-4*A*C>0. If not no zeroes exist and since P_1(0)=E<0 we conclude P_1(x)<0 for all x and hence no solution exists.
    if (B^2-4*A*E<0)
    {
      print("The constraints do not intersect. Select another target_correlation or a larger L")
      return()
    }
    # Distinguish quadratic and linear cases:  
    if (abs(A)>0)
    {
      # P_1 is quadratic    
      # Next we determine the range of admissibility. Note that P_1(0)=E<0. If A>0 then P_1(x) is convex and P_1(x)>0 for all large x. If A<0 then P_1(x) is concave and P_1(x) < 0 for large x. Since B^2-4*A*C > 0 (checked above) zeroes must exist but they could be negative. If all zeroes are negative then no solution exists since then P_1(0)=E<0 implies P_1(x)<0 for all x>=0 (constraints do not have an intersection).
      z1<-(-B+sqrt(B^2-4*A*E))/(2*A)
      z2<-(-B-sqrt(B^2-4*A*E))/(2*A)
      if (z1<0&z2<0)
      {
        print("All zeroes of P_1 are negative: the constraints do not intersect. Select another target_correlation or a larger L")
        return()
      }
      # Since z1<0&z2<0 is excluded by the above check, we have at least one positive root of P_1(x) and since P_1(0)=E<0 we infer that there exist x>0 such that P_1(x)>0. We distinguish two cases:
      # i) A>0: P_1(x) convex. Since P_1(0)<0 we infer that z1=min(z1,z2)<0 and z2=max(z1,z2)>0. Hence for all x>=z2 we must have P_1(x)>= 0 (noting that x >= 0).
      if (A>0)
        admissible_range<-c(max(z1,z2),Inf)
      # ii) A<0: P_1(x) concave. Note that we have already excluded the case of  no roots above. 
      # Since P_1(0)<0 we infer that either z1 and z 2 <0  or z1 and z2 > 0 due to concavity. The former case (both negative) has already been checked above (rejected). 
      # Hence z1=min(z1,z2)>0 and z2=max(z1,z2)>0 and for all z1<=x<=z2 we must have P_1(x)>= 0
      if (A<0)
        admissible_range<-c(min(z1,z2),max(z1,z2))
    } else
    {
      # Quadratic P_1 degenerates to linear function: check that function is not constant:    
      if (abs(B)>0)
      {
        # Not constant function.  Zero:      
        z1<--E/B
        # Determine admissible range of positive values of P_1(x)
        if (B<0)
        {
          # Monotonically decreasing P_1    
          if (z1>0)
          {
            admissible_range<-c(0,z1)
          } else
          {
            print("Degenerate linear case without solution: select another target correlation or a larger L")
            return()
          }
        } else
        {
          # Increasing P_1: all x to the right of z1 are admissible but they must be positive.   
          admissible_range<-c(max(0,z1),Inf)
        }
      } else
      {
        print("P_1 quadratic polynomial degenerates to a constant")
        print("This case has not been analyzed yet")
        return()
      }
      
      
    }
    
    
    # Now that we have determined the feasible region where P_1(x)>=0 so that sqrt(P_1(x)) and thus the objective W(x) are well defined we look for the minimum of W(x) in this region. For this purpose we compute the derivative \dot{W}(x):
    # \dot{W}(x)= D-sqrt(P_perp)*\frac{2*A*x+B}{2*sqrt(P_1(x))}
    # We also compute the second order derivative of the objective W(x), see proof:
    # \dot{\dot{W}}(x)=\frac{B^2-4*A*E}{4*P_1(x)^(3/2)} 
    # This is interesting because in the admissible range B^2-4*A*E > 0 and hence \dot{\dot{W}}(x) > 0 implies that W(x) is convex in the admissible range (it might become non-convexe outside). 
    
    # We now distinguish two cases: I) The first derivative of the objective has a zero in the admissible range and II) no zero in the admissible range. 
    # Case I): Convexity of W(x) implies that the minimal value of the objective is attained at the zero of its derivative. Let's compute the zero: \dot{W}(x)=0 gives after rearranging:
    # D*sqrt(P_1(x))=\sqrt(P_perp)*(A*x+B). Since the left side has sign equal to sign(D), we require sign(A*x+B)=sign(D) at the admissible zero. 
    # Compute the zero(es) of the first derivative: this leads to a new squared function P_2(x) with coefficients:
    a<-as.double(4*A*(D^2-K_perp*A))
    b<-as.double(4*B*(D^2-K_perp*A))
    c<-as.double(4*D^2*E-K_perp*B^2)
    
    if (b^2-4*a*c>0)
    {
      # Compute the roots of the quadratic: to select the correct root we impose sign(2*A*root+B)==sign(D). Otherwise, the result of squaring is not correct.
      eps1<-(-b-sqrt(b^2-4*a*c))/(2*a)
      eps2<-(-b+sqrt(b^2-4*a*c))/(2*a)
      # Check which of the two roots is admissble and effectively a zero of the first order derivative.
      # Note that since W is convex in the admissible region it can only have one zero in its derivative in this region: so the solution must be either z1, or z2 or none.
      if (sign(2*A*eps1+B)==sign(D))
      {
        # It is eps1: in which case we do not have to check for eps2 which cannot be a second different root of the derivative in the admissibility region due to convexity.      
        z<-eps1
      } else
      {
        # If not eps1 it could be eps2      
        if (sign(2*A*eps2+B)==sign(D))
        {
          z<-eps2
        } else
        {
          # Roots exist but not in admissible region. This is the same case as no roots at all.        
          z<-NULL
        }
      }
    } else
    {
      # No roots: this is the same case as no roots in admissible region, above.    
      z<-NULL
    }
    # If a root exists and is admissible, then the minimum is taken there since W is convex in admissible region. In all other cases 
    # the minimum is taken at the boundary of the admissible region. Which boundary depends on the sign of the derivative.
    if (!is.null(z))
    {
      # Case I): the first order derivative vanishes in the admissible range and by convexity this must be the minimum of the objective function
      x_min<-z
    } else
    {
      # Case II): the first order derivative of the objective does not vanish (at all or in the admissible range). If the derivative at any single admissible point is positive then it must be positive in the entire admissible region (but not necessarily everywhere) and the minimum of the objective function is taken at the left boundary of admissibility (otherwise at the right one). Note that we can not use the boundary values of admissible_range to verify the sign of the derivative because the square root vanishes and thus the ratio is not well-defined. But we can use any point strictly within the admissibility region since then the square root will not vanish.
      # We first check that the admissible range is not a single point:    
      if (admissible_range[1]<admissible_range[2])
      {
        if (admissible_range[2]<Inf) 
        {
          point_within_range<-admissible_range[1]+(admissible_range[2]-admissible_range[1])/2
        } else
        {
          # Since the upper bound is infinity we can add any positive number to admissible_range[1]: here 10        
          point_within_range<-admissible_range[1]+10
        }
        # Derivative at point_within_range positive?      
        if (D-sqrt(K_perp)*(2*A*point_within_range+B)/(2*sqrt(A*point_within_range^2+B*point_within_range+E))>0)
        {
          # If yes: minimum of objective is attained at lower bound of admissibility region        
          x_min<-admissible_range[1]
        } else
        {
          # Otherwise at upper bound        
          x_min<-admissible_range[2]
        }
      } else
      {
        # If the admissible range is a single point then we can use either one      
        x_min<-admissible_range[1]
        # This is the same...      
        x_min<-admissible_range[2]
      }
    }  # 
    # Note that x_min cannot be zero since f=1/x_min=infty would conflict with the unit scaling (zero is not an admissible value for x). Therefore we can invert:  
    if (x_min<Inf)
    {  
      f_min<-1/x_min  
    } else
    {
      # A small positive number rather than zero because of numerical precision (this gives a safety margin for computing the square root in the expression for nu below)    
      # Note that this number is small relative to the unit-scaling t(b)%*%b=1, i.e., it is small in absolute terms.    
      f_min<-10^(-12)
    }
    b_LS <- u + f_min * v
    # Compute nu: note that for x_min within the admissibility range, the value under the square root is positive but due to numerical imprecision it might become `minus zero': we therefore rely on the absolute value to avoid errors.    
    nu<-as.double(sqrt(abs((1-t(b_LS)%*%b_LS))/K_perp))
    
    # Final filter weights
    b_opt <- b_LS - nu * (P_C_perp %*% k_vec)
    # Shift at reference frequency
    if (omega0==0)
    {
      tau_max<-as.double(t(b_opt)%*%(0:(L-1))/sum(b_opt))
    } else
    {
      tau_max<-atan(t(b_opt)%*%sin(omega0*(0:(L-1)))/t(b_opt)%*%cos(omega0*(0:(L-1))))/omega0
    }
    # Return results as a list
    return(list(
      b_opt = as.vector(b_opt),
      f_opt = epsilon1,
      tau_max=tau_max
    ))
  }
  
  
  # --- Example Usage ---
  # Smaller L are less effective in generating a lead
  L <- 50
  L_MA<-L
  a1 <- 0.9
  h<-5
  gamma_h <- a1^(h+0:(L-1))
  
  
  
  # Replicate gamma_h
  if (F)
  {
    # Correlation close to one  
    target_correlation <- 0.9999
    # epsilon matches sum(gamma_h): note that we must scale by 1/sqrt(gamma_h%*%gamma_h) because |b|=1 (unit length constraint)  
    epsilon<-as.double(sum(gamma_h)/sqrt(gamma_h%*%gamma_h))
  }
  
  # Replicate MSE
  target_correlation <- 0.999999
  epsilon_mse<-as.numeric(gamma_h%*%cos(omega0*(0:(L-1))))
  epsilon<-epsilon_mse#/2
  
  # Deviate from MSE
  target_correlation <- 0.9
  epsilon_mse<-as.numeric(gamma_h%*%cos(omega0*(0:(L-1))))
  epsilon<-epsilon_mse#/2
  
  # Reference frequency
  omega0<-pi/12
  # Run the optimization
  result <- tryCatch({
    max_tau_func(gamma_h, target_correlation,epsilon,omega0)
  }, error = function(e) {
    cat("Error:", e$message, "\n")
    NULL
  })
  
  epsilon_min<-result$f_opt
  
  
  cat("\n--- Optimization Results ---\n")
  cat(sprintf("Optimal f (f*): %f\n", result$f_opt))
  cat(sprintf("Minimum Objective Value: %f\n\n", result$objective_value))
  
  cat("Optimal filter weights (b*):\n")
  print(round(result$b_opt, 4))
  
  # Verify constraints
  cat("\n--- Constraint Verification ---\n")
  b <- result$b_opt
  f <- result$f_opt
  k_vec <- 0:(L-1)
  
  cat(sprintf("1) target correlation constraint = %f (Target: %f)\n", sum(b * gamma_h)/sqrt(gamma_h%*%gamma_h), target_correlation))
  cat(sprintf("2) time-shift constraint         = %f (Target: %f)\n", sum(b*cos(omega0*(0:(L-1)))), f))
  cat(sprintf("3) unit length constraint         = %f (Target: 1.000000)\n", sum(b^2)))
  if (omega0>0)
  {
    cat(sprintf("Theoretical objective atan/omega = %f\n", atan(t(b)%*%sin(omega0*(0:(L-1)))/epsilon)/omega0))
    cat(sprintf("Theoretical objective atan/omega = %f\n", atan(t(b)%*%sin(omega0*(0:(L-1)))/epsilon_min)/omega0))
    
  } else
  {
    cat(sprintf("Theoretical objective (sum(k*b)/epsilon) = %f\n", sum(k_vec * b) / epsilon))
    cat(sprintf("Effective objective (sum(k*b)/epsilon_min) = %f\n", sum(k_vec * b) / epsilon_min))
  }
  
  ts.plot(b,main="Max-Tau predictor")
  
  amp_obj<-amp_shift_func(600,b,T)
  
  # ----------------
  # Plot and compare predictors: max-tau is left shifted without being unsmoother
  
  set.seed(544)
  len<-150
  len<-600
  x<-rnorm(len)
  ma_vec<-gamma_h
  x<-arima.sim(n = len, list( ma = ma_vec))
  ymse<-filter(x,gamma_h,sides=1)
  ymaxtau<-filter(x,b,sides=1)
  
  mploth<-na.exclude(cbind(ymse,ymaxtau))
  
  anf<-80
  enf<-min(500,nrow(mploth))
  
  mplot<-scale(mploth)[anf:enf,]
  par(mfrow=c(1,1))
  ts.plot(mplot,col=c("black","blue"))
  abline(h=0)
  
  # max-tau is as smooth as MSE: virtually identical curvature for identical unit scaling
  apply(apply(apply(scale(mploth),2,diff),2,diff)^2,2,mean)
  
  
  
  #-----------------------------------
  # Compute lead
  # 1. Shift
  K<-600
  # The shift at frequency zero matches the objective
  trffkt_obj<-amp_shift_func(K,b,T)
  shift<-trffkt_obj$shift
  
  shift[shift<(-1)]<-0
  par(mfrow=c(1,1))
  ts.plot(shift,main=c("Time-Shift Max-Tau Predictor",paste("Theoretical epsilon=",round(epsilon,3),", Effectively feasible minimal epsilon=",round(epsilon_min,3),sep=""),paste("Minium shift: ",round(trffkt_obj$shift[1],3),", matches the feasible Tau")))
  
  
  #------------
  # 2. Lead at zero-crossings:
  filter_mat<-scale(mploth)[,2:1]
  filter_mat<-mplot[,2:1]
  max_lead=6;vicinity=4;last_crossing_or_closest_crossing=F;outlier_limit=10
  
  compute_min_tau_func(filter_mat,max_lead,vicinity,last_crossing_or_closest_crossing,outlier_limit)
  
  #-------------
  # 3. CCF
  # CCF: peak stays at zero (AR(1) process: peak cannot be right-shifted).
  # Max Tau induces a right-skew of the CCF.
  new_ccf_comp_func<-function(gamma1,gamma_ref)
  {
    if (length(gamma1)<length(gamma_ref))
    {
      print("length(gamma1)<length(gamma_ref): gamma1 is zero-padded")
      gamma1<-c(gamma1,rep(0,length(gamma_ref)-length(gamma1)))
    }
    if (length(gamma_ref)<length(gamma1))
    {
      print("length(gamma1_ref<length(gamma1): gamma_ref is zero-padded")
      gamma_ref<-c(gamma_ref,rep(0,length(gamma1)-length(gamma_ref)))
    }
    cor_vec_lead_LA<-cor_vec_lag_LA<-i_lead<-i_lag<-NULL
    # Leads: 0 up to h
    for (i in 0:(L-1))#i<-1
    {
      cor_vec_lead_LA<-c(cor_vec_lead_LA,gamma1[1:(min(L+i,L)-i)]%*%gamma_ref[(i+1):min(L+i,L)]/(sqrt(gamma1%*%gamma1)*sqrt(gamma_ref%*%gamma_ref)))
      i_lead<-c(i_lead,i)
    }
    # Lags 
    for (i in 1:(L-1))#i<-1
    {
      cor_vec_lag_LA<-c(cor_vec_lag_LA,gamma_ref[1:(min(L+i,L)-i)]%*%gamma1[(i+1):min(L+i,L)]/(sqrt(gamma1%*%gamma1)*sqrt(gamma_ref%*%gamma_ref)))
      i_lag<-c(i_lag,-i)
    }
    cor_vec<-c(cor_vec_lag_LA[length(cor_vec_lag_LA):1],cor_vec_lead_LA)
    names(cor_vec)<-c(i_lag[length(i_lag):1],i_lead)  
    return(cor_vec)
  }
  
  
  
  ccf<-new_ccf_comp_func(b,gamma_h)
  plot(ccf,main="True/Expected CCF: Peaks at lag = 0", type = "l", axes = FALSE,
       xlab = "Lag", ylab = "CCF")
  axis(1, at = 1:length(ccf),
       labels = names(ccf))
  axis(2)
  abline(v=which(ccf==max(ccf)),lty=2)
  box()
  
  # Decay is exponential with parameter a1 for leads (positive: right tail).
  #   The deviations from exact a1^k decay are due to finite MA representation (increasing L leads to tighter approximation)
  # The left tail depends on b and is no more exponential.
  ccf[2:length(ccf)]/ccf[1:(length(ccf)-1)]
  
  
  
  
  
}

