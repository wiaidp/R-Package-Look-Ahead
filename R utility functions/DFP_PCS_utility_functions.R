

# Convolution of two functions
conv_two_filt_func<-function(filt1,filt2)
{
  L<-max(length(filt1),length(filt2))
  if (length(filt1)<L)
    filt1<-c(filt1,rep(0,L-length(filt1)))
  if (length(filt2)<L)
    filt2<-c(filt2,rep(0,L-length(filt2)))
  conv<-filt1
  L<-length(filt1)
  for (i in 1:L)
  {
    conv[i]<-sum(filt1[1:i]*filt2[i:1])
  }  
  return(list(conv=conv))
}




# Multivariate convolution for M-SSA: 
# Care: ordering of filters is relevant since matrix multiplication is generally not commutative
#   -Ordering is irrelevant if one of the sequence is diagonal (so that matrix multiplication is commutative)
# It is assumed that filters are transposed i.e. nrow=n and ncol=n*L
M_conv_two_filt_func<-function(filt1,filt2)
{
  # filt1<-t(bk_mat)  filt2<-xi
  n<-dim(filt1)[1]
  if (n!=dim(filt2)[1])
  {
    print("Filter dimension n differ")
    return()
  }
  if (ncol(filt1)!=ncol(filt2))
  {
    print("Filter lengths differ")
    return()
  }
  # Filter length (for each series, i=1,...,n)  
  L<-ncol(filt1)/n
  # Initialize convolution as a corresponding matrix with zeroes  
  conv<-0*filt1
  for (i in 1:L)#i<-2
  {
    for (j in 1:i)#j<-2
    {
      conv[,i+(0:(n-1))*L]<-conv[,i+(0:(n-1))*L]+filt1[,j+(0:(n-1))*L]%*%filt2[,i+1-j+(0:(n-1))*L]
      #        filt1[1:i]*filt2[i:1]
    }
  }  
  return(list(conv=conv))
}







# Convolution with summation filter (unit-root assumption)
conv_with_unitroot_func<-function(filt)
{
  conv<-filt
  L<-length(filt)
  for (i in 1:L)
  {
    conv[i]<-sum(filt[1:i])
  }  
  return(list(conv=conv))
}

# Deconvolute filt2 from filt1: filt1 is the convolution
# See section 2 of JBCY paper
deconvolute_func<-function(filt1,filt2)
{
  filt1<-as.vector(filt1)
  filt2<-as.vector(filt2)
  if (length(filt1)<length(filt2))
    filt1<-c(filt1,rep(0,length(filt2)-length(filt1)))
  if (length(filt2)<length(filt2))
    filt2<-c(filt2,rep(0,length(filt1)-length(filt2)))
  
  # filt1<-as.vector(gammak_mse)    filt2<-as.vector(hp_mse)
  L<-length(filt1)
  dec_filt<-filt1
  dec_filt[1]<-filt1[1]/filt2[1]
  for (i in 2:L)
  {
    dec_filt[i]<-(filt1[i]-sum(dec_filt[1:(i-1)]*filt2[i:2]))/filt2[1]
  }  
  return(list(dec_filt=dec_filt))
}


# Multivariate deconvolution for M-SSA: ordering is relevant
#   Deconvolute filt2 from filt1: filt1 is the convolution
M_deconvolute_func<-function(filt1,filt2)
{
  # filt1<-t(bk_best)  gamma_target  filt1<-xi  filt2<-gamma_mse
  n<-dim(filt1)[1]
  if (n!=dim(filt2)[1])
  {
    print("Filter dimension n differ")
    return()
  }
  if (ncol(filt1)!=ncol(filt2))
  {
    print("Filter lengths differ")
    return()
  }
  # Filter length (for each series, i=1,...,n)  
  L<-ncol(filt1)/n
  # Initialize convolution as a corresponding matrix with zeroes  
  deconv<-0*filt1
  # Compute first element of deconvolution: ig filt2 ist MA-inversion xi then f2inv is just the identity 
  f2inv<-solve(filt2[,1+(0:(n-1))*L])
  deconv[,1+(0:(n-1))*L]<-filt1[,1+(0:(n-1))*L]%*%f2inv
  for (i in 2:L)#i<-2
  {
    for (j in 1:(i-1))#j<-2
    {
      deconv[,i+(0:(n-1))*L]<-deconv[,i+(0:(n-1))*L]+deconv[,j+(0:(n-1))*L]%*%filt2[,i+1-j+(0:(n-1))*L]
    }
    deconv[,i+(0:(n-1))*L]<-(filt1[,i+(0:(n-1))*L]-deconv[,i+(0:(n-1))*L])%*%f2inv 
  }  
  return(list(deconv=deconv))
}




# Compute polynomial coefficients from roots: used when computing minimum phase DFP
poly_from_roots <- function(roots, lead = 1, make_real = TRUE, tol = 1e-12) {
  roots <- as.vector(roots)
  n <- length(roots)
  # Start with P(x) = 1
  coeffs <- 1
  if (n > 0) {
    for (r in roots) {
      # Multiply current polynomial by (x - r):
      # new(x) = x*P(x) - r*P(x)
      coeffs <- c(0, coeffs) - r * c(coeffs, 0)
    }
  }
  # Scale leading coefficient
  coeffs <- Re(lead) * coeffs
  # Optionally coerce to real if coefficients are (numerically) real
  if (make_real) {
    if (max(abs(Im(coeffs))) < tol) coeffs <- Re(coeffs)
  }
  coeffs
}


compute_acf_at_lags_zero_delta_func<-function(max_lag,h,b,gamma)
{
  L<-length(b)
  cor_vec_lead<-cor_vec_lag<-NULL
# Leads:
  L_gamma<-length(gamma)
  for (i in 0:(L-1))#i<-2
    cor_vec_lead<-c(cor_vec_lead,b[1:min(L,L_gamma-i)]%*%gamma[i+(1:min(L,L_gamma-i))]/(sqrt(b%*%(b))*sqrt(gamma%*%gamma)))
# Lags
  if (max_lag>0)
  {
    for (i in 1:(min(L-1,max_lag)))#i<-1
      cor_vec_lag<-c(cor_vec_lag,b[(i+1):L]%*%gamma[1:((L)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
  }
  if (F)
  {
    # Leads: 0 up to h
    for (i in 0:h)#i<-2
      cor_vec_lead<-c(cor_vec_lead,b[1:(min((L-h)+i,L)-i)]%*%gamma[(i+1):min((L-h)+i,L)]/(sqrt(b%*%(b))*sqrt(gamma%*%gamma)))
    # Leads: h+1,...,L (after L the best forecast is zero)
    for (i in 1:((L-h)-1))#i<-1
      cor_vec_lead<-c(cor_vec_lead,b[1:((L-h)-i)]%*%gamma[(h+i)+1:((L-h)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
    # Lags
    for (i in 1:(max_lag-1))#i<-1
      cor_vec_lag<-c(cor_vec_lag,b[(i+1):(L-h)]%*%gamma[1:((L-h)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
  }
  cor_vec<-c(cor_vec_lag[length(cor_vec_lag):1],cor_vec_lead)
  return(list(cor_vec=cor_vec))
}


compute_acf_at_lags_zero_delta_func_old<-function(max_lag,h,b,gamma)
{
  L<-length(gamma)
  cor_vec_lead<-cor_vec_lag<-NULL
  # Leads:
  for (i in 0:(L-1))#i<-2
    cor_vec_lead<-c(cor_vec_lead,b[1:(L-i)]%*%gamma[(i+1):L]/(sqrt(b%*%(b))*sqrt(gamma%*%gamma)))
  # Lags
  if (max_lag>0)
  {
    for (i in 1:(min(L-1,max_lag)))#i<-1
      cor_vec_lag<-c(cor_vec_lag,b[(i+1):L]%*%gamma[1:((L)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
  }
  if (F)
  {
    # Leads: 0 up to h
    for (i in 0:h)#i<-2
      cor_vec_lead<-c(cor_vec_lead,b[1:(min((L-h)+i,L)-i)]%*%gamma[(i+1):min((L-h)+i,L)]/(sqrt(b%*%(b))*sqrt(gamma%*%gamma)))
    # Leads: h+1,...,L (after L the best forecast is zero)
    for (i in 1:((L-h)-1))#i<-1
      cor_vec_lead<-c(cor_vec_lead,b[1:((L-h)-i)]%*%gamma[(h+i)+1:((L-h)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
    # Lags
    for (i in 1:(max_lag-1))#i<-1
      cor_vec_lag<-c(cor_vec_lag,b[(i+1):(L-h)]%*%gamma[1:((L-h)-i)]/(sqrt(b%*%b)*sqrt(gamma%*%gamma)))
  }
  cor_vec<-c(cor_vec_lag[length(cor_vec_lag):1],cor_vec_lead)
  return(list(cor_vec=cor_vec))
}



# Compute CCF between predictor gamma and data generating process gamma1
compute_ccf_func<-function(gamma1,gamma_ref)
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



# Computes amplitude and time shifts (mainly for illustration purposes)
amp_shift_func<-function(K,b,plot_T)
{
  #  if (sum(b)<0)
  #  {
  #    print("Sign of coefficients has been changed")
  #    b<-b*sign(sum(b))
  #  }
  omega_k<-(0:K)*pi/K
  trffkt<-0:K
  for (i in 0:K)
  {
    trffkt[i+1]<-b%*%exp(1.i*omega_k[i+1]*(0:(length(b)-1)))
  }
  amp<-abs(trffkt)
  shift<-Arg(trffkt)/omega_k
  shift[1]<-sum((0:(length(b)-1))*b)/sum(b)
  if (plot_T)
  {
    par(mfrow=c(2,1))
    plot(amp,type="l",axes=F,xlab="Frequency",ylab="Amplitude",main="Amplitude")
    axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
    axis(2)
    box()
    plot(shift,type="l",axes=F,xlab="Frequency",ylab="Shift",main="Shift",ylim=c(min(min(shift,na.rm=T),0),max(shift,na.rm=T)))
    axis(1,at=1+0:6*K/6,labels=c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi"))
    axis(2)
    box()
  }
  return(list(trffkt=trffkt,amp=amp,shift=shift))
}

# Filter function: applies a filter b to a series x which can be xts or double
#   If x is xts then time ordering of b is reversed
filt_func<-function(x,b)
{
  L<-length(b)
  yhat<-x
  if (is.matrix(x))
  {
    length_time_series<-nrow(x)
  } else
  {
    if (is.vector(x))
    {
      length_time_series<-length(x)
    } else
    {
      print("Error: x is neither a matrix nor a vector!!!!")
    }
  }
  for (i in L:length_time_series)
  {
    # If x is an xts object then we cannot reorder x in desceding time i.e. x[i:(i-L+1)] is the same as  x[(i-L+1):i]
    #   Therefore, in this case, we have to revert the ordering of the b coefficients.
    if (is.xts(x))
    {
      yhat[i]<-as.double(b[L:1]%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
    } else
    {
      yhat[i]<-as.double(b%*%x[i:(i-L+1)])#tail(x) x[(i-L+1):i]
    }
  }
  #  names(yhat)<-index(x)#index(yhat)  index(x)
  #  yhat<-as.xts(yhat,tz="GMT")
  return(list(yhat=yhat))
}




# Relies on older code (which could deviate from unit-sphere) to infer MSE and new code (optimization closed-form
# conditional on a single Lagrangian for the unit-sphere constraint which is computed numerically) to infer new estimate
mse_fast_optim_func<-function(gamma,h,L,sum_par_pos,shrink,sup_vec,modify_target=T)
{
  l<-length(gamma)
  ts.plot(gamma)
  # Forecast horizon
  delta<-h
  #delta<-70
  # Support vector: at which leads the cross-correlations are (collectively) optimized
  sup_vec_mse<-delta
  # Split/break between positive and negative cross-correlations
  threshold_plus_minus_mse<-min(delta-min(sup_vec_mse),2)
  # Sum of parameters positive
  sum_par_pos<-T
  # Skrink the crosscorrelation further (inequality is cross-cor>=acf/shrink)
  shrink<-1

  opt_obj<-fast_optim_func(sup_vec_mse,delta,threshold_plus_minus_mse,gamma,L,sum_par_pos,shrink,modify_target)

  b_mse<-opt_obj$b
  ts.plot(b_mse)
  # Check unity sum
  sum(b_mse^2)

  ts.plot(scale(cbind(gamma,b_mse)))



  # Compute crosscorrelations and acf-target
  anf<-0
  enf<-delta-1

  cross_cor_obj<-cross_cor_func(anf,enf,b_mse,gamma)

  # The point of crossing is a natural estimate for threshold_plus_minus
  ts.plot(cbind(cross_cor_obj$target_cor,cross_cor_obj$cross_cor),col=c("black","blue"))


  if (length(which(cross_cor_obj$target_cor<cross_cor_obj$cross_cor))!=0)
  {
    # I a crossing exists
    threshold_plus_minus_nat<-which(cross_cor_obj$target_cor<cross_cor_obj$cross_cor)[1]-1
  } else
  {
    # No crossing
    threshold_plus_minus_nat<-L
  }
  #------------------------------------------------------------
  #-------------------------------------------------------------------------------------
  if (F)
  {
    sup_vec<-c(delta,delta-1)
    sup_vec<-c(delta:(delta-5),0)
    sup_vec<-(delta):0
  }
  #sup_vec<-(delta):(-30)
  # Split/break between positive and negative cross-correlations
  threshold_plus_minus<-0
  threshold_plus_minus<-threshold_plus_minus_nat
  #threshold_plus_minus<-min(delta-min(sup_vec),40)

  # Simpler optimization: quadratic length-constraint is an inequality (to ensure convexity)
  #   This solution often works because maximum of linear objective is obtained at the boundary of the unit-ball i.e. at the unit-sphere
  #   But in certain cases (for example AR(1) target) the solution is inside the unit-ball i.e. the optimization does not work (because simplified expressions are nomore correlations if length is smaller one)
  opt_obj<-fast_optim_func(sup_vec,delta,threshold_plus_minus,gamma,L,sum_par_pos,shrink,modify_target)

  b<-opt_obj$b
  a<-opt_obj$a
  a[2:L]/a[1:(L-1)]
  b[2:L]/b[1:(L-1)]

  # Compare outputs and peak-correlations with target
  # Length of simulated series to which MSE and new filter are applied
  len<-100000

  check_perf_func(len,L,gamma,b_mse,b,delta)

  b_fast_optim<-b

  return(list(b_mse=b_mse,b_fast_optim=b_fast_optim,threshold_plus_minus=threshold_plus_minus))

}


ar_to_ma_func <- function(phi, theta,n_lags) {
  p <- length(phi)
  psi <- numeric(n_lags)
  
  # Psi_0 is always 1 by definition, but usually, we output psi_1 to psi_k
  # Let's handle indices carefully. 
  # R's ARMAtoMA function is convenient for this.
  
  # Note: arima() returns phi such that X_t = phi_1*X_{t-1} + ... + e_t
  # ARMAtoMA expects the same convention.
  psi_weights <- ARMAtoMA(ar = phi, ma = theta, lag.max = n_lags)
  return(psi_weights)
}


# This function computes the variances of the MA-inverted weights (not the full covariance matrix)

ma_inv_pred_interval_func<-function(data,p_fit,q_fit,k_lags)
{
  fit <- arima(data, order = c(p_fit, 0, q_fit), include.mean = FALSE)
  
  # Extract estimated coefficients (phi) and their covariance matrix (Sigma_phi)
  if (p_fit>0)
  {
    phi_hat <- fit$coef[1:p_fit]
  } else
  {
    phi_hat<-NULL
  }
  if (q_fit>0)
  {
    theta_hat <- fit$coef[p_fit+1:q_fit]
  } else
  {
    theta_hat<-NULL
  }
  sigma <- fit$var.coef
  
  # Define the number of MA weights (lags) you want to compute
  k_lags <- k_lags
  # 3. Define a function that converts AR params to MA weights
  # The relationship is: Phi(B) * Psi(B) = 1
  # This can be solved recursively: psi_j = sum(phi_k * psi_{j-k})
  
  # 4. Compute the Estimates and Variances (Delta Method)
  
  # A. Point Estimates of MA weights
  psi_hat <- ar_to_ma_func(phi_hat,theta_hat, k_lags)
  
  # B. Compute the Jacobian of the transformation at phi_hat
  # We need d(psi)/d(phi). Result is a (k_lags x p) matrix.
  jacobian_mat <- jacobian(func = ar_to_ma_func, x = phi_hat,theta=theta_hat, n_lags = k_lags)
  
  # C. Compute Covariance Matrix of MA weights: Cov(Psi) = J * Cov(Phi) * J'
  cov_psi <- jacobian_mat %*% sigma %*% t(jacobian_mat)
  
  # D. Extract Standard Errors (sqrt of diagonal elements)
  se_psi <- sqrt(diag(cov_psi))
  
  return(list(phi_hat=phi_hat,theta_hat=theta_hat,psi_hat=psi_hat,se_psi=se_psi,sigma=sigma))
}




# This function returns the full covariance matrix of the MA inverted weights
compute_ma_distribution <- function(data, p, lag.max = 10) {
  
  # 1. Fit the AR(p) model
  # We use ar.mle (Maximum Likelihood) or ar.ols. 
  # arima() with order=c(p,0,0) is generally robust for standard errors.
  fit <- arima(data, order = c(p, 0, 0), include.mean = FALSE)
  
  # Extract coefficients (phi) and their covariance matrix (Sigma_phi)
  phi_hat <- fit$coef[1:p]
  sigma_phi <- fit$var.coef[1:p, 1:p]
  
  # 2. Compute the MA weights (psi) from AR coefficients
  # The relationship is defined by the filter inverse.
  # psi(B) = 1 / phi(B)
  # We can use ARMAtoMA to get the point estimates
  psi_hat <- ARMAtoMA(ar = phi_hat, ma = 0, lag.max = lag.max)
  
  # 3. Apply the Delta Method to find the Covariance Matrix of Psi
  # We need the Jacobian matrix J, where J[i, j] = d(psi_i) / d(phi_j)
  #
  # The recursive relationship is:
  # psi_0 = 1
  # psi_j = sum_{k=1}^{min(j, p)} phi_k * psi_{j-k}  for j > 0
  
  # Initialize Jacobian matrix (rows = lag.max, cols = p)
  J <- matrix(0, nrow = lag.max, ncol = p)
  
  # To compute derivatives numerically or analytically:
  # Analytically is cleaner. 
  # d(psi_j)/d(phi_k) depends on previous psi values.
  
  # Let's perform the derivative calculation iteratively.
  # We denote psi vector including psi_0 = 1 for easier indexing
  psi_full <- c(1, psi_hat) 
  
  for (j in 1:lag.max) { # For each MA weight psi_j
    for (k in 1:p) {     # With respect to each AR coeff phi_k
      
      # Derivative Logic:
      # psi_j = phi_1*psi_{j-1} + ... + phi_k*psi_{j-k} + ... + phi_p*psi_{j-p}
      # The derivative has two parts:
      # 1. Direct dependence: if the term phi_k * psi_{j-k} exists, derivative is psi_{j-k}
      # 2. Indirect dependence: phi_1 * d(psi_{j-1})/d(phi_k) + ...
      
      term1 <- 0
      if ((j - k) >= 0) {
        term1 <- psi_full[(j - k) + 1] # +1 because R is 1-indexed
      }
      
      term2 <- 0
      # Sum over m=1 to p of (phi_m * d(psi_{j-m})/d(phi_k))
      for (m in 1:p) {
        if ((j - m) > 0) {
          # We look up the derivative calculated in previous row of J
          term2 <- term2 + phi_hat[m] * J[j - m, k]
        }
      }
      
      J[j, k] <- term1 + term2
    }
  }
  
  if (F)
  {
# This is the same up to numerical error (jacobian is a numerical approximation of exact derivative above)
    J <- jacobian(func = ar_to_ma_func, x = phi_hat,theta=0, n_lags = lag.max)
  }

  # 4. Compute Sigma_psi = J * Sigma_phi * J'
  sigma_psi <- J %*% sigma_phi %*% t(J)
  
  return(list(
    ar_coeffs = phi_hat,
    ar_cov_matrix = sigma_phi,
    ma_weights = psi_hat,
    ma_cov_matrix = sigma_psi
  ))
}






#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# Exact equality constraint based on linearization of optimization problem
# New optimization: solves MSE-LA for given arbitrarily many equality constraints (no inequalities)
# Closed-form solution in paper timeliness_paper.rnw is not used. Instead we rely on closed-form for
#  linear equalities conditionally on a single Lagrangian for unit-sphere, which is computed numerically.
# The Lagrange-parameter for sphere is obtained by ordinary numerical optimization (not closed-form)

# threshold_plus_minus sets the lead at which the cross correlation of the forecast and the DGP drops below (or exceeds)  the shifted acf of the DGP (shifted and centered at forecast horizon h)
# Background:
#   -We minimize the absolute difference of both cross correlations
#   -Absolute difference is sign(difference)*difference
#   -We now linearize the absolute difference by supplying signs: positive sign for leads>threshold_plus_minus and negative signs for leads<threshold_plus_minus
# The (correct) signs are then imposed by constraints: either inequality constraints or equalities

new_optimization_given_constraints<-function(gamma,h,L,shrink,sup_vec,set_val,val_vec,threshold_plus_minus,modify_target=T)
{
  if (max(sup_vec)>delta)
  {
    print("delta<max(delta_sup): this particular extension has not been implemented/verified yet")
    return()
  }
  if (!delta%in%sup_vec)
  {
    print("delta must be in sup_vec")
    return()
  }
  if (delta!=sup_vec[1])
  {
    print("delta must be first element of sup_vec")
    return()
  }
  delta<-h
  ls<-length(sup_vec)
  l<-length(gamma)
  # Assign weights to each lead/lag
  weight_vec<-rep(1,ls)
  # Setting-up weights of Objective function (of target vector)
  # 1. At lag delta (always positive sign i.e. maximization)
  a<-weight_vec[1]*c(gamma[delta+1:min(L,l-delta)],rep(0,L-min(L,l-delta)))/sqrt(sum(gamma^2))
# Here we can modify target in objective function: targeting more leads than the single delta-step ahead performance
  if (modify_target)
  {
#    if (threshold_plus_minus>0)
#    {
      for (i in 2:(ls))#i<-ls
      {
        # Meaning of sup_vec[i]
        # sup_vec[i]=delta: lead delta; sup_vec[i]=k>=0: lead k; sup_vec[i]=-k<0: lag k
        if (delta-sup_vec[i]<threshold_plus_minus)
        {
          # Maximization of cross-correlation
          # We use abs(sup_vec[i]) because if sup_vec[i]<0 then we use the mirrored (left tail)
          # i.e. correlation of forecast with lagged (instead of leading) target
          a<-a+weight_vec[i]*c(gamma[abs(sup_vec[i])+1:min(L,l-abs(sup_vec[i]))],rep(0,L-min(L,l-abs(sup_vec[i]))))/sqrt(sum(gamma^2))
        } else
        {
          # Minimization of cross-correlation
          a<-a-weight_vec[i]*c(gamma[abs(sup_vec[i])+1:min(L,l-abs(sup_vec[i]))],rep(0,L-min(L,l-abs(sup_vec[i]))))/sqrt(sum(gamma^2))
        }
      }
#    }
  }
  ts.plot(a)


  val<-rep(NA,ls)
# Matrix of linear constraints
  A<-matrix(rep(NA,L*ls),nrow=ls,ncol=L)
# Optimization requires column and row names (because constraints could be applied to subset of coefficients)
# Forecast horizon delta
# 1. Upper value of constraint: maximal value of correlation (i.e. 1)
  val[1]<-1
# 2. Linear constraint
#   -The forecast depends on eps_t,eps_{t-1},...;
#   -The future data point at horizon delta depends on eps_{t+delta},eps_{t+delta-1},....
#   -Therefore the correlation of predictor b is with gamma_{delta}, gamma_{delta+1},...
  A[1,]<-c(gamma[delta+1:min(L,l-delta)],rep(0,L-min(L,l-delta)))/sqrt(sum(gamma^2))
# This is the same as (because delta is always first element of sup_vec: this condition is checked above)
  A[1,]<-c(gamma[sup_vec[1]+1:min(L,l-delta)],rep(0,L-min(L,l-delta)))/sqrt(sum(gamma^2))
# Initialize all smaller-equal constraints (maximization)
  if (ls>1)#(threshold_plus_minus>0)
  {
# Meaning of sup_vec[i]
# sup_vec[i]=delta: lead delta; sup_vec[i]=k>=0: lead k; sup_vec[i]=-k<0: lag k
    for (i in 2:ls)#i<-ls
    {
# Distinguish maximization from minimization
      if (delta-sup_vec[i]<threshold_plus_minus)
      {
# Bound linear constraint: autocorrelation at lag
# Constraint: val[i]-A[i,]%*%b>=0 i.e. A[i,]%*%b<=val[i]
        val[i]<-(gamma[(delta+1-sup_vec[i]):l]%*%gamma[1:(l-(delta-sup_vec[i]))]/sum(gamma^2))
# Maximization of cross-correlation
# We use abs(sup_vec[i]) because if sup_vec[i]<0 then we use the mirrored (left tail)
# i.e. correlation of forecast with lagged (instead of leading) target
        A[i,]<--c(gamma[abs(sup_vec[i])+1:min(L,l-abs(sup_vec[i]))],rep(0,L-min(L,l-abs(sup_vec[i]))))/sqrt(sum(gamma^2))
      } else
      {
        val[i]<--(gamma[(delta+1-sup_vec[i]):l]%*%gamma[1:(l-(delta-sup_vec[i]))]/sum(gamma^2))
# One can shrink further the conditions where the cross-correlation is minimized
        val[i]<-val[i]/shrink
        A[i,]<-c(gamma[abs(sup_vec[i])+1:min(L,l-abs(sup_vec[i]))],rep(0,L-min(L,l-abs(sup_vec[i]))))/sqrt(sum(gamma^2))
      }
    }
  }
# Constraints could be set to prespecified values
  if (set_val)
  {
    val[1]<-val_vec[1]
    if (ls>1)#(threshold_plus_minus>0)
    {
      for (i in 2:ls)#i<-ls
      {
# Distinguish maximization from minimization
        if (delta-sup_vec[i]<threshold_plus_minus)
        {
          val[i]<-val_vec[i]
        } else
        {
          val[i]<--val_vec[i]
        }
      }
    }
  }
  dim(A)
  lambda_start<-3

  sum_bsquared<-function(lambda_start,A,j0,L,val,a)
  {
# lambda_start<-1.e-99
# Avoid singular designs: keep size of lambda bounded
    lambda_start<-sign(lambda_start)*min(abs(lambda_start),1.e+6)
    lambda_start<-sign(lambda_start)*max(abs(lambda_start),1.e-6)
    if (length(j0)>1)
    {
      M<-rbind(cbind(diag(-2*lambda_start,L),t(A[j0,])),cbind(A[j0,],diag(rep(0,length(j0)))))
    } else
    {
      M<-rbind(cbind(diag(-2*lambda_start,L),A[j0,]),c(A[j0,],0))
    }
    a_vec<--c(a,val[j0])
    x_vec<-solve(M)%*%a_vec
    b<-x_vec[1:L]
   # Checks
   # 1. Correct lambda_start should lead to 1 (unit-sphere)
    return(sum(b^2))
  }

  crit_func<-function(lambda_start,A,j0,L,val,a)
  {

    return(abs(1-sum_bsquared(lambda_start,A,j0,L,val,a)))
  }
  crit_func<-function(lambda_start)
  {

    return(abs(1-sum_bsquared(lambda_start,A,j0,L,val,a)))
  }


#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Step 1: run through all Single constraints and check which one is not feasible
#   Non feasible: if acf-value is too high at a given lead (for example acf=1 at lead delta is impossible to achieve)
#   Therefore we could in principle omit 1 and rely on j0_mat<-matrix(2:ls,nrow=ls-1)
  j0_mat<-matrix(1:ls,nrow=ls)
  const_mat<-NULL
# Introduce a slack variable for resolving limit cases of constraint violations
  slack<-1.e-5
  b_opt<-crit_opt<-const_mat_opt<-NULL
  length_check_vec<-NULL
  lambda_start<-1


#-------------------------------
# Step 2: impose all admissible constraints
#j0_mat<-matrix(c(1,5,8,9,3,10),nrow=1)
#j0_mat<-matrix(c(5,8,9,3,10),nrow=1)

  const_mat<-NULL
# Introduce a slack variable for resolving limit cases of constraint violations
  slack<-1.e-5
  b_opt<-crit_opt<-const_mat_opt<-NULL
# Loop over all j0 and select solution which fulfills all constraints AND which maximizes criterion (minimizes negative criterion)
  for (i in 1:nrow(j0_mat))#i<-1
  {
    j0<-j0_mat[i,]

# Optimize: find lambda such that length of b is one
    opt_obj<-optim(lambda_start,crit_func)

    lambda_opt<-opt_obj$par
    a_vec<--c(a,val[j0])
#  M<-rbind(cbind(diag(-2*lambda_opt,L),A[j0,]),c(A[j0,],0))
    if (length(j0)>1)
    {
      M<-rbind(cbind(diag(-2*lambda_opt,L),t(A[j0,])),cbind(A[j0,],diag(rep(0,length(j0)))))
    } else
    {
      M<-rbind(cbind(diag(-2*lambda_opt,L),A[j0,]),c(A[j0,],0))
    }

    x_vec<-solve(M)%*%a_vec
    b<-x_vec[1:L]
# Lagrange parameters at optimum: they are obtained in closed-form except for lambda_opt which specifies the unit-sphere constraint
    Lagrange<-c(lambda_opt,x_vec[(L+1):length(x_vec)])
    b[2:L]/b[1:(L-1)]
    ts.plot(b)
    #  ts.plot(b_aug)
# Verify constraints: should be all positive for optimal j0
    const_mat<-cbind(const_mat,A%*%b+val)
    print(sum(b^2))
# Store all solutions satisfying all constraints
    if (sum(A%*%b+val+slack>0)==ls&abs(1-sum(b^2))<slack)
    {
      b_opt<-cbind(b_opt,b)
# Criterion: select maximum value implies we must use -a_vec (since a_vec relies on -a above)
      crit_opt<-c(crit_opt,-a_vec[1:L]%*%b)
# Check all constraints satisfied (all results >= 0)
      const_mat_opt<-cbind(const_mat_opt,A%*%b+val)
    }
  }

#-------------
# Check constraints and criterion value(s)
# 1. All designs
  const_mat
# 2. Selected designs only (satisfy all constraints)
  const_mat_opt
# 3. Criteria of solutions: select the one with the largest criterion value
  crit_opt

# Maximal criterion value
  opt_design<-which(crit_opt==max(crit_opt))

# Plot potential solutions (sastisfying all constraints)
  colo<-rainbow(1)
  ts.plot(b_opt[,opt_design],col=colo)

  sum(b_opt[,opt_design])
  A%*%b_opt[,opt_design]+val



  filter_mat<-cbind(gamma,a,b_opt[,opt_design])
  colnames(filter_mat)[ncol(filter_mat)]<-"MSE-LA"
  colnames(filter_mat)[2]<-"MSE-LA target"
  colo<-c("violet","red","blue")

  ts.plot(scale(filter_mat,center=F,scale=T),col=colo)
  for (i in 1:ncol(filter_mat))
    mtext(colnames(filter_mat)[i],col=colo[i],line=-i)

  return(list(filter_mat=filter_mat,val=val))

}




















# Regularization criterion with weight lambda assigned to a single constraint only
# This function computes closed-form solutions in the rank two case: a single constraint
#   -Closed form solution in rank-two case based on quartic equation:
#   -M has rank 2: two non-vanishing eigenvalues (A_target and A_constraint should be linearly independent:
#       -this assumption does not hold in the AR(1) case for example (in the AR(1) case no solution exists becuase the requirements are conflicting: MSE-LA should not correlate with lag 0 but correlate with lead delta and both are linearly dependent...)
#   -For each of the non-vanishing eigenvalues of M two solutions of the quartic equation correspond to the left and right of the eigenvalue such that t(b)%*%b=1
#     2*2=four different solutions i.e. polynomial of order four
# The function returns closed_form_obj:
#   closed_form_obj$crit_opt: t(b)%*%b-1 should vanish
#   closed_form_obj$constraint_opt: deviation from single contrainst. Should ideally vanish as lambda\to\infty
#   closed_form_obj$objective_opt: objective or target correlation. Should be maximized
#   closed_form_obj$b_opt
# The function also returns b i.e. the third solution b_opt[,3] corresponding to the left-side of the smaller eigenvalue
#   -The smaller eigenvalue corresponds to the eigenvector orthogonal to lag zero (the larger eigenvalue corresponds to the eigenvector parallel to lag zero)
#   -It thus decouples from present time
#   -the left side generally corresponds to positive objective function
# In typical applications b=b_opt[,3] is the best choice; but not always. Therefore we return closed_form_obj with all solutions
#   -The user can always decide to use any other solution of b_opt: they are all compliant with t(b)%*%b=1, at least for non-singular designs
MSE_LA_closed_form_rank_two_func<-function(criterion_number,h,lambda,gamma_target,gamma_constraint,val_vec_target,val_vec_constraint,L=length(gamma_constraint),gamma_target_scaling=gamma_target)
{
  #criterion_number,h,lambda<-lambda_vec[i],gamma_target,gamma_constraint,val_vec_target,val_vec_constraint,L)  L<-L_ma
  # lambda<-100
  # In the following A_target is completed at the end in order to be colinear to A_constraint in the case of an AR(1)
  if (L!=length(gamma_target_scaling)|L!=length(gamma_target)|L!=length(gamma_constraint))
  {
    print("The length of any of the gammas differs from L: check the lengths first and rerun the function once verified")
    return()
  }
  delta<-h
  if (lambda<0)
  {
    print("lambda must be positive")
    return()
  }
  if (!(criterion_number%in%1:3))
  {
    print("criterion_number must be 1,2 or 3")
    return()
  }
  if (criterion_number==2&sum(abs(val_vec_constraint))<0.001)
  {
    print("Warning: Criterion 2 cannot account for full decoupling: val_vec_constraint is set to 0.001")
    val_vec_constraint<-rep(0.001,length(val_vec_constraint))
  }
# Scaling of cross correlations: use effective target (for example two sided filter) or one-sided MSE benchmark
  scaling<-as.double(sqrt(t(gamma_target_scaling)%*%gamma_target_scaling))
  # We can either multiply val_target with scaling and left A_target without scaling or left val_target as is
  #   and divide A_target by scaling: we here do the latter (in paper we do the former)
  val_target<-val_vec_target#*scaling
  # Matrix of linear constraints
  A_target<-matrix(rep(NA,L),nrow=1,ncol=L)
  A_target[1,]<-gamma_target/scaling
  val_constraint<-val_vec_constraint#*scaling
  # Matrix of linear constraints
  A_constraint<-matrix(rep(NA,L),nrow=1,ncol=L)
# Scale to unit-length
  A_constraint[1,]<-gamma_constraint/sqrt(sum(gamma_constraint^2))

# Closed form solution in rank-two case based on quartic equation:
# M has rank 2: two non-vanishing eigenvalues (A_target and A_constraint should be linearly independent: this assumption does not hold in the AR(1) case for example)
# For each of the non-vanishing eigenvalues two solutionsof the quartic equation correspond to the left and right of the eigenvalue such that t(b)%*%b=1
#   2*2=four different solutions i.e. polynomial of order four
# Warning messages are issued in the case of singular or nearly singular designs
  closed_form_obj<-closed_form_lambda2_rank2_quartic_func(criterion_number,A_target,A_constraint,val_constraint,val_target,lambda)
# For criteria 1 and 2: rank M is 2, for criterion 3 rank of M is 1
# Rank 2 cases (criteria 1,2):
#   -The first two solutions are to the left and to the right of the largest eigenvalue (function eigen orders eigenvalues from largest to smallest);
#   -The third and fourth are to the left and to the right of the second largest (non-vanishing) eigenvalue
#   -Left and right solutions are nearly (but not perfectly) equal in absolute value
#   -Opposed sign because dominant (nearly singular) eigenvalue of (M-lambda2*I)^{-1} changes sign when lambda2 is to the left or right of eigenvalue of M
  closed_form_obj$crit_opt
  closed_form_obj$constraint_opt
  closed_form_obj$objective_opt

# Some checks
# 0. Comparison of optimal lambda2 and eigenvalues Lambda of M
#   One should change sign of lambda2 (in R-code below we use M+lambda*I instead of M-lambda2*I)
#   One can check that first two -lambda2 are to the left and right of first (largest) eigenvalue and similarly for last two lambda2
  closed_form_obj$lambda2_opt
  closed_form_obj$Lambda[1:2]
# 1. Scaling
# Scaling: should give one if solution exists
  apply(closed_form_obj$b_opt^2,2,sum)
# 2. Target correlation and constraint correlation: these are correlations because the filters are scaled
  val_constraint-A_constraint%*%closed_form_obj$b_opt
  A_target%*%closed_form_obj$b_opt
# 3. Amplitude at frequency zero
  apply(closed_form_obj$b_opt,2,sum)

#---------------------------------------------------------
# 1. Select best filter (out of four from quartic equation): generally third filter
#     Third filter is around second largest eigenvalue and mostly positively autocorrelated with target (fourth is generally negatively autocorrelated)
  closed_form_obj$crit_opt
  closed_form_obj$constraint_opt
  closed_form_obj$objective_opt
  # Select best solution out of four possible of quartic equation: in general the one corresponding to the smaller eigenvalue
  #   i.e. i=3 or i=4
  if (ncol(closed_form_obj$b_opt)>2)
  {
    select_b<-3
  } else
  {
    select_b<-2
  }

# Order according to length constraint: select smallest two i.e. those who comply with length constraint (otherwise linearized criteria don't match correlations)
  select_vec<-which(abs(1-apply(closed_form_obj$b_opt^2,2,sum))<0.001)
# From these two select the one which fits equality constraint best
  select_best<-which(abs(closed_form_obj$constraint_opt[select_vec])==min(abs(closed_form_obj$constraint_opt[select_vec])))
  select_b<-select_vec[select_best]
  b_unscaled<-closed_form_obj$b_opt[,select_b]
# Change sign if criterion is negative
  if (A_target%*%b_unscaled<0)
    b_unscaled<--b_unscaled
# Align scaling on gamma
  b<-b_unscaled*sqrt(sum(A_target^2))/sqrt(sum(b_unscaled^2))
# Check scaling
  sum(b^2)/sum(A_target^2)

  # Plot filters
  ts.plot(cbind(t(A_target),b),col=c("green","blue"),main=paste("MSE (green) vs. MSE-LA (blue): filter ",select_b,sep=""))
  return(list(closed_form_obj=closed_form_obj,b=b,scaling=scaling,A_target=A_target))
}






# Regularization criterion with weight lambda assigned to possibly multiple constraints
# This function computes MSE-LA for arbitrary many restrictions numerically
#   The above closed-form function addresses the rank 2 case i.e. a single constraint only
# The Lagrangian of the optimization problem involves lambda2: the weight of the length constraint t(b)%*%b=1
#   -Closed-form solution for lambda2 such that t(b)%*%b=1 is possible in rank-two case only (polynomial is of order 2*(1+#constraints))
#   -With more than one constraint the polynomial of order 2*(1+#constraints) cannot be solved in lambda2 in closed-form anymore
# For each eigenvalue of M the function computes two solutions lambda2 such that t(b)%*%b=1
#   -to the left and to the right of each eigenvalue: this determines the sign of b i.e. the sign of target correlation
# For each of these solutions, the function returns
#   a. the criterion value crit_opt: is vanishing if t(b)%*%b=1
#   b. the objective function or target correlation objective_opt: must be maximized
#   c. the deviation from the constraint: constraint_opt: is vanishing if the constraint holds exactly (if lambda\to\infty)
#   d. the MSE-LA predictor b_opt
# The best solution in b_opt must
#   1. meet the length constraing t(b)%*%b=1
#   2. maximize the objective function
#   3. meet the imposed constraints (differences should be small in absolute value)
# Since these are tradeoffs the `best' solution is not determined unequivocally:
#   -we therefore return all of them
#   -the user must then select his preferred one based on the best compromise of criteria 1-3 listed above
# Note: for arbitrarily large lambda all constraints are imposed perfectly at least for one of the solutions
MSE_LA_num_opt_func<-function(h,lambda,gamma_target,gamma_constraint,gamma_target_scaling,val_vec_target,val_vec_constraint,L)
{
  # In the following A_target is completed at the end in order to be colinear to A_constraint in the case of an AR(1)
  if (L!=length(gamma_target_scaling))
  {
    print("length of gamma_target_scaling differs from L")
    return()
  }
  delta<-h
  if (lambda<0)
  {
    print("lambda must be positive")
    return()
  }
  scaling<-as.double(sqrt(t(gamma_target_scaling)%*%gamma_target_scaling))
  # We can either multiply val_target with scaling and left A_target without scaling or left val_target as is
  #   and divide A_target by scaling: we here do the latter (in paper we do the former)
  val_target<-val_vec_target#*scaling
  # Matrix of linear constraints
  if (is.vector(gamma_target))
  {
    A_target<-matrix(gamma_target,nrow=1)
  } else
  {
    if (!(L%in%dim(gamma_target)))
    {
      print("Dimension of gamma_target does not match L")
      return()
    }
    if (dim(gamma_target)[2]==L)
    {
      A_target<-gamma_target
    } else
    {
      A_target<-t(gamma_target)
    }
  }
  # Impose scaling (otherwise interpretation is not correlation i.e. lambda of regularization differs)
  A_target<-A_target/scaling
  if (dim(A_target)[2]!=L)
  {
    print("Dimension of gamma_target does not match L")
    return()
  }


  # We can either multiply val_constraint with scaling and left A_constraint without scaling or left val_constraint as is
  #   and divide A_constraint by scaling: we here do the latter (in paper we do the former)
  val_constraint<-val_vec_constraint#*scaling
  # Matrix of linear constraints
  if (is.vector(gamma_constraint))
  {
    A_constraint<-matrix(gamma_constraint,nrow=1)
  } else
  {
    if (!(L%in%dim(gamma_constraint)))
    {
      print("Dimension of gamma_constraint does not match L")
      return()
    }
    if (dim(gamma_constraint)[2]==L)
    {
      A_constraint<-gamma_constraint
    } else
    {
      A_constraint<-t(gamma_constraint)
    }
  }
# Impose scaling (otherwise interpretation is not correlation i.e. lambda of regularization differs)
  A_constraint<-A_constraint/scaling
  if (dim(A_constraint)[2]!=L)
  {
    print("Dimension of gamma_constraint does not match L")
    return()
  }

  # General proceeding:
  #   -lambda2 must be found such that unit-sphere constraint t(b)%*%b=1 holds
  #   -One can always find lambda2 to the left and to the right (in the vicinity) of each non-vanishing eigenvalue of M such that constraint holds
  #     Explanation:
  #       -for |lambda2|\to\infty b\to 0 and t(b)%*%b\to 0;
  #       -for lambda2\to -eigenvalue (of M) of of the eigenvalues of M+lambda2*I goes to zero and therefore the
  #         inverse diverges and therefore b diverges and therefore t(b)%*%b diverges
  #       -By the mean value theorem there exists lambda2 in between both extreme cases such that t(b)%*%b=1
  # Strategy: we first find lambda_up such that t(b)%*%b>1 and lambda_lo such that t(b)%*%b<1 and then we triangulate
  #   for optimal lmabda2 in [lambda_lo,lambda_up]
  # anz=Number of triangulation points: precision will be 2^(-anz)
  anz<-30
  print(anz)
  # 1.Specify M, V, r
  if (lambda<=1)
  {
    M<-t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
  } else
  {
    M<-t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
  }
  eigen_M<-eigen(M)
  Lambda<-eigen_M$values
  D<-diag(Lambda)
  V<-eigen_M$vectors
  lambda2<-1
  # Check: should vanish
  if (F)
  {
    V%*%(diag(1/(Lambda+lambda2)))%*%t(V)%*%r-
      compute_b_func(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)
  }
  # 2. Select a non-vanishing eigenvalue of M:
  #   -For each non-vanishing eigenvalue a lambda2 can be found such that t(b)%*%b=1
  #   -The function find_lambda2_around_eigenvalue_of_M_func does the search and finds a corresponding lambda2 such that objective function is positive
  # The solution of the optimization problem is obtained for that eigenvalue and corresponding lambda2 for which
  #   a. The constraints hold `best': if lambda<infty then deviations are allowed
  #   b. The objective function is maximized
  # Finding the `absolute best' solution is not always straightforward because of the tradeoff: largest objective vs. exact constraint

  # Loop through all non-vanishing eigenvalues and compute lambda2 such that unit-sphere constraint holds:
  #   Compute b_opt, constraint deviation and objective function
  non_vanishing_eigen<-which (abs(Lambda)>10^(-10))
  crit_opt<-b_opt<-lambda2_opt<-objective_opt<-constraint_opt<-NULL
  for (i in non_vanishing_eigen)#i<-2
  {
    lambda_try<-Lambda[i]

    b_opt_obj<-find_lambda2_around_eigenvalue_of_M_func(lambda_try,delta_delta,S,V,Lambda,r,lambda,A_target,A_constraint,L,val_target,val_constraint,anz)
    b<-b_opt_obj$b_opt
    b_opt<-cbind(b_opt,b)
    lambda2_opt<-c(lambda2_opt,b_opt_obj$lambda2_opt)

    ts.plot(b)  #  b_opt[2:L]/b_opt[1:(L-1)]
    # Checks
    # 1. Length constraint: should vanish for optimal lambda2
    crit_opt<-c(crit_opt,1-t(b)%*%b)
    # 2. Constraint: should be close to zero for large lambda
    constraint_opt<-c(constraint_opt,val_constraint-A_constraint%*%b)
    # 3. Should be minimized: difference of target acf and acf of predictor at target forecast horizon delta
    target_cor_lambda2<-A_target%*%b
    objective_opt<-c(objective_opt,target_cor_lambda2)
  }
  # b'b-1: should vanish
  crit_opt
  # Should vanish if constraint holds
  constraint_opt
  # Should be maximized
  objective_opt


  return(list(crit_opt=crit_opt,objective_opt=objective_opt,constraint_opt=constraint_opt,b_opt=b_opt))
}





# For given eigenvalue lambda_try of M this function searches for lambda2 such that unit-sphere constraint t(b)%*%b=1 holds
# It is a fast numerical optimization (not closed-form)
# It can handle arbitrarily many constraints
# A closed-form solution in the case of a single constraint is proposed below: closed_form_lambda2_rank2_quartic_func
find_lambda2_around_eigenvalue_of_M_func<-function(lambda_try,delta_delta,S,V,Lambda,r,lambda,A_target,A_constraint,L,val_target,val_constraint,anz)
{
  #  lambda_largest_absolute<-0
  # Set lambda2 close to lambda_largest_absolute: then
  #   -t(b)%*%b will be large (exceed 1)
  #   -b>0 if delta_delta>0 because sign of largest (in absolute value) eigenvalue will be positive
  #   -b<0 if delta_delta<0 because sign of largest (in absolute value) eigenvalue will be negative
  if (abs(lambda_try)>0)
  {
    delta_delta<-0.1*lambda_try
  } else
  {
    delta_delta<-0.1
  }
  lambda_up<--lambda_try+delta_delta
  S<-V%*%diag(1/(Lambda+lambda_up))%*%t(V)
  b<-S%*%r
  crit_up<-t(b)%*%b
  i_up<-i_lo<-0
  while (crit_up<1)
  {
    i_up<-i_up+1
    delta_delta<-delta_delta/10
    lambda_up<--lambda_try+delta_delta
    b<-compute_b_func(lambda_up,lambda,A_target,A_constraint,L,val_target,val_constraint)
    crit_up<-t(b)%*%b
  }
  # Search for lower boundary: lambda_lo such that t(b)%*%b<1
  lambda_lo<-lambda_up
  b<-compute_b_func(lambda_lo,lambda,A_target,A_constraint,L,val_target,val_constraint)
  crit_lo<-t(b)%*%b
  while (crit_lo>1)
  {
    i_lo<-i_lo+1
    delta_delta<-delta_delta*2
    lambda_lo<--lambda_try+delta_delta
    b<-compute_b_func(lambda_lo,lambda,A_target,A_constraint,L,val_target,val_constraint)
    crit_lo<-t(b)%*%b
  }
  i_up
  i_lo
  # Triangulate search for optimal lambda2 between lambda_up and lambda_lo
  for (i in 1:anz)
  {
    lambda_mid<-(lambda_up+lambda_lo)/2
    b<-compute_b_func(lambda_mid,lambda,A_target,A_constraint,L,val_target,val_constraint)
    crit_mid<-t(b)%*%b
    if (crit_mid<1)
    {
      lambda_lo<-lambda_mid
    } else
    {
      lambda_up<-lambda_mid
    }
  }
  lambda2_opt_positive<-lambda_mid
  b_opt_positive<-compute_b_func(lambda2_opt_positive,lambda,A_target,A_constraint,L,val_target,val_constraint)
  target_cor_positive_lambda2<-A_target%*%b_opt_positive
  if (target_cor_positive_lambda2>0)
  {
    b_opt<-b_opt_positive
    lambda2_opt<-lambda2_opt_positive
  } else
  {

    # Same as above but with negative delta_delta
    if (abs(lambda_try)>0)
    {
      delta_delta<--0.1*lambda_try
    } else
    {
      delta_delta<--0.1
    }

    lambda_up<--lambda_try+delta_delta
    S<-V%*%diag(1/(Lambda+lambda_up))%*%t(V)
    b<-S%*%r
    crit_up<-t(b)%*%b
    i_up<-i_lo<-0
    while (crit_up<1)
    {
      i_up<-i_up+1
      delta_delta<-delta_delta/10
      lambda_up<--lambda_try+delta_delta
      b<-compute_b_func(lambda_up,lambda,A_target,A_constraint,L,val_target,val_constraint)
      crit_up<-t(b)%*%b
    }
    # Search for lower boundary: lambda_lo such that t(b)%*%b<1
    lambda_lo<-lambda_up
    b<-compute_b_func(lambda_lo,lambda,A_target,A_constraint,L,val_target,val_constraint)
    crit_lo<-t(b)%*%b
    while (crit_lo>1)
    {
      i_lo<-i_lo+1
      delta_delta<-delta_delta*10
      lambda_lo<--lambda_try+delta_delta
      b<-compute_b_func(lambda_lo,lambda,A_target,A_constraint,L,val_target,val_constraint)
      crit_lo<-t(b)%*%b
    }
    i_up
    i_lo
    # Triangulate search for optimal lambda2 between lambda_up and lambda_lo
    for (i in 1:anz)
    {
      lambda_mid<-(lambda_up+lambda_lo)/2
      b<-compute_b_func(lambda_mid,lambda,A_target,A_constraint,L,val_target,val_constraint)
      crit_mid<-t(b)%*%b
      if (crit_mid<1)
      {
        lambda_lo<-lambda_mid
      } else
      {
        lambda_up<-lambda_mid
      }
    }
    lambda2_opt_negative<-lambda_mid
    b_opt_negative<-compute_b_func(lambda2_opt_negative,lambda,A_target,A_constraint,L,val_target,val_constraint)
    target_cor_negative_lambda2<-A_target%*%b_opt_negative
    if (target_cor_negative_lambda2<0)
    {
      print("No solution with positive objective function found: problem cannot be solved")
      return()
    } else
    {
      b_opt<-b_opt_negative
      lambda2_opt<-lambda2_opt_negative

    }
  }
  return(list(b_opt=b_opt,lambda2_opt=lambda2_opt))
}









# The following functions are used for numerical optimization of b
compute_b_func<-function(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)
{
  # lambda2<-0.0001 lambda2_start   lambda<-10  lambda2<-lambda2[i]
  # Avoid singular designs: keep size of lambda bounded
  lambda2<-sign(lambda2)*min(abs(lambda2),1.e+6)
  lambda2<-sign(lambda2)*max(abs(lambda2),1.e-10)
  # Avoid singularities when computing inverse for very large or very small lambda
  # Note that lambda2 is scaled differently in both cases but we can ignore this effect if
  if (lambda<=1)
  {
    b<-solve(t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint+lambda2*diag(rep(1,L)))%*%(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
  } else
  {
    b<-solve(t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint+lambda2*diag(rep(1,L)))%*%(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
  }
    #    eigen(t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint)$values
  return(b)
}

crit_func<-function(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)
{

  return(abs(1-sum(compute_b_func(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)^2)))
}
crit_func<-function(lambda2)
{

  return(abs(1-sum(compute_b_func(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)^2)))
}







# This function computes a closed-form solution of lambda2 ensuring unit-sphere constraint t(b)%*%b=1 in the rank-one/two cases
# Rank-two case: a single constraint and target is not self-similar AR(1)
# Rank-one case: a single (or multiple) constraint(s) and target is self-similar AR(1): in this case all constraints are colinear to target
# Solution is root of quartic equation
closed_form_lambda2_rank2_quartic_func<-function(criterion_number,A_target,A_constraint,val_constraint,val_target,lambda)
{
  L<-dim(A_target)[2]
  if (criterion_number==1)
  {
    if (lambda<=1)
    {
      M<-t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint
      r<-(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
    } else
    {
      M<-t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint
      r<-(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
    }
  }

#  lambda<-1.01

  if (criterion_number==2)
  {
    if (lambda<=1)
    {
      M<-t(A_target)%*%A_target-lambda*t(A_constraint)%*%A_constraint
      r<--lambda*val_constraint*t(A_constraint)
    } else
    {
      M<-t(A_target)%*%A_target/lambda-t(A_constraint)%*%A_constraint
      r<--val_constraint*t(A_constraint)
    }
  }
  if (criterion_number==3)
  {
    if (lambda<=1)
    {
      M<-lambda*t(A_constraint)%*%A_constraint
      r<-t(A_target)/2+lambda*val_constraint*t(A_constraint)
    } else
    {
      M<-t(A_constraint)%*%A_constraint
      r<-t(A_target)/(2*lambda)+val_constraint*t(A_constraint)
    }
  }
  eigen_M<-eigen(M)
  Lambda<-eigen_M$values
  D<-diag(Lambda)
  V<-eigen_M$vectors

# Closed form solution for the rank two (and rank one) case
  V[2:L,1]/V[1:(L-1),1]
  xi<-t(V)%*%r
# Eigenvalues are sorted from largest to smallest by eigen
  eta<-Lambda

# Coefficients of quartic function: depend on criterion
  if (criterion_number==1|criterion_number==2)
  {
    a<-1
    b<-2*eta[1]+2*eta[2]
    c<-eta[1]^2+4*eta[1]*eta[2]+eta[2]^2-xi[1]^2-xi[2]^2
    d<-2*eta[1]^2*eta[2]+2*eta[1]*eta[2]^2-2*xi[1]^2*eta[2]-2*xi[2]^2*eta[1]
    e<-eta[1]^2*eta[2]^2-xi[1]^2*eta[2]^2-xi[2]^2*eta[1]^2
  }
  if (criterion_number==3)
  {
    a<-1
    b<-2*eta[1]
    c<-eta[1]^2-sum(xi^2)
    d<--2*eta[1]*sum(xi[2:L]^2)
    e<--eta[1]^2*sum(xi[2:L]^2)
  }
  # Determine roots
  x_vec<-root4_func(a,b,c,d,e)

# Check: should vanish
  c(a,b,c,d,e)%*%x_vec[1]^(4:0)
  c(a,b,c,d,e)%*%x_vec[2]^(4:0)
  c(a,b,c,d,e)%*%x_vec[3]^(4:0)
  c(a,b,c,d,e)%*%x_vec[4]^(4:0)


# Check singular cases: either infty or unit-length constraint does not hold for at least one solution
# Special cases of the quartic equation when S=0 (singularity): can be addressed by quadratic equation
  if (sum(!abs(x_vec)<Inf)>1|abs(c(a,b,c,d,e)%*%x_vec[1]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[2]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[3]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[4]^(4:0))>1.e-8)
  {
    print("quartic equation is singular: quadratic equation for lambda2 is used")
    # In rank-one case the quartic equation degenerates to a quadratic equation
    if (4*eta[1]^2-4*(eta[1]^2-xi[1]^2)<0)
    {
      print("No real solution available")
      return()
    }
    # Solutions of the quadratic equation: they differ by the sign of b
    lambda2<-c(0.5*(-2*eta[1]+sqrt(4*eta[1]^2-4*(eta[1]^2-xi[1]^2))),0.5*(-2*eta[1]-sqrt(4*eta[1]^2-4*(eta[1]^2-xi[1]^2))))
  } else
  {
# Solutions of Quartic equation: select the one which maximizes target correlation
# Check: each one of the following terms should vanish (root of quartic equation): if it doesn't then a warning is issued
    if (abs(c(a,b,c,d,e)%*%x_vec[1]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[2]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[3]^(4:0))>1.e-8|abs(c(a,b,c,d,e)%*%x_vec[4]^(4:0))>1.e-8)
      print("Warning: closed-form solution of quartic equation for lambda2 is nearly singular")
    if (abs(Im(x_vec[1]))>1.e-10|abs(Im(x_vec[4]))>1.e-10|abs(Im(x_vec[4]))>1.e-10|abs(Im(x_vec[4]))>1.e-10)
      print("Warning: nearly singular case. Non-vanishing imaginary part in at least one solution of quartic equation")
    lambda2<-Re(x_vec)
  }
  # Compute b for all lambda2 solutions (4 or 2 solutions)
  # Correct lambda2 maximizes objective subject to constraint (the latter is a soft constraint
  #  whose deviation/error is controlled by lambda)
  crit_opt<-b_opt<-lambda2_opt<-objective_opt<-constraint_opt<-NULL
  for (i in 1:length(lambda2))#i<-2
  {
    lambda2_opt<-c(lambda2_opt,lambda2[i])
# Old code: does not account for different criteria i.e. different M and r specifications
    b_vec<-compute_b_func(lambda2[i],lambda,A_target,A_constraint,L,val_target,val_constraint)
# New code: uses M and r as specified above by different criteria
#   If lambda2 is zero we shift is slightly to avoid singular designs (M is rank deficient) : this is copied from function compute_b_func
    lambda2_a<-sign(lambda2[i])*min(abs(lambda2[i]),1.e+6)
    lambda2_a<-sign(lambda2[i])*max(abs(lambda2[i]),1.e-10)
    b_vec<-solve(M+lambda2_a*diag(rep(1,dim(M)[1])))%*%r  #eigen(M+lambda2[i]*diag(rep(1,dim(M)[1])))
    b_opt<-cbind(b_opt,b_vec)
    ts.plot(b_vec)
    # Vanishes for solutions of quadratic and quartic equations
    crit_opt<-c(crit_opt,t(b_vec)%*%b_vec-1)
    # Correct solution of quartic or quadratic equations must maximize the target correlation
    objective_opt<-c(objective_opt,A_target%*%b_vec)
    # Constraint: ideally the error vanishes
    constraint_opt<-c(constraint_opt,val_constraint-A_constraint%*%b_vec)
  }
  return(list(b_opt=b_opt,lambda2_opt=lambda2_opt,crit_opt=crit_opt,objective_opt=objective_opt,constraint_opt=constraint_opt,Lambda=Lambda,M=M,V=V))
}


# This function computes close-form solution of quartic equation
# Note: it does not work for all cases: should be generalized...
root4_func<-function(a,b,c,d,e)
{
  p<-(8*a*c-3*b^2)/(8*a^2)
  q<-(b^3-4*a*b*c+8*a^2*d)/(8*a^3)
  Delta0<-c^2-3*b*d+12*a*e
  Delta1<-2*c^3-9*b*c*d+27*b^2*e+27*a*d^2-72*a*c*e
  Q<-((0*1.i+Delta1+sqrt(0*1.i+Delta1^2-4*Delta0^3))/2)^(1/3)
  if (Q==0)
  {
    print("Q=0 in quartic equation. General solution has not been implemented yet. Change slightly the hyperparameters to obtain a regular case")
  }
  S<-0.5*sqrt(0*1.i-2*p/3+(Q+Delta0/Q)/(3*a))
  x1<--b/(4*a)-S+0.5*sqrt(0*1.i-4*S^2-2*p+q/S)
  x2<--b/(4*a)-S-0.5*sqrt(0*1.i-4*S^2-2*p+q/S)
  x3<--b/(4*a)+S+0.5*sqrt(0*1.i-4*S^2-2*p-q/S)
  x4<--b/(4*a)+S-0.5*sqrt(0*1.i-4*S^2-2*p-q/S)
  # Check
  c(a,b,c,d,e)%*%x1^(4:0)

  return(c(x1,x2,x3,x4))
}






# Play with solution lambda2 of unit-sphere constraint: assumes all variables were previously initialized
# This function is not part of the optimization: it is here for illustration of solution:
#   lambda2 to the left or to the right of non-vanishing eigenvalues of M
play_with_lambda2_func<-function()
{

#----------------------------
# General idea
# For each eigenvalue!=0 of M we can find lambda2 in its vicinity such that t(b)%*%b=1 and
#   either b>0 or b<0 (objective>/<0) depending on sign of delta  (on which side of eigenvalue)
# If target is AR(1), then A_target~A_constraint are colinear and rank of M is one even when adding more constraints

# Depending on lambda>/<1 we implement M and r such that solution is less singular (M%*%r is indifferent to both implementations)
  if (lambda<=1)
  {
    M<-t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
  } else
  {
    M<-t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
  }

# Eigenvalues of M and diagonalization
  Lambda<-eigen(M)$values
  D<-diag(Lambda)
  V<-eigen(M)$vectors
# Checks
  if (F)
  {
# Check: is zero
    eigen(M-(V)%*%D%*%t(V))$values
# Check: is zero
    eigen(solve(M+lambda2*diag(rep(1,L)))-V%*%diag(1/(Lambda+lambda2))%*%t(V))$values
  }

# Select largest (in absolute value) eigenvalue: its the first one because the function eigen orders according to size of eigenvalue
  lambda_largest_absolute<-Lambda[which(abs(Lambda)==max(abs(Lambda)))]
# The above is the same as
  lambda_largest_absolute<-Lambda[1]
# Set lambda2 close to lambda_largest_absolute: then
#   -t(b)%*%b will be large (exceed 1)
#   -b>0 if delta>0 because sign of largest (in absolute value) eigenvalue will be positive
#   -b<0 if delta<0 because sign of largest (in absolute value) eigenvalue will be negative
  delta<--0.001
  lambda2<--lambda_largest_absolute+delta
  S<-V%*%diag(1/(Lambda+lambda2))%*%t(V)
  b<-S%*%r
  t(b)%*%b
  ts.plot(b)

# Select large lambda2: then b~0 and t(b)%*%b~0
  lambda2<-10000
  S<-V%*%diag(1/(Lambda+lambda2))%*%t(V)
  b<-S%*%r
  t(b)%*%b
}












# Function returns a rotation matrix transforming a vector x into a vector y
# The rotation matrix transforming x to y is not unique (n dimensions for x or y and n*n for rotation matrix)
#   but ANY matrix will do for the purpose of deriving a closed-form solution to the MSE-LA forecast
rotation = function(x,y){
  u=x/sqrt(sum(x^2))

  v=y-sum(u*y)*u
  v=v/sqrt(sum(v^2))

  cost=sum(x*y)/sqrt(sum(x^2))/sqrt(sum(y^2))

  sint=sqrt(1-cost^2);

  rot_mat<- as.double(sqrt(y%*%y))* t(cbind(u,v) %*% matrix(c(cost,-sint,sint,cost), 2) %*% t(cbind(u,v)))/as.double(sqrt(x%*%x))
  return(list(rot_mat=rot_mat))
}





# We here relate alpha0 in DFP-constraint to phase lead of DFP over MSE
# Inputs: alpha0, gamma_constraint (gamma0 or nowcast),gamma_target (gammah or or MSE predictor),b_opt (DFP predictor). Note that theta=acos(alpha0) is not needed explicitly since it is obtained from b_opt and gamma_constraint

# The function returns:
#   1. Phi=Phi: first order linearization of lead of DFP over MSE as a function of lead of MSE over nowcast
#     -lead DFP=Phi*lead MSE
#     -The approximation holds in a vicinity of frequency zero
#     -Phi depends on lambda0 which in turn depends on theta=acos(alpha0) where alpha0 is hyperparameter of DFP-constraint
#   2. beta_formula=beta_formula: lead of DFP over MSE exact frequency-dependent non-linear function
#     -The above linearization is good proxy for omega small
#     -This formula does not rely on DFP: it is based on trigonometric identities which involve gamma_constraint (gamma0, nowcast) and gamma_target (gammah, MSE predictor) only
#   3 beta_direct=beta_direct: lead of DFP over MSE based on explicit phase-shifts of both predictors.
#     -In constrast to previous beta_formula, this expression relies explicitly on DFP predictor
#     -In principle, if optimization came out perfectly, then b_opt would lie in space spanned by gamma_constraint and gamma_target and therefore beta_direct=beta_formula
#     -However, in general both expressions differ a bit due to numerical errors. The difference should be small though.

shift_alpha0_func<-function(alpha0,gamma_constraint,gamma_target,b_opt)
{# gamma_constraint<-gamma_constraint1   gamma_target<-gamma_target1
  theta<-acos(alpha0)
# Scale
  gamma_constraint<-gamma_constraint/as.double(sqrt(gamma_constraint%*%gamma_constraint))
  gamma_target<-gamma_target/as.double(sqrt(gamma_target%*%gamma_target))
  # Compute theta0: angle between gammah (target, MSE) and gamma0 (constraint, nowcast)
  theta0<-acos(gamma_constraint%*%gamma_target)
  theta0
  # Compute transferfunctions
  K<-600
  filt_obj<-amp_shift_func(K,gamma_constraint,F)
  trffkt0<-filt_obj$trffkt
  shift0<-filt_obj$shift
  filt_obj<-amp_shift_func(K,gamma_target,F)
  shift1<-filt_obj$shift
  trffkt1<-filt_obj$trffkt
  filt_obj<-amp_shift_func(K,b_opt,F)
  shiftb<-filt_obj$shift
  trffktb<-filt_obj$trffkt

  # Formula for lamba0 in regression of gamma_target and gamma_constraint on b_opt, assuming weight of gamma_traget is 1:
  #   -the resulting linear combination is proportional to b_opt but generally length is different from one #   -lambda0 depends on alpha0 by means of theta=acos(alpha0)
  lambda0<-as.double(-sin(theta-theta0)/sin(pi-theta))
# This is a linearized proxy for lambda0
  lambda_0_prox<-theta0/theta-1
  # Check
  if (F)
  {
    # Check formula relating theta (i.e. acos(alpha0)) to replication of DFP by regression weights
    #   The formula in notes assumes weight of MSE (gamma_target) is scaled to one
    #   Therefore we look at ratio of coefficient of gamma0 (i.e. nowcast) to coefficient of gammah (MSE or target)
    lm_hp_month<-lm(b_opt~gamma_constraint+gamma_target)
    summary(lm_hp_month)
    lm_hp_month$coef[2]/lm_hp_month$coef[3]
    # Check 1: Should match above ratio of coefficients
    lambda0
    # Check 2: proportionality: OK up to small rounding errors (probably due to root of quartic equation and matrix inversions or numerical computation of lambda 2 in above function which is not in closed-form)
    ratio<-(gamma_target+lambda0*gamma_constraint)/b_opt
    # Should be ideally constant (proportionality)
    ratio
  }

  # Print warning if proportionality is less than `stellar'
  ratio<-(gamma_target+lambda0*gamma_constraint)/b_opt
  if (max(abs(ratio/mean(ratio)-1))>0.001)
  {
    print("Proportinality error is larger than 0.001: eventually check predictor")
  }

  if (F)
  {
    # We can rely either on filter_mat[,"MSE"]+lambda0*filter_mat[,"HP-concurrent"] or on filter_mat[,"MSE-LA"]
    #   -The former lies in plane spanned by gamma0,gammah (HP-concurrent and MSE) and therefore some results are exact
    #   -The latter is not exatcly in plane due to rounding errors of solution of quartic equation. Therefore some results are not perfectly exact.
    b_new<-gamma_target+lambda0*gamma_constraint
    filt_obj<-amp_shift_func(K,b_new,F)
    trffktb_new<-filt_obj$trffkt
  }
  b_new<-b_opt
  trffktb_new<-trffktb

  if (F)
  {
    # Checks: There may be evidence of rounding errors with b_opt due to quartic solution or numerical computation of lambda2 such that b_opt does not perfectly lie in space spanned by gamma_constraint and gamma_target. Using b_new<-gamma_target+lambda0*gamma_constraint cancels these effects.
    (trffkt1+lambda0*trffkt0)/trffktb
    # Triangle of transferfunctions in frequency domain
    trffkt1+lambda0*trffkt0-trffktb_new
  }
  # Define phases and amplitude functions
  Phi1<-Arg(trffkt1)
  Phi0<-Arg(trffkt0)
  Phib_new<-Arg(trffktb_new)
  # One angle of triangle of transferfunctions: this is frequency dependent!!!
  gamma_angle<-Phi1-Phi0
  # One length of triangle of transferfunctions: this is frequency dependent!!!
  b<-abs(lambda0*trffkt0)
  # Second length of triangle of transferfunctions: this is frequency dependent!!!
  a<-abs(trffkt1)
  # Derive angle beta in triangle which corresponds to excess phase of DFP (b_opt) over MSE (gamma_target)
  #   -This formula replicates the direct calculation below
  #   -But is does not depend directly on b_opt: it is based on geometry not involving DFP predictor
  #   -The formula is frequency-dependent and non-linear: below we derive a simple linear first order Taylor-proxy
  #   -The formula derives the phase excess of DFP over MSE (target), i.e., beta_formula, to gamma_angle which is the phase difference between MSE (target) and nowcast (constraint). Therefore we can derive the phase-excess or lead of DFP over MSE as a function of phase-excess or lead of MSE (gammah, target)  over nowcast (gamma0, constraint)
  #   -The formula depends on b which depends on lambda0 which depends on theta which depends on alpha0: links DFP constraint alpha0 with phase excess of DFP!
  beta_formula<-atan((sin(gamma_angle)*b)/(a-cos(gamma_angle)*b))
  # Same expression but we divide by b
  beta_formula<-atan((sin(gamma_angle))/(a/b-cos(gamma_angle)))
  # Direct computation of lead (phase excess) of DFP over MSE: in contrast to beta_formula this expression assumes knowledge of DFP predictor (at least its phase)
  beta_direct<-(Phib_new-Phi1)
  if (F)
  {
    # Check: should match
    cbind(beta_formula,beta_direct)
  }
  # Various approximations
  if (F)
  {
    # Replacing a/b by 1 is more or less OK though now differences become perceptible: forget that one!
    beta_formula_approx<-atan((sin(gamma_angle))/(1/abs(lambda0)-cos(gamma_angle)))
    # First order Taylor approx of atan is OK
    beta_formula_approx<-((sin(gamma_angle))/(a/b-cos(gamma_angle)))
    # First order Taylor approx of atan and sin is OK
    beta_formula_approx<-((gamma_angle)/(a/b-cos(gamma_angle)))
    # First order Taylor approx of atan and sin and cos is OK
    beta_formula_approx<-((gamma_angle)/(a/b-1))
    # First order Taylor approx of atan, sin, cos and using amp-ratio at frequency zero, i.e., (a/b)[1], is OK for small frequencies but if amplitudes change relative to each other with frequency then differences become perceptible
    beta_formula_approx<-((gamma_angle)/((a/b)[1]-1))
    # First order Taylor approx of atan, sin, cos and replacing a/b by 1/lambda0, is not very good_skip
    beta_formula_approx<-gamma_angle/(1/abs(lambda0)-1)
    # check quality of various approximations:
    cbind(beta_direct,beta_formula_approx)[1:60,]
    # Note that difference between last and up to last approximation are due to the fact that sum of coefficients of gamma0 (nowcast or constraint) and of gammah (MSE or target) differ, see below
    # Lengths (variances) are the same
    gamma_constraint%*%gamma_constraint
    gamma_target%*%gamma_target
    # But sums differ i.e. amplitudes at frequency zero are slightly different
    sum(gamma_constraint)
    sum(gamma_target)
  }
  # This is the best/simple linear proxy as used in paper: it relies on linearization and amplitude ratio at frequency 0
  #   -The formula depends on b[1] ate frequency zero which depends on lambda0 which depends on theta which depends on alpha0: links DFP constraint alpha0 with phase excess of DFP towards frequency zero!
  Phi<-1/((a/b)[1]-1)
  cbind(Phi*gamma_angle,beta_direct)
  return(list(Phi=Phi,beta_formula=beta_formula,beta_direct=beta_direct,Phi1=Phi1,Phi0=Phi0,Phib_new=Phib_new,trffkt0=trffkt0,trffkt1=trffkt1,trffktb=trffktb,theta0=theta0,shift1=shift1,shift0=shift0,shiftb=shiftb,lambda0=lambda0))
}



# Kind of inverse of above function: it computes alpha0 for given rho0=(excess shift of FFP over MSE)/(excess shift MSE over nowcast)
# Inputs: rho0, gamma_constraint (gamma0 or nowcast),gamma_target (gammah or or MSE predictor)
# Implements formula of corollary 1 in timeliness paper
alpha0_rho0_func<-function(rho0,gamma_constraint,gamma_target)
{# gamma_constraint<-gamma_constraint1   gamma_target<-gamma_target1

# Scale
  gamma_constraint<-gamma_constraint/as.double(sqrt(gamma_constraint%*%gamma_constraint))
  gamma_target<-gamma_target/as.double(sqrt(gamma_target%*%gamma_target))
# Compute theta0: angle between gammah (target, MSE) and gamma0 (constraint, nowcast)
  theta0<-acos(gamma_constraint%*%gamma_target)
  theta0
  # Compute transferfunctions
  K<-600
  filt_obj<-amp_shift_func(K,gamma_constraint,F)
  trffkt0<-filt_obj$trffkt
  shift0<-filt_obj$shift
  filt_obj<-amp_shift_func(K,gamma_target,F)
  shift1<-filt_obj$shift
  trffkt1<-filt_obj$trffkt
  alpha0<-as.double(cos((abs(trffkt0[1])*(1+rho0)*theta0)/(abs(trffkt0[1])*(1+rho0)-abs(trffkt1[1]*rho0))))
  return(list(alpha0=alpha0))
}


# This function computes lamba0, the weight of gamma_constraint, in the linear combination b_opt=gamma_traget+lambda0*gamma_constraint
# See Corollary 2 in timeliness paper
lambda0_rho0_func<-function(rho0,gamma_constraint,gamma_target)
{# gamma_constraint<-gamma_constraint1   gamma_target<-gamma_target1

  # Scale
  gamma_constraint<-gamma_constraint/as.double(sqrt(gamma_constraint%*%gamma_constraint))
  gamma_target<-gamma_target/as.double(sqrt(gamma_target%*%gamma_target))
  # Compute transferfunctions
  K<-600
  filt_obj<-amp_shift_func(K,gamma_constraint,F)
  trffkt0<-filt_obj$trffkt
  shift0<-filt_obj$shift
  filt_obj<-amp_shift_func(K,gamma_target,F)
  shift1<-filt_obj$shift
  trffkt1<-filt_obj$trffkt
  lambda0<--as.double(abs(trffkt1[1])*rho0/(abs(trffkt0[1])*(1+rho0)))
  return(list(lambda0=lambda0))
}




#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# Older desgins

MSE_LA_num_opt_func_old<-function(h,sup_vec_target,lambda,gamma,val_vec_target,val_vec_constraint)
{
  # In the following A_target is completed at the end in order to be colinear to A_constraint in the case of an AR(1)
  L<-length(gamma)
  delta<-h
  if (!delta%in%sup_vec_target)
  {
    print("delta must be in sup_vec_target")
    return()
  }
  if (lambda<0)
  {
    print("lambda must be positive")
    return()
  }

  ls<-length(sup_vec_target)
  l<-length(gamma)
  scaling<-as.double(sqrt(t(gamma)%*%gamma))
  # We can either multiply val_target with scaling and left A_target without scaling or left val_target as is
  #   and divide A_target by scaling: we here do the latter (in paper we do the former)
  val_target<-val_vec_target#*scaling
  # Matrix of linear constraints
  A_target<-matrix(rep(NA,L*ls),nrow=ls,ncol=L)
  # Optimization requires column and row names (because constraints could be applied to subset of coefficients)
  # Forecast horizon delta
  # 1. Upper value of constraint: maximal value of correlation (i.e. 1)
  # 2. Linear constraint
  #   -The forecast depends on eps_t,eps_{t-1},...;
  #   -The future data point at horizon delta depends on eps_{t+delta},eps_{t+delta-1},....
  #   -Therefore the correlation of predictor b is with gamma_{delta}, gamma_{delta+1},...
  A_target[1,]<-c(gamma[sup_vec_target[1]+1:min(L,l-sup_vec_target[1])],rep(0,L-min(L,l-sup_vec_target[1])))/scaling
  # Initialize all smaller-equal constraints (maximization)
  if (ls>1)#(threshold_plus_minus>0)
  {
    for (i in 2:ls)#i<-ls
    {
      A_target[i,]<-c(gamma[sup_vec_target[i]+1:min(L,l-sup_vec_target[i])],rep(0,L-min(L,l-sup_vec_target[i])))/scaling
    }
  }
  # Constraints could be set to prespecified values
  #  if (set_val)
  #    val_target<-val_vec_target
  dim(A_target)
  # MSE predictor: last delta entries are zero
  # If MSE_TF==F then we complete the MSE predictor such that A_target can be colinear to A_constraint: singular AR(1) case
  if (F)
  {
    if (!MSE_TF)
    {
      # A_target is completed at the end in order to be colinear to A_constraint in the case of an AR(1)
      # This is no more the MSE predictor (the latter has vanishing last delta entries)
      A_target[(L-delta+1):L]<-ar1^(1:delta)*A_target[L-delta]
    }
  }
  ls<-length(sup_vec_constraint)
  l<-length(gamma)
  # We can either multiply val_constraint with scaling and left A_constraint without scaling or left val_constraint as is
  #   and divide A_constraint by scaling: we here do the latter (in paper we do the former)
  val_constraint<-val_vec_constraint#*scaling
  # Matrix of linear constraints
  A_constraint<-matrix(rep(NA,L*ls),nrow=ls,ncol=L)
  # Optimization requires column and row names (because constraints could be applied to subset of coefficients)
  # Forecast horizon delta
  # 1. Upper value of constraint: maximal value of correlation (i.e. 1)
  #  val_constraint[1]<-1
  # 2. Linear constraint
  #   -The forecast depends on eps_t,eps_{t-1},...;
  #   -The future data point at horizon delta depends on eps_{t+delta},eps_{t+delta-1},....
  #   -Therefore the correlation of predictor b is with gamma_{delta}, gamma_{delta+1},...
  A_constraint[1,]<-c(gamma[delta+1:min(L,l-delta)],rep(0,L-min(L,l-delta)))/scaling
  # This is the same as (because delta is always first element of sup_vec: this condition is checked above)
  A_constraint[1,]<-c(gamma[sup_vec_constraint[1]+1:min(L,l-sup_vec_constraint[1])],rep(0,L-min(L,l-sup_vec_constraint[1])))/scaling
  # Initialize all smaller-equal constraints (maximization)
  if (ls>1)#(threshold_plus_minus>0)
  {
    for (i in 2:ls)#i<-ls
    {
      #      val_constraint[i]<-(gamma[(delta+1-sup_vec_constraint[i]):l]%*%gamma[1:(l-(delta-sup_vec_constraint[i]))]/sum(gamma^2))
      A_constraint[i,]<-c(gamma[sup_vec_constraint[i]+1:min(L,l-sup_vec_constraint[i])],rep(0,L-min(L,l-sup_vec_constraint[i])))/scaling
    }
  }
  # Constraints could be set to prespecified values
  #  if (set_val)
  #    val_constraint<-val_vec_constraint
  dim(A_constraint)

  # General proceeding:
  #   -lambda2 must be found such that unit-sphere constraint t(b)%*%b=1 holds
  #   -One can always find lambda2 to the left and to the right (in the vicinity) of each non-vanishing eigenvalue of M such that constraint holds
  #     Explanation:
  #       -for |lambda2|\to\infty b\to 0 and t(b)%*%b\to 0;
  #       -for lambda2\to -eigenvalue (of M) of of the eigenvalues of M+lambda2*I goes to zero and therefore the
  #         inverse diverges and therefore b diverges and therefore t(b)%*%b diverges
  #       -By the mean value theorem there exists lambda2 in between both extreme cases such that t(b)%*%b=1
  # Strategy: we first find lambda_up such that t(b)%*%b>1 and lambda_lo such that t(b)%*%b<1 and then we triangulate
  #   for optimal lmabda2 in [lambda_lo,lambda_up]
  # anz=Number of triangulation points: precision will be 2^(-anz)
  anz<-30
  print(anz)
  # 1.Specify M, V, r
  if (lambda<=1)
  {
    M<-t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
  } else
  {
    M<-t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
  }
  eigen_M<-eigen(M)
  Lambda<-eigen_M$values
  D<-diag(Lambda)
  V<-eigen_M$vectors
  lambda2<-1
  # Check: should vanish
  if (F)
  {
    V%*%(diag(1/(Lambda+lambda2)))%*%t(V)%*%r-
      compute_b_func(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)
  }
  # 2. Select a non-vanishing eigenvalue of M:
  #   -For each non-vanishing eigenvalue a lambda2 can be found such that t(b)%*%b=1
  #   -The function find_lambda2_around_eigenvalue_of_M_func does the search and finds a corresponding lambda2 such that objective function is positive
  # The solution of the optimization problem is obtained for that eigenvalue and corresponding lambda2 for which
  #   a. The constraints hold `best': if lambda<infty then deviations are allowed
  #   b. The objective function is maximized
  # Finding the `absolute best' solution is not always straightforward because of the tradeoff: largest objective vs. exact constraint

  # Loop through all non-vanishing eigenvalues and compute lambda2 such that unit-sphere constraint holds:
  #   Compute b_opt, constraint deviation and objective function
  non_vanishing_eigen<-which (abs(Lambda)>10^(-10))
  crit_opt<-b_opt<-lambda2_opt<-objective_opt<-constraint_opt<-NULL
  for (i in non_vanishing_eigen)#i<-2
  {
    lambda_try<-Lambda[i]

    b_opt_obj<-find_lambda2_around_eigenvalue_of_M_func(lambda_try,delta_delta,S,V,Lambda,r,lambda,A_target,A_constraint,L,val_target,val_constraint,anz)
    b<-b_opt_obj$b_opt
    b_opt<-cbind(b_opt,b)
    lambda2_opt<-c(lambda2_opt,b_opt_obj$lambda2_opt)

    ts.plot(b)  #  b_opt[2:L]/b_opt[1:(L-1)]
    # Checks
    # 1. Length constraint: should vanish for optimal lambda2
    crit_opt<-c(crit_opt,1-t(b)%*%b)
    # 2. Constraint: should be close to zero for large lambda
    constraint_opt<-c(constraint_opt,val_constraint-A_constraint%*%b)
    # 3. Should be minimized: difference of target acf and acf of predictor at target forecast horizon delta
    target_cor_lambda2<-A_target%*%b
    objective_opt<-c(objective_opt,target_cor_lambda2)
  }
  # b'b-1: should vanish
  crit_opt
  # Should vanish if constraint holds
  constraint_opt
  # Should be maximized
  objective_opt


  return(list(crit_opt=crit_opt,objective_opt=objective_opt,constraint_opt=constraint_opt,b_opt=b_opt))
}


MSE_LA_closed_form_rank_two_func_old<-function(criterion_number,h,sup_vec_target,lambda,gamma_target,gamma_constraint,gamma_target_scaling,val_vec_target,val_vec_constraint,L)
{
  # lambda<-0.5 L<-10
  # In the following A_target is completed at the end in order to be colinear to A_constraint in the case of an AR(1)
  if (L!=length(gamma_target_scaling))
  {
    print("length of gamma_target_scaling differs from L")
    return()
  }
  delta<-h
  if (!delta%in%sup_vec_target)
  {
    print("delta must be in sup_vec_target")
    return()
  }
  if (lambda<0)
  {
    print("lambda must be positive")
    return()
  }
  # Scaling of cross correlations: use effective target (for example two sided filter) or one-sided MSE benchmark
  scaling<-as.double(sqrt(t(gamma_target_scaling)%*%gamma_target_scaling))
  # We can either multiply val_target with scaling and left A_target without scaling or left val_target as is
  #   and divide A_target by scaling: we here do the latter (in paper we do the former)
  val_target<-val_vec_target#*scaling
  # Matrix of linear constraints
  A_target<-matrix(rep(NA,L),nrow=1,ncol=L)
  A_target[1,]<-gamma_target/scaling
  val_constraint<-val_vec_constraint#*scaling
  # Matrix of linear constraints
  A_constraint<-matrix(rep(NA,L),nrow=1,ncol=L)
  A_constraint[1,]<-gamma_constraint/scaling


  # General proceeding:
  #   -lambda2 must be found such that unit-sphere constraint t(b)%*%b=1 holds
  #   -The solution to this problem can be formulated as roots of a quartic equation, see paper
  # 1.Specify M, V, r
  # We distinguish the cases lambda>/< 1 to facilitate numerical precision  (the transformation is unnecessary in the case of infinite numerical precision)
  if (lambda<=1)
  {
    M<-t(A_target)%*%A_target+lambda*t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)+lambda*val_constraint*t(A_constraint))
  } else
  {
    M<-t(A_target)%*%A_target/lambda+t(A_constraint)%*%A_constraint
    r<-(+val_target*t(A_target)/lambda+val_constraint*t(A_constraint))
  }
  eigen_M<-eigen(M)
  Lambda<-eigen_M$values
  D<-diag(Lambda)
  V<-eigen_M$vectors
  lambda2<-1
  # Check: should vanish
  if (F)
  {
    V%*%(diag(1/(Lambda+lambda2)))%*%t(V)%*%r-
      compute_b_func(lambda2,lambda,A_target,A_constraint,L,val_target,val_constraint)
  }

  # Closed form solution in rank-two case based on quartic equation:
  # M has rank 2: two non-vanishing eigenvalues (A_target and A_constraint should be linearly independent: this assumption does not hold in the AR(1) case for example)
  # For each of the non-vanishing eigenvalues two solutionsof the quartic equation correspond to the left and right of the eigenvalue such that t(b)%*%b=1
  #   2*2=four different solutions i.e. polynomial of order four
  # Warning messages are issued in the case of singular or nearly singular designs
  closed_form_obj<-closed_form_lambda2_rank2_quartic_func(criterion_number,A_target,A_constraint,val_constraint,val_target,lambda)

  # The first two solutions are to the left and to the right of the largest eigenvalue;
  # The third and fourth are to the left and to the right of the second largest (non-vanishing) eigenvalue
  # Left and right solutions are nearly (but not perfectly) equal in absolute value
  #   Opposed sign because dominant (nearly singular) eigenvalue of (M-lambda2*I)^{-1} changes sign when lambda2 is to the left or right of eigenvalue of M
  closed_form_obj$crit_opt
  closed_form_obj$constraint_opt
  closed_form_obj$objective_opt

  # Some checks
  # 0. Comparison of optimal lambda2 and eigenvalues Lambda of M
  closed_form_obj$lambda2_opt
  Lambda[1:2]
  # 1. Scaling
  # Scaling
  sum(A_constraint^2)
  apply(closed_form_obj$b_opt^2,2,sum)
  # 2. Target correlation and constraint correlation: these are correlations because the filters are scaled
  val_constraint-A_constraint%*%closed_form_obj$b_opt
  A_target%*%closed_form_obj$b_opt
  # 3. Amplitude at frequency zero
  apply(closed_form_obj$b_opt,2,sum)

  #---------------------------------------------------------
  # Filter data
  # 1. Select best filter (out of four from quartic equation): generally third filter
  #     Third filter is around second largest eigenvalue and mostly positively autocorrelated with target (fourth is generally negatively autocorrelated)
  closed_form_obj$crit_opt
  closed_form_obj$constraint_opt
  closed_form_obj$objective_opt
  # Select best solution out of four possible of quartic equation: in general the one corresponding to the smaller eigenvalue
  #   i.e. i=3 or i=4
  if (ncol(closed_form_obj$b_opt)>2)
  {
    select_b<-3
  } else
  {
    select_b<-2
  }
  b_unscaled<-closed_form_obj$b_opt[,select_b]
  # Align scaling on gamma
  b<-b_unscaled*sqrt(sum(A_target^2))/sqrt(sum(b_unscaled^2))
  # Check scaling
  sum(b^2)/sum(A_target^2)
  # Check transfer function at ferquency zero
  sum(b)
  sum(A_target)

  if (sum(b)<0)
  {
    print("transferfunction of MSE-LA is negative at frequency zero: apply centering")
    #  return()
  }


  # Plot filters
  ts.plot(cbind(t(A_target),b),col=c("green","blue"),main=paste("MSE (green) vs. MSE-LA (blue): filter ",select_b,sep=""))
  return(list(closed_form_obj=closed_form_obj,b=b,scaling=scaling,A_target=A_target))
}





plot_func<-function()
{
  colo <- rainbow(ncol(filter_mat))
  par(mfrow = c(2, 2))
  
  # Scale all filters to unit energy for visual comparability.
  mplot <- scale(filter_mat, center = FALSE, scale = TRUE)
  
  # Verify filter energies after scaling (should all equal 1).
  apply(mplot^2, 2, sum)
  
  # ── Panel 1: Scaled predictor coefficient profiles ────────────────────────────
  plot(mplot[, 1],
       main = "Scaled Predictors", axes = FALSE, type = "l",
       xlab = "Lags", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(mplot), max(mplot)))
  mtext(colnames(mplot)[1], col = colo[1], line = -1)
  
  for (i in 2:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
    mtext(colnames(mplot)[i], col = colo[i], line = -i)
  }
  lines(mplot[, 2], col = colo[2])   # Redraw second filter on top for visibility.
  
  axis(1, at     = c(0, (1:(nrow(mplot) / 10)) * 10),
       labels = c(0, (1:(nrow(mplot) / 10)) * 10))
  axis(2)
  box()
  
  # ── Panel 2: CCF against xi (Wold decomposition) ──────────────────────────────
  # For each predictor, compute the CCF against xi at lags 0, 1, ..., h.
  # The dashed vertical line marks the target horizon h; the horizontal line
  # marks zero correlation.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], xi)$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against xi", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)   # Vertical marker at target horizon h.
  abline(h = 0)                 # Zero-correlation reference line.
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  
  # ── Panel 3: CCF against V1 (leading eigenvector of M) ────────────────────────
  # V[,1] ≈ gamma_0 (up to sign) when delta is small, so this panel isolates
  # the gamma_0 contribution to the CCF.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], V[, 1])$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against V1", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)
  abline(h = 0)
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  
  # ── Panel 4: CCF against V2 (second eigenvector of M) ─────────────────────────
  # V[,2] is orthogonal to V[,1] ≈ gamma_0 and captures the full decoupling
  # direction introduced by the perturbation.
  max_lag <- 0
  ccf_mat <- NULL
  for (i in 1:ncol(filter_mat))
    ccf_mat <- cbind(ccf_mat,
                     compute_acf_at_lags_zero_delta_func(
                       max_lag, h, filter_mat[, i], V[, 2])$cor_vec)
  colnames(ccf_mat) <- colnames(filter_mat)
  rownames(ccf_mat) <- paste("CCF at lead:", -max_lag - 1 + 1:nrow(ccf_mat))
  
  mplot <- ccf_mat
  
  plot(mplot[, 1],
       main = "CCF against V2", axes = FALSE, type = "l",
       xlab = "", ylab = "",
       col  = colo[1], lwd = 1,
       ylim = c(min(0, min(mplot)), max(mplot)))
  
  for (i in 1:ncol(mplot)) {
    lines(mplot[, i],
          col = colo[i],
          lwd = ifelse(colnames(mplot)[i] == "MSE", 2, 1),
          lty = ifelse(colnames(mplot)[i] == "MSE", 2, 1))
  }
  
  abline(v = 1 + h, lty = 2)
  abline(h = 0)
  
  axis(1, at = 1:nrow(mplot), labels = -1 + 1:nrow(mplot))
  axis(2)
  box()
  return(colo)
}


