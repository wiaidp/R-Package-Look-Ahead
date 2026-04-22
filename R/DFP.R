unitary_DFP_func<-function(gamma0,gammah,alpha0)
{
  if (abs(abs(gamma0%*%gammah)-sqrt(sum(gamma0^2)*sum(gammah^2)))<10^{-10})
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  if (abs(alpha0)>1)
  {
    print("|alpha0| must be smaller one: it is a correlation!")
    return()
  }
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




# Compute MSE DFP
compute_mse_dfp<-function(alpha0,gamma0,gammah,plot_T=F)
{  
  if (abs(abs(gamma0%*%gammah)-sqrt(sum(gamma0^2)*sum(gammah^2)))<10^{-10})
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  L<-length(gamma0)
  B<-rbind(-gamma0[2:L]/gamma0[1],diag(rep(1,L-1)))
  alpha0_vec<-c(alpha0/gamma0[1],rep(0,L-1))
  
  b<-solve(t(B)%*%B)%*%t(B)%*%(gammah-alpha0_vec)
  
  b0<-alpha0_vec+B%*%b
  if (plot_T)
    ts.plot(b0)
  
  # check: should vanish
  t(b0)%*%gamma0-alpha0
  return(list(b0=b0))
}


mse_dfp_from_tau_func<-function(gamma0,gammah,lead)
{
  if (abs(abs(gamma0%*%gammah)-sqrt(sum(gamma0^2)*sum(gammah^2)))<10^{-15})
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  # Compute shifts at frequency zero
  tau0<-sum((0:(L-1))*gamma0)/sum(gamma0)
  tauh<-sum((0:(L-1))*gammah)/sum(gammah)

  tau<-lead
  # Formula for lambda0
  lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
  b<-gammah+lambda0*gamma0
  if (tauh>tau0)
  {
    print("Non-standard case: tauh>tau0")
    print("The MSE predictors lags the nowcast at frequency zero")
    print("This requires inversion of the DFP solution")
    print("b must lie on side of gamma0 opposite to gammah")
    print("Equivalently, we minimize the target correlation")
    print("This means b=-gammah+lambda0*gamma0")
    print("The sign of gammah is inverted in the non-standard case")
    lambda0<--lambda0
    b<--gammah+lambda0*gamma0
  }

  if (b%*%gammah<0)
  {
    print("Warning: the target correlation is negative")
#    lambda0<--lambda0
#    b<--b
  }
  
  return(list(tau0=tau0,tauh=tauh,lambda0=lambda0,b=b))
}



mse_dfp_from_alpha0_func<-function(gamma0,gammah,alpha0)
{
  if (abs(abs(gamma0%*%gammah)-sqrt(sum(gamma0^2)*sum(gammah^2)))<10^{-10})
  {
    print("Warning: gammah and gamma0 are nearly collinear: the DFP predictor is not computed")
    return()
  }
  
  lambda<-as.double((alpha0-t(gamma0)%*%gammah)/(t(gamma0)%*%gamma0))
  b<-gammah+lambda*gamma0
  return(list(lambda=lambda,b=b))
}



compute_alpha_0_func<-function(gamma0,gammah,lambda0)
{
  alpha0<-as.double(t(gamma0)%*%(gammah+lambda0*gamma0)/sqrt(t(gammah+lambda0*gamma0)%*%(gammah+lambda0*gamma0)))
  return(list(alpha0=alpha0))
}
