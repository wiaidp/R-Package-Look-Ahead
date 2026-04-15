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




# Compute MSE DFP
compute_mse_dfp<-function(alpha0,gamma0,gammah,plot_T=F)
{
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


dfp_from_tau_func<-function(gamma0,gammah,lead)
{
  # Compute shifts at frequency zero
  tau0<-sum((0:(L-1))*gamma0)/sum(gamma0)
  tauh<-sum((0:(L-1))*gammah)/sum(gammah)
  tau<-lead
  # Formula for lambda0
  lambda0<--(tau*sum(gammah))/((tau+tauh-tau0)*sum(gamma0))
  b<-gammah+lambda0*gamma0
  
  return(list(tau0=tau0,tauh=tauh,lambda0=lambda0,b=b))
}



dfp_from_alpha0_func<-function(gamma0,gammah,alpha0)
{
  lambda<-as.double((alpha0-t(gamma0)%*%gammah)/(t(gamma0)%*%gamma0))
  b<-gammah+lambda*gamma0
  return(list(lambda=lambda,b=b))
}



compute_alpha_0_func<-function(gamma0,gammah,lambda0)
{
  alpha0<-as.double(t(gamma0)%*%(gammah+lambda0*gamma0)/sqrt(t(gammah+lambda0*gamma0)%*%(gammah+lambda0*gamma0)))
  return(list(alpha0=alpha0))
}
