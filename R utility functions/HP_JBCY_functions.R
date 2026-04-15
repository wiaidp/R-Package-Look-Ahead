# Data  function: loads data for BCA in paper

data_load_func<-function(path.data) 
{  
  
  indpro_mat<-read.csv(paste(path.data,"/indpro.csv",sep=""),sep=",",header=T,na.strings="NA",dec=".",row.names=1)
  
  indpro<-indpro_level<-NULL
  for (i in 1:ncol(indpro_mat))
  {
    indpro<-cbind(indpro,diff(log(indpro_mat[,i])))
    indpro_level<-cbind(indpro_level,indpro_mat[,i])
  }
  typeof(indpro)
  colnames(indpro)<-colnames(indpro_level)<-colnames(indpro_mat)
  rownames(indpro)<-rownames(indpro_mat)[2:nrow(indpro_mat)]
  rownames(indpro_level)<-rownames(indpro_mat)
  mean(indpro_mat[,1],na.rm=T)
  
  indpro_mat_eu<-read.csv(paste(path.data,"/indpro_eu_sa.csv",sep=""),sep=",",header=T,na.strings="NA",dec=".",row.names=1)
  
  indpro_eu<-NULL
  for (i in 1:ncol(indpro_mat_eu))
    indpro_eu<-cbind(indpro_eu,diff(log(as.double(indpro_mat_eu[,i]))))
  typeof(indpro_eu)
  tail(indpro_eu)
  colnames(indpro_eu)<-colnames(indpro_mat_eu)
  rownames(indpro_eu)<-rownames(indpro_mat_eu)[2:nrow(indpro_mat_eu)]
  
  indpro<-as.xts(indpro,order.by=as.Date(rownames(indpro),"%d/%m/%Y"))
  indpro_level<-as.xts(indpro_level,order.by=as.Date(rownames(indpro_level),"%d/%m/%Y"))
  return(list(indpro=indpro,indpro_level=indpro_level,indpro_eu=indpro_eu))
} 

#-------------------------------------------------------------------
# HP-functions

# Computes target, MSE, HP-trend and HP-gap original and modified: it is used in BCA-section of paper
HP_target_mse_modified_gap<-function(L,lambda_monthly)
{
  #   MSE relies on white noise assumption while HP-concurrent relies on implicit ARIMA(0,2,2) model
  #   L<-100 is OK i.e. recession datings are nearly identical with L<-200
  setseed<-1
  
  hp_obj<-hp_func(L,lambda_monthly,setseed)
  
  # Concurrent trend
  hp_trend<-hp_obj$concurrent
  ts.plot(hp_trend)
  # Concurrent gap
  hp_gap<-c(1-hp_trend[1],-hp_trend[2:L])
  ts.plot(hp_gap)
  # Modified concurrent gap (as applied to first differences)
  modified_hp_gap<-hp_gap
  for (i in 1:length(hp_gap))
  {
    modified_hp_gap[i]<-sum(hp_gap[1:i])
  }
  ts.plot(modified_hp_gap)
  # Symmetric target
  target<-hp_obj$target
  ts.plot(target)
  # One-sided MSE: must double length in order to retrieve right half of target
  L_target<-2*(L-1)+1
  hp_obj<-hp_func(L_target,lambda_monthly,setseed)
  target_long<-hp_obj$target
  hp_mse<-target_long[(1+(L_target-1)/2):L_target]
  ts.plot(hp_mse)
  return(list(hp_mse=hp_mse,hp_gap=hp_gap,modified_hp_gap=modified_hp_gap,hp_trend=hp_trend,target=target))
}


# Generic HP function relying on R-package. Computes holding-times according to formula in paper
hp_func<-function(L,lambda,setseed)
{
  
  set.seed(setseed)
  eps<-rnorm(L)
  
  hp_filt_obj<-hp_filt_obj <- hpfilter(eps,type="lambda", freq=lambda)
  
  gap_matrix<-hp_filt_obj$fmatrix
  # Extract the coefficients of the symmetric trend:
  #   hpfilter generates coefficients of the HP-gap (see below):
  #   we here transform back to trend filter
  parm_hp<-(diag(rep(1,L))-hp_filt_obj$fmatrix)
  target<-parm_hp[,(L-1)/2+1]
#  rho_ht_hp<-compute_holding_time_func(target)
#  ht_target<-rho_ht_hp$ht
  concurrent<-parm_hp[,1]
#  ht_concurrent<-compute_holding_time_func(concurrent)$ht
  b_mse<-target[((length(target)-1)/2+1):length(target)]
  if (length(b_mse)>L)
    b_mse<-b_mse[1:L]
  if (length(b_mse)<L)
    b_mse<-c(b_mse,rep(0,L-length(b_mse)))
  #  ts.plot(b_mse)
  return(list(target=target,concurrent=concurrent,b_mse=b_mse,gap_matrix=gap_matrix))
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





per<-function(x,plot_T)
{
  len<-length(x)
  per<-0:(len/2)
  DFT<-per
  
  for (k in 0:(len/2))
  {
    cexp <- exp(1.i*(1:len)*2*pi*k/len)
    DFT[k+1]<-sum(cexp*x*sqrt(1/(2*pi*len)))
  }
  # Frequency zero receives weight 1/sqrt(2)
  #   The periodogram in frequency zero appears once only whereas all other frequencies are doubled
  
  # This is omitted now in order to comply with MDFA
  #   We now change the periodogram in the dfa estimation routines
  #  DFT[1]<-DFT[1]/sqrt(2)
  # Weighths wk: if length of data sample is even then DFT in frequency pi is scaled by 1/sqrt(2) (Periodogram in pi is weighted by 1/2)
  if (abs(as.integer(len/2)-len/2)<0.1)
    DFT[k+1]<-DFT[k+1]/sqrt(2)
  per<-abs(DFT)^2
  if (plot_T)
  {
    par(mfrow=c(2,1))
    plot(per,type="l",axes=F,xlab="Frequency",ylab="Periodogram",
         main="Periodogram")
    axis(1,at=1+0:6*len/12,labels=c("0","pi/6","2pi/6","3pi/6",
                                    "4pi/6","5pi/6","pi"))
    axis(2)
    box()
    plot(log(per),type="l",axes=F,xlab="Frequency",ylab="Log-periodogram",
         main="Log-periodogram")
    axis(1,at=1+0:6*len/12,labels=c("0","pi/6","2pi/6","3pi/6",
                                    "4pi/6","5pi/6","pi"))
    axis(2)
    box()
  }
  return(list(DFT=DFT,per=per))
}




heat_map_func<-function(scale_column,select_acausal_target,MSE_mat,target_mat)
{ 
  # We can also rely on ggplot for drawing a heat map
  # For this purpose we have to specify a matrix with three columns corresponding to holding-time, forecast horizon and criterion values
  
  heat_mat_mse<-matrix(ncol=3,nrow=nrow(MSE_mat)*ncol(MSE_mat))
  heat_mat_target<-matrix(ncol=3,nrow=nrow(target_mat)*ncol(target_mat))
  
  range_mse<-range_aucausal_target<-NULL
  for (i in 1:ncol(MSE_mat))
  {
    if (scale_column)
    {  
      # Scaling along columns: emphasize ht effect better    
      heat_mat_mse[(i-1)*nrow(MSE_mat)+1:nrow(MSE_mat),]<-cbind(as.double(rownames(MSE_mat)),as.double(colnames(MSE_mat)[i]),scale(MSE_mat[,i]))
      heat_mat_target[(i-1)*nrow(MSE_mat)+1:nrow(MSE_mat),]<-cbind(as.double(rownames(target_mat)),as.double(colnames(target_mat)[i]),scale(target_mat[,i]))
      colnames(heat_mat_mse)<-colnames(heat_mat_target)<-c("Smoothness_holding_time","Timeliness_forecast_horizon","Scaled_criterion")
      
    } else
    {
      heat_mat_mse[(i-1)*nrow(MSE_mat)+1:nrow(MSE_mat),]<-cbind(as.double(rownames(MSE_mat)),as.double(colnames(MSE_mat)[i]),(MSE_mat[,i]))
      heat_mat_target[(i-1)*nrow(MSE_mat)+1:nrow(MSE_mat),]<-cbind(as.double(rownames(target_mat)),as.double(colnames(target_mat)[i]),(target_mat[,i]))
      colnames(heat_mat_mse)<-colnames(heat_mat_target)<-c("Smoothness_holding_time","Timeliness_forecast_horizon","Criterion")
      
    }
    range_mse<-c(range_mse,MSE_mat[,i])
    range_aucausal_target<-c(range_aucausal_target,target_mat[,i])
  }
  
  # Correlations of SSA with causal MSE are larger than with acausal (effective) target
  tail(heat_mat_mse)
  tail(heat_mat_target)
  
  # One can draw heat-map for Trilemma based on correlation with MSE or on correlation with target
  #   Select either one
  if (select_acausal_target)
  {
    heat_mat<-as.data.frame((heat_mat_target))
    range<-range_aucausal_target
  } else
  {
    heat_mat<-as.data.frame(heat_mat_mse)
    range<-range_mse
  }
  
  
  if (scale_column)
  {  
    if (select_acausal_target)
    {
      main<-"Correlations against acausal target: scaled values"
    } else
    {
      main<-"Correlations against causal MSE: scaled values"
    }
    ggplot(heat_mat , aes(x = Timeliness_forecast_horizon, y = Smoothness_holding_time),main=main) +
      geom_raster(aes(fill = Scaled_criterion), interpolate=TRUE) +
      scale_fill_gradient2(low="red",mid="yellow", high="black", 
                           midpoint=mean(heat_mat$Scaled_criterion), limits=range(heat_mat$Scaled_criterion)) +
      theme_classic()
  } else
  {
    if (select_acausal_target)
    {
      main<-"Correlations against acausal target"
    } else
    {
      main<-"Correlations against causal MSE"
    }
    
    ggplot(heat_mat , aes(x = Timeliness_forecast_horizon, y = Smoothness_holding_time),main=main) +
      geom_raster(aes(fill = Criterion), interpolate=TRUE) +
      scale_fill_gradient2(low="red", mid="yellow", high="black", 
                           midpoint=mean(heat_mat$Criterion), limits=range(heat_mat$Criterion)) +
      theme_classic()
    

  }
}


# This function shifts filter2 against filter1: for each lead or lag the function computes the sum of distances between crossings
# The function plots this sum as a function of lead/lag
# The minimum of the fuction (sum of timing-distances) indicates the lad or lag of filter2 relative to filter1 
compute_min_tau_func<-function(filter_mat,max_lead=6,vicinity=4,last_crossing_or_closest_crossing=F,outlier_limit=10)
{  
  
  #------------------------------------------------------------
  # Empirical lead/lag at zero-crossings
  # Skip all crossings with lead/lag>outlier_limit
  skip_larger<-outlier_limit
  # Index of series with more crossings: this is measured against the crossings of the reference series
  con_ind<-2
  # Index of reference series: this one has less crossings and shift is measured with reference to thse crossings only
  ref_ind<-1
  mean_shift_vec<-mean_shift_adjusted_vec<-NULL
  for (i in 1:max_lead)
  {
    shift_series<-cbind(filter_mat[i:nrow(filter_mat),1],filter_mat[1:(nrow(filter_mat)-i+1),2])
    
    
    lead_lag_cross_obj<-new_lead_at_crossing_func(ref_ind,con_ind,shift_series,last_crossing_or_closest_crossing,vicinity)
    
    tau_vec<-c(lead_lag_cross_obj$cum_ref_con[1],diff(lead_lag_cross_obj$cum_ref_con))
    remove_tp<-which(abs(tau_vec)>skip_larger)
    if (length(remove_tp)>0)
    {  
      tau_vec_adjusted<-tau_vec[-remove_tp]
    } else
    {
      print("No outlier adjustment for tau-statistic")    
      tau_vec_adjusted<-tau_vec
    }
    tau<-lead_lag_cross_obj$mean_lead_ref_con
    tau_adjusted<-mean(tau_vec_adjusted)
    mean_shift_vec<-c(mean_shift_vec,tau)
    mean_shift_adjusted_vec<-c(mean_shift_adjusted_vec,tau_adjusted)
  }
  # revert ordering  
  mean_shift_vec<-mean_shift_vec[max_lead:1]
  mean_shift_adjusted_vec<-mean_shift_adjusted_vec[max_lead:1]
  # Shift the other series  
  for (i in 2:max_lead)# i<-3
  {
    shift_series<-cbind(filter_mat[1:(nrow(filter_mat)-i+1),1],filter_mat[i:nrow(filter_mat),2])
    
    
    lead_lag_cross_obj<-new_lead_at_crossing_func(ref_ind,con_ind,shift_series,last_crossing_or_closest_crossing,vicinity)
    
    tau_vec<-c(lead_lag_cross_obj$cum_ref_con[1],diff(lead_lag_cross_obj$cum_ref_con))
    remove_tp<-which(abs(tau_vec)>skip_larger)
    if (length(remove_tp)>0)
    {  
      tau_vec_adjusted<-tau_vec[-remove_tp]
    } else
    {
      print("no outlier adjustment necessary in tau-statistic")    
      tau_vec_adjusted<-tau_vec
    }
    tau<-lead_lag_cross_obj$mean_lead_ref_con
    tau_adjusted<-mean(tau_vec_adjusted)
    mean_shift_vec<-c(mean_shift_vec,tau)
    mean_shift_adjusted_vec<-c(mean_shift_adjusted_vec,tau_adjusted)
  }
  # Revert time once again to conform with peak-cor plot: leads of reference filter (first column) correspond to troughs to the left  
  mean_shift_vec<-mean_shift_vec[length(mean_shift_vec):1]
  mean_shift_adjusted_vec<-mean_shift_adjusted_vec[length(mean_shift_adjusted_vec):1]
  if (F)
  {  
    par(mfrow=c(2,1))
    plot(abs(mean_shift_vec),col="blue",main="Min-tau shift",axes=F,type="l", xlab="Lead/lag",ylab="")
    abline(v=max_lead)
    at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
    axis(1,at=at_vec,labels=at_vec-max_lead)
    axis(2)
    box()
    plot(abs(mean_shift_adjusted_vec),col="blue",main="Min-tau adjusted shift",axes=F,type="l", xlab="Lead/lag",ylab="")
    abline(v=max_lead)
    at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
    axis(1,at=at_vec,labels=at_vec-max_lead)
    axis(2)
    box()
    
    
  }
  
  par(mfrow=c(1,1))
  plot(abs(mean_shift_adjusted_vec),col="blue",main="Min-tau adjusted shift",axes=F,type="l", xlab="Lead/lag",ylab="")
  abline(v=max_lead)
  at_vec<-c(1,max_lead/2,max_lead,3*max_lead/2,2*max_lead-1)
  axis(1,at=at_vec,labels=at_vec-max_lead)
  axis(2)
  box()
  
  
  min_tau_plot<-recordPlot()
  
  
  
  
  
  
  
  return(list(mean_shift_vec=mean_shift_vec,mean_shift_adjusted_vec=mean_shift_adjusted_vec,min_tau_plot=min_tau_plot))
  
}


# This function implements the mean-shift statistic in paper
#   It accounts for sign of crossings (up-swing/down-swing)
# There is a new additional feature when compared with earlier version above
#   1. If last_crossing_or_closest_crossing==F then it replicates the function used and described in paper
#   2. If last_crossing_or_closest_crossing==T then for each up- or down-turn of the reference filter 
#     one selects the last up- or down-turn of contender in a vicinity of crossing (otherwise the closest will be selected) 
# The setting last_crossing_or_closest_crossing==T is more realistic but still a bit optimistic because 
#   the user generally does not know that a particular crossing will be the last one.
# Results depend on vicinity which is the size of the neighborhood +/-vicinity about crossing
#   Can be chosen with a link to holding time: if holding time is large, then vicinity could be larger, too
# In general last_crossing_or_closest_crossing==F will lead to smaller lead-times (closest crossing) than
#   last_crossing_or_closest_crossing==T (latest crossing in vicinity)
# In the paper we use last_crossing_or_closest_crossing==F exclusively (closest crossings)
new_lead_at_crossing_func<-function(ref_ind,con_ind,filter_mat,last_crossing_or_closest_crossing,vicinity)
{
  ref_cross<-which(sign(filter_mat[1:(nrow(filter_mat)-1),ref_ind])!=sign(filter_mat[2:(nrow(filter_mat)),ref_ind]))
  con_cross<-which(sign(filter_mat[1:(nrow(filter_mat)-1),con_ind])!=sign(filter_mat[2:(nrow(filter_mat)),con_ind]))
  if (filter_mat[ref_cross[1],ref_ind]<0)
  {
    ref_cross_sign<-(-1)^(0:(length(ref_cross)-1))*ref_cross  
  } else
  {
    ref_cross_sign<--1*((-1)^(0:(length(ref_cross)-1))*ref_cross)
  }
  if (filter_mat[con_cross[1],con_ind]<0)
  {
    con_cross_sign<-(-1)^(0:(length(con_cross)-1))*con_cross  
  } else
  {
    con_cross_sign<--1*((-1)^(0:(length(con_cross)-1))*con_cross)
  }
  # Crossings from above and from below  
  con_cross_sign_plus<-con_cross_sign[con_cross_sign>0]
  con_cross_sign_negative<-con_cross_sign[con_cross_sign<0]
  
  disc_vec<-NULL
  # Check if there are crossings    
  if (length(ref_cross)>0&length(con_cross)>0)
  {
    
    
    ref_len<-length(ref_cross)#length(con_cross)
    
    # For the j-th zero-crossing of reference
    for (j in 1:ref_len)#  j=1
    {
      if (ref_cross_sign[j]>0)
      {
        # Up-turns      
        if (!last_crossing_or_closest_crossing)
        {    
          # For upturns of reference: select nearest upturn of contender        
          min_i<-min(abs(abs(con_cross_sign_plus)-abs(ref_cross[j])))
          which_min<-which(abs(abs(con_cross_sign_plus)-abs(ref_cross[j]))==min_i)
          disc_vec<-c(disc_vec,max(abs(con_cross_sign_plus[which_min]))-abs(ref_cross[j]))
          #          disc_vec<-c(disc_vec,min_i)
        } else
        {
          # For upturns of reference: select latest upturn of contender in vicinity of reference up-turn
          # This is closer to real-time application though it is still optimistic because the user doesn't know 
          #   that this will be the last up-turn in practice      
          # 1 All upturns in vicinity of reference up-turn        
          rert<-which(abs(abs(con_cross_sign_plus)-abs(ref_cross[j]))<vicinity)
          # 2. Select latest one or border of vicinity
          if (length(rert)>0)
          {  
            # If there is a TP in vicinity: select last one            
            max_rert<-max(rert)
            last_tp<-abs(con_cross_sign_plus[max_rert])
          } else
          {
            # Otherwise select border of vicinity (one again a bit optimistic)            
            last_tp<-abs(ref_cross[j])+vicinity
            last_tp<-abs(ref_cross[j])
          }
          disc_vec<-c(disc_vec,last_tp-abs(ref_cross[j]))
          
        }  
      } else
      {
        # For downturns      
        if (!last_crossing_or_closest_crossing)
        {  
          # For downturns of reference: select nearest downturn of contender        
          min_i<-min(abs(abs(con_cross_sign_negative)-abs(ref_cross[j])))
          which_min<-which(abs(abs(con_cross_sign_negative)-abs(ref_cross[j]))==min_i)
          # Time-difference at down-turn: 
          #   in case of two possible nearest crossings we select the max (because the contrary direction is on at the time point of the crossing of the reference filter)
          disc_vec<-c(disc_vec,max(abs(con_cross_sign_negative[which_min]))-abs(ref_cross[j]))
          #          disc_vec<-c(disc_vec,min_i)
          
        } else
        {
          # For down-turns of reference: select latest down-turn of contender in vicinity of reference down-turn
          # This is closer to real-time application though it is still optimistic because the user doesn't know 
          #   that this will be the last down-turn in practice      
          # 1 All down-pturns in vicinity of reference down-turn        
          rert<-which(abs(abs(con_cross_sign_negative)-abs(ref_cross[j]))<vicinity)
          # 2. Select latest one or border of vicinity
          if (length(rert)>0)
          {  
            # If there is a TP in vicinity: select last one            
            max_rert<-max(rert)
            last_tp<-abs(con_cross_sign_negative[max_rert])
          } else
          {
            # Otherwise select border of vicinity (one again a bit optimistic)            
            last_tp<-abs(ref_cross[j])+vicinity
            last_tp<-abs(ref_cross[j])
          }
          disc_vec<-c(disc_vec,last_tp-abs(ref_cross[j]))
        }  
      }
    }
    # Total number of crossings of contender and of reference
    number_crossings_per_sample<-c(length(con_cross),length(ref_cross))
  } else
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("no zero-crossings observed")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    number_crossings_per_sample<-rbind(number_crossings_per_sample,c(0,0))
  }
  if (length(names(ref_cross))>0)
    names(disc_vec)<-names(ref_cross)
  number_crossings_per_sample<-t(as.matrix(number_crossings_per_sample,nrow=1))
  cum_ref_con<-cumsum(na.exclude(disc_vec))
  #  ts.plot(cum_ref_con)
  mean_lead_ref_con<-cum_ref_con[length(cum_ref_con)]/length(cum_ref_con)
  mean_lead_ref_con
  tail(cum_ref_con)
  return(list(cum_ref_con=cum_ref_con,mean_lead_ref_con=mean_lead_ref_con,number_crossings_per_sample=number_crossings_per_sample,ref_cross=ref_cross))
}



