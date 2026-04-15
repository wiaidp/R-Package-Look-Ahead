# ════════════════════════════════════════════════════════════════════
# TUTORIAL 2 — TIMELINESS: A MEASURE FOR LEAD 
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# Measuring timeliness or lead of a predictor poses several challenges:
# -Simple interpretable measure
# -Absolute vs. relative measure
# -Actionable measure: can it be deployed to refine or design predictors
# -


# time-domain Tau statistic, peak correlation; frequency-domain shift

# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND ────────────────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     a Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5

#   Wildi, M. (2026). Forecasting on the Accuracy–Timeliness Frontier:
#   Two Novel "Look-Ahead" Predictors.
#   https://doi.org/10.48550/arXiv.2602.23087


# ════════════════════════════════════════════════════════════════════


rm(list = ls())



# Load tau-statistic (measures lead/lag performance)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction functions used in JBCY (requires mFilter)
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

#----------------------------------------------------------------------------
# Exercise 1: Data Samples

# 1.1 Generate Series With General (Non-Integer) Lead/Lag

len<-120
periodicity<-3*12
omega<-2*pi/periodicity
shift<-5
A1<-A2<-1
sigma1<-1
sigma2<-1
set.seed(36)

eps1<-sigma1*rnorm(len)
eps2<-sigma2*rnorm(len)

x<-A1*cos((1:len)*omega)+eps1
y<-A2*cos((shift+1:len)*omega)+eps2

par(mfrow=c(1,1))
ts.plot(cbind(x,y))
#-------------------------------------------------------------------------

# 1.2 Cross Correlation Function (CCF)
ccf(x, y, lag.max = 10, plot = TRUE)



#-------------------------------------------------------------------------

# 1.3 Dynamic regression with distributed lags

x_ts <- ts(x, start = 1, frequency = 1)
y_ts <- ts(y, start = 1, frequency = 1)
model <- dynlm(x_ts ~ L(y_ts, 0:8))   # lags 0 through 8
summary(model)

#-------------------------------------------------------------------------
# 1.4 Lead/lag at zero crossings (see Wildi 2024)
filter_mat<-cbind(x,y)
max_lead<-10
compute_min_tau_func(filter_mat,max_lead)
  

#-------------------------------------------------------------------------
# 1.5 Frequency Domain: Coherence and Phase
library(astsa)


# ── Compute cross-spectral analysis ──────────────────────────────────────────
# spans: smoothing spans for the periodogram (reduces noise)
# taper: proportion of data tapered at each end (reduces leakage)
# plot:  TRUE produces a 4-panel plot automatically

spec_result <- mvspec(
  x       = cbind(x, y),
  spans   = c(7, 7),       # smoothing: adjust for more/less smoothing
  taper   = 0.1,           # 10% tapering at each end
  plot    = TRUE           # produces spectrum, coherence, and phase plots
)

# ── Frequencies and spectra ───────────────────────────────────────────────────
freq      <- spec_result$freq          # frequencies (cycles per time unit)
period    <- 1 / freq                  # corresponding periods

# Individual spectra
spec_x    <- spec_result$fxx[1, 1, ]  # spectrum of x (real part)
spec_y    <- spec_result$fxx[2, 2, ]  # spectrum of y (real part)

# Cross-spectrum
cross_xy  <- spec_result$fxx[1, 2, ]  # complex cross-spectrum x vs y

# ── Coherence: strength of co-movement at each frequency ─────────────────────
coherence <- Mod(cross_xy)^2 / (Re(spec_x) * Re(spec_y))
coherence <- pmin(coherence, 1)        # clamp to [0, 1] for numerical safety

# ── Phase: lead/lag at each frequency ────────────────────────────────────────
phase     <- Arg(cross_xy)             # phase in radians

# Convert phase to time lag at each frequency
# time_lag > 0: x leads y  |  time_lag < 0: y leads x
time_lag  <- phase / (2 * pi * freq)

# ── Find dominant frequency (peak in spectrum of x) ──────────────────────────
peak_idx  <- which.max(Re(spec_x))
cat("Dominant frequency :", round(freq[peak_idx], 4),   "\n")
cat("Dominant period    :", round(period[peak_idx], 2),  "\n")
cat("Phase at peak (rad):", round(phase[peak_idx], 4),   "\n")
cat("Estimated time lag :", round(time_lag[peak_idx], 2), "periods\n")
# Expected: close to shift

#---------------------------------------------------------------------------
if (F)
{
# Other: Non-Linear
  library(RTransferEntropy)

# Estimate transfer entropy in both directions
  te_result <- transfer_entropy(x, y, lx = 2*shift, ly = 2*shift, nboot = 100)
  print(te_result)

# Cross-spectral analysis: coherence and phase
  mvspec(cbind(x, y), spans = c(5, 5), plot = TRUE)
}


#----------------------------------------------------------------------------
# Exercise 2: Filters







