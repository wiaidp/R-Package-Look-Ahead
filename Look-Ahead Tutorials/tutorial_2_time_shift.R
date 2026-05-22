# ════════════════════════════════════════════════════════════════════
# TUTORIAL 2 — TIMELINESS: A MEASURE FOR LEAD
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────────
# This tutorial explores how to measure the timeliness (lead/lag) of a
# predictor relative to a target series. Key challenges addressed include:
#
#   - Constructing simple, interpretable timeliness measures
#   - Distinguishing between absolute and relative measures
#   - Deriving exact (sample-independent) measures
#   - Producing actionable measures that guide predictor refinement or design
#
# ── TIMELINESS MEASURES FOR TIME SERIES ──────────────────────────────
#   1. Sample Cross-Correlation Function (CCF)
#        Estimates the lag at which two series are most linearly correlated.
#        Simple and widely used: sample CCFs are potentially sensitive to 
#        sample noise.
#
#   2. Dynamic regression with distributed lags
#        Regresses the target on current and lagged values of the predictor.
#        The lag with the most significant coefficient estimates the lead.
#
#   3. Lead/lag at zero crossings (Tau statistic)
#        Counts how consistently one series crosses zero before the other.
#        Requires a zero-mean stationary series; see Wildi (2024).
#
#   4. Phase coherence (frequency domain)
#        Decomposes the lead/lag relationship by frequency via cross-spectral
#        analysis. Coherence measures co-movement strength; phase converted
#        to time units gives the frequency-specific lag.
#
# ── TIMELINESS MEASURES FOR FILTERS ──────────────────────────────────
#   5. True/Expected CCF
#        Cross-correlates two filter coefficient vectors to estimate their
#        relative timing. Measure the aggregate timeliness effect and are 
#        coarser than frequency-domain measures.
#
#   6. Amplitude and time-shift functions
#        Describe, at every frequency, how a filter scales (amplitude/gain)
#        and delays (time-shift/phase delay) a sinusoidal input. Together
#        they provide a complete, exact, sample-independent characterisation
#        of filter behaviour.
#
# ── VALIDATION ────────────────────────────────────────────────────────
# All measures are applied to simulated data where the true shift is known,
# allowing direct verification of each method's accuracy and reliability.
# ════════════════════════════════════════════════════════════════════

# ── BACKGROUND / REFERENCES ───────────────────────────────────────────
#   Wildi, M. (2024)
#     Business Cycle Analysis and Zero-Crossings of Time Series:
#     A Generalized Forecast Approach.
#     https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026)
#     Forecasting on the Accuracy–Timeliness Frontier:
#     Two Novel "Look-Ahead" Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
# ════════════════════════════════════════════════════════════════════

# ── INITIALISATION ────────────────────────────────────────────────────
# Clear the workspace to ensure a clean environment before execution
rm(list = ls())

# Load the tau-statistic utility
# Provides compute_min_tau_func(): measures lead/lag at zero crossings
# of a zero-mean stationary series (Wildi 2024)
source(paste(getwd(), "/R utility functions/Tau_statistic.r", sep = ""))

# Load signal extraction and filter utility functions from the 2024 paper
# Provides amp_shift_func(), compute_ccf_func(), and related helpers
# Dependency: the mFilter package must be installed
source(paste(getwd(), "/R utility functions/DFP_PCS_utility_functions.r", sep = ""))

library(mFilter)

# ════════════════════════════════════════════════════════════════════
# EXERCISE 1: Data Samples — Lead/Lag Measures for Time Series
# ════════════════════════════════════════════════════════════════════

# ── 1.1 Simulate Two Series with a Known (Non-Integer) Lead/Lag ───────
# Both x and y are cosine waves with the same frequency (omega), but y is
# phase-shifted forward by `shift` time units, so y LEADS x by `shift`.
# Independent Gaussian noise is added to each series.

len         <- 120            # number of observations
periodicity <- 3 * 12         # cycle length in time units (36-month business cycle)
omega       <- 2 * pi / periodicity  # angular frequency (radians per time unit)
shift       <- 5              # true lead of y over x (in time units)
A1 <- A2    <- 1              # equal amplitudes for x and y
sigma1      <- 1              # noise standard deviation for x
sigma2      <- 1              # noise standard deviation for y

set.seed(36)                  # fix seed for reproducibility
eps1 <- sigma1 * rnorm(len)   # noise component for x
eps2 <- sigma2 * rnorm(len)   # noise component for y

x <- A1 * cos((1:len) * omega) + eps1               # target series (lagging)
y <- A2 * cos((shift + 1:len) * omega) + eps2       # predictor series (leading)

# Visual inspection of the two series
par(mfrow = c(1, 1))
ts.plot(cbind(x, y), main = "Simulated series: x (black) and y (red)",
        col = c("black", "red"))


# ── 1.2 Sample Cross-Correlation Function (CCF) ──────────────────────────────
# The CCF measures the linear correlation between x and y at various lags.
# The lag at which the CCF peaks provides a simple estimate of the lead/lag.
# Expected peak: near lag = shift (y leads x by `shift` units).
ccf(x, y, lag.max = 10, plot = TRUE,
    main = "CCF of x and y — peak lag estimates the lead")
# Result: the peak CCF confirms shift=5. But the data is noisy (tuning noise 
# through A1/sigma1 and A2/sigma2: signal noise ratios)

# ── 1.3 Dynamic Regression with Distributed Lags ──────────────────────
# Regress x on current and lagged values of y (lags 0 through 8).
# The lag with the largest (most significant) coefficient indicates
# how far back in y is most informative for x — i.e., the lead of y.
library(dynlm)
x_ts  <- ts(x, start = 1, frequency = 1)
y_ts  <- ts(y, start = 1, frequency = 1)

# Fit model: x_t = alpha + sum_{k=0}^{8} beta_k * y_{t-k} + error
model <- dynlm(x_ts ~ L(y_ts, 0:8))
summary(model)   # inspect coefficient significance across lags
# The most significant lag confirms shift=5.


# ── 1.4 Lead/Lag at Zero Crossings (Tau Statistic) ────────────────────
# The tau statistic counts the mean lead or lag at the zero crossings.
# See Wildi (2024) for theoretical background.
# `max_lead` sets the maximum lead horizon to test.
filter_mat <- cbind(x, y)
max_lead   <- 10
tau<-compute_min_tau_func(filter_mat, max_lead)
# The measure is noisy (it looks only at zero crossings). We see dips 
# at -4 and -8.

# ── 1.5 Frequency-Domain: Coherence and Phase ─────────────────────────
# Cross-spectral analysis decomposes the lead/lag relationship by frequency.
#   - Coherence: measures the strength of co-movement at each frequency
#                (analogous to R² in frequency domain; ranges from 0 to 1)
#   - Phase:     measures the phase offset between x and y at each frequency;
#                converting phase to time units gives the frequency-specific lag
library(astsa)

# Compute cross-spectral quantities using smoothed periodogram
#   spans : smoothing bandwidths applied to the periodogram (reduces noise)
#   taper : proportion of data tapered at each end (reduces spectral leakage)
#   plot  : automatically produces spectrum, coherence, and phase panels
spec_result <- mvspec(
  x     = cbind(x, y),
  spans = c(7, 7),     # adjust spans for more (larger) or less (smaller) smoothing
  taper = 0.1,         # 10% cosine taper applied to each end of the series
  plot  = TRUE         # display the four-panel cross-spectral plot
)

# ── Extract frequencies and spectral components ───────────────────────
freq     <- spec_result$freq           # frequencies in cycles per time unit
period   <- 1 / freq                   # corresponding cycle lengths (time units)

spec_x   <- spec_result$fxx[1, 1, ]   # power spectrum of x (real part)
spec_y   <- spec_result$fxx[2, 2, ]   # power spectrum of y (real part)
cross_xy <- spec_result$fxx[1, 2, ]   # complex cross-spectrum of x vs. y

# ── Coherence: fraction of variance in x explained by y at each frequency ──
coherence <- Mod(cross_xy)^2 / (Re(spec_x) * Re(spec_y))
coherence <- pmin(coherence, 1)        # clamp to [0, 1] to guard against numerical noise

# ── Phase: angular lead/lag between x and y at each frequency ────────
phase    <- Arg(cross_xy)              # phase difference in radians

# Convert phase (radians) to time units using:  time_lag = phase / (2π × freq)
# Interpretation: time_lag > 0  →  x leads y
#                 time_lag < 0  →  y leads x
time_lag <- phase / (2 * pi * freq)

# ── Report results at the dominant frequency (spectral peak of x) ─────
peak_idx <- which.max(Re(spec_x))
cat("Dominant frequency :", round(freq[peak_idx],     4), "\n")
cat("Dominant period    :", round(period[peak_idx],   2), "\n")
cat("Phase at peak (rad):", round(phase[peak_idx],    4), "\n")
cat("Estimated time lag :", round(time_lag[peak_idx], 2), "time units\n")
# Expected result: estimated time lag is close to the true `shift` (= 5)


# ── 1.6 Optional: Non-Linear Transfer Entropy (skipped by default) ────
if (FALSE) {
  # Transfer entropy quantifies directional information flow between x and y
  # beyond linear cross-correlation, useful when relationships are non-linear.
  library(RTransferEntropy)
  
  # lx / ly: embedding lags for x and y (set to twice the known shift)
  te_result <- transfer_entropy(x, y, lx = 2 * shift, ly = 2 * shift, nboot = 100)
  print(te_result)
  
  # Cross-spectral analysis (alternative quick call)
  mvspec(cbind(x, y), spans = c(5, 5), plot = TRUE)
}


# ════════════════════════════════════════════════════════════════════
# EXERCISE 2: Filters — Simple Unit Lag
# Compare a filter with a one-period lag against its unlagged counterpart.
# ════════════════════════════════════════════════════════════════════

# ── 2.1 Define Filters ────────────────────────────────────────────────
# Both filters are length-L exponentially weighted moving averages (EWMA),
# but b1 has its weights shifted by `shift` positions (introducing lag).
L     <- 10    # filter length (number of coefficients)
a1    <- 0.8   # geometric decay factor for the exponential weights
shift <- 1     # number of positions by which b1 is delayed relative to b2

# b1: EWMA with a one-period lag (first `shift` coefficients are zero)
b1 <- c(rep(0, shift), a1^(0:(L - 1 - shift)) / sum(a1^(0:(L - 1 - shift))))

# b2: Standard EWMA (no lag), used as the reference filter
b2 <- a1^(0:(L - 1)) / sum(a1^(0:(L - 1)))
# Need to set last lag to zero:
b2[L]<-0
ts.plot(cbind(b1, b2),xlab="",ylab="",
        main = "Filter coefficients: lagged (b1: blue) vs. unlagged (b2: red) EWMA",col=c("blue","red"))
h <- 5   # forecast horizon parameter passed to CCF utility


# ── 2.2 True/Expected (Filter) CCF ────────────────────────────────────────────────────
# Compute the true (expected) cross-correlation between the two filter (predictor) 
# coefficient vectors.
# The lag at the CCF peak estimates the relative lag of b1 with respect to b2.
# Expected peak: at lag = shift (= 1), confirming the one-period offset.
max_lead <- 5
ccf<-compute_ccf_func(b2, b1 )
plot(ccf,main="True/Expected CCF: Peaks at lag = 1", type = "l", axes = FALSE,
     xlab = "Lag", ylab = "CCF")
axis(1, at = 1:length(ccf),
     labels = names(ccf))
axis(2)
abline(v=which(ccf==max(ccf)),lty=2)
box()




# ── 2.3 Time-Shift Function ───────────────────────────────────────────
# The time-shift function shows the phase delay (in time units) introduced
# by each filter as a function of frequency.
# A flat, positive time-shift indicates a constant lag across all frequencies
K      <- 600      # number of frequency grid points
plot_T <- FALSE    # suppress internal plotting; we build a custom plot below

as_obj1 <- amp_shift_func(K, b1, plot_T)   # time-shift for lagged filter (b1)
as_obj2 <- amp_shift_func(K, b2, plot_T)   # time-shift for reference filter (b2)

# Plot time-shift functions for both filters across frequencies [0, π]
par(mfrow = c(1, 1))
colo  <- c("blue", "red")
mplot <- cbind(as_obj1$shift, as_obj2$shift)
colnames(mplot) <- c("Lagged (b1)", "Unlagged (b2)")

plot(mplot[, 1], type = "l", axes = FALSE,
     xlab = "Frequency", ylab = "Time shift (periods)",
     main = "Time-shift function: lagged vs. unlagged EWMA filter",
     ylim = c(min(mplot), max(mplot)), col = colo[1])
mtext(colnames(mplot)[1], line = -1, col = colo[1])

for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colo[i])
  mtext(colnames(mplot)[i], col = colo[i], line = -i)
}
# Label frequency axis from 0 to π in sixths
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()
# We can see that the time-shift difference is consistently one (shift=1) 
# at all frequencies.


# ════════════════════════════════════════════════════════════════════
# EXERCISE 3: Filters — Equally Weighted vs. Exponentially Weighted MA
# Compares a simple moving average (equal weights) to an EWMA.
# Equal weights assign uniform importance to all past observations;
# For roughly similar (but not identical) effects, the EWMA down-weights older 
# observations geometrically, making it potentially more responsive to recent 
# data and less lagging on average.
# ════════════════════════════════════════════════════════════════════

# ── 3.1 Define Filters ────────────────────────────────────────────────
L  <- 10    # filter length
a1 <- 0.8   # geometric decay factor for the EWMA
# Note: L and a1 should be `aligned' so that their filters produce roughly 
# similar effects.

# b1: equally weighted MA — each of the L observations has weight 1/L
b1 <- rep(1 / L, L)

# b2: exponentially weighted MA — weights decay geometrically from lag 0
b2 <- a1^(0:(L - 1)) / sum(a1^(0:(L - 1)))

ts.plot(cbind(b1, b2),xlab="",ylab="",
        main = "Filter coefficients: equal weights (b1) vs. EWMA (b2)")


# ── 3.2 Filter CCF ────────────────────────────────────────────────────
# Cross-correlate the two filter coefficient vectors to compare their
# relative timing. Because both filters are centred at the same horizon 
# (neither introduces an artificial lag), and both generate similar lowpass 
# behavior (a1 and L are roughly aligned), we expect the CCF peak at lag 0,
# indicating the two filters are contemporaneously aligned.
max_lag <- 5
# Note: CCF peaks at lag 0 — both filters respond at the same time point,
#       though they differ in how they weight past observations.
max_lead <- 5
ccf<-compute_ccf_func(b2, b1 )
plot(ccf,main="True/Expected CCF: Peaks at lag = 0", type = "l", axes = FALSE,
     xlab = "Lag", ylab = "CCF")
axis(1, at = 1:length(ccf),
     labels = names(ccf))
axis(2)
abline(v=which(ccf==max(ccf)),lty=2)
box()


#----------------------------------------------------------------------------
# 3.3 Amplitude and Time-Shift Functions
#----------------------------------------------------------------------------
# For a linear filter applied to a sinusoid at frequency omega, the output is
# again a sinusoid at the same frequency, but scaled by the filter's AMPLITUDE
# (gain) and delayed by the filter's TIME-SHIFT (phase delay).
#
# Plotting both functions across all frequencies [0, π] allows a direct
# comparison of how each filter distorts the magnitude and timing of signals
# at every frequency — a more informative picture than a single CCF lag.

K      <- 600      # number of frequency grid points spanning [0, π]
plot_T <- FALSE    # suppress internal plotting; custom plots are built below

# Compute amplitude and time-shift for each filter across the frequency grid
as_obj1 <- amp_shift_func(K, b1, plot_T)   # equally-weighted MA
as_obj2 <- amp_shift_func(K, b2, plot_T)   # exponentially-weighted MA

par(mfrow = c(2, 1))

# ── Amplitude plot ────────────────────────────────────────────────────────────
# Amplitude (gain) at each frequency: values close to 1 mean the filter passes
# that frequency with little attenuation; values near 0 indicate suppression.
colos <- c("blue", "red")
mplot <- cbind(as_obj1$amp, as_obj2$amp)
colnames(mplot) <- c("Equally weighted", "Exponentially weighted")

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Amplitude (Gain) Function",
     ylim = c(min(mplot), max(mplot)), col = colos[1])
mtext(colnames(mplot)[1], line = -1, col = colos[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colos[i])
  mtext(colnames(mplot)[i], col = colos[i], line = -i)
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# ── Time-shift plot ────────────────────────────────────────────────────────────
# Time-shift (phase delay in time units) at each frequency:
# larger values indicate greater lag introduced by the filter.
# The EWMA has a smaller shift magnitude than the equally-weighted MA because
# (for similar effects) it concentrates more weight on recent observations, 
# making it potentially more responsive.
mplot <- cbind(as_obj1$shift, as_obj2$shift)
colnames(mplot) <- c("Equally weighted", "Exponentially weighted")

plot(mplot[, 1], type = "l", axes = FALSE, xlab = "Frequency", ylab = "",
     main = "Time-Shift Function",
     ylim = c(min(mplot), max(mplot)), col = colos[1])
#mtext(colnames(mplot)[1], line = -1, col = colos[1])
for (i in 2:ncol(mplot)) {
  lines(mplot[, i], col = colos[i])
  # Legend labels omitted here to avoid overlap; colours identify each filter
}
axis(1, at = 1 + 0:6 * K / 6,
     labels = expression(0, pi/6, 2*pi/6, 3*pi/6, 4*pi/6, 5*pi/6, pi))
axis(2)
box()

# Key takeaway:
# The EWMA's time-shift is uniformly smaller in magnitude than that of the
# equally-weighted MA, meaning it introduces less lag at every frequency.
# This makes the EWMA possibly (though not uniformly so) a better choice when 
# timeliness is a priority.


#----------------------------------------------------------------------------
# 3.3.1 Verification on a Sinusoidal Input
#----------------------------------------------------------------------------
# Theory: applying a filter to cos(omega * t) yields amp * cos(omega * (t - shift)).
# Here we verify this identity numerically by:
#   1. Filtering the raw sinusoid with b1 and b2.
#   2. Constructing the predicted output using the amplitude and shift values
#      read from the frequency-grid objects above.
# The two curves should overlap almost perfectly.

# Select a frequency index on the grid (must be between 1 and K)
K_select <- 12
if (K_select > K) {
  print(paste("K_select must be smaller than", K))
}

# Corresponding angular frequency (radians per time unit)
omega <- K_select * pi / K

# Generate a pure cosine at that frequency
len <- 120
x   <- cos(omega * (1:len))

par(mfrow = c(1, 1))
ts.plot(x, main = paste0("Input sinusoid at frequency ", round(omega, 4), " rad"))

# Read amplitude and time-shift for each filter at this specific frequency
# (index K_select + 1 because the grid is stored with index 1 corresponding to omega = 0)
shift1 <- as_obj1$shift[K_select + 1]
amp1   <- as_obj1$amp[K_select + 1]
shift2 <- as_obj2$shift[K_select + 1]
amp2   <- as_obj2$amp[K_select + 1]

# Apply filters to x (side = 1 uses only past values, i.e., one-sided / causal filter)
y1 <- filter(x, b1, side = 1)
y2 <- filter(x, b2, side = 1)

# Construct analytical predictions: scaled and time-shifted cosines
# These should coincide with the filtered output once the filter has warmed up
x_shift_amp1 <- amp1 * cos(omega * (-shift1 + 1:len))
x_shift_amp2 <- amp2 * cos(omega * (-shift2 + 1:len))

par(mfrow = c(2, 1))
ts.plot(cbind(x_shift_amp1, y1),
        main = "Equally-weighted MA: analytical prediction vs. filtered output",
        col  = c("blue", "black"))
ts.plot(cbind(x_shift_amp2, y2),
        main = "Exponentially-weighted MA: analytical prediction vs. filtered output",
        col  = c("red", "black"))

# Conclusion:
# The analytical prediction (amplitude × shifted cosine) overlaps exactly with
# the filtered output — confirming that amplitude and time-shift fully characterise
# a linear filter's effect on any sinusoidal (and hence, by superposition, any
# stationary) input.
# Extension to non-stationary series: see McElroy & Wildi.


#----------------------------------------------------------------------------
# 3.3.2 Extension to a Trend Input (Zero Frequency)
#----------------------------------------------------------------------------
# At omega = 0 the formula shift = Phi(omega) / omega is indeterminate (0/0).
# Applying l'Hôpital's rule (first-order Taylor expansion of the phase around 0)
# yields a well-defined limit: the zero-frequency time-shift equals the
# weighted mean lag of the filter coefficients (see Wildi 2026a):
#
#   shift_0 = sum_{k=0}^{L-1} k * b_k
#
# This is also the centroid (centre of gravity) of the filter's impulse response.

shift1_trend <- as.double((0:(L - 1)) %*% b1)   # centroid of equally-weighted MA
shift2_trend <- as.double((0:(L - 1)) %*% b2)   # centroid of exponentially-weighted MA

cat("Zero-frequency shift (equally weighted MA)      :", shift1_trend, "\n")
cat("Zero-frequency shift (exponentially weighted MA) :", shift2_trend, "\n")
# The EWMA centroid is smaller, confirming it is less lagged on trend-like inputs.

# ── Numerical verification on a linear trend ─────────────────────────────────
# A linear trend x_t = t can be thought of as a zero-frequency signal.
# Applying the filter should shift the trend by exactly shift_trend time units.

x        <- 1:len
y1_trend <- filter(x, b1, side = 1)
y2_trend <- filter(x, b2, side = 1)

# Amplitude at zero frequency (omega = 0) equals the sum of filter coefficients;
# for normalised filters this is 1, but we read it from the computed object for
# generality (stored at index 1, corresponding to omega = 0).
amp1_trend <- as_obj1$amp[1]
amp2_trend <- as_obj2$amp[1]
  
# Construct analytical predictions for the trend case
x_shift_amp_trend1 <- amp1_trend * (-shift1_trend + (1:len))
x_shift_amp_trend2 <- amp2_trend * (-shift2_trend + (1:len))

par(mfrow = c(2, 1))
ts.plot(cbind(x_shift_amp_trend1, y1_trend),
        main = "Equally-weighted MA: analytical trend prediction vs. filtered trend",
        col  = c("blue", "black"),lty=1:2)
ts.plot(cbind(x_shift_amp_trend2, y2_trend),
        main = "Exponentially-weighted MA: analytical trend prediction vs. filtered trend",
        col  = c("red", "black"),lty=1:2)

# ════════════════════════════════════════════════════════════════════
# MAIN FINDINGS
# ════════════════════════════════════════════════════════════════════
#
# 1. SERIES-DEPENDENT (SAMPLE) MEASURES (Exercise 1) ARE NOISY
#    ─────────────────────────────────────────────────
#    CCF, dynamic regression, transfer entropy, and phase coherence all
#    depend on the realised sample. Estimation noise makes them unreliable
#    guides for comparing or designing predictors, especially in short series.
#
# 2. TRUE (EXPECTED) MEASURES ARE PREFERRED
#    ─────────────────────────────────────
#    Measures derived directly from filter coefficients (amplitude, time-shift,
#    centroid) are exact and sample-independent: they describe what the filter
#    does in population, not what happened to be observed in one realisation.
#
# 3. CCF AS AGGREGATE MEASURE
#    ─────────────────────────────────
#    The CCF of two filter coefficient vectors collapses the full
#    frequency-specific lead/lag structure into a single scalar lag estimate.
#    This summary is too coarse to capture how filters differ across frequencies.
#    However, the CCF is a potentially important measure for AGGREGATE time-shift
#    effects.
#
# 4. THE TIME-SHIFT FUNCTION IS INFORMATIVE BUT FREQUENCY-SPECIFIC
#    ───────────────────────────────────────────────────────────────
#    The time-shift function reveals the lag introduced by a filter at every
#    frequency, but any single-number summary requires choosing a frequency.
#    The choice of which frequency to report should be guided by the signal
#    of interest (e.g., the trend or the business-cycle band).
#
# 5. ZERO-FREQUENCY SHIFT IS A PRACTICAL PROXY FOR Trend and BUSINESS-CYCLE TIMELINESS
#    ─────────────────────────────────────────────────────────────────────────
#    The zero-frequency (trend) shift — derived as the centroid of the filter's
#    impulse response — is a closed-form, exact measure. Because macro-economic
#    trends and cycles sit at or close to zero frequency, continuity of the 
#    time-shift function implies that the zero-frequency shift can also be a 
#    reliable indicator of the lag a filter introduces at business-cycle frequencies.
#
# 6. AMPLITUDE AND TIME-SHIFT FULLY CHARACTERISE FILTER EFFECTS ON SINUSOIDS
#    ─────────────────────────────────────────────────────────────────────────
#    For a sinusoidal input at frequency omega, a linear filter produces a
#    sinusoidal output scaled by the amplitude (gain) and delayed by the
#    time-shift (phase delay). Together, these two functions provide a complete
#    description of what the filter does to any single-frequency component.
#
# 7. THIS CHARACTERISATION EXTENDS TO ALL STATIONARY (AND INTEGRATED) SERIES
#    ─────────────────────────────────────────────────────────────────────────
#    By the spectral representation theorem, any stationary series can be
#    decomposed into a (continuous) superposition of sinusoids. Integrated
#    (unit-root) series admit an analogous decomposition. Consequently,
#    amplitude and time-shift jointly describe the effect of a linear filter
#    on the entire class of stationary and integrated time series — not just
#    pure sinusoids.
#
# 8. AMPLITUDE AND TIME-SHIFT ARE EXACT, SAMPLE-INDEPENDENT MEASURES
#    ──────────────────────────────────────────────────────────────────
#    Because they are computed analytically from filter coefficients alone,
#    amplitude and time-shift do not depend on the length, distribution, or
#    realisation of any observed series. This makes them robust benchmarks
#    for comparing competing filters.
#
# 9. THESE MEASURES ARE ACTIONABLE FOR FILTER DESIGN
#    ──────────────────────────────────────────────────
#    Amplitude and time-shift or true/expected CCF can be embedded directly 
#    into optimisation criteria, enabling the systematic design of filters that 
#    achieve a desired balance on the accuracy–timeliness frontier — without 
#    requiring any simulation or in-sample fitting.
# ════════════════════════════════════════════════════════════════════



















