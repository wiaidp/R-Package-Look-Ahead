# ════════════════════════════════════════════════════════════════════
# TUTORIAL 2 — DFP PREDICTOR -
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────────────────
# Tutorial 1 demonstrates that the classical Mean Squared Error (MSE)
# multi-step-ahead predictor can become "stuck" at the current time point:
# rather than targeting x_{t+h}, it correlates most strongly with x_t,
# where h > 0 denotes the forecast horizon. This phenomenon motivates the
# introduction of the so-called Decouple-From-Present (DFP) criterion,
# a novel approach designed to explicitly enforce look-ahead behavior in
# the predictor.
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════
# Theoretical background:
#   Wildi, M. (2026). Forecasting on the Accuracy–Timeliness Frontier:
#   Two Novel "Look-Ahead" Predictors.
#   https://doi.org/10.48550/arXiv.2602.23087

# ════════════════════════════════════════════════════════════════════


