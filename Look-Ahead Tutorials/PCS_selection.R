
# Various types and various geometries:

# Type I: potentially highest rank; b is linear combination of gamma_h and gamma_k-gamma_{k-1}, k=1,...,h
# Type II: b is linear combination of gamma_h, gamma_h-gamma_{h-1}
# Type III: b is linear combination of gammah, gamma_h-gamma_0
# Type IV: b is linear combination of gamma_h, gamma_1-gamma_0.

# Notes: 
# 1.for type I b is linear combination of gamma_0,...,gamma_h: this is because gamma_h and gamma_k-gamma_{k-1}, k=1,...,h 
# is the same space as gamma_0,...,gamma_h.
# 2. This equivalence also applies to Type II and Type III, but not Type IV
# For type IV the space is given by gamma_h and gamma_1-gamma_0.

# Geometry: use Type III for illustration
# -The space is given by gamma_0 and gamma_h: b \propto gamma_h + lambda * gamma_0 (assume weight on gamma_h is positive)
# -gamma_0 = lambda_1 * gamma_h + lambda_2 * gamma_h_orthogonal (Graham-Schmidt orthogonalization)

# The different types differ with respect to gamma_h_orthogonal.
# gamma_h_orthogonal is fully decoupled from gamma_h.
# 1. b ' * gamma_h = lambda_1 * ||gamma_h||^2 does not depend on lambda_2.
# 2. ||b||^2 = lambda_1^2 * ||gamma_h||^2 + lambda_2^2 * ||gamma_0||^2 + 2*lambda_1*lambda_2*||gamma_h' * gamma_0|| depends on lambda_2.
# In order to maximize the target correlation b' * gamma_h / ||b||*||gamma|| we should 

# A. Minimize lambda_2^2 * ||gamma_0||^2 + 2*lambda_1*lambda_2* (gamma_h' * gamma_0) in the denominator of the target correlation (to maximize the target)
# B. Push the peak of the CCF to the right.

# Note: lambda_1 and lambda_2 are fixed by gamma_0 (Graham Schmidt). 
# The `variable' part in the minimization of A (and the push in B) is the choice of gamma_k (here k=0): 
# should we use gamma_0 or another gamma_k in the PCS constraint.
# 

# How is the peak of the CCF pushed to the right?

# CCF(0) = b' * gamma_0 / ||b|| * ||gamma|| \ propto b' * gamma_0
# CCF(1) \propto b' * gamma_1

# 0 > CCF(1) - CCF(0) = b' * (gamma_1 - gamma_0).

# Note: ||gamma_1|| < ||gamma_0|| if L is sufficiently long (||gamma_0||^2 = gamma_{00}^2 + ||gamma_1||^2 if L=\infty, and gamma_{00}=1 per definition).

# Therefore b' * (gamma_1 - gamma_0) > 0 together with ||gamma_1|| < ||gamma_0|| implies that 
# b must lie at a favorable angle to gamma_1 in the plane spanned by gamma_0 and gamma_h (b lies in the plane of gamma_0 and gamma_h).

