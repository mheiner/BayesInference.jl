#=
conjugateUpdates.jl

The function naming convention for this file is:
post_likelihood_unknownParams
=#

export post_norm_mean, post_norm_var, rpost_normlm_beta1;


"""
    post_norm_mean(ȳ, n, σ2, μ0, v0)

  Returns conjugate normal posterior under independent normal likelihood
  with known variance and normal prior on the mean.

  Accepts the sample mean and returns a normal distribution.
"""
function post_norm_mean(ȳ::Real, n::Int, σ2::Real, μ0::Real, v0::Real)
  σ2 > 0.0 || throw(ArgumentError("σ must be positive."))
  v0 > 0.0 || throw(ArgumentError("v0 must be positive."))
  n > 0.0 || throw(ArgumentError("n must be positive."))

  # n = convert(Float, n)

  v1 = 1.0 / (1.0/v0 + n/σ2)
  sd1 = sqrt(v1)
  μ1 = v1 * (μ0/v0 + n*ȳ/σ2)

  Normal(μ1, sd1)
end


"""
    post_norm_var(ss, n, a0, b0)

  Returns conjugate inverse-gamma posterior for variance under
  independent normal likelihood with known mean and
  inverse-gamma prior on the variance.

  Accepts `ss```= ∑(y_i - μ_i)^2`` and sample size `n`.

  `a0` is the shape parameter of the inverse-gamma prior.
  `b0` is the scale parameter of the inverse-gamma prior.
"""
function post_norm_var(ss::Real, n::Real, a0::Real, b0::Real)
  a0 > 0.0 || throw(ArgumentError("a0 must be positive."))
  b0 > 0.0 || throw(ArgumentError("b0 must be positive."))

  # n = convert(Float, n)
  a1 = a0 + 0.5*n
  b1 = b0 + 0.5*ss

  InverseGamma(a1, b1)
end



"""
    rpost_normlm_beta1(y, X, σ2, τ2, β0=0.0)

  Returns draw from conjugate normal posterior for linear model
  coefficients under independent normal likelihood with
  (optionally) known intercept `β0` and observation variance `σ2`.

  Assumes independent normals prior with mean `0.0` and variance `τ2`.
"""
function rpost_normlm_beta1(y::Vector{::Real},
  X::Matrix{::Real},
  σ2::Real, τ2::Real, β0::Real=0.0)
  # n > 1 and p > 1 case

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
  τ2 > 0.0 || throw(ArgumentError("τ2 must be positive."))

  n,p = size(X) # assumes X was a matrix
  length(y) == n || throw(ArgumentError("y and X dimension mismatch"))
  ystar = y .- β0

  A = Matrix(1.0I, p, p) / τ2 + X'X / σ2
  U = (cholesky(A)).U
  Ut = transpose(U)

  # μ_a = At_ldiv_B(U, (X'ystar/σ2))
  μ_a = Ut \ (X'ystar/σ2)
  μ = U \ μ_a

  z = randn(p)
  β = U \ z + μ
end
function rpost_normlm_beta1(y::Real, X::Vector{::Real},
  σ2::Real, τ2::Real, β0::Real=0.0)
  # n = 1 and p > 1 case

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
  τ2 > 0.0 || throw(ArgumentError("τ2 must be positive."))

  p = length(X)
  ystar = y .- β0

  A = Matrix(1.0I, p, p) / τ2 + (X * transpose(X)) / σ2
  U = (cholesky(A)).U
  Ut = transpose(U)

  # μ_a = At_ldiv_B(U, (ystar*X/σ2))
  μ_a = Ut \ (ystar*X/σ2)
  μ = U \ μ_a

  z = randn(p)
  β = U \ z + μ
end
function rpost_normlm_beta1(y::Vector{::Real},
  X::Vector{::Real},
  σ2::Real, τ2::Real, β0::Real=0.0)
  # n > 1 and p = 1 case

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
  τ2 > 0.0 || throw(ArgumentError("τ2 must be positive."))
  length(y) == length(X) || throw(ArgumentError("y and X dimension mismatch"))

  ystar = y .- β0

  A = 1.0 / τ2 + dot(X,X) / σ2
  μ = (X'ystar/σ2) / A

  z = randn(1)
  β = z / sqrt(A) .+ μ
end
function rpost_normlm_beta1(y::Real, X::Real,
  σ2::Real, τ2::Real, β0::Real=0.0)
  # n = 1 and p = 1 case

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
  τ2 > 0.0 || throw(ArgumentError("τ2 must be positive."))

  ystar = y .- β0

  A = 1.0 / τ2 + dot(X,X) / σ2
  μ = (X'ystar/σ2) / A

  z = randn(1)
  β = z / sqrt(A) .+ μ
end
