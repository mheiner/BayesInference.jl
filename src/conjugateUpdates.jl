#=
conjugateUpdates.jl

The function naming convention for this file is:
post_likelihood_unknownParams
=#

export post_norm_mean, post_norm_var;


"""
    post_norm_mean(ȳ, n, σ2, μ0, v0)

  Returns conjugate normal posterior under independent normal likelihood
  with known variance and normal prior on the mean.

  Accepts the sample mean and returns a normal distribution.
"""
function post_norm_mean(ȳ::Float64, n::Int, σ2::Float64, μ0::Float64, v0::Float64)
  σ2 > 0.0 || throw(ArgumentError("σ must be positive."))
  v0 > 0.0 || throw(ArgumentError("v0 must be positive."))
  n > 0.0 || throw(ArgumentError("n must be positive."))

  const n = convert(Float64, n)

  const v1 = 1.0 / (1.0/v0 + n/σ2)
  const sd1 = sqrt(v1)
  const μ1 = v1 * (μ0/v0 + n*ȳ/σ2)

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
function post_norm_var(ss::Float64, n::Int64, a0::Float64, b0::Float64)
  a0 > 0.0 || throw(ArgumentError("a0 must be positive."))
  b0 > 0.0 || throw(ArgumentError("b0 must be positive."))

  const n = convert(Float64, n)
  const a1 = a0 + 0.5*n
  const b1 = b0 + 0.5*ss

  InverseGamma(a1, b1)
end
