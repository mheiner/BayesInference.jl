"""
The function naming convention for this file is:
post_likelihood_unknownParams
"""


export post_norm_mean;


"""
    post_norm_mean(ȳ, n, σ2, μ0, v0)

  Returns conjugate normal posterior under independent normal loglikelihood
  with known variance and normal prior on the mean.

  Accepts the sample mean and returns a normal distribution.
"""
function post_norm_mean(ȳ::Float64, n::Int, σ2::Float64, μ0::Float64, v0::Float64)
  σ2 > 0.0 || throw(ArgumentError("σ must be positive."))
  v0 > 0.0 || throw(ArgumentError("v0 must be positive."))
  n > 0.0 || throw(ArgumentError("n must be positive."))

  n = convert(Float64, n)

  v1 = 1.0 / (1.0/v0 + n/σ2)
  sd1 = sqrt(v1)
  μ1 = v1 * (μ0/v0 + n*ȳ/σ2)

  Normal(μ1, sd1)
end
