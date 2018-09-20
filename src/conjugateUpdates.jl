#=
conjugateUpdates.jl

The function naming convention for this file is:
post_likelihood_unknownParams
=#

export post_norm_mean, post_norm_var, rpost_normlm_beta1, post_alphaDP,
       rpost_MvN_knownPrec, rpost_MvNprec_knownMean;


"""
    post_norm_mean(ȳ, n, σ2, μ0, v0)

  Returns conjugate normal posterior under independent normal likelihood
  with known variance and normal prior on the mean.

  Accepts the sample mean and returns a normal distribution.
"""
function post_norm_mean(ȳ::T, n::Int, σ2::T, μ0::T, v0::T) where T <: Real
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
function post_norm_var(ss::T, n::T, a0::T, b0::T) where T <: Real
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
function rpost_normlm_beta1(y::Vector{T},
  X::Matrix{T},
  σ2::T, τ2::T, β0::T=0.0) where T <: Real
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
function rpost_normlm_beta1(y::T, X::Vector{T},
  σ2::T, τ2::T, β0::T=0.0) where T <: Real
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
function rpost_normlm_beta1(y::Vector{T},
  X::Vector{T},
  σ2::T, τ2::T, β0::T=0.0) where T <: Real
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
function rpost_normlm_beta1(y::T, X::T,
  σ2::T, τ2::T, β0::T=0.0) where T <: Real
  # n = 1 and p = 1 case

  σ2 > 0.0 || throw(ArgumentError("σ2 must be positive."))
  τ2 > 0.0 || throw(ArgumentError("τ2 must be positive."))

  ystar = y .- β0

  A = 1.0 / τ2 + dot(X,X) / σ2
  μ = (X'ystar/σ2) / A

  z = randn(1)
  β = z / sqrt(A) .+ μ
end



"""
    post_alphaDP(H::Int, lω_last::T, a_α::T, b_α::T)

    Returns posterior full conditional gamma distribution
    for the Dirichlet Process mass parameter `α`
    when using the truncated stick breaking representation.

    `H` is the the DP truncation level.
    `lω_last` is the Hth and last log weight.
    `a_α` is the shape of the gamma prior.
    `b_α` is the rate of the gamma prior.
"""
function post_alphaDP(H::Int, lω_last::T, a_α::T, b_α::T) where T <: Real
    a1 = a_α + H - 1.0
    b1 = b_α - lω_last # rate parameter
    Gamma(a1, 1.0/b1)
end


function rpost_MvN_knownPrec(Y::Array{T,2}, Λ::PDMat{T},
                             μ0::Array{T,1}, Λ0::PDMat{T}) where T <: Real
    n = float(T)(size(Y,1))
    ybar = vec(mean(Y, dims=1))
    return rpost_MvN_knownPrec(n, ybar, Λ, μ0, Λ0)
end
function rpost_MvN_knownPrec(n::T, ybar::Array{T,1}, Λ::PDMat{T},
                             μ0::Array{T,1}, Λ0::PDMat{T}) where T <: Real
    d = length(ybar)
    Λ1 = PDMat(Λ0 + (n .* Λ))
    a = Λ0*μ0 + (n .* Λ * ybar)
    μ1 = Λ1 \ a
    return μ1 + Λ.chol.U \ randn(d)
end


function rpost_MvNprec_knownMean(Y::Array{T,2}, μ::Array{T,1},
                                 df::T, invSc::PDMat{T}) where T <: Real
    n = size(Y,1)
    K = length(μ)
    SS = zeros(T, K, K)
    for i = 1:n
        dev = Y[i,:] - μ
        SS +=  dev * dev'
    end
    return rpost_MvNprec_knownMean(float(T)(n), PDMat(SS), df, invSc)
end
function rpost_MvNprec_knownMean(n::T, SS::PDMat{T}, df::T, invSc::PDMat{T}) where T <: Real
    df1 = df + n
    invSc1 = PDMat(invSc + SS)
    Sc1 = inv(invSc1) # is there some way around this?
    return rand(Distributions.Wishart(df1, Sc1))
end
