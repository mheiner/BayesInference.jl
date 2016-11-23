
export logsumexp, rDirichlet, condNorm;

"""
    logsumexp(x[, usemax])

Computes `log(sum(exp(x)))` in a stable manner.

### Example
```julia
  x = rand(5)
  logsumexp(x)
  log(sum(exp(x)))
```
"""
function logsumexp(x::Array{Float64}, usemax::Bool=true)
  if usemax
    m = maximum(x)
  else
    m = minimum(x)
  end

  m + log(sum(exp(x - m)))
end



"""
    rDirichlet(α[, logscale])

  Single draw from Dirichlet distribution, option for log scale.
"""
function rDirichlet(α::Array{Float64,1}, logscale::Bool=false)
  assert( all(α .> 0.0) )

  const k = length(α)
  const xx = Vector{Float64}(k) # allows changes to elements
  s = 0.0

  if logscale
    for i in 1:k
      xx[i] = log(rand(Gamma(α[i], 1.0)))
    end
    s = logsumexp(xx)
    out = xx - s

  else
    for i in 1:k
      xx[i] = rand(Gamma(α[i], 1.0))
      s += xx[i]
    end
    out = xx / s

  end

out
end



"""
    condNorm(μx, μy, Σxx, Σxy, Σyy, x)

  Returns the conditional distribution of a the ``y`` subvector
  of ``(x,y) ∼ MultivariateNormal``.

  ### Example
  ```julia
    xx = randn(10, 5)
    Sig = xx'xx
    mu = collect(1:5)*1.0
    x_indx = 1:3
    y_indx = 4:5
    μx = mu[x_indx]
    μy = mu[y_indx]
    Σxx = Sig[x_indx, x_indx]
    Σyy = Sig[y_indx, y_indx]
    Σxy = Sig[x_indx, y_indx]
    x = [0.3, 2.1, 3.4]
    @time condNorm(μx, μy, Σxx, Σxy, Σyy, x)
  ```
"""
function condNorm(μx::Union{Float64, Vector{Float64}},
  μy::Union{Float64, Vector{Float64}},
  Σxx::Union{Float64, Array{Float64, 2}},
  Σxy::Union{Float64, Vector{Float64}, Array{Float64, 2}},
  Σyy::Union{Float64, Array{Float64, 2}}, x::Union{Float64, Vector{Float64}})

  isposdef(Σxx) || throw(ArgumentError("Σxx must be positive definite."))
  isposdef(Σyy) || throw(ArgumentError("Σyy must be positive definite."))

  # Cholesky method (about 2 times faster than direct method)
  U = chol(Σxx)

  C = U' \ Σxy
  D = U' \ (x - μx)

  μ = μy + C'D
  Σ = Σyy - C'C

  # Direct method
  # A = Σxx \ (x - μx)
  # B = Σxx \ Σxy
  #
  # μ = μy + Σxy'A
  # Σ = Σyy - Σxy'B
  # Σ = 0.5 * (Σ' + Σ)

  # because univariate normal parameterized with stdev
  if length(μ) == 1
    Σ = sqrt(Σ)
  end

  out = MultivariateNormal(μ, Σ)
end
