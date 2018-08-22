
export logsumexp, rDirichlet, embed, condNorm, ldnorm, lmvbeta;

"""
    logsumexp(x[, usemax])

Computes `log(sum(exp(x)))` in a stable manner.

### Example
```julia
  x = rand(5)
  logsumexp(x)
  log(sum(exp.(x)))
```
"""
function logsumexp(x::Array{Float64}, usemax::Bool=true)
  if usemax
    m = maximum(x)
  else
    m = minimum(x)
  end

  m + log(sum(exp.(x .- m)))
end

"""
    logsumexp(x, region[, usemax])

Computes `log(sum(exp(x)))` in a stable manner along dimensions specified.

### Example
```julia
  x = reshape(collect(1:24)*1.0, (2,3,4))
  logsumexp(x, 2)
```
"""
function logsumexp(x::Array{Float64}, region, usemax::Bool=true)
  if usemax
    ms = maximum(x, region)
  else
    ms = minimum(x, region)
  end
  bc_xminusms = broadcast(-, x, ms)

  expxx = exp.(bc_xminusms)
  sumexpxx = sum(expxx, region)

  log.(sumexpxx) .+ ms
end




"""
    rDirichlet(α[, logscale])

  Single draw from Dirichlet distribution, option for log scale.

  ### Example
  ```julia
  rDirichlet(ones(5),true)
  ```
"""
function rDirichlet(α::Array{Float64, 1}, logscale::Bool=false)
  @assert( all(α .> 0.0) )

  k = length(α)
  xx = Vector{Float64}(undef, k) # allows changes to elements
  s = 0.0

  if logscale
    for i in 1:k
      xx[i] = log(rand(Gamma(α[i], 1.0)))
    end
    s = logsumexp(xx)
    out = xx .- s

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
    embed(y::Union{Vector{Int}, Vector{Float64}}, nlags::Int)

### Example
```julia
embed(collect(1:5), 2)
```
"""
function embed(y::Union{Vector{Int}, Vector{Float64}}, nlags::Int)
    TT = length(y)
    n = TT - nlags
    emdim = nlags + 1
    out = zeros(Float64, n, emdim)
    for i in 1:n
        tt = i + nlags
        out[i,:] = copy(y[range(tt, step=-1, length=emdim)])
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
  U = (cholesky(Σxx)).U
  Ut = transpose(U)

  # C = U' \ Σxy
  # C = At_ldiv_B(U, Σxy)
  C = Ut \ Σxy
  # D = U' \ (x - μx)
  # D = At_ldiv_B(U, (x - μx))
  D = Ut \ (x - μx)

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


## triangular matrix from vector
# https://groups.google.com/forum/#!topic/julia-users/UARlZBCNlng
# [ i<=j ? v[j*(j-1)>>>1+i] : 0 for i=1:n, j=1:n ]

"""
    ldnorm(x, μ, σ2)
### Example
```julia
ldnorm(0.5, 0.0, 1.0)
```
"""
function ldnorm(x::Float64, μ::Float64, σ2::Float64)
  -0.5*log(2*π*σ2) - 0.5 * (x - μ)^2 / σ2
end


"""
    lmvbeta(x)

Computes the natural log of ``∏(Γ(x)) / Γ(sum(x))``.

### Example
```julia
  x = rand(5)
  lmvbeta(x)
```
"""
function lmvbeta(x::Array{Float64})
    sum(lgamma.(x)) - lgamma(sum(x))
end
