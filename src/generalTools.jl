
export logsumexp, rDirichlet, embed, condNorm,
ldnorm, lmvbeta, vech, xpnd_tri;

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
function logsumexp(x::Array{Real}, usemax::Bool=true)
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
function logsumexp(x::Array{Real}, region, usemax::Bool=true)
  if usemax
    ms = maximum(x, dims=region)
  else
    ms = minimum(x, dims=region)
  end
  bc_xminusms = broadcast(-, x, ms)

  expxx = exp.(bc_xminusms)
  sumexpxx = sum(expxx, dims=region)

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
function rDirichlet(α::Array{Real, 1}, logscale::Bool=false)
  @assert( all(α .> 0.0) )

  k = length(α)
  xx = Vector{Real}(undef, k) # allows changes to elements
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
    embed(y::Union{Vector{Int}, Vector{Real}}, nlags::Int)

### Example
```julia
embed(collect(1:5), 2)
```
"""
function embed(y::Union{Vector{Int}, Vector{Real}}, nlags::Int)
    TT = length(y)
    n = TT - nlags
    emdim = nlags + 1
    out = zeros(Real, n, emdim)
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
function condNorm(μx::Union{Real, Vector{Real}},
  μy::Union{Real, Vector{Real}},
  Σxx::Union{Real, Array{Real, 2}},
  Σxy::Union{Real, Vector{Real}, Array{Real, 2}},
  Σyy::Union{Real, Array{Real, 2}}, x::Union{Real, Vector{Real}})

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
function ldnorm(x::Real, μ::Real, σ2::Real)
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
function lmvbeta(x::Array{Real})
    sum(lgamma.(x)) - lgamma(sum(x))
end


"""
    vech(A)

Extracts the lower triangular elements of a matrix as a vector.

Adapted from https://discourse.julialang.org/t/half-vectorization/7399/4

### Example
```julia
    A = reshape(collect(1:16), 4,4)
    v = vech(A)
    xpnd_tri(v)

    v = vech(A, false)
    xpnd_tri(v)

    vech(A', false)
```
"""
function vech(A::AbstractArray{T, 2},
    diag::Bool=true) where {T}
    m = LinearAlgebra.checksquare(A)
    k = 0
    if diag
        v = Vector{T}(undef, (m*(m+1))>>1)
        for j = 1:m, i = j:m
            @inbounds v[k += 1] = A[i,j]
        end
    else
        v = Vector{T}(undef, (m*(m-1))>>1)
        for j = 1:(m-1), i = (j+1):m
            @inbounds v[k += 1] = A[i,j]
        end
    end
    return v
end

"""
    xpnd(v)

Creates a lower triangular matrix from the elements of a vector.

Adapted from https://groups.google.com/forum/#!topic/julia-users/UARlZBCNlng
and xpnd function from R package MCMCpack.

### Example
```julia
  A = reshape(collect(1:16), 4,4)
  v = vech(A)
  xpnd_tri(v)

  v = vech(A, false)
  xpnd_tri(v, false)
```
"""
function xpnd_tri(v::Vector{T}, diag::Bool=true) where {T}
    k = 0
    n = length(v)
    if diag
        m = Int((sqrt(1 + 8*n) - 1) / 2)
        A0 = Array{T, 2}(undef, m, m)
        for j = 1:m, i = j:m
            k += 1
            @inbounds A0[i,j] = v[k]
        end
        A = LowerTriangular(A0)
    else
        m = Int((sqrt(1 + 8*n) + 1) / 2)
        A0 = Array{T, 2}(undef, m, m)
        for j = 1:(m-1), i = (j+1):m
            k += 1
            @inbounds A0[i,j] = v[k]
        end
        A = UnitLowerTriangular(A0)
    end
    return A
end
