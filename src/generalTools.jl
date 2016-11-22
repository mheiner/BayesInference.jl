
export logsumexp, rDirichlet;

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
function rDirichlet(α::Array{Float64}, logscale::Bool=false)

  k = length(α)

  xx = zeros(Float64, k)
  s = 0.0

  if logscale
    for i in 1:k
      α[i] > 0 || throw(ArgumentError("α must be a positive vector."))
      xx[i] = log(rand(Gamma(α[i], 1.0)))
    end
    s = logsumexp(xx)
    out = xx - s

  else
    for i in 1:k
      α[i] > 0 || throw(ArgumentError("α must be a positive vector."))
      xx[i] = rand(Gamma(α[i], 1.0))
      s += xx[i]
    end
    out = xx / s

  end

out
end
