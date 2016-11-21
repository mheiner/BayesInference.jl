module BayesInference


"""
LogSumExp trick
    logsumexp(x)

Computes `log(sum(exp(x)))` in a stable manner.

# Example
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




end # module
