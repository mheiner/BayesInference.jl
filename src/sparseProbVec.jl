# sparseProbVec.jl

export SparseDirMixPrior, SparseSBPrior, SparseSBPriorP, SparseSBPriorFull,
  logSDMweights, logSDMmarginal, rSparseDirMix,
  rpost_sparseStickBreak, slice_mu, logSBMmarginal;


immutable SparseDirMixPrior
  α::Union{Float64, Vector{Float64}}
  β::Float64
end

immutable SparseSBPrior
  α::Float64
  p1::Float64
  μ::Float64
  M::Float64
end

type SparseSBPriorP
  α::Float64
  μ::Float64
  M::Float64
  a_p1::Float64
  b_p1::Float64
  p1_now::Float64
end

type SparseSBPriorFull
  α::Float64
  M::Float64
  a_p1::Float64
  b_p1::Float64
  a_μ::Float64
  b_μ::Float64
  p1_now::Float64
  μ_now::Float64
end


### Sparse Dirichlet Mixture

"""
    logSDMweights(α, β)

  Calculate the log of mixture weights for the SDM prior.

"""
function logSDMweights(α::Vector{Float64}, β::Float64)
  assert(β > 1.0)
  K = length(α)
  X = reshape(repeat(α, inner=K), (K,K)) + β .* eye(K)
  lgX = lgamma(X)
  lpg = reshape(sum(lgX, 2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg - lpg_denom
  lw
end

"""
    logSDMmarginal(x, α, β)

  Calculate the log of the SDM prior predictive probability mass function.

"""
function logSDMmarginal(x::Vector{Int}, α::Vector{Float64}, β::Float64)
    lwSDM = logSDMweights(α, β)

    K = length(α)
    A = reshape(repeat(α, inner=K), (K,K)) + β .* eye(K)
    AX = A + reshape(repeat(x, inner=K), (K,K))

    lnum = [ lmvbeta(AX[k,:]) for k in 1:K ]
    ldenom = [ lmvbeta(A[k,:]) for k in 1:K ]

    lv = lwSDM .+ lnum .- ldenom

    logsumexp( lv )
end


"""
    rSparseDirMix(α, β[, logscale=FALSE])

  Draw from sparse Dirichlet mixture: p(Θ) ∝ Dir(α)⋅∑Θ^β

"""
function rSparseDirMix(α::Vector{Float64}, β::Float64, logscale=false)
  assert(β > 1.0)
  K = length(α)
  X = reshape(repeat(α, inner=K), (K,K)) + β .* eye(K)
  lgX = lgamma.(X)
  lpg = reshape(sum(lgX, 2), K)
  lpg_denom = logsumexp(lpg)

  lw = lpg - lpg_denom
  z = StatsBase.sample(WeightVec( exp.(lw) ))

  rDirichlet(X[z,:], logscale)
end




### Sparse Stick Breaking mixture


function h_slice_mu(z2::Vector{Float64}, n2::Int, M::Float64, μ::Float64,
  logscale::Bool=false)

  a = -n2*(lgamma(M*μ) + lgamma(M*(1.0 - μ)))
  b = M*μ*sum( log(z2) .- log(1.0 .- z2))
  out = a + b
  if (!logscale)
    out = exp(out)
  end
  out
end


function slice_mu(z::Vector{Float64}, ξ::Vector{Int}, μ_old::Float64, M::Float64,
  a_μ::Float64, b_μ::Float64)

  # assumes xi=1 for group 1, xi=2 for group 2

  n2 = sum( ξ.==2 )
  betadist = Beta(a_μ, b_μ)
  if n2 == 0
    μ_out = rand( betadist )
  else
    z2 = copy( z[ find(ξ.==2) ] )
    b_u = h_slice_mu(z2, n2, M, μ_old)
    u = rand( Uniform(0.0, b_u) )
    lu = log(u)

    ii = 0
    keepgoing = true
    μ_cand = 0.0
    while keepgoing
      if ii > 1000
        throw("too many slices!")
      end

      μ_cand = rand( betadist )
      lh = h_slice_mu(z2, n2, M, μ_cand, true)

      keepgoing = lu > lh
      ii += 1
    end

    μ_out = copy(μ_cand)
  end

  μ_out
end



function rpost_sparseStickBreak(x::Vector{Int}, p1::Float64, α::Float64,
  μ::Float64, M::Float64)

  const K = length(x)
  const n = K - 1
  const rcrx = cumsum( x[range(K, -1, n)] )[range(n, -1, n)]  # ∑_{k+1}^K x_k
  const γ = M * μ
  const δ = M * (1.0 - μ)

  ## mixture component 1 update
  const a1 = 1.0 .+ x[1:n]
  const b1 = α .+ rcrx

  ## mixture component 2 update
  const a2 = γ .+ x[1:n]
  const b2 = δ .+ rcrx

  ## calculate posterior mixture weights
  const lgwt1 = log(p1) .- lbeta.(1.0, α) .+ lbeta.(a1, b1)
  const lgwt2 = log(1.0 - p1) .- lbeta.(γ, δ) .+ lbeta.(a2, b2)
  const ldenom = [ logsumexp( [lgwt1[i], lgwt2[i]] ) for i in 1:n ]

  const post_p1 = exp.( lgwt1 .- ldenom )
  const is1 = rand(n) .< post_p1
  const ξ = -1*(is1 .* 1 - 1) + 1

  z = Vector{Float64}(n)
  for i in 1:n
    if is1[i]
      z[i] = rand( Beta(a1[i], b1[i]) )
    else
      z[i] = rand( Beta(a2[i], b2[i]) )
    end
  end

  ## break the Stick
  w = Vector{Float64}(K)
  whatsleft = 1.0

  for i in 1:n
    w[i] = z[i] * whatsleft
    whatsleft -= copy(w[i])
  end
  w[K] = copy(whatsleft)

  (w, z, ξ)
end
function rpost_sparseStickBreak(x::Vector{Int}, p1_old::Float64, α::Float64,
  μ::Float64, M::Float64, a_p1::Float64, b_p1::Float64)

  ## inference for p1 only

  w, z, ξ = rpost_sparseStickBreak(x, p1_old, α, μ, M)
  p1_now = rand( Beta(a_p1 + sum(ξ==1), b_p1 + sum(ξ==2) ) )

  (w, z, ξ, p1_now)
end
function rpost_sparseStickBreak(x::Vector{Int}, p1_old::Float64, α::Float64,
  μ_old::Float64, M::Float64, a_p1::Float64, b_p1::Float64, a_μ::Float64, b_μ::Float64)

  ## inference for p1 and μ as well

  w, z, ξ = rpost_sparseStickBreak(x, p1_old, α, μ_old, M)
  μ_now = slice_mu(z, ξ, μ_old, M, a_μ, b_μ)
  p1_now = rand( Beta(a_p1 + sum(ξ==1), b_p1 + sum(ξ==2) ) )

  (w, z, ξ, μ_now, p1_now)
end



"""
    logSBMmarginal(x, p1, α, μ, M)

  Calculate the log of the SBM prior predictive probability mass function.

"""
function logSBMmarginal(x::Vector{Int}, p1::Float64, α::Float64,
  μ::Float64, M::Float64)

      const K = length(x)
      const n = K - 1
      const rcrx = cumsum( x[range(K, -1, n)] )[range(n, -1, n)]  # ∑_{k+1}^K x_k
      const γ = M * μ
      const δ = M * (1.0 - μ)

      ## mixture component 1 update
      const a1 = 1.0 .+ x[1:n]
      const b1 = α .+ rcrx

      ## mixture component 2 update
      const a2 = γ .+ x[1:n]
      const b2 = δ .+ rcrx

      ## calculate posterior mixture weights
      const lgwt1 = log(p1) .- lbeta.(1.0, α) .+ lbeta.(a1, b1)
      const lgwt2 = log(1.0 - p1) .- lbeta.(γ, δ) .+ lbeta.(a2, b2)
      const lsum = [ logsumexp( [lgwt1[i], lgwt2[i]] ) for i in 1:n ]

      sum( lsum )
end
