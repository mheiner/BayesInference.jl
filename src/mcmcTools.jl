# mcmcTools.jl

export adapt_cΣ;


"""
    adapt_cΣ(accpt_rate, cΣ, tries, accpt_bnds=[0.23, 0.40], adjust=[0.5, 2.0]
"""
function adapt_cΣ(accpt_rate::Float, cΣ::Matrix{Float}, tries::Int,
  accpt_bnds::Vector{Float}=[0.23, 0.40], adjust::Vector{Float}=[0.5, 2.0])

  pass = true
  cΣ_out = cΣ

  too_low = accpt_rate < accpt_bnds[1]
  too_high = accpt_rate > accpt_bnds[2]

  if too_low
    pass = false
    cΣ_out = adjust[1] .* cΣ
  elseif too_high
    pass = false
    cΣ_out = adjust[2] .* cΣ
  end

  return (pass, tries+1, cΣ_out)
end
