# mcmcTools.jl

export adapt_cΣ, deepcopyFields, etr;


"""
    adapt_cΣ(accpt_rate, cΣ, tries, accpt_bnds=[0.23, 0.40], adjust=[0.5, 2.0]
"""
function adapt_cΣ(accpt_rate::T, cΣ::Matrix{T}, tries::Int,
  accpt_bnds::Vector{T}=[0.23, 0.40], adjust::Vector{T}=[0.5, 2.0]) where T <: Real

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

## from Arthur Lui
function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  for field in fields
    substate[field] = deepcopy(getfield(state, field))
  end

  return substate
end




## estimate time remaining
function etr(timestart::DateTime, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / 1.0e5 # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end
function etr(timestart::DateTime; n_iter_timed::Int, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / float(n_iter_timed) # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end
