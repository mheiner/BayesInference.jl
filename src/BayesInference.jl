module BayesInference

using Distributions
using StatsBase

include("generalTools.jl")
include("conjugateUpdates.jl")
include("mcmcTools.jl")

end # module
