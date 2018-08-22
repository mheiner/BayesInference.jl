module BayesInference

using Distributions
using StatsBase
using LinearAlgebra
using SpecialFunctions

include("generalTools.jl")
include("conjugateUpdates.jl")
include("mcmcTools.jl")
include("sparseProbVec.jl")

end # module
