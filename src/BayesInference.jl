module BayesInference

using LinearAlgebra
using SpecialFunctions

using Distributions
using PDMats
using StatsBase

include("generalTools.jl")
include("conjugateUpdates.jl")
include("mcmcTools.jl")

end # module
