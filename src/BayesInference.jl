module BayesInference

using LinearAlgebra
using SpecialFunctions

using Distributions
using StatsBase
using PDMats

include("generalTools.jl")
include("conjugateUpdates.jl")
include("mcmcTools.jl")

end # module
