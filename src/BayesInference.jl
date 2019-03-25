module BayesInference

using LinearAlgebra
using SpecialFunctions

using Distributions
using PDMats
using StatsBase
using Dates

include("generalTools.jl")
include("conjugateUpdates.jl")
include("mcmcTools.jl")

end # module
