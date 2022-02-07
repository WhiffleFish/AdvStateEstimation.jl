module AdvStateEstimation
using Reexport

@reexport using CairoMakie
@reexport using MAT
@reexport using StaticArrays
@reexport using Distributions
@reexport using StatsBase
using LaTeXStrings

include(joinpath("HW1", "P2.jl"))
export BernoulliMLE, MLE
export GaussianMLE, likelihood_contour

end # module
