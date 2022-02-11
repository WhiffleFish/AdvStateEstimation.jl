module AdvStateEstimation
using Reexport

@reexport using CairoMakie
@reexport using MAT
@reexport using StaticArrays
@reexport using Distributions
@reexport using StatsBase
@reexport using LinearAlgebra
using LaTeXStrings

include(joinpath("HW1", "P2.jl"))
export BernoulliMLE, MLE
export GaussianMLE, likelihood_contour

include(joinpath("HW1", "P4.jl"))
export GaussNewtonNLS, Rocket, update!

end # module
