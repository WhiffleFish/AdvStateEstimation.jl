module HW4

using LinearAlgebra
using Distributions
using StatsBase
using CairoMakie

include("state_space.jl")
export CTStateSpace, DTStateSpace, c2d

include("information_filter.jl")
export InformationFilter, IFSimulator, simulate

include("ukf.jl")
export UnscentedKF, KFSimulator

include("pf.jl")
export BootstrapPF, PFSimulator

end
