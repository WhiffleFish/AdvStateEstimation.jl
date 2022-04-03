module HW3

using StaticArrays
using Distributions
using LinearAlgebra


include("P1.jl")
export ProbGrid, GridFilter, predict!, update!, step!
export MMSE, MAP

end
