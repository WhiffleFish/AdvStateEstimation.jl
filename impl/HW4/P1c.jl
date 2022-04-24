using AdvStateEstimation.HW4
using LinearAlgebra
using StaticArrays
using Distributions
using CairoMakie

##
x0 = zeros(2)
P0 = 10.0*I(2) |> Matrix
initialdist = MvNormal(x0,P0)
pf = HW4.BootstrapPF(ss_dt, initialdist, 500)

sim = PFSimulator(pf, x0, u)
HW4.load!(sim, TIMES, X_DATA, Y_DATA)

simulate(sim)
plot(sim; Ness=true)
##
x0 = zeros(2)
P0 = 10.0*I(2) |> Matrix
initialdist = MvNormal(x0,P0)
pf = HW4.BootstrapPF(ss_dt, initialdist, 500)

sim = PFSimulator(pf, x0, u)

simulate(sim, 15.0)

lines(sim.Ness)

plot(sim; Ness=true)
