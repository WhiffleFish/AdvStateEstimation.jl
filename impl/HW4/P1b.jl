using AdvStateEstimation.HW4
using LinearAlgebra
using StaticArrays
using CairoMakie

A = Float64[0 1; 0 0]
B = reshape(Float64[0;1],2,1)
C = Float64[1 0]
D = zeros(1,1)
Γ = reshape(Float64[0;1], 2, 1)
W = 1.0
V = 1.0
Δt = 0.1

ss_ct = CTStateSpace(A, B, C, D, Γ, W, V)
ss_dt = c2d(ss_ct, Δt)
u(t) = SA[2cos(0.75t)]

##
x0 = zeros(2)
P0 = 10.0*I(2) |> Matrix
kf = UnscentedKF(ss_dt, x0, P0; α=0.01)

sim = KFSimulator(kf, x0, u)
HW4.load!(sim, TIMES, X_DATA, Y_DATA)
simulate(sim)

fig = HW4.plot_error(sim)
save(joinpath(@__DIR__, "img", "P1bStateError.pdf"), fig)

HW4.plot_meas(sim)
save(joinpath(@__DIR__, "img", "P1bMeasInnovation.pdf"), fig)
