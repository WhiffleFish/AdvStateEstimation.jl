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
kf = UnscentedKF(ss_dt, x0, P0; α=sqrt(1/2 + 1))

sim = KFSimulator(kf, x0, u)
simulate(sim, 15.)

kf.sigma_points

λ = HW4.lambda(kf)
λ / (2 + λ)

kf.Pkm

scatter(first.(kf.sigma_points),last.(kf.sigma_points))

sqrt(1/n  + 1)
