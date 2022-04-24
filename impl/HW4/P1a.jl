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

P0 = 10*I(2)
I0 = inv(P0) |> Matrix
x0 = zeros(2)
i0 = I0*x0
IF = InformationFilter(ss_dt, I0, i0)
u(t) = SA[2cos(0.75t)]
sim = IFSimulator(IF, x0, u)

t, x, y = simulate(ss_dt, x0, u, 15.0)
const TIMES = t
const X_DATA = x
const Y_DATA = y

sim.yhist
HW4.load!(sim, TIMES, X_DATA, Y_DATA)

sim.yhist

simulate(sim)

##  plot the individual state errors with 2σ estimation error covariance bounds vs. time
covar = HW4.covariance(sim) .|> diag
fig = Figure()
for i in eachindex(x0)
    ax = Axis(fig[i,1], xlabel="t", ylabel="x_$i error")
    scatter!(ax, sim.thist, getindex.(sim.xpredhist, i) .- getindex.(sim.xhist, i), markersize=5)
    σ = sqrt.(getindex.(covar, i))
    lines!(sim.thist, 2 .* σ, color=:red, linestyle=:dash)
    lines!(sim.thist, - 2 .* σ, color=:red, linestyle=:dash)
end
fig
save(joinpath(@__DIR__, "img", "P1aStateError.pdf"), fig)
##plot the measurement innovations with innovation error covariance 2σ bounds vs. time

Pkms = diag.(inv.(sim.Imhist[2:end]))
innovations = Vector{Float64}(undef, length(sim.thist) - 1)
for (i,t) in enumerate(sim.thist[1:end-1])
    x_p = sim.xpredhist[i]
    x_m = HW4.dynamics(ss_dt, x_p, u(t))
    y_hat = HW4.meas(ss_dt, x_m, u(t))
    y = sim.yhist[i+1]
    innovations[i] = first(y_hat .- y)
end

fig = Figure()
ax = Axis(fig[1,1], xlabel="t", ylabel="Meas Innovation")
scatter!(ax, sim.thist[2:end], innovations)
lines!(ax, sim.thist[2:end], 2 .* sqrt.(first.(Pkms)), color=:red, linestyle=:dash )
lines!(ax, sim.thist[2:end], - 2 .* sqrt.(first.(Pkms)), color=:red, linestyle=:dash )
fig
save(joinpath(@__DIR__, "img", "P1aMeasInnovation.pdf"), fig)
