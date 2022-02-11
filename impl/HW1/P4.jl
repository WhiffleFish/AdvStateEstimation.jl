using AdvStateEstimation
import AdvStateEstimation as SE
BLAS.set_num_threads(1)
vars = matread(joinpath(@__DIR__, "homework1_missileprob_data.mat"))

ystacked = dropdims(vars["ystacked"]; dims=2)
x0true = dropdims(vars["x0true"]; dims=2)

rocket = Rocket()
nls = GaussNewtonNLS(rocket, ystacked, α=0.5)

x0 = x0true + [10,5,1,10]

x, opt_hist = update!(nls, x0, 100; hist=true)

plot(opt_hist)
plot(getindex.(opt_hist.x,3))

Y = Iterators.partition(SE.measvec(rocket, x, 40),2) |> collect

idx = 2
p = plot(getindex.(Y,idx),marker='o', color=:blue, markersize=15)
plot!(getindex.(collect(Iterators.partition(ystacked, 2)),idx), marker='x', color=:red, markersize=15)
current_figure()

AdvStateEstimation.stack_meas_jac(rocket, x0, 40)

##
using Zygote
rocket_meas(x) = meas(rocket, x)

using Test
@test first(Zygote.jacobian(rocket_meas, x0true)) ≈ meas_jac(rocket, x0true)
