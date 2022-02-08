using MAT

vars = matread(joinpath(@__DIR__, "homework1_missileprob_data.mat"))

ystacked = dropdims(vars["ystacked"]; dims=2)
x0true = dropdims(vars["x0true"]; dims=2)

rocket = Rocket()
meas(rocket, x0true)





##
using Zygote
rocket_meas(x) = meas(rocket, x)

using Test
@test first(Zygote.jacobian(rocket_meas, x0true)) ≈ meas_jac(rocket, x0true)
