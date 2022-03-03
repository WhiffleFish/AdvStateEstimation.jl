## imports
using AdvStateEstimation
import AdvStateEstimation as SE
BLAS.set_num_threads(1)

## setup + read data
vars = matread(joinpath(@__DIR__, "homework1_missileprob_data.mat"))
ystacked = dropdims(vars["ystacked"]; dims=2)
x0true = SVector{4,Float64}(dropdims(vars["x0true"]; dims=2))

rocket = Rocket()

nls = GaussNewtonNLS(rocket, ystacked, α=0.5)

## b
fig = Figure()
ax = Axis(
    fig[1, 1:2],
    yscale = log10,
    yminorgridvisible = true,
    xlabel = "Gauss-Newton Iteration",
    ylabel = L"J(x)",
    ylabelsize = 20.0)
x0 = SA[10.,40.,2300.,30.]
l = []
α_list = [1.0, 0.5, 0.1, 0.05, 0.01]
for (i,α) in enumerate(α_list)
    nls = GaussNewtonNLS(rocket, ystacked, α=α)
    x , opt = update!(nls, x0, 200; hist=true)
    l_i = lines!(ax, opt.J, label="α=$α")
    push!(l, l_i)
end
fig[1, 3] = Legend(fig, ax, "Learning Rates")
fig
save(joinpath(@__DIR__,"img", "LearningRateComparison.svg"), fig)

## 'Good' initial condition
x0 = SA[10.,40.,2300.,30.]
x , opt_hist = update!(nls, x0, 30; hist=true)

f1 = plot(opt_hist, ystacked)
f2 = SE.plotx(opt_hist, x0true)
Cdata = SE.MCTrials(nls, x0true, x0, 50, 1000)
f3 = plot(MCdata, x0true)
SE.bias(MCdata, x0true)
SE.error_cov(MCdata, x0true)

save(joinpath(@__DIR__,"img", "GoodOptHist.svg"), f1)
save(joinpath(@__DIR__,"img", "GoodxHist.svg"), f2)
save(joinpath(@__DIR__,"img", "GoodMCSims.svg"), f3)

## 'Bad' initial condition
x0 = SA[130.,4.,2.,6050.]
x , opt_hist = update!(nls, x0, 30; hist=true)

f1 = plot(opt_hist, ystacked)
f2 = SE.plotx(opt_hist, x0true)
MCdata = SE.MCTrials(nls, x0true, x0, 50, 1000)
f3 = plot(MCdata, x0true)
SE.bias(MCdata, x0true)
SE.error_cov(MCdata, x0true)

save(joinpath(@__DIR__,"img", "BadOptHist.svg"), f1)
save(joinpath(@__DIR__,"img", "BadxHist.svg"), f2)
save(joinpath(@__DIR__,"img", "BadMCSims.svg"), f3)
