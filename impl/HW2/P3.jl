using AdvStateEstimation
const ASE = AdvStateEstimation
BLAS.set_num_threads(1)

## setup + read data

vars1 = matread(joinpath(@__DIR__, "../HW1/homework1_missileprob_data.mat"))
ystacked1 = dropdims(vars1["ystacked"]; dims=2)
x0true1 = SVector{4,Float64}(dropdims(vars1["x0true"]; dims=2))

vars2 = matread(joinpath(@__DIR__, "hw2_missileprob_data.mat"))
ystacked2 = dropdims(vars2["ystacked"]; dims=2)
x0true2 = SVector{4,Float64}(dropdims(vars2["x0true"]; dims=2))

## a
R = Symmetric(SA[70. 0; 0 5e-3])
rocket = Rocket(R=R)

ea1 = easting_alt(rocket.L, ystacked1)
ea2 = easting_alt(rocket.L, ystacked2)
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ξ (m)", ylabel="a (m)")
    s1 = scatter!(ax, ea1, marker='+', markersize=20, label="Gaussian")
    s2 = scatter!(ax, ea2, marker='o', markersize=20, label="TDist")
    axislegend(ax)
    display(fig)
end
save(joinpath(@__DIR__,"img/P3a.pdf"), fig)

## b

vars2 = matread(joinpath(@__DIR__, "hw2_missileprob_data.mat"))
ystacked = dropdims(vars2["ystacked"]; dims=2)
x0true = SVector{4,Float64}(dropdims(vars2["x0true"]; dims=2))

rocket = Rocket(R=Symmetric(SA[1e4 0 ; 0 1e-4]))

nls = GaussNewtonNLS(rocket, copy(ystacked), α=0.1)

x0 = SA[10.,40.,2300.,30.]
x , opt_hist = update!(nls, x0, 100; hist=true)

f1 = plot(opt_hist, ystacked)
save(joinpath(@__DIR__,"img/P3b_OptHist.pdf"), f1)
f2 = ASE.plotx(opt_hist, x0true)
save(joinpath(@__DIR__,"img/P3b_StateHist.pdf"), f2)


dist_vec = [TDist(.5), Normal(0,5e-3)]
rocket = Rocket(R=Symmetric(SA[70. 0 ; 0 1.5e-3]))
nls = GaussNewtonNLS(rocket, copy(ystacked), α=0.1)
mct = ASE.MCTrials(nls, x0true, x0, 100, 100, dist_vec)
fig = plot(mct, x0true)
save(joinpath(@__DIR__,"img/P3b.pdf"), fig)
ASE.bias(mct, x0true)
ASE.error_cov(mct, x0true)

##
