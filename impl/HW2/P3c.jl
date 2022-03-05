using AdvStateEstimation
const ASE = AdvStateEstimation
using Optim

sr = ASE.StudentRocket()

obj(x) = -ASE.log_likelihood(sr, x, ystacked)

function gen_opt_samples(dist, N)
    opt_vec = []
    xf = []
    for _ in 1:N
        x0 = rand(sample_dist)
        opt = optimize(obj, x0)
        push!(opt_vec, opt)
        push!(xf, opt.minimizer)
    end
    return opt_vec, xf
end

sample_dist = MvNormal(x0true, diagm([20.,20.,20.,100.]))

opt_vec, xfs = gen_opt_samples(sample_dist, 1000)

errors = [xf .- x0true for xf in xfs]

mean_and_var(errors)

min_sq_err = minimum(Base.Fix1(sum, abs2), errors)
min_err = argmin(Base.Fix1(sum, abs2), errors)

best_xf_idx = findfirst(x-> all(x .≈ min_err), errors)
xf_best = xfs[best_xf_idx]
opt_vec[best_xf_idx].initial_x .- x0true

MLE_meas = ASE.measvec(sr.r, xf_best, length(ystacked)÷2)
MMSE_meas = ASE.measvec(sr.r, x, length(ystacked)÷2)
ea_MLE = ASE.easting_alt(sr.r.L, MLE_meas)
ea_MMSE = ASE.easting_alt(sr.r.L, MMSE_meas)
ea_meas = easting_alt(sr.r.L, ystacked)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ξ (m)", ylabel="a (m)")
    l_MLE = lines!(ax, ea_MLE, label="MLE Trajectory")
    s = scatter!(ax, ea_meas, marker='+', markersize=20, color=:red, label="True Measurements")

    axislegend(ax, position=:rb)
    display(fig)
end

save(joinpath(@__DIR__,"img/P3MLE.pdf"), fig)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "ξ (m)", ylabel="a (m)")
    l_MLE = lines!(ax, ea_MLE, label="MLE Trajectory")
    l_MMSE = lines!(ax, ea_MMSE, label="MMSE Trajectory")
    s = scatter!(ax, ea_meas, marker='+', markersize=20, color=:red, label="True Measurements")

    axislegend(ax, position=:rb)
    display(fig)
end

save(joinpath(@__DIR__,"img/MMSEvsMLE.pdf"), fig)
