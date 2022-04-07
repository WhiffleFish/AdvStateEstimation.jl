using AdvStateEstimation
const SE = AdvStateEstimation
using AdvStateEstimation.HW3

vars = matread(joinpath(@__DIR__, "asen6519hw3_problem2.mat"))
zMeasHist = vars["zMeasHist"] |> vec
zetaHist = vars["zetaHist"] |> vec

model = Jalopy()
b = zeros(model.Nx)
b[[3,19,35]] .= 1/3
upd = DiscreteBayesFilter(model, b)

belief_hist = Vector{Float64}[]
push!(belief_hist, copy(b))
for z in zMeasHist[2:end]
    AdvStateEstimation.HW3.update!(upd, z)
    push!(belief_hist, copy(upd.b))
end

MMSE_traj = MMSE.(belief_hist)
MAP_traj = argmax.(belief_hist)

fig = Figure()
ax = Axis(fig[1,1], xlabel="Timestep", ylabel="ζ", title="Discrete Updater")
hm = heatmap!(ax, reduce(hcat, belief_hist)', colormap=:grayC)
l1 = lines!(ax, zetaHist)
l2 = lines!(ax, MMSE_traj)
l3 = lines!(ax, MAP_traj)
axislegend(
    ax,
    [MarkerElement(color=(:black, 0.5), marker=:rect, markersize=10), l1,l2,l3],
    ["Discrete Belief","True ζ", "MMSE estimate", "MAP Estimate"],
    position=:lt
)
fig

save(joinpath(@__DIR__, "img", "P2_Discrete.pdf"), fig)

K = [7, 50, 180, 350]

begin
    fig = Figure()
    legend_list = nothing
    for (i,k) in enumerate(K)
        ax = Axis(
            fig[i, 1],
            title = "Step $k",
            ylabel = L"\rho_k^+",
            xlabel = i == 4 ? "ζ" : "")
        b = belief_hist[k]
        @show k
        @show MMSE_traj[k]
        @show MAP_traj[k]
        bp = barplot!(ax, eachindex(b), b, color=:black)
        v1 = vlines!(ax, MMSE_traj[k:k], linewidth=3)
        v2 = vlines!(ax, MAP_traj[k:k], linewidth=3)
        isone(i) && (legend_list = [bp, v1, v2])
    end
    Legend(
        fig[:,2],
        legend_list,
        [L"\rho_k^+", L"\hat{\zeta}_{MMSE}", L"\hat{\zeta}_{MAP}"],
        "Discrete Updater")
    fig
end

save(joinpath(@__DIR__, "img", "P2_Discrete_selected.pdf"), fig)
