using AdvStateEstimation
const SE = AdvStateEstimation
using AdvStateEstimation.HW3

vars = matread(joinpath(@__DIR__, "asen6519hw3_problem2.mat"))
zMeasHist = vars["zMeasHist"] |> vec
zetaHist = vars["zetaHist"] |> vec

model = Jalopy()
b = zeros(model.Nx)
b[[3,19,35]] .= 1/3
b0 = Categorical(b)
pf = SIRParticleFilter(model, b0, 500)

belief_hist = Vector{Int}[]
push!(belief_hist, copy(pf.particles))
for z in zMeasHist[2:end]
    AdvStateEstimation.HW3.update!(pf, z)
    push!(belief_hist, copy(pf.particles))
end
x_hist = reduce(vcat, fill(i,500) for i in 1:length(belief_hist))
p_hist = reduce(vcat, belief_hist)
MMSE_traj = mean.(belief_hist)
MAP_traj = mode.(belief_hist)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Timestep", ylabel="ζ", title="SIR Particle Filter")
scatter!(ax, x_hist, p_hist, color=(:black, 0.01), markersize=5)
l1 = lines!(ax, zetaHist)
l2 = lines!(ax, MMSE_traj)
l3 = lines!(ax, MAP_traj)
axislegend(
    ax,
    [MarkerElement(color=:black, marker=:circle, markersize=5), l1,l2,l3],
    ["Particle Belief","True ζ", "MMSE estimate", "MAP Estimate"],
    position=:lt
)
fig
save(joinpath(@__DIR__, "img", "P2_Particle.pdf"), fig)


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
        bp = hist!(ax, b, color=:black, normalization=:pdf, bins=0:100)
        v1 = vlines!(ax, MMSE_traj[k:k], linewidth=3)
        v2 = vlines!(ax, MAP_traj[k:k], linewidth=3)
        isone(i) && (legend_list = [bp, v1, v2])
    end
    Legend(
        fig[:,2],
        legend_list,
        [L"\rho_k^+", L"\hat{\zeta}_{MMSE}", L"\hat{\zeta}_{MAP}"],
        "SIR Particle Filter")
    fig
end
save(joinpath(@__DIR__, "img", "P2_Particle_selected.pdf"), fig)
