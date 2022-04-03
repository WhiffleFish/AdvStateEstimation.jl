pf = SE.HW3.ISParticleFilter(P1Prior(),P1Meas(),500)
Y = [91.56, 70.43, 108.67]
SE.HW3.step!(pf, Y)

mmse = SE.HW3.MMSE(pf)
map = SE.HW3.MAP(pf)
max_w = maximum(pf.weights)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x", ylabel="z")
    scatter!(ax,
        pf.particles,
        # color = pf.weights ./ max_w,
        # colormap = :grayC,
        color=[(:red, w/max_w) for w in pf.weights]
    )
    sc1 = scatter!(ax, [mmse], marker=:+, markersize=20, color=:blue)
    sc2 = scatter!(ax, [map], marker=:+, markersize=20, color=:green)
    axislegend(ax, [sc1,sc2], ["MMSE", "MAP"])
    fig
end
save(joinpath(@__DIR__, "img", "P1c_scatter500.pdf"), fig)

x = first.(pf.particles)
y = last.(pf.particles)
z = pf.weights / max_w
m = meshscatter(
    x, y, z,
    color = log.(z.+1e-3),
    colormap = :grayC,
    textsize = 100
)

save(joinpath(@__DIR__, "img", "P1c_scatter3D.pdf"), m)
