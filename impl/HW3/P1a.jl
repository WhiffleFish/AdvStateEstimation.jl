using AdvStateEstimation
using AdvStateEstimation.HW3
const SE = AdvStateEstimation

struct P1Meas end

Distributions.pdf(m::P1Meas, x, y::Float64) = pdf(Uniform(x[1],(x[1]+x[2])^2), y)
function Distributions.pdf(m::P1Meas, x, y::Vector{Float64})
    p = 1.0
    for y_i in y
        p *= pdf(Uniform(x[1],(x[1]+x[2])^2), y_i)
    end
    return p
end

struct P1Trans end

function Distributions.pdf(t::P1Trans, x′, x)
    return Float64(x ≈ x′)
end

struct P1Prior end

function Distributions.pdf(p::P1Prior, x)
    px = pdf(Uniform(1,11), x[1])
    pz = pdf(Gamma(2,2), x[2])
    return px*pz
end

Base.rand(prior::P1Prior) = SA[rand(Uniform(1,11)), rand(Gamma(2,2))]

SE.HW3.predict!(gf::GridFilter{P1Trans,P1Meas}) = gf.grid.prob

gf = GridFilter(
    P1Trans(),
    P1Meas(),
    ProbGrid([1.:0.05:14.,0.05:0.05:30.], P1Prior())
)

HW3.marginal_meas!(gf, Y[1:3])

Y = [91.56, 70.43, 108.67]

fig = Figure()
for i in 1:4
    xlabel = i == 4 ? "x" : ""
    ax = Axis(fig[i, 1], xlabel=xlabel, ylabel="z", limits=(nothing, nothing, 0.05, 10.))
    heatmap!(ax, gf.grid.grid...,gf.grid.prob; colormap=:magma)
    i < 4 && step!(gf, Y[i])
end
fig
save(joinpath(@__DIR__, "img", "P1a_heatmap.pdf"), fig)

##
pg = ProbGrid([1.:0.05:14.,0.05:0.05:30.], P1Prior())

gf = GridFilter(
    P1Trans(),
    P1Meas(),
    pg
)

Y = [91.56, 70.43, 108.67]
@show SE.HW3.MAP(gf)
@show SE.HW3.MMSE(gf)
for i in eachindex(Y)
    @show i
    step!(gf, Y[i])
    heatmap(
        pg.grid...,
        pg.prob;
        colormap = :magma,
        xlabel = "x",
        ylabel = "z",
        axis = (;limits=(nothing, nothing, 0.05, 10.))
    ) |> display

    @show SE.HW3.MAP(gf)
    @show SE.HW3.MMSE(gf)
end
