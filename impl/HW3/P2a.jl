using AdvStateEstimation
const SE = AdvStateEstimation
model = Jalopy()
Nx = model.Nx

T = [SE.HW3.transition(model, i, j) for i in 1:Nx, j in 1:Nx]

λ, v = eigen(T)
λ

v_I = real(v[:,100])
T*v_I ≈ v_I # true

##
ρ = zeros(Nx)
ρ[[3,19,35]] .= 1/3

K = [10, 50, 180, 300, 600]
m = Matrix{Float64}(undef, Nx, length(K))
for i in eachindex(K)
    m[:,i] .= (T^K[i])*ρ
end

begin
    fig = Figure()
    for i in eachindex(K)
        ax = Axis(
            fig[i, 1],
            title="k = $(K[i])",
            xlabel = i == length(K) ? "ζ" : "",
            ylabel = "ρₖ"
            )
        barplot!(ax, 1:Nx, @view m[:,i])
    end
    fig
end

save(joinpath(@__DIR__, "img", "P2b.pdf"), fig)
