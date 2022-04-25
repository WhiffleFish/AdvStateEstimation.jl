struct UnscentedKF{SS<:DTStateSpace, V}
    ss::SS
    sigma_points::Vector{V}
    κ::Float64
    α::Float64
    β::Float64
    xm::Vector{Float64}
    xp::Vector{Float64}
    wm::Vector{Float64}
    wc::Vector{Float64}
    Pkm::Matrix{Float64}
    Pkp::Matrix{Float64}
end

function UnscentedKF(ss::DTStateSpace, x0, P0; α=0.01, β=2.0, κ=0.0)
    n = statedim(ss)
    sigma_points = Vector{Vector{Float64}}(undef, 2n+1)
    λ = α^2 * (n + κ) - n
    wm = zeros(2n+1)
    wm[1] = λ / (n + λ)
    wm[2:end] .= 1 / (2*(n+λ))

    wc = zeros(2n+1)
    wc[1] = (λ / (n + λ)) + 1 - α^2 + β
    wc[2:end] .= wm[2:end]

    return UnscentedKF(ss, sigma_points, κ, α, β, similar(x0), x0, wm, wc, similar(P0), P0)
end

function force_symm!(x)
    x .+= x'
    x ./= 2.
end

statedim(kf::UnscentedKF) = size(kf.Pkp, 1)

function lambda(kf::UnscentedKF)
    (;α,κ) = kf
    n = statedim(kf)
    return α^2 * (n + κ) - n
end

function gen_sigma_points!(kf::UnscentedKF, x, S)
    λ = lambda(kf)
    n = statedim(kf)
    χ = kf.sigma_points

    χ[1] = x

    for i in 2:n+1
        χ[i] = x .+ sqrt(n+λ)*S[i-1,:]
    end

    for j in 2:n+1
        i = j + n
        χ[i] = x .- sqrt(n+λ)*S[j-1,:]
    end
    return χ
end

function predict!(kf::UnscentedKF, u)
    χ = kf.sigma_points
    force_symm!(kf.Pkp)
    S = cholesky(kf.Pkp).U
    n = statedim(kf)

    gen_sigma_points!(kf, kf.xp, S)

    χ_bar = similar(χ)
    for (i,χ_i) in enumerate(χ)
        χ_bar[i] = dynamics(kf.ss, χ_i, u)
    end

    kf.xm .= 0.0
    for (i,χ_i) in enumerate(χ_bar)
        kf.xm .+= kf.wm[i]*χ_i
    end

    kf.Pkm .= 0.0
    for i in eachindex(χ_bar)
        kf.Pkm .+= kf.wc[i]*(χ_bar[i] - kf.xm)*(χ_bar[i] - kf.xm)'
    end
    kf.Pkm .+= kf.ss.Q
    return kf.xm
end

function correct!(kf::UnscentedKF, u, y)
    χ = kf.sigma_points
    n = statedim(kf)
    force_symm!(kf.Pkm)
    S̄ = cholesky(kf.Pkm).U

    gen_sigma_points!(kf, kf.xm, S̄)

    γ = Vector{Vector{Float64}}(undef, length(χ))
    for i in eachindex(χ)
        γ[i] = meas(kf.ss, χ[i], u)
    end

    ŷ⁻ = zero(first(γ))
    for i in eachindex(γ)
        ŷ⁻ += kf.wm[i]*γ[i]
    end

    Skp1 = zeros(length(ŷ⁻), length(ŷ⁻))
    for i in eachindex(γ)
        Skp1 .+= kf.wc[i]*(γ[i] - ŷ⁻)*(γ[i] - ŷ⁻)'
    end
    Skp1 .+= kf.ss.R

    Cxy = zeros(n, length(ŷ⁻))
    for i in eachindex(χ)
        Cxy .+= kf.wc[i]*(χ[i] - kf.xm)*(γ[i] - ŷ⁻)'
    end

    K = Cxy*inv(Skp1)

    kf.xp .= kf.xm .+ K*(y - ŷ⁻)
    kf.Pkp .= kf.Pkm .- K*Skp1*K'

    return kf.xp
end

function update!(kf::UnscentedKF, u, y)
    predict!(kf, u)
    correct!(kf, u, y)
end


struct KFSimulator{VEC,MAT,KF<:UnscentedKF,F}
    thist::Vector{Float64}
    xhist::Vector{VEC}
    xmhist::Vector{VEC}
    xphist::Vector{VEC}
    yhist::Vector{VEC}
    Pkmhist::Vector{MAT}
    Pkphist::Vector{MAT}
    kf::KF
    u::F
end

function KFSimulator(kf::UnscentedKF, x0, u)
    return KFSimulator(
        Float64[],
        [copy(x0)],
        [Float64[]],
        [copy(kf.xp)],
        [Float64[]],
        [copy(kf.Pkm)],
        [copy(kf.Pkp)],
        kf,
        u
    )
end

function load!(sim::KFSimulator, t::Vector, x::Vector, y::Vector)
    (;thist, xhist, yhist) = sim
    resize!(sim, length(t))
    copyto!(thist, t)
    copyto!(xhist, x)
    @views copyto!(yhist[2:end], y[2:end])
end

function Base.resize!(sim::KFSimulator, sz::Int)
    resize!(sim.thist, sz)
    resize!(sim.xhist, sz)
    resize!(sim.xmhist, sz)
    resize!(sim.xphist, sz)
    resize!(sim.yhist, sz)
    resize!(sim.Pkmhist, sz)
    resize!(sim.Pkphist, sz)
end

function simulate(sim::KFSimulator, T::Float64)
    (;thist, xhist, xmhist, xphist, yhist, Pkmhist, Pkphist, kf) = sim
    ss = kf.ss
    times = 0.0:ss.Δt:T
    resize!(sim, length(times))
    copyto!(thist, times)

    x = first(xhist)

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = noisy_dynamics(ss, x, u)
        y = noisy_meas(ss, x, u)
        update!(kf, u, y)

        xhist[i+1] = x
        xmhist[i+1] = copy(kf.xm)
        xphist[i+1] = copy(kf.xp)
        yhist[i+1] = y
        Pkmhist[i+1] = copy(kf.Pkm)
        Pkphist[i+1] = copy(kf.Pkp)
    end
end

function simulate(sim::KFSimulator)
    (;thist, xhist, xmhist, xphist, yhist, Pkmhist, Pkphist, kf) = sim
    ss = kf.ss

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = xhist[i+1]
        y = yhist[i+1]
        update!(kf, u, y)

        xmhist[i+1] = copy(kf.xm)
        xphist[i+1] = copy(kf.xp)
        Pkmhist[i+1] = copy(kf.Pkm)
        Pkphist[i+1] = copy(kf.Pkp)
    end
    return sim
end

function plot_error(sim::KFSimulator)
    Pkp = diag.(sim.Pkphist)
    n = statedim(sim.kf)
    fig = Figure()
    for i in 1:n
        ax = Axis(fig[i , 1], xlabel="t", ylabel="x_$i error")
        σ = sqrt.(getindex.(Pkp, i))
        e = getindex.(sim.xphist, i) .- getindex.(sim.xhist, i)
        scatter!(ax, sim.thist, e, markersize=5)
        lines!(sim.thist, 2 .* σ, color=:red, linestyle=:dash)
        lines!(sim.thist, - 2 .* σ, color=:red, linestyle=:dash)
    end
    return fig
end

function measurements(ss, times, xm, u_f)
    y_vec = Vector{Vector{Float64}}(undef, length(xm))
    for (i,t) in enumerate(times[1:end-1])
        y_vec[i+1] = meas(ss, xm[i+1], u_f(t))
    end
    return y_vec
end

function plot_meas(sim::KFSimulator)
    Pkm = diag.(sim.Pkmhist)
    ŷ = measurements(sim.kf.ss, sim.thist, sim.xmhist, sim.u)[2:end]
    y = sim.yhist[2:end]
    e_y = ŷ .- y
    p = length(first(ŷ))
    fig = Figure()
    for i in 1:p
        ax = Axis(fig[p, 1], xlabel="t", ylabel=L"e_{y,%$i}")
        scatter!(ax, sim.thist[2:end], getindex.(e_y, p))
        σ = sqrt.(getindex.(Pkm, p)[2:end])
        lines!(sim.thist[2:end], 2 .* σ, color=:red, linestyle=:dash)
        lines!(sim.thist[2:end], -2 .* σ, color=:red, linestyle=:dash)
    end
    return fig
end
