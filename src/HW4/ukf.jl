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

function force_symm!(x)
    x .+= x'
    x ./= 2.
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
        χ[i] = x .+ S[i-1,:]
    end

    for j in 2:n+1
        i = j + n
        χ[i] = x .- S[j-1,:]
    end
    return χ
end

function predict!(kf::UnscentedKF, u)
    χ = kf.sigma_points
    n = size(kf.ss.F,1)

    S = cholesky(Symmetric(n*kf.Pkp)).U
    n = statedim(kf)

    gen_sigma_points!(kf, kf.xp, S)

    for i in eachindex(χ)
        χ[i] = dynamics(kf.ss, χ[i], u)
    end

    kf.xm .= mean(χ)

    kf.Pkm .= force_symm!(cov(χ)) .+ kf.ss.Q
    return kf.xm
end


function correct!(kf::UnscentedKF, u, y)
    χ = kf.sigma_points
    n = statedim(kf)
    S̄ = cholesky(kf.Pkm).U

    gen_sigma_points!(kf, kf.xm, S̄)

    γ = Vector{Vector{Float64}}(undef, length(χ))
    for i in eachindex(χ)
        γ[i] = meas(kf.ss, χ[i], u)
    end

    ŷ⁻ = mean(γ)

    Skp1 = force_symm!(cov(γ)) .+ kf.ss.R

    Cxy = zeros(n, length(ŷ⁻))
    for i in eachindex(χ)
        Cxy .+= (χ[i] - kf.xm)*(γ[i] - ŷ⁻)'
    end
    Cxy ./= length(γ)

    K = Cxy/cholesky(Skp1)

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

        xmhist[i+1] = kf.xm
        xphist[i+1] = kf.xp
        Pkmhist[i+1] = copy(kf.Pkm)
        Pkphist[i+1] = copy(kf.Pkp)
    end
    return sim
end
