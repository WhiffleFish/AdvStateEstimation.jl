Base.@kwdef struct Rocket{RT<:AbstractMatrix}
    L::Float64 = 80_000. # m
    R::RT = Symmetric(SA[70. 0; 0 5e-3])
    Δt::Float64 = 5.0
end

# Re-doing a lot of RobotDynamics.jl work...
state_dim(::Rocket) = 4
meas_dim(::Rocket) = 2

function dynamics(r::Rocket, x)
    ξ, ξ_dot, a, a_dot = x
    g = 9.80665
    Δt = r.Δt
    return SA[
        ξ + ξ_dot*Δt,
        ξ_dot,
        a + a_dot*Δt - 0.5*g*Δt^2,
        a_dot - g*Δt
        ]
end

function meas(r::Rocket, x::AbstractArray)
    ξ, _, a, _ = x
    ρ = sqrt((r.L - ξ)^2 + a^2)
    θ = atan(a,r.L - ξ)
    return SA[ρ, θ]
end

function noisy_meas(r::Rocket, x::AbstractArray)
    R = r.R
    dist = MvNormal(SA[0,0], R)
    return meas(r,x) + SVector{2,Float64}(rand(dist))
end

function meas_jac(r::Rocket, x::AbstractArray)
    ξ, _, a, _ = x
    L = r.L

    p_denom = sqrt((L - ξ)^2 + a^2)
    dpdξ = -(L - ξ) / p_denom
    dpda = a / p_denom

    θdenom = ((a/(L-ξ))^2 + 1)
    dθdξ = a / (θdenom*(L-ξ)^2)
    dθda = 1 / (θdenom*(L-ξ))
    return SA[
        dpdξ 0. dpda 0.;
        dθdξ 0. dθda 0.
        ]
end

function state_jac(r::Rocket, t)
    return SA[
        1 t 0 0;
        0 1 0 0;
        0 0 1 t;
        0 0 0 1
    ]
end

function stack_meas_jac(r::Rocket, x, N)
    np = meas_dim(r)
    n = state_dim(r)
    Δt = r.Δt

    Hc = Matrix{Float64}(undef, N*np, n)
    for i in 0:(N-1)
        x = dynamics(r, x)
        t = (i)*Δt
        ∇h = meas_jac(r, x)*state_jac(r,t)
        start_idx = 1+(np*i)
        Hc[start_idx:(start_idx+np-1),:] .= ∇h
    end
    return Hc
end

##

Base.@kwdef mutable struct GaussNewtonNLS{M, R<:AbstractMatrix}
    model::M
    Y::Vector{Float64}
    N::Int
    Rbig_inv::R
    α::Float64 = 1.0
    β::Float64 = 0.5
    max_splits::Int = 100
    _y_cache::Vector{Float64} = similar(Y)
end

function blkdiag(A::AbstractMatrix, N::Int)
    return cat(Iterators.repeated(A,N)..., dims=(1,2))
end

function GaussNewtonNLS(model::Rocket, Y::AbstractVector; kwargs...)
    R = model.R
    N = length(Y) ÷ 2
    Rbig_inv = blkdiag(R,N) |> Symmetric |> inv
    return GaussNewtonNLS(;model, Y, N, Rbig_inv, kwargs...)
end

function GaussNewtonNLS(model::Rocket, Y)
    return GaussNewtonNLS(Y, model, length(Y))
end

function measvec(r::Rocket, x0::AbstractVector, N::Int)
    v = Vector{Float64}(undef, N*meas_dim(r))
    return measvec!(v, r, x0)
end

function noisy_measvec(r::Rocket, x0::AbstractVector, N::Int)
    v = Vector{Float64}(undef, N*meas_dim(r))
    return noisy_measvec!(v, r, x0)
end

function measvec!(v, r::Rocket, x::AbstractVector)
    np = meas_dim(r)
    N = length(v) ÷ np
    for i in 0:(N-1)
        x = dynamics(r, x)
        y = meas(r,x)
        start_idx = 1+(np*i)
        v[start_idx:(start_idx+np-1)] .= y
    end
    return v
end

function noisy_measvec!(v, r::Rocket, x::AbstractVector)
    np = meas_dim(r)
    N = length(v) ÷ np
    for i in 0:(N-1)
        x = dynamics(r, x)
        y = noisy_meas(r,x)
        start_idx = 1+(np*i)
        v[start_idx:(start_idx+np-1)] .= y
    end
    return v
end

function update!(upd::GaussNewtonNLS, x0)
    model = upd.model
    Rbig_inv = upd.Rbig_inv
    y = upd.Y

    ŷ = measvec!(upd._y_cache, model, x0)
    Hc = stack_meas_jac(model, x0, upd.N)
    Jcurr = dot(y - ŷ, Rbig_inv, y - ŷ)
    Δx = inv(transpose(Hc)*Rbig_inv*Hc)*transpose(Hc)*Rbig_inv*(y - ŷ)

    α = upd.α
    β = upd.β
    Jn = 0.
    for _ in 1:upd.max_splits
        x̂n = x0 + α*Δx
        ŷn = measvec!(upd._y_cache, model, x̂n)
        @. upd._y_cache = y - ŷn
        Jn = dot(upd._y_cache, Rbig_inv, upd._y_cache)

        if Jn > Jcurr
            α *= β
        else
            x0 = x̂n
            break
        end
    end
    return x0, Jn
end

struct OptHist{M,S}
    model::M
    J::Vector{Float64}
    x::Vector{S}
end

function update!(upd::GaussNewtonNLS, x0::S, N::Int; hist::Bool=false) where S
    if hist
        Jhist = Vector{Float64}(undef, N+1)
        xhist = Vector{S}(undef, N+1)

        y = upd.Y
        ŷ = measvec!(upd._y_cache, upd.model, x0)
        Jcurr = dot(y - ŷ, upd.Rbig_inv, y - ŷ)
        Jhist[1] = Jcurr
        xhist[1] = x0

        for i in 2:(N+1)
            x0, J = update!(upd, x0)
            Jhist[i] = J
            xhist[i] = x0
        end
        return x0, OptHist(upd.model, Jhist, xhist)
    else
        for i in 1:N
            x0, J = update!(upd, x0)
        end
        return x0
    end
end

struct GaussNewtonMCTrials{M,D<:AbstractVector}
    model::M
    data::D
end

bias(d::GaussNewtonMCTrials, x0true) = mean(d.data) - x0true
error_cov(d::GaussNewtonMCTrials, x0true) = cov(map(x->x-x0true, d.data))

function MCTrials(upd::GaussNewtonNLS, x0_true::S, x0_guess::S, N_steps, N_samples) where S
    xs = Vector{S}(undef, N_samples)
    for i in 1:N_samples
        noisy_measvec!(upd.Y, upd.model, x0_true)
        x = update!(upd, x0_guess, N_steps)
        xs[i] = x
    end
    return GaussNewtonMCTrials(upd.model, xs)
end

function CairoMakie.plot(opt::OptHist; kwargs...)
    np = meas_dim(opt.model)
    fig = Figure(; kwargs...)#resolution=(800,600))
    ax1 = Axis(
        fig[1, 1],
        yscale = log10,
        yminorgridvisible = true,
        title = "Objective Function",
        xlabel = "Gauss-Newton Iteration",
        ylabel = L"J(x)"
    )
    plot!(ax1, opt.J)
    x0 = last(opt.x)
    Y = Iterators.partition(measvec(opt.model, x0, 40), np) |> collect
    for idx in 1:np
        ax = Axis(fig[idx+1,1])
        plot!(ax, getindex.(Y,idx), marker='o', color=:blue, markersize=15)
    end
    return fig
end

function CairoMakie.plot(opt::OptHist{M}, ytrue::AbstractArray; kwargs...) where M <: Rocket
    np = meas_dim(opt.model)
    fig = Figure(;kwargs...)
    ax1 = Axis(
        fig[1, 1],
        yscale = log10,
        yminorgridvisible = true,
        title = "Objective Function",
        xlabel = "Gauss-Newton Iteration",
        ylabel = L"J(x)"
    )
    lines!(ax1, opt.J)
    x0 = last(opt.x)
    Y = Iterators.partition(measvec(opt.model, x0, 40), np) |> collect
    Ytrue = Iterators.partition(ytrue, np) |> collect
    labels = ["ρ","θ"]
    for idx in 1:np
        ax = Axis(
            fig[idx+1,1],
            xlabel = L"t_k",
            ylabel = labels[idx])
        plot!(ax, getindex.(Y,idx), marker='o', color=:blue, markersize=15)
        plot!(ax, getindex.(Ytrue,idx), marker='x', color=:red, markersize=15)
    end
    return fig
end

function plotx(opt::OptHist{M}, xtrue::AbstractVector; kwargs...) where M <: Rocket
    fig = Figure()
    X = opt.x
    axes = [Axis(fig[1, 1], ylabel=L"\xi"), Axis(fig[1, 2], ylabel=L"\dot{\xi}"),
            Axis(fig[2, 1], ylabel=L"a"), Axis(fig[2, 2], ylabel=L"\dot{a}")]

    labels = [L"\xi", L"\dot{\xi}", L"a", L"\dot{a}"]
    for (i,ax) in enumerate(axes)
        Xi = getindex.(X,i)
        abline!(ax, xtrue[i], 0; ls =:dash, color=:red)
        lines!(ax, Xi; marker='⊡', markersize=15)
    end
    return fig
end

function CairoMakie.plot(sims::GaussNewtonMCTrials{M}, x0true; kwargs...) where M <: Rocket
    xs = sims.data

    fig = Figure(;kwargs...)

    axes = [
        Axis(fig[1, 1], title=L"\xi"), Axis(fig[1, 2], title=L"\dot{\xi}"),
        Axis(fig[2, 1], title=L"a"), Axis(fig[2, 2], title=L"\dot{a}")
    ]

    for (idx, ax) in enumerate(axes)
        estimates = getindex.(xs,idx)
        hist!(ax,estimates)
        vlines!(ax, [x0true[idx]]; color=:green)
        vlines!(ax, [mean(estimates)]; color=:red, linestyle=:dash)
    end
    return fig
end
