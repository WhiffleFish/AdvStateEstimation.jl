Base.@kwdef struct Rocket{RT<:AbstractMatrix}
    L::Float64 = 80_000. # m
    R::RT = Symmetric([70. 0; 0 5e-3])
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
        t = (i)*Δt
        ∇h = meas_jac(r, x)*state_jac(r,t)
        start_idx = 1+(np*i)
        Hc[start_idx:(start_idx+np-1),:] .= ∇h
        x = dynamics(r, x)
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

function GaussNewtonNLS(model::Rocket, Y; kwargs...)
    R = model.R
    N = length(Y) ÷ 2
    Rbig_inv = cat((R for _ in 1:N)..., dims=(1,2)) |> Symmetric |> inv
    return GaussNewtonNLS(;model, Y, N, Rbig_inv, kwargs...)
end

function GaussNewtonNLS(model::Rocket, Y)
    return GaussNewtonNLS(Y, model, length(Y))
end

function measvec(r::Rocket, x0::AbstractVector, N::Int)
    v = Vector{Float64}(undef, N*meas_dim(r))
    return measvec!(v, r, x0)
end

function measvec!(v, r::Rocket, x::AbstractVector)
    np = meas_dim(r)
    N = length(v) ÷ np
    for i in 0:(N-1)
        y = meas(r,x)
        start_idx = 1+(np*i)
        v[start_idx:(start_idx+np-1)] .= y
        x = dynamics(r, x)
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
        Jn = dot(y - ŷn, Rbig_inv, y - ŷn)

        if Jn > Jcurr
            α *= β
        else
            x0 = x̂n
            break
        end
    end
    return x0, Jn
end

struct OptHist{S}
    J::Vector{Float64}
    x::Vector{S}
end

function update!(upd::GaussNewtonNLS, x0::S, N::Int; hist::Bool=false) where S
    if hist
        Jhist = Vector{Float64}(undef, N)
        xhist = Vector{S}(undef, N)
        for i in 1:N
            x0, J = update!(upd, x0)
            Jhist[i] = J
            xhist[i] = x0
        end
        return x0, OptHist(Jhist, xhist)
    else
        for i in 1:N
            x0, J = update!(upd, x0)
        end
        return x0
    end
end

function CairoMakie.plot(opt::OptHist)
    plot(opt.J)
end
