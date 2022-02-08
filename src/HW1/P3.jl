Base.@kwdef struct Rocket
    L::Float64 = 80_000. # m
    R::Matrix{Float64} = [70. 0; 0 5e-3]
    Δt::Float64 = 5.0
end

# Re-doing a lot of RobotDynamics.jl work...
state_dim(::Rocket) = 2
meas_dim(::Rocket) = 4

function dynamics(r::Rocket, x)
    ξ, ξ_dot, a, a_dot = x
    g = 9.80665
    Δt = r.Δt
    return SA[
        ξ + ξ_dot*Δt,
        ξ_dot,
        a + a_dot*Δt - 0.5*g*Δt^2
        ]
end

function meas(r::Rocket, x::AbstractArray)
    ξ, _, a, _ = x
    ρ = sqrt((r.L - ξ)^2 + a^2)
    θ = atan(a/(r.L - ξ))
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

##

Base.@kwdef mutable struct GaussNewtonNLS{M, S <: AbstractVector}
    α::Float64 = 1.0
    β::Float64 = 0.5
    Y::Vector{Float64} = Float64[]
    model::M
    state::S = @SVector zeros(1)
end

function measvec(r::Rocket, x0::AbstractVector, N::Int)
    v = Vector{Float64}(undef, N*measdim(r))
    return measvec!(v, r, x0, N)
end

function measvec!(v, r::Rocket, x::AbstractVector, N::Int)
    L = meas_dim(r)
    for i in 1:L:length(v)
        y = meas(x)
        v[i:(i+L-1)] .= y
        x = dynamics(r, x)
    end
    return v
end

function update!(upd::GaussNewtonNLS) end
