Base.@kwdef struct StudentRocket{R<:Rocket, RD<:Distribution, TD<:Distribution}
    r::R = Rocket()
    vρ::RD = TDist(0.5)
    vθ::TD = Normal(0, 5e-3)
end

function easting_alt(L, ρ, θ)
    tθ = tan(θ)
    x = ρ*cos(θ)
    a = ρ*sin(θ)
    ξ = L - x
    return ξ, a
end

function easting_alt(L, v::AbstractVector)
    np = 2
    N = length(v) ÷ np
    v′ = Matrix{Float64}(undef, N, 2)
    for i in 0:(N-1)
        start_idx = 1+(np*i)
        end_idx = (start_idx+np-1)
        ρ, θ = v[start_idx:end_idx]
        ξ, a = easting_alt(L, ρ, θ)
        v′[i+1,:] .= SA[ξ,a]
    end
    return v′
end

function noisy_meas(r::Rocket, x::AbstractArray, dist_vec::Vector{Distribution{Univariate, Continuous}})
    noise = SA[rand(first(dist_vec)), rand(last(dist_vec))]
    return meas(r,x) .+ noise
end

function noisy_measvec!(v, r::Rocket, x::AbstractVector, dist_vec::Vector{Distribution{Univariate, Continuous}})
    np = meas_dim(r)
    N = length(v) ÷ np
    for i in 0:(N-1)
        x = dynamics(r, x)
        y = noisy_meas(r,x, dist_vec)
        start_idx = 1+(np*i)
        v[start_idx:(start_idx+np-1)] .= y
    end
    return v
end

function noisy_measvec(r::Rocket, x0::AbstractVector, N::Int, dist_vec::Vector{Distribution{Univariate, Continuous}})
    v = Vector{Float64}(undef, N*meas_dim(r))
    return noisy_measvec!(v, r, x0, dist_vec)
end

function MCTrials(
    upd::GaussNewtonNLS,
    x0_true::S,
    x0_guess::S,
    N_steps,
    N_samples,
    dist_vec::Vector{Distribution{Univariate, Continuous}}) where S

    xs = Vector{S}(undef, N_samples)
    for i in 1:N_samples
        noisy_measvec!(upd.Y, upd.model, x0_true, dist_vec)
        x = update!(upd, x0_guess, N_steps)
        xs[i] = x
    end
    return GaussNewtonMCTrials(upd.model, xs)
end

##
function likelihood(sr::StudentRocket, x0, Y)
    r = sr.r
    N = div(length(Y),2)
    l = 1.0
    x = x0
    for i in 0:(N-1)
        start_idx = 1+(2*i)
        end_idx = (start_idx+2-1)

        x = dynamics(r, x)
        ρ_pred, θ_pred = meas(r,x)
        ρ, θ = @view Y[start_idx:end_idx]

        P_ρ = pdf(sr.vρ, ρ - ρ_pred)
        P_θ = pdf(sr.vρ, θ - θ_pred)
        l *= P_ρ
        l *= P_θ
    end
    return l
end

function log_likelihood(sr::StudentRocket, x0, Y)
    r = sr.r
    N = div(length(Y),2)
    l = 0.0
    x = x0
    for i in 0:(N-1)
        start_idx = 1+(2*i)
        end_idx = (start_idx+2-1)

        x = dynamics(r, x)
        ρ_pred, θ_pred = meas(r,x)
        ρ, θ = @view Y[start_idx:end_idx]

        P_ρ = pdf(sr.vρ, ρ - ρ_pred)
        P_θ = pdf(sr.vρ, θ - θ_pred)
        l += log(P_ρ)
        l += log(P_θ)
    end
    return l
end
