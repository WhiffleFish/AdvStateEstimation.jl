struct BernoulliMLE
    x0::Float64
    N::Int
    Y::Vector{Int}
end

function BernoulliMLE(x0::Float64, N::Int)
    return BernoulliMLE(x0, N, rand(N) .< x0)
end

function log_likelihood(x::Float64, mle::BernoulliMLE)
    m = sum(mle.Y)
    N = mle.N
    return m*log(x) + (N-m)*log(1-x)
end

function log_likelihood(x::AbstractVector{Float64}, mle::BernoulliMLE)
    m = sum(mle.Y)
    N = mle.N
    return @. m*log(x) + (N-m)*log(1-x)
end

MLE(mle::BernoulliMLE) = sum(mle.Y)/mle.N

function CairoMakie.plot(mle_vec::Vector{BernoulliMLE})
    L = length(mle_vec)
    x0_vec = 0.01:0.01:0.99

    fig = Figure()
    for (i,mle) in enumerate(mle_vec)
        ax = Axis(
            fig[i,1],
            title ="N = $(mle.N)",
            xlabel = i === L ? L"x_0" : "",
            ylabel = L"-l(x_0;Y_{1:N})"
        )
        ll = -log_likelihood(x0_vec, mle)
        lines!(ax, x0_vec, ll)
        ml_estimate = MLE(mle)
        ml_likelihood = -log_likelihood(ml_estimate, mle)
        scatter!(ax, [ml_estimate], [ml_likelihood], color=:red)
    end
    return fig
end

##

struct GaussianMLE{D<:Normal}
    dist::D
    N::Int
    Y::Vector{Float64}
end

GaussianMLE(dist::Normal, N::Int) = GaussianMLE(dist, N, rand(dist,N))

StatsBase.mean(mle::GaussianMLE) = sum(mle.Y)/mle.N

StatsBase.var(mle::GaussianMLE, μ::Number) = sum((y - μ)^2 for y in mle.Y)/mle.N

function StatsBase.var(mle::GaussianMLE)
    μ = mean(mle.dist) # default to true mean
    return sum((y - μ)^2 for y in mle.Y)/mle.N
end

function MLE(mle::GaussianMLE)
    return mean(mle), var(mle)
end

function log_likelihood(μ, σ, mle::GaussianMLE)
    return -inv(2*σ^2)*sum((y-μ)^2 for y in mle.Y) - mle.N*log(σ*sqrt(2π))
end

function logspace(x1, x2, N::Int=100)
    return exp.(LinRange(log(x1), log(x2), N))
end

function likelihood_contour(
    mle::GaussianMLE,
    μ_vec::AbstractVector,
    σ2_vec::AbstractVector;
    n_contours::Int=100,
    kwargs...
    )

    ll = [-log_likelihood(μ, sqrt(σ²), mle) for μ in μ_vec, σ² in σ2_vec]
    μ = mean(mle)
    σ² = var(mle, μ)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = L"\hat{\mu}", ylabel = L"\hat{\sigma}^2")
    hm = contour!(
        ax,
        μ_vec,
        σ2_vec,
        ll;
        levels = logspace(extrema(ll)..., n_contours),
        kwargs...)
    scatter!(ax, [μ], [σ²], marker='x', markersize=50)
    Colorbar(fig[1, 2], hm, label = L"-l(Y_{1:N}; \mu, \sigma^2)")
    return fig
end
