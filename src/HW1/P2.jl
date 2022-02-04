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
