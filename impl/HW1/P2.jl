using AdvStateEstimation
## a

x0 = 0.75

mles = [BernoulliMLE(x0,5), BernoulliMLE(x0,10), BernoulliMLE(x0,50)]

fig = plot(mles)

save(joinpath(@__DIR__, "img","P2a.svg"), fig)

## b

rel_error(x̂, x) = abs(x - x̂)/x

N = 5000
Ny = 10
mle_vec = [BernoulliMLE(x0, Ny) for _ in 1:N]

estimates = MLE.(mle_vec)
μ_MLE, var_MLE = mean_and_var(estimates)

experimental_bias = μ_MLE - x0
analytical_bias = 0.0

experimental_var = var_MLE
analytical_var = x0*(1-x0)/Ny

mean_error = rel_error(μ_MLE, x0)
var_error = rel_error(experimental_var, analytical_var)

## c
p2cplot(m) = likelihood_contour(
    m,
    LinRange(-3,3,1000),
    LinRange(0.5,8,1000);
    n_contours = 100,
    colormap = :magma
)

μ = 2
σ² = 3
dist = Normal(μ,sqrt(σ²))

mle = GaussianMLE(dist, 5)
fig = p2cplot(mle)
save(joinpath(@__DIR__, "img","P2C5.svg"), fig)

mle = GaussianMLE(dist, 10)
fig = p2cplot(mle)

save(joinpath(@__DIR__, "img","P2C10.svg"), fig)

mle = GaussianMLE(dist, 50)
fig = p2cplot(mle)

save(joinpath(@__DIR__, "img","P2C50.svg"), fig)

## d
μ = 2
σ² = 3
dist = Normal(μ,sqrt(σ²))
N = 5000
mles = [GaussianMLE(dist, 10) for _ in 1:N]
estimates = MLE.(mles)

μ_MLE, v = mean_and_var(first.(estimates))

estim_var = σ² / 10

rel_error(μ_MLE, μ)
rel_error(v, estim_var)
