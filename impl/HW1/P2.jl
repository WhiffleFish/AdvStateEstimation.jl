using AdvStateEstimation
## a

x0 = 0.75

mles = [BernoulliMLE(x0,5), BernoulliMLE(x0,10), BernoulliMLE(x0,50)]

fig = plot(mles)

save(joinpath("img","P2a.svg"), fig)

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

bias_error = rel_error(experimental_bias, analytical_bias)
var_error = rel_error(experimental_var, analytical_var)

## c

μ = 2
σ² = 3
dist = Normal(μ,σ²)
