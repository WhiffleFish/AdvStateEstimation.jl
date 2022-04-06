struct ProbGrid{N, G}
    prob    :: Array{Float64, N}
    grid    :: Vector{G}
end

function Base.getindex(pg::ProbGrid{N}, ci::CartesianIndex) where N
    t = Tuple(ci)
    return SVector{N, Float64}(pg.grid[i][t[i]] for i in 1:N)
end

Base.size(pg::ProbGrid) = size(pg.prob)

function ProbGrid(loc)
    Ls = Tuple(length(d)-1 for d in loc)
    numel = prod(Ls)
    return ProbGrid(fill(inv(numel), Ls), loc)
end

function ProbGrid(loc::AbstractVector{<:AbstractRange}, prior)
    Ls = Tuple(length(d)-1 for d in loc)
    L = length(Ls)
    prob = Matrix{Float64}(undef, Ls)
    Δx = _dx(loc)
    x = similar(Δx)
    for ci in CartesianIndices(size(prob))
        x .= (loc[i][ci[i]] for i in 1:L) .+ Δx./2
        prob[ci] = pdf(prior, x)
    end
    return ProbGrid(prob, loc)
end

function _dx(v::AbstractVector{<:AbstractRange})
    return [step(v[i]) for i in eachindex(v)]
end

function _dx(grid::ProbGrid{N,<:AbstractRange}) where N
    SVector{N,Float64}(step(grid.grid[i]) for i in 1:N)
end

struct GridFilter{T,M,G<:ProbGrid}
    trans::T # Tiny
    meas::M # Meat
    grid::G # Gang
end

Base.size(gf::GridFilter) = size(gf.grid)

function predict!(gf::GridFilter)
    CI = CartesianIndices(size(gf))
    Δx = _dx(gf.grid)
    pg = gf.grid
    prob = pg.prob
    prob′ = similar(prob) # FIXME: alloc, cache maybe
    for ci_i in CI
        w_sum = 0.0
        x_kp1 = pg[ci_i] + Δx/2
        for ci_j in CI
            x_k = pg[ci_j] + Δx/2
            w_sum += prob[ci_j]*pdf(gf.trans, x_kp1, x_k)
        end
        prob′[ci_i] = w_sum
    end
    copyto!(prob, prob′)
    normalize!(prob, 1)
end

function update!(gf::GridFilter, y)
    CI = CartesianIndices(size(gf))
    Δx = _dx(gf.grid)
    pg = gf.grid
    prob = pg.prob
    for ci in CI
        prob[ci] *= pdf(gf.meas, pg[ci] + Δx/2, y)
    end
    normalize!(prob, 1)
end

function marginal_meas!(gf::GridFilter, Y)
    CI = CartesianIndices(size(gf))
    Δx = _dx(gf.grid)
    ΔA = prod(Δx)
    pg = gf.grid
    prob = pg.prob

    py = 0.0
    for ci in CI
        pX = prob[ci]
        p = 1.0
        for y in Y
            p *= pdf(gf.meas, pg[ci] + Δx/2, y)
        end
        py += p*pX*ΔA
    end
    return py
end

function step!(gf::GridFilter, y)
    predict!(gf)
    update!(gf, y)
end

## (b) Point Estimates

function MAP(gf::GridFilter)
    prob = gf.grid.prob
    idx = argmax(prob)
    return gf.grid[idx]
end

function MMSE(gf::GridFilter)
    x = zeros(Float64, length(size(gf)))
    pg = gf.grid
    prob = pg.prob
    grid = pg.grid
    normalize!(prob, 1)
    for ci in CartesianIndices(size(gf))
        x += prob[ci]*pg[ci]
    end
    return x
end


## (c) Importance Sampling

struct ISParticleFilter{Q,P,S}
    q::Q # proposal q(x;Y)
    p::P # joint p(x,Y)
    particles::Vector{S}
    weights::Vector{Float64}
end

ISParticleFilter(q,p,n::Int) = ISParticleFilter(q, p, [rand(q) for _ in 1:n], Vector{Float64}(undef, n))

function step!(pf::ISParticleFilter, y)
    ps = pf.particles
    for i in eachindex(pf.weights)
        p = ps[i]
        pf.weights[i] = pdf(pf.p, p, y) / pdf(pf.q, p)
    end
    normalize!(pf.weights, 1)
end

function MMSE(pf::ISParticleFilter)
    x = zero(first(pf.particles))
    for (p,w) in zip(pf.particles, pf.weights)
        x += w*p
    end
    return x
end

function MAP(pf::ISParticleFilter)
    return pf.particles[argmax(pf.weights)]
end
