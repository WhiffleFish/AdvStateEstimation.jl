Base.@kwdef struct Jalopy
    r   ::Float64 = 25.
    L   ::Float64 = 100.
    dx  ::Float64 = 10.
    dT  ::Float64 = 5.
    Nx  ::Int     = 100
end

function transition(model::Jalopy, i::Int, j::Int)
    Nx = model.Nx
    return if (i == j) && (2 ≤ i ≤ Nx - 1)
        0.3
    elseif (i == j + 1) && (3 ≤ i ≤ Nx) && (j ≥ 2)
        0.5
    elseif (i == j - 1) && (1 ≤ i ≤ Nx - 2) && (j ≤ Nx - 1)
        0.2
    elseif (i == j == 1)
        0.1
    elseif (i == j == Nx)
        0.1
    elseif (i == 2) && (j == 1)
        0.9
    elseif (i == Nx - 1) && (j == Nx)
        0.9
    else
        0.0
    end
end

function transition(model::Jalopy, j::Int)
    p = Vector{Float64}(undef, model.Nx)
    for i in eachindex(p)
        p[i] = transition(model, i, j)
    end
    return Categorical(p)
end

function observation(model::Jalopy, j::Int)
    Nx = model.Nx
    r = model.r
    return Uniform(
        max(1, j - r),
        min(Nx, j + r)
    )
end

## Discrete Filter

struct DiscreteBayesFilter{M}
    model::M
    b::Vector{Float64}
    b′::Vector{Float64}
end

DiscreteBayesFilter(m, b::Vector) = DiscreteBayesFilter(m, b, similar(b))

Distributions.pdf(upd::DiscreteBayesFilter, s) = upd.b[s]

function update!(upd::DiscreteBayesFilter, z)
    model = upd.model
    b = upd.b
    b′ = upd.b′
    for s′ in eachindex(b)
        t_sum = 0.0
        for (s,p) in enumerate(upd.b)
            t_sum += transition(model, s′, s)*p
        end
        b′[s′] = pdf(observation(model, s′), z)*t_sum
    end
    normalize!(b′, 1)
    return copyto!(b, b′)
end

function MMSE(v::Vector{Float64})
    s = 0.0
    for (i,v_i) in enumerate(v)
        s += i*v_i
    end
    return s
end

## SIR Particle Filter
struct SIRParticleFilter{M,S}
    model::M
    particles::Vector{S}
    particle_cache::Vector{S}
    weights::Vector{Float64}
end

function SIRParticleFilter(model, d::Distribution, n::Int)
    ps = rand(d, n)
    return SIRParticleFilter(
        model,
        ps,
        similar(ps),
        Vector{Float64}(undef, length(ps))
    )
end

n_particles(pf::SIRParticleFilter) = length(pf.particles)

function propagate!(pf::SIRParticleFilter)
    model = pf.model
    particles = pf.particles
    for (i,p) in enumerate(pf.particles)
        particles[i] = rand(transition(model, p)) # lots of GC
    end
    return particles
end

function reweight!(pf::SIRParticleFilter, z)
    model = pf.model
    particles = pf.particles
    weights = pf.weights
    for (i,p) in enumerate(pf.particles)
        weights[i] = pdf(observation(model, p), z)
    end
    return particles
end

function resample!(pf::SIRParticleFilter)
    p = pf.particles
    p′ = pf.particle_cache
    w = pf.weights
    ws = sum(w)
    n_p = n_particles(pf)

    r = rand()*ws/n_p
    c = first(w)
    i = 1
    U = r
    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws / n_p
        p′[m] = p[i]
    end
    return copyto!(p, p′)
end

function update!(upd::SIRParticleFilter, z)
    propagate!(upd)
    reweight!(upd, z)
    resample!(upd)
end
