struct BootstrapPF{M,S}
    model::M
    particles::Vector{S}
    particle_cache::Vector{S}
    weights::Vector{Float64}
end

function BootstrapPF(model, dist, n::Int)
    ps = [rand(dist) for _ in 1:n]
    return BootstrapPF(
        model,
        ps,
        similar(ps),
        Vector{Float64}(undef, length(ps)),
    )
end

n_particles(pf::BootstrapPF) = length(pf.particles)

function propagate!(pf::BootstrapPF, u)
    model = pf.model
    particles = pf.particles
    for (i,p) in enumerate(pf.particles)
        particles[i] = noisy_dynamics(model, p, u)
    end
    return particles
end

function reweight!(pf::BootstrapPF, u, z)
    model = pf.model
    particles = pf.particles
    weights = pf.weights
    for (i,p) in enumerate(pf.particles)
        weights[i] = pdf(observation(model, p, u), z)
    end
    return particles
end

function resample!(pf::BootstrapPF)
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

function update!(upd::BootstrapPF, u, z)
    propagate!(upd, u)
    reweight!(upd, u, z)
    resample!(upd)
end

struct PFSimulator{VEC,PF<:BootstrapPF,F}
    thist::Vector{Float64}
    xhist::Vector{VEC}
    yhist::Vector{VEC}
    particle_hist::Vector{Vector{VEC}}
    Ness::Vector{Float64}
    pf::PF
    u::F
end

function PFSimulator(pf::BootstrapPF, x0, u)
    return PFSimulator(
        Float64[],
        [x0],
        [Float64[]],
        [copy(pf.particles)],
        [effective_size(pf.weights)],
        pf,
        u
    )
end

function load!(sim::PFSimulator, t::Vector, x::Vector, y::Vector)
    (;thist, xhist, yhist) = sim
    resize!(sim, length(t))
    copyto!(thist, t)
    copyto!(xhist, x)
    @views copyto!(yhist[2:end], y[2:end])
end

function Base.resize!(sim::PFSimulator, sz::Int)
    resize!(sim.thist, sz)
    resize!(sim.xhist, sz)
    resize!(sim.yhist, sz)
    resize!(sim.particle_hist, sz)
    resize!(sim.Ness, sz)
end

function simulate(sim::PFSimulator, T::Float64)
    (;thist, xhist, yhist, particle_hist, Ness, pf) = sim
    ss = pf.model
    times = 0.0:ss.Δt:T
    resize!(sim, length(times))
    copyto!(thist, times)

    x = first(xhist)

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = noisy_dynamics(ss, x, u)
        y = noisy_meas(ss, x, u)
        update!(pf, u, y)

        xhist[i+1] = x
        yhist[i+1] = y
        particle_hist[i+1] = copy(pf.particles)
        Ness[i+1] = effective_size(pf.weights)
    end
end

function simulate(sim::PFSimulator)
    (;thist, xhist, yhist, particle_hist, Ness, pf) = sim
    ss = pf.model

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = xhist[i+1]
        y = yhist[i+1]
        update!(pf, u, y)

        particle_hist[i+1] = copy(pf.particles)
        Ness[i+1] = effective_size(pf.weights)
    end
end


function MMSE(sim::PFSimulator)
    return mean.(sim.particle_hist)
end

function MAP(sim::PFSimulator)
    return mode.(sim.particle_hist)
end

function effective_size(w::Vector{Float64})
    cv = var(w) / mean(w)^2
    return length(w) / (1+cv)
end

function state_hist(sim::PFSimulator, idx::Int)
    p_hist = sim.particle_hist
    T = length(p_hist)
    n_p = length(first(p_hist))

    x = Vector{Float64}(undef, n_p*T)
    y = Vector{Float64}(undef, n_p*T)
    for (i,ci) in enumerate(CartesianIndices((n_p, T)))
        p, t = Tuple(ci)
        y[i] = p_hist[t][p][idx]
        x[i] = sim.thist[t]
    end
    return x, y
end

function CairoMakie.plot(sim::PFSimulator; Ness=false)
    mmse_hist = MMSE(sim)
    map_hist = MAP(sim)
    n = length(first(map_hist))
    fig = Figure()
    for i in 1:n
        ax = Axis(fig[i,1], ylabel=L"x_{%$i}")
        x, y = state_hist(sim, i)
        scatter!(ax, x, y, color=(:black,0.01), label="Particle Belief")
        lines!(ax, sim.thist, getindex.(sim.xhist, i), lable="True State")
        lines!(sim.thist, getindex.(mmse_hist, i), label="MMSE")
        lines!(sim.thist, getindex.(map_hist, i), label="MAP")
    end
    if Ness
        ax = Axis(fig[n+1,1], ylabel = L"N_{ess}")
        lines!(ax, sim.thist, sim.Ness)
    end
    return fig
end
