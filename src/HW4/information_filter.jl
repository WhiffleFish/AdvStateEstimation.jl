struct InformationFilter{MAT,VEC}
    F_inv::MAT
    H::MAT
    Q_inv::MAT
    R_inv::MAT
    M::MAT
    G::MAT
    I_m::MAT
    I_p::MAT
    i_m::VEC
    i_p::VEC
end

function InformationFilter(ss::DTStateSpace, I₀, i₀)
    F_inv = inv(ss.F)
    M = Matrix{Float64}(undef, size(F_inv)...)
    return InformationFilter(F_inv, ss.H, inv(ss.Q), inv(ss.R), M, ss.G, similar(I₀), I₀, similar(i₀), i₀)
end

function predict!(IF::InformationFilter, u)
    (;F_inv, M, Q_inv, i_p, I_p, G) = IF
    M .= F_inv' * I_p * F_inv
    IF.i_m .= (I - M*inv(M + Q_inv))*(F_inv'*i_p + M*G*u)
    IF.I_m .= M - M*inv(M + Q_inv)*M
    return IF.i_m
end

function correct!(IF::InformationFilter, y)
    (;i_m, H, R_inv, I_m) = IF
    IF.i_p .= i_m .+ H'*R_inv*y
    IF.I_p .= I_m .+ H'*R_inv*H
    return IF.i_p
end

function update!(IF::InformationFilter, u, y)
    predict!(IF, u)
    correct!(IF, y)
end

struct IFSimulator{VEC,MAT,F}
    thist::Vector{Float64}
    xhist::Vector{VEC}
    xpredhist::Vector{VEC}
    yhist::Vector{VEC}
    imhist::Vector{VEC}
    iphist::Vector{VEC}
    Imhist::Vector{MAT}
    Iphist::Vector{MAT}
    IF::InformationFilter{MAT, VEC}
    u::F
end

function IFSimulator(IF::InformationFilter, x0, u)
    return IFSimulator(
        Float64[],
        [x0],
        [x0],
        [Float64[]],
        [IF.I_p*x0],
        [IF.I_p*x0],
        [copy(IF.I_p)],
        [copy(IF.I_p)],
        IF,
        u
    )
end

function Base.resize!(IF::IFSimulator, sz::Int)
    resize!(IF.thist, sz)
    resize!(IF.xhist, sz)
    resize!(IF.xpredhist, sz)
    resize!(IF.yhist, sz)
    resize!(IF.imhist, sz)
    resize!(IF.iphist, sz)
    resize!(IF.Imhist, sz)
    resize!(IF.Iphist, sz)
end

function simulate(sim::IFSimulator, ss::DTStateSpace, T::Float64)
    (;thist, xhist, xpredhist, yhist, imhist, iphist, Imhist, Iphist, IF) = sim
    times = 0.0:ss.Δt:T
    resize!(sim, length(times))
    copyto!(thist, times)

    x = first(xhist)

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = noisy_dynamics(ss, x, u)
        y = noisy_meas(ss, x, u)
        update!(IF, u, y)

        xhist[i+1] = x
        xpredhist[i+1] = inv(IF.I_p)*IF.i_p
        yhist[i+1] = y
        imhist[i+1] = copy(IF.i_m)
        iphist[i+1] = copy(IF.i_p)
        Imhist[i+1] = copy(IF.I_m)
        Iphist[i+1] = copy(IF.I_p)
    end
end

function load!(sim::IFSimulator, t::Vector, x::Vector, y::Vector)
    (;thist, xhist, yhist) = sim
    resize!(sim, length(t))
    copyto!(thist, t)
    copyto!(xhist, x)
    @views copyto!(yhist[2:end], y[2:end])
end

function simulate(sim::IFSimulator)
    (;thist, xhist, xpredhist, yhist, imhist, iphist, Imhist, Iphist, IF) = sim

    for (i,t) in enumerate(thist[1:end-1])
        u = sim.u(t)
        x = xhist[i+1]
        y = yhist[i+1]
        update!(IF, u, y)

        xpredhist[i+1] = inv(IF.I_p)*IF.i_p
        imhist[i+1] = copy(IF.i_m)
        iphist[i+1] = copy(IF.i_p)
        Imhist[i+1] = copy(IF.I_m)
        Iphist[i+1] = copy(IF.I_p)
    end
end

covariance(sim::IFSimulator) = inv.(sim.Iphist)
