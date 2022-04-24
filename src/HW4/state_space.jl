struct CTStateSpace{T}
    A::T
    B::T
    C::T
    D::T
    Γ::T
    W::Float64
    V::Float64
end

struct DTStateSpace{T}
    F::T
    G::T
    H::T
    M::T
    Q::T
    R::T
    Δt::Float64
end

statedim(ss::DTStateSpace) = size(ss.F,1)

function c2d(ss::CTStateSpace, Δt::Float64)
    (;A,B,C,D,Γ,W,V) = ss
    nx = size(A, 1)
    ny = size(C, 1)
    nu = size(B, 2)

    M = exp(
        [A*Δt B*Δt;
        zeros(nu, nx + nu)])

    F = M[1:nx,1:nx]
    G = M[1:nx, (nx+1):end]
    H = C
    M = D

    Z = Δt*[-A Γ*W*Γ';
            zeros(nx,nx) A']
    Ẑ = exp(Z)

    F = Ẑ[(nx+1):end,(nx+1):end]' |> copy
    Q = F*Ẑ[1:nx,(nx+1):end]
    R = reshape([V*Δt], 1, 1)
    return DTStateSpace(F,G,H,M,Q,R,Δt)
end

dynamics(ss::DTStateSpace, x, u) = ss.F*x .+ ss.G*u

noisy_dynamics(ss::DTStateSpace, x, u) = rand(MvNormal(dynamics(ss, x, u), ss.Q))

meas(ss::DTStateSpace, x, u) = ss.H*x .+ ss.M*u

noisy_meas(ss::DTStateSpace, x, u) = rand(MvNormal(meas(ss, x, u), ss.R))

observation(ss::DTStateSpace, x, u) = MvNormal(meas(ss,x,u), ss.R)

function simulate(ss::DTStateSpace, x, u_f, T)
    times = 0.0:ss.Δt:T |> collect
    xhist = Vector{Vector{Float64}}(undef, length(times))
    yhist = Vector{Vector{Float64}}(undef, length(times))
    xhist[1] = x

    for (i,t) in enumerate(times[1:end-1])
        u = u_f(t)
        x = noisy_dynamics(ss, x, u)
        y = noisy_meas(ss, x, u)

        xhist[i+1] = x
        yhist[i+1] = y
    end
    return times, xhist, yhist
end
