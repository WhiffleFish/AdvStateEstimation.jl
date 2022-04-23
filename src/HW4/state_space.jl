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
end

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

    F = Ẑ[(nx+1):end,(nx+1):end]'
    Q = F*Ẑ[1:nx,(nx+1):end]
    R = V*Δt
    return DTStateSpace(F,G,H,M,Q,R)
end

dynamics(ss::DTStateSpace, x, u) = ss.F*x + ss.G*u

noisy_dynamics(ss::DTStateSpace, x, u) = rand(MvNormal(dynamics(ss, x, u), ss.Q))

meas(ss::DTStateSpace, x, u) = ss.H*x + ss.M*u

noisy_meas(ss::DTStateSpace, x, u) = rand(MvNormal(meas(ss, x, u), ss.R))
