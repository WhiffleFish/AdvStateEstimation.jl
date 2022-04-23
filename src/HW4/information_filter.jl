struct InformationFilter{MAT,VEC}
    F_inv::MAT
    H::MAT
    Q_inv::MAT
    R_inv::MAT
    M::MAT
    I_m::MAT
    I_p::MAT
    i_m::VEC
    i_p::VEC
end

function InformationFilter(ss::DiscreteStateSpace, I₀, i₀)
    F_inv = inv(ss.F)
    M = F_inv*I₀*F_inv
    return InformationFilter(F_inv, ss.H, inv(ss.Q), inv(ss.R), M, similar(I₀), I₀, similar(i₀), i₀)
end

function predict!(IF::InformationFilter)
    IF.i_m .= (I - M*inv(M + Q_inv))*(F_inv'*i_p + M*G*u)
    IF.I_m .= M - M*inv(M + Q_inv)*M
    return IF.i_m
end

function update!(IF::InformationFilter, y)
    IF.i_p .= i_m .+ H'*R_inv*y
    IF.I_p .= I_m .+ H'*R_inv*H
    return IF.i_p
end

struct IFSimulator{X,Y,I,F}
    xhist::X
    yhist::Y
    Ihist::I
    u::F
end
