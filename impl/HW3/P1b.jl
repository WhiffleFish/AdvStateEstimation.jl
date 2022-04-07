gf = GridFilter(
    P1Trans(),
    P1Meas(),
    ProbGrid([1.:0.05:14.,0.05:0.05:30.], P1Prior())
)

Y = [91.56, 70.43, 108.67]

for i in eachindex(Y)
    @show HW3.marginal_meas!(gf, Y[1:i])
end
