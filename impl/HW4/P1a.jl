using AdvStateEstimation.HW3
const SE = AdvStateEstimation

A = [0 1; 0 0]
B = [0;1]
C = [1 0]
D = [0]
Γ = [0;1]
W = 1
V = 1

ss_ct = CTStateSpace(A, B, C, D, Γ, W, V)
ss_dt = c2d(ss_ct)
