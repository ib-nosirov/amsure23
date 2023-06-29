using LinearAlgebra
using SparseArrays
using Random
using Plots

Random.seed!(1)
gr(size = (1750, 1565)) # use gr as the plot backend

A = randn(n,n) + diagm(λ)
b = randn(n)

xtrue = A\b

q = 500
h = 0.0001
k = 20

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

println("q = 0")
println("  norm(r) = $(norm(b-A*x))")

r_nrm = zeros(q)
rc_nrm = zeros(q)
AB = zeros(n,k+1)
AB[:,1:k] = A*Y

for s=1:q
    global x, Y, AB

    r = b - A*x
    x = x + h.*r

    Y = Y - h.*AB[:,1:k]

    Q, R = qr(hcat(Y,x))

    B = Matrix(Q)

    Y = B[:,1:k]

    AB = A*B
    c = (B'*AB)\(B'*b)

    rc = b - AB*c

    r_nrm[s] = norm(r)
    rc_nrm[s] = norm(rc)

    println("q = $s")
    println("  norm(b-Ax) = $(r_nrm[s])")
    println("  norm(b-ABc) = $(rc_nrm[s])")
    println("  norm(b-ABc)/norm(b-Ax) = $(rc_nrm[s]/r_nrm[s])")
end

plot([1:q], log.(r_nrm),
title="r_norm",
yaxis=:log10,
xlab="q",
ylab="Accuracy",
label=q,
linewidth=2,
titlefontsize=30,
guidefontsize=30,
tickfontsize=30)

plot!([1:q], log.(rc_nrm),
title="rc_norm",
yaxis=:log10,
xlab="q",
ylab="Accuracy",
label=q,
linewidth=2,
titlefontsize=30,
guidefontsize=30,
tickfontsize=30)
