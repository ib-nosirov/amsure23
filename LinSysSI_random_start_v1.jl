using LinearAlgebra
using SparseArrays
using Random
using Plots

Random.seed!(1)
gr(size=(1800,1600))

n = 10000
λ = @. 10 + (1:n)
A = randn(n,n) + diagm(λ)
b = randn(n)

xtrue = A\b

q = 2000
h = 0.0001
k = 20

x0 = zeros(Float64,n)

x = copy(x0)
r = b .- A*x

Y = randn(n,k)

println("q = 0")
println("  norm(r) = $(norm(r))")

r_nrm = zeros(q)

for s=1:q
    global x, Y, r

    Q, R = qr(hcat(Y,r))

    B = Matrix(Q)

    Y = B[:,1:k]

    AB = A*B
    c = (B'*AB)\(B'*r)

    x = x .+ B*c
    r = r .- AB*c

    Y = Y .- h.*AB[:,1:k]

    r_nrm[s] = norm(r)

    println("q = $s")
    println("  norm(b-Ax) = $(r_nrm[s])")
end

plot([1:q], log.(r_nrm))
