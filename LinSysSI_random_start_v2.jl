using LinearAlgebra
using SparseArrays
using Random
using Plots

Random.seed!(1)
gr(size = (1750, 1565)) # use gr as the plot backend

# function user_sparse_matvec(x::Array{Float64})
#
#     n = length(x)
#
#     ee = ones(n-1)
#     dd = range(1,length=n)
#     dd = dd./n
#
#     A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)
#
#     return A*x
#
# end


# n = 1000
# ee = ones(n-1)
# dd = range(1,length=n)
# dd = dd./n
# A = spdiagm(-1=>ee,0=>dd.*3,1=>ee.*2)
# b = randn(n)
# b = b./norm(b)

# n = 10000
# λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
# b = rand(n)

n = 1000 # 10000
λ = @. 10 + (1:n)
# λ = zeros(n)
# λ = n*ones(n)
# λ[1] = 10
# A = triu(rand(n,n),1) + diagm(λ)
# A = diagm(λ)
A = randn(n,n) + diagm(λ)
# A = (A+A') ./2
# A = A'*A
b = randn(n)

# N = 32
# n = N^3
# A = spdiagm(-1=>fill(-1.0, N - 1), 0=>fill(3.0, N), 1=>fill(-2.0, N - 1))
# Id = copy(sparse(1.0*I, N, N));
# A = kron(A, Id) + kron(Id, A)
# A = kron(A, Id) + kron(Id, A)
# x = ones(n)
# # x = zeros(n)
# # x[1] = 1
# b = A * x


xtrue = A\b

q = 500
h = 0.0001
k = 20

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

# ef = eigen(Symmetric(A), n:n)
# @show (1.0 .- h.*(ef.values))
#
# ef = eigen(Symmetric(A), 1:k+1)   #k smallest eigenvalues/vectors
# @show (1.0 .- h.*(ef.values))

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

    # AB = A1*B
    # c = AB\b

    rc = b - AB*c

    r_nrm[s] = norm(r)
    rc_nrm[s] = norm(rc)

    println("q = $s")
    println("  norm(b-Ax) = $(r_nrm[s])")
    println("  norm(b-ABc) = $(rc_nrm[s])")
    println("  norm(b-ABc)/norm(b-Ax) = $(rc_nrm[s]/r_nrm[s])")
end

plot([1:q], log.(r_nrm),
title="rc_norm",
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
