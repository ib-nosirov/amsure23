using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)

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

n = 1000
λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,n) + diagm(λ)
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
h = 0.001
k = 10

x0 = zeros(Float64,n)

x = copy(x0)

Y = randn(n,k)

println("q = 0")
println("norm(r) = $(norm(b-A*x))")

for s=1:q
    global x, Y

    r = b - A*x
    x = x + h.*r

    Y = Y - h.*(A*Y)

    Q, R = qr(hcat(x,Y))
    B = Matrix(Q)

    AB = A*B
    c = (B'*AB)\(B'*b)

    rc = b - AB*c

    println("q = $s")
    println("  norm(b-Ax) = $(norm(r))")
    println("  norm(b-ABc) = $(norm(rc))")
    println("  norm(b-ABc)/norm(b-Ax) = $(norm(rc)/norm(r))")
end