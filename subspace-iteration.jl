using LinearAlgebra
using Random
using Plots

function SubspaceIteration(A::Array{Float64,2},k::Int,maxIter=100,tol=1e-6)
# Check if `A` is square
if size(A,1) != size(A,2)
    throw(ArgumentError("Input matrix `A` must be square"))
end
# Check if `k` is less than the number of columns of `A`
if k > size(A,2)
    throw(ArgumentError("`k` must be less than or equal to the number of columns
    of `A`"))
end

n = size(A,1)
# Initialize a random matrix Q of size n x k
Q = rand(n,k)
R = zeros(n,k)
for iter = 1:maxIter
    Q_old = Q
    # Compute the matrix product AQ
    Z = A*Q
    # Perform QR factorization of Z
    Q,R = qr(Z)
    # Check for convergence
    if norm(Q-Q_old,Inf) < tol
        break
    end
end
# Compute the eigenvalues using the Rayleigh quotient
eigenvalues = diag(R)

return eigenvalues,Q
end

n=10
A = randn(n,n)
A = A*A' # real eigenvalues
k = n
eigenvalues,Q = SubspaceIteration(A,k)
sort(eigenvalues) - sort(eigen(A).values)