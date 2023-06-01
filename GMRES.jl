using LinearAlgebra
using Random

# where does the reorthogonalization step come in?
function gmres(A, b, k)
    Q, H = arnoldi(A, b, k)
    e1 = zeros(k)
    e1[1] = norm(b)
    y = H[1:k, 1:k] \ e1
    return Q[:, 1:k] * y
end

function arnoldi(A, b, k)
    # A matrix
    # b starting vector
    # k number of iterations
    # starts with n=1
    n = length(b)
    H = zeros(k+1,k)
    Q = zeros(n,k+1)
    Q[:,1] = b / norm(b)
    for i in 1:k
        v = A * Q[:,i]
        for j in 1:i
            q = Q[:, j]
            H[j, i] = dot(q,v) # take a bunch of inner products
            v -= H[j,i]*q # get a residual
            # the above two lines are a reorthogonalization step... very
            # expensive.
        end
        H[i+1, i] = norm(v)
        Q[:, i+1] = v / H[i+1, i]
    end
    return Q, H
end

# Example usage
n = 10
Random.seed!(123)  # To get consistent results
A = randn(n, n)
b = randn(n)
k = n

x = gmres(A, b, k)

x_true = A \ b

println("Solution x:")
println(x)
println("True x:")
println(x_true)
println("Error:")
println((x - x_true)/norm(x_true))
