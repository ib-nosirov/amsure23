using LinearAlgebra
using Random

# implemented from https://arxiv.org/pdf/2111.00113.pdf
function sGMRES(A,f,x,d,k) # will need to add tolerance
    n = length(f)
    S = randn(2*(d+1),n)
    r = f - A*x
    g = S*r
    B = zeros(n,d)
    B[:,1] = r/norm(r)
    M = zeros(n,d)
    M[:,1] = A*B[:,1]
    M,B = arnoldi(A,M,B,d,k)
    C = S*M
    U,T = qr(C)
    yhat = T \ (U*g)
    xhat = x + M*yhat
end

function arnoldi(A,M,B,d,k)
    for i in 2:d
        w = bSum(i,k,M,B)
        B[:,i] = w/norm(w)
        M[:,i] = A*B[:,i]
    end
end

function bSum(i,k,M,B)
    n = maximum(size(B))
    W = zeros(n,n)
    for j in 1:k
        if i-j > 0
            W -= B[:,i-j]*B[:,i-j]'
        else
            continue
        end
    end
    w = W*M[:,i-1]
    return w
end

# Example usage
n = 10
Random.seed!(123)  # To get consistent results
A = randn(n,n)
f = randn(n)
d = Int(floor(2*n/3))
x = randn(n)
k = n-2

x = sGMRES(A,f,x,d,k)

x_true = A \ f

println("Solution x:")
println(x)
println("True x:")
println(x_true)
println("Error:")
println((x - x_true)/norm(x_true))