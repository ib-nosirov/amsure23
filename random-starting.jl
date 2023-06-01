using LinearAlgebra
using Random

function RandomStarting(A,b,ε,q,k)
    n = size(A, 1)

    # Ensure that \varepsilon is small enough so that I - \varepsilon * A has spectral radius < 1
    eigenvalues = eigvals(I - ε*A)
    spectral_radius = maximum(abs.(eigenvalues))
    if spectral_radius >= 1
        throw(ArgumentError("Choose a smaller ε such that spectral radius of I - εA is less than 1"))
    end

    # Initialize sketch matrix
    Pi = randn(n,k-1)

    # Calculate Y
    Y = zeros(n,k)
    Y[:,1] =  ε*sum([(I - ε*A)^j*b for j=0:q-1])
    Y[:,2:end] = (I - ε*A)^q*Pi

    # Perform QR decomposition
    Q, R = qr(Y)

    # Solve the k x k linear system
    AQ = A*Q
    c = (Q'*AQ)\(Q'*b)

    # Get the approximate solution
    x_approx = Q*c

    return x_approx
end

n = 100
A = randn(n,n)
A = A*A'
b = randn(n)
ε = 0.001
q = 5
k = 10
x_approx = RandomStarting(A,b,ε,q,k)
x_true = A\b
norm(x_approx-x_true)/(norm(x_true))