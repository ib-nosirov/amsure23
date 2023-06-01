using LinearAlgebra
using Random
using Plots

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

function RandomStartingAccuracy(A,b,eps,q,k)
    x_approx = RandomStarting(A,b,eps,q,k)
    x_true = A\b
    return norm(x_true-x_approx)/norm(x_true)
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
println(norm(x_approx-x_true)/(norm(x_true)))

n = 100
A = randn(n, n)
A = A*A' # otherwise the matrix is too ill-conditioned.
b = randn(n)
ε = 0.001
max_q = 50
max_k = 50

q_values = 1:max_q
k_values = 1:max_k

q_accuracies = [RandomStartingAccuracy(A,b,ε,q,10) for q in q_values]
k_accuracies = [RandomStartingAccuracy(A,b,ε,10,k) for k in k_values]

p1 = plot(q_values, q_accuracies, title="Accuracy vs q", xlab="q",
ylab="Accuracy", legend=false,linewidth=2,titlefontsize=30, guidefontsize=30,
tickfontsize=30)
p2 = plot(k_values, k_accuracies, title="Accuracy vs k", xlab="k",
ylab="Accuracy", legend=false,linewidth=2,titlefontsize=30, guidefontsize=30,
tickfontsize=30)


plot(p1, p2, layout = (2, 1),size = (2000, 1800))

# how fast is convergence as you increase each one?