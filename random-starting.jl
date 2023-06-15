using LinearAlgebra
using Random
using Plots

function RandomStarting(A,b,ε,q,k,x_true)
    n = size(A, 1)
    # Ensure that ε is small enough so that I - ε*A has
    # spectral radius < 1
    eigenvalues = eigvals(I - ε*A)
    spectral_radius = maximum(abs.(eigenvalues))
    if spectral_radius >= 1
        throw(ArgumentError("Choose a smaller ε such that spectral radius of I -
        εA is less than 1"))
    end
    # Initialize sketch matrix
    Pi = randn(n,k-1)
    # Calculate Y
    Y = zeros(n,k)
    # start at Pi
    Y[:,2:end] = Pi
    c_arr = zeros(q)
    x_approx = zeros(k)
    for j in 0:q-1
        Y[:,2:end] = (I - ε*A)*Y[:,2:end]
        Y[:,1] += ε*b - ε*A*Y[:,1]
        Q,R = qr(Y)
        Q = Matrix(Q) # skinny qr
        # Solve the k x k linear system
        AQ = A*Q
        c = (Q'*AQ)\(Q'*b)
        # Get the approximate solution
        x_approx = Q*c
        c_arr[j+1] = norm(x_approx-x_true)/norm(x_true)
    end
    return c_arr
end

function RandomStartingAccuracy(A,b,ε,q,k,x_true)
    x_approx = RandomStarting(A,b,ε,q,k,x_true)
    #@show(maximum(x_true))
    #@show(minimum(x_true))
    return norm(x_approx-x_true)/norm(x_true)
end

# main
n = 1000
λ = @. 10 + (1:n)
# A = triu(rand(n,n),1) + diagm(λ)
A = randn(n,n) + diagm(λ)
b = randn(n)
ε = 0.001
q = 500
k = 10 # error just drops when k = n
x_true = A\b
c_arr = RandomStarting(A,b,ε,q,k,x_true)
plot(1:length(c_arr),c_arr,title="Accuracy vs q", xlab="q", ylab="Accuracy",
legend=false,linewidth=2,titlefontsize=30, guidefontsize=30,
tickfontsize=30,size=(1200,1800))
#n = 100
#A = randn(n, n)
#A = A*A' # otherwise the matrix is too ill-conditioned.
#b = randn(n)
#ε = 0.001
#max_q = 1000
#max_k = 1000
#
#q_values = 1:100:max_q
#k_values = 1:100:max_k
#
#q_accuracies = [RandomStartingAccuracy(A,b,ε,q,1000,x_true) for q in q_values]
#k_accuracies = [RandomStartingAccuracy(A,b,ε,1000,k) for k in k_values]

#p1 = plot(q_values, q_accuracies, title="Accuracy vs q", xlab="q",
#ylab="Accuracy", legend=false,linewidth=2,titlefontsize=30, guidefontsize=30,
#tickfontsize=30)
#p2 = plot(k_values, k_accuracies, title="Accuracy vs k", xlab="k",
#ylab="Accuracy", legend=false,linewidth=2,titlefontsize=30, guidefontsize=30,
#tickfontsize=30)
#
#plot(p1, p2, layout = (2, 1),size = (2000, 1800))