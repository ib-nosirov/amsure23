function Richardson(A,b,h,q,k)
    n = size(A,1)
    
    x = zeros(n)
    
    r_nrm = zeros(q)
    rc_nrm = zeros(q)
    
    
    for s=1:q
    
        r = b - A*x
        x = x + h.*r
    
        rc = b - A*x
    
        rc_nrm[s] = norm(rc)
    end
    
    return rc_nrm
    end

function LinSysSI(A,b,h,q,k)
    n = size(A,1)
    
    x = zeros(n)
    Y = randn(n,k)
    
    rc_nrm = zeros(q)
    
    AB = zeros(n,k+1)
    AB[:,1:k] = A*Y
    
    for s=1:q
        r = b - A*x
        x = x + h.*r
    
        Y = Y - h.*AB[:,1:k]
        Q,R = qr(hcat(Y,x))
        B = Matrix(Q)
    
        Y = B[:,1:k]
    
        AB = A*B
        c = (B'*AB)\(B'*b)
    
        rc = b - AB*c
    
        rc_nrm[s] = norm(rc)
    end
    
    return rc_nrm
    end

function RandomStarting(A,b,h,q,k)
    n = size(A,1)
    
    #Y = hcat(Pi,zeros(n))
    Y = randn(n,k+1)
    Y[:,end] = zeros(n)
    
    rc_nrm = zeros(q)
    
    AY = A*Y
    for s=1:q
        Y[:,1:k] -= h.*AY[:,1:k]
    
        Y[:,end] += h.*(b-AY[:,end])
        Q,R = qr(Y)
        Q = Matrix(Q)
    
        Y[:,1:k] = Q[:,1:k]
    
        AY = A*Y
        c = (Y'*AY)\(Y'*b)
    
        rc = b - AY*c
    
        rc_nrm[s] = norm(rc)
    end
    
    return rc_nrm
    end

# main
#Random.seed!(1)

#n = 1000
#λ = @. 10 + (1:n)
#A = randn(n,n) + diagm(λ)
#b = randn(n)
#h = 0.001
#q = 200
#k = 20
#Pi = randn(n,k)

#si_error = LinSysSI(A,b,h,q,k);
#rs_error = RandomStarting(A,b,h,q,k);
#ri_error = Richardson(A,b,h,q,k);

## separate code block for plotting.
#gr(size=(400,300))

#p = plot()

#plot!(1:size(si_error,1),si_error,
    #title="Accuracy vs q",
    #yaxis=:log10,
    #xlab="q",
    #ylab="Accuracy",
    #label="LinSysSI")

#plot!(1:size(rs_error,1),rs_error,
    #title="Accuracy vs q",
    #yaxis=:log10,
    #xlab="q",
    #ylab="Accuracy",
    #label="RandomStarting")

#plot!(1:size(ri_error,1),ri_error,
    #title="Accuracy vs q",
    #yaxis=:log10,
    #xlab="q",
    #ylab="Accuracy",
    #label="Richardson")

#display(p)
