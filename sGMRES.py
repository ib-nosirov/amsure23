import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def arnoldi_iteration(A,b,k):
    n = len(b)
    Q = np.zeros((n,k+1))
    H = np.zeros((n,k+1))

    Q[:,0] = b / norm(b)

    for j in range(k):
        v = A @ Q[:,j]
        for i in range(j+1):
            H[i,j] = Q[:,i] @ v
            v -= H[i,j]*Q[:,i]
        H[j+1,j] = norm(v)
        if H[j+1,j] == 0:
            break
        Q[:,j+1] = v/H[j+1,j]

    #H = H[:k,:]
    Q = Q[:,:k]

    return H

# Display the matrix using imshow
#plt.imshow(np.abs(H), cmap='hot', interpolation='nearest')
#plt.colorbar()

# Add labels and title
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#plt.title('Matrix Visualization')

# Show the plot
plt.show()

def gmres(A,f,x,d,k,tol):
    # ARGS:
    # A: initial matrix.
    # f: right-hand side.
    # x: initial guess.
    # d: basis dimension.
    # k: truncation threshol for Arnoldi.
    # tol: tolerance on condition number.

    # draw a sketching matrix
    n = len(A)
    S = np.random.randn(2*(d+1),n)
    r = f - A@x
    g = S@r
    n = len(x)
    # we will run the full GMRES algorithm
    # issue: Arnoldi returns a square and it should not.
    H = arnoldi_iteration(A,x,k)
    C = S @ H
    Q,R = np.linalg.qr(C)
#    if np.linalg.cond(R) > tol:
        # form new residual and restart
    # Solve the linear system Ax = x
    y = Q.T @ g
    print(C.shape)
    print(y.shape)
    x = np.linalg.solve(R,y)
    x += H @ y
    res = norm((np.eye(n) - Q*Q.T)@g)
    return x, res

# Example usage
A = np.random.randn(10,10)
x = np.random.randn(10)
f = A @ x
d = 3
k = 4
tol = 10**5

print(gmres(A,f,x,d,k,tol))

#def sgmres()
