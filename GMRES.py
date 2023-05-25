import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def arnoldi_iteration(A,V,n):
    H = np.zeros((n+1,n+1))
    v = A @ V[:,n]
    for j in range(n):
        H[j,n] = V[:,j]@v
        v -= H[j,n]*V[:,j]
    H[n,n] = norm(v)
    if H[n,n] != 0:
        V[:,n+1] = v/H[n+1,n]

    # Display the matrix using imshow
    plt.imshow(np.abs(H), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

    return V,H

def gmres(A,b,k):
    n = A.shape[0]
    b_norm = norm(b)
    x = np.zeros(n)
    #residual = b - A@x
    #residual_norm = norm(residual)

    # Krylov subspace-at most k iterations
    V = np.zeros((n,k))
    # Hessenberg matrix
    # my personal addition
    for n in range(k):
        V,H = arnoldi_iteration(A,V,n)
        e1 = np.zeros(n+1)
        e1[0] = b_norm 
        print(H.shape)
        print(e1.shape)

        y, _, _, _ = np.linalg.lstsq(H,e1)

        # same as x_n = V_n @ y, but cheaper
        x += V[:,n] @ y
    return x


# Example usage
A = np.random.randn(10, 10)
b = np.random.randn(10)
k = 9

x = gmres(A,b,k)

true_x = np.linalg.solve(A,b)

print("Solution x:")
print(x)
print("True x:")
print(true_x)
print("Error:")
print(x-true_x)
