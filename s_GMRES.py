import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

def arnoldi_iteration(A,V,H,i):
    v = A @ V[:,i]
    for j in range(i):
        H[i,j] = np.dot(V[:,i],v)
        v -= H[i,j]*V[:,i]
    H[j+1,j] = norm(v)
    if H[j+1,j] != 0:
        V[:,j+1] = v/H[j+1,j]
    # M[:,i] = A@v -> this is an extra matrix vector multiply
    # C = S @ M
    # Q,R = QR(C)
    # solve least-squares problem inv(T)(U*g) -> how does this work?
    # x += M*yhat

    return V,H

def gmres(A, b, max_iterations=None, restart=None, tol=1e-6):
    n = A.shape[0]
    if max_iterations is None:
        max_iterations = n
    if restart is None:
        restart = n

    x = np.zeros(n)
    residual = b - A@x
    residual_norm = norm(residual)

    V = np.zeros((n,restart+1))
    H = np.zeros((restart+1,restart))

    converged = False
    iteration = 0

    while not converged and iteration < max_iterations:
        V[:, 0] = residual / residual_norm

        Q, H = arnoldi_iteration(A,V,H,restart)

        e1 = np.zeros(restart+1)
        e1[0] = residual_norm

        y, _, _, _ = np.linalg.lstsq(H,e1)

        x += Q[:,:-1] @ y
        residual = b - A@x
        residual_norm = norm(residual)

        if residual_norm < tol:
            converged = True

        iteration += 1

    return x


# Example usage
A = np.random.randn(10, 10)
b = np.random.randn(10)

x = gmres(A,b)

print("Solution x:")
print(x)
