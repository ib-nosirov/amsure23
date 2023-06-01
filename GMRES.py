# import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

# where does the reorthogonalization step come in?
def gmres(A,b,k):
    Q,H = arnoldi(A,b,k)
    e1 = np.zeros(k)
    e1[0] = norm(b)
    y,res = np.linalg.lstsq(H[:k,:k],e1)[:2] # what is [:2] for?
    return Q[:,:k]@y

def arnoldi(A,b,k):
    # A matrix
    # b starting vector
    # k number of iterations
    # starts with n=1
    n = len(b)
    H = np.zeros((k+1,k))
    Q = np.zeros((n,k+1)) # why k+1?
    Q[:,0] = b/norm(b)
    for i in range(k):
        v = A @ Q[:,i]
        for j in range(i+1):
            q = Q[:,j]
            H[j,i] = q@v# take a bunch of inner products
            v -= H[j,i]*q # get a residual

        H[i+1,i] = norm(v)
        Q[:,i+1] = v/H[i+1,i]
    return Q,H

    # Example usage
n = 10
A = np.random.randn(n,n)
b = np.random.randn(n)
k = n

x = gmres(A,b,k)

true_x = np.linalg.solve(A,b)

print("Solution x:")
print(x)
print("True x:")
print(true_x)
print("Error:")
print(x-true_x)