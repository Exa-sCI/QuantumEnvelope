#!/bin/python
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time

''' Block Davidson method for finding the lowest eigenvalues of a Hamiltonian
    H: Hamiltonian
    n: dimension of the Hamiltonian
    eps: covergeance tolerance
    max_iter: maximum number of iterations
    n_guess: number of initial guess vectors
    n_eig: number of eigenvalues to find
'''

def davidson(H, n, eps, max_iter, n_guess, n_eig):
    t = np.eye(n,n_guess) # set of n_guess unit vectors as guess
    V = np.zeros((n,n)) # array of zeros to hold guess vec
    I = np.eye(n) # identity matrix same dimension as A
    for m in range(n_guess,max_iter,n_guess):
        if m <= n_guess:
            for j in range(0,n_guess):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
            theta_old = 1 
        elif m > n_guess:
            theta_old = theta[:n_eig]
        V[:,:m],R = np.linalg.qr(V[:,:m])
        T = np.dot(V[:,:m].T,np.dot(H,V[:,:m]))
        THETA,S = np.linalg.eig(T)
        idx = THETA.argsort()
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(0,n_guess):
            w = np.dot((H - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
            q = w/(theta[j]-H[j,j])
            V[:,(m+j)] = q
        norm = np.linalg.norm(theta[:n_eig] - theta_old)
        if norm < eps:
            return theta

n = 1200				
n_eig = 4
A = np.zeros((n,n))  
for i in range(0,n):  
    A[i,i] = i + 1  
A = A + 0.0001*np.random.randn(n,n)  
A = (A.T + A)/2

start_davidson = time.time()

theta = davidson(A, n, 1e-8, n//2, 8, n_eig)

end_davidson = time.time()

# Print results.

print("davidson = ", theta[:n_eig],";",
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:n_eig],";",
     end_numpy - start_numpy, "seconds") 
