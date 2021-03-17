#!/bin/python
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time
from mpi4py import MPI
import sys

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
    for m in range(n_guess, max_iter, n_guess):
        if m <= n_guess:
            for j in range(0, n_guess):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j]) # initialization of V
            theta_old = 1 
        elif m > n_guess:
            theta_old = theta[:n_eig]
        V[:,:m],R = np.linalg.qr(V[:,:m]) # QR decomposition of V
        T = np.dot(V[:,:m].T,np.dot(H,V[:,:m])) # V^T H V
        THETA,S = np.linalg.eig(T)  # diagonalizing V^T H V
        idx = THETA.argsort() # sorting the eigenvalues
        theta = THETA[idx] # getting an array of sorted eigenvalues
        s = S[:,idx] # getting an array of sorted eigenvectors
        for j in range(0,n_guess):
            # (H - theta_j * I) *(V * s_j)
            r = np.dot((H - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
            q = r/(theta[j]-H[j,j])
            V[:,(m+j)] = q
        norm = np.linalg.norm(theta[:n_eig] - theta_old)
        if norm < eps:
            return theta

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#sys.stdout.write("Hello! I'm processor %d of %d. \n" % (rank, size))

n = 24#1200
n_eig = 4
n_guess = n_eig * 2
max_iter = n//2

A = np.zeros((n,n))  
for i in range(0,n):  
    A[i,i] = i + 1  
A = A + 0.0001*np.random.randn(n,n)  
A = (A.T + A)/2

t = np.eye(n,n_guess)
V = np.zeros((n,n))
I = np.eye(n)
for j in range(0, n_guess):
    V[:,j] = t[:,j]/np.linalg.norm(t[:,j])

n_p = n // size
offset = rank * n_p
A_p = np.zeros((n_p,n))
V_p = np.zeros((n_p, n))
for i in range(n_p):
    A_p[i,:] = A[i + offset,:]
    V_p[i,:] = V[i + offset,:]
    W_p = []
    Y = []

L = []
dim_S = []

for j in range(n_eig):
    for k in range(max_iter):
        # Broadcast vk
        if rank == 0:
            comm.Bcast(V, root=0)
        # Receive vk
        # compute w_pk = A_p vk and update W_pk = [W_p(k-1),w_pk]
        w_p = np.dot(A_p, V[:,k])
        W_p = [W_p, w_p]
        s_p = np.dot(V_p.T, w_p)
        # send to host

        # if rank == 0:
            # s_sum = sum(s_p for all processors)
            # eigen decomposition L_k and Y_k
        LAMBDA, Y = np.linalg.eig(A) # should be S
            # dim_S += 1
            # check and apply restart when necessary
        idx = LAMBDA.argsort()
        l = LAMBDA[idx]
        y = Y[:,idx]
            # broadcast the l smallest eigenpair l_k, y_k from L_k and Y_k, where l= min(dim_S, j)
        
        # receive y_k and l_k
        # compute r_pk = W_pk * y_k - l * V_pk * y_k
        #r_p = np.dot(W_p, y)  - l * np.dot(V_p, y)
        temp = np.dot(W_p, y)

        if rank == 0:
            # broadcast r by appending all r_p (residual vector)
            # r = np.dot((H - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
            # q = r/(theta[j]-H[j,j])
            # V[:,(m+j)] = q (see preconditioning)
            # norm = np.linalg.norm(theta[:n_eig] - theta_old)
            norm = 1
            # norm = np.linalg.norm(r)
            if norm < eps:
                break
            # t = M * r (preconditioning)
            # send slice t_p to the correspoding node
            # v = modified gram schmidt to get it from V and t

sendBuf = None
if rank == 0:
    A = np.zeros((n,n))  
    for i in range(0,n):  
        A[i,i] = i + 1  
    A = A + 0.0001*np.random.randn(n,n)  
    A = (A.T + A)/2
    sendBuf = np.empty([size, n])
    sendBuf.T[:,:] = range(size)
else:
    A = np.zeros((n,n))
recvBuf= np.empty(n)
comm.Scatter(sendBuf, recvBuf, root=0)
assert np.allclose(recvBuf, rank)

#comm.Bcast(A, root=0)
#start_davidson = time.time()

#theta = davidson(A, n, 1e-8, n//2, 8, n_eig)

#end_davidson = time.time()

# Print results.
if rank == 0:
#    print("davidson = ", theta[:n_eig],";")
       # , end_davidson - start_davidson, "seconds")
    #start_numpy = time.time()
    E,Vec = np.linalg.eig(A)
    E = np.sort(E)
    #end_numpy = time.time()
    print("numpy = ", E[:n_eig],";")
       # , end_numpy - start_numpy, "seconds") 
