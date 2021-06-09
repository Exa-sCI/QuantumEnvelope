#!/usr/bin/env python3
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


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    print("init: I'm processor %d of %d. \n" % (rank, world_size))

    # davidson initialization, since we're going for distributed data, that
    # stuff is actually just the buffers required on one MPI rank.

    # global size of the system
    n = 2**8
    local_size = n // world_size
    assert(local_size * world_size == n)

    # number of total eigenvalues we want
    n_eig = 4

    # number of seed eigenvectors
    n_guess = n_eig * 2

    # max number of iterations we allow (divergence cutoff)
    max_iter = n//2

    # cutoff for error
    epsilon = 1e-8

    print(f"init: {n}, {local_size}, {n_eig}, {n_guess}, {max_iter}, {epsilon}")

    # lets build the initial decomposition, per the paper:
    # the global matrix is evenly split among rows.
    # A is only the input, so we can split it at once, V_k and W_k are
    # iteratively growing in columns during execution so it's easier to keep
    # them as lists of columns.
    A_i = np.zeros((local_size, n))
    V_ik = []  # list of np vects (local_size)
    W_ik = []  # list of np vects (local_size)
    r_ik = np.zeros((local_size))

    # lets build a garbage initial matrix
    A_i = 0.0001*np.random.randn(local_size, n)

    x_1 = np.random.randn(local_size)
    V_ik.append(np.linalg.norm(x_1))

    S_k = []
    dim_S = 0
    it = 0  # number of iterations
    err = 1e-8

    for j in range(n_eig):
        for k in range(1, max_iter):

            # gather the entire V_k on each rank
            V_k = np.zeros(n)
            comm.Allgather([V_ik[k-1], MPI.DOUBLE], [V_k, MPI.DOUBLE])

            # compute the new column of W_ik
            new_w = np.dot(A_i, V_k)
            W_ik.append(new_w)

            # paper uses k-2 here?
            new_s = np.dot(V_ik[k-1].T, new_w)

            # send to the host and compute sum
            s_k = np.copy(new_s)
            comm.Reduce(new_s, s_k)

            # algorithm2
            if rank == 0:
                L_k = []
                Q_k = []
                ns_k = np.dot(


            print("checkpoint:")


def relicats():
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
            #if norm < eps:
            #    break
            # t = M * r (preconditioning)
            # send slice t_p to the correspoding node
            # v = modified gram schmidt to get it from V and t

#comm.Bcast(A, root=0)
#start_davidson = time.time()

#theta = davidson(A, n, 1e-8, n//2, 8, n_eig)

#end_davidson = time.time()

# Print results.
#if rank == 0:
#    print("davidson = ", theta[:n_eig],";")
       # , end_davidson - start_davidson, "seconds")
    #start_numpy = time.time()
#    E,Vec = np.linalg.eig(A)
#    E = np.sort(E)
    #end_numpy = time.time()
#    print("numpy = ", E[:n_eig],";")
       # , end_numpy - start_numpy, "seconds")
