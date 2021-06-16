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


def parallel_arrowhead_decomposition(Y_p, s_k, L_p):
    # prepare arrowhead matrix
    s_nk = np.dot(Y_p.T, s_k[:-1])
    s_nk = np.append(s_nk, s_k[-1])
    L_p = np.append(L_p, 0)
    S_nk = np.diag(L_p)
    S_nk[-1, :] = s_nk
    S_nk[:, -1] = s_nk

    # needs to be switched with a better implementation
    L_k, Q_k = np.linalg.eig(S_nk)

    # recover Y_k from Q_k
    p = Y_p.shape[0]
    Y_ext = np.zeros((p+1, p+1))
    Y_ext[:-1, :-1] = Y_p
    Y_ext[-1, -1] = 1
    Y_k = np.dot(Y_ext, Q_k)
    return L_k, Y_k


def parallel_restart(m, q, dim_S, L_k, Y_k, V_ik, W_ik):
    if m + q <= dim_S:
        L_k = L_k[:m]
        # TODO: missing orthonormalization of Y_k[:,:m] here
        Y_k = Y_k[:, :m]
        V_ik = np.dot(V_ik, Y_k)
        W_ik = np.dot(W_ik, Y_k)
        Y_k = Y_k[:m, :]
        dim_S = m
    return dim_S, L_k, Y_k, V_ik, W_ik


def preconditioning(A_i, l_k, r_ik):
    M_k = np.diag(np.reciprocal(np.diag(A_i) - l_k))
    return np.dot(M_k, r_ik)


def mgs(comm, n, V_ik, t_ik):
    for v in range(1, V_ik.shape[1]):
        c_j = np.copy(np.inner(V_ik[:, v], t_ik))
        comm.Allreduce([c_j, MPI.DOUBLE], [c_j, MPI.DOUBLE])
        t_ik = t_ik - c_j*V_ik[:, v]
    t_k = np.zeros(n)
    comm.Allgather([t_ik, MPI.DOUBLE], [t_k, MPI.DOUBLE])
    return t_ik/np.linalg.norm(t_k)


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

    m, q = 10, 15

    print(f"init: {n}, {local_size}, {n_eig}, {n_guess}, {max_iter}, {epsilon}")

    # lets build the initial decomposition, per the paper:
    # the global matrix is evenly split among rows.
    # A is only the input, so we can split it at once, V_k and W_k are
    # iteratively growing in columns during execution so it's easier to keep
    # them as lists of columns.
    A_i = np.zeros((local_size, n))
    V_ik = np.zeros((local_size, 0))
    W_ik = np.zeros((local_size, 0))
    r_ik = np.zeros((local_size))

    # each node needs to keep the previous Y_k
    Y_k = np.eye(1)
    L_k = []

    # lets build a garbage initial matrix
    A_i = 0.0001*np.random.randn(local_size, n)

    x_1 = np.random.randn(local_size)
    V_ik = np.c_[V_ik, x_1/np.linalg.norm(x_1)]

    dim_S = 0
    err = 1e-8

    for j in range(n_eig):
        for k in range(1, max_iter):
            print(f"eig: {j}, iter: {k}")

            # gather the entire V_k on each rank
            V_k = np.zeros(n)
            local_vik = np.array(V_ik[:, -1])
            comm.Allgather([local_vik, MPI.DOUBLE], [V_k, MPI.DOUBLE])

            # compute the new column of W_ik
            new_w = np.dot(A_i, V_k)
            W_ik = np.c_[W_ik, new_w]

            # paper uses k-2 here?
            new_s = np.dot(V_ik.T, new_w)

            # send to the host and compute sum
            s_k = np.copy(new_s)
            comm.Reduce(new_s, s_k)

            print("arrow", f"dim_S: {dim_S}")
            if k == 1:
                L_k, Y_k = np.linalg.eig(np.diag(s_k))
            else:
                L_k, Y_k = parallel_arrowhead_decomposition(Y_k, s_k, L_k)
            dim_S += 1

            print("restart", f"dim_S: {dim_S}")
            dim_S, L_k, Y_k, V_ik, W_ik = parallel_restart(m, q, dim_S, L_k,
                                                           Y_k, V_ik, W_ik)

            print("residual", f"dim_S: {dim_S}", f"L_k: {L_k.shape}",
                  f"Y_k: {Y_k.shape}")
            lmin = min(dim_S, j)
            # L_k and Y_k are sync'd (the sort of one is the same on the other)
            indices = L_k.argsort()
            l_k = L_k[indices[lmin]]
            y_k = Y_k[:, indices[lmin]]

            print(f"{V_ik.shape}, {W_ik.shape}, {y_k.shape}")
            r_ik = np.dot(W_ik, y_k) - l_k*np.dot(V_ik, y_k)
            r_k = np.zeros(n)
            comm.Allgather([r_ik, MPI.DOUBLE], [r_k, MPI.DOUBLE])

            res = np.linalg.norm(r_k)
            if res < epsilon:
                break

            print("preconditioning", f"||r_k||: {res}")
            t_ik = preconditioning(A_i, l_k, r_ik)

            print("mgs")
            V_ik = np.c_[V_ik, mgs(comm, n, V_ik, t_ik)]

# fuse A across all nodes
A = np.zeros((n, n))
comm.Allgather([A_i, MPI.DOUBLE], [A, MPI.DOUBLE])
# simple algo

if rank == 0:
    ref_l, ref_y = np.linalg.eig(A)
    ref_l = np.sort(ref_l)

    print(L_k, ref_l: len(L_k))
