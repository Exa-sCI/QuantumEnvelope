#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
import itertools


class DavidsonManager:
    def __init__(self, comm, problem_size):
        self.comm = comm
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        # put ceil(n/world) on n%world nodes, and floor(n/world) on the others
        self.problem_size = problem_size
        local_min = problem_size // self.world_size
        local_max = local_min + 1
        remainder = problem_size % self.world_size
        self.distribution = [local_max] * remainder + [local_min] * (self.world_size - remainder) 
        self.local_size = self.distribution[self.rank]
        print(f"dist: {self.distribution}, local: {self.local_size}")
        # compute offsets (start of the local section) for all nodes
        self.offsets = [0] + list(itertools.accumulate(self.distribution))
        del self.offsets[-1]

    def sequential(self, H, n, eps, max_iter, n_guess, n_eig):
        """Block Davidson method for finding the lowest eigenvalues of a
        Hamiltonian, sequential version

        :param H: Hamiltonian
        :param n: dimension of the Hamiltonian
        :param eps: convergence tolerance
        :param max_iter: maximum number of iterations
        :param n_guess: number of initial guess vectors
        :param n_eig: number of eigenvalues to find
        """
        t = np.eye(n, n_guess)  # set of n_guess unit vectors as guess
        V = np.zeros((n, n))  # array of zeros to hold guess vec
        I = np.eye(n)  # identity matrix same dimension as A
        for m in range(n_guess, max_iter, n_guess):
            if m <= n_guess:
                for j in range(0, n_guess):
                    V[:, j] = t[:, j] / np.linalg.norm(
                        t[:, j]
                    )  # initialization of V
                theta_old = 1
            elif m > n_guess:
                theta_old = theta[:n_eig]
            V[:, :m], R = np.linalg.qr(V[:, :m])  # QR decomposition of V
            T = np.dot(V[:, :m].T, np.dot(H, V[:, :m]))  # V^T H V
            THETA, S = np.linalg.eig(T)  # diagonalizing V^T H V
            idx = THETA.argsort()  # sorting the eigenvalues
            theta = THETA[idx]  # getting an array of sorted eigenvalues
            s = S[:, idx]  # getting an array of sorted eigenvectors
            for j in range(0, n_guess):
                # (H - theta_j * I) *(V * s_j)
                r = np.dot((H - theta[j] * I), np.dot(V[:, :m], s[:, j]))
                q = r / (theta[j] - H[j, j])
                V[:, (m + j)] = q
            norm = np.linalg.norm(theta[:n_eig] - theta_old)
            if norm < eps:
                return theta

    def parallel_arrowhead_decomposition(self, Y_p, s_k, L_p):
        """Builds an arrowhead matrix, diagonalizes it, and returns its
        eigenvalues and eigenvectors.

        The arrowhead is in the form:
          | L_p s_k  |
          | s_k s_kk |
        :param Y_p: a list of eigenvectors, as a numpy matrix
        :param s_k: a numpy vector
        :param L_p: a list of eigenvalues, as a numpy vector
        Y_p and L_p are of the same size.
        :return (L_k, Y_k) a list of eigenvalues as a numpy vector, and a list
        of eigenvectors, as a matrix. Consistent with np.linalg.eig.

        L_k.shape == L_p.shape + 1
        Y_k.shape == Y_p.shape + (1, 0)
        """
        # prepare arrowhead matrix
        s_nk = np.dot(Y_p.T, s_k[:-1])
        s_nk = np.append(s_nk, s_k[-1])
        L_p = np.append(L_p, 0)
        S_nk = np.diag(L_p)
        S_nk[-1, :] = s_nk
        S_nk[:, -1] = s_nk

        # TODO: needs to be switched with a better implementation
        L_k, Q_k = np.linalg.eig(S_nk)

        # recover Y_k from Q_k
        p = Y_p.shape[0]
        Y_ext = np.zeros((p + 1, p + 1))
        Y_ext[:-1, :-1] = Y_p
        Y_ext[-1, -1] = 1
        Y_k = np.dot(Y_ext, Q_k)
        return L_k, Y_k

    def parallel_restart(self, m, q, dim_S, L_k, Y_k, V_ik, W_ik):
        """Given two parameters `m` and `q`, resize (restart) the `Y_k` matrix
        and its associated data structures to `m` columns. Basically a memory
        footprint reduction pass.

        :param m: number of columns to keep
        :param q:
        :param dim_S: size of local working variables (Y_k and friends)
        :param L_k: list of eigenvalues, as a numpy vector
        :param Y_k: list of eigenvectors, as a numpy matrix
        :param V_ik, W_ik: local working variables, numpy vectors
        :return new values for dim_S, L_k, Y_k, V_ik, and W_ik
        """
        if m + q <= dim_S:
            L_k = L_k[:m]
            # TODO: missing orthonormalization of Y_k[:,:m] here
            Y_k = Y_k[:, :m]
            V_ik = np.dot(V_ik, Y_k)
            W_ik = np.dot(W_ik, Y_k)
            Y_k = Y_k[:m, :]
            dim_S = m
        return dim_S, L_k, Y_k, V_ik, W_ik

    def preconditioning(self, A_i, l_k, r_ik):
        """Precondition input matrix A_i.

        :param A_i: local section of input to davidson, as a numpy matrix (full
        columns).
        :param l_k: an eigenvalue, as a scalar
        :param r_ik: residual, a numpy vector
        :return well conditioned numpy matrix
        """
        M_k = np.diag(np.reciprocal(np.diag(A_i) - l_k))
        return np.dot(M_k, r_ik)

    def mgs(self, n, V_ik, t_ik):
        """Modified Graham-Schmidt, orthonormalization of V_ik.

        :param comm: MPI communicator
        :param n: size of entire input
        :param V_ik: local work variable, numpy matrix
        :param t_ik: local work variable, numpy vector
        :return normal V_ik
        """
        for v in range(1, V_ik.shape[1]):
            c_j = np.copy(np.inner(V_ik[:, v], t_ik))
            self.comm.Allreduce([c_j, MPI.DOUBLE], [c_j, MPI.DOUBLE])
            t_ik = t_ik - c_j * V_ik[:, v]
        t_k = np.zeros(n)
        self.comm.Allgatherv([t_ik, MPI.DOUBLE], [t_k, self.distribution, self.offsets,  MPI.DOUBLE])
        return t_ik / np.linalg.norm(t_k)

    def distributed(self, A_i, n, n_eig, eps, max_iter, m, q):
        """Distributed davidson implementation, of a matrix distributed
        column-wise across MPI ranks.

        :param A_i: a number of columns of an Hamiltonian.
        :param n: the size of the Hamiltonian
        :param n_eig: the number of eigenvalues to find
        :param eps: convergence tolerance
        :param max_iter: iteration number cutoff (per eigenvalue)
        :param m, q: memory footprint tuning (max size of local work variables
        :return a list of `n` eigenvalues, as a numpy vector
        """
        assert n == self.problem_size

        # lets build the initial decomposition, per the paper:
        # the global matrix is evenly split among rows.
        # A is only the input, already split, V_k and W_k are
        # iteratively growing in columns during execution so it's easier to
        # keep them as lists of columns.
        V_ik = np.zeros((self.local_size, 0))
        W_ik = np.zeros((self.local_size, 0))
        r_ik = np.zeros((self.local_size))

        x_1 = np.random.randn(self.local_size)
        V_ik = np.c_[V_ik, x_1 / np.linalg.norm(x_1)]

        dim_S = 0

        for j in range(n_eig):
            for k in range(1, max_iter):
                print(f"eig: {j}, iter: {k}")

                # gather the entire V_k on each rank
                V_k = np.zeros(n)
                local_vik = np.array(V_ik[:, -1])
                self.comm.Allgatherv([local_vik, MPI.DOUBLE], [V_k, self.distribution, self.offsets, MPI.DOUBLE])

                # compute the new column of W_ik
                new_w = np.dot(A_i, V_k)
                W_ik = np.c_[W_ik, new_w]

                # paper uses k-2 here?
                new_s = np.dot(V_ik.T, new_w)

                # send to the host and compute sum
                s_k = np.copy(new_s)
                self.comm.Reduce(new_s, s_k)

                print("arrow", f"dim_S: {dim_S}")
                if k == 1:
                    L_k, Y_k = np.linalg.eig(np.diag(s_k))
                else:
                    L_k, Y_k = self.parallel_arrowhead_decomposition(
                        Y_k, s_k, L_k
                    )
                dim_S += 1

                print("restart", f"dim_S: {dim_S}")
                dim_S, L_k, Y_k, V_ik, W_ik = self.parallel_restart(
                    m, q, dim_S, L_k, Y_k, V_ik, W_ik
                )

                print(
                    "residual",
                    f"dim_S: {dim_S}",
                    f"L_k: {L_k.shape}",
                    f"Y_k: {Y_k.shape}",
                )
                lmin = min(dim_S, j)
                # L_k and Y_k are sync'd (the sort of one is the same
                # on the other)
                indices = L_k.argsort()
                l_k = L_k[indices[lmin]]
                y_k = Y_k[:, indices[lmin]]

                print(f"{V_ik.shape}, {W_ik.shape}, {y_k.shape}")
                r_ik = np.dot(W_ik, y_k) - l_k * np.dot(V_ik, y_k)
                r_k = np.zeros(n)
                self.comm.Allgatherv([r_ik, MPI.DOUBLE], [r_k, self.distribution, self.offsets, MPI.DOUBLE])

                res = np.linalg.norm(r_k)
                if res < epsilon:
                    break

                print("preconditioning", f"||r_k||: {res}")
                t_ik = self.preconditioning(A_i, l_k, r_ik)

                print("mgs")
                V_ik = np.c_[V_ik, self.mgs(n, V_ik, t_ik)]
        return L_k


if __name__ == "__main__":

    # davidson initialization, since we're going for distributed data, that
    # stuff is actually just the buffers required on one MPI rank.

    # global size of the system
    n = 2 ** 8

    # number of total eigenvalues we want
    n_eig = 4

    # max number of iterations we allow (divergence cutoff)
    max_iter = n // 2

    # cutoff for error
    epsilon = 1e-8

    m, q = 10, 15

    print(f"init: {n}, {n_eig}, {max_iter}, {epsilon}")

    DM = DavidsonManager(MPI.COMM_WORLD,n)

    # lets build a garbage initial matrix
    A_i = np.zeros((DM.local_size, n))
    A_i = 0.0001 * np.random.randn(DM.local_size, n)

    print("init: I'm processor %d of %d. \n" % (DM.rank, DM.world_size))

    L_k = DM.distributed(A_i, n, n_eig, epsilon, max_iter, m, q)

    # fuse A across all nodes
    A = np.zeros((n, n))
    DM.comm.Allgatherv([A_i, MPI.DOUBLE], [A, DM.distribution, DM.offsets, MPI.DOUBLE])

    if DM.rank == 0:
        ref_l, ref_y = np.linalg.eig(A)
        ref_l = np.sort(ref_l)

    print(L_k, ref_l, len(L_k))
