#!/usr/bin/env python3
from qe.drivers import *
import mpi4py
from mpi4py import (
    MPI,
)  # Note this initializes and finalizes MPI session automatically
import sys

# Move on to testing // PT2

# 1. Test // computation of E_var

comm = mpi4py.MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Generate internal wave=function
N_orb = 6
if rank == 0:  # Arbitrarily say 0 is master
    print(N_orb**4)
psi_full = [Determinant((0, 1), (0, 1))]  # Generate full wavefunctin
psi_full += Excitation(N_orb).gen_all_connected_determinant(psi_full)

d_two_e_integral = {}  # Create dictionary of two-electron integrals
for (i, j, k, l) in product(range(N_orb), repeat=4):
    d_two_e_integral[compound_idx4(i, j, k, l)] = 1

d_one_e_integral = {}  # Create dictionary of one-electron integrals
for (i, j) in product(range(N_orb), repeat=2):
    d_one_e_integral[(i, j)] = 1
E0 = 1  # Some arbitrary float for energy

# Create instance of Hamiltonian_generator class
print(f"Hi, I'm rank {rank}. I'm generating the local Hamiltonian")
HG_det = Hamiltonian_generator(
    comm, E0, d_one_e_integral, d_two_e_integral, psi_full, driven_by="determinant"
)
HG_int = Hamiltonian_generator(
    comm, E0, d_one_e_integral, d_two_e_integral, psi_full, driven_by="integral"
)

# 1. Test whether we are correctly building the Hamiltonian

H_full_det = HG_det.H
H_full_int = HG_int.H
if rank == 0:
    H_full_ref_det = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by="determinant").H(
        psi_full, psi_full
    )
    H_full_ref_int = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by="integral").H(
        psi_full, psi_full
    )
    print("||H_full_int - H_full_ref_int||", np.linalg.norm(H_full_int - H_full_ref_int))
    print("||H_full_det - H_full_ref_det||", np.linalg.norm(H_full_det - H_full_ref_det))
    # Make sure it's not empty..
    print("H_full_int", H_full_int)
    print("H_full_det", H_full_det)
    print("||H_full_int - H_full_det||", np.linalg.norm(H_full_int - H_full_det))

# 2. Test whether we properly generate the Hamiltonian matrix elements

k = 1  # Col. size of text matrix
if rank == 0:  # So all workers have same guess vectors, build on one and bcast
    V = np.random.rand(len(psi_full), k)  # Generate some random guess vectors
else:
    V = np.zeros(shape=(len(psi_full), k), dtype="float")

V = comm.bcast(V, root=0)  # This will throw an error if V is not preallocated on worker processes!

# Compute local matrix-matrix product
# W_det_i = HG_det.H_i_implicit_matrix_product(V)
W_local = HG_int.H_i_implicit_matrix_product(V)

W_full = None
recvcounts = None
# No. of elements received by each process, W_local has dimensions len(psi_local) \times k
# Get local size from the instance of generator classes
local_size = HG_int.local_size
assert local_size == HG_det.local_size
recvcounts = np.array(comm.gather(local_size * k, root=0))

if rank == 0:
    # pre-allocate space for full matrix-matrix product on master
    W_full = np.zeros(shape=(len(psi_full), k), dtype="float")

# Each node sends local chunk of matrix-matrix to master
comm.Gatherv(W_local, (W_full, recvcounts), root=0)
if rank == 0:
    # re-use H saved from earlier
    W_full_ref = np.dot(H_full_ref_int, V)
    print(np.concatenate((W_full, W_full_ref), axis=1))
    print("||W_full - W_full_ref||:", np.linalg.norm(W_full - W_full_ref))

# 3. Test the Davidson implementation

# Have Hamiltonians from earlier for ref..

# Pass instance of Hamiltonian generator
DM = Davidson_manager(comm, HG_int)  # Instance of Davidson_Manager class
print("init: I'm processor %d of %d. \n" % (DM.rank, DM.world_size))

# Set inputs
n = len(psi_full)  # Full problem size
n_eig = 3  # No. of eigenvalues we wan
m = 3  # Minimal subspace dim
max_iter = 100  # Max number of iterations we allow (divergence cutoff)
epsilon = 1e-8  # Cutoff for error
q = 30  # Max subspace dimension
if rank == 0:
    print(f"init: {n}, {n_eig}, {max_iter}, {epsilon}")

(Lambda_k, X_k) = DM.distributed_davidson(None, n_eig, epsilon, 1e-10, max_iter, m, q)

if rank == 0:
    ref_L, ref_Y = np.linalg.eigh(H_full_ref_int)
    ref_L = np.sort(ref_L)
    ref_L = ref_L[:n_eig]
    print(Lambda_k, ref_L)

# Move on to testing // PT2

# Have generator classes from earlier
print(f"Hi, I'm rank {rank}. I'm generating the local Hamiltonian")

# Integral-driven powerplant
PP_int = Powerplant_manager(comm, HG_int)
# Determinant-driven powerplant
PP_det = Powerplant_manager(comm, HG_det)

# Now, diagonalize Hamiltonian to get ground-state energy
E_0_ref, X_0 = DM.distributed_davidson(None, 1, epsilon, 1e-10, max_iter, 1, q)
psi_coef = X_0.tolist()
# print(psi_coef)
# Ref powerplants
lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by="integral")
PP_ref_int = Powerplant(lewis, psi_full, DM)

# Compute variatonal energy and compare
# TODO: Compare to old PP
E_0 = PP_int.E(psi_coef)
# E_0_ref = PP_ref_int.E(psi_coef)
if rank == 0:
    print("||E_0 - E_0_ref||", np.linalg.norm(E_0 - E_0_ref))

# Compare to old PP functions
external_pt2 = PP_int.psi_external_pt2(psi_coef)
# _, external_pt2_ref= PP_ref_int.psi_external_pt2(psi_coef, N_orb)
# print(external_pt2)
# print(external_pt2_ref)
