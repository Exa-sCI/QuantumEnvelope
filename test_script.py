#!/usr/bin/env python3
from qe.drivers import *
import mpi4py
from mpi4py import (
    MPI,
)  # Note this initializes and finalizes MPI session automatically
import sys

# Test the // implementation

comm = mpi4py.MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Load test files
# Carbon dimer, C2
fcidump_path = "data/c2_eq_hf_dz.fcidump.gz"
wf_path = "data/c2_eq_hf_dz_1.1det.ref.gz"
driven_by = "integral"

# Load integrals
n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
# Load wave function
psi_coef, psi_det = load_wf(wf_path)
k = 4  # No. of iterations to run
N_det_target = (2**k) * len(
    psi_det
)  # Run k iterations where we double the no. of inital dets each time
# Hamiltonian engine

lewis = Hamiltonian_generator(
    comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
)

# Let's time Davidson's
# if rank == 0:
#    print(f"No. of ranks: {size}")
#    print(f"{rank}: Diagonalizing H and computing ground state energy")
#    t1 = MPI.Wtime()
# E, psi_coef = Powerplant_manager(comm, lewis).E_and_psi_coef
#
# if rank == 0:
#    t2 = MPI.Wtime()
#    print(f"{rank}: E0 = {E}, Time took locally:", t2 - t1)
# sys.exit()
## How long does it take to build H
#
# if rank == 0:
#    print(f"No. of ranks: {size}")
#    print(f"{rank}: Building the local Hamiltonian in //")
#    t1 = MPI.Wtime()
# Hamiltonian_i = lewis.H_i
# if rank == 0:
#    t2 = MPI.Wtime()
#    print(f"{rank}: Time took locally", t2 - t1)
#
# sys.exit()
if rank == 0:
    print("Initiate PT2 iteration")
t_tot = 0
while len(psi_det) < N_det_target:
    # How aRe these distriubuted
    local_dets = lewis.psi_local
    if rank == 0:
        print(rank, "Local dets", local_dets)

    t1 = MPI.Wtime()
    E, psi_coef, psi_det = selection_step(comm, lewis, n_ord, psi_coef, psi_det, len(psi_det))
    # Update Hamiltonian engine
    lewis = Hamiltonian_generator(
        comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
    )
    t2 = MPI.Wtime()
    if rank == 0:
        print(f"N_det: {len(psi_det)}, E {E}")
        print(rank, "Time elapsed:", t2 - t1)

    t_tot += t2

if rank == 0:
    print(rank, "Total time elapsed", t_tot)
