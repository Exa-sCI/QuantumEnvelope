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
wf_path = "data/c2_eq_hf_dz_1.1det.wf.gz"
driven_by = "integral"

# Load integrals
n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
# Load wave function
psi_coef, psi_det = load_wf(wf_path)
k = 12  # No. of iterations to run
N_det_target = (2**k) * len(
    psi_det
)  # Run k iterations where we double the no. of inital dets each time
# Hamiltonian engine

lewis = Hamiltonian_generator(
    comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
)

# How long does it take to build H

print(f"{rank}: Building the local Hamiltonian in //")
t1 = MPI.Wtime()
Hamiltonian_i = lewis.H_i
t2 = MPI.Wtime()
print(f"{rank}: Time took locally", t2 - t1)

sys.exit()

print("Initiate PT2 iteration")
while len(psi_det) < N_det_target:
    # How aRe these distriubuted
    local_dets = lewis.psi_local
    print(rank, "Local dets", local_dets)

    t1 = MPI.Wtime()
    E, psi_coef, psi_det = selection_step(comm, lewis, n_ord, psi_coef, psi_det, len(psi_det))
    # Update Hamiltonian engine
    lewis = Hamiltonian_generator(
        comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
    )
    t2 = MPI.Wtime()
    print(f"N_det: {len(psi_det)}, E {E}")
    print(rank, "Time elapsed:", t2 - t1)


t3 = MPI.Wtime()
print(rank, "Total time elapsed", t3)
