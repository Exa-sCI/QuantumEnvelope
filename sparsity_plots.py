#!/usr/bin/env python3
from qe.drivers import *
import mpi4py
from mpi4py import (
    MPI,
)  # Note this initializes and finalizes MPI session automatically
import sys
import matplotlib
from matplotlib import pyplot

# Test the // implementation

comm = mpi4py.MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# Load test files

fcidump_path = "data/f2_631g.161det.fcidump"
wf_path = "data/f2_631g.161det.wf"
driven_by = "integral"

# Load integrals
n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
# Load wave function
psi_coef, psi_det = load_wf(wf_path)

lewis_int = Hamiltonian_generator(
    comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by="integral"
)

Hamiltonian = lewis_int.H
# E,_ = Powerplant_manager(comm, lewis_int).E_and_psi_coef
# print(E, -198.80842697960500)

# Build H and plot
plt = matplotlib.pyplot
plt.spy(Hamiltonian, precision=1e-15, markersize=2)
plt.show()
