#!/usr/bin/env python3
from qe.drivers import *
import sys

if __name__ == "__main__":

    fcidump_path = sys.argv[1]
    wf_path = sys.argv[2]
    N_det_target = int(sys.argv[3])
    try:
        driven_by = sys.argv[4]
    except IndexError:
        driven_by = "integral"

    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
    # Load wave function
    psi_coef, psi_det = load_wf(wf_path)
    # Hamiltonian engine
    lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by=driven_by)

    while len(psi_det) < N_det_target:
        E, psi_coef, psi_det = selection_step(lewis, n_ord, psi_coef, psi_det, len(psi_det))
        print(f"N_det: {len(psi_det)}, E {E}")
