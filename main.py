#!/usr/bin/env python3
from qe.drivers import *
from qe.io import *
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process  input arguments for Quantum Envelope calculation"
    )

    parser.add_argument(
        "--fcidump_path", help="path/filename of the FCIDUMP file containing integrals"
    )
    parser.add_argument(
        "--wf_path",
        help="path/filename of the wf file containing the wf coefficients and determinant list to begin with",
    )

    parser.add_argument(
        "-N_det_target",
        type=int,
        default=int(1000),
        required=False,
        help="Number of determinants to target",
    )

    parser.add_argument(
        "-driven_by",
        choices=["integral", "determinant"],
        default="integral",
        required=False,
        help="Way in which Hamiltonian is generated. Integral driven: local set of integrals, determinants are found on each node. Determinant driven: local set of determinants, all integrals are on each node.",
    )

    args = parser.parse_args()
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(args.fcidump_path)
    # Load wave function
    psi_coef, psi_det = load_wf(args.wf_path)
    # Hamiltonian engine
    comm = MPI.COMM_WORLD
    lewis = Hamiltonian_generator(
        comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=args.driven_by
    )

    while len(psi_det) < args.N_det_target:
        E, psi_coef, psi_det = selection_step(comm, lewis, n_ord, psi_coef, psi_det, len(psi_det))
        # Update Hamiltonian engine
        lewis = Hamiltonian_generator(
            comm,
            E0,
            d_one_e_integral,
            d_two_e_integral,
            psi_det,
            driven_by=args.driven_by,
        )
        print(f"N_det: {len(psi_det)}, E {E}")
