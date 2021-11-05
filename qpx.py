#!/usr/bin/env python3

from qpx.types import *
from qpx.input import load_integrals, load_wf, load_eref
from qpx.hamiltonian import Hamiltonian
from qpx.powerplant import Powerplant
from dataclasses import dataclass
import numpy as np
from numpy.testing import assert_almost_equal

#  __
# (_   _  |  _   _ _|_ o  _  ._
# __) (/_ | (/_ (_  |_ | (_) | |
#
def selection_step(lewis: Hamiltonian, n_ord, psi_coef: Psi_coef, psi_det: Psi_det, n) -> Tuple[Energy, Psi_coef, Psi_det]:
    # 1. Generate a list of all the external determinant and their pt2 contribution
    # 2. Take the n  determinants who have the biggest contribution and add it the wave function psi
    # 3. Diagonalize H corresponding to this new wave function to get the new variational energy, and new psi_coef.

    # In the main code:
    # -> Go to 1., stop when E_pt2 < Threshold || N < Threshold
    # See example of chained call to this function in `test_f2_631g_1p5p5det`

    # 1.
    psi_external_det, psi_external_energy = Powerplant(lewis, psi_det).psi_external_pt2(psi_coef, n_ord)

    # 2.
    idx = np.argpartition(psi_external_energy, n)[:n]
    psi_det_extented = psi_det + [psi_external_det[i] for i in idx]

    # 3.
    return (*Powerplant(lewis, psi_det_extented).E_and_psi_coef, psi_det_extented)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fcidump", help="path to fcidump file, can be a glob")
    parser.add_argument("wf", help="path to wf file, can be a glob")
    parser.add_argument("eref", help="path to eref file, can be a glob")
    args = parser.parse_args()

    # load data
    n_orb, E0, d_one_e_integral, d_two_e_integral = load_integrals(args.fcidump)
    psi_coef, psi_det = load_wf(args.wf)
    # prepare generator
    lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0)
    E =  Powerplant(lewis, psi_det).E(psi_coef)

    # 
    E_ref = load_eref(args.eref)
    assert_almost_equal(E_ref, E)
