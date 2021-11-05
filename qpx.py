#!/usr/bin/env python3

from qpx.types import *
from qpx.input import load_integrals, load_wf, load_eref, save_wf
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

def make_simple_wf(n_orb: int, n_elec: int) -> Tuple[List[float], List[Determinant]]:
    """
    make a single-determinant wavefunction with the lowest orbitals occupied
    """
    psi_coef = [1.0]
    nb = n_elec//2
    na = n_elec - nb
    psi_det = [Determinant(tuple(range(1,na+1)), tuple(range(1,nb+1)))]
    return psi_coef, psi_det


def run_cipsi(fcidump_path: str, n_elec: int, n_iter: int, outpath: str, iter_expansion_factor: float=1.0, do_pt2: bool=False):
    """
    using integrals from fcidump, do `n_iter` cipsi iterations starting from a simple single-det wf
    at each iteration, add a number of dets equal to current n_det times iter_expansion_factor

    """
    # load integrals, n_orb and create Hamiltonian
    n_orb, E_0, int1e, int2e = load_integrals(fcidump_path)
    ham = Hamiltonian(int1e,int2e,E_0)

    # make simple single-det starting wf
    psi_coef, psi_det = make_simple_wf(n_orb,n_elec)

    pt2str=''
    pt2header='E_PT2' if do_pt2 else ''
    # iterate
    print(f'iter       n_det      E_var                      '+pt2header)
    for i in range(n_iter):
        E_i, psi_coef, psi_det = selection_step(ham,n_orb,psi_coef,psi_det,int(iter_expansion_factor*len(psi_det)))
        if do_pt2:
            E_pt2_i = Powerplant(ham, psi_det).E_pt2(psi_coef, n_orb)
            pt2str = f'{E_pt2_i:15.5f}'
        print(f'{i:4d}  {len(psi_det):10d}{E_i:25.15f}'+pt2str)
        save_wf(psi_coef,psi_det,f'{outpath}.wf_{i:05d}_{len(psi_det)}_det',n_orb)

    return E_i,  psi_coef, psi_det


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
    print(args.eref)
    E_ref = load_eref(args.eref)
    assert_almost_equal(E_ref, E)
