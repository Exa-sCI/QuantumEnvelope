from .hamiltonian_utils import get_phase_idx_single_exc, get_exc_degree
from .types import *
from itertools import product
import numpy as np
from dataclasses import dataclass

@dataclass
class Hamiltonian_2e(object):
    """
    Now, we consider the Hamiltonian matrix in the basis of Slater determinants.
    Slater-Condon rules are used to compute the matrix elements <I|H|J> where I
    and J are Slater determinants.

     ~
     Slater-Condon Rules
     ~

     https://en.wikipedia.org/wiki/Slater%E2%80%93Condon_rules
     https://arxiv.org/abs/1311.6244

     * H is symmetric
     * If I and J differ by more than 2 orbitals, <I|H|J> = 0, so the number of
       non-zero elements of H is bounded by N_det x ( N_alpha x (n_orb - N_alpha))^2,
       where N_det is the number of determinants, N_alpha is the number of
       alpha-spin electrons (N_alpha >= N_beta), and n_orb is the number of
       molecular orbitals.  So the number of non-zero elements scales linearly with
       the number of selected determinant.
    """

    d_one_e_integral: One_electron_integral
    E0: Energy

    def H_one_e(self, i: OrbitalIdx, j: OrbitalIdx) -> float:
        """One-electron part of the Hamiltonian: Kinetic energy (T) and
        Nucleus-electron potential (V_{en}). This matrix is symmetric."""
        return self.d_one_e_integral[(i, j)]

    # ~ ~ ~
    # H_2e
    # ~ ~ ~
    def H_i_i_2e(self, det_i: Determinant) -> Energy:
        """Diagonal element of the Hamiltonian : <I|H|I>."""
        res = self.E0
        res += sum(self.H_one_e(i, i) for i in det_i.alpha)
        res += sum(self.H_one_e(i, i) for i in det_i.beta)
        return res

    def H_i_j_single_2e(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Energy:
        """<I|H|J>, when I and J differ by exactly one orbital."""
        phase, m, p = get_phase_idx_single_exc(sdet_i, sdet_j)
        return self.H_one_e(m, p) * phase

    def H_i_j_2e(self, det_i: Determinant, det_j: Determinant) -> Energy:
        """General function to dispatch the evaluation of H_ij"""
        ed_up, ed_dn = get_exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up, ed_dn) == (0, 0):
            return self.H_i_i_2e(det_i)
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            return self.H_i_j_single_2e(det_i.alpha, det_j.alpha)
        elif (ed_up, ed_dn) == (0, 1):
            return self.H_i_j_single_2e(det_i.beta, det_j.beta)
        else:
            return 0.0

    def H_2e(self, psi_i, psi_j) -> List[List[Energy]]:
        h = np.array([self.H_i_j_2e(det_i, det_j) for det_i, det_j in product(psi_i, psi_j)])
        return h.reshape(len(psi_i), len(psi_j))

