from dataclasses import dataclass
from .types import *
from .hamiltonian_utils import get_phase_idx_single_exc, get_phase_idx_double_exc,get_exc_degree
import numpy as np
from itertools import chain, product, combinations

@dataclass
class Hamiltonian_4e_determinant_driven(Hamiltonian_engine):

    d_two_e_integral: Two_electron_integral

    def H_two_e(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        """Assume that *all* the integrals are in
        `d_two_e_integral` In this function, for simplicity we don't use any
        symmetry sparse representation.  For real calculations, symmetries and
        storing only non-zeros needs to be implemented to avoid an explosion of
        the memory requirements."""
        return self.d_two_e_integral[(i, j, k, l)]

    # ~ ~ ~
    # H_4e
    # Helper functions for different excitation types (A,AA,B,BB,AB)
    # ~ ~ ~

    def H_i_i_4e_index(self, det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """indices of integrals that contribute to diagonal element <I|H|I>."""
        for i, j in combinations(det_i.alpha, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in combinations(det_i.beta, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in product(det_i.alpha, det_i.beta):
            yield (i, j, i, j), 1

    def H_i_j_single_4e_index(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant, sdet_k: Spin_determinant) -> Iterator[Two_electron_integral_index_phase]:
        """indices of integrals that contribute to <I|H|J>,
        when I and J differ by exactly one orbital."""
        phase, m, p = get_phase_idx_single_exc(sdet_i, sdet_j)
        for i in sdet_i:
            yield (m, i, p, i), phase
            yield (m, i, i, p), -phase
        for i in sdet_k:
            yield (m, i, p, i), phase

    def H_i_j_doubleAA_4e_index(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Iterator[Two_electron_integral_index_phase]:
        """indices of integrals that contribute to <I|H|J>,
        when I and J differ by exactly two orbitals within the same spin."""
        phase, h1, h2, p1, p2 = get_phase_idx_double_exc(sdet_i, sdet_j)
        yield (h1, h2, p1, p2), phase
        yield (h1, h2, p2, p1), -phase

    def H_i_j_doubleAB_4e_index(self, det_i: Determinant, det_j: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """indices of integrals that contribute to <I|H|J>,
        when I and J differ by exactly one alpha spin-orbital and one beta spin-orbital."""
        phaseA, h1, p1 = get_phase_idx_single_exc(det_i.alpha, det_j.alpha)
        phaseB, h2, p2 = get_phase_idx_single_exc(det_i.beta, det_j.beta)
        yield (h1, h2, p1, p2), phaseA * phaseB

    def H_i_j_4e_index(self, det_i: Determinant, det_j: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """General function to dispatch the evaluation of H_ij"""


        # ~ ~ ~
        # find exc_degree and call appropriate index function
        # ~ ~ ~

        ed_up, ed_dn = get_exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up, ed_dn) == (0, 0):
            yield from self.H_i_i_4e_index(det_i)
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            yield from self.H_i_j_single_4e_index(det_i.alpha, det_j.alpha, det_i.beta)
        elif (ed_up, ed_dn) == (0, 1):
            yield from self.H_i_j_single_4e_index(det_i.beta, det_j.beta, det_i.alpha)
        # Double excitation of same spin
        elif (ed_up, ed_dn) == (2, 0):
            yield from self.H_i_j_doubleAA_4e_index(det_i.alpha, det_j.alpha)
        elif (ed_up, ed_dn) == (0, 2):
            yield from self.H_i_j_doubleAA_4e_index(det_i.beta, det_j.beta)
        # Double excitation of opposite spins
        elif (ed_up, ed_dn) == (1, 1):
            yield from self.H_i_j_doubleAB_4e_index(det_i, det_j)

    def H_4e_index(self, psi_i, psi_j) -> Iterator[Tuple[Tuple[int,int],Two_electron_integral_index_phase]]:
        """
        for all integrals connecting each pair of determinants, return
        pair of det indices, integral index, and phase
        """
        for (a,det_i),(b,det_j) in product(enumerate(psi_i),enumerate(psi_j)):
            for idx, phase in self.H_i_j_4e_index(det_i, det_j):
                yield (a, b), idx, phase

    def H_4e(self, psi_i, psi_j) -> List[List[Energy]]:
        """
        build H by iterating over all pairs of dets and all corresponding integrals
        and accumulating sums in np array
        """
        # This is the function who will take foreever
        h = np.zeros(shape=(len(psi_i), len(psi_j)))
        for (a, b), (i, j, k, l), phase in self.H_4e_index(psi_i, psi_j):
            h[a, b] += phase * self.H_two_e(i, j, k, l)
        return h
