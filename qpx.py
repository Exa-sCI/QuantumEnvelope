#!/usr/bin/env python3

from qpx import *
from dataclasses import dataclass

# Yes, I like itertools
from itertools import chain, product, combinations, takewhile
import numpy as np

#  _
# |_     _ o _|_  _. _|_ o  _  ._
# |_ >< (_ |  |_ (_|  |_ | (_) | |
#
class Excitation(object):
    def __init__(self, n_orb):
        self.all_orbs = frozenset(range(1, n_orb + 1))

    def gen_all_excitation(self, spindet: Spin_determinant, ed: int) -> Iterator:
        """
        Generate list of pair -> hole from a determinant spin.

        >>> sorted(Excitation(4).gen_all_excitation( (1,2),2))
        [((1, 2), (3, 4))]
        >>> sorted(Excitation(4).gen_all_excitation( (1,2),1))
        [((1,), (3,)), ((1,), (4,)), ((2,), (3,)), ((2,), (4,))]
        """
        holes = combinations(spindet, ed)
        not_spindet = self.all_orbs - set(spindet)
        parts = combinations(not_spindet, ed)
        return product(holes, parts)

    def gen_all_connected_spindet(self, spindet: Spin_determinant, ed: int) -> Iterator:
        """
        Generate all the posible spin determinant relative to a excitation degree

        >>> sorted(Excitation(4).gen_all_connected_spindet( (1,2), 1))
        [(1, 3), (1, 4), (2, 3), (2, 4)]
        """

        def apply_excitation(exc: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Spin_determinant:
            # Warning use global variable spindet.
            lh, lp = exc
            s = (set(spindet) - set(lh)) | set(lp)
            return tuple(sorted(s))

        l_exc = self.gen_all_excitation(spindet, ed)
        return map(apply_excitation, l_exc)

    def gen_all_connected_det_from_det(self, det: Determinant) -> Iterator[Determinant]:
        """
        Generate all the determinant who are single or double exictation (aka connected) from the input determinant

        >>> sorted(Excitation(3).gen_all_connected_det_from_det( Determinant( (1,2), (1,) )))
        [Determinant(alpha=(1, 2), beta=(2,)),
         Determinant(alpha=(1, 2), beta=(3,)),
         Determinant(alpha=(1, 3), beta=(1,)),
         Determinant(alpha=(1, 3), beta=(2,)),
         Determinant(alpha=(1, 3), beta=(3,)),
         Determinant(alpha=(2, 3), beta=(1,)),
         Determinant(alpha=(2, 3), beta=(2,)),
         Determinant(alpha=(2, 3), beta=(3,))]
        """

        # All single exitation from alpha or for beta determinant
        # Then the production of the alpha, and beta (it's a double)
        # Then the double exitation form alpha or beta

        # We use l_single_a, and l_single_b twice. So we store them.
        l_single_a = set(self.gen_all_connected_spindet(det.alpha, 1))
        l_double_aa = self.gen_all_connected_spindet(det.alpha, 2)

        s_a = (Determinant(det_alpha, det.beta) for det_alpha in chain(l_single_a, l_double_aa))

        l_single_b = set(self.gen_all_connected_spindet(det.beta, 1))
        l_double_bb = self.gen_all_connected_spindet(det.beta, 2)

        s_b = (Determinant(det.alpha, det_beta) for det_beta in chain(l_single_b, l_double_bb))

        l_double_ab = product(l_single_a, l_single_b)

        s_ab = (Determinant(det_alpha, det_beta) for det_alpha, det_beta in l_double_ab)

        return chain(s_a, s_b, s_ab)

    def gen_all_connected_determinant(self, psi_det: Psi_det) -> Psi_det:
        """
        >>> d1 = Determinant( (1,2), (1,) ) ; d2 = Determinant( (1,3), (1,) )
        >>> len(Excitation(4).gen_all_connected_determinant( [ d1,d2 ] ))
        22

        We remove the connected determinant who are already inside the wave function. Order doesn't matter
        """
        return list(set(chain.from_iterable(map(self.gen_all_connected_det_from_det, psi_det))) - set(psi_det))


#
# |_|  _. ._ _  o | _|_  _  ._  o  _. ._
# | | (_| | | | | |  |_ (_) | | | (_| | |
#
@dataclass
class Hamiltonian(object):
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
    d_two_e_integral: Two_electron_integral
    E0: Energy

    def H_one_e(self, i: OrbitalIdx, j: OrbitalIdx) -> float:
        """One-electron part of the Hamiltonian: Kinetic energy (T) and
        Nucleus-electron potential (V_{en}). This matrix is symmetric."""
        return self.d_one_e_integral[(i, j)]

    def H_two_e(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        """Assume that *all* the integrals are in
        `d_two_e_integral` In this function, for simplicity we don't use any
        symmetry sparse representation.  For real calculations, symmetries and
        storing only non-zeros needs to be implemented to avoid an explosion of
        the memory requirements."""
        return self.d_two_e_integral[(i, j, k, l)]

    @staticmethod
    def get_phase_idx_single_exc(sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Hamiltonian.get_phase_idx_single_exc((1, 5, 7), (1, 23, 7))
        (1, 5, 23)
        >>> Hamiltonian.get_phase_idx_single_exc((1, 2, 9), (1, 9, 18))
        (-1, 2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        phase = 1
        for det, idx in ((sdet_i, h), (sdet_j, p)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase

        return (phase, h, p)

    @staticmethod
    def get_phase_idx_double_exc(sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Hamiltonian.get_phase_idx_double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (1, 3, 4, 12, 13)
        >>> Hamiltonian.get_phase_idx_double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (-1, 3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = 1
        for det, idx in ((sdet_i, h1), (sdet_j, p1), (sdet_j, p2), (sdet_i, h2)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase

        # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
        # Look like to be always true in our tests
        if (min(h2, p2) < max(h1, p1)) != (h2 < p1 or p2 < h1):
            phase = -phase
            print(">>> get_phase_idx_double_exc QP conditional was trigered! Please repport to the developpers", sdet_i, sdet_j)

        return (phase, h1, h2, p1, p2)

    @staticmethod
    def get_exc_degree(det_i: Determinant, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ
           between the two determinants.
        >>> Hamiltonian.get_exc_degree(Determinant(alpha=(1, 2), beta=(1, 2)),
        ...                            Determinant(alpha=(1, 3), beta=(5, 7)))
        (1, 2)
        """
        ed_up = len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
        ed_dn = len(set(det_i.beta).symmetric_difference(set(det_j.beta))) // 2
        return ed_up, ed_dn

    # ~ ~ ~
    # H_2e
    # ~ ~ ~
    def H_i_i_2e(self, det_i: Determinant) -> Energy:
        """Diagonal element of the Hamiltonian : <I|H|I>."""
        res = self.E0
        res += sum(self.H_one_e(i, i) for i in det_i.alpha)
        res += sum(self.H_one_e(i, i) for i in det_i.beta)
        return res

    def H_i_j_single_2e(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant, sdet_k: Spin_determinant) -> Energy:
        """<I|H|J>, when I and J differ by exactly one orbital."""
        phase, m, p = Hamiltonian.get_phase_idx_single_exc(sdet_i, sdet_j)
        return self.H_one_e(m, p) * phase

    def H_i_j_2e(self, det_i: Determinant, det_j: Determinant) -> Energy:
        """General function to dispatch the evaluation of H_ij"""
        ed_up, ed_dn = Hamiltonian.get_exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up, ed_dn) == (0, 0):
            return self.H_i_i_2e(det_i)
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            return self.H_i_j_single_2e(det_i.alpha, det_j.alpha, det_i.beta)
        elif (ed_up, ed_dn) == (0, 1):
            return self.H_i_j_single_2e(det_i.beta, det_j.beta, det_i.alpha)
        else:
            return 0.0

    def H_2e(self, psi_i, psi_j) -> List[List[Energy]]:
        h = np.array([self.H_i_j_2e(det_i, det_j) for det_i, det_j in product(psi_i, psi_j)])
        return h.reshape(len(psi_i), len(psi_j))

    # ~ ~ ~
    # H_4e
    # ~ ~ ~
    def H_i_i_4e_index(self, det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        for i, j in combinations(det_i.alpha, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in combinations(det_i.beta, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in product(det_i.alpha, det_i.beta):
            yield (i, j, i, j), 1

    def H_i_j_single_4e_index(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant, sdet_k: Spin_determinant) -> Iterator[Two_electron_integral_index_phase]:
        """<I|H|J>, when I and J differ by exactly one orbital."""
        phase, m, p = Hamiltonian.get_phase_idx_single_exc(sdet_i, sdet_j)
        for i in sdet_i:
            yield (m, i, p, i), phase
            yield (m, i, i, p), -phase
        for i in sdet_k:
            yield (m, i, p, i), phase

    def H_i_j_doubleAA_4e_index(self, sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Iterator[Two_electron_integral_index_phase]:
        """<I|H|J>, when I and J differ by exactly two orbitals within
        the same spin."""
        phase, h1, h2, p1, p2 = Hamiltonian.get_phase_idx_double_exc(sdet_i, sdet_j)
        yield (h1, h2, p1, p2), phase
        yield (h1, h2, p2, p1), -phase

    def H_i_j_doubleAB_4e_index(self, det_i: Determinant, det_j: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
        one beta spin-orbital."""
        phaseA, h1, p1 = Hamiltonian.get_phase_idx_single_exc(det_i.alpha, det_j.alpha)
        phaseB, h2, p2 = Hamiltonian.get_phase_idx_single_exc(det_i.beta, det_j.beta)
        yield (h1, h2, p1, p2), phaseA * phaseB

    def H_i_j_4e_index(self, det_i: Determinant, det_j: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """General function to dispatch the evaluation of H_ij"""
        ed_up, ed_dn = Hamiltonian.get_exc_degree(det_i, det_j)
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

    def H_4e_index(self, psi_i, psi_j) -> Iterator[Two_electron_integral_index_phase]:
        for a, det_i in enumerate(psi_i):
            for b, det_j in enumerate(psi_j):
                for idx, phase in self.H_i_j_4e_index(det_i, det_j):
                    yield (a, b), idx, phase

    def H_4e(self, psi_i, psi_j) -> List[List[Energy]]:
        # This is the function who will take foreever
        h = np.zeros(shape=(len(psi_i), len(psi_j)))
        for (a, b), (i, j, k, l), phase in self.H_4e_index(psi_i, psi_j):
            h[a, b] += phase * self.H_two_e(i, j, k, l)
        return h

    # ~ ~ ~
    # H_i_i
    # ~ ~ ~
    def H_i_i(self, det_i) -> List[Energy]:
        H_i_i_4e = sum(phase * self.H_two_e(*idx) for idx, phase in self.H_i_i_4e_index(det_i))
        return self.H_i_i_2e(det_i) + H_i_i_4e

    # ~ ~ ~
    # H
    # ~ ~ ~
    def H(self, psi_i: Psi_det, psi_j: Psi_det) -> List[List[Energy]]:
        """Return a matrix of size psi_i x psi_j containing the value of the Hamiltonian.
        Note that when psi_i == psi_j, this matrix is an hermitian."""
        return self.H_2e(psi_i, psi_j) + self.H_4e(psi_i, psi_j)


#  _                  _
# |_) _        _  ._ |_) |  _. ._ _|_
# |  (_) \/\/ (/_ |  |   | (_| | | |_
#
@dataclass
class Powerplant(object):
    """
    Compute all the Energy and associated value from a psi_det.
    E denote the variational energy
    """

    lewis: Hamiltonian
    psi_det: Psi_det

    def E(self, psi_coef: Psi_coef) -> Energy:
        # Vector * Vector.T * Matrix
        return np.einsum("i,j,ij ->", psi_coef, psi_coef, self.lewis.H(self.psi_det, self.psi_det))

    @property
    def E_and_psi_coef(self) -> Tuple[Energy, Psi_coef]:
        # Return lower eigenvalue (aka the new E) and lower evegenvector (aka the new psi_coef)
        psi_H_psi = self.lewis.H(self.psi_det, self.psi_det)
        energies, coeffs = np.linalg.eigh(psi_H_psi)
        return energies[0], coeffs[:, 0]

    def psi_external_pt2(self, psi_coef: Psi_coef, n_orb) -> Tuple[Psi_det, List[Energy]]:
        # Compute the pt2 contrution of all the external (aka connected) determinant.
        #   eα=⟨Ψ(n)∣H∣∣α⟩^2 / ( E(n)−⟨α∣H∣∣α⟩ )
        psi_external = Excitation(n_orb).gen_all_connected_determinant(self.psi_det)

        nomitator = np.einsum("i,ij -> j", psi_coef, self.lewis.H(self.psi_det, psi_external))  # vector * Matrix -> vector
        denominator = np.divide(1.0, self.E(psi_coef) - np.array([self.lewis.H_i_i(d) for d in psi_external]))

        return psi_external, np.einsum("i,i,i -> i", nomitator, nomitator, denominator)  # vector * vector * vector -> scalar

    def E_pt2(self, psi_coef: Psi_coef, n_orb) -> Energy:
        # The sum of the pt2 contribution of each external determinant
        _, psi_external_energy = self.psi_external_pt2(psi_coef, n_orb)
        return sum(psi_external_energy)


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

    args = parser.parse_args()

    # load data
    n_orb, E0, d_one_e_integral, d_two_e_integral = load_integrals(args.fcidump)
    psi_coef, psi_det = load_wf(args.wf)

    print(psi_det)

    # prepare generator
    lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0)
