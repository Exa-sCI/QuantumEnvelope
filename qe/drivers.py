# Yes, I like itertools
from dataclasses import dataclass
from itertools import chain, product, combinations, takewhile, permutations, accumulate
from functools import partial, cached_property, cache
from collections import defaultdict
import numpy as np

# Import mpi4py and utilities
from mpi4py import MPI  # Note this initializes and finalizes MPI session automatically

from qe.fundamental_types import (
    Spin_determinant,
    Determinant,
    Psi_det,
    OrbitalIdx,
    Energy,
    Psi_coef,
    One_electron_integral,
    Two_electron_integral,
    Two_electron_integral_index_phase,
)
from typing import Iterator, Set, Tuple, List, Dict
from qe.integral_indexing_utils import compound_idx4_reverse, compound_idx4, canonical_idx4

#  _____      _                       _   _
# |_   _|    | |                     | | | |
#   | | _ __ | |_ ___  __ _ _ __ __ _| | | |_ _   _ _ __   ___  ___
#   | || '_ \| __/ _ \/ _` | '__/ _` | | | __| | | | '_ \ / _ \/ __|
#  _| || | | | ||  __/ (_| | | | (_| | | | |_| |_| | |_) |  __/\__ \
#  \___/_| |_|\__\___|\__, |_|  \__,_|_|  \__|\__, | .__/ \___||___/
#                      __/ |                   __/ | |
#                     |___/                   |___/|_|


def integral_category(i, j, k, l):
    """
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    | label |                   | ik/jl i/k j/l | i/j j/k k/l i/l | singles                       | doubles                      | diagonal                    |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   A   | i=j=k=l (1,1,1,1) |   =    =   =  |  =   =   =   =  |                               |                              | coul. (1 occ. both spins?)  |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   B   | i=k<j=l (1,2,1,2) |   <    =   =  |  <   >   <   <  |                               |                              | coul. (1,2 any spin occ.?)  |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i=k<j<l (1,2,1,3) |   <    =   <  |  <   >   <   <  | 2<->3, 1 occ. (any spin)      |                              |                             |
    |   C   | i<k<j=l (1,3,2,3) |   <    <   =  |  <   >   <   <  | 1<->2, 3 occ. (any spin)      |                              |                             |
    |       | j<i=k<l (2,1,2,3) |   <    =   <  |  >   <   <   <  | 1<->3, 2 occ. (any spin)      |                              |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   D   | i=j=k<l (1,1,1,2) |   <    =   <  |  =   =   <   <  | 1<->2, 1 occ. (opposite spin) |                              |                             |
    |       | i<j=k=l (1,2,2,2) |   <    <   =  |  <   =   =   <  | 1<->2, 2 occ. (opposite spin) |                              |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i=j<k<l (1,1,2,3) |   <    <   <  |  =   <   <   <  | 2<->3, 1 occ. (same spin)     | 1a<->2a x 1b<->3b, (and a/b) |                             |
    |   E   | i<j=k<l (1,2,2,3) |   <    <   <  |  <   =   <   <  | 1<->3, 2 occ. (same spin)     | 1a<->2a x 2b<->3b, (and a/b) |                             |
    |       | i<j<k=l (1,2,3,3) |   <    <   <  |  <   <   =   <  | 1<->2, 3 occ. (same spin)     | 1a<->3a x 2b<->3b, (and a/b) |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |   F   | i=j<k=l (1,1,2,2) |   =    <   <  |  =   <   =   <  |                               | 1a<->2a x 1b<->2b            | exch. (1,2 same spin occ.?) |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    |       | i<j<k<l (1,2,3,4) |   <    <   <  |  <   <   <   <  |                               | 1<->3 x 2<->4                |                             |
    |   G   | i<k<j<l (1,3,2,4) |   <    <   <  |  <   >   <   <  |                               | 1<->2 x 3<->4                |                             |
    |       | j<i<k<l (2,1,3,4) |   <    <   <  |  >   <   <   <  |                               | 1<->4 x 2<->3                |                             |
    +-------+-------------------+---------------+-----------------+-------------------------------+------------------------------+-----------------------------+
    """
    assert (i, j, k, l) == canonical_idx4(i, j, k, l)
    if i == l:
        return "A"
    elif (i == k) and (j == l):
        return "B"
    elif (i == k) or (j == l):
        if j == k:
            return "D"
        else:
            return "C"
    elif j == k:
        return "E"
    elif (i == j) and (k == l):
        return "F"
    elif (i == j) or (k == l):
        return "E"
    else:
        return "G"


#   _____         _ _        _   _
#  |  ___|       (_) |      | | (_)
#  | |____  _____ _| |_ __ _| |_ _  ___  _ __
#  |  __\ \/ / __| | __/ _` | __| |/ _ \| '_ \
#  | |___>  < (__| | || (_| | |_| | (_) | | | |
#  \____/_/\_\___|_|\__\__,_|\__|_|\___/|_| |_|
#
class Excitation:
    def __init__(self, n_orb):
        self.all_orbs = frozenset(range(n_orb))

    def gen_all_excitation(self, spindet: Spin_determinant, ed: int) -> Iterator:
        """
        Generate list of pair -> hole from a determinant spin.

        >>> sorted(Excitation(4).gen_all_excitation((0, 1),2))
        [((0, 1), (2, 3))]
        >>> sorted(Excitation(4).gen_all_excitation((0, 1),1))
        [((0,), (2,)), ((0,), (3,)), ((1,), (2,)), ((1,), (3,))]
        """
        holes = combinations(spindet, ed)
        not_spindet = self.all_orbs - set(spindet)
        parts = combinations(not_spindet, ed)
        return product(holes, parts)

    @staticmethod
    def apply_excitation(spindet, exc: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        lh, lp = exc
        s = (set(spindet) - set(lh)) | set(lp)
        return tuple(sorted(s))

    def gen_all_connected_spindet(self, spindet: Spin_determinant, ed: int) -> Iterator:
        """
        Generate all the posible spin determinant relative to a excitation degree

        >>> sorted(Excitation(4).gen_all_connected_spindet((0, 1), 1))
        [(0, 2), (0, 3), (1, 2), (1, 3)]
        """
        l_exc = self.gen_all_excitation(spindet, ed)
        apply_excitation_to_spindet = partial(Excitation.apply_excitation, spindet)
        return map(apply_excitation_to_spindet, l_exc)

    def gen_all_connected_det_from_det(self, det: Determinant) -> Iterator[Determinant]:
        """
        Generate all the determinant who are single or double exictation (aka connected) from the input determinant

        >>> sorted(Excitation(3).gen_all_connected_det_from_det( Determinant((0, 1), (0,))))
        [Determinant(alpha=(0, 1), beta=(1,)),
         Determinant(alpha=(0, 1), beta=(2,)),
         Determinant(alpha=(0, 2), beta=(0,)),
         Determinant(alpha=(0, 2), beta=(1,)),
         Determinant(alpha=(0, 2), beta=(2,)),
         Determinant(alpha=(1, 2), beta=(0,)),
         Determinant(alpha=(1, 2), beta=(1,)),
         Determinant(alpha=(1, 2), beta=(2,))]
        """

        # All single exitation from alpha or for beta determinant
        # Then the production of the alpha, and beta (its a double)
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
        >>> d1 = Determinant((0, 1), (0,) ) ; d2 = Determinant((0, 2), (0,) )
        >>> len(Excitation(4).gen_all_connected_determinant( [ d1,d2 ] ))
        22

        We remove the connected determinant who are already inside the wave function. Order doesn't matter
        """
        return list(
            set(chain.from_iterable(map(self.gen_all_connected_det_from_det, psi_det)))
            - set(psi_det)
        )

    @staticmethod
    @cache
    def exc_degree_spindet(spindet_i: Spin_determinant, spindet_j: Spin_determinant) -> int:
        return len(set(spindet_i).symmetric_difference(set(spindet_j))) // 2

    @staticmethod
    # @cache
    def exc_degree(det_i: Determinant, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ
           between the two determinants.
        >>> Excitation.exc_degree(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                     Determinant(alpha=(0, 2), beta=(4, 6)))
        (1, 2)
        """
        ed_up = Excitation.exc_degree_spindet(det_i.alpha, det_j.alpha)
        ed_dn = Excitation.exc_degree_spindet(det_i.beta, det_j.beta)
        return (ed_up, ed_dn)


#  ______ _                        _   _       _
#  | ___ \ |                      | | | |     | |
#  | |_/ / |__   __ _ ___  ___    | |_| | ___ | | ___
#  |  __/| '_ \ / _` / __|/ _ \   |  _  |/ _ \| |/ _ \
#  | |   | | | | (_| \__ \  __/_  | | | | (_) | |  __/
#  \_|   |_| |_|\__,_|___/\___( ) \_| |_/\___/|_|\___|
#                             |/
#
#                   _  ______          _   _      _
#                  | | | ___ \        | | (_)    | |
#    __ _ _ __   __| | | |_/ /_ _ _ __| |_ _  ___| | ___  ___
#   / _` | '_ \ / _` | |  __/ _` | '__| __| |/ __| |/ _ \/ __|
#  | (_| | | | | (_| | | | | (_| | |  | |_| | (__| |  __/\__ \
#   \__,_|_| |_|\__,_| \_|  \__,_|_|   \__|_|\___|_|\___||___/
#
#
class PhaseIdx(object):
    @staticmethod
    def single_phase(sdet_i, sdet_j, h, p):
        phase = 1
        for det, idx in ((sdet_i, h), (sdet_j, p)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase
        return phase

    @staticmethod
    def single_exc_no_phase(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[OrbitalIdx, OrbitalIdx]:
        """hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> PhaseIdx.single_exc_no_phase((1, 5, 7), (1, 23, 7))
        (5, 23)
        >>> PhaseIdx.single_exc_no_phase((1, 2, 9), (1, 9, 18))
        (2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        return h, p

    @staticmethod
    def single_exc(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> PhaseIdx.single_exc((0, 4, 6), (0, 22, 6))
        (1, 4, 22)
        >>> PhaseIdx.single_exc((0, 1, 8), (0, 8, 17))
        (-1, 1, 17)
        """
        h, p = PhaseIdx.single_exc_no_phase(sdet_i, sdet_j)

        return PhaseIdx.single_phase(sdet_i, sdet_j, h, p), h, p

    @staticmethod
    def double_phase(sdet_i, sdet_j, h1, h2, p1, p2):
        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = PhaseIdx.single_phase(sdet_i, sdet_j, h1, p1) * PhaseIdx.single_phase(
            sdet_j, sdet_i, p2, h2
        )
        if h2 < h1:
            phase *= -1
        if p2 < p1:
            phase *= -1
        return phase

    @staticmethod
    def double_exc_no_phase(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> PhaseIdx.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (3, 4, 12, 13)
        >>> PhaseIdx.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        return h1, h2, p1, p2

    @staticmethod
    def double_exc(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> PhaseIdx.double_exc((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 4, 5, 6, 7, 8, 11, 12))
        (1, 2, 3, 11, 12)
        >>> PhaseIdx.double_exc((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 3, 4, 5, 6, 7, 11, 17))
        (-1, 2, 8, 11, 17)
        """

        h1, h2, p1, p2 = PhaseIdx.double_exc_no_phase(sdet_i, sdet_j)

        return PhaseIdx.double_phase(sdet_i, sdet_j, h1, h2, p1, p2), h1, h2, p1, p2


#   _   _                 _ _ _              _
#  | | | |               (_) | |            (_)
#  | |_| | __ _ _ __ ___  _| | |_ ___  _ __  _  __ _ _ __
#  |  _  |/ _` | '_ ` _ \| | | __/ _ \| '_ \| |/ _` | '_ \
#  | | | | (_| | | | | | | | | || (_) | | | | | (_| | | | |
#  \_| |_/\__,_|_| |_| |_|_|_|\__\___/|_| |_|_|\__,_|_| |_|
#

#    _             _
#   / \ ._   _    |_ |  _   _ _|_ ._ _  ._
#   \_/ | | (/_   |_ | (/_ (_  |_ | (_) | |
#
@dataclass
class Hamiltonian_one_electron(object):
    """
    One-electron part of the Hamiltonian: Kinetic energy (T) and
    Nucleus-electron potential (V_{en}). This matrix is symmetric.
    """

    integrals: One_electron_integral
    E0: Energy

    def H_ij_orbital(self, i: OrbitalIdx, j: OrbitalIdx) -> float:

        return self.integrals[(i, j)]

    def H_ii(self, det_i: Determinant) -> Energy:
        """Diagonal element of the Hamiltonian : <I|H|I>."""
        res = self.E0
        res += sum(self.H_ij_orbital(i, i) for i in det_i.alpha)
        res += sum(self.H_ij_orbital(i, i) for i in det_i.beta)
        return res

    def H_ij(self, det_i: Determinant, det_j: Determinant) -> Energy:
        """General function to dispatch the evaluation of H_ij"""

        def H_ij_spindet(sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Energy:
            """<I|H|J>, when I and J differ by exactly one orbital."""
            phase, m, p = PhaseIdx.single_exc(sdet_i, sdet_j)
            return self.H_ij_orbital(m, p) * phase

        ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up, ed_dn) == (0, 0):
            return self.H_ii(det_i)
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            return H_ij_spindet(det_i.alpha, det_j.alpha)
        elif (ed_up, ed_dn) == (0, 1):
            return H_ij_spindet(det_i.beta, det_j.beta)
        else:
            return 0.0

    def H(self, psi_i, psi_j) -> List[List[Energy]]:
        h = np.array([self.H_ij(det_i, det_j) for det_i, det_j in product(psi_i, psi_j)])
        return h.reshape(len(psi_i), len(psi_j))


#   ___            _
#    |       _    |_ |  _   _ _|_ ._ _  ._   _
#    | \/\/ (_)   |_ | (/_ (_  |_ | (_) | | _>
#    _                                          _
#   | \  _ _|_  _  ._ ._ _  o ._   _. ._ _|_   | \ ._ o     _  ._
#   |_/ (/_ |_ (/_ |  | | | | | | (_| | | |_   |_/ |  | \/ (/_ | |
#
@dataclass
class Hamiltonian_two_electrons_determinant_driven(object):
    d_two_e_integral: Two_electron_integral

    def H_ijkl_orbital(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        """Assume that *all* the integrals are in
        `d_two_e_integral` In this function, for simplicity we don't use any
        symmetry sparse representation.  For real calculations, symmetries and
        storing only non-zeros needs to be implemented to avoid an explosion of
        the memory requirements."""
        key = compound_idx4(i, j, k, l)
        return self.d_two_e_integral[key]

    @staticmethod
    def H_ii_indices(det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """Diagonal element of the Hamiltonian : <I|H|I>.
        >>> sorted(Hamiltonian_two_electrons_determinant_driven.H_ii_indices( Determinant((0,1),(2,3))))
        [((0, 1, 0, 1), 1), ((0, 1, 1, 0), -1), ((0, 2, 0, 2), 1), ((0, 3, 0, 3), 1),
         ((1, 2, 1, 2), 1), ((1, 3, 1, 3), 1), ((2, 3, 2, 3), 1), ((2, 3, 3, 2), -1)]
        """
        for i, j in combinations(det_i.alpha, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in combinations(det_i.beta, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in product(det_i.alpha, det_i.beta):
            yield (i, j, i, j), 1

    @staticmethod
    def H_ij_indices(
        det_i: Determinant, det_j: Determinant
    ) -> Iterator[Two_electron_integral_index_phase]:
        """General function to dispatch the evaluation of H_ij"""

        def H_ij_single_indices(
            sdet_i: Spin_determinant, sdet_j: Spin_determinant, sdet_k: Spin_determinant
        ) -> Iterator[Two_electron_integral_index_phase]:
            """<I|H|J>, when I and J differ by exactly one orbital"""
            phase, h, p = PhaseIdx.single_exc(sdet_i, sdet_j)
            for i in sdet_i:
                yield (h, i, p, i), phase
                yield (h, i, i, p), -phase
            for i in sdet_k:
                yield (h, i, p, i), phase

        def H_ij_doubleAA_indices(
            sdet_i: Spin_determinant, sdet_j: Spin_determinant
        ) -> Iterator[Two_electron_integral_index_phase]:
            """<I|H|J>, when I and J differ by exactly two orbitals within
            the same spin."""
            phase, h1, h2, p1, p2 = PhaseIdx.double_exc(sdet_i, sdet_j)
            yield (h1, h2, p1, p2), phase
            yield (h1, h2, p2, p1), -phase

        def H_ij_doubleAB_2e_index(
            det_i: Determinant, det_j: Determinant
        ) -> Iterator[Two_electron_integral_index_phase]:
            """<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
            one beta spin-orbital."""
            phaseA, h1, p1 = PhaseIdx.single_exc(det_i.alpha, det_j.alpha)
            phaseB, h2, p2 = PhaseIdx.single_exc(det_i.beta, det_j.beta)
            yield (h1, h2, p1, p2), phaseA * phaseB

        ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up, ed_dn) == (0, 0):
            yield from Hamiltonian_two_electrons_determinant_driven.H_ii_indices(det_i)
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            yield from H_ij_single_indices(det_i.alpha, det_j.alpha, det_i.beta)
        elif (ed_up, ed_dn) == (0, 1):
            yield from H_ij_single_indices(det_i.beta, det_j.beta, det_i.alpha)
        # Double excitation of same spin
        elif (ed_up, ed_dn) == (2, 0):
            yield from H_ij_doubleAA_indices(det_i.alpha, det_j.alpha)
        elif (ed_up, ed_dn) == (0, 2):
            yield from H_ij_doubleAA_indices(det_i.beta, det_j.beta)
        # Double excitation of opposite spins
        elif (ed_up, ed_dn) == (1, 1):
            yield from H_ij_doubleAB_2e_index(det_i, det_j)

    @staticmethod
    def H_indices(
        psi_internal: Psi_det, psi_j: Psi_det
    ) -> Iterator[Two_electron_integral_index_phase]:
        for a, det_i in enumerate(psi_internal):
            for b, det_j in enumerate(psi_j):
                for idx, phase in Hamiltonian_two_electrons_determinant_driven.H_ij_indices(
                    det_i, det_j
                ):
                    yield (a, b), idx, phase

    def H(self, psi_internal: Psi_det, psi_external: Psi_det) -> List[List[Energy]]:
        # det_external_to_index = {d: i for i, d in enumerate(psi_external)}
        # This is the function who will take foreever
        h = np.zeros(shape=(len(psi_internal), len(psi_external)))
        for (a, b), (i, j, k, l), phase in self.H_indices(psi_internal, psi_external):
            h[a, b] += phase * self.H_ijkl_orbital(i, j, k, l)
        return h

    def H_ii(self, det_i: Determinant):
        return sum(
            phase * self.H_ijkl_orbital(i, j, k, l)
            for (i, j, k, l), phase in self.H_ii_indices(det_i)
        )


#   ___            _
#    |       _    |_ |  _   _ _|_ ._ _  ._   _
#    | \/\/ (_)   |_ | (/_ (_  |_ | (_) | | _>
#   ___                           _
#    |  ._ _|_  _   _  ._ _. |   | \ ._ o     _  ._
#   _|_ | | |_ (/_ (_| | (_| |   |_/ |  | \/ (/_ | |
#                   _|


@dataclass
class Hamiltonian_two_electrons_integral_driven(object):
    d_two_e_integral: Two_electron_integral

    def H_ijkl_orbital(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        """Assume that *all* the integrals are in
        `d_two_e_integral` In this function, for simplicity we don't use any
        symmetry sparse representation.  For real calculations, symmetries and
        storing only non-zeros needs to be implemented to avoid an explosion of
        the memory requirements."""
        key = compound_idx4(i, j, k, l)
        return self.d_two_e_integral[key]

    @staticmethod
    def H_ii_indices(det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """Diagonal element of the Hamiltonian : <I|H|I>.
        >>> sorted(Hamiltonian_two_electrons_integral_driven.H_ii_indices( Determinant((0,1),(2,3))))
        [((0, 1, 0, 1), 1), ((0, 1, 1, 0), -1), ((0, 2, 0, 2), 1), ((0, 3, 0, 3), 1),
         ((1, 2, 1, 2), 1), ((1, 3, 1, 3), 1), ((2, 3, 2, 3), 1), ((2, 3, 3, 2), -1)]
        """
        for i, j in combinations(det_i.alpha, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in combinations(det_i.beta, 2):
            yield (i, j, i, j), 1
            yield (i, j, j, i), -1

        for i, j in product(det_i.alpha, det_i.beta):
            yield (i, j, i, j), 1

    @staticmethod
    def get_dets_occ_in_orbitals(
        spindet_occ: Dict[OrbitalIdx, Set[int]],
        oppspindet_occ: Dict[OrbitalIdx, Set[int]],
        d_orbitals: Dict[str, Set[OrbitalIdx]],
        which_orbitals,
    ):
        """
        Get indices of determinants that are occupied in the orbitals d_orbitals.
        Input which_orbitals = "all" or "any" indicates if we want dets occupied in all of the indices, or just any of the indices
        >>> Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals({0: {0}, 1: {0, 1}, 3: {1}}, {1: {0}, 2: {0}, 4: {1}, 5: {1}},  {"same": {0, 1}, "opposite": {}}, "all")
        {0}
        >>> Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals({0: {0}, 1: {0, 1}, 3: {1}}, {1: {0}, 2: {0}, 4: {1}, 5: {1}},  {"same": {0}, "opposite": {4}}, "all")
        set()
        """
        l = []
        for spintype, indices in d_orbitals.items():
            if spintype == "same":
                l += [spindet_occ[o] for o in indices]
            if spintype == "opposite":
                l += [oppspindet_occ[o] for o in indices]
        if which_orbitals == "all":
            return set.intersection(*l)
        else:
            return set.union(*l)

    @staticmethod
    def get_dets_via_orbital_occupancy(
        spindet_occ: Dict[OrbitalIdx, Set[int]],
        oppspindet_occ: Dict[OrbitalIdx, Set[int]],
        d_occupied: Dict[str, Set[OrbitalIdx]],
        d_unoccupied: Dict[str, Set[OrbitalIdx]],
    ):
        """
        If psi_i == psi_j, return indices of determinants occupied in d_occupied and empty
        in d_unoccupied.
        >>> Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy({0: {0}, 1: {0, 1}, 3: {1}}, {}, {"same": {1}}, {"same": {0}})
        {1}
        >>> Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy({0: {0}, 1: {0, 1, 2}, 3: {1}}, {1: {0}, 2: {0}, 4: {1}, 5: {1}}, {"same": {1}, "opposite": {1}}, {"same": {3}})
        {0}
        """

        return Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
            spindet_occ, oppspindet_occ, d_occupied, "all"
        ) - Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
            spindet_occ, oppspindet_occ, d_unoccupied, "any"
        )

    @staticmethod
    def do_diagonal(det_indices, psi_i, det_to_index_j, phase):
        # contribution from integrals to diagonal elements
        for a in det_indices:
            # Handle PT2 case when psi_i != psi_j. In this case, psi_i[a] won't be in the external space and so error will be thrown
            if psi_i[a] in det_to_index_j:
                yield (
                    a,
                    det_to_index_j[psi_i[a]],
                ), phase  # Yield (a, J) v. (a, a) for MPI implementation
            else:
                pass

    @staticmethod
    def do_single(det_indices_i, phasemod, occ, h, p, psi_i, det_to_index_j, spin, exci):
        # Single excitation from h to p, occ is index of orbital occupied
        # Excitation is from internal to external space
        for a in det_indices_i:  # Loop through candidate determinants in internal space
            det = psi_i[a]
            excited_spindet = exci.apply_excitation(getattr(det, spin), [[h], [p]])
            if spin == "alpha":
                excited_det = Determinant(excited_spindet, getattr(det, "beta"))
            else:
                excited_det = Determinant(getattr(det, "alpha"), excited_spindet)
            if excited_det in det_to_index_j:  # Check if excited det is in external space
                phase = phasemod * PhaseIdx.single_phase(getattr(det, spin), excited_spindet, h, p)
                yield (a, det_to_index_j[excited_det]), phase
            else:
                pass

    @staticmethod
    def do_double_samespin(
        hp1, hp2, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i, spin, exci
    ):
        # hp1 = i, j or j, i, hp2 = k, l or l, k, particle-hole pairs
        # double excitation from i to j and k to l, electrons are of the same spin
        i, k = hp1
        j, l = hp2
        det_indices_AA = Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
            spindet_occ_i, {}, {"same": {i, j}}, {"same": {k, l}}
        )
        for a in det_indices_AA:  # Loop through candidate determinants in internal space
            det = psi_i[a]
            excited_spindet = exci.apply_excitation(getattr(det, spin), [[i, j], [k, l]])
            if spin == "alpha":
                excited_det = Determinant(excited_spindet, getattr(det, "beta"))
            else:
                excited_det = Determinant(getattr(det, "alpha"), excited_spindet)
            if excited_det in det_to_index_j:  # Check if excited det is in external space
                phase = PhaseIdx.double_phase(getattr(det, spin), excited_spindet, i, j, k, l)
                yield (a, det_to_index_j[excited_det]), phase
            else:
                pass

    @staticmethod
    def do_double_oppspin(
        hp1, hp2, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i, spin, exci
    ):
        # hp1 = i, j or j, i, hp2 = k, l or l, k, particle-hole pairs
        # double excitation from i to j and k to l, electrons are of opposite spin spin
        i, k = hp1
        j, l = hp2
        det_indices_AB = Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
            spindet_occ_i,
            oppspindet_occ_i,
            {"same": {i}, "opposite": {j}},
            {"same": {k}, "opposite": {l}},
        )
        for a in det_indices_AB:  # Look through candidate determinants in internal space
            det = psi_i[a]
            excited_spindet_A = exci.apply_excitation(getattr(det, spin), [[i], [k]])
            phaseA = PhaseIdx.single_phase(getattr(det, spin), excited_spindet_A, i, k)
            if spin == "alpha":
                excited_spindet_B = exci.apply_excitation(getattr(det, "beta"), [[j], [l]])
                phaseB = PhaseIdx.single_phase(getattr(det, "beta"), excited_spindet_B, j, l)
                excited_det = Determinant(excited_spindet_A, excited_spindet_B)
            else:
                excited_spindet_B = exci.apply_excitation(getattr(det, "alpha"), [[j], [l]])
                phaseB = PhaseIdx.single_phase(getattr(det, "alpha"), excited_spindet_B, j, l)
                excited_det = Determinant(excited_spindet_B, excited_spindet_A)
            if excited_det in det_to_index_j:  # Check if excited det is in external space
                yield (a, det_to_index_j[excited_det]), phaseA * phaseB
            else:
                pass

    @staticmethod
    def category_A(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category A, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i = j = k = 1: (1,1,1,1). This category will contribute to diagonal elements of the Hamiltonian matrix, only
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "A"

        def do_diagonal_A(i, j, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i):
            # Get indices of determinants occupied in ia and ib
            det_indices = Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                spindet_occ_i, oppspindet_occ_i, {"same": {i}, "opposite": {j}}, "all"
            )

            # phase is always 1
            yield from Hamiltonian_two_electrons_integral_driven.do_diagonal(
                det_indices, psi_i, det_to_index_j, 1
            )

        yield from do_diagonal_A(i, j, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i)

    @staticmethod
    def category_B(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category B, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i = k < j = l: (1,2,1,2). This category will contribute to diagonal elements of the Hamiltonian matrix, only
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "B"

        def do_diagonal_B(i, j, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i):
            # Get indices of determinants occupied in ia and ja, jb and jb, ia and jb, and ib and ja
            det_indices = chain(
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i}, "opposite": {j}}, "all"
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i, j}}, "all"
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    oppspindet_occ_i, spindet_occ_i, {"same": {i}, "opposite": {j}}, "all"
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    oppspindet_occ_i, spindet_occ_i, {"same": {i, j}}, "all"
                ),
            )

            # phase is always 1
            yield from Hamiltonian_two_electrons_integral_driven.do_diagonal(
                det_indices, psi_i, det_to_index_j, 1
            )

        yield from do_diagonal_B(i, j, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i)

    @staticmethod
    def category_C(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, exci):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category C, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i = k < j < l: (1,2,1,3), i < k < j = l: (1,3,2,3), j < i = k < l: (2,1,2,3)
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "C"

        # Hopefully, can remove hashes (det_to_index_i,j, psi_i,j, spindets... ) and call them as properties of the class (self. ...)
        def do_single_C(
            i, j, k, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i, spin, exci
        ):
            # Get indices of determinants that are possibly related by excitations from internal --> external space
            # phasemod, occ, h, p = 1, j, i, k
            det_indices_1 = chain(
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, {}, {"same": {j, i}}, {"same": {k}}
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i}, "opposite": {j}}, {"same": {k}}
                ),
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_1, 1, j, i, k, psi_i, det_to_index_j, spin, exci
            )
            # Get indices of determinants that are possibly related by excitations from external --> internal space
            # phasemod, occ, h, p = 1, j, k, i
            det_indices_2 = chain(
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, {}, {"same": {j, k}}, {"same": {i}}
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {k}, "opposite": {j}}, {"same": {i}}
                ),
            )

            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_2, 1, j, k, i, psi_i, det_to_index_j, spin, exci
            )

        if i == k:  # <ij|il> = <ji|li>, ja(b) to la(b) where ia or ib is occupied
            yield from do_single_C(
                j,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_C(
                j,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )
        else:  # j == l, <ji|jk> = <ij|kj>, ia(b) to ka(b) where ja or jb or is occupied
            yield from do_single_C(
                i, j, k, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
            )
            yield from do_single_C(
                i,
                j,
                k,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )

    @staticmethod
    def category_D(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, exci):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category D, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i=j=k<l (1,1,1,2), i<j=k=l (1,2,2,2)
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "D"

        def do_single_D(
            i, j, l, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i, spin, exci
        ):
            # Get indices of determinants that are possibly related by excitations from external --> internal space
            # phasemod, occ, h, p = 1, i, j, l
            det_indices_1 = (
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {j}, "opposite": {i}}, {"same": {l}}
                )
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_1, 1, i, j, l, psi_i, det_to_index_j, spin, exci
            )
            # Get indices of determinants that are possibly related by excitations from external --> internal space
            # phasemod, occ, h, p = 1, i, l, j
            det_indices_2 = (
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {l}, "opposite": {i}}, {"same": {j}}
                )
            )

            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_2, 1, i, l, j, psi_i, det_to_index_j, spin, exci
            )

        if i == j:  # <ii|il>, ia(b) to la(b) while ib(a) is occupied
            yield from do_single_D(
                i,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_D(
                i,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )
        else:  # i < j == k == l, <ij|jj> = <jj|ij> = <jj|ji>, ja(b) to ia(b) where jb(a) is occupied
            yield from do_single_D(
                j,
                j,
                i,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_D(
                j,
                j,
                i,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )

    @staticmethod
    def category_E(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, exci):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category E, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i=j<k<l (1,1,2,3), i<j=k<l (1,2,2,3), i<j<k=l (1,2,3,3)
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "E"

        def do_single_E(
            i, k, l, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i, spin, exci
        ):
            # Get indices of determinants that are possibly related by excitations from external --> internal space
            # phasemod, occ, h, p = -1, i, k, l
            det_indices_1 = (
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i, k}}, {"same": {l}}
                )
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_1, -1, i, k, l, psi_i, det_to_index_j, spin, exci
            )
            # Get indices of determinants that are possibly related by excitations from external --> internal space
            # phasemod, occ, h, p = -1, i, l, k
            det_indices_2 = (
                Hamiltonian_two_electrons_integral_driven.get_dets_via_orbital_occupancy(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i, l}}, {"same": {k}}
                )
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_single(
                det_indices_2, -1, i, l, k, psi_i, det_to_index_j, spin, exci
            )

        # doubles, ia(b) to ka(b) and jb(a) to lb(a)
        for hp1, hp2 in product(permutations([i, k], 2), permutations([j, l], 2)):
            yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
                hp1, hp2, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
                hp1, hp2, psi_i, det_to_index_j, spindet_b_occ_i, spindet_a_occ_i, "beta", exci
            )

        if i == j:  # <ii|kl> = <ii|lk> = <ik|li> -> - <ik|il>
            # singles, ka(b) to la(b) where ia(b) is occupied
            yield from do_single_E(
                i,
                k,
                l,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_E(
                i,
                k,
                l,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )
        elif j == k:  # <ij|jl> = - <ij|lj>
            # singles, ia(b) to la(b) where ja(b) is occupied
            yield from do_single_E(
                j,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_E(
                j,
                i,
                l,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )
        else:  # k == l, <ij|kk> = <ji|kk> = <jk|ki> -> -<jk|ik>
            # singles, ja(b) to ia(b) where ka(b) is occupied
            yield from do_single_E(
                k,
                i,
                j,
                psi_i,
                det_to_index_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                "alpha",
                exci,
            )
            yield from do_single_E(
                k,
                i,
                j,
                psi_i,
                det_to_index_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                "beta",
                exci,
            )

    @staticmethod
    def category_F(
        idx,
        psi_i,
        det_to_index_j,
        spindet_a_occ_i,
        spindet_b_occ_i,
        exci,
    ):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category F, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i=j<k=l (1,1,2,2). Contributes to diagonal elements and doubles
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "F"

        def do_diagonal_F(i, k, psi_i, det_to_index_j, spindet_occ_i, oppspindet_occ_i):
            # should have negative phase, since <11|22> = <12|21> -> <12|12> with negative factor
            # Get indices of determinants occupied in ia, ja and jb, jb
            det_indices = chain(
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    spindet_occ_i, oppspindet_occ_i, {"same": {i, k}}, "all"
                ),
                Hamiltonian_two_electrons_integral_driven.get_dets_occ_in_orbitals(
                    oppspindet_occ_i, spindet_occ_i, {"same": {i, k}}, "all"
                ),
            )
            # phase is always -1
            yield from Hamiltonian_two_electrons_integral_driven.do_diagonal(
                det_indices, psi_i, det_to_index_j, -1
            )

        yield from do_diagonal_F(i, k, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i)

        # Only call for a single spin variable. Each excitation involves ia, ib to ka, kb. Flipping the spin just double counts it
        yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
            [i, k], [i, k], psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
        )
        # Need to do ph1, ph2 pairing twice, once for each spin. ia -> ka, kb -> ib
        yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
            [i, k], [k, i], psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
        )
        # Double from ia -> ka, kb -> ib
        yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
            [i, k], [k, i], psi_i, det_to_index_j, spindet_b_occ_i, spindet_a_occ_i, "beta", exci
        )
        # Only call for a single spin variable. Each excitation involves ia, ib to ka, kb. Flipping the spin just double counts it
        yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
            [k, i], [k, i], psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
        )

    @staticmethod
    def category_G(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, exci):
        """
        psi_i psi_j: Psi_det, lists of determinants
        idx: i,j,k,l, index of integral <ij|kl>
        For an integral i,j,k,l of category G, yield all dependent determinant pairs (I,J) and associated phase
        Possibilities are i<j<k<l (1,2,3,4), i<k<j<l (1,3,2,4), j<i<k<l (2,1,3,4)
        """
        i, j, k, l = idx
        assert integral_category(i, j, k, l) == "G"

        # doubles, i to k and j to l, same spin and opposite-spin excitations allowed
        for hp1, hp2 in product(permutations([i, k], 2), permutations([j, l], 2)):
            yield from Hamiltonian_two_electrons_integral_driven.do_double_samespin(
                hp1, hp2, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_double_samespin(
                hp1, hp2, psi_i, det_to_index_j, spindet_b_occ_i, spindet_a_occ_i, "beta", exci
            )
        # doubles, i to k and j to l, same spin and opposite-spin excitations allowed
        for hp1, hp2 in product(permutations([i, k], 2), permutations([j, l], 2)):
            yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
                hp1, hp2, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, "alpha", exci
            )
            yield from Hamiltonian_two_electrons_integral_driven.do_double_oppspin(
                hp1, hp2, psi_i, det_to_index_j, spindet_b_occ_i, spindet_a_occ_i, "beta", exci
            )

    @cached_property
    def N_orb(self):
        key = max(self.d_two_e_integral)
        return max(compound_idx4_reverse(key)) + 1

    @cached_property
    def exci(self):
        # Create single instance of excitation class to avoid doing so repeatedly in category functions
        return Excitation(self.N_orb)

    def H_indices(self, psi_i, psi_j) -> Iterator[Two_electron_integral_index_phase]:
        # Returns H_indices, and idx of associated integral
        generator = H_indices_generator(psi_i, psi_j)
        spindet_a_occ_i, spindet_b_occ_i = generator.spindet_occ_int
        det_to_index_j = generator.det_to_index_ext
        for idx4, integral_values in self.d_two_e_integral.items():
            idx = compound_idx4_reverse(idx4)
            for (
                (a, b),
                phase,
            ) in self.H_indices_idx(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i):
                yield (a, b), idx, phase

    def H_indices_idx(
        self, idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i
    ) -> Iterator[Two_electron_integral_index_phase]:
        # Call to get indices of determinant pairs + associated phase for a given integral idx
        category = integral_category(*idx)
        if category == "A":
            yield from self.category_A(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i)
        if category == "B":
            yield from self.category_B(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i)
        if category == "C":
            yield from self.category_C(
                idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, self.exci
            )
        if category == "D":
            yield from self.category_D(
                idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, self.exci
            )
        if category == "E":
            yield from self.category_E(
                idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, self.exci
            )
        if category == "F":
            yield from self.category_F(
                idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, self.exci
            )
        if category == "G":
            yield from self.category_G(
                idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, self.exci
            )

    def H(self, psi_i, psi_j) -> List[List[Energy]]:
        generator = H_indices_generator(psi_i, psi_j)
        spindet_a_occ_i, spindet_b_occ_i = generator.spindet_occ_int
        det_to_index_j = generator.det_to_index_ext
        # This is the function who will take foreever
        h = np.zeros(shape=(len(psi_i), len(psi_j)))
        for idx4, integral_values in self.d_two_e_integral.items():
            idx = compound_idx4_reverse(idx4)
            for (
                (a, b),
                phase,
            ) in self.H_indices_idx(idx, psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i):
                h[a, b] += phase * integral_values
        return h

    def H_ii(self, det_i: Determinant):
        return sum(phase * self.H_ijkl_orbital(*idx) for idx, phase in self.H_ii_indices(det_i))


@dataclass
class H_indices_generator(object):
    """Generate and cache necessary utilities for building the
    two-electron Hamiltonian in an integral-driven fashion.
    Re-created at each CIPSI iteration; i.e. for each new list of internal determinants."""

    psi_internal: Psi_det  # Internal wavefunction
    psi_external: Psi_det  # External wavefunction

    def __init__(self, psi_internal: Psi_det, psi_external: Psi_det = None):
        if psi_external is None:
            psi_external = psi_internal
        self.psi_i = psi_internal
        self.psi_j = psi_external

    @staticmethod
    def get_spindet_a_occ_spindet_b_occ(
        psi_i: Psi_det,
    ) -> Tuple[Dict[OrbitalIdx, Set[int]], Dict[OrbitalIdx, Set[int]]]:
        """
        >>> H_indices_generator.get_spindet_a_occ_spindet_b_occ([Determinant(alpha=(0,1),beta=(1,2)),Determinant(alpha=(1,3),beta=(4,5))])
        (defaultdict(<class 'set'>, {0: {0}, 1: {0, 1}, 3: {1}}),
         defaultdict(<class 'set'>, {1: {0}, 2: {0}, 4: {1}, 5: {1}}))
        >>> H_indices_generator.get_spindet_a_occ_spindet_b_occ([Determinant(alpha=(0,),beta=(0,))])[0][1]
        set()
        """
        # Can generate det_to_indices hash in here
        def get_dets_occ(psi_i: Psi_det, spin: str) -> Dict[OrbitalIdx, Set[int]]:
            ds = defaultdict(set)
            for i, det in enumerate(psi_i):
                for o in getattr(det, spin):
                    ds[o].add(i)
            return ds

        return tuple(get_dets_occ(psi_i, spin) for spin in ["alpha", "beta"])

    @cached_property
    def det_to_index_ext(self):
        # Create and cache dictionary mapping connected determinants \in psi_j to associated indices.
        return {det: i for i, det in enumerate(self.psi_j)}

    @cached_property
    def spindet_occ_int(self):
        # Create and cache dictionaries mapping spin-orbital indices (alpha) to det indices
        return self.get_spindet_a_occ_spindet_b_occ(self.psi_i)


#   _   _                 _ _ _              _
#  | | | |               (_) | |            (_)
#  | |_| | __ _ _ __ ___  _| | |_ ___  _ __  _  __ _ _ __
#  |  _  |/ _` | '_ ` _ \| | | __/ _ \| '_ \| |/ _` | '_ \
#  | | | | (_| | | | | | | | | || (_) | | | | | (_| | | | |
#  \_| |_/\__,_|_| |_| |_|_|_|\__\___/|_| |_|_|\__,_|_| |_|
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
    driven_by: str = "determinant"

    @cached_property
    def H_one_electron(self):
        return Hamiltonian_one_electron(self.d_one_e_integral, self.E0)

    @cached_property
    def H_two_electrons(self):
        if self.driven_by == "determinant":
            return Hamiltonian_two_electrons_determinant_driven(self.d_two_e_integral)
        elif self.driven_by == "integral":
            return Hamiltonian_two_electrons_integral_driven(self.d_two_e_integral)
        else:
            raise NotImplementedError

    # ~ ~ ~
    # H_ii
    # ~ ~ ~
    def H_ii(self, det_i: Determinant) -> List[Energy]:
        return self.H_one_electron.H_ii(det_i) + self.H_two_electrons.H_ii(det_i)

    # ~ ~ ~
    # H
    # ~ ~ ~
    def H(self, psi_internal: Psi_det, psi_external: Psi_det = None) -> List[List[Energy]]:
        """Return a matrix of size psi x psi_j containing the value of the Hamiltonian.
        If psi_j == None, then assume a return psi x psi hermitian Hamiltonian,
        if not not overlap exist between psi and psi_j"""
        if psi_external is None:
            psi_external = psi_internal

        return self.H_one_electron.H(psi_internal, psi_external) + self.H_two_electrons.H(
            psi_internal, psi_external
        )


#   _   _                 _ _ _              _
#  | | | |               (_) | |            (_)
#  | |_| | __ _ _ __ ___  _| | |_ ___  _ __  _  __ _ _ __
#  |  _  |/ _` | '_ ` _ \| | | __/ _ \| '_ \| |/ _` | '_ \
#  | | | | (_| | | | | | | | | || (_) | | | | | (_| | | | |
#  \_| |_/\__,_|_| |_| |_|_|_|\__\___/|_| |_|_|\__,_|_| |_|
#
#   _____                           _
#  |  __ \                         | |
#  | |  \/ ___ _ __   ___ _ __ __ _| |_ ___  _ __
#  | | __ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
#  | |_\ \  __/ | | |  __/ | | (_| | || (_) | |
#   \____/\___|_| |_|\___|_|  \__,_|\__\___/|_|
#


@dataclass
class Hamiltonian_generator(object):
    """Generator class for matrix Hamiltonian; compute matrix elements of H
    in the basis of Slater determinants in a distributed fashion.
    Each rank handles local H_i \in len(psi_local) x len(psi_internal) chunk of the full H.
    Slater-Condon rules are used to compute the matrix elements <I|H|J> where I
    and J are Slater determinants.

    Called and re-created at each CIPSI iteration; i.e. each time determinants
    are added to the internal wave-function.

    :param comm: MPI.COMM_WORLD communicator
    :param E0: Float, energy
    :param d_one_e_integral: Dictionary of one-electorn integrals
    :param d_two_e_integral: Dictionary of two-electorn integrals
    :param driven_by: generate H in a an integral/determinant-driven fashion.

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
    * Matrix elements of H are stored as a hash lookup.
    """

    # Only pass internal determinant, since we'll only want to cache the Hamiltonian matrix elts. for an iteration
    # For ex., we iterate through the (psi_int) x (psi_ext) matrix elts. once to compute the PT2 contribution
    def __init__(
        self,
        comm,
        E0: Energy,
        d_one_e_integral: One_electron_integral,
        d_two_e_integral: Two_electron_integral,
        psi_internal: Psi_det,
        driven_by="integral",
    ):
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        # Full problem size is no. of internal determinants
        self.full_problem_size = len(psi_internal)
        # At initialization, each rank computes distribution of determinants
        floor, remainder = divmod(self.full_problem_size, self.world_size)
        ceiling = floor + 1
        self.distribution = [ceiling] * remainder + [floor] * (self.world_size - remainder)
        # Local problem size (no. of local determinants)
        self.local_size = self.distribution[self.rank]
        # Compute offsets (start of the local section) for all nodes
        self.offsets = [0] + list(accumulate(self.distribution[:-1]))
        # Save lists of determinants/integral dictionaries with instance of class
        self.psi_internal = psi_internal
        self.E0 = E0
        self.d_one_e_integral = d_one_e_integral
        self.d_two_e_integral = d_two_e_integral
        self.driven_by = driven_by

    @cached_property
    def psi_local(self):
        # TODO: Right now, having each rank do this. Will re-do each iteration, but can be optimized somehow
        # Each rank computes local determinants
        return self.psi_internal[
            self.offsets[self.rank] : (self.offsets[self.rank] + self.distribution[self.rank])
        ]

    @cached_property
    def N_orb(self):
        key = max(self.d_two_e_integral)
        return max(compound_idx4_reverse(key)) + 1

    # Create instances of 1e and 2e `driver' classes
    @cached_property
    def Hamiltonian_1e_driver(self):
        return Hamiltonian_one_electron(self.d_one_e_integral, self.E0)

    @cached_property
    def Hamiltonian_2e_driver(self):
        if self.driven_by == "determinant":
            return Hamiltonian_two_electrons_determinant_driven(self.d_two_e_integral)
        elif self.driven_by == "integral":
            return Hamiltonian_two_electrons_integral_driven(self.d_two_e_integral)
        else:
            raise NotImplementedError

    # ~ ~ ~
    # H_ii
    # ~ ~ ~
    def H_ii(self, det_i: Determinant) -> Energy:
        # Diagonal elements of local Hamiltonian
        return self.Hamiltonian_1e_driver.H_ii(det_i) + self.Hamiltonian_2e_driver.H_ii(det_i)

    @cached_property
    def D_i(self):
        """Return `diagonal' of local H_i. (Diagonal meaning, entries of H_i
        corresponding to the diagonal part of H) as a numpy vector.
        Used for pre-conditioning step in Davidson's iteration."""
        D_i = np.zeros(self.local_size, dtype="float")
        # Iterate through local determinants m
        for j, det in enumerate(self.psi_local):
            D_i[j] = self.H_ii(det)
        return D_i

    # ~ ~ ~
    # H
    # ~ ~ ~
    @cached_property
    def H_i(self):
        """Build row-wise portion of Hamiltonian matrix (H_i).
        :return H_i: len(self.psi_local) \times len(self.psi_j), as numpy array"""
        H_i_1e = self.Hamiltonian_1e_driver.H(self.psi_local, self.psi_internal)
        H_i_2e = self.Hamiltonian_2e_driver.H(self.psi_local, self.psi_internal)
        return H_i_1e + H_i_2e

    @cached_property
    def H(self):
        """Build full Hamiltonian matrix, H = [H_1; ...; H_N].
        Local portions are computed by each rank, gathered and sent to all."""
        H_i = self.H_i
        # TODO: Optimize this? Mostly using for testing so not a huge deal.
        # Size of sendbuff
        sendcounts = np.array(self.local_size * self.full_problem_size, dtype="i")
        recvcounts = None
        if self.rank == self.MPI_master_rank:
            # Size of recvbuff
            recvcounts = np.zeros(self.world_size, dtype="i")
        self.comm.Gather(sendcounts, recvcounts, root=self.MPI_master_rank)
        H_full = np.zeros((self.full_problem_size, self.full_problem_size), dtype="float")
        # Master process gathers and sends full matrix Hamiltonian
        self.comm.Gatherv(H_i, (H_full, recvcounts), root=self.MPI_master_rank)
        self.comm.Bcast(np.array(H_full, dtype="float"), root=self.MPI_master_rank)

        return H_full

    @cached_property
    def H_i_1e_matrix_elements(self):
        """Generate elements of H_i_1e (local row-wise portion of one-electron Hamiltonian).
        Elements are gathered `on-the-fly' at first iteration, and then cached in a dict to be re-used later.
        """
        H_i_1e_matrix_elements = defaultdict(int)
        for I, det_I in enumerate(self.psi_local):
            for J, det_J in enumerate(self.psi_internal):
                H_i_1e_matrix_elements[(I, J)] += self.Hamiltonian_1e_driver.H_ij(det_I, det_J)
        # Remove the default dict
        return dict(H_i_1e_matrix_elements)

    @cached_property
    def H_i_2e_matrix_elements(self):
        """Generate elements of H_i_1e (local row-wise portion of one-electron Hamiltonian).
        Elements are gathered `on-the-fly' at first iteration, and then cached in a dict to be re-used later.
        Works for integral-driven or determinant-driven implementation.
        """
        H_i_2e_matrix_elements = defaultdict(int)
        for (I, J), idx, phase in self.Hamiltonian_2e_driver.H_indices(
            self.psi_local, self.psi_internal
        ):
            # Update (I, J)th 2e matrix element
            H_i_2e_matrix_elements[(I, J)] += phase * self.Hamiltonian_2e_driver.H_ijkl_orbital(
                *idx
            )
        # Remove the default dict
        return dict(H_i_2e_matrix_elements)

    def H_i_implicit_matrix_product(self, M):
        """Function to implicitly compute matrix-matrix product W_i = H_i * M
        At first call, matrix elements of H_i are built `on-the-fly'. Matrix elements are cached
        for later use, and iterated through to compute H_i * V implicitly.

        :param H_i: local (self.local_size \times n) row-wise portion of Hamiltonian (never explicitly formed)
        :param V:  (self.full_size \times k) diensional numpy array

        :return W_i: locally computed chunk of matrix-matrix product (self.local_size \times k), as a numpy array
        """

        def H_i_implicit_matrix_product_step(self, M, matrix_elements):
            # Implicitly compute the 1e/2e Hamiltonain matrix-matrix product W_i = H_i * M
            # :param matrix_elements: python dictionary of 1e or 2e Hamiltonian matrix elements
            if M.ndim == 1:  # Handle case when M is a vector
                M = M.reshape(len(M), 1)
            k = M.shape[1]  # Column dimension
            # Pre-allocate space for local brick of matrix-matrix product
            W_i = np.zeros((self.local_size, k), dtype="float")
            # Iterate through nonzero matrix elements to compute matrix-matrix product
            for (I, J), matrix_elt in matrix_elements.items():
                W_i[I, :] += matrix_elt * M[J, :]  # Update row I of W_i
            return W_i

        # On first call, these will do the same things regardless of the `driven_by' option
        # If option to cache, compute elements and store in a sparse representation, then multiply
        return H_i_implicit_matrix_product_step(
            self, M, self.H_i_1e_matrix_elements
        ) + H_i_implicit_matrix_product_step(self, M, self.H_i_2e_matrix_elements)


#  ______            _     _
#  |  _  \          (_)   | |
#  | | | |__ ___   ___  __| |___  ___  _ __
#  | | | / _` \ \ / / |/ _` / __|/ _ \| '_ \
#  | |/ / (_| |\ V /| | (_| \__ \ (_) | | | |
#  |___/ \__,_| \_/ |_|\__,_|___/\___/|_| |_|
#
#

class Davidson_manager(object):
    """A matrix-free implementation of Davidson's method in parallel.
    All matrix products involving the Hamiltonian matrix are computed implicitly, and
    matrix entries are cached if memory allows.

    References:
    * `A Parallel Davidson-Type Algorithm for Several Eigenvalues' [L. Borges, S. Oliveira, 1998]
    * `The Davidson Method' [M. Crouzeix, B. Philippe, M. Sadkane, 1994]

    Each process will have local access to instance of the `Hamiltonian_generator()' class, to construct
    the local portion Hamiltonian on the fly.
    """

    def __init__(self, comm, H_i_generator: Hamiltonian_generator):
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        # Instance of generator class for building local portion of the Hamiltonian
        self.H_i_generator = H_i_generator
        # TODO: Is there a best practice with this sort of thing? (Computing dist. of work and the like)
        # Hamiltonian_generator computes dist. of work, so pass these to this class for easier reference.
        self.full_problem_size = H_i_generator.full_problem_size
        self.distribution = H_i_generator.distribution
        self.offsets = H_i_generator.offsets
        self.local_size = H_i_generator.local_size

    def parallel_iteration_restart(self, dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik):
        """Restart Davidson's iteration; resize the trial subspace V_k
        and its associated data structures. Prevent a significant
        blow-up of column dimension.

        :param dim_S: column-size of local working variables (V_ik and friends)
        :param n_eig: number of eigenvalues to look for (minimally allowed dim_S)
        :param n_newvecs: number of new vectors added at the previous iteration
        :param X_ik: current Ritz vectors, numpy array
        :param V_ik, W_ik: local working variables, numpy arrays

        :return new values for dim_S, V_ik, and W_ik following implicit restart
        """
        if n_newvecs == 0:
            V_inew = X_ik[:, :n_eig]
        else:
            V_inew = np.c_[X_ik[:, :n_eig], V_ik[:, -n_newvecs:]]
        # Take leading n_eig Ritz vectors as new guess vectors
        dim_S = V_inew.shape[1]  # New subspace dimension
        V_ik = np.zeros((self.local_size, 0), dtype="float")
        W_ik = np.zeros((self.local_size, 0), dtype="float")
        # Initialize; normalize first basis vector
        v_inew = np.array(V_inew[:, 0], dtype="float")
        v_new = np.zeros(self.full_problem_size, dtype="float")
        self.comm.Allgatherv(
            [v_inew, MPI.DOUBLE], [v_new, self.distribution, self.offsets, MPI.DOUBLE]
        )
        V_ik = np.c_[V_ik, v_inew / np.linalg.norm(v_new)]
        for j in range(1, dim_S):
            # Orthogonalize next vector against previous ones in restart basis
            v_inew, norm_vnew = self.mgs(V_ik[:, :j], V_inew[:, j])
            V_ik = np.c_[V_ik, v_inew]  # Update basis
        n_newvecs = dim_S
        return dim_S, n_newvecs, V_ik, W_ik

    def mgs(self, V_ik, t_ik):
        """Parallel implementation of Modified Graham-Schmidt (MGS).
        Takes piece of new guess vector t_ik, and orthogonalizes it against trial subspace V_k.

        :param V_ik: local work variable, numpy matrix
        :param t_ik: local work variable, numpy vector

        :return orthonormalized vector t_ik
        """
        for j in range(V_ik.shape[1]):  # Iterate through k basis vectors
            c_ij = np.copy(
                np.inner(V_ik[:, j], t_ik)
            )  # Each process computes partial inner-product
            c_j = np.zeros(1, dtype="float")  # Pre-allocate
            self.comm.Allreduce([c_ij, MPI.DOUBLE], [c_j, MPI.DOUBLE])  # Default op=SUM
            t_ik = t_ik - c_j * V_ik[:, j]  # Remove component of t_ik in V_ik
        t_k = np.zeros(
            self.full_problem_size, dtype="float"
        )  # Pre-allocate space to receive new guess vector
        self.comm.Allgatherv([t_ik, MPI.DOUBLE], [t_k, self.distribution, self.offsets, MPI.DOUBLE])
        norm_tk = np.linalg.norm(t_k)
        return t_ik / norm_tk, norm_tk  # Return new orthonormalized vector

    def preconditioning(self, D_i, l_k, r_ik):
        """Preconditon next guess vector

        :param D_i: diagonal portion of local Hamiltonian, as a numpy vector
        :param l_k: an eigenvalue, as a scalar
        :param r_ik: residual, a numpy vector

        :return numpy vector
        """
        # Build diagonal preconditioner
        M_k = np.diag(np.clip(np.reciprocal(D_i - l_k), a_min=-1e5, a_max=1e5))
        return np.dot(M_k, r_ik)

    def print_master(self, str_):
        """Master rank prints inputted str"""
        if self.rank == self.MPI_master_rank:
            print(str_)

    def initial_guess_vectors(self, n, dim_S):
        """Generate standard initial guess vectors for Davidson's iteration.
        Locally distributed canonical basis vectors

        :param n: full problem size
        :param dim_S: initial subspace dimension
        """
        I = np.eye(n, dim_S)
        V_iguess = I[
            self.offsets[self.rank] : (self.offsets[self.rank] + self.distribution[self.rank]), :
        ]

        return V_iguess

    def distributed_davidson(
        self, V_iguess=None, n_eig=1, conv_tol=1e-8, subspace_tol=1e-10, max_iter=1000, m=1, q=100
    ):
        """Davidson's method implemented in parallel. The Hamiltonian
        matrix is distrubted row-wise across MPI rank.
        Finds the n_eig smallest eigenvalues of a symmetric Hamiltonian.

        :param H: self.full_problem_size \times self.full_problem_size symmetric Hamiltonian
        :param H_i: self.local_size \times self.full_problem_size `short and fat' locally distributed H
        :param V_iguess: self.local_size \times dim_S initial guess vectors
        :param n_eig: number of desired eigenpairs
        :param conv_tol: convergence tolerance
        :param subspace_tol: tolerance for adding new basis vectors to trial subspace (avoids ill-conditioning)
        :param max_iter: max no. of iterations for Davidson to run
        :param m: minimal subspace dimension, m <= dim_S
        :param q: memory footprint tuning, q is maximally allowed subspace dimension

        :return a list of `n_eig` eigenvalues/associated eigenvectors, as numpy vector/array resp.
        """
        # Initialization steps
        n = self.full_problem_size  # Save full problem size
        # Establish local vars: trial subspace (V_ik) and action of H_i on full V_k (W_ik = H_i * V_k)
        V_ik = np.zeros((self.local_size, 0), dtype="float")
        W_ik = np.zeros((self.local_size, 0), dtype="float")
        # Set initial guess vectors and minimal initial subspace dimension
        dim_S = min(m, n)
        assert m >= n_eig  # No. of initial guess vectors must be >= no. of desired energy values
        # Set initial guess vectors TODO: For now, have some default guess vectors for testing
        if V_iguess is None:
            V_iguess = self.initial_guess_vectors(n, dim_S)
        else:  # Else, check dimensions of initial guess vectors align with other inputs
            assert (n, m) == V_iguess.shape
        V_ik = np.c_[V_ik, V_iguess]
        # Build `diagonal` of local Hamiltonian
        D_i = self.H_i_generator.D_i

        n_newvecs = dim_S  # No. of vectors added is initial subspace dimension
        restart = True
        for k in range(1, max_iter):
            self.print_master(
                f"Process rank: {self.rank}, Iterate: {k}, Subspace dimension: {dim_S}"
            )
            # Gather full trial vectors added during previous iteration on each rank
            V_new = np.zeros((n, n_newvecs), dtype="float")
            V_inew = np.array(V_ik[:, -n_newvecs:], dtype="float")
            self.comm.Allgatherv(
                [V_inew, MPI.DOUBLE],
                [
                    V_new,
                    n_newvecs * np.array(self.distribution),
                    n_newvecs * np.array(self.offsets),
                    MPI.DOUBLE,
                ],
            )
            # Compute new columns of W_ik, W_inew = H_i * V_new
            # TODO: Some maximal allowed dimension before we switch to on the fly?
            W_inew = self.H_i_generator.H_i_implicit_matrix_product(V_new)  # Default is to cache
            W_ik = np.c_[W_ik, W_inew]

            # Each rank computes partial update to the projected Hamiltonian S_k
            if restart:  # If True, need to compute full S_k explicitly
                S_ik = np.dot(V_ik.T, W_ik)
                restart = False
            else:  # Else, append new rows & columns
                S_inew_c = np.dot(V_ik[:, :-n_newvecs].T, W_inew)
                S_ik = np.c_[S_ik, S_inew_c]
                S_inew_r = np.c_[np.dot(V_inew.T, W_ik[:, :-n_newvecs]), np.dot(V_inew.T, W_inew)]
                S_ik = np.r_[S_ik, S_inew_r]
            # Reduce contributions and form new S_k
            S_k = np.zeros((dim_S, dim_S), dtype="float")
            self.comm.Allreduce([S_ik, MPI.DOUBLE], [S_k, MPI.DOUBLE])
            # All ranks diagonalize S_k (dim_S kept small, so inexpensive)
            L_k, Y_k = np.linalg.eigh(S_k)  # TODO: Some check that this is symmetric?
            L_k, Y_k = L_k[:n_eig], Y_k[:, :n_eig]

            n_newvecs = 0  # Initialize counter; no. of new vectors added to trial subspace
            X_ik = np.dot(V_ik, Y_k)  # Pre-compute Ritz vectors (V_ik updated each iteration)
            # Each rank computes local portion of residuals simultaneously
            R_i = np.dot(W_ik, Y_k) - np.dot(X_ik, np.diag(L_k))
            R = np.zeros((n, n_eig), dtype="float")  # Pre-allocate space for residuals
            self.comm.Allgatherv(
                [R_i, MPI.DOUBLE],
                [
                    R,
                    n_eig * np.array(self.distribution),
                    n_eig * np.array(self.offsets),
                    MPI.DOUBLE,
                ],
            )  # Gather full residuals on each rank to compute norm
            # Track converged eigenpairs; True if R[:, j] < eps -> jth pair has converged
            converged, working_indices = [], []
            for j in range(n_eig):
                res = np.linalg.norm(R[:, j])
                self.print_master(f"||r_j||: {res}")
                converged.append(res < conv_tol)
                # If jth eigenpair not converged, add to list of working indices
                if not (res < conv_tol):
                    working_indices.append(j)
                else:
                    self.print_master(
                        f"Eigenvalue {j}: {L_k[j]} converged, no new trial vector added"
                    )

            if all(converged):  # Convergence check
                self.print_master("All eigenvalues converged, exiting iteration")
                break
            for j in working_indices:  # Iterate through non-converged eigenpairs
                self.print_master(
                    f"Eigenvalue {j}: not converged, preconditioning next trial vector"
                )
                # Precondition next trial vector
                t_ik = self.preconditioning(D_i, L_k[j], R_i[:, j])
                # Orthogonalize new trial vector against previous basis vectors via parallel-MGS
                t_k = np.zeros(n, dtype="float")
                self.comm.Allgatherv(
                    [np.array(t_ik, dtype="float"), MPI.DOUBLE],
                    [t_k, self.distribution, self.offsets, MPI.DOUBLE],
                )
                t_ik = t_ik / np.linalg.norm(t_k)
                t_ik, norm_tk = self.mgs(V_ik, t_ik)
                # If new trial vector is `small`, ignore. Avoids ill-conditioning
                if norm_tk > subspace_tol:
                    V_ik = np.c_[V_ik, t_ik]  # Append new vector to trial subspace
                    n_newvecs += 1

            dim_S = V_ik.shape[1]  # Update dimension of trial subspace

            if q <= dim_S:  # Collapose trial basis
                self.print_master(f"q <= dim_S: {dim_S}, restarting Davidson's")
                dim_S, n_newvecs, V_ik, W_ik = self.parallel_iteration_restart(
                    dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik
                )
                restart = True  # Indicate restart

            elif n_newvecs == 0:
                self.print_master(
                    "No new vectors added at previous iteration, restarting Davidson's"
                )
                dim_S, n_newvecs, V_ik, W_ik = self.parallel_iteration_restart(
                    dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik
                )
                restart = True  # Indicate restart

        else:
            raise NotImplementedError("Davidson not converged")

        m = X_ik.shape[1]  # Same across all ranks
        X_k = np.zeros((n, m), dtype="float")
        # Gather Ritz vectors on all ranks
        self.comm.Allgatherv(
            [X_ik, MPI.DOUBLE],
            [X_k, m * np.array(self.distribution), m * np.array(self.offsets), MPI.DOUBLE],
        )

        return L_k, X_k


#  _                  _
# |_) _        _  ._ |_) |  _. ._ _|_
# |  (_) \/\/ (/_ |  |   | (_| | | |_
#


class Powerplant_manager(object):
    """Class to compute all Energy associated with psi_internal (Psi_det);
    E denotes the variational energy <psi_det|H|psi_det>.
    E_PT2 denotes the PT2 contribution for the connected determinants."""

    # Generator class for current basis of determinants
    # Each rank has instance of this corresponding to locally stored dets psi_local \subset psi_internal
    def __init__(self, comm, H_i_generator: Hamiltonian_generator):
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        self.H_i_generator = H_i_generator
        # TODO: Is there a best practice with this sort of thing? (Computing dist. of work and the like)
        # Hamiltonian_generator computes dist. of work, so pass these to this class for easier reference.
        self.full_problem_size = H_i_generator.full_problem_size
        self.distribution = H_i_generator.distribution
        self.offsets = H_i_generator.offsets
        self.local_size = H_i_generator.local_size
        self.psi_local = self.H_i_generator.psi_local

    @cached_property
    def DM(self):
        # Instance of Davidson_manager() class for diagonalizing the Hamiltonian
        return Davidson_manager(self.comm, self.H_i_generator)

    @property
    def E_and_psi_coef(self) -> Tuple[Energy, Psi_coef]:
        """Diagonalize Hamiltonian in // and return ground state energy (new E) and corresponding eigenvector (new psi_coef)
        Done per CIPSI iteration"""
        try:
            energies, coeffs = self.DM.distributed_davidson()
        except NotImplementedError:
            print("Davidson Failed, fallback to numpy eigh")
            psi_H_psi = self.lewis.H  # Build full Hamiltonian
            energies, coeffs = np.linalg.eigh(psi_H_psi)

        E, psi_coef = energies[0], coeffs[:, 0]
        return E, psi_coef

    def E(self, psi_coef: Psi_coef) -> Energy:
        """Compute the variatonal energy associated with psi_det

        :param psi_coef: list of determinant coefficients in expansion of trial WF, as python list

        We assume the wavefunction is normalized; np.linalg.norm(c) = 1.
        Each rank will necessarily have access to the full list of determinant coefficients
        Vector * Vector.T * Matrix"""
        c = np.array(psi_coef, dtype="float")  # Coef. vector as np array
        # Compute local portion of matrix * vector product H_i * |psi_det>
        H_i_psi_det = self.H_i_generator.H_i_implicit_matrix_product(c)
        # Each rank computes portion of Vector.T * Vector inner-product (E_i, portion of variational energy)
        # Get coeffs. of local determinants
        c_i = c[self.offsets[self.rank] : (self.offsets[self.rank] + self.distribution[self.rank])]
        E_i = np.copy(np.dot(c_i.T, H_i_psi_det))
        E = np.zeros(1, dtype="float")  # Pre-allocate
        self.comm.Allreduce(
            [E_i, MPI.DOUBLE], [E, MPI.DOUBLE]
        )  # Default op=SUM, reduce contributions
        # All ranks return varitonal energy
        # TODO: Fix this if there's a more efficient way to convert to a float
        return E.item()

    @cached_property
    def psi_external(self):
        # Generate all external (connected) determinants for current CIPSI iteration
        return Excitation(self.H_i_generator.N_orb).gen_all_connected_determinant(
            self.H_i_generator.psi_internal
        )

    @cached_property
    def psi_external_local(self):
        # While we're assuming we can store the external determinants (bad assumption)
        # Let's distribute them, to make some of the computations better

        external_size = len(self.psi_external)  # Size of external space
        # Distribute connected determinants
        floor, remainder = divmod(external_size, self.world_size)
        ceiling = floor + 1
        # Save these distributions + offsets for some MPI communication later
        self.external_distribution = [ceiling] * remainder + [floor] * (self.world_size - remainder)
        # Compute offsets (start of the local section) for all nodes
        self.external_offsets = [0] + list(accumulate(self.external_distribution[:-1]))
        # Distribute external dets
        return self.psi_external[
            self.external_offsets[self.rank] : (
                self.external_offsets[self.rank] + self.external_distribution[self.rank]
            )
        ]

    def psi_external_pt2(self, psi_coef: Psi_coef) -> Tuple[Psi_det, List[Energy]]:
        # Compute the pt2 contrution of all the external (aka connected) determinant.
        #   eα=⟨Ψ(n)∣H∣∣α⟩^2 / ( E(n)−⟨α∣H∣∣α⟩ )

        # Compute len(psi_local) \times len(psi_external) `Hamiltonian'
        c = np.array(psi_coef, dtype="float")  # Coef. vector as np array
        # Coeffs. of local determinants
        c_i = c[self.offsets[self.rank] : (self.offsets[self.rank] + self.distribution[self.rank])]
        # Pre-allocate space for PT2 contributions
        E_pt2_conts = np.zeros(len(self.psi_external), dtype="float")
        # Two-electron matrix elements
        for (I, J), idx, phase in self.H_i_generator.Hamiltonian_2e_driver.H_indices(
            self.psi_local, self.psi_external
        ):
            E_pt2_conts[J] += (
                c_i[I] * phase * self.H_i_generator.Hamiltonian_2e_driver.H_ijkl_orbital(*idx)
            )
        # One-electron matrix elements
        for I, det_I in enumerate(self.psi_local):
            for J, det_J in enumerate(self.psi_external):
                E_pt2_conts[J] += c_i[I] * self.H_i_generator.Hamiltonian_1e_driver.H_ij(
                    det_I, det_J
                )
        E_pt2_conts = np.array(E_pt2_conts, dtype="float")
        E_pt2 = np.zeros(len(self.psi_external), dtype="float")
        self.comm.Allreduce(
            [E_pt2_conts, MPI.DOUBLE], [E_pt2, MPI.DOUBLE]
        )  # Default op=SUM, reduce contributions
        nominator = E_pt2
        # Distribute this computation and Gather
        denominator_conts = np.divide(
            1.0,
            self.E(psi_coef)
            - np.array([self.H_i_generator.H_ii(det) for det in self.psi_external_local]),
        )
        denominator = np.zeros(len(self.psi_external), dtype="float")
        self.comm.Allgatherv(
            [denominator_conts, MPI.DOUBLE],
            [denominator, self.external_distribution, self.external_offsets, MPI.DOUBLE],
        )

        return np.einsum(
            "i,i,i -> i", nominator, nominator, denominator
        )  # vector * vector * vector -> scalar

    def E_pt2(self, psi_coef: Psi_coef) -> Energy:
        # The sum of the pt2 contribution of each external determinant
        psi_external_energy = self.psi_external_pt2(psi_coef)

        return sum(psi_external_energy)


#  __
# (_   _  |  _   _ _|_ o  _  ._
# __) (/_ | (/_ (_  |_ | (_) | |
#
def selection_step(
    comm, lewis: Hamiltonian_generator, n_ord, psi_coef: Psi_coef, psi_det: Psi_det, n
) -> Tuple[Energy, Psi_coef, Psi_det]:
    # 1. Generate a list of all the external determinant and their pt2 contribution
    # 2. Take the n  determinants who have the biggest contribution and add it the wave function psi
    # 3. Diagonalize H corresponding to this new wave function to get the new variational energy, and new psi_coef.

    # In the main code:
    # -> Go to 1., stop when E_pt2 < Threshold || N < Threshold
    # See example of chained call to this function in `test_f2_631g_1p5p5det`
    # 1.
    psi_external_energy = Powerplant_manager(comm, lewis).psi_external_pt2(psi_coef)

    # 2.
    idx = np.argpartition(psi_external_energy, n)[:n]
    psi_external_det = Powerplant_manager(comm, lewis).psi_external
    psi_det_extented = psi_det + [psi_external_det[i] for i in idx]

    # 3.
    # New instance of Davidson manager class for the extended wavefunction
    lewis_new = Hamiltonian_generator(
        comm,
        lewis.E0,
        lewis.d_one_e_integral,
        lewis.d_two_e_integral,
        psi_det_extented,
    )  # Can optimize to only do this once

    return (*Powerplant_manager(comm, lewis_new).E_and_psi_coef, psi_det_extented)
