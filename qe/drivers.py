# Types
# -----
from typing import Tuple, Dict, NewType, NamedTuple, List, Set, Iterator, NewType
from dataclasses import dataclass

# Yes, I like itertools
from itertools import chain, product, combinations, takewhile, permutations, accumulate
from functools import partial, cached_property, cache
from collections import defaultdict
import numpy as np
from math import sqrt
import random

# Import mpi4py and utilities
import mpi4py
from mpi4py import MPI  # Note this initializes and finalizes MPI session automatically
import sys

# Orbital index (0,1,2,...,n_orb-1)
OrbitalIdx = NewType("OrbitalIdx", int)
# Two-electron integral :
# $<ij|kl> = \int \int \phi_i(r_1) \phi_j(r_2) \frac{1}{|r_1 - r_2|} \phi_k(r_1) \phi_l(r_2) dr_1 dr_2$
Two_electron_integral_index = Tuple[OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]
Two_electron_integral = Dict[Two_electron_integral_index, float]

Two_electron_integral_index_phase = Tuple[Two_electron_integral_index, bool]

# One-electron integral :
# $<i|h|k> = \int \phi_i(r) (-\frac{1}{2} \Delta + V_en ) \phi_k(r) dr$
One_electron_integral = Dict[Tuple[OrbitalIdx, OrbitalIdx], float]
Spin_determinant = Tuple[OrbitalIdx, ...]


class Determinant(NamedTuple):
    """Slater determinant: Product of 2 determinants.
    One for $\alpha$ electrons and one for \beta electrons."""

    alpha: Spin_determinant
    beta: Spin_determinant


Psi_det = List[Determinant]
Psi_coef = List[float]
# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det
Energy = NewType("Energy", float)

# _____          _           _               _   _ _   _ _
# |_   _|        | |         (_)             | | | | | (_) |
#  | | _ __   __| | _____  ___ _ __   __ _  | | | | |_ _| |___
#  | || '_ \ / _` |/ _ \ \/ / | '_ \ / _` | | | | | __| | / __|
# _| || | | | (_| |  __/>  <| | | | | (_| | | |_| | |_| | \__ \
# \___/_| |_|\__,_|\___/_/\_\_|_| |_|\__, |  \___/ \__|_|_|___/
#                                     __/ |
#                                    |___/


@cache
def compound_idx2(i, j):
    """
    get compound (triangular) index from (i,j)

    (first few elements of lower triangle shown below)
          j
        │ 0   1   2   3
     ───┼───────────────
    i 0 │ 0
      1 │ 1   2
      2 │ 3   4   5
      3 │ 6   7   8   9

    position of i,j in flattened triangle

    >>> compound_idx2(0,0)
    0
    >>> compound_idx2(0,1)
    1
    >>> compound_idx2(1,0)
    1
    >>> compound_idx2(1,1)
    2
    >>> compound_idx2(1,2)
    4
    >>> compound_idx2(2,1)
    4
    """
    p, q = min(i, j), max(i, j)
    return (q * (q + 1)) // 2 + p


@cache
def compound_idx4(i, j, k, l):
    """
    nested calls to compound_idx2
    >>> compound_idx4(0,0,0,0)
    0
    >>> compound_idx4(0,1,0,0)
    1
    >>> compound_idx4(1,1,0,0)
    2
    >>> compound_idx4(1,0,1,0)
    3
    >>> compound_idx4(1,0,1,1)
    4
    """
    return compound_idx2(compound_idx2(i, k), compound_idx2(j, l))


@cache
def compound_idx2_reverse(ij):
    """
    inverse of compound_idx2
    returns (i, j) with i <= j
    >>> compound_idx2_reverse(0)
    (0, 0)
    >>> compound_idx2_reverse(1)
    (0, 1)
    >>> compound_idx2_reverse(2)
    (1, 1)
    >>> compound_idx2_reverse(3)
    (0, 2)
    """
    j = int((sqrt(1 + 8 * ij) - 1) / 2)
    i = ij - (j * (j + 1) // 2)
    return i, j


def compound_idx4_reverse(ijkl):
    """
    inverse of compound_idx4
    returns (i, j, k, l) with ik <= jl, i <= k, and j <= l (i.e. canonical ordering)
    where ik == compound_idx2(i, k) and jl == compound_idx2(j, l)
    >>> compound_idx4_reverse(0)
    (0, 0, 0, 0)
    >>> compound_idx4_reverse(1)
    (0, 0, 0, 1)
    >>> compound_idx4_reverse(2)
    (0, 0, 1, 1)
    >>> compound_idx4_reverse(3)
    (0, 1, 0, 1)
    >>> compound_idx4_reverse(37)
    (0, 2, 1, 3)
    """
    ik, jl = compound_idx2_reverse(ijkl)
    i, k = compound_idx2_reverse(ik)
    j, l = compound_idx2_reverse(jl)
    return i, j, k, l


@cache
def compound_idx4_reverse_all(ijkl):
    """
    return all 8 permutations that are equivalent for real orbitals
    returns 8 4-tuples, even when there are duplicates
    for complex orbitals, they are ordered as:
    v, v, v*, v*, u, u, u*, u*
    where v == <ij|kl>, u == <ij|lk>, and * denotes the complex conjugate
    >>> compound_idx4_reverse_all(0)
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    >>> compound_idx4_reverse_all(1)
    ((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
    >>> compound_idx4_reverse_all(37)
    ((0, 2, 1, 3), (2, 0, 3, 1), (1, 3, 0, 2), (3, 1, 2, 0), (0, 3, 1, 2), (3, 0, 2, 1), (1, 2, 0, 3), (2, 1, 3, 0))
    """
    i, j, k, l = compound_idx4_reverse(ijkl)
    return (
        (i, j, k, l),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, j, i),
        (i, l, k, j),
        (l, i, j, k),
        (k, j, i, l),
        (j, k, l, i),
    )


@cache
def compound_idx4_reverse_all_unique(ijkl):
    """
    return only the unique 4-tuples from compound_idx4_reverse_all
    """
    return tuple(set(compound_idx4_reverse_all(ijkl)))


def canonical_idx4(i, j, k, l):
    """
    for real orbitals, return same 4-tuple for all equivalent integrals
    returned (i,j,k,l) should satisfy the following:
        i <= k
        j <= l
        (k < l) or (k==l and i <= j)
    the last of these is equivalent to (compound_idx2(i,k) <= compound_idx2(j,l))
    >>> canonical_idx4(1, 0, 0, 0)
    (0, 0, 0, 1)
    >>> canonical_idx4(4, 2, 3, 1)
    (1, 3, 2, 4)
    >>> canonical_idx4(3, 2, 1, 4)
    (1, 2, 3, 4)
    >>> canonical_idx4(1, 3, 4, 2)
    (2, 1, 3, 4)
    """
    i, k = min(i, k), max(i, k)
    ik = compound_idx2(i, k)
    j, l = min(j, l), max(j, l)
    jl = compound_idx2(j, l)
    if ik <= jl:
        return i, j, k, l
    else:
        return j, i, l, k


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


#   _____      _ _   _       _ _          _   _
#  |_   _|    (_) | (_)     | (_)        | | (_)
#    | | _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
#    | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
#   _| || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#   \___/_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|

# ~
# Integrals of the Hamiltonian over molecular orbitals
# ~
def load_integrals(
    fcidump_path,
) -> Tuple[int, float, One_electron_integral, Two_electron_integral]:
    """Read all the Hamiltonian integrals from the data file.
    Returns: (E0, d_one_e_integral, d_two_e_integral).
    E0 : a float containing the nuclear repulsion energy (V_nn),
    d_one_e_integral : a dictionary of one-electron integrals,
    d_two_e_integral : a dictionary of two-electron integrals.
    """
    import glob

    if len(glob.glob(fcidump_path)) == 1:
        fcidump_path = glob.glob(fcidump_path)[0]
    elif len(glob.glob(fcidump_path)) == 0:
        print("no matching fcidump file")
    else:
        print(f"multiple matches for {fcidump_path}")
        for i in glob.glob(fcidump_path):
            print(i)

    # Use an iterator to avoid storing everything in memory twice.
    if fcidump_path.split(".")[-1] == "gz":
        import gzip

        f = gzip.open(fcidump_path)
    elif fcidump_path.split(".")[-1] == "bz2":
        import bz2

        f = bz2.open(fcidump_path)
    else:
        f = open(fcidump_path)

    # Only non-zero integrals are stored in the fci_dump.
    # Hence we use a defaultdict to handle the sparsity
    n_orb = int(next(f).split()[2])

    for _ in range(3):
        next(f)

    d_one_e_integral = defaultdict(int)
    d_two_e_integral = defaultdict(int)

    for line in f:
        v, *l = line.split()
        v = float(v)
        # Transform from Mulliken (ik|jl) to Dirac's <ij|kl> notation
        # (swap indices)
        i, k, j, l = list(map(int, l))

        if i == 0:
            E0 = v
        elif j == 0:
            # One-electron integrals are symmetric (when real, not complex)
            d_one_e_integral[
                (i - 1, k - 1)
            ] = v  # index minus one to be consistent with determinant orbital indexing starting at zero
            d_one_e_integral[(k - 1, i - 1)] = v
        else:
            # Two-electron integrals have many permutation symmetries:
            # Exchange r1 and r2 (indices i,k and j,l)
            # Exchange i,k
            # Exchange j,l
            key = compound_idx4(i - 1, j - 1, k - 1, l - 1)
            d_two_e_integral[key] = v

    f.close()

    return n_orb, E0, d_one_e_integral, d_two_e_integral


def load_wf(path_wf) -> Tuple[List[float], List[Determinant]]:
    """Read the input file :
    Representation of the Slater determinants (basis) and
    vector of coefficients in this basis (wave function)."""

    import glob

    if len(glob.glob(path_wf)) == 1:
        path_wf = glob.glob(path_wf)[0]
    elif len(glob.glob(path_wf)) == 0:
        print(f"no matching wf file: {path_wf}")
    else:
        print(f"multiple matches for {path_wf}")
        for i in glob.glob(path_wf):
            print(i)

    if path_wf.split(".")[-1] == "gz":
        import gzip

        with gzip.open(path_wf) as f:
            data = f.read().decode().split()
    elif path_wf.split(".")[-1] == "bz2":
        import bz2

        with bz2.open(path_wf) as f:
            data = f.read().decode().split()
    else:
        with open(path_wf) as f:
            data = f.read().split()

    def decode_det(str_):
        for i, v in enumerate(str_):
            if v == "+":
                yield i

    def grouper(iterable, n):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return zip(*args)

    det = []
    psi_coef = []
    for (coef, det_i, det_j) in grouper(data, 3):
        psi_coef.append(float(coef))
        det.append(Determinant(tuple(decode_det(det_i)), tuple(decode_det(det_j))))

    # Normalize psi_coef

    norm = sqrt(sum(c * c for c in psi_coef))
    psi_coef = [c / norm for c in psi_coef]

    return psi_coef, det


def load_eref(path_ref) -> Energy:
    """Read the input file :
    Representation of the Slater determinants (basis) and
    vector of coefficients in this basis (wave function)."""

    import glob

    if len(glob.glob(path_ref)) == 1:
        path_ref = glob.glob(path_ref)[0]
    elif len(glob.glob(path_ref)) == 0:
        print(f"no matching ref file: {path_ref}")
    else:
        print(f"multiple matches for {path_ref}")
        for i in glob.glob(path_ref):
            print(i)

    if path_ref.split(".")[-1] == "gz":
        import gzip

        with gzip.open(path_ref) as f:
            data = f.read().decode()
    elif path_ref.split(".")[-1] == "bz2":
        import bz2

        with bz2.open(path_ref) as f:
            data = f.read().decode()
    else:
        with open(path_ref) as f:
            data = f.read()

    import re

    return float(re.search(r"E +=.+", data).group(0).strip().split()[-1])


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
            try:
                J = det_to_index_j[psi_i[a]]
            except KeyError:
                pass
            else:
                yield (a, J), phase  # Yield (a, J) v. (a, a) for MPI implementation

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
            try:  # Check if excited det is in external space
                J = det_to_index_j[excited_det]
            except KeyError:
                pass
            else:
                phase = phasemod * PhaseIdx.single_phase(getattr(det, spin), excited_spindet, h, p)
                yield (a, J), phase

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
            try:  # Check if excited det is in external space
                J = det_to_index_j[excited_det]
            except KeyError:
                pass
            else:
                phase = PhaseIdx.double_phase(getattr(det, spin), excited_spindet, i, j, k, l)
                yield (a, J), phase

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
            try:  # Check if excited det is in external space
                J = det_to_index_j[excited_det]
            except KeyError:
                pass
            else:
                yield (a, J), phaseA * phaseB

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
            phase = 1
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

    @staticmethod
    def get_spindet_a_occ_spindet_b_occ(
        psi_i: Psi_det,
    ) -> Tuple[Dict[OrbitalIdx, Set[int]], Dict[OrbitalIdx, Set[int]]]:
        """
        maybe use dict with spin as key instead of tuple?
        >>> Hamiltonian_two_electrons_integral_driven.get_spindet_a_occ_spindet_b_occ([Determinant(alpha=(0,1),beta=(1,2)),Determinant(alpha=(1,3),beta=(4,5))])
        (defaultdict(<class 'set'>, {0: {0}, 1: {0, 1}, 3: {1}}),
         defaultdict(<class 'set'>, {1: {0}, 2: {0}, 4: {1}, 5: {1}}))
        >>> Hamiltonian_two_electrons_integral_driven.get_spindet_a_occ_spindet_b_occ([Determinant(alpha=(0,),beta=(0,))])[0][1]
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
    Re-created at each CIPSI iteration; i.e. for each new list of interanl determinants."""

    d_two_e_integral: Two_electron_integral
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
        maybe use dict with spin as key instead of tuple?
        >>> Hamiltonian_two_electrons_integral_driven.get_spindet_a_occ_spindet_b_occ([Determinant(alpha=(0,1),beta=(1,2)),Determinant(alpha=(1,3),beta=(4,5))])
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


#  _                  _
# |_) _        _  ._ |_) |  _. ._ _|_
# |  (_) \/\/ (/_ |  |   | (_| | | |_
#


def davidson(H, n_eig=1, n_guess=1, eps=1e-7, max_iter=1000, q=1000):
    """Davidson's method.
    Finds the n_eig smallest eigenvalues of a symmetric Hamiltonian.

    :param H: self.full_size \times self.full_size symmetric Hamiltonian
    :param H_i: self.local_size \times self.full_size `short and fat' locally distributed H
    :param n_guess: number of initial guess vectors per desired eigenvalue
    :param n_eig: number of eigenvalues to find
    :param eps: convergence tolerance
    :param max_iter: max no. of iterations for Davidson to run
    :param q: memory footprint tuning, q is maximally allowed subspace dimension

    :return a list of `n_eig` eigenvalues/associated eigenvectors, as numpy vector/array resp.
    """
    n = H.shape[0]  # Get problem dimension
    if n == 1:  # If Hamiltonian is one-dimensional
        return np.array([H[0][0]]), np.array([[1]])
    # Establish working vars: trial subspace (V_k) and action of H on V_k (W_k = H * V_k)
    V_k = np.zeros((n, 0), dtype="float")
    W_k = np.zeros((n, 0), dtype="float")
    D = np.diag(H)  # Save diagonal part of H for later use
    # Set initial guess vectors and subspace dimension
    dim_S = min(n_eig * n_guess, n)
    V_guess = np.eye(n, dim_S)
    V_k = np.c_[V_k, V_guess]

    n_newvecs = dim_S  # No. of vectors added is initial subspace dimension
    restart = False
    for k in range(1, max_iter):
        # Get new trial vectors and compute new columns of W_k
        V_new = np.array(V_k[:, -n_newvecs:], dtype="float")
        W_new = np.dot(H, V_new)
        W_k = np.c_[W_k, W_new]
        # Rayleigh-Ritz; Update projected Hamiltonian
        if (k == 1) or (
            restart
        ):  # If first iterate (or following a restart), need to compute full S_k explicitly
            S_k = np.dot(V_k.T, W_k)
            restart = False
        else:  # Else, append new rows & columns
            S_new_c = np.dot(V_k[:, :-n_newvecs].T, W_new)
            S_k = np.c_[S_k, S_new_c]
            S_new_r = np.c_[np.dot(V_new.T, W_k[:, :-n_newvecs]), np.dot(V_new.T, W_new)]
            S_k = np.r_[S_k, S_new_r]
        # Diagonalize S_k and keep n_eig smallest estimates
        L_k, Y_k = np.linalg.eigh(S_k)
        L_k = L_k[:n_eig]
        Y_k = Y_k[:, :n_eig]

        n_newvecs = 0  # Initialize counter; no. of new vectors added to trial subspace
        X_k = np.dot(V_k, Y_k)  # Pre-compute Ritz vectors (V_k updated each iteration)
        converged = True  # Reset
        for j in range(n_eig):
            # Grab jth eigenpair, each rank computes local portion of jth residual vector
            l_j, y_j = L_k[j], Y_k[:, j]
            r_j = np.dot(W_k, y_j) - l_j * X_k[:, j]
            res = np.linalg.norm(r_j)
            if res > eps:  # If ||r_j|| > eps, jth eigenpair hasn't converged
                converged = False
                # Precondition next trial vector
                M_j = np.diag(
                    np.clip(np.reciprocal(D - l_j), a_min=-1e5, a_max=1e5)
                )  # Build diagonal preconditioner
                t_j = np.dot(M_j, r_j)
                # Orthogonalize new trial vector against previous basis vectors using Modified Gram-Schmidt (MG)S
                t_j = t_j / np.linalg.norm(t_j)
                for i in range(V_k.shape[1]):  # Iterate through k basis vectors
                    c_i = np.copy(
                        np.inner(V_k[:, i], t_j)
                    )  # Each process computes partial inner-product
                    t_j -= c_i * V_k[:, i]  # Remove component of t_j in V_k

                norm_tj = np.linalg.norm(t_j)
                if norm_tj > 1e-4:  # If new vector is `small`, ignore. Avoids ill-conditioning
                    V_k = np.c_[V_k, t_j / norm_tj]  # Append new vector to trial subspace
                    n_newvecs += 1

        if converged:
            break

        dim_S += n_newvecs  # Update dimension of trial subspace

        if q <= dim_S:  # Collapose trial basis
            V_new = np.c_[X_k[:, :n_eig], V_k[:, -n_newvecs:]]
            # Take leading n_eig Ritz vectors as new guess vectors
            dim_S = V_new.shape[1]  # New subspace dimension
            n_newvecs = dim_S
            W_k = np.zeros((self.local_size, 0), dtype="float")
            V_k = np.linalg.qr(V_new)  # Brute-force a full QR
            restart = True  # Indicate restart # TODO: Cleaner way to do this?
        elif n_newvecs == 0:
            V_new = X_k[:, :n_eig]
            # Take leading n_eig Ritz vectors as new guess vectors
            dim_S = V_new.shape[1]  # New subspace dimension
            n_newvecs = dim_S
            W_k = np.zeros((self.local_size, 0), dtype="float")
            V_k = np.linalg.qr(V_new)  # Brute-force a full QR
            restart = True  # Indicate restart

    else:
        raise NotImplementedError(f"Not converged. Returning {n_eig} lowest eigenvalue estimates")

    return L_k, X_k


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
        return np.einsum("i,j,ij ->", psi_coef, psi_coef, self.lewis.H(self.psi_det))

    @property
    def E_and_psi_coef(self) -> Tuple[Energy, Psi_coef]:
        # Return lower eigenvalue (aka the new E) and lower evegenvector (aka the new psi_coef)
        psi_H_psi = self.lewis.H(self.psi_det)
        try:
            energies, coeffs = davidson(psi_H_psi)
        except NotImplementedError:
            print("Davidson Failed, fallback to numpy eigh")
            energies, coeffs = np.linalg.eigh(psi_H_psi)

        return energies[0], coeffs[:, 0]

    def psi_external_pt2(self, psi_coef: Psi_coef, n_orb) -> Tuple[Psi_det, List[Energy]]:
        # Compute the pt2 contrution of all the external (aka connected) determinant.
        #   eα=⟨Ψ(n)∣H∣∣α⟩^2 / ( E(n)−⟨α∣H∣∣α⟩ )
        psi_external = Excitation(n_orb).gen_all_connected_determinant(self.psi_det)

        nomitator = np.einsum(
            "i,ij -> j", psi_coef, self.lewis.H(self.psi_det, psi_external)
        )  # vector * Matrix -> vector
        denominator = np.divide(
            1.0, self.E(psi_coef) - np.array([self.lewis.H_ii(d) for d in psi_external])
        )

        return psi_external, np.einsum(
            "i,i,i -> i", nomitator, nomitator, denominator
        )  # vector * vector * vector -> scalar

    def E_pt2(self, psi_coef: Psi_coef, n_orb) -> Energy:
        # The sum of the pt2 contribution of each external determinant
        _, psi_external_energy = self.psi_external_pt2(psi_coef, n_orb)
        return sum(psi_external_energy)


#  __
# (_   _  |  _   _ _|_ o  _  ._
# __) (/_ | (/_ (_  |_ | (_) | |
#
def selection_step(
    lewis: Hamiltonian, n_ord, psi_coef: Psi_coef, psi_det: Psi_det, n
) -> Tuple[Energy, Psi_coef, Psi_det]:
    # 1. Generate a list of all the external determinant and their pt2 contribution
    # 2. Take the n  determinants who have the biggest contribution and add it the wave function psi
    # 3. Diagonalize H corresponding to this new wave function to get the new variational energy, and new psi_coef.

    # In the main code:
    # -> Go to 1., stop when E_pt2 < Threshold || N < Threshold
    # See example of chained call to this function in `test_f2_631g_1p5p5det`

    # 1.
    psi_external_det, psi_external_energy = Powerplant(lewis, psi_det).psi_external_pt2(
        psi_coef, n_ord
    )

    # 2.
    idx = np.argpartition(psi_external_energy, n)[:n]
    psi_det_extented = psi_det + [psi_external_det[i] for i in idx]

    # 3.
    return (*Powerplant(lewis, psi_det_extented).E_and_psi_coef, psi_det_extented)


#  ___  _________ _____
#  |  \/  || ___ \_   _|
#  | .  . || |_/ / | |
#  | |\/| ||  __/  | |
#  | |  | || |    _| |_
#  \_|  |_/\_|    \___/
#


def dispatch_psi(comm, psi_i):
    """Function to partition and dispatch pieces of trial wavefunction psi_i to worker processes

    :param comm: mpi4py.MPI.COMM_WORLD communicator object
    :param psi_i: Psi_det, list of determinants"""

    rank = comm.Get_rank()  # Rank of current process
    world_size = comm.Get_size()  # No. of processes currently running
    MPI_master_rank = 0  # Denote master rank

    if rank == 0:
        indices = np.arange(len(psi_i), dtype="i")
        number_of_dets, res = divmod(len(psi_i), world_size)
        # Number of determinants sent to each rank; First `res` processes sent one extra
        sendcounts = np.array(
            [number_of_dets + 1 if i < res else number_of_dets for i in range(world_size)],
            dtype="i",
        )
        displacement = np.array(
            [sum(sendcounts[:i]) for i in range(world_size)], dtype="i"
        )  # Index of first determinant rank i receives
    else:
        indices = None  # Worker processes send nothing
        sendcounts = np.zeros(world_size, dtype="i")  # Preallocate space for count on workers
        displacement = None

    comm.Bcast(
        sendcounts, root=MPI_master_rank
    )  # Broadcast count to each process, which will need this to preallocate appropriate space
    local_indices = np.zeros(
        sendcounts[rank], dtype="i"
    )  # Preallocate space for process to receive count[rank] determinants
    comm.Scatterv([indices, sendcounts, displacement, MPI.INT], local_indices, root=MPI_master_rank)

    # Create new list of local determinants on each node
    return [psi_i[i] for i in local_indices]


@dataclass
class Hamiltonian_one_electron_manager(object):
    """Instance of `Hamiltonian_one_electron()` class for particular
    lists of internal/external determinants. Used to cache `local' Hamiltonian.
    Re-created at each CIPSI iteration."""

    psi_i: Psi_det  # Internal (or local) list of determinants
    psi_j: Psi_det  # External (or full internal) list of determinants
    d_one_e_integral: One_electron_integral
    E0: Energy

    def __init__(
        self,
        psi_i: Psi_det,
        psi_j: Psi_det,
        d_one_e_integral: One_electron_integral,
        E0: Energy,
        comm=MPI.COMM_WORLD,
    ):
        self.d_one_e_integral = d_one_e_integral
        self.E0 = E0
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        # Dispatch wavefunction and save psi_local, dictionary of 2e integrals with local instance of class
        if self.world_size == 1:  # One process running, use entire wavefunction
            self.psi_i = psi_j
        else:  # Else, dispatch chunks of the wavefunction to all processes
            self.psi_i = psi_i
        self.psi_j = psi_j
        # Local problem dimension (size of internal wavefunction or no. of local determinants)
        self.local_size = len(self.psi_i)
        # Full problem dimension (no. of connected determinants or size of full internal wavefunction)
        self.full_size = len(self.psi_j)

    @cached_property
    def Hamiltonian_one_electron_driver(self):
        # Re-reate and cache instance of driver class for particular internal/external determinants
        return Hamiltonian_one_electron(self.d_one_e_integral, self.E0)

    @cached_property
    def H(self):
        """Construct and cache len(psi_i) \times len(psi_j) one-electron
        Hamiltonian in a determinant-driven fashion.
        Used in building Hamiltonian to be diagonalized."""

        return self.Hamiltonian_one_electron_driver.H(self.psi_i, self.psi_j)

    @cached_property
    def H_full(self):
        """Build full one-electron Hamiltonian H = [H_1; ...; H_N]."""
        H_i = self.H  # Each process builds local Hamiltonian
        sendcounts = np.array(self.local_size * self.full_size, dtype="i")
        recvcounts = None
        if self.rank == 0:
            recvcounts = np.zeros(self.world_size, dtype="i")
        self.comm.Gather(
            sendcounts, recvcounts, root=self.MPI_master_rank
        )  # No. of elements received by each process
        H_full = np.zeros((self.full_size, self.full_size), dtype="float")
        # Master process gathers and sends full two-electron Hamiltonian
        self.comm.Gatherv(H_i, (H_full, recvcounts), root=self.MPI_master_rank)
        self.comm.Bcast(np.array(H_full, dtype="float"), root=self.MPI_master_rank)

        return H_full

    def D_i(self):
        """
        Build local diagonal of one-electron Hamiltonian in determinant-driven fashion.
        Needed for pre-conditioning step of Davidson's method.
        :return D_i: n_local dimensional numpy array
        """
        D_i = np.zeros(self.local_size, dtype="float")  # Save diagonal as a numpy vector
        for j, det in enumerate(
            self.psi_i
        ):  # Iterate through determinants in local portion of wavefunction
            D_i[j] = self.Hamiltonian_one_electron_driver.H_ii(det)
        return D_i

    def implicit_1e_matrix_product(self, V_k):
        """Function to implicitly compute matrix-matrix product W_ik = H_i * V_k by building the local Hamiltonian
        `on the fly', where H_i is the one-electron portion of the local Hamiltonian.

        :param H_i: local (self.local_size \times n) one-electron portion of the Hamiltonian, never explicitly stored
        :param V_k: full column basis for trial subspace (n \times k), as a numpy array

        :return W_ik: locally computed chunk of matrix-matrix product (self.local_size \times k), as a numpy array
        """
        try:
            k = V_k.shape[1]
        except IndexError:  # Handle case when V is a vector
            V_k = V_k.reshape(len(V_k), 1)
            k = 1
        W_ik = np.zeros(
            (self.local_size, k), dtype="float"
        )  # Pre-allocate space for local brick of matrix-matrix product
        # One-electron part
        for I, det_I in enumerate(self.psi_i):
            for J, det_J in enumerate(self.psi_j):
                W_ik[I, :] += (
                    self.Hamiltonian_one_electron_driver.H_ij(det_I, det_J) * V_k[J, :]
                )  # Update row I of W_ik

        return W_ik

    def explicit_1e_matrix_product(self, V_k):

        return np.dot(self.H, V_k)


@dataclass
class Hamiltonian_two_electrons_integral_driven_manager(object):
    """Instance of `Hamiltonian_two_electrons_integral_driven()` class for particular
    lists of internal/external determinants. Used to cache `local' Hamiltonian.
    Re-created at each CIPSI iteration."""

    psi_i: Psi_det  # Internal (or local) list of determinants
    psi_j: Psi_det  # External (or full internal) list of determinants
    d_two_e_integral: Two_electron_integral

    def __init__(
        self,
        psi_i: Psi_det,
        psi_j: Psi_det,
        d_two_e_integral: Two_electron_integral,
        comm=MPI.COMM_WORLD,
    ):
        self.d_two_e_integral = d_two_e_integral
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        # Dispatch wavefunction and save psi_local, dictionary of 2e integrals with local instance of class
        if self.world_size == 1:  # One process running, use entire wavefunction
            self.psi_i = psi_j
        else:  # Else, dispatch chunks of the wavefunction to all processes
            self.psi_i = psi_i
        self.psi_j = psi_j
        # Local problem dimension (size of internal wavefunction or no. of local determinants)
        self.local_size = len(self.psi_i)
        # Full problem dimension (no. of connected determinants or size of full internal wavefunction)
        self.full_size = len(self.psi_j)
        # Generate utilities
        generator = H_indices_generator(self.psi_i, self.psi_j)
        self.spindet_a_occ_i, self.spindet_b_occ_i = generator.spindet_occ_int
        self.det_to_index_j = generator.det_to_index_ext

    @cached_property
    def Hamiltonian_two_electrons_driver(self):
        # Create instance of driver class
        return Hamiltonian_two_electrons_integral_driven(self.d_two_e_integral)

    @cached_property
    def H(self):
        """Build and cache len(psi_i) \times len(psi_j) two-electron
        Hamiltonian in an integral-driven fashion.
        Used in building Hamiltonian to be diagonalized."""

        return self.Hamiltonian_two_electrons_driver.H(self.psi_i, self.psi_j)

    @cached_property
    def H_full(self):
        """Build full two-electron Hamiltonian H = [H_1; ...; H_N]."""
        H_i = self.H  # Each process builds local Hamiltonian
        sendcounts = np.array(self.local_size * self.full_size, dtype="i")
        recvcounts = None
        if self.rank == 0:
            recvcounts = np.zeros(self.world_size, dtype="i")
        self.comm.Gather(
            sendcounts, recvcounts, root=self.MPI_master_rank
        )  # No. of elements received by each process
        H_full = np.zeros((self.full_size, self.full_size), dtype="float")
        # Master process gathers and sends full two-electron Hamiltonian
        self.comm.Gatherv(H_i, (H_full, recvcounts), root=self.MPI_master_rank)
        self.comm.Bcast(np.array(H_full, dtype="float"), root=self.MPI_master_rank)

        return H_full

    def D_i(self):
        """
        Build local diagonal of two-electron Hamiltonian in determinant-driven fashion.
        Needed for pre-conditioning step of Davidson's method.
        :return D_i: n_local dimensional numpy array
        """
        D_i = np.zeros(self.local_size, dtype="float")  # Save diagonal as a numpy vector
        for j, det in enumerate(
            self.psi_i
        ):  # Iterate through determinants in local portion of wavefunction
            D_i[j] = self.Hamiltonian_two_electrons_driver.H_ii(det)

        return D_i

    def implicit_2e_matrix_product(self, V_k):
        """Function to implicitly compute matrix-matrix product W_ik = H_i * V_k by building the local Hamiltonian
        `on the fly', where H_i is the two-electron portion of the local Hamiltonian.

        :param H_i: local (self.local_size \times n) two-electron portion of the Hamiltonian, never explicitly stored
        :param V_k: full column basis for trial subspace (n \times k), as a numpy array

        :return W_ik: locally computed chunk of matrix-matrix product (self.local_size \times k), as a numpy array
        """
        try:
            k = V_k.shape[1]
        except IndexError:  # Handle case when V is a vector
            V_k = V_k.reshape(len(V_k), 1)
            k = 1
        W_ik = np.zeros(
            (self.local_size, k), dtype="float"
        )  # Pre-allocate space for local brick of matrix-matrix product
        # Two-electron part
        for (
            idx4,
            integral_values,
        ) in self.d_two_e_integral.items():  # Build two-electron elements of H_i `on the fly'
            idx = compound_idx4_reverse(idx4)
            for (I, J,), phase in self.Hamiltonian_two_electrons_driver.H_indices_idx(
                idx,
                self.psi_i,
                self.det_to_index_j,
                self.spindet_a_occ_i,
                self.spindet_b_occ_i,
            ):
                W_ik[I, :] += phase * integral_values * V_k[J, :]  # Update row I of W_ik

        return W_ik

    def explicit_2e_matrix_product(self, V_k):

        return np.dot(self.H, V_k)


@dataclass
class Hamiltonian_manager(object):
    psi_j: Psi_det
    d_one_e_integral: One_electron_integral
    d_two_e_integral: Two_electron_integral
    E0: Energy
    driven_by: str = "integral"

    def __init__(
        self,
        psi_j: Psi_det,
        d_one_e_integral: One_electron_integral,
        d_two_e_integral: Two_electron_integral,
        E0: Energy,
        comm,
    ):
        self.d_one_e_integral = d_one_e_integral
        self.d_two_e_integral = d_two_e_integral
        self.E0 = E0
        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        # Dispatch wavefunction and save psi_local, dictionary of 2e integrals with local instance of class
        if self.world_size == 1:  # One process running, use entire wavefunction
            self.psi_i = psi_j
        else:  # Else, dispatch chunks of the wavefunction to all processes
            self.psi_i = dispatch_psi(comm, psi_j)
        self.psi_j = psi_j
        # Local problem dimension (size of internal wavefunction or no. of local determinants)
        self.local_size = len(self.psi_i)
        # Full problem dimension (no. of connected determinants or size of full internal wavefunction)
        self.full_size = len(self.psi_j)

    @cached_property
    def H_i_one_electron(self):
        return Hamiltonian_one_electron_manager(
            self.psi_i, self.psi_j, self.d_one_e_integral, self.E0, self.comm
        )

    @cached_property
    def H_i_two_electrons(self):
        if self.driven_by == "integral":
            return Hamiltonian_two_electrons_integral_driven_manager(
                self.psi_i, self.psi_j, self.d_two_e_integral, self.comm
            )
        else:
            raise NotImplementedError

    def H_i(self) -> List[List[Energy]]:
        """Build a local portion of the Hamiltonian H_i of size (n_local \times n_full).
        Used for benchmarking, or applications where n_full is small enough to
        store the full H_i."""

        return self.H_i_one_electron.H + self.H_i_two_electrons.H

    def H_full(self):
        """Build full Hamiltonian of size (n_full \times n_fulL).
        Used for benchmarking, or applications where n_full small enough to
        store the full H."""

        return self.H_i_one_electron.H_full + self.H_i_two_electrons.H_full

    def D_i(self) -> List[Energy]:
        """Build local diagonal part of local Hamiltonian.
        By diagonal, we mean the part of H_i that is part of the diagonal in H.
        Used for preconditioning step in Davidson's."""

        return self.H_i_one_electron.D_i() + self.H_i_two_electrons.D_i()

    def implicit_matrix_product(self, V_k):
        """Function to implicitly compute matrix-matrix product W_ik = H_i * V_k by building the local Hamiltonian
        `on the fly'.

        :param H_i: local (self.local_size \times n) portion of the Hamiltonian, never explicitly stored
        :param V_k: full column basis for trial subspace (n \times k), as a numpy array

        :return W_ik: locally computed chunk of matrix-matrix product (self.local_size \times k), as a numpy array
        """

        return self.H_i_one_electron.implicit_1e_matrix_product(
            V_k
        ) + self.H_i_two_electrons.implicit_2e_matrix_product(V_k)

    def explicit_matrix_product(self, V_k):
        """Function to explicit compute matrix-matrix product W_ik = H_i * V_k.

        :param H_i: local (self.local_size \times n) portion of the Hamiltonian, cached.
        :param V_k: full column basis for trial subspace (n \times k), as a numpy array

        :return W_ik: locally computed chunk of matrix-matrix product (self.local_size \times k), as a numpy array
        """

        return self.H_i_one_electron.explicit_1e_matrix_product(
            V_k
        ) + self.H_i_two_electrons.explicit_2e_matrix_product(V_k)


@dataclass
class Davidson_manager(object):
    """A matrix-free implementation of Davidson's method in parallel.
    All matrix products involving the Hamiltonian are computed implicitly and on-the-fly.

    References:
    * `A Parallel Davidson-Type Algorithm for Several Eigenvalues' [L. Borges, S. Oliveira, 1998]
    * `The Davidson Method' [M. Crouzeix, B. Philippe, M. Sadkane, 1994]

    Each process will have local access to d_one_e_integral, d_two_e_integral, E0,
    and psi_full to build H_i on the fly.
    :param d_two_e_integral: Dictionary of two-electron integrals
    :param psi_full: Current trial wave-function, list of determinants. Problem size is n = len(psi_full)
    """

    d_one_e_integral: One_electron_integral
    d_two_e_integral: Two_electron_integral
    psi_full: Psi_det
    E0: Energy

    def __init__(
        self,
        comm,
        psi_full: Psi_det,
        d_one_e_integral: One_electron_integral,
        d_two_e_integral: Two_electron_integral,
        E0: Energy,
        problem_size,
    ):

        self.comm = comm
        self.world_size = self.comm.Get_size()  # No. of processes running
        self.rank = self.comm.Get_rank()  # Rank of current process
        self.MPI_master_rank = 0  # Master rank
        # Create instance of Hamiltonian_Manger class(). Dispatches wavefunction to worker processes
        self.Hamiltonian_manager = Hamiltonian_manager(
            psi_full, d_one_e_integral, d_two_e_integral, E0, comm
        )
        # For easier reference, full problem size
        self.full_size = problem_size
        # Set local problem size
        floor = self.full_size // self.world_size
        ceiling = floor + 1
        remainder = self.full_size % self.world_size
        self.distribution = [ceiling] * remainder + [floor] * (self.world_size - remainder)
        self.local_size = self.distribution[self.rank]  # Size of local chunk of Hamiltonian
        if self.rank == 0:
            print(
                f"Distribution of work: {self.distribution}, Local problem size: {self.local_size}"
            )
        # Compute offsets (start of the local section) for all nodes
        self.offsets = [0] + list(accumulate(self.distribution))
        del self.offsets[-1]
        # TODO: Just change one of these. I'm redoing some work in this and dispatch_psi
        # Throw error if allocation of work is not consistent
        assert self.local_size == self.Hamiltonian_manager.local_size

    def parallel_restart(self, dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik):
        """Perform an implicit restart; resize the trial subspace V_k
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
        v_new = np.zeros(self.full_size, dtype="float")
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
            c_j = np.copy(np.inner(V_ik[:, j], t_ik))  # Each process computes partial inner-product
            self.comm.Allreduce([c_j, MPI.DOUBLE], [c_j, MPI.DOUBLE])  # Default op=SUM
            t_ik = t_ik - c_j * V_ik[:, j]  # Remove component of t_ik in V_ik
        t_k = np.zeros(
            self.full_size, dtype="float"
        )  # Pre-allocate space to receive new guess vector
        self.comm.Allgatherv([t_ik, MPI.DOUBLE], [t_k, self.distribution, self.offsets, MPI.DOUBLE])
        norm_tk = np.linalg.norm(t_k)
        return t_ik / norm_tk, norm_tk  # Return new orthonormalized vector

    def preconditioning(self, D_i, lambda_k, r_ik):
        """Preconditon next guess vector

        :param D_i: diagonal portion of local Hamiltonian, as a numpy vector
        :param lambda_k: an eigenvalue, as a scalar
        :param r_ik: residual, a numpy vector

        :return numpy vector
        """
        M_k = np.diag(
            np.clip(np.reciprocal(D_i - lambda_k), a_min=-1e5, a_max=1e5)
        )  # Build diagonal preconditioner
        return np.dot(M_k, r_ik)

    def print_master(self, str):
        """Master rank prints inputted str"""
        if self.rank == 0:
            print(str)

    def distributed_davidson(
        self, H_i, n_eig=1, n_guess=1, eps=1e-7, max_iter=1000, q=1000, driven_by="explicit"
    ):
        """Davidson's method implemented in parallel. The Hamiltonian
        matrix is distrubted row-wise across MPI rank.
        Finds the n_eig smallest eigenvalues of a symmetric Hamiltonian.

        :param H: self.full_size \times self.full_size symmetric Hamiltonian
        :param H_i: self.local_size \times self.full_size `short and fat' locally distributed H
        :param n_guess: number of initial guess vectors per desired eigenvalue
        :param n_eig: number of eigenvalues to find
        :param eps: convergence tolerance
        :param max_iter: max no. of iterations for Davidson to run
        :param q: memory footprint tuning, q is maximally allowed subspace dimension

        :return a list of `n_eig` eigenvalues/associated eigenvectors, as numpy vector/array resp.
        """
        n = self.full_size  # Save full problem size
        # Establish local vars: trial subspace (V_ik) and action of H_i on full V_k (W_ik = H_i * V_k)
        V_ik = np.zeros((self.local_size, 0), dtype="float")
        W_ik = np.zeros((self.local_size, 0), dtype="float")
        # Set initial guess vectors and minimal initial subspace dimension
        dim_S = min(n_eig * n_guess, n)
        I = np.eye(n, dim_S)
        V_guess = I[
            self.offsets[self.rank] : (self.offsets[self.rank] + self.distribution[self.rank]), :
        ]
        V_ik = np.c_[V_ik, V_guess]
        # Build `diagonal` of local Hamiltonian
        D_i = self.Hamiltonian_manager.D_i()

        n_newvecs = dim_S  # No. of vectors added is initial subspace dimension
        restart = False
        for k in range(1, max_iter):
            if self.rank == 0:
                print(
                    f"Process rank: {self.rank}, Iterate: {k}, Working subspace dimension: {dim_S}"
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
            if driven_by == "explicit":
                W_inew = self.Hamiltonian_manager.explicit_matrix_product(V_new)
            elif driven_by == "implicit":
                W_inew = self.Hamiltonian_manager.implicit_matrix_product(V_new)
            else:
                raise NotImplementedError
            W_ik = np.c_[W_ik, W_inew]

            # TODO: Don't need to reduce full matrix at each step. Just new columns/rows
            # Compute rank computes partial update to the projected Hamiltonian S_k
            if (k == 1) or (
                restart
            ):  # If first iterate (or following a restart), need to compute full S_k explicitly
                S_ik = np.dot(V_ik.T, W_ik)
                restart = 0
            else:  # Else, append new rows & columns
                S_inew_c = np.dot(V_ik[:, :-n_newvecs].T, W_inew)
                S_ik = np.c_[S_ik, S_inew_c]
                S_inew_r = np.c_[np.dot(V_inew.T, W_ik[:, :-n_newvecs]), np.dot(V_inew.T, W_inew)]
                S_ik = np.r_[S_ik, S_inew_r]
            # Reduce contributions and form new S_k
            S_k = None  # Initialize
            if self.rank == 0:
                S_k = np.zeros((dim_S, dim_S), dtype="float")
            self.comm.Reduce([S_ik, MPI.DOUBLE], [S_k, MPI.DOUBLE])
            if self.rank == 0:  # Master diagonalizes S_k
                L_k, Y_k = np.linalg.eigh(S_k)
                L_k = np.array(L_k[:n_eig], dtype="float")
                Y_k = np.array(Y_k[:, :n_eig], dtype="float")
            else:  # Pre-allocate space to receive
                L_k = np.zeros(n_eig, dtype="float")
                Y_k = np.zeros((dim_S, n_eig), dtype="float")
            # Broadcast smallest n_eig estimates to all ranks to compute residuals
            self.comm.Bcast([L_k, MPI.DOUBLE], root=self.MPI_master_rank)
            self.comm.Bcast([Y_k, MPI.DOUBLE], root=self.MPI_master_rank)

            n_newvecs = 0  # Initialize counter; no. of new vectors added to trial subspace
            X_ik = np.dot(V_ik, Y_k)  # Pre-compute Ritz vectors (V_ik updated each iteration)
            converged = True  # Reset
            for j in range(n_eig):  # TODO: Remove converged eigenpairs
                # Grab jth eigenpair, each rank computes local portion of jth residual vector
                l_j, y_j = L_k[j], Y_k[:, j]
                r_ij = np.dot(W_ik, y_j) - l_j * X_ik[:, j]
                r_j = np.zeros(n, dtype="float")
                # Gather residual on each rank and compute its norm
                self.comm.Allgatherv(
                    [r_ij, MPI.DOUBLE], [r_j, self.distribution, self.offsets, MPI.DOUBLE]
                )
                res = np.linalg.norm(r_j)
                if self.rank == 0:
                    print(f"||r_k||: {res}")
                if res > eps:  # If ||r_j|| > eps, jth eigenpair hasn't converged
                    converged = False
                    if self.rank == 0:
                        print(f"Eigenvalue {j}: not converged, preconditioning next trial vector")
                    # Precondition next trial vector
                    t_ik = self.preconditioning(D_i, l_j, r_ij)
                    # Orthogonalize new trial vector against previous basis vectors via parallel-MGS
                    t_k = np.zeros(self.full_size, dtype="float")
                    self.comm.Allgatherv(
                        [np.array(t_ik, dtype="float"), MPI.DOUBLE],
                        [t_k, self.distribution, self.offsets, MPI.DOUBLE],
                    )
                    t_ik = t_ik / np.linalg.norm(t_k)
                    t_ik, norm_tk = self.mgs(V_ik, t_ik)
                    if norm_tk > 1e-4:  # If new vector is `small`, ignore. Avoids ill-conditioning
                        V_ik = np.c_[V_ik, t_ik]  # Append new vector to trial subspace
                        n_newvecs += 1
                else:
                    if self.rank == 0:
                        print(f"Eigenvalue {j}: {l_j}, converged, no new trial vector added")

            if converged:
                if self.rank == 0:
                    print("All eigenvalues converged, exiting iteration")
                break

            dim_S += n_newvecs  # Update dimension of trial subspace

            if q <= dim_S:  # Collapose trial basis
                if self.rank == 0:
                    print(f"q <= dim_S: {dim_S}, restarting Davidson's")
                dim_S, n_newvecs, V_ik, W_ik = self.parallel_restart(
                    dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik
                )
                restart = True  # Indicate restart

            elif n_newvecs == 0:
                if self.rank == 0:
                    print("No new vectors added at previous iteration, restarting Davidson's")
                dim_S, n_newvecs, V_ik, W_ik = self.parallel_restart(
                    dim_S, n_eig, n_newvecs, X_ik, V_ik, W_ik
                )
                restart = True  # Indicate restart

        else:
            raise NotImplementedError(f"Davidson not converged")

        m = X_ik.shape[1]  # Same across all ranks
        X_k = np.zeros((n, m), dtype="float")
        # Gather Ritz vectors on all ranks
        self.comm.Allgatherv(
            [X_ik, MPI.DOUBLE],
            [X_k, m * np.array(self.distribution), m * np.array(self.offsets), MPI.DOUBLE],
        )

        return L_k, X_k
