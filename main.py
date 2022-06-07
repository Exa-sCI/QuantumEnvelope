#!/usr/bin/env python3

# Types
# -----
from typing import Tuple, Dict, NewType, NamedTuple, List, Set, Iterator, NewType
from dataclasses import dataclass

# Yes, I like itertools
from itertools import chain, product, combinations, takewhile
from functools import partial, cached_property, cache
from collections import defaultdict
import numpy as np
from math import sqrt


# Orbital index (1,2,...,n_orb)
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

class DetDiff(NamedTuple):

    holes: Determinant
    particles: Determinant
    def __repr__(self):
        return(f'DetDiff(hA={self.holes.alpha}, pA={self.particles.alpha}; hB={self.holes.beta}, pB={self.particles.beta})')


Psi_det = List[Determinant]
Psi_coef = List[float]
# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det
Energy = NewType("Energy", float)


def det_diff(det1: Determinant, det2: Determinant) -> DetDiff:
    return DetDiff(
            Determinant( tuple(sorted(set(det1.alpha) - set(det2.alpha))),
                         tuple(sorted(set(det1.beta)  - set(det2.beta)))),
            Determinant( tuple(sorted(set(det2.alpha) - set(det1.alpha))),
                         tuple(sorted(set(det2.beta)  - set(det1.beta)))))
# _____          _           _               _   _ _   _ _
#|_   _|        | |         (_)             | | | | | (_) |
#  | | _ __   __| | _____  ___ _ __   __ _  | | | | |_ _| |___
#  | || '_ \ / _` |/ _ \ \/ / | '_ \ / _` | | | | | __| | / __|
# _| || | | | (_| |  __/>  <| | | | | (_| | | |_| | |_| | \__ \
# \___/_| |_|\__,_|\___/_/\_\_|_| |_|\__, |  \___/ \__|_|_|___/
#                                     __/ |
#                                    |___/

@cache
def compound_idx2(i,j):
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
    p,q = min(i,j),max(i,j)
    return (q*(q+1))//2+p

def compound_idx4(i,j,k,l):
    """
    nested calls to compound_idx2
    """
    return compound_idx2(compound_idx2(i,k),compound_idx2(j,l))

@cache
def compound_idx2_reverse(ij):
    """
    >>> all(compound_idx2(*compound_idx2_reverse(A)) == A for A in range(10000))
    True
    """
    j=int((sqrt(1+8*ij)-1)/2)
    i=ij-(j*(j+1)//2)
    return i,j

@cache
def compound_idx4_reverse(ijkl):
    """
    >>> all(compound_idx4(*compound_idx4_reverse(A)) == A for A in range(10000))
    True
    """
    ik,jl = compound_idx2_reverse(ijkl)
    i,k = compound_idx2_reverse(ik)
    j,l = compound_idx2_reverse(jl)
    return i,j,k,l

@cache
def compound_idx4_reverse_all(ijkl):
    """
    return all 8 permutations that are equivalent for real orbitals
    for complex orbitals, they are ordered as:
    v, v, v*, v*, u, u, u*, u*
    where v == <ij|kl>, u == <ij|lk>, and * denotes the complex conjugate
    >>> def check_idx(A):
    ...     return all(compound_idx4(i,j,k,l)==A for i,j,k,l in compound_idx4_reverse_all(A))
    >>> all(check_idx(A) for A in range(1000))
    True
    """
    i,j,k,l = compound_idx4_reverse(ijkl)
    return (i,j,k,l),(j,i,l,k),(k,l,i,j),(l,k,j,i),(i,l,k,j),(l,i,j,k),(k,j,i,l),(j,k,l,i)

@cache
def compound_idx4_reverse_all_unique(ijkl):
    """
    return only the unique 4-tuples from compound_idx4_reverse_all
    """
    return tuple(set(compound_idx4_reverse_all(ijkl)))

@cache
def canonical_idx4(i,j,k,l):
    """
    for real orbitals, return same 4-tuple for all equivalent integrals
    returned (i,j,k,l) should satisfy the following:
        i <= k
        j <= l
        (k < l) or (k==l and i <= j)
    the last of these is equivalent to (compound_idx2(i,k) <= compound_idx2(j,l))
    >>> all(\
            all(\
                canonical_idx4(*compound_idx4_reverse(A)) == B \
            for B in (canonical_idx4(i,j,k,l) for (i,j,k,l) in compound_idx4_reverse_all(A)))\
        for A in range(1000))
    True
    """
    i,k = min(i,k),max(i,k)
    ik = compound_idx2(i,k)
    j,l = min(j,l),max(j,l)
    jl = compound_idx2(j,l)
    if ik<=jl:
        return i,j,k,l
    else:
        return j,i,l,k
@cache
def canonical_idx4_reverse(ijkl):
    return canonical_idx4(*compound_idx4_reverse(ijkl))

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
            d_one_e_integral[(i, k)] = v
            d_one_e_integral[(k, i)] = v
        else:
            # Two-electron integrals have many permutation symmetries:
            # Exchange r1 and r2 (indices i,k and j,l)
            # Exchange i,k
            # Exchange j,l
            d_two_e_integral[compound_idx4(i, j, k, l)] = v

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
        for i, v in enumerate(str_, start=1):
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

    @staticmethod
    def apply_excitation(spindet, exc: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        lh, lp = exc
        s = (set(spindet) - set(lh)) | set(lp)
        return tuple(sorted(s))

    def gen_all_connected_spindet(self, spindet: Spin_determinant, ed: int) -> Iterator:
        """
        Generate all the posible spin determinant relative to a excitation degree

        >>> sorted(Excitation(4).gen_all_connected_spindet( (1,2), 1))
        [(1, 3), (1, 4), (2, 3), (2, 4)]
        """
        l_exc = self.gen_all_excitation(spindet, ed)
        apply_excitation_to_spindet = partial(Excitation.apply_excitation, spindet)
        return map(apply_excitation_to_spindet, l_exc)

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
        >>> Excitation.exc_degree(Determinant(alpha=(1, 2), beta=(1, 2)),
        ...                     Determinant(alpha=(1, 3), beta=(5, 7)))
        (1, 2)
        """
        ed_up = Excitation.exc_degree_spindet(det_i.alpha,det_j.alpha) 
        ed_dn = Excitation.exc_degree_spindet(det_i.beta,det_j.beta)
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
    def single_exc(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> PhaseIdx.single_exc((1, 5, 7), (1, 23, 7))
        (1, 5, 23)
        >>> PhaseIdx.single_exc((1, 2, 9), (1, 9, 18))
        (-1, 2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        return PhaseIdx.single_phase(sdet_i, sdet_j, h, p), h, p

    @staticmethod
    def double_phase(sdet_i, sdet_j, h1, h2, p1, p2):
        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = PhaseIdx.single_phase(sdet_i, sdet_j, h1, p1) * PhaseIdx.single_phase(
            sdet_j, sdet_i, p2, h2
        )
        # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
        # Look like to be always true in our tests
        if (min(h2, p2) < max(h1, p1)) != (h2 < p1 or p2 < h1):
            phase = -phase
            raise NotImplementedError(
                f"double_exc QP conditional was trigered! Please repport to the developpers {sdet_i}, {sdet_j}"
            )
        return phase

    @staticmethod
    def double_exc(
        sdet_i: Spin_determinant, sdet_j: Spin_determinant
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> PhaseIdx.double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (1, 3, 4, 12, 13)
        >>> PhaseIdx.double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (-1, 3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

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
        key = compound_idx4(i,j,k,l)
        return self.d_two_e_integral[key]

    @staticmethod
    def H_ii_indices(det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """Diagonal element of the Hamiltonian : <I|H|I>.
        >>> sorted(Hamiltonian_two_electrons_determinant_driven.H_ii_indices( Determinant((1,2),(3,4))))
        [((1, 2, 1, 2), 1), ((1, 2, 2, 1), -1), ((1, 3, 1, 3), 1), ((1, 4, 1, 4), 1),
         ((2, 3, 2, 3), 1), ((2, 4, 2, 4), 1), ((3, 4, 3, 4), 1), ((3, 4, 4, 3), -1)]
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
        return sum(phase * self.H_ijkl_orbital(*idx) for idx, phase in self.H_ii_indices(det_i))


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
        key = compound_idx4(i,j,k,l)
        return self.d_two_e_integral[key]

    @staticmethod
    def H_ii_indices(det_i: Determinant) -> Iterator[Two_electron_integral_index_phase]:
        """Diagonal element of the Hamiltonian : <I|H|I>.
        >>> sorted(Hamiltonian_two_electrons_integral_driven.H_ii_indices( Determinant((1,2),(3,4))))
        [((1, 2, 1, 2), 1), ((1, 2, 2, 1), -1), ((1, 3, 1, 3), 1), ((1, 4, 1, 4), 1),
         ((2, 3, 2, 3), 1), ((2, 4, 2, 4), 1), ((3, 4, 3, 4), 1), ((3, 4, 4, 3), -1)]
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
    def single_Ss(
        psi_i,
        psi_j,
        idx,
        spindet_a_occ_i,
        spindet_b_occ_i,
        spindet_a_occ_j,
        spindet_b_occ_j,
        exc,
        spin,
    ):
        """
        yield all det pairs (a,b) and phase where <a|H|b> depends on the integral with index idx
        filter out anything that isn't a single excitation
        """
        i, j, k, l = idx

        # limit to only single excitations
        if j == l:
            # <ij|kj>
            # from ia to ka where ja is occupied
            S1 = (spindet_a_occ_i[i] & spindet_a_occ_i[j]) - spindet_a_occ_i[k]
            R1 = (spindet_a_occ_j[k] & spindet_a_occ_j[j]) - spindet_a_occ_j[i]
            # <ij|kj>
            # from ia to ka where jb is occupied
            S2 = (spindet_a_occ_i[i] & spindet_b_occ_i[j]) - spindet_a_occ_i[k]
            R2 = (spindet_a_occ_j[k] & spindet_b_occ_j[j]) - spindet_a_occ_j[i]
            # j==l
            # Kevin: separating these two is incorrect because it double counts the intersection of the two products
            # Thomas & Brice: Realy?!
            # might be useful to form (a_i[i] - a_i[k]) (used in S1 and S2)
            #                         (a_j[k] - a_j[i]) (used in R1 and R2)
            for a, b in chain(product(S1, R1), product(S2, R2)):
                det_i, det_j = psi_i[a], psi_j[b]
                ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
                if (ed_up, ed_dn) == exc:
                    yield (a, b), PhaseIdx.single_phase(
                        getattr(det_i, spin), getattr(det_j, spin), i, k
                    )

        # <ij|jl> = -<ij|lj>
        # from ia to la where ja is occupied
        if j == k:
            S3 = (spindet_a_occ_i[i] & spindet_a_occ_i[j]) - spindet_a_occ_i[l]
            R3 = (spindet_a_occ_j[l] & spindet_a_occ_j[j]) - spindet_a_occ_j[i]

            for a, b in product(S3, R3):
                det_i, det_j = psi_i[a], psi_j[b]
                ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
                if (ed_up, ed_dn) == exc:
                    yield (a, b), -PhaseIdx.single_phase(
                        getattr(det_i, spin), getattr(det_j, spin), i, l
                    )


    @staticmethod
    def double_different(
        idx,
        psi_i,
        psi_j,
        spindet_a_occ_i,
        spindet_b_occ_i,
        spindet_a_occ_j,
        spindet_b_occ_j,
        spin_a,
        spin_b,
    ):
        i, j, k, l = idx

        S1 = (spindet_a_occ_i[i] & spindet_b_occ_i[j]) - (spindet_a_occ_i[k] | spindet_b_occ_i[l])
        R1 = (spindet_a_occ_j[k] & spindet_b_occ_j[l]) - (spindet_a_occ_j[i] | spindet_b_occ_j[j])
        for a, b in product(S1, R1):
            det_i, det_j = psi_i[a], psi_j[b]
            ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
            if (ed_up, ed_dn) == (1, 1):
                phaseA, hA, pA = PhaseIdx.single_exc(getattr(det_i, spin_a), getattr(det_j, spin_a))
                phaseB, hB, pB = PhaseIdx.single_exc(getattr(det_i, spin_b), getattr(det_j, spin_b))
                yield (a, b), phaseA * phaseB

    @staticmethod
    def double_same_unique_external(idx, psi_i, psi_j, spindet_a_occ_i, spindet_a_occ_j, exc, spin):
        def foo(i,j,k,l):
            if i==j:
                return
            if k == l: # p1 == p2, both branch should have been take, 0 contribution
                return
            phasemod = 1
            if j<i:
                phasemod *= -1
            if l<k:
                phasemod *= -1
            S1 = (spindet_a_occ_i[i] & spindet_a_occ_i[j]) - (spindet_a_occ_i[k] | spindet_a_occ_i[l])
            R1 = (spindet_a_occ_j[k] & spindet_a_occ_j[l]) - (spindet_a_occ_j[i] | spindet_a_occ_j[j])

            for a, b in product(S1, R1):
                det_i, det_j = psi_i[a], psi_j[b]
                ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
                # Should some preselection to only double or at list only double+ but nothing in the other spin
                if (ed_up, ed_dn) == exc:
                    phase, h1, h2, p1, p2 = PhaseIdx.double_exc(
                        getattr(det_i, spin), getattr(det_j, spin)
                    )
                    yield (a,b),phase*phasemod
        i, j, k, l = idx
        yield from foo(i,j,k,l)
        yield from foo(i,l,k,j)
        yield from foo(k,l,i,j)
        yield from foo(j,k,l,i)

    @staticmethod
    def double_same_unique_internal(idx, psi_i, spindet_a_occ_i, exc, spin):
        def foo(i,j,k,l,pfac=1):
            if i==j:
                return
            if k == l: # p1 == p2, both branch should have been take, 0 contribution
                return
            phasemod = 1
            if j<i:
                phasemod *= -1
            if l<k:
                phasemod *= -1
            S1 = (spindet_a_occ_i[i] & spindet_a_occ_i[j]) - (spindet_a_occ_i[k] | spindet_a_occ_i[l])
            R1 = (spindet_a_occ_i[k] & spindet_a_occ_i[l]) - (spindet_a_occ_i[i] | spindet_a_occ_i[j])

            for a, b in product(S1, R1):
                det_i, det_j = psi_i[a], psi_i[b]
                ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
                # Should some preselection to only double or at list only double+ but nothing in the other spin
                if (ed_up, ed_dn) == exc:
                    phase, h1, h2, p1, p2 = PhaseIdx.double_exc(
                        getattr(det_i, spin), getattr(det_j, spin)
                    )
                    yield (a,b),phase*phasemod
                    yield (b,a),phase*phasemod
        i, j, k, l = idx
        yield from foo(i,j,k,l)
        yield from foo(i,l,k,j)


    def double_same_unique(self,idx, psi_i, psi_j, spindet_a_occ_i, spindet_a_occ_j, exc, spin):
        if psi_i == psi_j:
            yield from self.double_same_unique_internal(idx,psi_i,spindet_a_occ_i,exc,spin)
        else:
            yield from self.double_same_unique_external(idx, psi_i, psi_j, spindet_a_occ_i, spindet_a_occ_j, exc, spin)
        

    def H_pair_phase_from_idx_unique(
        self, idx, spindet_a_occ_i, spindet_b_occ_i, psi_i, spindet_a_occ_j, spindet_b_occ_j, psi_j
    ):
        i, j, k, l = idx

        if i != j:
            yield from self.double_same_unique(
                idx, psi_i, psi_j, spindet_a_occ_i, spindet_a_occ_j, (2, 0), "alpha"
            )
            yield from self.double_same_unique(
                idx, psi_i, psi_j, spindet_b_occ_i, spindet_b_occ_j, (0, 2), "beta"
            )


    def H_pair_phase_from_idx(
        self, idx, spindet_a_occ_i, spindet_b_occ_i, psi_i, spindet_a_occ_j, spindet_b_occ_j, psi_j
    ):
        i, j, k, l = idx

        yield from self.single_Ss(
            psi_i,
            psi_j,
            idx,
            spindet_a_occ_i,
            spindet_b_occ_i,
            spindet_a_occ_j,
            spindet_b_occ_j,
            (1, 0),
            "alpha",
        )
        yield from self.single_Ss(
            psi_i,
            psi_j,
            idx,
            spindet_b_occ_i,
            spindet_a_occ_i,
            spindet_b_occ_j,
            spindet_a_occ_j,
            (0, 1),
            "beta",
        )

        if i < j:
            yield from self.double_different(
                idx,
                psi_i,
                psi_j,
                spindet_a_occ_i,
                spindet_b_occ_i,
                spindet_a_occ_j,
                spindet_b_occ_j,
                "alpha",
                "beta",
            )
        # above we do (hA < hB) so (hB < hA) and (hA==hB)
        if i <= j:
            yield from self.double_different(
                idx,
                psi_i,
                psi_j,
                spindet_b_occ_i,
                spindet_a_occ_i,
                spindet_b_occ_j,
                spindet_a_occ_j,
                "beta",
                "alpha",
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

        def get_dets_occ(psi_i: Psi_det, spin: str) -> Dict[OrbitalIdx, Set[int]]:
            ds = defaultdict(set)
            for i, det in enumerate(psi_i):
                for o in getattr(det, spin):
                    ds[o].add(i)
            return ds

        return tuple(get_dets_occ(psi_i, spin) for spin in ["alpha", "beta"])

    def H_indices(self, psi_i, psi_j) -> Iterator[Two_electron_integral_index_phase]:
        # This only need iijj, ijji integral
        # This can be stored in memory, hence we will do the determinant driven way
        # Only one node will be responsible for it
        for a, det_i in enumerate(psi_i):
            for b, det_j in enumerate(psi_j):
                ed_up, ed_dn = Excitation.exc_degree(det_i, det_j)
                if (ed_up, ed_dn) == (0, 0):
                    for idx, phase in self.H_ii_indices(det_i):
                        yield (a, b), idx, phase

        spindet_a_occ_i, spindet_b_occ_i = self.get_spindet_a_occ_spindet_b_occ(psi_i)
        spindet_a_occ_j, spindet_b_occ_j = self.get_spindet_a_occ_spindet_b_occ(psi_j)

        for key in self.d_two_e_integral:

            #TODO: fix H_pair_phase_from_idx so we can loop over only the canonical ijkl
            for idx in compound_idx4_reverse_all_unique(key):
            #idx = compound_idx4_reverse(key)
                for (a, b), phase in self.H_pair_phase_from_idx(
                    idx,
                    spindet_a_occ_i,
                    spindet_b_occ_i,
                    psi_i,
                    spindet_a_occ_j,
                    spindet_b_occ_j,
                    psi_j
                ):
                    yield (a, b), idx, phase

            idx = canonical_idx4_reverse(key)
            for (a, b), phase in self.H_pair_phase_from_idx_unique(
                idx,
                spindet_a_occ_i,
                spindet_b_occ_i,
                psi_i,
                spindet_a_occ_j,
                spindet_b_occ_j,
                psi_j
            ):
                yield (a, b), idx, phase


    def H(self, psi_i, psi_j) -> List[List[Energy]]:
        # This is the function who will take foreever
        h = np.zeros(shape=(len(psi_i), len(psi_j)))
        for (a, b), (i, j, k, l), phase in self.H_indices(psi_i, psi_j):
            h[a, b] += phase * self.H_ijkl_orbital(i, j, k, l)
        return h

    def H_ii(self, det_i: Determinant):
        return sum(phase * self.H_ijkl_orbital(*idx) for idx, phase in self.H_ii_indices(det_i))


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


#   _____         _   _
#  |_   _|       | | (_)
#    | | ___  ___| |_ _ _ __   __ _
#    | |/ _ \/ __| __| | '_ \ / _` |
#    | |  __/\__ \ |_| | | | | (_| |
#    \_/\___||___/\__|_|_| |_|\__, |
#                              __/ |
#                             |___/
import unittest
import time


class Timing:
    def setUp(self):
        print(f"{self.id()} ... ", end="", flush=True)
        self.startTime = time.perf_counter()
        if PROFILING:
            import cProfile

            self.pr = cProfile.Profile()
            self.pr.enable()

    def tearDown(self):
        t = time.perf_counter() - self.startTime
        print(f"ok ({t:.3f}s)")
        if PROFILING:
            from pstats import Stats

            self.pr.disable()
            p = Stats(self.pr)
            p.strip_dirs().sort_stats("tottime").print_stats(0.05)


class Test_VariationalPowerplant:
    def test_c2_eq_dz_3(self):
        fcidump_path = "c2_eq_hf_dz.fcidump*"
        wf_path = "c2_eq_hf_dz_3.*.wf*"
        E_ref = load_eref("data/c2_eq_hf_dz_3.*.ref*")
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_c2_eq_dz_4(self):
        fcidump_path = "c2_eq_hf_dz.fcidump*"
        wf_path = "c2_eq_hf_dz_4.*.wf*"
        E_ref = load_eref("data/c2_eq_hf_dz_4.*.ref*")
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_1det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        E_ref = -198.646096743145
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.10det.wf"
        E_ref = -198.548963
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_30det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.30det.wf"
        E_ref = -198.738780989106
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_161det(self):
        fcidump_path = "f2_631g.161det.fcidump"
        wf_path = "f2_631g.161det.wf"
        E_ref = -198.8084269796
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_296det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.296det.wf"
        E_ref = -198.682736076007
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)


def load_and_compute(fcidump_path, wf_path, driven_by):
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
    # Load wave function
    psi_coef, psi_det = load_wf(f"data/{wf_path}")
    # Computation of the Energy of the input wave function (variational energy)
    lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by)
    return Powerplant(lewis, psi_det).E(psi_coef)


class Test1_VariationalPowerplant_Determinant(
    Timing, unittest.TestCase, Test_VariationalPowerplant
):
    def load_and_compute(self, fcidump_path, wf_path):
        return load_and_compute(fcidump_path, wf_path, "determinant")


class Test1_VariationalPowerplant_Integral(Timing, unittest.TestCase, Test_VariationalPowerplant):
    def load_and_compute(self, fcidump_path, wf_path):
        return load_and_compute(fcidump_path, wf_path, "integral")


class Test_VariationalPT2Powerplant:
    def test_f2_631g_1det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        E_ref = -0.367587988032339
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_2det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.2det.wf"
        E_ref = -0.253904406461572
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.10det.wf"
        E_ref = -0.24321128
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_28det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.28det.wf"
        E_ref = -0.244245625775444
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)


def load_and_compute_pt2(fcidump_path, wf_path, driven_by):
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
    # Load wave function
    psi_coef, psi_det = load_wf(f"data/{wf_path}")
    # Computation of the Energy of the input wave function (variational energy)
    lewis = Hamiltonian(d_one_e_integral, d_two_e_integral, E0, driven_by)
    return Powerplant(lewis, psi_det).E_pt2(psi_coef, n_ord)


class Test_VariationalPT2_Determinant(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path):
        return load_and_compute_pt2(fcidump_path, wf_path, "determinant")


class Test_VariationalPT2_Integral(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path):
        return load_and_compute_pt2(fcidump_path, wf_path, "integral")


class TestSelection(unittest.TestCase):
    def load(self, fcidump_path, wf_path):
        # Load integrals
        n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        return (
            n_ord,
            psi_coef,
            psi_det,
            Hamiltonian(d_one_e_integral, d_two_e_integral, E0),
        )

    def test_f2_631g_1p0det(self):
        # Verify that selecting 0 determinant is egual that computing the variational energy
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        E_var = Powerplant(lewis, psi_det).E(psi_coef)

        E_selection, _, _ = selection_step(lewis, n_ord, psi_coef, psi_det, 0)

        self.assertAlmostEqual(E_var, E_selection, places=6)

    def test_f2_631g_1p10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        # No a value optained with QP
        E_ref = -198.72696793971556
        # Selection 10 determinant and check if the result make sence

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        E, _, _ = selection_step(lewis, n_ord, psi_coef, psi_det, 10)

        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_1p5p5det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        # We will select 5 determinant, than 5 more.
        # The value is lower than the one optained by selecting 10 deterinant in one go.
        # Indeed, the pt2 get more precise whith the number of selection
        E_ref = -198.73029308564543

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        _, psi_coef, psi_det = selection_step(lewis, n_ord, psi_coef, psi_det, 5)
        E, psi_coef, psi_det = selection_step(lewis, n_ord, psi_coef, psi_det, 5)

        self.assertAlmostEqual(E_ref, E, places=6)


if __name__ == "__main__":
    import doctest
    import sys

    try:
        sys.argv.remove("--profiling")
    except ValueError:
        PROFILING = False
    else:
        PROFILING = True
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE, raise_on_error=True)
    unittest.main(failfast=True, verbosity=0)
