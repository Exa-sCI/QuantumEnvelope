from typing import Tuple, Dict, NamedTuple, List, NewType, Iterator
from itertools import chain, product, combinations
from functools import partial, cache

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

#   ______     _                      _                   _
#   |  _  \   | |                    (_)                 | |
#   | | | |___| |_ ___ _ __ _ __ ___  _ _ __   __ _ _ __ | |_
#   | | | / _ \ __/ _ \ '__| '_ ` _ \| | '_ \ / _` | '_ \| __|
#   | |/ /  __/ ||  __/ |  | | | | | | | | | | (_| | | | | |_
#   |___/ \___|\__\___|_|  |_| |_| |_|_|_| |_|\__,_|_| |_|\__|
#
#

Spin_determinant_tuple = Tuple[OrbitalIdx, ...]
Spin_determinant_bitstring = int


class Determinant(NamedTuple):
    """Slater determinant: Product of 2 determinants.
    One for $\alpha$ electrons and one for \beta electrons.
    Generic |Determinant| class; inherited by |Determinant_tuple| and |Determinant_bitstring|"""

    alpha: object
    beta: object


class Determinant_tuple(Determinant):
    """Slater determinant: Product of 2 determinants.
    One for \alpha electrons and one for \beta electrons.

    Internal representation of |Determinants| as |tuple| of integers
    Handle via set operations.
    """

    alpha: Spin_determinant_tuple
    beta: Spin_determinant_tuple

    def to_bitstring(self, Norb):
        """Convert instance of |Determinant_tuple| -> |Determinant_bitstring|
        >>> Determinant_tuple((0, 2), (1,)).to_bitstring(4)
        Determinant_bitstring(alpha=5, beta=2)
        >>> Determinant_tuple((0, 2), ()).to_bitstring(4)
        Determinant_bitstring(alpha=5, beta=0)
        """
        # Do for each |Spin_determinant_tuple|, then join
        # Strings immutable -> Work with lists for ON assignment, convert to |str| when done
        alpha_str = ["0", "b"]
        beta_str = ["0", "b"]
        alpha_str.extend(["0"] * Norb)
        beta_str.extend(["0"] * Norb)
        for o in self.alpha:
            alpha_str[-(o + 1)] = "1"
        for o in self.beta:
            beta_str[-(o + 1)] = "1"

        # Return |Determinant_bitstring| representation of instance of |Determinant_tuple|
        return Determinant_bitstring(int(("".join(alpha_str)), 2), int(("".join(beta_str)), 2))

    #     _
    #    |_     _ o _|_  _. _|_ o  _  ._
    #    |_ >< (_ |  |_ (_|  |_ | (_) | |
    #

    # Here, we put all the necessary driver functions for applying excitations, computing exc_degree, etc.
    # All operations performed on |Determinant_tuple| objects handled via set operations

    @staticmethod
    def apply_excitation(
        sdet: Spin_determinant_tuple, exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]]
    ) -> Tuple[OrbitalIdx, ...]:
        """Function to `apply' excitation to |Spin_determinant_tuple| object
        Implemented via symmetric set difference (^)

        :param exc: variable length tuple containing [[holes], [particles]] that determine the excitation

        >>> Determinant_tuple.apply_excitation((0, 1), ((1,), (2,)))
        (0, 2)
        >>> Determinant_tuple.apply_excitation((1, 3), ((1,), (2,)))
        (2, 3)
        >>> Determinant_tuple.apply_excitation((0, 1), ((), ()))
        (0, 1)
        """
        lh, lp = exc  # Unpack
        return tuple(sorted(set(sdet) ^ (set(lh) | set(lp))))

    def gen_all_connected_spindet(
        self, ed: int, n_orb: int, spin="alpha"
    ) -> Iterator[Tuple[OrbitalIdx, ...]]:
        """Generate all connected spin determinants to self relative to a particular excitation degree
        :param n_orb: global parameter
        >>> sorted(Determinant_tuple((0, 1), ()).gen_all_connected_spindet(1, 4))
        [(0, 2), (0, 3), (1, 2), (1, 3)]
        >>> sorted(Determinant_tuple((0, 1), ()).gen_all_connected_spindet(2, 4))
        [(2, 3)]
        >>> sorted(Determinant_tuple((0, 1), ()).gen_all_connected_spindet(2, 2))
        []
        """
        sdet = getattr(self, spin)
        # Compute all possible holes (occupied orbitals in self) and particles (empty orbitals in self)
        holes = combinations(sdet, ed)
        particles = combinations(set(range(n_orb)) - set(sdet), ed)
        l_hp_pairs = product(holes, particles)
        apply_excitation_to_spindet = partial(self.apply_excitation, sdet)
        return map(apply_excitation_to_spindet, l_hp_pairs)

    def gen_all_connected_det(self, n_orb: int) -> Iterator[Determinant]:
        """Generate all determinants that are singly or doubly connected to self
        :param n_orb: global parameter, needed to cap possible excitations

        >>> sorted(Determinant_tuple((0, 1), (0,)).gen_all_connected_det(3))
        [Determinant_tuple(alpha=(0, 1), beta=(1,)),
         Determinant_tuple(alpha=(0, 1), beta=(2,)),
         Determinant_tuple(alpha=(0, 2), beta=(0,)),
         Determinant_tuple(alpha=(0, 2), beta=(1,)),
         Determinant_tuple(alpha=(0, 2), beta=(2,)),
         Determinant_tuple(alpha=(1, 2), beta=(0,)),
         Determinant_tuple(alpha=(1, 2), beta=(1,)),
         Determinant_tuple(alpha=(1, 2), beta=(2,))]
        """
        # Generate all singles from constituent alpha and beta spin determinants
        # Then, opposite-spin and same-spin doubles

        # We use l_single_a, and l_single_b twice. So we store them.
        l_single_a = set(self.gen_all_connected_spindet(1, n_orb, "alpha"))
        l_double_aa = self.gen_all_connected_spindet(2, n_orb, "alpha")

        # Singles and doubles; alpha spin
        exc_a = (
            Determinant_tuple(det_alpha, self.beta) for det_alpha in chain(l_single_a, l_double_aa)
        )

        l_single_b = set(self.gen_all_connected_spindet(1, n_orb, "beta"))
        l_double_bb = self.gen_all_connected_spindet(2, n_orb, "beta")

        # Singles and doubles; beta spin
        exc_b = (
            Determinant_tuple(self.alpha, det_beta) for det_beta in chain(l_single_b, l_double_bb)
        )

        l_double_ab = product(l_single_a, l_single_b)

        # Doubles; opposite-spin
        exc_ab = (Determinant_tuple(det_alpha, det_beta) for det_alpha, det_beta in l_double_ab)

        return chain(exc_a, exc_b, exc_ab)

    @staticmethod
    @cache
    def exc_degree_spindet(sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]) -> int:
        # Cache since many of these will be re-used in computation of exc_degree between two dets
        return len(set(sdet_i) ^ set(sdet_j)) // 2

    @staticmethod
    def exc_degree(det_i: Determinant, det_j: Determinant) -> Tuple[int, int]:
        """Compute excitation degree; the number of orbitals which differ between two |Determinants| det_i, det_J
        >>> Determinant_tuple.exc_degree(Determinant_tuple(alpha=(0, 1), beta=(0, 1)), Determinant_tuple(alpha=(0, 2), beta=(4, 6)))
        (1, 2)
        >>> Determinant_tuple.exc_degree(Determinant_tuple(alpha=(0, 1), beta=(0, 1)),Determinant_tuple(alpha=(0, 1), beta=(4, 6)))
        (0, 2)
        """
        ed_up = Determinant_tuple.exc_degree_spindet(det_i.alpha, det_j.alpha)
        ed_dn = Determinant_tuple.exc_degree_spindet(det_i.beta, det_j.beta)
        return (ed_up, ed_dn)

    @staticmethod
    def is_connected(det_i: Determinant, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ between the two determinants.
        Return bool; `Are det_i and det_j (singley or doubley) connected?
        >>> Determinant_tuple.is_connected(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                         Determinant(alpha=(0, 1), beta=(0, 2)))
        True
        >>> Determinant_tuple.is_connected(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                         Determinant(alpha=(0, 2), beta=(0, 2)))
        True
        >>> Determinant_tuple.is_connected(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                         Determinant(alpha=(2, 3), beta=(0, 1)))
        True
        >>> Determinant_tuple.is_connected(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                         Determinant(alpha=(2, 3), beta=(0, 2)))
        False
        >>> Determinant_tuple.is_connected(Determinant(alpha=(0, 1), beta=(0, 1)),
        ...                         Determinant(alpha=(0, 1), beta=(0, 1)))
        False
        """
        return sum(Determinant_tuple.exc_degree(det_i, det_j)) in [1, 2]


class Determinant_bitstring(Determinant):
    """Slater determinant: Product of 2 spin determinants.
    One for $\alpha$ electrons and one for \beta electrons.

    Internal representation as a binary string of |int|, i.e. `Occupation number' (ON) representation
    Uses convention that rightmost bit is most significant (corresponds to orbital 0)"""

    alpha: Spin_determinant_bitstring
    beta: Spin_determinant_bitstring

    # No. of 64-bit integers required to store ON representation of |Bitstring|
    # self.Nint = -(Norb // -64)

    def to_tuple(self):
        """Conver |Determinant_bitstring| to |Determinant_tuple| representation
        >>> Determinant_bitstring(0b0101, 0b0001).to_tuple()
        Determinant_tuple(alpha=(0, 2), beta=(0,))
        >>> Determinant_bitstring(0b0101, 0b0000).to_tuple()
        Determinant_tuple(alpha=(0, 2), beta=())
        """

        # Run through spin-bits right -> left, for each |Spin_determinant_bitstring|
        # `0b` at end of bitstring is skipped due to if
        alpha_tup = tuple([i for i, on in enumerate(bin(self.alpha)[::-1]) if on == "1"])
        beta_tup = tuple([i for i, on in enumerate(bin(self.beta)[::-1]) if on == "1"])

        # Return |Determinant_tuple| representation of |Determinant_Bitstring|
        return Determinant_tuple(alpha_tup, beta_tup)


Psi_det = List[Determinant]
Psi_coef = List[float]
# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det
Energy = NewType("Energy", float)
