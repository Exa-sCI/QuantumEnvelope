# For forward declaration in type hints
from __future__ import annotations

from typing import Tuple, Dict, NamedTuple, List, NewType, Iterator
from itertools import chain, product, combinations, takewhile
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

# #     _
# #    |_) o _|_  _  _ _|_
# #    |_) |  |_ _> (/_ |_
# #

# class Spin_determinant_bitset:
#     def __init__(self, int):
#         # TODO: Define these things
#         self.tag = qelib.SPIN_DET_TYPE_BITSET
#         npa = np.array(sorted(t), dtype=np.intc)
#         npa_size = len(t)
#         self.handle = qelib.qe_spin_det_bitset_create(npa, npa_size)

#    ___
#     |     ._  |  _
#     | |_| |_) | (/_
#           |


class Spin_determinant_tuple(Tuple[OrbitalIdx, ...]):
    """NewType for |Spin_determinant| as tuple of occupied orbital indices
    Certain logical operations overloaded as set operations
    """

    def convert_repr(self, Norb):
        """Convert |Spin_determinant_tuple| to |Spin_determinant_bitstring| (occupation number) representation
        >>> bin(Spin_determinant_tuple((0, 2)).convert_repr(4))
        '0b101'
        >>> bin(Spin_determinant_tuple(()).convert_repr(4))
        '0b0'
        >>> type(Spin_determinant_tuple((0, 2)).convert_repr(4))
        <class 'fundamental_types.Spin_determinant_bitstring'>
        """
        # Strings immutable -> Work with lists for ON assignment, convert to |str| when done
        bitstr = ["0", "b"]
        bitstr.extend(["0"] * Norb)
        for o in self:
            bitstr[-(o + 1)] = "1"

        # Return |Bitstring| representation of this |Determinant|
        return Spin_determinant_bitstring(int(("".join(bitstr)), 2))

    # Use iterator methods inherent to |Tuple|; no need to overload as in bitset representation

    def __and__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Overload `&` operator to perform set intersection
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((0, 2))
        (0,)
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((2, 3))
        ()
        """
        return Spin_determinant_tuple(sorted(set(self) & set(s_tuple)))

    def __rand__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Reverse overloaded __and__
        >>> (0, 1) & Spin_determinant_tuple((0, 2))
        (0,)
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((0, 2))
        (0,)
        """
        return self.__and__(s_tuple)

    def __or__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Overload `|` operator to perform set union
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((0, 2))
        (0, 1, 2)
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((0, 1))
        (0, 1)
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((2, 3))
        (0, 1, 2, 3)
        """
        return Spin_determinant_tuple(sorted(set(self) | set(s_tuple)))

    def __ror__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Reverse overloaded __or__
        >>> (0, 1) | Spin_determinant_tuple((0, 2))
        (0, 1, 2)
        """
        return self.__or__(s_tuple)

    def __xor__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Overload `^` operator to perform symmetric set difference
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((0, 2))
        (1, 2)
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((0, 1))
        ()
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((2, 3))
        (0, 1, 2, 3)
        """
        return Spin_determinant_tuple(sorted(set(self) ^ set(s_tuple)))

    def __rxor__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Reverse overloaded __xor__
        >>> (0, 1) ^  Spin_determinant_tuple((0, 2))
        (1, 2)
        """
        return self.__xor__(s_tuple)

    def __sub__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Overload `-` operator to perform set difference
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) - Spin_determinant_tuple((0, 2))
        (1,)
        >>> Spin_determinant_tuple((0, 2)) - Spin_determinant_tuple((0, 1))
        (2,)
        >>> Spin_determinant_tuple((0, 1)) - Spin_determinant_tuple((0, 1))
        ()
        """
        return Spin_determinant_tuple(sorted(set(self) - set(s_tuple)))

    def __rsub__(self, s_tuple: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Reverse overloaded __sub__
        Convert arg `s_tuple' to |Spin_determinant_tuple|, then perform __sub__ since operation is not communative
        >>> (0, 1, 2, 3) - Spin_determinant_tuple((0, 2))
        (1, 3)
        """
        return Spin_determinant_tuple(s_tuple).__sub__(self)

    def popcnt(self) -> int:
        """Perform a `popcount'; return length of the tuple"""
        return int(len(self))

    #     _
    #    |_     _ o _|_  _. _|_ o  _  ._
    #    |_ >< (_ |  |_ (_|  |_ | (_) | |
    #

    def apply_single_excitation(
        self, hole: OrbitalIdx, particle: OrbitalIdx
    ) -> Spin_determinant_tuple:
        """Apply single hole -> particle excitation to instance of |Spin_determinant|
        >>> Spin_determinant_tuple((0, 1)).apply_single_excitation(0, 2)
        (1, 2)
        >>> Spin_determinant_tuple((1, 4)).apply_single_excitation(4, 0)
        (0, 1)
        >>> Spin_determinant_tuple((0, 1, 3, 4, 7, 10)).apply_single_excitation(4, 8)
        (0, 1, 3, 7, 8, 10)
        >>> Spin_determinant_tuple((0, 1, 3, 4, 7, 10)).apply_single_excitation(4, 5)
        (0, 1, 3, 5, 7, 10)
        >>> Spin_determinant_tuple((0, 1, 3, 4, 7, 10)).apply_single_excitation(7, 2)
        (0, 1, 2, 3, 4, 10)
        >>> Spin_determinant_tuple((0, 1, 3, 4, 7, 10)).apply_single_excitation(7, 5)
        (0, 1, 3, 4, 5, 10)
        """
        hole_index = self.index(hole)
        # Where to put the particle so things are sorted?
        particle_index = 0
        while (particle_index < self.popcnt() - 1) & (self[particle_index] < particle):
            particle_index += 1
        # Upon return, either i == to self.popcnt() -1, or self[particle_index] > particle
        # print(particle, self[hole_index + 1])
        if hole_index < particle_index:
            if self[particle_index] > particle:
                # Place partice before self[particle_index] in new tuple
                t = (
                    self[:hole_index]
                    + self[hole_index + 1 : particle_index]
                    + (particle,)
                    + self[particle_index:]
                )
            else:
                # Here, self[particle_index] < particle, so place particle after
                t = (
                    self[:hole_index]
                    + self[hole_index + 1 : particle_index + 1]
                    + (particle,)
                    + self[particle_index + 1 :]
                )
        else:
            t = (
                self[:particle_index]
                + (particle,)
                + self[particle_index:hole_index]
                + self[hole_index + 1 :]
            )
        return Spin_determinant_tuple(t)

    def apply_double_excitation(
        self, h1: OrbitalIdx, p1: OrbitalIdx, h2: OrbitalIdx, p2: OrbitalIdx
    ) -> Spin_determinant_tuple:
        """Apply double h1 -> p1, h2 -> p2 excitation to instance of |Spin_determinant|
        >>> Spin_determinant_tuple((0, 1)).apply_double_excitation(0, 2, 1, 3)
        (2, 3)
        >>> Spin_determinant_tuple((0, 4)).apply_double_excitation(4, 1, 0, 2)
        (1, 2)
        >>> Spin_determinant_tuple((0, 1, 3, 4, 7, 10)).apply_double_excitation(4, 8, 10, 2)
        (0, 1, 2, 3, 7, 8)
        """
        temp = self.apply_single_excitation(h1, p1)
        return temp.apply_single_excitation(h2, p2)

    def exc_degree_spindet(self, right: Spin_determinant_tuple) -> int:
        """Return excitation degree between two |Spin_determinant|
        >>> Spin_determinant_tuple((0, 1)).exc_degree_spindet((1, 2))
        1
        >>> Spin_determinant_tuple((0, 1)).exc_degree_spindet((2, 3))
        2
        >>> Spin_determinant_tuple((0, 1)).exc_degree_spindet((0, 1))
        0
        """
        return ((self ^ right).popcnt()) // 2

    def gen_all_connected_spindet(self, ed: int, n_orb: int) -> Iterator[Spin_determinant_tuple]:
        """Generate all connected spin determinants to self relative to a particular excitation degree
        :param n_orb: global parameter
        >>> sorted(Spin_determinant_tuple((0, 1)).gen_all_connected_spindet(1, 4))
        [(0, 2), (0, 3), (1, 2), (1, 3)]
        >>> sorted(Spin_determinant_tuple((0, 1)).gen_all_connected_spindet(2, 4))
        [(2, 3)]
        >>> sorted(Spin_determinant_tuple((0, 1)).gen_all_connected_spindet(2, 2))
        []
        """
        # Compute all possible holes (occupied orbitals in self) and particles (empty orbitals in self)
        holes = combinations(self, ed)
        particles = combinations(Spin_determinant_tuple(range(n_orb)) - self, ed)
        l_hp_pairs = product(holes, particles)

        return [self ^ tuple((set(h) | set(p))) for h, p in l_hp_pairs]

    #     _
    #    |_) |_   _.  _  _      |_|  _  |  _
    #    |   | | (_| _> (/_ o   | | (_) | (/_
    #                   _   /
    #     _. ._   _|   |_) _. ._ _|_ o  _ |  _
    #    (_| | | (_|   |  (_| |   |_ | (_ | (/_

    # Driver functions for computing phase, hole and particle between determinant pairs

    def get_holes(self, right: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Get holes involved in excitation between two |Spin_determinant|
        >>> Spin_determinant_tuple((0, 1)).get_holes((1, 2))
        (0,)
        >>> Spin_determinant_tuple((0, 1)).get_holes((2, 3))
        (0, 1)
        >>> Spin_determinant_tuple((0, 1)).get_holes((0, 1))
        ()
        """
        return (self ^ right) & self

    def get_particles(self, right: Spin_determinant_tuple) -> Spin_determinant_tuple:
        """Get particles involved in excitation between two |Spin_determinant|
        >>> Spin_determinant_tuple((0, 1)).get_particles((1, 2))
        (2,)
        >>> Spin_determinant_tuple((0, 1)).get_particles((2, 3))
        (2, 3)
        >>> Spin_determinant_tuple((0, 1)).get_particles((0, 1))
        ()
        """
        return (self ^ right) & right

    def single_phase(
        self,
        h: OrbitalIdx,
        p: OrbitalIdx,
    ):
        """Function to compute phase for <I|H|J> when I and J differ by exactly one orbital h <-> pd

        >>> Spin_determinant_tuple((0, 4, 6)).single_phase(4, 5)
        1
        >>> Spin_determinant_tuple((0, 1, 8)).single_phase(1, 17)
        -1
        >>> Spin_determinant_tuple((0, 1, 4, 8)).single_phase(1, 17)
        1
        >>> Spin_determinant_tuple((0, 1, 4, 7, 8)).single_phase(1, 17)
        -1
        """
        # Naive; compute phase for |Spin_determinant| pairs related by excitataion from h <-> p
        j, k = min(h, p), max(h, p)
        pmask = tuple((i for i in range(j + 1, k)))
        parity = (self & pmask).popcnt() % 2
        return -1 if parity else 1

    def double_phase(self, h1: OrbitalIdx, p1: OrbitalIdx, h2: OrbitalIdx, p2: OrbitalIdx):
        """Function to compute phase for <I|H|J> when I and J differ by exactly two orbitals h1, h2 <-> p1, p2
        Only for same spin double excitations
        h1, h2 is occupied in self, p1, p2 is unoccupied
        >>> Spin_determinant_tuple((0, 1, 2, 3, 4, 5, 6, 7, 8)).double_phase(2, 11, 3, 12)
        1
        >>> Spin_determinant_tuple((0, 1, 2, 3, 4, 5, 6, 7, 8)).double_phase(2, 11, 8, 17)
        -1
        """
        # TODO: NOTE, elsewhere in qe.drivers code, will have to switch hole particle order in args!
        # Compute phase. Loopless as in https://arxiv.org/abs/1311.6244
        phase = self.single_phase(h1, p1) * self.single_phase(h2, p2)
        # if max(h1, p1) > min(h2, p2):
        #     return -phase
        # else:
        #     return phase
        if h2 < p1:
            phase *= -1
        if p2 < h1:
            phase *= -1
        return phase

    def single_exc(self, right: Spin_determinant_tuple) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Spin_determinant_tuple((0, 4, 6)).single_exc((0, 5, 6))
        (1, 4, 5)
        >>> Spin_determinant_tuple((0, 4, 6)).single_exc((0, 22, 6))
        (-1, 4, 22)
        >>> Spin_determinant_tuple((0, 1, 8)).single_exc((0, 8, 17))
        (-1, 1, 17)
        """
        # Get holes, particle in exc

        (h,) = self.get_holes(right)
        (p,) = self.get_particles(right)

        return self.single_phase(h, p), h, p

    def double_exc(
        self, right: Spin_determinant_tuple
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Spin_determinant_tuple((0, 1, 2, 3, 4, 5, 6, 7, 8)).double_exc((0, 1, 4, 5, 6, 7, 8, 11, 12))
        (1, 2, 3, 11, 12)
        >>> Spin_determinant_tuple((0, 1, 2, 3, 4, 5, 6, 7, 8)).double_exc((0, 1, 3, 4, 5, 6, 7, 11, 17))
        (-1, 2, 8, 11, 17)
        """
        # Holes
        h1, h2 = self.get_holes(right)
        # Particles
        p1, p2 = self.get_particles(right)

        return self.double_phase(h1, p1, h2, p2), h1, h2, p1, p2


#     _
#    |_) o _|_  _ _|_ ._ o ._   _
#    |_) |  |_ _>  |_ |  | | | (_|
#                               _|


class Spin_determinant_bitstring(int):
    """NewType for |Spin_determinant| represented as integer bitstring
    Occupation number (ON) representation of determinants; most significant bits are rightmost
        e.g., occupation of OrbitalIdx = 0 is given by rightmost bit
    Certain bitwise logical operators overloaded.
    """

    def convert_repr(self, Norb=None):
        """Conver |Spin_determinant_bitstring| to |Spin_determinant_tuple| representation
        Norb is global param; needed for |Spin_determinant_tuple|, so pass as dummy arg
        >>> Spin_determinant_bitstring(0b0101).convert_repr()
        (0, 2)
        >>> Spin_determinant_bitstring(0b0000).convert_repr()
        ()
        >>> type(Spin_determinant_bitstring(0b0101).convert_repr())
        <class 'fundamental_types.Spin_determinant_tuple'>
        """
        # Run through spin-bits right -> left
        # `0b` at end of bitstring is skipped due to if
        tuple_rep = tuple([i for i, on in enumerate(bin(self)[::-1]) if on == "1"])

        # Return |Spin_determinant_tuple| representation of |Spin_determinant_bitstring|
        return Spin_determinant_tuple(tuple_rep)

    @staticmethod
    def create_bitmask(tuple_mask: Tuple[OrbitalIdx, ...]) -> int:
        """Create bitmask from tuple of integers `tuple_mask'
        >>> bin(Spin_determinant_bitstring.create_bitmask((0, 1, 7, 8)))
        '0b110000011'
        """
        bitstring_mask = 0b0
        for indice in tuple_mask:
            # Create mask with bits in lh, lp set to 1
            bitstring_mask = bitstring_mask | (1 << indice)
        return bitstring_mask

    def __and__(self, mask: int or Tuple[OrbitalIdx, ...]) -> int:
        """Overload `&` operator to hole-particle mask as |tuple| or |int| and return |int|
        >>> bin(Spin_determinant_bitstring(0b1010) & 0b0110)
        '0b10'
        >>> bin(Spin_determinant_bitstring(0b0101) & 0b1111)
        '0b101'
        >>> bin(Spin_determinant_bitstring(0b1010) & 0b0)
        '0b0'

        >>> bin(Spin_determinant_bitstring(0b1010) & (1, 2))
        '0b10'
        >>> bin(Spin_determinant_bitstring(0b0101) & (0, 1, 2, 3))
        '0b101'
        >>> bin(Spin_determinant_bitstring(0b1010) & ())
        '0b0'

        >>> isinstance((Spin_determinant_bitstring(0b1010) & 0b0110), int)
        True
        >>> isinstance((Spin_determinant_bitstring(0b1010) & 0b0110), Spin_determinant_bitstring)
        True
        """
        # Perform correct operation based on input
        if isinstance(mask, int):
            # Operator overloaded function on first operand
            # In case that first argument is also |Spin_determinant bitstring|, convert to int
            # Else, infinite recursion
            return Spin_determinant_bitstring(int(mask) & self)
        elif isinstance(mask, tuple):
            # Create mask with bits in lh, lp set to 1
            bitstring_mask = self.create_bitmask(mask)
            return Spin_determinant_bitstring(self & bitstring_mask)
        else:
            raise TypeError(f"Unsupported operand type(s) for &: '{type(self)}' and '{type(mask)}'")

    def __or__(self, mask: int or Tuple[OrbitalIdx, ...]) -> int:
        """Overload `|` operator to hole-particle mask as |tuple| or |int| and return |int|
        >>> bin(Spin_determinant_bitstring(0b1010) | 0b0110)
        '0b1110'
        >>> bin(Spin_determinant_bitstring(0b0101) | 0b1111)
        '0b1111'
        >>> bin(Spin_determinant_bitstring(0b1010) | 0b0)
        '0b1010'

        >>> bin(Spin_determinant_bitstring(0b1010) | (1, 2))
        '0b1110'
        >>> bin(Spin_determinant_bitstring(0b0101) | (0, 1, 2, 3))
        '0b1111'
        >>> bin(Spin_determinant_bitstring(0b1010) | ())
        '0b1010'

        >>> isinstance((Spin_determinant_bitstring(0b1010) | 0b0110), int)
        True
        >>> isinstance((Spin_determinant_bitstring(0b1010) | 0b0110), Spin_determinant_bitstring)
        True
        """
        # Perform correct operation based on input
        if isinstance(mask, int):
            # Operator overloaded function on first operand
            # In case that first argument is also |Spin_determinant bitstring|, convert to int
            # Else, infinite recursion
            return Spin_determinant_bitstring(int(mask) | self)
        elif isinstance(mask, tuple):
            # Create mask with bits in lh, lp set to 1
            bitstring_mask = self.create_bitmask(mask)
            return Spin_determinant_bitstring(self | bitstring_mask)
        else:
            raise TypeError(f"Unsupported operand type(s) for |: '{type(self)}' and '{type(mask)}'")

    def __xor__(self, mask: int or Tuple[OrbitalIdx, ...]) -> int:
        """Overload `^` operator to hole-particle mask as |tuple| or |int| and return |int|
        >>> bin(Spin_determinant_bitstring(0b1010) ^ 0b0110)
        '0b1100'
        >>> bin(Spin_determinant_bitstring(0b0101) ^ 0b1111)
        '0b1010'
        >>> bin(Spin_determinant_bitstring(0b1010) ^ 0b0)
        '0b1010'

        >>> bin(Spin_determinant_bitstring(0b1010) ^ (1, 2))
        '0b1100'
        >>> bin(Spin_determinant_bitstring(0b0101) ^ (0, 1, 2, 3))
        '0b1010'
        >>> bin(Spin_determinant_bitstring(0b1010) ^ ())
        '0b1010'

        >>> isinstance((Spin_determinant_bitstring(0b1010) ^ 0b0110), int)
        True
        >>> isinstance((Spin_determinant_bitstring(0b1010) ^ 0b0110), Spin_determinant_bitstring)
        True
        """
        # Perform correct operation based on input
        if isinstance(mask, int):
            # Operator overloaded function on first operand
            # In case that first argument is also |Spin_determinant bitstring|, convert to int
            # Else, infinite recursion
            return Spin_determinant_bitstring(int(mask) ^ self)
        elif isinstance(mask, tuple):
            # Create mask with bits in lh, lp set to 1
            bitstring_mask = self.create_bitmask(mask)
            return Spin_determinant_bitstring(self ^ bitstring_mask)
        else:
            raise TypeError(f"Unsupported operand type(s) for ^: '{type(self)}' and '{type(mask)}'")

    def __sub__(self, spin_bs: int) -> int:
        """Overload `-` operator to perform logical bitwise comparison
        Remove common bits between `self` and `spin_bs` -> (self) & ~(spin_bs)
        >>> bin(Spin_determinant_bitstring(0b1010) - Spin_determinant_bitstring(0b0011))
        '0b1000'
        >>> bin(Spin_determinant_bitstring(0b0101) - Spin_determinant_bitstring(0b0101))
        '0b0'
        >>> bin(Spin_determinant_bitstring(0b1010) - Spin_determinant_bitstring(0b0101))
        '0b1010'
        """
        return Spin_determinant_bitstring(self & ~(spin_bs))

    def __rsub__(self, mask: int or Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx]:
        """Reverse overloaded __sub__
        Convert arg `mask', then perform __sub__ since operation is not communative
        Take hole-particle mask as |tuple| or |int| and return |int|
        >>> bin(0b1111 - Spin_determinant_bitstring(0b0101))
        '0b1010'
        >>> bin((0, 1, 2, 3) -  Spin_determinant_bitstring(0b0101))
        '0b1010'
        >>> bin(0b1010 - Spin_determinant_bitstring(0b0101))
        '0b1010'
        >>> bin((0, 1, 2, 3) - Spin_determinant_bitstring(0b0101))
        '0b1010'
        """
        # Perform correct operation based on input
        if isinstance(mask, int):
            # Operator overloaded function on first operand
            return Spin_determinant_bitstring(mask) - self
        elif isinstance(mask, tuple):
            # Create mask with bits in lh, lp set to 1
            bitstring_mask = self.create_bitmask(mask)
            return Spin_determinant_bitstring(bitstring_mask) - self
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(mask)}'")

    def popcnt(self) -> int:
        """Perform a `popcount'; number of bits set to True in self"""
        return self.bit_count()


#   ______     _                      _                   _
#   |  _  \   | |                    (_)                 | |
#   | | | |___| |_ ___ _ __ _ __ ___  _ _ __   __ _ _ __ | |_
#   | | | / _ \ __/ _ \ '__| '_ ` _ \| | '_ \ / _` | '_ \| __|
#   | |/ /  __/ ||  __/ |  | | | | | | | | | | (_| | | | | |_
#   |___/ \___|\__\___|_|  |_| |_| |_|_|_| |_|\__,_|_| |_|\__|
#
#


class Determinant:
    """Generic Slater determinant claass: Product of 2 determinants.
    One for $\alpha$ electrons and one for \beta electrons."""

    def __init__(self, alpha, beta, representation="tuple"):
        self.flag = representation
        if self.flag == "tuple":
            self.alpha = Spin_determinant_tuple(alpha)
            self.beta = Spin_determinant_tuple(beta)
        else:
            raise NotImplementedError

    def __repr__(self):
        """Return a nicely formatted representation string"""
        return "Determinant(alpha=%r, beta=%r)" % (self.alpha, self.beta)

    def __eq__(self, right: Determinant):
        """Comparison operator"""
        if (self.alpha == right.alpha) & (self.beta == right.beta):
            return True
        else:
            return False

    def __hash__(self):
        """Hash custom Det"""
        return hash((self.alpha, self.beta))

    def __iter__(self):
        """Unpack"""
        return iter((self.alpha, self.beta))

    #     _
    #    |_     _ o _|_  _. _|_ o  _  ._
    #    |_ >< (_ |  |_ (_|  |_ | (_) | |
    #

    def apply_single_excitation(self, h: OrbitalIdx, p: OrbitalIdx, spin: str) -> Determinant:
        """Apply single excitation to self, return new abstract |Determinant| class
        Each fundamental type has own implementation of spindet excitaations
        Inputs
            :param h, p: Specifies hole, particle involved in excitation of |Spin_determinant|
            :param spin: Spin-type of excitation, "alpha" or "beta"

        >>> Determinant((0, 1), (0, 1), "tuple").apply_single_excitation(1, 2, "alpha")
        Determinant(alpha=(0, 2), beta=(0, 1))
        >>> Determinant((0, 1, 2), (0, 1), "tuple").apply_single_excitation(0, 4, "alpha")
        Determinant(alpha=(1, 2, 4), beta=(0, 1))
        >>> Determinant((0, 1), (0, 1), "tuple").apply_single_excitation(0, 2, "beta")
        Determinant(alpha=(0, 1), beta=(1, 2))
        """
        if spin == "alpha":
            return Determinant(self.alpha.apply_single_excitation(h, p), self.beta, self.flag)
        elif spin == "beta":
            return Determinant(self.alpha, self.beta.apply_single_excitation(h, p), self.flag)
        else:
            raise NotImplementedError

    def apply_same_spin_double_excitation(
        self, h1: OrbitalIdx, p1: OrbitalIdx, h2: OrbitalIdx, p2: OrbitalIdx, spin: str
    ) -> Determinant:
        """Apply double excitation to self, same-spin, return new abstract |Determinant| class
        Each fundamental type has own implementation of spindet excitaations
        Inputs
            :param h1, p1, h2, p2:  Specifies holes, particles involved in excitation of |Spin_determinant|
            :param spin:            Spin-type of excitation, "alpha" or "beta"


        >>> Determinant((0, 1), (0, 1), "tuple").apply_same_spin_double_excitation(0, 3, 1, 2, "alpha")
        Determinant(alpha=(2, 3), beta=(0, 1))
        >>> Determinant((0, 1, 3), (0, 1), "tuple").apply_same_spin_double_excitation(3, 2, 1, 4, "alpha")
        Determinant(alpha=(0, 2, 4), beta=(0, 1))
        >>> Determinant((0, 1), (0, 1), "tuple").apply_same_spin_double_excitation(0, 2, 1, 3, "beta")
        Determinant(alpha=(0, 1), beta=(2, 3))
        """
        if spin == "alpha":
            return Determinant(
                self.alpha.apply_double_excitation(h1, p1, h2, p2), self.beta, self.flag
            )
        elif spin == "beta":
            return Determinant(
                self.alpha, self.beta.apply_double_excitation(h1, p1, h2, p2), self.flag
            )
        else:
            raise NotImplementedError

    def apply_opposite_spin_double_excitation(
        self, h1: OrbitalIdx, p1: OrbitalIdx, h2: OrbitalIdx, p2: OrbitalIdx
    ) -> Determinant:
        """Apply double excitation to self, opposite-spin, return new abstract |Determinant| class
        Each fundamental type has own implementation of spindet excitaations
        Inputs
            :param h1, p1, h2, p2:  Specifies holes, particles involved in excitation. Assumed that h1, p1 (h2, p2) -> alpha exc (beta exc)

        >>> Determinant((0, 1), (0, 1), "tuple").apply_opposite_spin_double_excitation(0, 3, 1, 2)
        Determinant(alpha=(1, 3), beta=(0, 2))
        >>> Determinant((0, 1, 3), (0, 1), "tuple").apply_opposite_spin_double_excitation(3, 2, 1, 4)
        Determinant(alpha=(0, 1, 2), beta=(0, 4))
        """
        return Determinant(
            self.alpha.apply_single_excitation(h1, p1),
            self.beta.apply_single_excitation(h2, p2),
            self.flag,
        )

    def exc_degree(self, right: Determinant) -> int:
        """Compute excitation degree; the number of orbitals which differ between two |Determinants| self, det_J

        >>> Determinant((0, 1), (0, 1), "tuple").exc_degree(Determinant((0, 2), (4, 6), "tuple"))
        (1, 2)
        >>> Determinant((0, 1), (0, 1), "tuple").exc_degree(Determinant((0, 1), (4, 6), "tuple"))
        (0, 2)
        """
        ed_up = (self.alpha ^ right.alpha).popcnt() // 2
        ed_dn = (self.beta ^ right.beta).popcnt() // 2
        return (ed_up, ed_dn)

    def is_connected(self, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ between the two determinants.
        Return bool; `Is det_j (singley or doubley) connected to instance of self?

        >>> Determinant((0, 1), (0, 1), "tuple").is_connected(Determinant((0, 1), (0, 2), "tuple"))
        True
        >>> Determinant((0, 1), (0, 1), "tuple").is_connected(Determinant((0, 2), (0, 2), "tuple"))
        True
        >>> Determinant((0, 1), (0, 1), "tuple").is_connected(Determinant((2, 3), (0, 1), "tuple"))
        True
        >>> Determinant((0, 1), (0, 1), "tuple").is_connected(Determinant((2, 3), (0, 2), "tuple"))
        False
        >>> Determinant((0, 1), (0, 1), "tuple").is_connected(Determinant((0, 1), (0, 1), "tuple"))
        False
        """
        return sum(self.exc_degree(det_j)) in [1, 2]

    def gen_all_connected_det(self, n_orb: int) -> Iterator[Determinant]:
        """Generate all determinants that are singly or doubly connected to self
        :param n_orb: global parameter, needed to cap possible excitations

        >>> [d for d in (Determinant((0, 1), (0,), "tuple").gen_all_connected_det(3))]
        [Determinant(alpha=(0, 2), beta=(0,)),
        Determinant(alpha=(1, 2), beta=(0,)),
        Determinant(alpha=(0, 1), beta=(1,)),
        Determinant(alpha=(0, 1), beta=(2,)),
        Determinant(alpha=(0, 2), beta=(1,)),
        Determinant(alpha=(0, 2), beta=(2,)),
        Determinant(alpha=(1, 2), beta=(1,)),
        Determinant(alpha=(1, 2), beta=(2,))]
        """

        # Generate all singles from constituent alpha and beta spin determinants
        # Then, opposite-spin and same-spin doubles

        # We use l_single_a, and l_single_b twice. So we store them.
        l_single_a = set(self.alpha.gen_all_connected_spindet(1, n_orb))
        l_double_aa = self.alpha.gen_all_connected_spindet(2, n_orb)

        # Singles and doubles; alpha spin
        exc_a = (
            Determinant(det_alpha, self.beta, self.flag)
            for det_alpha in chain(l_single_a, l_double_aa)
        )

        l_single_b = set(self.beta.gen_all_connected_spindet(1, n_orb))
        l_double_bb = self.beta.gen_all_connected_spindet(2, n_orb)

        # Singles and doubles; beta spin
        exc_b = (
            Determinant(self.alpha, det_beta, self.flag)
            for det_beta in chain(l_single_b, l_double_bb)
        )

        l_double_ab = product(l_single_a, l_single_b)

        # Doubles; opposite-spin
        exc_ab = (
            Determinant(det_alpha, det_beta, self.flag) for det_alpha, det_beta in l_double_ab
        )

        return chain(exc_a, exc_b, exc_ab)

    def triplet_constrained_single_excitations_from_det(
        self, constraint: Tuple[OrbitalIdx, ...], n_orb: int, spin="alpha"
    ) -> Iterator[Determinant]:
        """Called by inherited classes; Generate singlet excitations from constraint
        :param spin: Refers to the spin type of `constraint`"""

        ha, pa, hb, pb = self.get_holes_particles_for_constrained_singles(constraint, n_orb, spin)
        # Excitations of argument `spin`
        for h, p in product(ha, pa):
            if spin == "alpha":
                # Then, det_a is alpha spindet
                excited_det = self.apply_single_excitation(h, p, "alpha")
            else:
                # det_a is beta spindet
                excited_det = self.apply_single_excitation(h, p, "beta")
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Generate opposite-spin excitations
        for h, p in product(hb, pb):
            if spin == "alpha":
                # Then, det_b is beta spindet
                excited_det = self.apply_single_excitation(h, p, "beta")
            else:
                # det_b is alpha spindet
                excited_det = self.apply_single_excitation(h, p, "alpha")
            # TODO: Assertion for bitstring? Shouldn't need though
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

    def triplet_constrained_double_excitations_from_det(
        self, constraint: Tuple[OrbitalIdx, ...], n_orb: int, spin="alpha"
    ) -> Iterator[Determinant]:
        """Called by inherited classes; Generate singlet excitations from constraint
        :param spin: Refers to the spin type of `constraint`"""

        # |Determinant_tuple| and |Determinant_bitstring| each have this method
        haa, paa, hbb, pbb, hab, pab = self.get_holes_particles_for_constrained_doubles(
            constraint, n_orb, spin
        )
        # Excitations of argument `spin`
        # Same-spin doubles, for argument `spin`
        for (h1, h2), (p1, p2) in product(haa, paa):
            if spin == "alpha":
                # Then, det_a is alpha spindet
                excited_det = self.apply_same_spin_double_excitation(h1, p1, h2, p2, "alpha")
            else:
                # det_a is beta spindet
                excited_det = self.apply_same_spin_double_excitation(h1, p1, h2, p2, "beta")
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Same-spin doubles, for opposite-spin to `spin`
        for (h1, h2), (p1, p2) in product(hbb, pbb):
            if spin == "alpha":
                # Then, det_b is beta spindet
                excited_det = self.apply_same_spin_double_excitation(h1, p1, h2, p2, "beta")
            else:
                # det_b is alpha spindet
                excited_det = self.apply_same_spin_double_excitation(h1, p1, h2, p2, "alpha")
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Opposite-spin doubles
        for holes, particles in product(hab, pab):
            ha, hb = holes
            pa, pb = particles
            if spin == "alpha":
                # det_a is alpha, det_b is beta
                excited_det = self.apply_opposite_spin_double_excitation(ha, pa, hb, pb)
            else:
                # det_a is beta, det_b is beta
                excited_det = self.apply_opposite_spin_double_excitation(hb, pb, ha, pa)
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

    def get_holes_particles_for_constrained_singles(
        self, constraint: Tuple[OrbitalIdx, ...], n_orb: int, spin="alpha"
    ) -> Tuple[List[OrbitalIdx], List[OrbitalIdx], List[OrbitalIdx], List[OrbitalIdx]]:
        """Get all hole, particle pairs that produce a singlet excitation of |Determinant| (self) that satisfy triplet constraint.
        By default: constraint T specifies 3 `most highly` occupied alpha spin orbitals allowed in the generated excitation
            e.g., if exciting |D> does not yield |J> such that o1, o2, o3 are the `largest` occupied alpha orbitals in |J> -> Excitation not generated
        Inputs:
            :param constraint: Triplet constraint as |Spin_determinant_tuple| -> tuple of three highest occupied `spin` orbitals
                                    T = [o1: |OrbitalIdx|, o2: |OrbitalIdx|, o3: |OrbitalIdx|]
            :param n_orb:      Global parameter
            :param spin:       Spin-type of constraint (e.g., are the orbitial indices in T alpha or beta spin?)

        Outputs:
            ((ha, pa), (hb, pb)) -> hole-particle pairs that identify alpha (beta) excitations of self that satisfy T
        """
        ha = []  # `Occupied` alpha orbitals to loop over
        pa = []  # `Virtual`  "                         "
        hb = []  # `Occupied` beta orbitals to loop over
        pb = []  # `Virtual`  "                         "

        all_orbs = tuple(range(n_orb))
        a1 = min(constraint)  # Index of `smallest` occupied constraint orbital
        B = tuple(range(a1 + 1, n_orb))  # B: `Bitmask' -> |Determinant| {a1 + 1, ..., Norb - 1}
        if spin == "alpha":
            det_a = getattr(
                self, spin
            )  # Get |Spin_determinant| of inputted |Determinant|, |D> (default is alpha)
            det_b = getattr(self, "beta")
        else:
            det_a = getattr(self, spin)  # Get |Spin_determinant| of inputted |Determinant|, |D>
            det_b = getattr(self, "alpha")

        # Some things can be pre-computed:
        #   Which of the `constraint` (spin) orbitals {a1, a2, a3} are occupied in |D_a>? (If any)
        constraint_orbitals_occupied = det_a & constraint
        #   Which `higher-order` (spin) orbitals o >= a1 that are not {a1, a2, a3} are occupied in |D_a>? (If any)
        #   TODO: Different from Tubman paper, which has an error if I reada it correctly
        #   Equivalent to `det_a & B - constraint` in set operations
        nonconstrained_orbitals_occupied = (det_a & B).get_holes(constraint)

        # If no double excitation of |D> will produce |J> satisfying constraint
        if (
            constraint_orbitals_occupied.popcnt() == 1
            or nonconstrained_orbitals_occupied.popcnt() > 1
        ):
            # No single excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            # These are empty lists
            return (ha, pa, hb, pb)

        # Create list of possible `particles` s.to constraint
        if constraint_orbitals_occupied.popcnt() == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> -> A single (a) x_a \in ha to a_unocc is necessary to satisfy the constraint
            # A single (b) will still not satisfy constraint
            # Operation equivalent to `(det_a ^ constraint) & constraint `
            (a_unocc,) = det_a.get_particles(constraint)  # The 1 unoccupied constraint orbital
            pa.append(a_unocc)
        if constraint_orbitals_occupied.popcnt() == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any single x_a \in ha to w_a where w_a < a1 will satisfy constraint
            # Operation below is equivalent to all_orbs - det_al; get orbs that are unoccupied in alpha spindet
            det_unocc_a_orbs = det_a.get_particles(all_orbs)
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                pa.append(w_a)
            #   Any single x_b \in hb to w_b
            # Operation below is equivalent to all_orbs - det_b; get orbs that are unoccupied in beta spindet
            det_unocc_b_orbs = det_b.get_particles(all_orbs)
            for w_b in det_unocc_b_orbs:
                pb.append(w_b)

        # Create list of possible `holes` s.to constraint
        if nonconstrained_orbitals_occupied.popcnt() == 1:
            # x_a > a1 \in |D_a> with x_a \not\in {a1, a2, a3} -> A single (a) x_a to w_a \in pa is necessary to satisfy constraint
            # A single (b) will not satisfy
            (x_a,) = nonconstrained_orbitals_occupied  # Has length 1; unpack
            ha.append(x_a)
        elif nonconstrained_orbitals_occupied.popcnt() == 0:
            # No `higher` orbitals \not\in {a1, a2, a3} occupied in |D> ->
            #   A single (a) x_a to w_a \in pa, where x_a < a1 (so as not to ruin constraint)
            for x_a in takewhile(lambda x: x < a1, det_a):
                ha.append(x_a)
            #   A single (b) x_b to w_b \in pb
            for x_b in det_b:
                hb.append(x_b)

        return (ha, pa, hb, pb)

    def get_holes_particles_for_constrained_doubles(
        self, constraint: Tuple[OrbitalIdx], n_orb: int, spin="alpha"
    ) -> Tuple[
        List[OrbitalIdx],
        List[OrbitalIdx],
        List[OrbitalIdx],
        List[OrbitalIdx],
        List[OrbitalIdx],
        List[OrbitalIdx],
    ]:
        """Get all hole, particle pairs that produce a doublet excitation of |Determinant| (self) that satisfy triplet constraint.
        By default: constraint T specifies 3 `most highly` occupied alpha spin orbitals allowed in the generated excitation
            e.g., if exciting |D> does not yield |J> such that o1, o2, o3 are the `largest` occupied alpha orbitals in |J> -> Excitation not generated
        Inputs:
            :param constraint: Triplet constraint as |Spin_determinant_tuple| -> tuple of three highest occupied `spin` orbitals
                                    T = [o1: |OrbitalIdx|, o2: |OrbitalIdx|, o3: |OrbitalIdx|]
            :param n_orb:      Global parameter
            :param spin:       Spin-type of constraint (e.g., are the orbitial indices in T alpha or beta spin?)

        Outputs:
            ((haa, paa), (hbb, pbb), (hab, pab)) -> hole-particle pairs that identify same-spin alpha, same-spin beta, and opposite-spin
                                                    double excitations of self that satisfy T.
        """
        # Same-spin alpha
        haa = []  # `Occupied` orbitals to loop over
        paa = []  # `Virtual`  "                   "
        # Same-spin beta
        hbb = []
        pbb = []
        # Opposite spin
        hab = []
        pab = []

        all_orbs = tuple(range(n_orb))
        a1 = min(constraint)  # Index of `smallest` occupied alpha constraint orbital
        B = tuple(range(a1 + 1, n_orb))  # B: `Bitmask' -> |Determinant| {a1 + 1, ..., Norb - 1}
        if spin == "alpha":
            det_a = getattr(
                self, spin
            )  # Get |Spin_determinant| of inputted |Determinant|, |D> (default is alpha)
            det_b = getattr(self, "beta")
        else:
            det_a = getattr(self, spin)  # Get |Spin_determinant| of inputted |Determinant|, |D>
            det_b = getattr(self, "alpha")

        # Some things can be pre-computed:
        #   Which of the `constraint` (spin) orbitals {a1, a2, a3} are occupied in |D>? (If any)
        constraint_orbitals_occupied = det_a & constraint
        #   Which `higher-order`(spin) orbitals o >= a1 that are not {a1, a2, a3} are occupied in |D>? (If any)
        #   TODO: Different from Tubman paper, which has an error if I read it correctly...
        #   Equivalent to `det_a & B - constraint` in set operations
        nonconstrained_orbitals_occupied = (det_a & B).get_holes(constraint)

        # If this -> no double excitation of |D> will produce |J> satisfying constraint |T>
        if (
            constraint_orbitals_occupied.popcnt() == 0
            or nonconstrained_orbitals_occupied.popcnt() > 2
        ):
            # No double excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            # These are empty lists
            return (haa, paa, hbb, pbb, hab, pab)

        # Create list of possible `particles` s.to constraint
        if constraint_orbitals_occupied.popcnt() == 1:
            # (One constraint orbital occupied) e.g., a1 \in |D_a> -> A same-spin (aa) double to (x_a, y_a) \in haa to (a2, a3) is necessary
            # No same-spin (bb) or opposite-spin (ab) excitations will satisfy constraint
            # New particles -> a2, a3
            # Operation equivalent to `(det_a ^ constraint) & constraint `
            a_unocc_1, a_unocc_2 = det_a.get_particles(
                constraint
            )  # This set will have length 2; unpack
            paa.append((a_unocc_1, a_unocc_2))

        elif constraint_orbitals_occupied.popcnt() == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> ->
            #   A same-spin (aa) double (x_a, y_a) \in haa to (z_a, a_unocc), where z_a\not\in|D_a>, and z_a < a1 (if excited to z_a > a1, constraint ruined!)
            # Operation equivalent to `(det_a ^ constraint) & constraint `
            (a_unocc,) = det_a.get_particles(constraint)  # The 1 unoccupied constraint orbital
            # Operation below is equivalent to `all_orbs - det_a`
            det_unocc_a_orbs = det_a.get_particles(all_orbs)  # Unocc orbitals in |D_a>
            for z_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                # z < a_unocc trivially, no need to check they are distinct
                paa.append((z_a, a_unocc))
            #   No same spin (bb) excitations will satisfy constraint
            #   An oppopsite spin (ab) double (x_a, y_b) \in \hab to (a_unocc, z_b), where z\not\in|D_b>
            det_unocc_b_orbs = all_orbs - det_b  # Unocc orbitals in |D_b>
            for z_b in det_unocc_b_orbs:
                pab.append((a_unocc, z_b))

        elif constraint_orbitals_occupied.popcnt() == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any same-spin (aa) double (x_a, y_a) \in haa to (w_a, z_a), where w_a, z_a < a1
            # Operations below is equivalent to `all_orbs - det_a`
            det_unocc_a_orbs = det_a.get_particles(all_orbs)
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                for z_a in takewhile(lambda z: z < w_a, det_unocc_a_orbs):
                    paa.append((w_a, z_a))
            # Any same-spin (bb) double (x_b, y_b) \in hbb to (w_b, z_b)
            # Operations below is equivalent to `all_orbs - det_b`
            det_unocc_b_orbs = det_b.get_particles(all_orbs)  # Unocc orbitals in |D_b>
            for w_b in det_unocc_b_orbs:
                for z_b in takewhile(lambda x: x < w_b, det_unocc_b_orbs):
                    pbb.append((w_b, z_b))
            #   Any oppospite-spin (ab) double (x_a, y_b) \in hab to (w_a, z_b), where w_a < a1
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                for z_b in det_unocc_b_orbs:
                    pab.append((w_a, z_b))

        # Create list of possible `holes` s.to constraint
        if nonconstrained_orbitals_occupied.popcnt() == 2:
            # x_a, y_a \in |D_a> with x_a, y_a > a1 and \not\in {a1, a2, a3} -> A same-spin (aa) double (x_a, y_a) to (w_a, z_a) \in paa is necessary
            # No same-spin (bb) or opposite-spin (ab) excitations will satisfy constraint
            # New holes -> x, y
            x_a, y_a = nonconstrained_orbitals_occupied  # This set will have length 2; unpack
            haa.append((x_a, y_a))
        elif nonconstrained_orbitals_occupied.popcnt() == 1:
            # x_a > a1 \in |D_a> with x_a \not\in {a1, a2, a3} ->
            #   A same-spin (aa) double (x_a, y_a) to (w_a, z_a) \in paa, where y_a < a1 (exciting y_a < a1 doesn't remove constraint)
            (x_a,) = nonconstrained_orbitals_occupied  # Has length 1; unpack
            for y_a in takewhile(lambda x: x < a1, det_a):
                if x_a != y_a:
                    haa.append((x_a, y_a))
            #   A same-spin (bb) double will not satisfy the constraint
            #   A opposite-spin (ab) double (x_a, y_b) -> (w_a, z_b) \in pab where y_b \in |D_b>
            for y_b in det_b:
                hab.append((x_a, y_b))

        elif nonconstrained_orbitals_occupied.popcnt() == 0:
            # No `higher` orbitals \not\in {a1, a2, a3} occupied in |D> ->
            #   A same-spin (aa) double (x_a, y_a) to (w_a, z_a) \in paa, where x_a, y_a < a1
            for x_a in takewhile(lambda x: x < a1, det_a):
                for y_a in takewhile(lambda y: y < x_a, det_a):
                    haa.append((x_a, y_a))
            #   A same-spin (bb) double (x_b, y_b) to (w_b, z_b) \in pbb
            for x_b in det_b:
                for y_b in takewhile(lambda x: x < x_b, det_b):
                    hbb.append((x_b, y_b))
            #   A opposite-spin (ab) double (x_a, y_b) to (w_a, z_b) \in pab, where x_a < a1
            for x_a in takewhile(lambda x: x < a1, det_a):
                for y_b in det_b:
                    hab.append((x_a, y_b))

        return (haa, paa, hbb, pbb, hab, pab)

    #     _
    #    |_) |_   _.  _  _      |_|  _  |  _
    #    |   | | (_| _> (/_ o   | | (_) | (/_
    #                   _   /
    #     _. ._   _|   |_) _. ._ _|_ o  _ |  _
    #    (_| | | (_|   |  (_| |   |_ | (_ | (/_

    @staticmethod
    def single_exc_no_phase(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[OrbitalIdx, OrbitalIdx]:
        """hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Determinant.single_exc_no_phase((1, 5, 7), (1, 23, 7))
        (5, 23)
        >>> Determinant.single_exc_no_phase((1, 2, 9), (1, 9, 18))
        (2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        return h, p

    @staticmethod
    def double_exc_no_phase(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Determinant.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (3, 4, 12, 13)
        >>> Determinant.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        return h1, h2, p1, p2


Psi_det = List[Determinant]
Psi_coef = List[float]
# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det
Energy = NewType("Energy", float)
