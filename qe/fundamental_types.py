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

    def __and__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx, ...]:
        """Overload `&` operator to perform set intersection
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((0, 2))
        (0,)
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((2, 3))
        ()
        """
        return Spin_determinant_tuple(set(self) & set(s_tuple))

    def __rand__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx]:
        """Reverse overloaded __and__
        >>> (0, 1) & Spin_determinant_tuple((0, 2))
        (0,)
        >>> Spin_determinant_tuple((0, 1)) & Spin_determinant_tuple((0, 2))
        (0,)
        """
        return self.__and__(s_tuple)

    def __or__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx, ...]:
        """Overload `|` operator to perform set union
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((0, 2))
        (0, 1, 2)
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((0, 1))
        (0, 1)
        >>> Spin_determinant_tuple((0, 1)) | Spin_determinant_tuple((2, 3))
        (0, 1, 2, 3)
        """
        return Spin_determinant_tuple(set(self) | set(s_tuple))

    def __ror__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx]:
        """Reverse overloaded __or__
        >>> (0, 1) | Spin_determinant_tuple((0, 2))
        (0, 1, 2)
        """
        return self.__or__(s_tuple)

    def __xor__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx, ...]:
        """Overload `^` operator to perform symmetric set difference
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((0, 2))
        (1, 2)
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((0, 1))
        ()
        >>> Spin_determinant_tuple((0, 1)) ^ Spin_determinant_tuple((2, 3))
        (0, 1, 2, 3)
        """
        return Spin_determinant_tuple(set(self) ^ set(s_tuple))

    def __rxor__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx]:
        """Reverse overloaded __xor__
        >>> (0, 1) ^  Spin_determinant_tuple((0, 2))
        (1, 2)
        """
        return self.__xor__(s_tuple)

    def __sub__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx, ...]:
        """Overload `-` operator to perform set difference
        Return type |Spin_determinant_tuple|
        >>> Spin_determinant_tuple((0, 1)) - Spin_determinant_tuple((0, 2))
        (1,)
        >>> Spin_determinant_tuple((0, 2)) - Spin_determinant_tuple((0, 1))
        (2,)
        >>> Spin_determinant_tuple((0, 1)) - Spin_determinant_tuple((0, 1))
        ()
        """
        return Spin_determinant_tuple(set(self) - set(s_tuple))

    def __rsub__(self, s_tuple: Tuple[OrbitalIdx, ...]) -> Tuple[OrbitalIdx]:
        """Reverse overloaded __sub__
        Convert arg `s_tuple' to |Spin_determinant_tuple|, then perform __sub__ since operation is not communative
        >>> (0, 1, 2, 3) - Spin_determinant_tuple((0, 2))
        (1, 3)
        """
        return Spin_determinant_tuple(s_tuple).__sub__(self)

    def popcnt(self) -> int:
        """Perform a `popcount'; return length of the tuple"""
        return len(self)

    def gen_all_connected_spindet(self, ed: int, n_orb: int) -> Iterator[Tuple[OrbitalIdx, ...]]:
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

    def gen_all_connected_spindet(self, ed: int, n_orb: int) -> Iterator[Tuple[OrbitalIdx, ...]]:
        """Generate all connected spin determinants to self relative to a particular excitation degree
        :param n_orb: global parameter, used to pad bitstring with necessary 0s

        >>> for excited_sdet in sorted(Spin_determinant_bitstring(0b11).gen_all_connected_spindet(1, 4)):
        ...     bin(excited_sdet)
        '0b101'
        '0b110'
        '0b1001'
        '0b1010'
        >>> for excited_sdet in sorted(Spin_determinant_bitstring(0b1).gen_all_connected_spindet(1, 4)):
        ...     bin(excited_sdet)
        '0b10'
        '0b100'
        '0b1000'
        >>> for excited_sdet in sorted(Spin_determinant_bitstring(0b11).gen_all_connected_spindet(2, 4)):
        ...     bin(excited_sdet)
        '0b1100'
        >>> sorted(Spin_determinant_bitstring(0b11).gen_all_connected_spindet(2, 2))
        []
        >>> for excited_sdet in sorted(Spin_determinant_bitstring(0b1000).gen_all_connected_spindet(1, 4)):
        ...     bin(excited_sdet)
        '0b1'
        '0b10'
        '0b100'
        """
        # Run through bitstring to create lists of particles and holes
        holes = []
        particles = []
        # Some pre-processing; Create intermediate bitstring that is padded with zeros s.t len(inter) = n_orb
        # If n_orb bit is already set -> No affect
        inter = format(self, "#0" + f"{n_orb + 2}" + "b")
        # One pass right -> left through reflected bitstring ([:-2] skips 'b0' in the flipped reflected bitstring)
        for i, bit in enumerate(inter[::-1][:-2]):
            # If ith bit is set, append to holes
            if bit == "1":
                holes.append(i)
            # Else ith bit is not set, append to particles
            else:
                particles.append(i)
        l_hp_pairs = product(combinations(tuple(holes), ed), combinations(tuple(particles), ed))

        return [self ^ tuple(sorted(set(h) | set(p))) for h, p in l_hp_pairs]


#   ______     _                      _                   _
#   |  _  \   | |                    (_)                 | |
#   | | | |___| |_ ___ _ __ _ __ ___  _ _ __   __ _ _ __ | |_
#   | | | / _ \ __/ _ \ '__| '_ ` _ \| | '_ \ / _` | '_ \| __|
#   | |/ /  __/ ||  __/ |  | | | | | | | | | | (_| | | | | |_
#   |___/ \___|\__\___|_|  |_| |_| |_|_|_| |_|\__,_|_| |_|\__|
#
#


class Determinant(tuple):
    """Slater determinant: Product of 2 determinants.
    One for $\alpha$ electrons and one for \beta electrons.
    Abstract |Determinant| class; mimics behaviour of |NamedTuple|"""

    __slots__ = ()

    _fields = ("alpha", "beta")

    def __new__(_cls, *args, **kwargs):
        """Create new |Determinant| instance
        If type of alpha, beta are |int| -> Return as is
        '                        ' |tuple| -> Convert to |Spin_determinant_tuple|, then return"""

        # Can either...
        #   Pass arguments in alpha, beta order as unnamed objects; e.g., Determinant(alpha_sdet, beta_sdet)
        if len(args) > 0:
            if (
                len(args) == 2
            ):  # Most often, Determinants will be created via Determinant(alpha_sdet, beta_sdet)
                alpha, beta = args
            elif len(args) == 1:
                # Arg passed might be tuple of sdets; Determinant(((alpha_sdet, beta_sdet))
                # Note: For some reason, mpi4py does this sometime when performing collectives
                try:
                    # Unpack length-1 tuple
                    (_sdets,) = args
                    alpha, beta = _sdets
                except:
                    raise TypeError(f"Expected 2 arguments, got {args}")
        # Or...
        #   Pass arguments with keywords; e.g., Determinant(alpha=alpha_sdet, beta=beta_sdet)
        elif len(kwargs) > 0:
            try:
                alpha = kwargs["alpha"]
                beta = kwargs["beta"]
            except:
                raise KeyError(f"Expected keyword arguments for 'alpha', 'beta', got {kwargs}")
        else:
            raise TypeError(f"Expected two keyword arguments for 'alpha', 'beta', got {kwargs}")

        # Once arguments are parsed, determine data representation
        if isinstance(alpha, tuple):
            _alpha = Spin_determinant_tuple(alpha)
        elif isinstance(alpha, int):
            _alpha = Spin_determinant_bitstring(alpha)
        else:
            raise TypeError(
                f"Expected 'alpha' argument of type <class 'tuple'> or <class 'int'>, got {type(alpha)}"
            )
        if isinstance(beta, tuple):
            _beta = Spin_determinant_tuple(beta)
        elif isinstance(beta, int):
            _beta = Spin_determinant_bitstring(beta)
        else:
            raise TypeError(
                f"Expected 'beta' argument of type <class 'tuple'> or <class 'int'>, got {type(beta)}"
            )
        return tuple.__new__(_cls, (_alpha, _beta))

    @classmethod
    def _make(cls, iterable):
        """Make a new Determinant object from a sequence or iterable
        Used in batch `apply_excitation' calls"""
        result = tuple.__new__(cls, iterable)
        if len(result) != 2:
            raise TypeError(f"Expected 2 arguments, got {len(result)}")
        return result

    def __repr__(self):
        """Return a nicely formatted representation string"""
        return "Determinant(alpha=%r, beta=%r)" % self

    @property
    def alpha(self):
        """Return alpha spin determinant"""
        return self[0]

    @property
    def beta(self):
        """Return beta spin determinant"""
        return self[1]

    def convert_repr(self, Norb=None):
        """Conver to bitstring (tuple) representation of |Determinant| if self is tuple (bitstring)
        >>> Determinant((0, 2), (1,)).convert_repr(4)
        Determinant(alpha=5, beta=2)
        >>> Determinant((0, 2), ()).convert_repr(4)
        Determinant(alpha=5, beta=0)

        >>> Determinant(0b0101, 0b0001).convert_repr()
        Determinant(alpha=(0, 2), beta=(0,))
        >>> Determinant(0b0101, 0b0000).convert_repr()
        Determinant(alpha=(0, 2), beta=())
        """
        # Each spin determinant class has member `convert_repr` function
        return Determinant(self.alpha.convert_repr(Norb), self.beta.convert_repr(Norb))

    #     _
    #    |_     _ o _|_  _. _|_ o  _  ._
    #    |_ >< (_ |  |_ (_|  |_ | (_) | |
    #

    def apply_excitation(
        self,
        alpha_exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]],
        beta_exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]],
    ) -> NamedTuple:
        """Apply excitation to self, produce new |Determinant|
        Each type |Determinant_tuple| and |Determinant_bitstring| has own implementation of `apply_excitation_to_spindet` based on type
        Inputs
            :param `alpha_exc` (`beta_exc`): Specifies holes, particles involved in excitation of alpha (beta) |Spin_determinant|

        If either argument is empty (), no excitation is applied
        >>> Determinant((0, 1), (0, 1)).apply_excitation(((1,), (2,)), ((1,), (2,)))
        Determinant(alpha=(0, 2), beta=(0, 2))
        >>> Determinant((0, 1), (0, 1)).apply_excitation(((0, 1), (2, 3)), ((0, 1), (3, 4)))
        Determinant(alpha=(2, 3), beta=(3, 4))
        >>> Determinant((0, 1), (0, 1)).apply_excitation(((), ()), ((), ()))
        Determinant(alpha=(0, 1), beta=(0, 1))

        >>> Determinant(0b11, 0b11).apply_excitation(((1,), (2,)), ((1,), (2,)))
        Determinant(alpha=5, beta=5)
        >>> Determinant(0b11, 0b11).apply_excitation(((0, 1), (2, 3)), ((0, 1), (3, 4)))
        Determinant(alpha=12, beta=24)
        >>> Determinant(0b11, 0b11).apply_excitation(((), ()), ((), ()))
        Determinant(alpha=3, beta=3)
        >>> Determinant(0b11, 0b11).apply_excitation(((1,), (2,)), ((), ()))
        Determinant(alpha=5, beta=3)
        """
        # Unpack alpha, beta holes
        lh_a, lp_a = alpha_exc
        lh_b, lp_b = beta_exc

        excited_sdet_a = self.alpha ^ (tuple(sorted(set(lh_a) | set(lp_a))))
        excited_sdet_b = self.beta ^ (tuple(sorted(set(lh_b) | set(lp_b))))
        return Determinant(excited_sdet_a, excited_sdet_b)

    def exc_degree(self, det_J: NamedTuple) -> Tuple[int, int]:
        """Compute excitation degree; the number of orbitals which differ between two |Determinants| self, det_J

        >>> Determinant((0, 1), (0, 1)).exc_degree(Determinant(alpha=(0, 2), beta=(4, 6)))
        (1, 2)
        >>> Determinant((0, 1), (0, 1)).exc_degree(Determinant(alpha=(0, 1), beta=(4, 6)))
        (0, 2)
        >>> Determinant(0b11, 0b11).exc_degree(Determinant(0b101, 0b101000))
        (1, 2)
        >>> Determinant(0b11, 0b11).exc_degree(Determinant(0b11, 0b101000))
        (0, 2)
        """
        ed_up = (self.alpha ^ det_J.alpha).popcnt() // 2
        ed_dn = (self.beta ^ det_J.beta).popcnt() // 2
        return (ed_up, ed_dn)

    def is_connected(self, det_j) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ between the two determinants.
        Return bool; `Is det_j (singley or doubley) connected to instance of self?

        >>> Determinant((0, 1), (0, 1)).is_connected(Determinant((0, 1), (0, 2)))
        True
        >>> Determinant((0, 1), (0, 1)).is_connected(Determinant((0, 2), (0, 2)))
        True
        >>> Determinant((0, 1), (0, 1)).is_connected(Determinant((2, 3), (0, 1)))
        True
        >>> Determinant((0, 1), (0, 1)).is_connected(Determinant((2, 3), (0, 2)))
        False
        >>> Determinant((0, 1), (0, 1)).is_connected(Determinant((0, 1), (0, 1)))
        False

        >>> Determinant(0b11, 0b11).is_connected(Determinant(0b11, 0b101))
        True
        >>> Determinant(0b11, 0b11).is_connected(Determinant(0b101, 0b101))
        True
        >>> Determinant(0b11, 0b11).is_connected(Determinant(0b1100, 0b11))
        True
        >>> Determinant(0b11, 0b11).is_connected(Determinant(0b1100, 0b101))
        False
        >>> Determinant(0b11, 0b11).is_connected(Determinant(0b11, 0b11))
        False
        """
        return sum(self.exc_degree(det_j)) in [1, 2]

    def gen_all_connected_det(self, n_orb: int) -> Iterator[NamedTuple]:
        """Generate all determinants that are singly or doubly connected to self
        :param n_orb: global parameter, needed to cap possible excitations

        >>> sorted(Determinant((0, 1), (0,)).gen_all_connected_det(3))
        [Determinant(alpha=(0, 1), beta=(1,)),
         Determinant(alpha=(0, 1), beta=(2,)),
         Determinant(alpha=(0, 2), beta=(0,)),
         Determinant(alpha=(0, 2), beta=(1,)),
         Determinant(alpha=(0, 2), beta=(2,)),
         Determinant(alpha=(1, 2), beta=(0,)),
         Determinant(alpha=(1, 2), beta=(1,)),
         Determinant(alpha=(1, 2), beta=(2,))]

        >>> for excited_det in sorted(Determinant(0b11, 0b1).gen_all_connected_det(3)):
        ...     [bin(excited_det.alpha), bin(excited_det.beta)]
        ['0b11', '0b10']
        ['0b11', '0b100']
        ['0b101', '0b1']
        ['0b101', '0b10']
        ['0b101', '0b100']
        ['0b110', '0b1']
        ['0b110', '0b10']
        ['0b110', '0b100']
        """
        # Generate all singles from constituent alpha and beta spin determinants
        # Then, opposite-spin and same-spin doubles

        # We use l_single_a, and l_single_b twice. So we store them.
        l_single_a = set(self.alpha.gen_all_connected_spindet(1, n_orb))
        l_double_aa = self.alpha.gen_all_connected_spindet(2, n_orb)

        # Singles and doubles; alpha spin
        exc_a = (Determinant(det_alpha, self.beta) for det_alpha in chain(l_single_a, l_double_aa))

        l_single_b = set(self.beta.gen_all_connected_spindet(1, n_orb))
        l_double_bb = self.beta.gen_all_connected_spindet(2, n_orb)

        # Singles and doubles; beta spin
        exc_b = (Determinant(self.alpha, det_beta) for det_beta in chain(l_single_b, l_double_bb))

        l_double_ab = product(l_single_a, l_single_b)

        # Doubles; opposite-spin
        exc_ab = (Determinant(det_alpha, det_beta) for det_alpha, det_beta in l_double_ab)

        return chain(exc_a, exc_b, exc_ab)

    def triplet_constrained_single_excitations_from_det(
        self, constraint: Tuple[OrbitalIdx, ...], n_orb: int, spin="alpha"
    ) -> Iterator[NamedTuple]:
        """Called by inherited classes; Generate singlet excitations from constraint"""

        ha, pa, hb, pb = self.get_holes_particles_for_constrained_singles(constraint, n_orb, spin)
        # Excitations of argument `spin`
        for h, p in product(ha, pa):
            if spin == "alpha":
                # Then, det_a is alpha spindet
                excited_det = self.apply_excitation(((h,), (p,)), ((), ()))
            else:
                # det_a is beta spindet
                excited_det = self.apply_excitation(((), ()), ((h,), (p,)))
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Generate opposite-spin excitations
        for h, p in product(hb, pb):
            if spin == "alpha":
                # Then, det_b is beta spindet
                excited_det = self.apply_excitation(((), ()), ((h,), (p,)))
            else:
                # det_b is alpha spindet
                excited_det = self.apply_excitation(((h,), (p,)), ((), ()))
            # TODO: Assertion for bitstring? Shouldn't need though
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

    def triplet_constrained_double_excitations_from_det(
        self, constraint: Tuple[OrbitalIdx, ...], n_orb: int, spin="alpha"
    ) -> Iterator[NamedTuple]:
        """Called by inherited classes; Generate singlet excitations from constraint"""

        # |Determinant_tuple| and |Determinant_bitstring| each have this method
        haa, paa, hbb, pbb, hab, pab = self.get_holes_particles_for_constrained_doubles(
            constraint, n_orb, spin
        )
        # Excitations of argument `spin`
        # Same-spin doubles, for argument `spin`
        for holes, particles in product(haa, paa):
            if spin == "alpha":
                # Then, det_a is alpha spindet
                excited_det = self.apply_excitation((holes, particles), ((), ()))
            else:
                # det_a is beta spindet
                excited_det = self.apply_excitation(((), ()), (holes, particles))
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Same-spin doubles, for opposite-spin to `spin`
        for holes, particles in product(hbb, pbb):
            if spin == "alpha":
                # Then, det_b is beta spindet
                excited_det = self.apply_excitation(((), ()), (holes, particles))
            else:
                # det_b is alpha spindet
                excited_det = self.apply_excitation((holes, particles), ((), ()))
            assert getattr(excited_det, spin)[-3:] == constraint
            yield excited_det

        # Opposite-spin doubles
        for holes, particles in product(hab, pab):
            ha, hb = holes
            pa, pb = particles
            if spin == "alpha":
                # det_a is alpha, det_b is beta
                excited_det = self.apply_excitation(((ha,), (pa,)), ((hb,), (pb,)))
            else:
                # det_a is beta, det_b is beta
                excited_det = self.apply_excitation(((hb,), (pb,)), ((ha,), (pa,)))
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
        nonconstrained_orbitals_occupied = (det_a & B) - constraint

        # If no double excitation of |D> will produce |J> satisfying constraint
        if len(constraint_orbitals_occupied) == 1 or len(nonconstrained_orbitals_occupied) > 1:
            # No single excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            # These are empty lists
            return (ha, pa, hb, pb)

        # Create list of possible `particles` s.to constraint
        if len(constraint_orbitals_occupied) == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> -> A single (a) x_a \in ha to a_unocc is necessary to satisfy the constraint
            # A single (b) will still not satisfy constraint
            (a_unocc,) = (
                (det_a | constraint) - (det_a & constraint)
            ) & constraint  # The 1 unoccupied constraint orbital
            pa.append(a_unocc)
        elif len(constraint_orbitals_occupied) == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any single x_a \in ha to w_a where w_a < a1 will satisfy constraint
            det_unocc_a_orbs = all_orbs - det_a
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                pa.append(w_a)
            #   Any single x_b \in hb to w_b
            det_unocc_b_orbs = all_orbs - det_b
            for w_b in det_unocc_b_orbs:
                pb.append(w_b)

        # Create list of possible `holes` s.to constraint
        if len(nonconstrained_orbitals_occupied) == 1:
            # x_a > a1 \in |D_a> with x_a \not\in {a1, a2, a3} -> A single (a) x_a to w_a \in pa is necessary to satisfy constraint
            # A single (b) will not satisfy
            (x_a,) = nonconstrained_orbitals_occupied  # Has length 1; unpack
            ha.append(x_a)
        elif len(nonconstrained_orbitals_occupied) == 0:
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
        nonconstrained_orbitals_occupied = (det_a & B) - constraint

        # If this -> no double excitation of |D> will produce |J> satisfying constraint |T>
        if len(constraint_orbitals_occupied) == 0 or len(nonconstrained_orbitals_occupied) > 2:
            # No double excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            # These are empty lists
            return (haa, paa, hbb, pbb, hab, pab)

        # Create list of possible `particles` s.to constraint
        if len(constraint_orbitals_occupied) == 1:
            # (One constraint orbital occupied) e.g., a1 \in |D_a> -> A same-spin (aa) double to (x_a, y_a) \in haa to (a2, a3) is necessary
            # No same-spin (bb) or opposite-spin (ab) excitations will satisfy constraint
            # New particles -> a2, a3
            a_unocc_1, a_unocc_2 = ((det_a | constraint) - (det_a & constraint)) & (
                constraint
            )  # This set will have length 2; unpack
            paa.append((a_unocc_1, a_unocc_2))

        elif len(constraint_orbitals_occupied) == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> ->
            #   A same-spin (aa) double (x_a, y_a) \in haa to (z_a, a_unocc), where z_a\not\in|D_a>, and z_a < a1 (if excited to z_a > a1, constraint ruined!)
            (a_unocc,) = (
                (det_a | constraint) - (det_a & constraint)
            ) & constraint  # The 1 unoccupied constraint orbital
            det_unocc_a_orbs = all_orbs - det_a  # Unocc orbitals in |D_a>
            for z_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                # z < a_unocc trivially, no need to check they are distinct
                paa.append((z_a, a_unocc))
            #   No same spin (bb) excitations will satisfy constraint
            #   An oppopsite spin (ab) double (x_a, y_b) \in \hab to (a_unocc, z_b), where z\not\in|D_b>
            det_unocc_b_orbs = all_orbs - det_b  # Unocc orbitals in |D_b>
            for z_b in det_unocc_b_orbs:
                pab.append((a_unocc, z_b))

        elif len(constraint_orbitals_occupied) == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any same-spin (aa) double (x_a, y_a) \in haa to (w_a, z_a), where w_a, z_a < a1
            det_unocc_a_orbs = all_orbs - det_a
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                for z_a in takewhile(lambda z: z < w_a, det_unocc_a_orbs):
                    paa.append((w_a, z_a))
            # Any same-spin (bb) double (x_b, y_b) \in hbb to (w_b, z_b)
            det_unocc_b_orbs = all_orbs - det_b  # Unocc orbitals in |D_a>
            for w_b in det_unocc_b_orbs:
                for z_b in takewhile(lambda x: x < w_b, det_unocc_b_orbs):
                    pbb.append((w_b, z_b))
            #   Any oppospite-spin (ab) double (x_a, y_b) \in hab to (w_a, z_b), where w_a < a1
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                for z_b in det_unocc_b_orbs:
                    pab.append((w_a, z_b))

        # Create list of possible `holes` s.to constraint
        if len(nonconstrained_orbitals_occupied) == 2:
            # x_a, y_a \in |D_a> with x_a, y_a > a1 and \not\in {a1, a2, a3} -> A same-spin (aa) double (x_a, y_a) to (w_a, z_a) \in paa is necessary
            # No same-spin (bb) or opposite-spin (ab) excitations will satisfy constraint
            # New holes -> x, y
            x_a, y_a = nonconstrained_orbitals_occupied  # This set will have length 2; unpack
            haa.append((x_a, y_a))
        elif len(nonconstrained_orbitals_occupied) == 1:
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

        elif len(nonconstrained_orbitals_occupied) == 0:
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

    # Driver functions for computing phase, hole and particle between determinant pairs
    # TODO: These are ONLY implemented for |Spin_determinant_tuple| at the moment;
    # So, might just keep as is

    def single_phase(
        self,
        h: OrbitalIdx,
        p: OrbitalIdx,
        spin: str,
    ):
        """Function to compute phase for <I|H|J> when I and J differ by exactly one orbital h <-> p
        h is occupied in sdet = getattr(self, spin), p is unoccupied

        >>> Determinant((0, 4, 6), ()).single_phase(4, 5, "alpha")
        1
        >>> Determinant((), (0, 1, 8)).single_phase(1, 17, "beta")
        -1
        >>> Determinant((0, 1, 4, 8), ()).single_phase(1, 17, "alpha")
        1
        >>> Determinant((0, 1, 4, 7, 8), ()).single_phase(1, 17, "alpha")
        -1

        >>> Determinant(0b1010001, 0b0).single_phase(4, 6, "alpha")
        1
        >>> Determinant(0b0, 0b100000011).single_phase(1, 17, "beta")
        -1
        >>> Determinant(0b100010011, 0b0).single_phase(1, 17, "alpha")
        1
        >>> Determinant(0b110010011, 0b0).single_phase(1, 17, "alpha")
        -1
        """
        # Naive; compute phase for |Spin_determinant| pairs related by excitataion from h <-> p
        sdet = getattr(self, spin)
        j, k = min(h, p), max(h, p)

        if isinstance(sdet, tuple):
            pmask = tuple((i for i in range(j + 1, k)))
        elif isinstance(sdet, int):
            # Strings immutable -> Work with lists for ON assignment, convert to |str| -> |int| when done
            pmask = ["0", "b"]
            pmask.extend(["0"] * k)
            for i in range(j + 1, k):
                pmask[-(i + 1)] = "1"
            pmask = int(("".join(pmask)), 2)

        parity = (sdet & pmask).popcnt() % 2
        return -1 if parity else 1

    def double_phase(self, holes: Tuple[OrbitalIdx, ...], particles: Tuple[OrbitalIdx, ...], spin):
        """Function to compute phase for <I|H|J> when I and J differ by exactly two orbitals h1, h2 <-> p1, p2
        Only for same spin double excitations
        h1, h2 is occupied in sdet = getattr(self, spin), p1, p2 is unoccupied
        >>> Determinant((0, 1, 2, 3, 4, 5, 6, 7, 8), ()).double_phase((2, 3), (11, 12), "alpha")
        1
        >>> Determinant((0, 1, 2, 3, 4, 5, 6, 7, 8), ()).double_phase((2, 8), (11, 17), "alpha")
        -1
        """
        # Compute phase. Loopless as in https://arxiv.org/abs/1311.6244
        h1, h2 = holes
        p1, p2 = particles
        phase = self.single_phase(h1, p1, spin) * self.single_phase(h2, p2, spin)
        # if max(h1, p1) > min(h2, p2):
        #     return -phase
        # else:
        #     return phase
        if h2 < p1:
            phase *= -1
        if p2 < h1:
            phase *= -1
        return phase

    def single_exc(self, sdet_j, spin: str) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Determinant((0, 4, 6), ()).single_exc((0, 5, 6), "alpha")
        (1, 4, 5)
        >>> Determinant((0, 4, 6), ()).single_exc((0, 22, 6), "alpha")
        (-1, 4, 22)
        >>> Determinant((), (0, 1, 8)).single_exc((0, 8, 17), "beta")
        (-1, 1, 17)
        """
        # Get holes, particle in exc
        sdet_i = getattr(self, spin)
        (h,) = sdet_i - sdet_j
        (p,) = sdet_j - sdet_i

        return self.single_phase(h, p, spin), h, p

    def double_exc(
        self, sdet_j: Tuple[OrbitalIdx, ...], spin: str
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Determinant((0, 1, 2, 3, 4, 5, 6, 7, 8), ()).double_exc((0, 1, 4, 5, 6, 7, 8, 11, 12), "alpha")
        (1, 2, 3, 11, 12)
        >>> Determinant((), (0, 1, 2, 3, 4, 5, 6, 7, 8)).double_exc((0, 1, 3, 4, 5, 6, 7, 11, 17), "beta")
        (-1, 2, 8, 11, 17)
        """
        sdet_i = getattr(self, spin)
        # Holes
        h1, h2 = sorted(sdet_i - sdet_j)
        # Particles
        p1, p2 = sorted(sdet_j - sdet_i)

        return self.double_phase((h1, h2), (p1, p2), spin), h1, h2, p1, p2

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
