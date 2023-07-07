from typing import Tuple, Dict, NamedTuple, List, NewType

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
