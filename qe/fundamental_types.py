from typing import Tuple, Dict, NamedTuple, List, NewType
import numpy as np

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

MO_1rdm = np.ndarray
no_occ = np.ndarray
no_coeff = np.ndarray

# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det
Energy = NewType("Energy", float)
