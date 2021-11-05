from dataclasses import dataclass
from .types import *
from .hamiltonian_2e import Hamiltonian_2e
from .hamiltonian_4e_determinant_driven import Hamiltonian_4e_determinant_driven

@dataclass
class Hamiltonian(object):

    d_one_e_integral: One_electron_integral
    d_two_e_integral: Two_electron_integral
    E0: Energy

    @property
    def H_4e_engine(self):
        return Hamiltonian_4e_determinant_driven(self.d_two_e_integral)

    @property
    def H_2e_engine(self):
        return Hamiltonian_2e(self.d_one_e_integral, self.E0)

    # ~ ~ ~
    # H
    # ~ ~ ~
    def H(self, psi_i: Psi_det, psi_j: Psi_det) -> List[List[Energy]]:
        """Return a matrix of size psi_i x psi_j containing the value of the Hamiltonian.
        Note that when psi_i == psi_j, this matrix is an hermitian."""
        H = self.H_2e_engine.H_2e(psi_i, psi_j) + self.H_4e_engine.H_4e(psi_i, psi_j)
        return H

                   
    # ~ ~ ~
    # H_i_i
    # ~ ~ ~
    def H_i_i(self, det_i) -> List[Energy]:
        """
        return diagonal element of H for det_i (used in denominator for PT2)
        """
        H_i_i_4e = sum(phase * self.H_2e_engine.H_two_e(*idx) for idx, phase in self.H_i_i_4e_index(det_i))
        return self.H_i_i_2e(det_i) + H_i_i_4e
 
