from dataclasses import dataclass
from .types import *
from .hamiltonian import Hamiltonian
import numpy as np

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

        # This doeesn't seem to be the correct abstraction for integral driven
        nomitator = np.einsum("i,ij -> j", psi_coef, self.lewis.H(self.psi_det, psi_external))  # vector * Matrix -> vector
        denominator = np.divide(1.0, self.E(psi_coef) - np.array([self.lewis.H_i_i(d) for d in psi_external]))

        return psi_external, np.einsum("i,i,i -> i", nomitator, nomitator, denominator)  # vector * vector * vector -> scalar

    def E_pt2(self, psi_coef: Psi_coef, n_orb) -> Energy:
        # The sum of the pt2 contribution of each external determinant
        _, psi_external_energy = self.psi_external_pt2(psi_coef, n_orb)
        return sum(psi_external_energy)

