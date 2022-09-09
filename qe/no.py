def get_MO_1rdm(comm, lewis, n_orb, psi_coef: Psi_coef, psi_det: Psi_det, n) -> MO_1rdm:
    return Powerplant_manager(comm, lewis).MO_1rdm(psi_coef, n, n_orb)


def natural_orbitals(
    comm, lewis: Hamiltonian_generator, n_orb, psi_coef: Psi_coef, psi_det: Psi_det, n
) -> Tuple[no_occ, no_coeff]:
    # create MO 1 rdm
    mo_1rdm = get_MO_1rdm(comm, lewis, n_orb, psi_coef, psi_det, n)
    # diagonalize MO 1 rdm
    no_occ, no_coeff_MO_basis = np.linalg.eigh(mo_1rdm)
    no_occ = no_occ[::-1]
    no_coeff = np.fliplr(no_coeff)
    # transform MO 1 rdm to ao basis
    # NEED AO BASIS FOR THIS
