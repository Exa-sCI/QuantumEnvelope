from .types import *
from itertools import takewhile

def get_exc_degree(det_i: Determinant, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ
           between the two determinants.
        >>> Hamiltonian.get_exc_degree(Determinant(alpha=(1, 2), beta=(1, 2)),
        ...                            Determinant(alpha=(1, 3), beta=(5, 7)))
        (1, 2)
        """
        ed_up = len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
        ed_dn = len(set(det_i.beta).symmetric_difference(set(det_j.beta))) // 2
        return ed_up, ed_dn

def get_phase_idx_single_exc(sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Hamiltonian.get_phase_idx_single_exc((1, 5, 7), (1, 23, 7))
        (1, 5, 23)
        >>> Hamiltonian.get_phase_idx_single_exc((1, 2, 9), (1, 9, 18))
        (-1, 2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        phase = 1
        for det, idx in ((sdet_i, h), (sdet_j, p)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase

        return (phase, h, p)

def get_phase_idx_double_exc(sdet_i: Spin_determinant, sdet_j: Spin_determinant) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Hamiltonian.get_phase_idx_double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (1, 3, 4, 12, 13)
        >>> Hamiltonian.get_phase_idx_double_exc((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (-1, 3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = 1
        for det, idx in ((sdet_i, h1), (sdet_j, p1), (sdet_j, p2), (sdet_i, h2)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase

        # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
        # Look like to be always true in our tests
        if (min(h2, p2) < max(h1, p1)) != (h2 < p1 or p2 < h1):
            phase = -phase
            print(">>> get_phase_idx_double_exc QP conditional was trigered! Please repport to the developpers", sdet_i, sdet_j)

        return (phase, h1, h2, p1, p2)
