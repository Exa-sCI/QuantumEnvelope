from itertools import chain, product, combinations, takewhile
import numpy as np

class Excitation(object):
    def __init__(self, n_orb):
        """
        we don't handle cases with frozen/inactive orbitals
        all orbitals are active
        """
        self.all_orbs = frozenset(range(1, n_orb + 1))

    def gen_all_excitation(self, spindet: Spin_determinant, exc_degree: int) -> Iterator:
        """
        Generate list of pair -> hole from a determinant spin.

        >>> sorted(Excitation(4).gen_all_excitation( (1,2),2))
        [((1, 2), (3, 4))]
        >>> sorted(Excitation(4).gen_all_excitation( (1,2),1))
        [((1,), (3,)), ((1,), (4,)), ((2,), (3,)), ((2,), (4,))]
        """
        holes = combinations(spindet, exc_degree)
        not_spindet = self.all_orbs - set(spindet)
        parts = combinations(not_spindet, exc_degree)
        return product(holes, parts)

    def gen_all_connected_spindet(self, spindet: Spin_determinant, exc_degree: int) -> Iterator:
        """
        Generate all the posible spin determinant relative to a excitation degree
        similar to `gen_all_excitation`, but excitations are then applied

        >>> sorted(Excitation(4).gen_all_connected_spindet( (1,2), 1))
        [(1, 3), (1, 4), (2, 3), (2, 4)]
        """

        def apply_excitation(exc: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Spin_determinant:
            # Warning use global variable spindet.
            lh, lp = exc
            s = (set(spindet) - set(lh)) | set(lp)
            return tuple(sorted(s))

        l_exc = self.gen_all_excitation(spindet, exc_degree)
        return map(apply_excitation, l_exc)

    def gen_all_connected_det_from_det(self, det: Determinant) -> Iterator[Determinant]:
        """
        Generate all the determinant who are single or double exictation (aka connected) from the input determinant with all n orbitals active


        >>> sorted(Excitation(3).gen_all_connected_det_from_det( Determinant( alpha=(1,2), beta=(1,) )))
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
        return list(set(chain.from_iterable(map(self.gen_all_connected_det_from_det, psi_det))) - set(psi_det))

