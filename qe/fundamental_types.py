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

# TODO: Will remove this later... But fixes build for now
Spin_determinant = Tuple[OrbitalIdx, ...]


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
    def apply_excitation_to_spindet(
        sdet: Spin_determinant_tuple, exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]]
    ) -> Tuple[OrbitalIdx, ...]:
        """Function to `apply' excitation to |Spin_determinant_tuple| object
        Implemented via symmetric set difference (^)

        :param exc: variable length tuple containing [[holes], [particles]] that determine the excitation

        >>> Determinant_tuple.apply_excitation_to_spindet((0, 1), ((1,), (2,)))
        (0, 2)
        >>> Determinant_tuple.apply_excitation_to_spindet((1, 3), ((1,), (2,)))
        (2, 3)
        >>> Determinant_tuple.apply_excitation_to_spindet((0, 1), ((), ()))
        (0, 1)
        """
        lh, lp = exc  # Unpack
        return tuple(sorted(set(sdet) ^ (set(lh) | set(lp))))

    def apply_excitation(
        self,
        alpha_exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]],
        beta_exc: Tuple[Tuple[OrbitalIdx, ...], Tuple[OrbitalIdx, ...]],
    ) -> Determinant:
        """Apply excitation to self, produce new |Determinant_tuple|
        Input `alpha_exc` (`beta_exc`) specifies holes, particles involved in excitation of alpha (beta) |Spin_determinant|
        If either argument is empty (), no excitation is applied
        >>> Determinant_tuple((0, 1), (0, 1)).apply_excitation(((1,), (2,)), ((1,), (2,)))
        Determinant_tuple(alpha=(0, 2), beta=(0, 2))
        >>> Determinant_tuple((0, 1), (0, 1)).apply_excitation(((0, 1), (2, 3)), ((0, 1), (3, 4)))
        Determinant_tuple(alpha=(2, 3), beta=(3, 4))
        >>> Determinant_tuple((0, 1), (0, 1)).apply_excitation(((), ()), ((), ()))
        Determinant_tuple(alpha=(0, 1), beta=(0, 1))
        """

        excited_sdet_a = Determinant_tuple.apply_excitation_to_spindet(self.alpha, alpha_exc)
        excited_sdet_b = Determinant_tuple.apply_excitation_to_spindet(self.beta, beta_exc)
        # Return excited det as instance of |Determinant_tuple|
        return Determinant_tuple(excited_sdet_a, excited_sdet_b)

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
        apply_excitation_fixed_sdet = partial(self.apply_excitation_to_spindet, sdet)
        return map(apply_excitation_fixed_sdet, l_hp_pairs)

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

    def exc_degree(self, det_j: Determinant) -> Tuple[int, int]:
        """Compute excitation degree; the number of orbitals which differ between two |Determinants| self, det_J
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).exc_degree(Determinant_tuple(alpha=(0, 2), beta=(4, 6)))
        (1, 2)
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).exc_degree(Determinant_tuple(alpha=(0, 1), beta=(4, 6)))
        (0, 2)
        """
        ed_up = Determinant_tuple.exc_degree_spindet(self.alpha, det_j.alpha)
        ed_dn = Determinant_tuple.exc_degree_spindet(self.beta, det_j.beta)
        return (ed_up, ed_dn)

    def is_connected(self, det_j: Determinant) -> Tuple[int, int]:
        """Compute the excitation degree, the number of orbitals which differ between the two determinants.
        Return bool; `Is det_j (singley or doubley) connected to instance of self?
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).is_connected(Determinant_tuple(alpha=(0, 1), beta=(0, 2)))
        True
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).is_connected(Determinant_tuple(alpha=(0, 2), beta=(0, 2)))
        True
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).is_connected(Determinant_tuple(alpha=(2, 3), beta=(0, 1)))
        True
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).is_connected(Determinant_tuple(alpha=(2, 3), beta=(0, 2)))
        False
        >>> Determinant_tuple(alpha=(0, 1), beta=(0, 1)).is_connected(Determinant_tuple(alpha=(0, 1), beta=(0, 1)))
        False
        """
        return sum(Determinant_tuple.exc_degree(self, det_j)) in [1, 2]

    def triplet_constrained_single_excitations_from_det(
        self, constraint: Spin_determinant, n_orb: int, spin="alpha"
    ) -> Iterator[Determinant]:
        """Compute all (single) excitations of a det (self) subject to a triplet contraint T = [o1: |OrbitalIdx|, o2: |OrbitalIdx|, o3: |OrbitalIdx|]
        By default: constraint T specifies 3 `most highly` occupied alpha spin orbitals allowed in the generated excitation
            e.g., if exciting |D> does not yield |J> such that o1, o2, o3 are the `largest` occupied alpha orbitals in |J> -> Excitation not generated
        Inputs:

        Outputs:
            Yield excitations of det |D> subject to specified constraint

        """
        ha = []  # `Occupied` orbitals to loop over
        pa = []  # `Virtual`  "                   "
        hb = []
        pb = []

        all_orbs = frozenset(range(n_orb))
        a1 = min(constraint)  # Index of `smallest` occupied constraint orbital
        B = set(
            range(a1 + 1, n_orb)
        )  # B: `Bitmask' -> |Determinant| {a1 + 1, ..., Norb - 1} # TODO: Maybe o1 inclusive...
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
        constraint_orbitals_occupied = set(det_a) & set(constraint)
        #   Which `higher-order` (spin) orbitals o >= a1 that are not {a1, a2, a3} are occupied in |D_a>? (If any)
        #   TODO: Different from Tubman paper, which has an error if I reada it correctly
        nonconstrained_orbitals_occupied = (set(det_a) & B) - set(constraint)

        # If no double excitation of |D> will produce |J> satisfying constraint
        if len(constraint_orbitals_occupied) == 1 or len(nonconstrained_orbitals_occupied) > 1:
            # No single excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            return None

        # Create list of possible `particles` s.to constraint
        if len(constraint_orbitals_occupied) == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> -> A single (a) x_a \in ha to a_unocc is necessary to satisfy the constraint
            # A single (b) will still not satisfy constraint
            (a_unocc,) = ((set(det_a) | set(constraint)) - (set(det_a) & set(constraint))) & set(
                constraint
            )  # The 1 unoccupied constraint orbital
            pa.append(a_unocc)
        elif len(constraint_orbitals_occupied) == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any single x_a \in ha to w_a where w_a < a1 will satisfy constraint
            det_unocc_a_orbs = all_orbs - set(det_a)
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                pa.append(w_a)
            #   Any single x_b \in hb to w_b
            det_unocc_b_orbs = all_orbs - set(det_b)
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

        # Finally, generate all excitations
        # Excitations of argument `spin`
        for h in ha:
            for p in pa:
                if spin == "alpha":
                    # Then, det_a is alpha spindet
                    excited_det = self.apply_excitation(((h,), (p,)), ((), ()))
                else:
                    # det_a is beta spindet
                    excited_det = self.apply_excitation(((), ()), ((h,), (p,)))
                assert getattr(excited_det, spin)[-3:] == constraint
                yield excited_det

        # Generate opposite-spin excitations
        for h in hb:
            for p in pb:
                if spin == "alpha":
                    # Then, det_b is beta spindet
                    excited_det = self.apply_excitation(((), ()), ((h,), (p,)))
                else:
                    # det_b is alpha spindet
                    excited_det = self.apply_excitation(((h,), (p,))((), ()))
                assert getattr(excited_det, spin)[-3:] == constraint
                yield excited_det

    def triplet_constrained_double_excitations_from_det(
        self, constraint: Spin_determinant, n_orb: int, spin="alpha"
    ) -> Iterator[Determinant]:
        """Compute all (double) excitations of a det (self) subject to a triplet contraint T = [a1: |OrbitalIdx|, a2: |OrbitalIdx|, a3: |OrbitalIdx|]
        By default: constraint T specifies 3 `most highly` occupied alpha spin orbitals allowed in the generated excitation
            e.g., if exciting |D> does not yield |J> such that a1, a2, a3 are the `largest` occupied alpha orbitals in |J> -> Excitation not generated
        Inputs:

        Outputs:
            Yield excitations of det |D> subject to specified constraint

        """
        # Same-spin alpha
        haa = []  # `Occupied` orbitals to loop over
        paa = []  # `Virtual`  "                   "
        # Same-spin beta
        hbb = []
        pbb = []
        # Oppositive spin
        hab = []
        pab = []

        all_orbs = frozenset(range(n_orb))
        a1 = min(constraint)  # Index of `smallest` occupied alpha constraint orbital
        B = set(range(a1 + 1, n_orb))  # B: `Bitmask' -> |Determinant| {a1 + 1, ..., Norb - 1}
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
        constraint_orbitals_occupied = set(det_a) & set(constraint)
        #   Which `higher-order`(spin) orbitals o >= a1 that are not {a1, a2, a3} are occupied in |D>? (If any)
        #   TODO: Different from Tubman paper, which has an error if I read it correctly...
        nonconstrained_orbitals_occupied = (set(det_a) & B) - set(constraint)

        # If this -> no double excitation of |D> will produce |J> satisfying constraint |T>
        if len(constraint_orbitals_occupied) == 0 or len(nonconstrained_orbitals_occupied) > 2:
            # No double excitations generated by the inputted |Determinant|: {det} satisfy given constraint: {constraint}
            return None

        # Create list of possible `particles` s.to constraint
        if len(constraint_orbitals_occupied) == 1:
            # (One constraint orbital occupied) e.g., a1 \in |D_a> -> A same-spin (aa) double to (x_a, y_a) \in haa to (a2, a3) is necessary
            # No same-spin (bb) or opposite-spin (ab) excitations will satisfy constraint
            # New particles -> a2, a3
            a_unocc_1, a_unocc_2 = (
                (set(det_a) | set(constraint)) - (set(det_a) & set(constraint))
            ) & set(
                constraint
            )  # This set will have length 2; unpack
            paa.append((a_unocc_1, a_unocc_2))

        elif len(constraint_orbitals_occupied) == 2:
            # (Two constraint orbitals occupied) e.g., a1, a2 \in |D_a> ->
            #   A same-spin (aa) double (x_a, y_a) \in haa to (z_a, a_unocc), where z_a\not\in|D_a>, and z_a < a1 (if excited to z_a > a1, constraint ruined!)
            (a_unocc,) = ((set(det_a) | set(constraint)) - (set(det_a) & set(constraint))) & set(
                constraint
            )  # The 1 unoccupied constraint orbital
            det_unocc_a_orbs = all_orbs - set(det_a)  # Unocc orbitals in |D_a>
            for z_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                # z < a_unocc trivially, no need to check they are distinct
                paa.append((z_a, a_unocc))
            #   No same spin (bb) excitations will satisfy constraint
            #   An oppopsite spin (ab) double (x_a, y_b) \in \hab to (a_unocc, z_b), where z\not\in|D_b>
            det_unocc_b_orbs = set(range(n_orb)) - set(det_b)  # Unocc orbitals in |D_b>
            for z_b in det_unocc_b_orbs:
                pab.append((a_unocc, z_b))

        elif len(constraint_orbitals_occupied) == 3:
            # a1, a2, a3 \in |D_a> -> |D> satisfies constraint! ->
            #   Any same-spin (aa) double (x_a, y_a) \in haa to (w_a, z_a), where w_a, z_a < a1
            det_unocc_a_orbs = all_orbs - set(det_a)
            for w_a in takewhile(lambda x: x < a1, det_unocc_a_orbs):
                for z_a in takewhile(lambda z: z < w_a, det_unocc_a_orbs):
                    paa.append((w_a, z_a))
            # Any same-spin (bb) double (x_b, y_b) \in hbb to (w_b, z_b)
            det_unocc_b_orbs = all_orbs - set(det_b)  # Unocc orbitals in |D_a>
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

        # Finally, generate all excitations
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

    #     _
    #    |_) |_   _.  _  _      |_|  _  |  _
    #    |   | | (_| _> (/_ o   | | (_) | (/_
    #                   _   /
    #     _. ._   _|   |_) _. ._ _|_ o  _ |  _
    #    (_| | | (_|   |  (_| |   |_ | (_ | (/_

    # Driver functions for computing phase, hole and particle between determinant pairs
    # TODO: All static methods for now... Feels weird to to pass self as an arg when we only do this for spindets
    # So, might just keep as is

    @staticmethod
    def single_phase(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...], h: OrbitalIdx, p: OrbitalIdx
    ):
        # Naive; compute phase for |Spin_determinant| pairs related by excitataion from h <-> p
        phase = 1
        for det, idx in ((sdet_i, h), (sdet_j, p)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase
        return phase

    @staticmethod
    def single_exc(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx]:
        """phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> Determinant_tuple.single_exc((0, 4, 6), (0, 22, 6))
        (1, 4, 22)
        >>> Determinant_tuple.single_exc((0, 1, 8), (0, 8, 17))
        (-1, 1, 17)
        """
        # Get holes, particle in exc
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        return Determinant_tuple.single_phase(sdet_i, sdet_j, h, p), h, p

    @staticmethod
    def single_exc_no_phase(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[OrbitalIdx, OrbitalIdx]:
        """hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J

        >>> PhaseIdx.single_exc_no_phase((1, 5, 7), (1, 23, 7))
        (5, 23)
        >>> PhaseIdx.single_exc_no_phase((1, 2, 9), (1, 9, 18))
        (2, 18)
        """
        (h,) = set(sdet_i) - set(sdet_j)
        (p,) = set(sdet_j) - set(sdet_i)

        return h, p

    @staticmethod
    def double_phase(
        sdet_i: Tuple[OrbitalIdx, ...],
        sdet_j: Tuple[OrbitalIdx, ...],
        h1: OrbitalIdx,
        h2: OrbitalIdx,
        p1: OrbitalIdx,
        p2: OrbitalIdx,
    ):
        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = Determinant_tuple.single_phase(
            sdet_i, sdet_j, h1, p1
        ) * Determinant_tuple.single_phase(sdet_j, sdet_i, p2, h2)
        if h2 < h1:
            phase *= -1
        if p2 < p1:
            phase *= -1
        return phase

    @staticmethod
    def double_exc(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[int, OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> Determinant_tuple.double_exc((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 4, 5, 6, 7, 8, 11, 12))
        (1, 2, 3, 11, 12)
        >>> Determinant_tuple.double_exc((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 3, 4, 5, 6, 7, 11, 17))
        (-1, 2, 8, 11, 17)
        """
        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))
        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        return Determinant_tuple.double_phase(sdet_i, sdet_j, h1, h2, p1, p2), h1, h2, p1, p2

    @staticmethod
    def double_exc_no_phase(
        sdet_i: Tuple[OrbitalIdx, ...], sdet_j: Tuple[OrbitalIdx, ...]
    ) -> Tuple[OrbitalIdx, OrbitalIdx, OrbitalIdx, OrbitalIdx]:
        """holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J

        >>> PhaseIdx.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5, 6, 7, 8, 9, 12, 13))
        (3, 4, 12, 13)
        >>> PhaseIdx.double_exc_no_phase((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 4, 5, 6, 7, 8, 12, 18))
        (3, 9, 12, 18)
        """

        # Holes
        h1, h2 = sorted(set(sdet_i) - set(sdet_j))

        # Particles
        p1, p2 = sorted(set(sdet_j) - set(sdet_i))

        return h1, h2, p1, p2


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
