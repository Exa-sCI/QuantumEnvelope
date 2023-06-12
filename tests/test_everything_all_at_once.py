#!/usr/bin/env python3

import unittest
import time
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qe.integral_indexing_utils import (
    compound_idx4_reverse,
    compound_idx4,
    canonical_idx4,
    compound_idx2,
    compound_idx2_reverse,
    compound_idx4_reverse_all,
)
from qe.drivers import (
    integral_category,
    PhaseIdx,
    Excitation,
    Hamiltonian_two_electrons_integral_driven,
    Hamiltonian_two_electrons_determinant_driven,
    H_indices_generator,
    Hamiltonian_generator,
    Powerplant_manager,
    selection_step,
)
from qe.io import load_eref, load_integrals, load_wf
from collections import defaultdict
from itertools import product, chain
from functools import cached_property
from qe.fundamental_types import Determinant
from mpi4py import MPI


class Timing:
    def setUp(self):
        print(f"{self.id()} ... ", end="", flush=True)
        self.startTime = time.perf_counter()
        if PROFILING:
            import cProfile

            self.pr = cProfile.Profile()
            self.pr.enable()

    def tearDown(self):
        t = time.perf_counter() - self.startTime
        print(f"ok ({t:.3f}s)")
        if PROFILING:
            from pstats import Stats

            self.pr.disable()
            p = Stats(self.pr)
            p.strip_dirs().sort_stats("tottime").print_stats(0.05)


class Test_Index(Timing, unittest.TestCase):
    def test_idx2_reverse(self, n=10000, nmax=(1 << 63) - 1):
        def check_idx2_reverse(ij):
            i, j = compound_idx2_reverse(ij)
            self.assertTrue(i <= j)
            self.assertEqual(ij, compound_idx2(i, j))

        for ij in random.sample(range(nmax), k=n):
            check_idx2_reverse(ij)

    def test_idx4_reverse(self, n=10000, nmax=(1 << 63) - 1):
        def check_idx4_reverse(ijkl):
            i, j, k, l = compound_idx4_reverse(ijkl)
            ik = compound_idx2(i, k)
            jl = compound_idx2(j, l)
            self.assertTrue(i <= k)
            self.assertTrue(j <= l)
            self.assertTrue(ik <= jl)
            self.assertEqual(ijkl, compound_idx4(i, j, k, l))

        for ijkl in random.sample(range(nmax), k=n):
            check_idx4_reverse(ijkl)

    def test_idx4_reverse_all(self, n=10000, nmax=(1 << 63) - 1):
        def check_idx4_reverse_all(ijkl):
            for i, j, k, l in compound_idx4_reverse_all(ijkl):
                self.assertEqual(compound_idx4(i, j, k, l), ijkl)

        for ijkl in random.sample(range(nmax), k=n):
            check_idx4_reverse_all(ijkl)

    def test_canonical_idx4(self, n=10000, nmax=(1 << 63) - 1):
        def check_canonical_idx4(ijkl):
            for i, j, k, l in compound_idx4_reverse_all(ijkl):
                self.assertEqual(
                    canonical_idx4(*compound_idx4_reverse(ijkl)), canonical_idx4(i, j, k, l)
                )

        for ijkl in random.sample(range(nmax), k=n):
            check_canonical_idx4(ijkl)

    def test_compound_idx4_reverse_is_canonical(self, n=10000, nmax=(1 << 63) - 1):
        def check_compound_idx4_reverse_is_canonical(ijkl):
            self.assertEqual(
                compound_idx4_reverse(ijkl), canonical_idx4(*compound_idx4_reverse(ijkl))
            )

        for ijkl in random.sample(range(nmax), k=n):
            check_compound_idx4_reverse_is_canonical(ijkl)


class Test_Category:
    def check_pair_idx_A(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "A")
        i, _, _, _ = idx
        self.assertEqual((i, i, i, i), idx)
        self.assertEqual(da, db)
        self.assertIn(i, da.alpha)
        self.assertIn(i, da.beta)

    def check_pair_idx_B(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "B")
        i, j, _, _ = idx
        self.assertEqual((i, j, i, j), idx)
        self.assertEqual(da, db)
        self.assertIn(i, da.alpha + da.beta)
        self.assertIn(j, da.alpha + da.beta)

    def check_pair_idx_C(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "C")
        i, j, k, l = idx
        exc = Excitation.exc_degree(da, db)
        self.assertIn(exc, ((1, 0), (0, 1)))
        if exc == (1, 0):
            (dsa, _), (dsb, _) = da, db
        elif exc == (0, 1):
            (_, dsa), (_, dsb) = da, db
        h, p = PhaseIdx.single_exc_no_phase(dsa, dsb)
        self.assertTrue(j == l or i == k)
        if j == l:
            self.assertEqual(sorted((h, p)), sorted((i, k)))
            self.assertIn(j, da.alpha + da.beta)
            self.assertIn(j, db.alpha + db.beta)
        elif i == k:
            self.assertEqual(sorted((h, p)), sorted((j, l)))
            self.assertIn(i, da.alpha + da.beta)
            self.assertIn(i, db.alpha + db.beta)

    def check_pair_idx_D(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "D")
        i, j, k, l = idx
        exc = Excitation.exc_degree(da, db)
        self.assertIn(exc, ((1, 0), (0, 1)))
        if exc == (1, 0):
            (dsa, dta), (dsb, dtb) = da, db
        elif exc == (0, 1):
            (dta, dsa), (dtb, dsb) = da, db
        h, p = PhaseIdx.single_exc_no_phase(dsa, dsb)
        self.assertTrue(j == l or i == k)
        if j == l:
            self.assertEqual(sorted((h, p)), sorted((i, k)))
            self.assertIn(j, dta)
            self.assertIn(j, dtb)
        elif i == k:
            self.assertEqual(sorted((h, p)), sorted((j, l)))
            self.assertIn(i, dta)
            self.assertIn(i, dtb)

    def check_pair_idx_E(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "E")
        i, j, k, l = idx
        self.assertTrue(i == j or j == k or k == l)
        exc = Excitation.exc_degree(da, db)
        self.assertIn(exc, ((1, 0), (0, 1), (1, 1)))
        if exc == (1, 1):
            (dsa, dta), (dsb, dtb) = da, db
            if i == j:
                p, r, s = i, k, l
            elif j == k:
                p, r, s = j, i, l
            elif k == l:
                p, r, s = k, j, i
            hs, ps = PhaseIdx.single_exc_no_phase(dsa, dsb)
            ht, pt = PhaseIdx.single_exc_no_phase(dta, dtb)
            self.assertEqual(
                sorted((sorted((hs, ps)), sorted((ht, pt)))),
                sorted((sorted((p, r)), sorted((p, s)))),
            )
        else:  # exc in ((1,0),(0,1))
            if exc == (1, 0):
                (dsa, _), (dsb, _) = da, db
            elif exc == (0, 1):
                (_, dsa), (_, dsb) = da, db
            h, p = PhaseIdx.single_exc_no_phase(dsa, dsb)
            if i == j:
                self.assertEqual(sorted((h, p)), sorted((k, l)))
                self.assertIn(i, dsa)
                self.assertIn(i, dsb)
            elif j == k:
                self.assertEqual(sorted((h, p)), sorted((i, l)))
                self.assertIn(j, dsa)
                self.assertIn(j, dsb)
            elif k == l:
                self.assertEqual(sorted((h, p)), sorted((i, j)))
                self.assertIn(k, dsa)
                self.assertIn(k, dsb)

    def check_pair_idx_F(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "F")
        i, _, k, _ = idx
        exc = Excitation.exc_degree(da, db)
        self.assertIn(exc, ((0, 0), (1, 1)))
        if exc == (0, 0):
            self.assertEqual(da, db)
            self.assertTrue(
                ((i in da.alpha) and (k in da.alpha)) or ((i in da.beta) and (k in da.beta))
            )
        elif exc == (1, 1):
            (dsa, dta), (dsb, dtb) = da, db
            hs, ps = PhaseIdx.single_exc_no_phase(dsa, dsb)
            ht, pt = PhaseIdx.single_exc_no_phase(dta, dtb)
            self.assertEqual(sorted((hs, ps)), sorted((i, k)))
            self.assertEqual(sorted((ht, pt)), sorted((i, k)))

    def check_pair_idx_G(self, dadb, idx):
        da, db = dadb
        self.assertEqual(integral_category(*idx), "G")
        i, j, k, l = idx
        exc = Excitation.exc_degree(da, db)
        self.assertIn(exc, ((1, 1), (2, 0), (0, 2)))
        if exc == (1, 1):
            (dsa, dta), (dsb, dtb) = da, db
            hs, ps = PhaseIdx.single_exc_no_phase(dsa, dsb)
            ht, pt = PhaseIdx.single_exc_no_phase(dta, dtb)
            self.assertEqual(
                sorted((sorted((hs, ps)), sorted((ht, pt)))),
                sorted((sorted((i, k)), sorted((j, l)))),
            )
        else:
            if exc == (2, 0):
                (dsa, _), (dsb, _) = da, db
            elif exc == (0, 2):
                (_, dsa), (_, dsb) = da, db
            h1, h2, p1, p2 = PhaseIdx.double_exc_no_phase(dsa, dsb)
            self.assertIn(
                sorted((sorted((h1, h2)), sorted((p1, p2)))),
                (
                    sorted((sorted((i, j)), sorted((k, l)))),
                    sorted((sorted((i, l)), sorted((k, j)))),
                ),
            )


class Test_Minimal(Timing, unittest.TestCase, Test_Category):
    @staticmethod
    def simplify_indices(l):
        d = defaultdict(int)
        for (a, b), idx, phase in l:
            key = ((a, b), compound_idx4(*idx))
            d[key] += phase
        return sorted((ab, idx, phase) for (ab, idx), phase in d.items() if phase)

    @property
    def psi_and_integral(self):
        # 4 Electron in 4 Orbital
        # I'm stupid so let's do the product
        psi = [Determinant((0, 1), (0, 1))]
        for det in Excitation(4).get_chunk_of_connected_determinants(psi):
            psi += det
        d_two_e_integral = {}
        for i, j, k, l in product(range(4), repeat=4):
            d_two_e_integral[compound_idx4(i, j, k, l)] = 1
        return psi, d_two_e_integral

    @property
    def psi_and_integral_PT2(self):
        # minimal psi_and_integral, psi_i != psi_j
        psi_i = [Determinant((0, 1), (0, 1)), Determinant((1, 2), (1, 2))]
        psi_j = list(chain.from_iterable(Excitation(4).get_chunk_of_connected_determinants(psi_i)))
        _, d_two_e_integral = self.psi_and_integral
        return psi_i, psi_j, d_two_e_integral

    def test_equivalence(self):
        # Does `integral` and `determinant` driven produce the same H
        psi, d_two_e_integral = self.psi_and_integral

        h = Hamiltonian_two_electrons_determinant_driven(d_two_e_integral)
        determinant_driven_indices = self.simplify_indices(h.H_indices(psi, psi))

        h = Hamiltonian_two_electrons_integral_driven(d_two_e_integral)
        integral_driven_indices = self.simplify_indices(h.H_indices(psi, psi))
        self.assertListEqual(determinant_driven_indices, integral_driven_indices)

    def test_equivalence_PT2(self):
        # Does `integral` and `determinant` driven produce the same H in the case where psi_i != psi_j
        # Test integral-driven matrix construction in case of PT2
        psi_i, psi_j, d_two_e_integral = self.psi_and_integral_PT2

        h = Hamiltonian_two_electrons_determinant_driven(d_two_e_integral)
        determinant_driven_indices = self.simplify_indices(h.H_indices(psi_i, psi_j))

        h = Hamiltonian_two_electrons_integral_driven(d_two_e_integral)
        integral_driven_indices = self.simplify_indices(h.H_indices(psi_i, psi_j))
        self.assertListEqual(determinant_driven_indices, integral_driven_indices)

    def test_category(self):
        # Does the assumtion of your Ingral category holds
        psi, d_two_e_integral = self.psi_and_integral
        h = Hamiltonian_two_electrons_integral_driven(d_two_e_integral)
        integral_driven_indices = self.simplify_indices(h.H_indices(psi, psi))
        for (a, b), idx, phase in integral_driven_indices:
            i, j, k, l = compound_idx4_reverse(idx)
            category = integral_category(i, j, k, l)
            getattr(self, f"check_pair_idx_{category}")((psi[a], psi[b]), (i, j, k, l))


class Test_Constrained_Excitation(Timing, unittest.TestCase):
    @property
    def n_orb(self):
        # lol
        return 6

    @cached_property
    def exci(self):
        # Cached instance of exci class
        return Excitation(self.n_orb)

    @property
    def reference_det(self):
        # Single reference determinant to generate connected space from
        return Determinant((0, 1, 2), (0, 1, 2))

    @cached_property
    def connected_space(self):
        # Lets do 6 electrons (3 a + 3 b) in 6 (spatial) orbitals
        # From one reference det, generate the whole connected space
        psi = [self.reference_det]
        psi_connected = []
        for det in self.exci.get_chunk_of_connected_determinants(psi):
            psi_connected += det
        return psi_connected

    @cached_property
    def generate_constraints(self):
        # Just a call to Excitation class to make things easier
        return self.exci.generate_constraints(3, 3)

    def check_constraint(self, det: Determinant, spin="alpha"):
        # Give me a determinant. What constraint does it satisfy? (Default constraint level m=3 -> Triplets)
        # Default is that constraints are determined by top m alpha electrons
        spindet = getattr(det, spin)
        con = spindet[-3:]
        return con

    @cached_property
    def connected_det_by_constraint(self):
        # Bin each connected determinant by constraint
        psi_connected = self.connected_space
        all_constraints = self.generate_constraints
        # Initialize defaultdict with keys as all possible constraints (accounts for case when no determinant satisfies a constraint)
        d = defaultdict(list, {con: [] for con in all_constraints})

        for _, det in enumerate(psi_connected):
            # What triplet constraint does this det satisfy?
            con = self.check_constraint(det, 3)
            d[con].append(det)
        return d

    def test_constrained_excitation_from_det(self):
        # Define ref determinant
        psi = self.reference_det
        # We need two things:
        #   1. When we generate determinants by constraint, do we exhaustively generate the connected space?
        #   2. Are the subsets of the connected space generated by constraint disjoint?
        # i.e., the constraint-based excitations form a complete partitioning of the connected space...
        for spin in ["alpha", "beta"]:
            # This dict will be for checking if the partitioning is disjoint
            d = defaultdict(list, {con: [] for con in self.generate_constraints})
            # Here, we will store the determinants by constraint for checking if we exhaustively generate the connected space
            psi_connected_by_constraint = []
            # So, let's loop through constraints...
            for con in self.generate_constraints:
                # And generate all excitation for this particular constraint
                for (
                    constrained_excitation
                ) in self.exci.triplet_constrained_single_excitations_from_det(psi, con, spin):
                    # d is for making sure the partitioning is disjoint
                    d[con].append(constrained_excitation)
                    # (Per constraint) Multiple internal determinant might touch one excited
                    # This is OK, so only tack it on if we haven't seen it yet in this pass
                    if constrained_excitation not in psi_connected_by_constraint:
                        # this wf is for checking that we exhaustively generate the space
                        psi_connected_by_constraint.append(constrained_excitation)

                for (
                    constrained_excitation
                ) in self.exci.triplet_constrained_double_excitations_from_det(psi, con, spin):
                    d[con].append(constrained_excitation)
                    if constrained_excitation not in psi_connected_by_constraint:
                        psi_connected_by_constraint.append(constrained_excitation)

            # Lets generate the full connected space for reference
            psi_connected_ref = self.connected_space
            print(len(psi_connected_ref), len(psi_connected_by_constraint))

            # Are these constraints disjoint??
            for ref_con, dets in d.items():
                for con in self.generate_constraints:
                    if ref_con != con:
                        for det in dets:
                            assert det not in d[con]

            # Do we exhaustively generate the onnected space?
            self.assertListEqual(sorted(psi_connected_ref), sorted(psi_connected_by_constraint))

    def test_constrained_excitation_from_wf(self):
        # Define ref determinant
        psi = [self.reference_det, Determinant((3, 4, 5), (0, 1, 2))]
        # We need two things:
        #   1. When we generate determinants by constraint, do we exhaustively generate the connected space?
        #   2. Are the subsets of the connected space generated by constraint disjoint?
        # i.e., the constraint-based excitations form a complete partitioning of the connected space...
        for spin in ["alpha", "beta"]:
            # This dict will be for checking if the partitioning is disjoint
            d = defaultdict(list, {con: [] for con in self.generate_constraints})
            # Here, we will store the determinants by constraint for checking if we exhaustively generate the connected space
            psi_connected_by_constraint = []
            # So, let's loop through constraints...
            # This function takes aa list of constraints
            for constrained_excitation in self.exci.gen_all_connected_by_triplet_constraints(
                psi, self.generate_constraints, spin
            ):
                con = self.check_constraint(constrained_excitation)
                # (Per constraint) Multiple internal determinant might touch one excited
                # This is OK, so only tack it on if we haven't seen it yet in this pass
                d[con].append(constrained_excitation)
                if constrained_excitation not in psi_connected_by_constraint:
                    psi_connected_by_constraint.append(constrained_excitation)

            # Lets generate the full connected space for reference
            psi_connected_ref = []
            for det in self.exci.get_chunk_of_connected_determinants(psi):
                psi_connected_ref += det
            print(len(psi_connected_ref), len(psi_connected_by_constraint))

            # Are these constraints disjoint??
            for ref_con, dets in d.items():
                for con in self.generate_constraints:
                    if ref_con != con:
                        for det in dets:
                            assert det not in d[con]

            # Do we exhaustively generate the onnected space?
            self.assertListEqual(sorted(psi_connected_ref), sorted(psi_connected_by_constraint))


class Test_Integral_Driven_Categories(Test_Minimal):
    @property
    def integral_by_category(self):
        # Bin each integral (with the 'idx4' representation) by integrals category
        """
        >>> Test_Integral_Driven_Categories().integral_by_category['A']
        [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)]
        """
        psi, d_two_e_integral = self.psi_and_integral
        d = defaultdict(list)
        for idx in d_two_e_integral:
            i, j, k, l = compound_idx4_reverse(idx)
            cat = integral_category(i, j, k, l)
            d[cat].append((i, j, k, l))
        return d

    @property
    def reference_indices_by_category(self):
        # Bin the indices (ab, idx4, phase) of the reference determinant implemetation by integrals category
        """
        >>> len(Test_Integral_Driven_Categories().reference_indices_by_category['C'])
        264
        """
        psi, _ = self.psi_and_integral
        indices = Hamiltonian_two_electrons_determinant_driven.H_indices(psi, psi)
        d = defaultdict(list)
        for ab, (i, j, k, l), phase in indices:
            p, q, r, s = canonical_idx4(i, j, k, l)
            cat = integral_category(p, q, r, s)
            d[cat].append((ab, (p, q, r, s), phase))

        for k in d:
            d[k] = self.simplify_indices(d[k])
        return d

    @property
    def reference_indices_by_category_PT2(self):
        # Bin the indices (ab, idx4, phase) of the reference determinant implemetation by integrals category
        """
        >>> len(Test_Integral_Driven_Categories().reference_indices_by_category['C'])
        264
        """
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        indices = Hamiltonian_two_electrons_determinant_driven.H_indices(psi_i, psi_j)
        d = defaultdict(list)
        for ab, (i, j, k, l), phase in indices:
            p, q, r, s = canonical_idx4(i, j, k, l)
            cat = integral_category(p, q, r, s)
            d[cat].append((ab, (p, q, r, s), phase))

        for k in d:
            d[k] = self.simplify_indices(d[k])
        return d

    def test_category_A(self):
        psi, _ = self.psi_and_integral
        indices = []
        det_to_index = {det: i for i, det in enumerate(psi)}
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["A"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_A(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["A"])

    def test_category_A_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["A"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_A(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, [])

    def test_category_B(self):
        psi, _ = self.psi_and_integral
        indices = []
        det_to_index = {det: i for i, det in enumerate(psi)}
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["B"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_B(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["B"])

    def test_category_B_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["B"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_B(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, [])

    def test_category_C(self):
        psi, _ = self.psi_and_integral
        det_to_index = {det: i for i, det in enumerate(psi)}
        indices = []
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["C"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_C(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ, Excitation(4)
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["C"])

    def test_category_C_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["C"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_C(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, Excitation(4)
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, self.reference_indices_by_category_PT2["C"])

    def test_category_D(self):
        psi, _ = self.psi_and_integral
        det_to_index = {det: i for i, det in enumerate(psi)}
        indices = []
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["D"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_D(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ, Excitation(4)
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["D"])

    def test_category_D_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["D"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_D(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, Excitation(4)
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, self.reference_indices_by_category_PT2["D"])

    def test_category_E(self):
        psi, _ = self.psi_and_integral
        det_to_index = {det: i for i, det in enumerate(psi)}
        indices = []
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["E"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_E(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ, Excitation(4)
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["E"])

    def test_category_E_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["E"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_E(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, Excitation(4)
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, self.reference_indices_by_category_PT2["E"])

    def test_category_F(self):
        psi, _ = self.psi_and_integral
        det_to_index = {det: i for i, det in enumerate(psi)}
        indices = []
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["F"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_F(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ, Excitation(4)
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["F"])

    def test_category_F_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["F"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_F(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, Excitation(4)
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, self.reference_indices_by_category_PT2["F"])

    def test_category_G(self):
        psi, _ = self.psi_and_integral
        det_to_index = {det: i for i, det in enumerate(psi)}
        indices = []
        (
            spindet_a_occ,
            spindet_b_occ,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi)
        for i, j, k, l in self.integral_by_category["G"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_G(
                (i, j, k, l), psi, det_to_index, spindet_a_occ, spindet_b_occ, Excitation(4)
            ):
                indices.append(((a, b), (i, j, k, l), phase))
        indices = self.simplify_indices(indices)
        self.assertListEqual(indices, self.reference_indices_by_category["G"])

    def test_category_G_PT2(self):
        psi_i, psi_j, _ = self.psi_and_integral_PT2
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        indices_PT2 = []
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        for i, j, k, l in self.integral_by_category["G"]:
            for (a, b), phase in Hamiltonian_two_electrons_integral_driven.category_G(
                (i, j, k, l), psi_i, det_to_index_j, spindet_a_occ_i, spindet_b_occ_i, Excitation(4)
            ):
                indices_PT2.append(((a, b), (i, j, k, l), phase))
        indices_PT2 = self.simplify_indices(indices_PT2)
        self.assertListEqual(indices_PT2, self.reference_indices_by_category_PT2["G"])


class Test_VariationalPowerplant:
    def test_c2_eq_dz_3(self):
        fcidump_path = "c2_eq_hf_dz.fcidump*"
        wf_path = "c2_eq_hf_dz_3.*.wf*"
        E_ref = load_eref("data/c2_eq_hf_dz_3.*.ref*")
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_c2_eq_dz_4(self):
        fcidump_path = "c2_eq_hf_dz.fcidump*"
        wf_path = "c2_eq_hf_dz_4.*.wf*"
        E_ref = load_eref("data/c2_eq_hf_dz_4.*.ref*")
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_1det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        E_ref = -198.646096743145
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.10det.wf"
        E_ref = -198.548963
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_30det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.30det.wf"
        E_ref = -198.738780989106
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_161det(self):
        fcidump_path = "f2_631g.161det.fcidump"
        wf_path = "f2_631g.161det.wf"
        E_ref = -198.8084269796
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_296det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.296det.wf"
        E_ref = -198.682736076007
        E = self.load_and_compute(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)


def load_and_compute(fcidump_path, wf_path, driven_by):
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
    # Load wave function
    psi_coef, psi_det = load_wf(f"data/{wf_path}")
    # Computation of the Energy of the input wave function (variational energy)
    comm = MPI.COMM_WORLD
    lewis = Hamiltonian_generator(comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by)
    return Powerplant_manager(comm, lewis).E(psi_coef)


class Test_VariationalPowerplant_Determinant(Timing, unittest.TestCase, Test_VariationalPowerplant):
    def load_and_compute(self, fcidump_path, wf_path):
        return load_and_compute(fcidump_path, wf_path, "determinant")


class Test_VariationalPowerplant_Integral(Timing, unittest.TestCase, Test_VariationalPowerplant):
    def load_and_compute(self, fcidump_path, wf_path):
        return load_and_compute(fcidump_path, wf_path, "integral")


class Test_VariationalPT2Powerplant:
    def test_f2_631g_1det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        E_ref = -0.367587988032339
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_2det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.2det.wf"
        E_ref = -0.253904406461572
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.10det.wf"
        E_ref = -0.24321128
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_28det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.28det.wf"
        E_ref = -0.244245625775444
        E = self.load_and_compute_pt2(fcidump_path, wf_path)
        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_10det_chunked(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.10det.wf"
        E_ref = -0.24321128

        # Connected space has 74262 determinants
        # Compute E_pt2 energy 5000 dets at a time
        chunk_size = 5000
        E = self.load_and_compute_pt2(fcidump_path, wf_path, chunk_size)
        self.assertAlmostEqual(E_ref, E, places=6)

        # What if chunk_size is > len(psi_connected)?
        chunk_size = 100000
        E = self.load_and_compute_pt2(fcidump_path, wf_path, chunk_size)
        self.assertAlmostEqual(E_ref, E, places=6)

        # What if chunk_size is = len(psi_connected)?
        chunk_size = 74262
        E = self.load_and_compute_pt2(fcidump_path, wf_path, chunk_size)
        self.assertAlmostEqual(E_ref, E, places=6)


def load_and_compute_pt2(fcidump_path, wf_path, driven_by, chunk_size=None):
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
    # Load wave function
    psi_coef, psi_det = load_wf(f"data/{wf_path}")
    # Computation of the Energy of the input wave function (variational energy)
    comm = MPI.COMM_WORLD
    lewis = Hamiltonian_generator(comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by)
    return Powerplant_manager(comm, lewis).E_pt2(psi_coef, chunk_size)


class Test_VariationalPT2_Determinant(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path, chunk_size=None):
        return load_and_compute_pt2(fcidump_path, wf_path, "determinant", chunk_size)


class Test_VariationalPT2_Integral(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path, chunk_size=None):
        return load_and_compute_pt2(fcidump_path, wf_path, "integral", chunk_size)


class Test_Selection(Timing, unittest.TestCase):
    def load(self, fcidump_path, wf_path):
        # Load integrals
        n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        return (
            n_ord,
            psi_coef,
            psi_det,
            Hamiltonian_generator(MPI.COMM_WORLD, E0, d_one_e_integral, d_two_e_integral, psi_det),
        )

    def test_f2_631g_1p0det(self):
        # Verify that selecting 0 determinant is egual that computing the variational energy
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        E_var = Powerplant_manager(lewis.comm, lewis).E(psi_coef)

        E_selection, _, _ = selection_step(lewis.comm, lewis, n_ord, psi_coef, psi_det, 0)

        self.assertAlmostEqual(E_var, E_selection, places=6)

    def test_f2_631g_1p10det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        # No a value optained with QP
        E_ref = -198.72696793971556
        # Selection 10 determinant and check if the result make sence

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        E, _, _ = selection_step(lewis.comm, lewis, n_ord, psi_coef, psi_det, 10)

        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_1p5p5det(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        # We will select 5 determinant, than 5 more.
        # The value is lower than the one optained by selecting 10 deterinant in one go.
        # Indeed, the pt2 get more precise whith the number of selection
        E_ref = -198.73029308564543

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)
        _, psi_coef, psi_det = selection_step(lewis.comm, lewis, n_ord, psi_coef, psi_det, 5)
        # New instance of Hamiltonian_generator
        lewis_new = Hamiltonian_generator(
            lewis.comm, lewis.E0, lewis.d_one_e_integral, lewis.d_two_e_integral, psi_det
        )
        E, psi_coef, psi_det = selection_step(
            lewis_new.comm, lewis_new, n_ord, psi_coef, psi_det, 5
        )

        self.assertAlmostEqual(E_ref, E, places=6)

    def test_f2_631g_1p10det_chunked(self):
        fcidump_path = "f2_631g.FCIDUMP"
        wf_path = "f2_631g.1det.wf"
        # No a value optained with QP
        E_ref = -198.72696793971556
        # Selection 10 determinant

        n_ord, psi_coef, psi_det, lewis = self.load(fcidump_path, wf_path)

        # Chunk the connected space by 1000 at a time, so it doesn't take forever
        E, _, _ = selection_step(lewis.comm, lewis, n_ord, psi_coef, psi_det, 10, 1000)

        self.assertAlmostEqual(E_ref, E, places=6)

        # What if chunk_size > len(psi_connected)?
        E, _, _ = selection_step(lewis.comm, lewis, n_ord, psi_coef, psi_det, 10, 100000)

        self.assertAlmostEqual(E_ref, E, places=6)


if __name__ == "__main__":
    try:
        sys.argv.remove("--profiling")
    except ValueError:
        PROFILING = False
    else:
        PROFILING = True
    unittest.main(failfast=True, verbosity=0)
