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
        # Na = 3, Nb = 3, Norb = 6 to account for triplet constraints
        psi_i = [Determinant((0, 1, 2), (0, 1, 2)), Determinant((1, 2, 3), (1, 2, 3))]
        psi_j = []
        for i, det in enumerate(psi_i):
            for det_connected in Excitation(6).gen_all_connected_det_from_det(det):
                # Remove determinant who are in psi_det
                if det_connected in psi_i:
                    continue
                # If it's was already generated by an old determinant, just drop it
                if any(Excitation.is_connected(det_connected, d) for d in psi_i[:i]):
                    continue

                psi_j.append(det_connected)

        d_two_e_integral = {}
        for i, j, k, l in product(range(6), repeat=4):
            d_two_e_integral[compound_idx4(i, j, k, l)] = 1
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
    def psi_and_norb_2det(self):
        # Do 5 e, 10 orb
        return 10, [
            Determinant((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
            Determinant((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)),
        ]

    @cached_property
    def psi_connected_2det(self):
        n_orb, psi = self.psi_and_norb_2det
        l_global = []
        for i, det in enumerate(psi):
            for det_connected in Excitation(n_orb).gen_all_connected_det_from_det(det):
                # Remove determinant who are in psi_det
                if det_connected in psi:
                    continue
                # If it's was already generated by an old determinant, just drop it
                if any(Excitation.is_connected(det_connected, d) for d in psi[:i]):
                    continue

                l_global.append(det_connected)

        # Return connected space
        return l_global

    @cached_property
    def connected_by_det(self):
        # For each det in WF, generate all singles and doubles and store with generator
        # Will sort out by constraint later
        n_orb, psi = self.psi_and_norb_2det
        # Initialize empty dict with generators as keys
        d = defaultdict(list, {det_I: [] for det_I in psi})
        for det_I in psi:
            # Generate all singles and doubles of det_I
            det_I_sd = [det_J for det_J in Excitation(n_orb).gen_all_connected_det_from_det(det_I)]
            for det_J in det_I_sd:
                if det_J not in psi:
                    d[det_I].append(det_J)

        return d

    @cached_property
    def connected_by_constraint(self):
        # Bin each connected determinant by constraint
        # Initialize defaultdict with keys as all possible constraints (accounts for case when no determinant satisfies a constraint)
        n_orb, psi = self.psi_and_norb_2det
        d = defaultdict(
            list,
            {
                con: []
                for con in self.generate_all_constraints(len(getattr(psi[0], "alpha")), n_orb)
            },
        )
        for det in self.psi_connected_2det:
            # What triplet constraint does this det satisfy?
            con = self.check_constraint(det, "alpha")
            d[con].append(det)
        return d

    def check_constraint(self, det: Determinant, spin="alpha"):
        # Give me a determinant. What constraint does it satisfy? (What are three most highly occupied alpha spin orbitas)
        spindet = getattr(det, spin)
        # Return constraint as |Spin_determinant|
        return spindet[-3:]

    def generate_all_constraints(self, n_a, n_orb):
        # Just a call to Excitation class to make things easier
        return Excitation(n_orb).generate_all_constraints(n_a, 3)

    def test_constrained_excitations(self, spin="alpha"):
        # 1. For each constraint, pass through wf
        n_orb, psi = self.psi_and_norb_2det
        for C in self.generate_all_constraints(len(getattr(psi[0], spin)), n_orb):
            # Initialiaze default dict with keys as dets in wf, store all constrained singles + doubles with generator
            sd_by_C = defaultdict(list, {det_I: [] for det_I in psi})
            for det_I in psi:
                # For this det_I, generate all singles + doubles subject to current constraint
                sd_by_C[det_I].extend(
                    [
                        det_J
                        for det_J in Excitation(
                            n_orb
                        ).triplet_constrained_double_excitations_from_det(det_I, C, spin)
                        if det_J not in psi
                    ]
                )
                sd_by_C[det_I].extend(
                    [
                        det_J
                        for det_J in Excitation(
                            n_orb
                        ).triplet_constrained_single_excitations_from_det(det_I, C, spin)
                        if det_J not in psi
                    ]
                )
            # For this constraint, get determinants from reference connected space that satisfy C
            # This is for ONE determinant.. So there should be no duplicates in the dict per generator key
            for gen_det_I, det_I_conn_by_C in sd_by_C.items():
                # This list contains all determinants connected to det_I
                ref_det_I_conn = list(self.connected_by_det[gen_det_I])
                # Filter out those that satisfy constraint C
                ref_det_I_conn_by_C = [
                    det_J for det_J in ref_det_I_conn if (self.check_constraint(det_J) == C)
                ]
                self.assertListEqual(sorted(ref_det_I_conn_by_C), sorted(det_I_conn_by_C))


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
    def integral_by_category_PT2(self):
        # Bin each integral (with the 'idx4' representation) by integrals category
        _, _, d_two_e_integral = self.psi_and_integral_PT2
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
        # Will need hash to map yielding determinants to a particular index
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        # Pass over constraints, pass over integrals
        for con in Excitation(6).generate_all_constraints(3):
            indices_PT2_con = []  # Reset list for each constraint
            for i, j, k, l in self.integral_by_category_PT2["C"]:
                for (I, det_J), phase in Hamiltonian_two_electrons_integral_driven.category_C_pt2(
                    (i, j, k, l), psi_i, con, spindet_a_occ_i, spindet_b_occ_i, Excitation(6)
                ):
                    if det_J not in psi_i:
                        indices_PT2_con.append(((I, det_to_index_j[det_J]), (i, j, k, l), phase))
                # `indices_PT2_con` contains all ((I, J), idx, phase) s.to:
                #   1. Integrals idx in category C;
                #   2. Determinants J satisfy constraint con
            indices_PT2_con = self.simplify_indices(indices_PT2_con)
            # Now, get all reference pairs subject to constraint C
            ref_indices_PT2_con = []
            for (I, J), idx, phase in self.reference_indices_by_category_PT2["C"]:
                if Excitation(6).check_constraint(psi_j[J]) == con:
                    ref_indices_PT2_con.append(((I, J), idx, phase))

            self.assertListEqual((indices_PT2_con), (ref_indices_PT2_con))

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
        # Will need hash to map yielding determinants to a particular index
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        # Pass over constraints, pass over integrals
        for con in Excitation(6).generate_all_constraints(3):
            indices_PT2_con = []  # Reset list for each constraint
            for i, j, k, l in self.integral_by_category_PT2["D"]:
                for (I, det_J), phase in Hamiltonian_two_electrons_integral_driven.category_D_pt2(
                    (i, j, k, l), psi_i, con, spindet_a_occ_i, spindet_b_occ_i, Excitation(6)
                ):
                    if det_J not in psi_i:
                        indices_PT2_con.append(((I, det_to_index_j[det_J]), (i, j, k, l), phase))
                # `indices_PT2_con` contains all ((I, J), idx, phase) s.to:
                #   1. Integrals idx in category D;
                #   2. Determinants J satisfy constraint con
            indices_PT2_con = self.simplify_indices(indices_PT2_con)
            # Now, get all reference pairs subject to constraint C
            ref_indices_PT2_con = []
            for (I, J), idx, phase in self.reference_indices_by_category_PT2["D"]:
                if Excitation(6).check_constraint(psi_j[J]) == con:
                    ref_indices_PT2_con.append(((I, J), idx, phase))

            self.assertListEqual((indices_PT2_con), (ref_indices_PT2_con))

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
        # Will need hash to map yielding determinants to a particular index
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        # Pass over constraints, pass over integrals
        for con in Excitation(6).generate_all_constraints(3):
            indices_PT2_con = []  # Reset list for each constraint
            for i, j, k, l in self.integral_by_category_PT2["E"]:
                for (I, det_J), phase in Hamiltonian_two_electrons_integral_driven.category_E_pt2(
                    (i, j, k, l), psi_i, con, spindet_a_occ_i, spindet_b_occ_i, Excitation(6)
                ):
                    if det_J not in psi_i:
                        indices_PT2_con.append(((I, det_to_index_j[det_J]), (i, j, k, l), phase))
                # `indices_PT2_con` contains all ((I, J), idx, phase) s.to:
                #   1. Integrals idx in category E;
                #   2. Determinants J satisfy constraint con
            indices_PT2_con = self.simplify_indices(indices_PT2_con)
            # Now, get all reference pairs subject to constraint C
            ref_indices_PT2_con = []
            for (I, J), idx, phase in self.reference_indices_by_category_PT2["E"]:
                if Excitation(6).check_constraint(psi_j[J]) == con:
                    ref_indices_PT2_con.append(((I, J), idx, phase))

            self.assertListEqual((indices_PT2_con), (ref_indices_PT2_con))

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
        # Will need hash to map yielding determinants to a particular index
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        # Pass over constraints, pass over integrals
        for con in Excitation(6).generate_all_constraints(3):
            indices_PT2_con = []  # Reset list for each constraint
            for i, j, k, l in self.integral_by_category_PT2["F"]:
                for (I, det_J), phase in Hamiltonian_two_electrons_integral_driven.category_F_pt2(
                    (i, j, k, l), psi_i, con, spindet_a_occ_i, spindet_b_occ_i, Excitation(6)
                ):
                    if det_J not in psi_i:
                        indices_PT2_con.append(((I, det_to_index_j[det_J]), (i, j, k, l), phase))
                # `indices_PT2_con` contains all ((I, J), idx, phase) s.to:
                #   1. Integrals idx in category F;
                #   2. Determinants J satisfy constraint con
            indices_PT2_con = self.simplify_indices(indices_PT2_con)
            # Now, get all reference pairs subject to constraint C
            ref_indices_PT2_con = []
            for (I, J), idx, phase in self.reference_indices_by_category_PT2["F"]:
                if Excitation(6).check_constraint(psi_j[J]) == con:
                    ref_indices_PT2_con.append(((I, J), idx, phase))

            self.assertListEqual((indices_PT2_con), (ref_indices_PT2_con))

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
        # Will need hash to map yielding determinants to a particular index
        det_to_index_j = {det: i for i, det in enumerate(psi_j)}
        (
            spindet_a_occ_i,
            spindet_b_occ_i,
        ) = H_indices_generator.get_spindet_a_occ_spindet_b_occ(psi_i)
        # Pass over constraints, pass over integrals
        for con in Excitation(6).generate_all_constraints(3):
            indices_PT2_con = []  # Reset list for each constraint
            for i, j, k, l in self.integral_by_category_PT2["G"]:
                for (I, det_J), phase in Hamiltonian_two_electrons_integral_driven.category_G_pt2(
                    (i, j, k, l), psi_i, con, spindet_a_occ_i, spindet_b_occ_i, Excitation(6)
                ):
                    if det_J not in psi_i:
                        indices_PT2_con.append(((I, det_to_index_j[det_J]), (i, j, k, l), phase))
                # `indices_PT2_con` contains all ((I, J), idx, phase) s.to:
                #   1. Integrals idx in category G;
                #   2. Determinants J satisfy constraint con
            indices_PT2_con = self.simplify_indices(indices_PT2_con)
            # Now, get all reference pairs subject to constraint C
            ref_indices_PT2_con = []
            for (I, J), idx, phase in self.reference_indices_by_category_PT2["G"]:
                if Excitation(6).check_constraint(psi_j[J]) == con:
                    ref_indices_PT2_con.append(((I, J), idx, phase))

            self.assertListEqual((indices_PT2_con), (ref_indices_PT2_con))


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


def load_and_compute_pt2(fcidump_path, wf_path, driven_by):
    # Load integrals
    n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
    # Load wave function
    psi_coef, psi_det = load_wf(f"data/{wf_path}")
    # Computation of the Energy of the input wave function (variational energy)
    comm = MPI.COMM_WORLD
    lewis = Hamiltonian_generator(comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by)
    return Powerplant_manager(comm, lewis).E_pt2(psi_coef)


class Test_VariationalPT2_Determinant(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path):
        return load_and_compute_pt2(fcidump_path, wf_path, "determinant")


class Test_VariationalPT2_Integral(Timing, unittest.TestCase, Test_VariationalPT2Powerplant):
    def load_and_compute_pt2(self, fcidump_path, wf_path):
        return load_and_compute_pt2(fcidump_path, wf_path, "integral")


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


if __name__ == "__main__":
    try:
        sys.argv.remove("--profiling")
    except ValueError:
        PROFILING = False
    else:
        PROFILING = True
    unittest.main(failfast=True, verbosity=0)
