from calculations import load_integrals, load_wf, selection_step, Powerplant, Hamiltonian
import unittest

class TestVariationalPowerplant(unittest.TestCase):

    def load_and_compute(self,fcidump_path,wf_path):
        # Load integrals
        N_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        # Computation of the Energy of the input wave function (variational energy)
        lewis = Hamiltonian(d_one_e_integral,d_two_e_integral, E0)
        return Powerplant(lewis, psi_det).E(psi_coef)

    def test_f2_631g_1det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.1det.wf'
        E_ref =  -198.646096743145
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_10det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.10det.wf'
        E_ref =  -198.548963
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_30det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.30det.wf'
        E_ref =  -198.738780989106
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_161det(self):
        fcidump_path='f2_631g.161det.fcidump'
        wf_path='f2_631g.161det.wf'
        E_ref =  -198.8084269796
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_296det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.296det.wf'
        E_ref =  -198.682736076007
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

class TestVariationalPT2Powerplant(unittest.TestCase):

    def load_and_compute_pt2(self,fcidump_path,wf_path):
        # Load integrals
        N_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        # Computation of the Energy of the input wave function (variational energy)
        lewis = Hamiltonian(d_one_e_integral,d_two_e_integral, E0)
        return Powerplant(lewis, psi_det).E_pt2(psi_coef,N_ord)

    def test_f2_631g_1det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.1det.wf'
        E_ref =  -0.367587988032339
        E =  self.load_and_compute_pt2(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_2det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.2det.wf'
        E_ref =  -0.253904406461572
        E =  self.load_and_compute_pt2(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_10det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.10det.wf'
        E_ref =  -0.24321128
        E =  self.load_and_compute_pt2(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_28det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.28det.wf'
        E_ref =  -0.244245625775444
        E =  self.load_and_compute_pt2(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E,places=6)

class TestSelection(unittest.TestCase):

    def load(self,fcidump_path,wf_path):
        # Load integrals
        N_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        return N_ord,psi_coef, psi_det, Hamiltonian(d_one_e_integral,d_two_e_integral, E0)

    def test_f2_631g_1p0det(self):
        # Verify that selecting 0 determinant is egual that computing the variational energy
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.1det.wf'

        N_ord, psi_coef, psi_det, lewis = self.load(fcidump_path,wf_path)
        E_var = Powerplant(lewis, psi_det).E(psi_coef)

        E_selection, _, _ = selection_step(lewis, N_ord, psi_coef, psi_det, 0)

        self.assertAlmostEqual(E_var,E_selection,places=6)


    def test_f2_631g_1p10det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.1det.wf'
        # No a value optained with QP
        E_ref =  -198.72696793971556
        # Selection 10 determinant and check if the result make sence

        N_ord, psi_coef, psi_det, lewis = self.load(fcidump_path,wf_path)
        E, _, _ = selection_step(lewis, N_ord, psi_coef, psi_det, 10)

        self.assertAlmostEqual(E_ref,E,places=6)

    def test_f2_631g_1p5p5det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.1det.wf'
        # We will select 5 determinant, than 5 more.
        # The value is lower than the one optained by selecting 10 deterinant in one go.
        # Indeed, the pt2 get more precise whith the number of selection
        E_ref =  -198.73029308564543

        N_ord, psi_coef, psi_det, lewis = self.load(fcidump_path,wf_path)
        _, psi_coef, psi_det = selection_step(lewis, N_ord, psi_coef, psi_det, 5)

        E, psi_coef, psi_det = selection_step(lewis, N_ord, psi_coef, psi_det, 5)

        self.assertAlmostEqual(E_ref,E,places=6)

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    unittest.main()
