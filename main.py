#!/usr/bin/env python3

# Types
# -----
from typing import Tuple, Dict, NewType, NamedTuple, List, Set, Iterator

# Orbital index (1,2,...,Norb)
OrbitalIdx = NewType('OrbitalIdx', int)

# Two-electron integral :
# $<ij|kl> = \int \int \phi_i(r_1) \phi_j(r_2) \frac{1}{|r_1 - r_2|} \phi_k(r_1) \phi_l(r_2) dr_1 dr_2$
Two_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]

# One-electron integral :
# $<i|h|k> = \int \phi_i(r) (-\frac{1}{2} \Delta + V_en ) \phi_k(r) dr$
One_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]

Determinant_Spin = Tuple[OrbitalIdx, ...]
Determinant_Spin_Set = Set[OrbitalIdx]
class Determinant(NamedTuple):
    '''Slater determinant: Product of 2 determinants.
       One for $\alpha$ electrons and one for \beta electrons.'''
    alpha: Determinant_Spin
    beta: Determinant_Spin

# ~
# Integrals of the Hamiltonian over molecular orbitals
# ~
def load_integrals(fcidump_path) -> Tuple[int, float, One_electron_integral, Two_electron_integral]:
    '''Read all the Hamiltonian integrals from the data file.
       Returns: (E0, d_one_e_integral, d_two_e_integral).
       E0 : a float containing the nuclear repulsion energy (V_nn),
       d_one_e_integral : a dictionary of one-electron integrals,
       d_two_e_integral : a dictionary of two-electron integrals.
       '''
    
    # Use an iterator to avoid storing everything in memory twice.
    f = open(fcidump_path)

    # Only non-zero integrals are stored in the fci_dump.
    # Hence we use a defaultdict to handle the sparsity
    N_orb = int(next(f).split()[2])

    for _ in range(3):
        next(f)

    from collections import defaultdict
    d_one_e_integral = defaultdict(int)
    d_two_e_integral = defaultdict(int)

    for line in f:
        v, *l = line.split()
        v = float(v)
        # Transform from Mulliken (ik|jl) to Dirac's <ij|kl> notation
        # (swap indices)
        i,k,j,l = list(map(int, l)) 
 
        if i == 0:
            E0 = v
        elif j == 0:
            # One-electron integrals are symmetric (when real, not complex)
            d_one_e_integral[ (i,k) ] = v            
            d_one_e_integral[ (k,i) ] = v
        else:
            # Two-electron integrals have many permutation symmetries:
            # Exchange r1 and r2 (indices i,k and j,l)
            # Exchange i,k 
            # Exchange j,l
            d_two_e_integral[ (i,j,k,l) ] = v
            d_two_e_integral[ (i,l,k,j) ] = v
            d_two_e_integral[ (j,i,l,k) ] = v
            d_two_e_integral[ (j,k,l,i) ] = v
            d_two_e_integral[ (k,j,i,l) ] = v
            d_two_e_integral[ (k,l,i,j) ] = v
            d_two_e_integral[ (l,i,j,k) ] = v
            d_two_e_integral[ (l,k,j,i) ] = v

    f.close()

    return N_orb, E0, d_one_e_integral, d_two_e_integral

def load_wf(path_wf) -> Tuple[ List[float] , List[Determinant] ]  :
    '''Read the input file :
       Representation of the Slater determinants (basis) and
       vector of coefficients in this basis (wave function).'''

    with open(path_wf) as f:
        data = f.read().split()

    def decode_det(str_):
        for i,v in enumerate(str_, start=1):
            if v == '+':
                yield i

    def grouper(iterable, n):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return zip(*args)

    det = []; psi_coef = []
    for (coef, det_i, det_j) in grouper(data,3):
        psi_coef.append(float(coef))
        det.append ( Determinant( tuple(decode_det(det_i)), tuple(decode_det(det_j) ) ) )

    # Normalize psi_coef
    from math import sqrt
    norm = sqrt(sum(c*c for c in psi_coef))
    psi_coef = [c / norm for c in psi_coef]

    return psi_coef, det


from itertools import chain, product
from itertools import combinations, product

class Excitation(object):

    def __init__(self, N_orb):
        self.all_orbs = set(range(1,N_orb+1))

    def gen_all_exc_from_detspin(self, detspin: Determinant_Spin, ed: int) -> Iterator:
        '''
        Generate list of pair -> hole from a determinant spin.

        >>> sorted(Excitation(4).gen_all_exc_from_detspin( (1,2),2))
        [((1, 2), (3, 4))]
        '''
        holes = combinations(detspin,ed)
        not_detspin = self.all_orbs - set(detspin)
        parts = combinations(not_detspin,ed)
        return product(holes,parts)

    def gen_all_connected_detspin_from_detspin(self, detspin: Determinant_Spin, ed: int) -> Iterator:
        '''
        Generate all the posible spin determinant relative to a excitation degree

        >>> sorted(Excitation(3).gen_all_connected_detspin_from_detspin( (1,2), 1))
        [(1, 3), (2, 3)]

        '''
        def apply_excitation(exc: Tuple[Tuple[int, ...],Tuple[int, ...]])-> Determinant_Spin_Set:
            # Warning use global variable detspin. 
            lh,lp = exc
            s = (set(detspin) - set(lh)) | set(lp)
            return tuple(s)

        l_exc = self.gen_all_exc_from_detspin(detspin, ed)
        return map(apply_excitation, l_exc)

    def gen_all_connected_det_from_det(self,det: Determinant)->Iterator:
        '''
        Generate all the determinant who are single or double exictation (aka connected) from the input determinant

        >>> sorted(Excitation(3).gen_all_connected_det_from_det( Determinant( (1,2), (1,) )))
        [Determinant(alpha=(1, 3), beta=(2,)), Determinant(alpha=(1, 3), beta=(3,)), 
         Determinant(alpha=(2, 3), beta=(2,)), Determinant(alpha=(2, 3), beta=(3,))]
        '''

        # All single exitation from alpha or for beta determinant
        # Then the production of the alpha, and beta (it's a double)
        # Then the double exitation form alpha or beta
        det_alpha, det_beta = det

        l_single_a = self.gen_all_connected_detspin_from_detspin(det_alpha, 1)
        l_double_aa =self.gen_all_connected_detspin_from_detspin(det_alpha, 2)

        s_a = ( Determinant(det, det_beta) for det in chain(l_single_a,l_double_aa) )

        l_single_b = self.gen_all_connected_detspin_from_detspin(det_beta, 1)
        l_double_bb =self.gen_all_connected_detspin_from_detspin(det_beta, 2)

        s_b = ( Determinant(det_alpha, det) for det in chain(l_single_b,l_double_bb) )

        l_double_ab = product(l_single_a,l_single_b)

        s_ab = ( Determinant(a,b) for a,b in l_double_ab )

        return chain(s_a,s_b,s_ab)

    def gen_all_connected_determinant_from_psi(self,psi: List[Determinant])-> Set:
        '''
        >>> d1 = Determinant( (1,2), (1,) ) ; d2 = Determinant( (1,3), (1,) )
        >>> len(Excitation(4).gen_all_connected_determinant_from_psi( [ d1,d2 ] ))
        20
        '''
        return set(chain.from_iterable(map(self.gen_all_connected_det_from_det,psi)))


def get_exc_degree(det_i: Determinant, det_j: Determinant) -> Tuple[int,int]:
    '''Compute the excitation degree, the number of orbitals which differ
       between the two determinants.
    >>> get_exc_degree(Determinant(alpha=(1, 2), beta=(1, 2)),
    ...                Determinant(alpha=(1, 3), beta=(5, 7)) )
    (1, 2)
    '''
    ed_up =  len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
    ed_dn =  len(set(det_i.beta).symmetric_difference(set(det_j.beta))) // 2
    return (ed_up, ed_dn)



# Now, we consider the Hamiltonian matrix in the basis of Slater determinants.
# Slater-Condon rules are used to compute the matrix elements <I|H|J> where I
# and J are Slater determinants.
# 
# ~
# Slater-Condon Rules
# ~
#
# https://en.wikipedia.org/wiki/Slater%E2%80%93Condon_rules
# https://arxiv.org/abs/1311.6244
#
# * H is symmetric
# * If I and J differ by more than 2 orbitals, <I|H|J> = 0, so the number of
#   non-zero elements of H is bounded by N_det x ( N_alpha x (N_orb - N_alpha))^2,
#   where N_det is the number of determinants, N_alpha is the number of
#   alpha-spin electrons (N_alpha >= N_beta), and N_orb is the number of
#   molecular orbitals.  So the number of non-zero elements scales linearly with
#   the number of selected determinant.
#

from itertools import takewhile
class Hamiltonian(object):

    def __init__(self, N_orb, d_one_e_integral: One_electron_integral, d_two_e_integral: Two_electron_integral, E0: float):
        self.d_one_e_integral = d_one_e_integral
        self.d_two_e_integral = d_two_e_integral
        self.E0 = E0
        self.N_orb = N_orb

    def H_one_e(self, i: OrbitalIdx, j: OrbitalIdx) -> float :
        '''One-electron part of the Hamiltonian: Kinetic energy (T) and
           Nucleus-electron potential (V_{en}). This matrix is symmetric.'''
        return self.d_one_e_integral[ (i,j) ]
    

    def H_two_e(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        '''Assume that *all* the integrals are in 
           `d_two_e_integral` In this function, for simplicity we don't use any
           symmetry sparse representation.  For real calculations, symmetries and
           storing only non-zeros needs to be implemented to avoid an explosion of
           the memory requirements.'''
        return self.d_two_e_integral[ (i,j,k,l) ]

    def get_phase_idx_single_exc(self, det_i: Determinant_Spin, det_j: Determinant_Spin) -> Tuple[int,int,int]:
        '''phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
           h is occupied only in I
           p is occupied only in J'''
    
        h, = set(det_i) - set(det_j)
        p, = set(det_j) - set(det_i)
    
        phase=1
        for det, idx in ((det_i,h),(det_j,p)):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase
    
        return (phase,h,p)
    
    def get_phase_idx_double_exc(self, det_i: Determinant_Spin, det_j: Determinant_Spin) -> Tuple[int,int,int,int,int]:
        '''phase, holes, particles of <I|H|J> when I and J differ by exactly two orbitals
           h1, h2 are occupied only in I
           p1, p2 are occupied only in J'''
    
        #Holes
        h1, h2 = sorted(set(det_i) - set(det_j))
        
        #Particles
        p1, p2 = sorted(set(det_j) - set(det_i))
    
        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = 1
        for det,idx in ( (det_i,h1), (det_j,p1),  (det_j,p2), (det_i,h2) ):
            for _ in takewhile(lambda x: x != idx, det):
                phase = -phase
                    
        # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
        if ( min(h2, p2) <  max(h1, p1) ) != ( h2 < p1 or p2 < h1):
            phase = -phase


        return (phase, h1, h2, p1, p2)


    def H_i_i(self, det_i: Determinant) -> float:
        from itertools   import product

        '''Diagonal element of the Hamiltonian : <I|H|I>.'''
        res  = self.E0
        res += sum(self.H_one_e(i,i) for i in det_i.alpha)
        res += sum(self.H_one_e(i,i) for i in det_i.beta)
        
        res += sum(self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) for i,j in product(det_i.alpha, det_i.alpha)) / 2.
        res += sum(self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) for i,j in product(det_i.beta, det_i.beta)) / 2.
           
        res += sum(self.H_two_e(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))
     
        return res


    def H_i_j_single(self, detspin_i: Determinant_Spin, detspin_j: Determinant_Spin, detspin_k: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one orbital.'''
        
        # Interaction 
        phase, m, p = self.get_phase_idx_single_exc(detspin_i,detspin_j)
        res = self.H_one_e(m,p)
    
        res += sum ( self.H_two_e(m,i,p,i) - self.H_two_e(m,i,i,p) for i in detspin_i)
        res += sum ( self.H_two_e(m,i,p,i) for i in detspin_k)
        return phase * res

    def H_i_j_doubleAA(self, li: Determinant_Spin, lj: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly two orbitals within
           the same spin.'''
        
        phase, h1, h2, p1, p2 = self.get_phase_idx_double_exc(li,lj)
        
        res = self.H_two_e(h1, h2, p1, p2) -  self.H_two_e(h1, h2, p2, p1)
        
        return phase * res
        
    
    def H_i_j_doubleAB(self, det_i: Determinant, det_j: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
           one beta spin-orbital.'''
        
        phaseA, hA, pA = self.get_phase_idx_single_exc(det_i.alpha, det_j.alpha)
        phaseB, hB, pB = self.get_phase_idx_single_exc(det_i.beta , det_j.beta)
        
        phase = phaseA * phaseB
        res = self.H_two_e(hA, hB, pA, pB)
      
        return phase * res
 
    def H_i_j(self, det_i: Determinant, det_j: Determinant) -> float:
        '''General function to dispatch the evaluation of H_ij'''
    
        ed_up, ed_dn = get_exc_degree(det_i, det_j)
        # Same determinant -> Diagonal element
        if (ed_up,ed_dn) == (0,0):
            return self.H_i_i(det_i)
    
        # Single excitation
        elif (ed_up, ed_dn) == (1, 0):
            return self.H_i_j_single(det_i.alpha, det_j.alpha, det_i.beta)
        elif (ed_up, ed_dn) == (0, 1):
            return self.H_i_j_single(det_i.beta, det_j.beta, det_i.alpha)
    
        # Double excitation of same spin
        elif (ed_up, ed_dn) == (2, 0):
            return self.H_i_j_doubleAA(det_i.alpha,det_j.alpha)
        elif (ed_up, ed_dn) == (0, 2):
            return self.H_i_j_doubleAA(det_i.beta,det_j.beta)
    
        # Double excitation of opposite spins
        elif (ed_up, ed_dn) == (1, 1):
            return self.H_i_j_doubleAB(det_i, det_j)
    
        # More than doubly excited, zero
        else:
            return 0.

def E_var(E0, N_ord, psi_coef, psi_det, d_one_e_integral,  d_two_e_integral):
    lewis = Hamiltonian(N_ord, d_one_e_integral,d_two_e_integral, E0)
    return sum(psi_coef[i] * psi_coef[j] * lewis.H_i_j(det_i,det_j) for (i,det_i),(j,det_j) in product(enumerate(psi_det),enumerate(psi_det)) )

import unittest
class TestVariationalEnergy(unittest.TestCase):

    def load_and_compute(self,fcidump_path,wf_path):
        # Load integrals
        N_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(f"data/{fcidump_path}")
        # Load wave function
        psi_coef, psi_det = load_wf(f"data/{wf_path}")
        # Computation of the Energy of the input wave function (variational energy)
        return E_var(E0, N_ord,psi_coef, psi_det, d_one_e_integral, d_two_e_integral) 

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


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

    unittest.main()

