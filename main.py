#!/usr/bin/env python3

# Types
# -----
from typing import Tuple, Dict, NewType, NamedTuple, List

# Orbital index (1,2,...,Norb)
OrbitalIdx = NewType('OrbitalIdx', int)

# Two-electron integral :
# $<ij|kl> = \int \int \phi_i(r_1) \phi_j(r_2) \frac{1}{|r_1 - r_2|} \phi_k(r_1) \phi_l(r_2) dr_1 dr_2$
Two_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]

# One-electron integral :
# $<i|h|k> = \int \phi_i(r) (-\frac{1}{2} \Delta + V_en ) \phi_k(r) dr$
One_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]

Determinant_Spin = Tuple[OrbitalIdx, ...]
class Determinant(NamedTuple):
    '''Slater determinant: Product of 2 determinants.
       One for $\alpha$ electrons and one for \beta electrons.'''
    alpha: Determinant_Spin
    beta: Determinant_Spin

# ~
# Integrals of the Hamiltonian over molecular orbitals
# ~
def load_integrals(fcidump_path) -> Tuple[int, Two_electron_integral, One_electron_integral]:
    '''Read all the Hamiltonian integrals from the data file.
       Returns: (E0, d_one_e_integral, d_two_e_integral).
       E0 : a float containing the nuclear repulsion energy (V_nn),
       d_one_e_integral : a dictionary of one-electron integrals,
       d_two_e_integral : a dictionary of two-electron integrals.
       '''
    from collections import defaultdict
    
    with open(fcidump_path) as f:
        data_int = f.readlines()

    # Only non-zero integrals are stored in the fci_dump.
    # Hence we use a defaultdict to handle the sparsity
    d_one_e_integral = defaultdict(int)
    d_two_e_integral = defaultdict(int)
    for line in data_int[4:]:
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
            # Exchange i,k (if complex, with a minus sign)
            # Exchange j,l (if complex, with a minus sign)
            d_two_e_integral[ (i,j,k,l) ] = v
            d_two_e_integral[ (i,l,k,j) ] = v
            d_two_e_integral[ (j,i,l,k) ] = v
            d_two_e_integral[ (j,k,l,i) ] = v
            d_two_e_integral[ (k,j,i,l) ] = v
            d_two_e_integral[ (k,l,i,j) ] = v
            d_two_e_integral[ (l,i,j,k) ] = v
            d_two_e_integral[ (l,k,j,i) ] = v

    return E0, d_one_e_integral, d_two_e_integral

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

    return psi_coef, det



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



class Hamiltonian(object):

    def __init__(self,d_one_e_integral, d_two_e_integral):
        self.d_one_e_integral = d_one_e_integral
        self.d_two_e_integral = d_two_e_integral

    def H_one_e(self, i: OrbitalIdx, j: OrbitalIdx) -> float :
        '''One-electron part of the Hamiltonian: Kinetic energy (T) and
           Nucleus-electron potential (V_{en}). This matrix is symmetric.'''
        return self.d_one_e_integral[ (i,j) ]
    
    def H_two_e(self, i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
        '''Assume that *all* the integrals are in the global_variable
           `d_two_e_integral` In this function, for simplicity we don't use any
           symmetry sparse representation.  For real calculations, symmetries and
           storing only non-zeros needs to be implemented to avoid an explosion of
           the memory requirements.'''
        return self.d_two_e_integral[ (i,j,k,l) ]

    def H_i_i(self, det_i: Determinant) -> float:
        from itertools   import product

        '''Diagonal element of the Hamiltonian : <I|H|I>.'''
        res  = sum(self.H_one_e(i,i) for i in det_i.alpha)
        res += sum(self.H_one_e(i,i) for i in det_i.beta)
        
        res += sum( (self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) ) for (i,j) in product(det_i.alpha, det_i.alpha)) / 2
        res += sum( (self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) ) for (i,j) in product(det_i.beta, det_i.beta)) / 2
           
        res += sum( self.H_two_e(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))
     
        return res


    def H_i_j_single(self, li: Determinant_Spin, lj: Determinant_Spin, lk: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one orbital.'''
        
        # Interaction 
        m, = set(li) - set(lj)
        p, = set(lj) - set(li)

        res = self.H_one_e(m,p)
    
        res += sum ( self.H_two_e(m,i,p,i)  -  self.H_two_e(m,i,i,p) for i in li)
        res += sum ( self.H_two_e(m,i,p,i)  -  self.H_two_e(m,i,i,p) for i in lk)
    
        # Phase
        phase = 1
        for l, idx in ( (li,m), (lj,p) ):
            for v in l:
                phase = -phase
                if v == idx:
                    break
    
        # Result    
        return phase*res


    def H_i_j_doubleAA(self, li: Determinant_Spin, lj: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly two orbitals within
           the same spin.'''
    
        #Hole
        i, j = sorted(set(li) - set(lj))
        #Particle
        k, l = sorted(set(lj) - set(li))
    
        res = ( self.H_two_e(i,j,k,l)  -  self.H_two_e(i,j,l,k) )
    
        # Compute phase. See paper to have a loopless algorithm
        # https://arxiv.org/abs/1311.6244
        phase = 1
        for l_,mp in ( (li,i), (lj,j),  (lj,k), (li,l) ):
            for v in l_:
                phase = -phase
                if v == mp:
                    break
        # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
        a = min(i, k)
        b = max(i, k)
        c = min(j, l)
        d = max(j, l)
        if ((a<c) and (c<b) and (b<d)):
            phase = -phase
    
        return phase * res 


    def H_i_j_doubleAB(self, det_i: Determinant, det_j: Determinant_Spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
           one beta spin-orbital.'''
        i, = set(det_i.alpha) - set(det_j.alpha)
        j, = set(det_i.beta) - set(det_j.beta)
        
        k, = set(det_j.alpha) - set(det_i.alpha)
        l, = set(det_j.beta) - set(det_i.beta)
    
        res =  self.H_two_e(i,j,k,l)
      
        phase = 1
        for l_,mp in ( (det_i.alpha,i), (det_i.beta,j), (det_j.alpha,k), (det_j.beta,l) ):
            for v in l_:
                phase = -phase
                if v == mp:
                     break
    
        return phase * res 
 
    def H_i_j(self, det_i: Determinant, det_j: Determinant, d_one_e_integral: One_electron_integral, d_two_e_integral: Two_electron_integral) -> float:
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

def E_var(psi_coef, psi_det, d_one_e_integral,  d_two_e_integral):
    
    lewis = Hamiltonian(d_one_e_integral,d_two_e_integral)
    norm2 = sum(c*c for c in psi_coef)

    from itertools import product
    r = sum(psi_coef[i] * psi_coef[j] * lewis.H_i_j(det_i,det_j, d_one_e_integral, d_two_e_integral) for (i,det_i),(j,det_j) in product(enumerate(psi_det),enumerate(psi_det)) )
    return r / norm2


import unittest
class TestVariationalEnergy(unittest.TestCase):

    def load_and_compute(self,fcidump_path,wf_path):
        # Load integrals
        E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
    
        # Load wave function
        psi_coef, psi_det = load_wf(wf_path)
    
        # Computation of the Energy of the input wave function (variational energy)
        return E0 + E_var(psi_coef, psi_det, d_one_e_integral, d_two_e_integral) 


    def test_f2_631g_10det(self):
        fcidump_path='f2_631g.FCIDUMP'
        wf_path='f2_631g.10det.wf'
        E_ref =  -198.548963
        E =  self.load_and_compute(fcidump_path,wf_path)
        self.assertAlmostEqual(E_ref,E)

if __name__ == "__main__":
    unittest.main()
