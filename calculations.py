#!/usr/bin/env python3

# Types
# -----
from typing import Tuple, Dict, NewType, NamedTuple, List, Set, Iterator, NewType

# Orbital index (1,2,...,Norb)
OrbitalIdx = NewType('OrbitalIdx', int)

# Two-electron integral :
# $<ij|kl> = \int \int \phi_i(r_1) \phi_j(r_2) \frac{1}{|r_1 - r_2|} \phi_k(r_1) \phi_l(r_2) dr_1 dr_2$
Two_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]

# One-electron integral :
# $<i|h|k> = \int \phi_i(r) (-\frac{1}{2} \Delta + V_en ) \phi_k(r) dr$
One_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]

Determinant_spin = Tuple[OrbitalIdx, ...]
class Determinant(NamedTuple):
    '''Slater determinant: Product of 2 determinants.
       One for $\alpha$ electrons and one for \beta electrons.'''
    alpha: Determinant_spin
    beta: Determinant_spin

Psi_det = List[Determinant_spin]
Psi_coef = List[float]
# We have two type of energy.
# The varitional Energy who correpond Psi_det
# The pt2 Energy who correnpond to the pertubative energy induce by each determinant connected to Psi_det 
Energy = NewType('Energy',float)

#
# ___                                          
#  |  ._  o _|_ o  _. | o _   _. _|_ o  _  ._  
# _|_ | | |  |_ | (_| | | /_ (_|  |_ | (_) | | 
#                                              

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

#  _ ___  _   __ ___     _                    
# /   |  |_) (_   |    _|_ _.  _ _|_  _  ._   
# \_ _|_ |   __) _|_    | (_| (_  |_ (_) | \/ 
#                                          /  

# Yes, I like itertools
from itertools import chain, product, combinations, takewhile
import numpy as np


class Excitation(object):

    def __init__(self, N_orb):
        self.all_orbs = frozenset(range(1,N_orb+1))

    def gen_all_exc_from_detspin(self, detspin: Determinant_spin, ed: int) -> Iterator:
        '''
        Generate list of pair -> hole from a determinant spin.

        >>> sorted(Excitation(4).gen_all_exc_from_detspin( (1,2),2))
        [((1, 2), (3, 4))]
        >>> sorted(Excitation(4).gen_all_exc_from_detspin( (1,2),1))
        [((1,), (3,)), ((1,), (4,)), ((2,), (3,)), ((2,), (4,))]
        '''
        holes = combinations(detspin,ed)
        not_detspin = self.all_orbs - set(detspin)
        parts = combinations(not_detspin,ed)
        return product(holes,parts)

    def gen_all_connected_detspin_from_detspin(self, detspin: Determinant_spin, ed: int) -> Iterator:
        '''
        Generate all the posible spin determinant relative to a excitation degree

        >>> sorted(Excitation(4).gen_all_connected_detspin_from_detspin( (1,2), 1))
        [(1, 3), (1, 4), (2, 3), (2, 4)]

        '''
        def apply_excitation(exc: Tuple[Tuple[int, ...],Tuple[int, ...]])-> Determinant_spin:
            # Warning use global variable detspin.
            lh,lp = exc
            s = (set(detspin) - set(lh)) | set(lp)
            return tuple(sorted(s))

        l_exc = self.gen_all_exc_from_detspin(detspin, ed)
        return map(apply_excitation, l_exc)

    def gen_all_connected_det_from_det(self,det_source: Determinant)->Iterator:
        '''
        Generate all the determinant who are single or double exictation (aka connected) from the input determinant

        >>> sorted(Excitation(3).gen_all_connected_det_from_det( Determinant( (1,2), (1,) )))
        [Determinant(alpha=(1, 2), beta=(2,)),
         Determinant(alpha=(1, 2), beta=(3,)),
         Determinant(alpha=(1, 3), beta=(1,)),
         Determinant(alpha=(1, 3), beta=(2,)),
         Determinant(alpha=(1, 3), beta=(3,)),
         Determinant(alpha=(2, 3), beta=(1,)),
         Determinant(alpha=(2, 3), beta=(2,)),
         Determinant(alpha=(2, 3), beta=(3,))]
        '''

        # All single exitation from alpha or for beta determinant
        # Then the production of the alpha, and beta (it's a double)
        # Then the double exitation form alpha or beta

        # We use l_single_a, and l_single_b twice. So we store them.
        l_single_a  = set(self.gen_all_connected_detspin_from_detspin(det_source.alpha, 1))
        l_double_aa = self.gen_all_connected_detspin_from_detspin(det_source.alpha, 2)

        s_a = ( Determinant(det_alpha, det_source.beta) for det_alpha in chain(l_single_a,l_double_aa) )

        l_single_b  = set(self.gen_all_connected_detspin_from_detspin(det_source.beta, 1))
        l_double_bb = self.gen_all_connected_detspin_from_detspin(det_source.beta, 2)

        s_b = ( Determinant(det_source.alpha, det_beta) for det_beta in chain(l_single_b,l_double_bb) )

        l_double_ab = product(l_single_a,l_single_b)

        s_ab = ( Determinant(det_alpha,det_beta) for det_alpha,det_beta in l_double_ab )

        return chain(s_a,s_b,s_ab)

    def gen_all_connected_determinant_from_psi(self,psi: List[Determinant])-> Set:
        '''
        >>> d1 = Determinant( (1,2), (1,) ) ; d2 = Determinant( (1,3), (1,) )
        >>> len(Excitation(4).gen_all_connected_determinant_from_psi( [ d1,d2 ] ))
        22
        '''
        return list(set(chain.from_iterable(map(self.gen_all_connected_det_from_det,psi))) - set(psi))


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

class Hamiltonian(object):

    def __init__(self, d_one_e_integral: One_electron_integral, d_two_e_integral: Two_electron_integral, E0: float):
        self.d_one_e_integral = d_one_e_integral
        self.d_two_e_integral = d_two_e_integral
        self.E0 = E0

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

    def get_phase_idx_single_exc(self, det_i: Determinant_spin, det_j: Determinant_spin) -> Tuple[int,int,int]:
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

    def get_phase_idx_double_exc(self, det_i: Determinant_spin, det_j: Determinant_spin) -> Tuple[int,int,int,int,int]:
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
        '''Diagonal element of the Hamiltonian : <I|H|I>.'''

        res  = self.E0
        res += sum(self.H_one_e(i,i) for i in det_i.alpha)
        res += sum(self.H_one_e(i,i) for i in det_i.beta)

        res += sum(self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) for i,j in combinations(det_i.alpha,2))
        res += sum(self.H_two_e(i,j,i,j) - self.H_two_e(i,j,j,i) for i,j in combinations(det_i.beta, 2))

        res += sum(self.H_two_e(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))

        return res


    def H_i_j_single(self, detspin_i: Determinant_spin, detspin_j: Determinant_spin, detspin_k: Determinant_spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one orbital.'''

        # Interaction
        phase, m, p = self.get_phase_idx_single_exc(detspin_i,detspin_j)
        res = self.H_one_e(m,p)

        res += sum ( self.H_two_e(m,i,p,i) - self.H_two_e(m,i,i,p) for i in detspin_i)
        res += sum ( self.H_two_e(m,i,p,i) for i in detspin_k)
        return phase * res

    def H_i_j_doubleAA(self, li: Determinant_spin, lj: Determinant_spin) -> float:
        '''<I|H|J>, when I and J differ by exactly two orbitals within
           the same spin.'''

        phase, h1, h2, p1, p2 = self.get_phase_idx_double_exc(li,lj)

        res = self.H_two_e(h1, h2, p1, p2) -  self.H_two_e(h1, h2, p2, p1)

        return phase * res


    def H_i_j_doubleAB(self, det_i: Determinant, det_j: Determinant_spin) -> float:
        '''<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
           one beta spin-orbital.'''

        phaseA, hA, pA = self.get_phase_idx_single_exc(det_i.alpha, det_j.alpha)
        phaseB, hB, pB = self.get_phase_idx_single_exc(det_i.beta , det_j.beta)

        phase = phaseA * phaseB
        res = self.H_two_e(hA, hB, pA, pB)

        return phase * res

    def get_exc_degree(self, det_i: Determinant, det_j: Determinant) -> Tuple[int,int]:
        '''Compute the excitation degree, the number of orbitals which differ
           between the two determinants.
        >>> Hamiltonian(_,_,_).get_exc_degree(Determinant(alpha=(1, 2), beta=(1, 2)),
        ...                                   Determinant(alpha=(1, 3), beta=(5, 7)) )
        (1, 2)
        '''
        ed_up =  len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
        ed_dn =  len(set(det_i.beta ).symmetric_difference(set(det_j.beta ))) // 2
        return ed_up, ed_dn

    def H_i_j(self, det_i: Determinant, det_j: Determinant) -> float:
        '''General function to dispatch the evaluation of H_ij'''

        ed_up, ed_dn = self.get_exc_degree(det_i, det_j)
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

    def H(self, psi_i, psi_j): 
        ''' Return a matrix of size psi_i x psi_j containing the value of the Hamiltionian.
         Note that when psi_i == psi_j, this matrix is an hermitian.'''

        h = np.array([self.H_i_j(det_i,det_j) for det_i, det_j in product(psi_i,psi_j)])
        return h.reshape(len(psi_i),len(psi_j))


class Powerplant(object):
    '''
    Compute all the Energy and associated value from a psi_det.
    E denote the variational energy
    '''
    def __init__(self, lewis, psi_det: Psi_det):
        self.lewis = lewis
        self.psi_det = psi_det

    def E(self,psi_coef: Psi_coef) -> Energy:
        # Vector * Vector.T * Matrix
        return np.einsum('i,j,ij ->', psi_coef,psi_coef,self.lewis.H(self.psi_det,self.psi_det))

    @property    
    def E_and_psi_coef(self) -> Tuple[Energy, Psi_coef]:
        # Return lower eigenvalue (aka the new E) and lower evegenvector (aka the new psi_coef)
        psi_H_psi = self.lewis.H(self.psi_det,self.psi_det)
        energies, coeffs = np.linalg.eigh(psi_H_psi)
        return energies[0], coeffs[:,0]

    def psi_external_pt2(self,psi_coef: Psi_coef, N_orb) -> Tuple[Psi_det, List[Energy] ]:
        # Compute the pt2 contrution of all the external (aka connected) determinant.
        #   eα=⟨Ψ(n)∣H∣∣α⟩^2 / ( E(n)−⟨α∣H∣∣α⟩ )

        psi_external = Excitation(N_orb).gen_all_connected_determinant_from_psi(self.psi_det)
        h = self.lewis.H(self.psi_det,psi_external)

        nomitator = np.einsum('i,ij -> j', psi_coef, h) # Matrix * vector -> vector

        denominator = np.divide(1., self.E(psi_coef) - np.array([self.lewis.H_i_i(det_external) for det_external in psi_external]))
        return psi_external, np.einsum('i,i,i -> i', nomitator, nomitator, denominator) # vector * vector * vector -> scalar

    def E_pt2(self,psi_coef,N_orb) -> Energy:
        # The sum of the pt2 contribution of each external determinant
        return sum(self.psi_external_pt2(psi_coef,N_orb)[1])


def selection_step(lewis, N_ord, psi_coef, psi_det, n) -> Tuple[Energy, Psi_coef, Psi_det]:
    # 1. Generate a list of all the external determinant and their pt2 contribution
    # 2. Take the n  determinants who have the biggest contribution and add it the wave function psi
    # 3. Diagonalize H corresponding to this new wave function to get the new variational energy, and new psi_coef.

    # In the main code:
    # -> Go to 1., stop when E_pt2 < Threshold || N < Threshold
    # See example of chained call to this function in `test_f2_631g_1p5p5det`

    # 1.
    psi_external_det, psi_external_energy = Powerplant(lewis, psi_det).psi_external_pt2(psi_coef,N_ord)

    # 2.
    idx = np.argpartition(psi_external_energy, n)[:n]
    psi_det_new = psi_det + [psi_external_det[i] for i in idx]

    # 3.
    return (*Powerplant(lewis, psi_det_new).E_and_psi_coef, psi_det_new)

