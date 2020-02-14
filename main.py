#!/usr/bin/env python3

from pprint import pprint 
from math import exp, sqrt

import random

from typing import Tuple, Dict, NewType, NamedTuple


OrbitalIdx = NewType('OrbitalIdx', int)
Determinant_Spin = Tuple[OrbitalIdx, ...]
Determinant_Alpha = NewType('Determinant_Alpha', Determinant_Spin)
Determinant_Beta = NewType('Determinant_Beta', Determinant_Spin)

class Determinant(NamedTuple):
    alpha: Determinant_Alpha
    beta: Determinant_Beta

Integral_Bielectronic = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]
Integral_Monoelectronic = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]


def load_integral(fcidump_path) -> Tuple[int, Integral_Bielectronic, Integral_Monoelectronic]:
    
    with open(fcidump_path) as f:
        data_int = f.readlines()

    from collections import defaultdict
    d_int = defaultdict(int)
    d_double = defaultdict(int)
    for line in data_int[4:]:
        v, *l = line.split()
        v = float(v)
        i,k,j,l = list(map(int, l)) 
 
        if i == 0:
            E0 = v
        elif j == 0:
            d_int[ (i,k) ] = v            
            d_int[ (k,i) ] = v
        else:
            # Physicist notation (storing)
            d_double[ (i,j,k,l) ] = v
            d_double[ (i,l,k,j) ] = v
            d_double[ (j,i,l,k) ] = v
            d_double[ (j,k,l,i) ] = v
            d_double[ (k,j,i,l) ] = v
            d_double[ (k,l,i,j) ] = v
            d_double[ (l,i,j,k) ] = v
            d_double[ (l,k,j,i) ] = v

    return E0, d_int, d_double


def H_mono(i: OrbitalIdx, j: OrbitalIdx) -> float : 
    return d_int[ (i,j) ]

def H_bi(i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
    return d_double[ (i,j,k,l) ]

def gen_det(N_det_max,N_elec_up, N_elec_dn):

    def gen_occ(n):
        s = set()
        while len(s) != n:
            s.add(random.randint(0,N_orb)+1)

        return tuple(sorted(s))

    det = set()
    det.add( Determinant( tuple(range(1,N_elec_up+1)), tuple(range(1,N_elec_dn+1)) )  ) 
    n_det = 1
    while n_det != N_det_max:
        det.add( (gen_occ(N_elec_up), gen_occ(N_elec_dn)) )
        n_det = len(det)

    return list(det)

def gen_psi_coef(N_det):
    from math import exp, sqrt
    psi_coef = [ exp(-i) if i % 3 else -exp(-i) for i in range(N_det) ]
    sum_ = sum(v*v for v in psi_coef)
    psi_coef = [v/sqrt(sum_) for v in psi_coef]
    return psi_coef


def get_ed(det_i: Determinant, det_j: Determinant) -> Tuple[int,int]:
    # Compute excitation degree
    ed_up =  len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
    ed_dn =  len(set(det_i.beta).symmetric_difference(set(det_j.beta))) // 2

    return (ed_up, ed_dn)


def H_i_i(det_i: Determinant) -> float:
    #Dirac Notation
    res = sum(H_mono(i,i) for i in det_i.alpha)
    res += sum(H_mono(i,i) for i in det_i.beta)
    
    from itertools import product
    res += sum( (H_bi(i,j,i,j)  -  H_bi(i,j,j,i) ) for (i,j) in product(det_i.alpha, det_i.alpha)) / 2
    res += sum( (H_bi(i,j,i,j)  -  H_bi(i,j,j,i) ) for (i,j) in product(det_i.beta, det_i.beta)) / 2
       
    res += sum( H_bi(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))
 
    return res

def H_i_j_single(li: Determinant_Spin, lj: Determinant_Spin, lk: Determinant_Spin):
    #https://arxiv.org/abs/1311.6244
    m, p = list(set(li).symmetric_difference(set(lj)))
    res = H_mono(m,p)

    for i in li:
        res += ( H_bi(m,i,p,i)  -  H_bi(m,i,i,p) ) 
      
    for i in lk:
        res += H_bi(m,i,p,i)
    
    phase = 1.

    for l,mp in ( (li,m), (lj,p) ):
        for v in l:
            phase = -phase
            if v == mp:
                break
         
    return phase*res

def H_i_j_doubleAA(li: Determinant_Spin, lj: Determinant_Spin):
    #Hole
    i, j = sorted(set(li) - set(lj))
    #Particle
    k, l = sorted(set(lj) - set(li))

    res = ( H_bi(i,j,k,l)  -  H_bi(i,j,l,k) )
    # Compute phase. See paper to have a loopless algorithm
    # https://arxiv.org/abs/1311.6244
    phase = 1.
    for l_,mp in ( (li,i), (lj,j),  (lj,k), (li,l) ):
        for v in l_:
            phase = -phase
            if v == mp:
                break
    # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:289
    a = min(i, k)
    b = max(i, k)
    c = min(j, l)
    d = max(j, l)
    if ((a<c) and (c<b) and (b<d)):
        phase = -phase
 
    return phase * res 


def H_i_j_doubleAB(det_i: Determinant, det_j: Determinant_Spin):
    i, = set(det_i.alpha) - set(det_j.alpha)
    j, = set(det_i.beta) - set(det_j.beta)

    k, = set(det_j.alpha) - set(det_i.alpha)
    l, = set(det_j.beta) - set(det_i.beta)

    res =  H_bi(i,j,k,l)

    phase = 1
    for l_,mp in ( (det_i.alpha,i), (det_i.beta,j),  (det_j.alpha,k), (det_j.beta,l) ):
        for v in l_:
            phase = -phase
            if v == mp:
                break


    return phase * res 

def H_i_j(det_i: Determinant, det_j: Determinant) -> float:

    ed_up, ed_dn = get_ed(det_i, det_j)
    # Apply slater rule for connected determinant (Only Singly and Double connected have a contribution)
    if ed_up + ed_dn == 0:
        return H_i_i(det_i)
    elif ed_up == 1 and  ed_dn == 0:
        return H_i_j_single(det_i.alpha, det_j.alpha, det_i.beta)
    elif ed_up == 0 and ed_dn == 1:
        return H_i_j_single(det_i.beta, det_j.beta, det_i.alpha)
    elif ed_up == 2 and  ed_dn == 0:
        return H_i_j_doubleAA(det_i.alpha,det_j.alpha)
    elif ed_up == 0 and  ed_dn == 2:
        return H_i_j_doubleAA(det_i.beta,det_j.beta)
    elif ed_up == 1 and  ed_dn == 1:
        return H_i_j_doubleAB(det_i, det_j)
    else:
        return 0.




N_orb = 18
N_elec_up = 9
N_elec_dn = 9
N_det = 1


# Fcidump contain the integral
fcidump_path='f2_631g.FCIDUMP'
# This reference value where generated by QP
Vnn  =    30.3586331075289
Ven  =   -536.871190427040
Vee  =    109.327080702748
Vecp =   0.000000000000000E+000
T    =    198.539379873618

print ('Ref', Vee+Vnn+Ven+T)

# Initilization
E0, d_int, d_double = load_integral(fcidump_path)
det  = gen_det(N_det,N_elec_up, N_elec_dn)
psi_coef = gen_psi_coef(N_det)

# Computation of the Energy
from itertools import product
variational_energy = sum(psi_coef[i] * psi_coef[j] * H_i_j(det_i,det_j)  for (i,det_i),(j,det_j) in product(enumerate(det),enumerate(det)) )
print (E0+variational_energy)


