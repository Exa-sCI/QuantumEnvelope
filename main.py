#!/usr/bin/env python3

from pprint import pprint 
from math import exp, sqrt

import random

N_det = 1

with open('f2_631g.FCIDUMP') as f:
    data_int = f.readlines()

Vnn  =    30.3586331075289
Ven  =   -536.871190427040
Vee  =    109.327080702748
Vecp =   0.000000000000000E+000
T    =    198.539379873618

print (Vnn+Ven+T)

from collections import defaultdict
d_int = defaultdict(int)
d_double = defaultdict(int)
E0=0
for line in data_int[4:]:
    v, *l = line.split()
    v = float(v)
    i,j,k,l = list(map(int, l)) 
        
 
    if i == 0:
        E0 = v
    elif k == 0:
        d_int[ (i,j) ] = v            
        d_int[ (j,i) ] = v
    else:
        d_double[ (i,j,k,l) ] = v
        d_double[ (i,l,k,j) ] = v
        d_double[ (j,i,l,k) ] = v
        d_double[ (j,k,l,i) ] = v
        d_double[ (k,j,i,l) ] = v
        d_double[ (k,l,i,j) ] = v
        d_double[ (l,i,j,k) ] = v
        d_double[ (l,k,j,i) ] = v

N_orb = 18
N_elec_up = 9
N_elec_dn = 9
N_det = 1


def gen_det(N_det_max,N_elec_up, N_elec_dn):

    def gen_occ(n):
        s = set()
        while len(s) != n:
            s.add(random.randint(0,N_orb)+1)

        return tuple(sorted(s))

    det = set()
    det.add( ( tuple(range(1,N_elec_up+1)), tuple(range(1,N_elec_dn+1)) )  ) 
    n_det = 1
    while n_det != N_det_max:
        det.add( (gen_occ(N_elec_up), gen_occ(N_elec_dn)) )
        n_det = len(det)

    return list(det)

# A determinant is a Tuple of Integer correpoding to the index of the orbital
# For example (1,2,3,4)
# Two type of determinant exist the Determinant up (know also as alpha) and down (beta).
# det is a list unique of pair of determinant up, and determinant down.

det  = gen_det(N_det,N_elec_up, N_elec_dn)
N_det = len(det)

def gen_psi_coef(N_det):
    from math import exp, sqrt
    psi_coef = [ exp(-i) if i % 3 else -exp(-i) for i in range(N_det) ]
    sum_ = sum(v*v for v in psi_coef)
    psi_coef = [v/sqrt(sum_) for v in psi_coef]
    return psi_coef


print (N_det)
psi_coef = gen_psi_coef(N_det)

def get_ed(det_i, det_j):
    # Compute excitation degree
    ed_up =  len(set(det_i[0]).symmetric_difference(set(det_j[0]))) // 2
    ed_dn =  len(set(det_i[1]).symmetric_difference(set(det_j[1]))) // 2

    return (ed_up, ed_dn)

def H_mono(i,j):
    # Todo read real integral
    return d_int[ (i,j) ]

def H_bi(i,j,k,l):
    # Todo read real integral
    return d_double[ (i,j,k,l) ]

def H_i_i(det_i):
    # Alpha + Beta
    res = 0
    for i in det_i[0] + det_i[1]:
        res += H_mono(i,i)

    for x in range(0,1):
        for i in det_i[x]:
            for j in det_i[x]:
                res += ( H_bi(i,j,i,j)  -  H_bi(i,j,j,i) ) / 2
        
    for i in det_i[0]:
        for j in det_i[1]:
            res += H_bi(i,j,i,j) 

    return res

def H_i_j_single(li, lj, lk):
    #https://arxiv.org/abs/1311.6244
    m, p = list(set(li).symmetric_difference(set(lj)))
    res = H_mono(m,p)

    for i in li:
        res += ( H_bi(m,i,p,i)  -  H_bi(m,i,i,p) ) 
      
    for i in lk:
        res += H_bi(m,i,p,i)
    
    #Todo phase traversal!
    phase = 1.

    for l,mp in ( (li,m), (lj,p) ):
        for v in l:
            phase = -phase
            if v == mp:
                break
         
    return phase*res

def H_i_j_doubleAA(li,lj):
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


def H_i_j_doubleAB(det_i,det_j):
    i, = set(det_i[0]) - set(det_j[0])
    j, = set(det_i[1]) - set(det_j[1])

    k, = set(det_j[0]) - set(det_i[0])
    l, = set(det_j[1]) - set(det_i[1])

    res =  H_bi(i,j,k,l)

    phase = 1
    for l_,mp in ( (det_i[0],i), (det_i[1],j),  (det_j[0],k), (det_j[1],l) ):
        for v in l_:
            phase = -phase
            if v == mp:
                break


    return phase * res 

def H_i_j(det_i, det_j):

    ed_up, ed_dn = get_ed(det_i, det_j)
    # Apply slater rule
    if ed_up + ed_dn == 0:
        return H_i_i(det_i)
    elif ed_up + ed_dn == 1:
        if ed_up == 1:
            assert (det_i[1] == det_j[1])
            return H_i_j_single(det_i[0], det_j[0], det_i[1])
        elif ed_dn == 1:
            assert (det_i[0] == det_j[0])
            return H_i_j_single(det_i[1], det_j[1], det_i[0])

    elif ed_up + ed_dn == 2:
        if ed_up == 2:
            assert (det_i[1] == det_j[1])
            return H_i_j_doubleAA(det_i[0],det_j[0])
        elif ed_dn == 2:
            assert (det_i[0] == det_j[0])
            return H_i_j_doubleAA(det_i[1],det_j[1])
        elif ed_up + ed_dn == 2:
            return H_i_j_doubleAB(det_i, det_j)
    else:
        return 0

from itertools import product

variational_energy = sum(psi_coef[i] * psi_coef[j] * H_i_j(det_i,det_j)  for (i,det_i),(j,det_j) in product(enumerate(det),enumerate(det)) )
print (E0+variational_energy)


