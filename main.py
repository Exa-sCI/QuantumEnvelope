#!/usr/bin/env python3

from pprint import pprint 
from math import exp, sqrt

import random

N_orb = 10
N_elec_up = 5
N_elec_dn = 4

def gen_occ(n):
    s = set()
    while len(s) != n:
        s.add(random.randint(0,N_orb)+1)
    
    return tuple(sorted(s))

N_det_max = 100
det = [ ( tuple(range(1,N_elec_up+1)), tuple(range(1,N_elec_dn+1)) )  ]
det += [ (gen_occ(N_elec_up), gen_occ(N_elec_dn)) for i in range(N_det_max-1) ]




det  = list(set(det))
N_det = len(det)

from math import exp, sqrt

psi_coef = [ exp(-i) if i % 3 else -exp(-i) for i in range(N_det) ]
sum_ = sum(v*v for v in psi_coef)
psi_coef = [v/sqrt(sum_) for v in psi_coef]


print (N_det)

def get_ed(det_i, det_j):
    # Compute execitation degree
    ed_up =  len(set(det_i[0]).symmetric_difference(set(det_j[0]))) // 2
    ed_dn =  len(set(det_i[1]).symmetric_difference(set(det_j[1]))) // 2

    return (ed_up, ed_dn)

def H_mono(i,j):
    # Todo read real integral
    return -i+j/100 

def H_bi(i,j,k,l):
    # Todo read real integral
    return H_mono(i,j)+ H_mono(k,l)*0.3

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
    
    m, p = list(set(li).symmetric_difference(set(lj)))
    res = H_mono(m,p)

    for i in li:
        res += ( H_bi(m,i,p,i)  -  H_bi(m,i,i,p) ) 
      
    for i in lk:
        res += H_bi(m,i,p,i)
    
    #Todo phase traversal!
    return res

def H_i_j_doubleAA(li,lj):
    i, j = set(li) - set(lj)
    k, l = set(lj) - set(li)

    res = ( H_bi(i,j,k,l)  -  H_bi(i,j,l,k) )
    # Todo phase
    return res


def H_i_j_doubleAB(det_i,det_j):
    i, = set(det_i[0]) - set(det_j[0])
    j, = set(det_i[1]) - set(det_j[1])

    k, = set(det_j[0]) - set(det_i[0])
    l, = set(det_j[1]) - set(det_i[1])

    res =  H_bi(i,j,k,l)
    # Todo phase
    return res

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
print (variational_energy)


