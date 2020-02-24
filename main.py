#!/usr/bin/env python3

from typing import Tuple, Dict, NewType, NamedTuple, List
OrbitalIdx = NewType('OrbitalIdx', int)
Determinant_Spin = Tuple[OrbitalIdx, ...]
class Determinant(NamedTuple):
    alpha: Determinant_Spin
    beta: Determinant_Spin

Integral_Bielectronic = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]
Integral_Monoelectronic = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]

#                         
# |   _   _.  _| o ._   _  
# |_ (_) (_| (_| | | | (_| 
#                       _| 

def load_integral_monoegral(fcidump_path) -> Tuple[int, Integral_Bielectronic, Integral_Monoelectronic]:
    
    with open(fcidump_path) as f:
        data_int = f.readlines()

    # Only non zero integrale are stored in the fci_dump.
    # Hence we use a defaultdict to handle the sparticity
    from collections import defaultdict
    d_integral_mono = defaultdict(int)
    d_integral_bi = defaultdict(int)
    for line in data_int[4:]:
        v, *l = line.split()
        v = float(v)
        # Transofrm to Diract Notation
        i,k,j,l = list(map(int, l)) 
 
        if i == 0:
            E0 = v
        elif j == 0:
            # Expend the symetrie 
            d_integral_mono[ (i,k) ] = v            
            d_integral_mono[ (k,i) ] = v
        else:
            # Physicist notation (storing)
            # Expend the 8-fold symetrie
            d_integral_bi[ (i,j,k,l) ] = v
            d_integral_bi[ (i,l,k,j) ] = v
            d_integral_bi[ (j,i,l,k) ] = v
            d_integral_bi[ (j,k,l,i) ] = v
            d_integral_bi[ (k,j,i,l) ] = v
            d_integral_bi[ (k,l,i,j) ] = v
            d_integral_bi[ (l,i,j,k) ] = v
            d_integral_bi[ (l,k,j,i) ] = v

    return E0, d_integral_mono, d_integral_bi


def load_wf(path_wf) -> Tuple[ List[float] , List[Determinant] ]  :
    with open(path_wf) as f:
        data = f.read().split()

    def grouper(iterable, n):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return zip(*args)

    def decode_det(str_):
        for i,v in enumerate(str_, start=1):
            if v == '+':
                yield i

    det = []; psi_coef = []
    for (coef, det_i, det_j) in grouper(data,3):
        psi_coef.append(float(coef))
        det.append ( Determinant( tuple(decode_det(det_i)), tuple(decode_det(det_j) ) ) )

    return psi_coef, det


#
#  _                                        
# /   _  ._ _  ._     _|_  _. _|_ o  _  ._  
# \_ (_) | | | |_) |_| |_ (_|  |_ | (_) | | 
#              |                            

# ~
# Integral
# ~

def H_mono(i: OrbitalIdx, j: OrbitalIdx) -> float :
    # Assume symetrie
    return d_integral_mono[ (i,j) ]

def H_bi(i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
    # Assume that *all* the integral are in the global_varaible `d_integral_bi`
    # In this function we don't use any symetrie or sparticity to reduce the storage, this is N4.
    # For large system (N>800) the sparticity should reduce the storage requirement
    return d_integral_bi[ (i,j,k,l) ]

# ~
# Slater condom Rule
# ~

def get_ed(det_i: Determinant, det_j: Determinant) -> Tuple[int,int]:
    # Compute excitation degree
    # Number of different orbital between determinant
    '''
    >>> get_ed(Determinant(alpha=(1, 2), beta=(1, 2)),
    ...       Determinant(alpha=(1, 3), beta=(5, 7)) )
    (1, 2)
    '''
    ed_up =  len(set(det_i.alpha).symmetric_difference(set(det_j.alpha))) // 2
    ed_dn =  len(set(det_i.beta).symmetric_difference(set(det_j.beta))) // 2
    return (ed_up, ed_dn)


def H_i_i(det_i: Determinant) -> float:
    #Dirac Notation
    res = sum(H_mono(i,i) for i in det_i.alpha)
    res += sum(H_mono(i,i) for i in det_i.beta)
    
    from itertools import product

    res += sum( (H_bi(i,j,i,j) - H_bi(i,j,j,i) ) for (i,j) in product(det_i.alpha, det_i.alpha)) / 2
    res += sum( (H_bi(i,j,i,j) - H_bi(i,j,j,i) ) for (i,j) in product(det_i.beta, det_i.beta)) / 2
       
    res += sum( H_bi(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))
 
    return res

def H_i_j_single(li: Determinant_Spin, lj: Determinant_Spin, lk: Determinant_Spin) -> float:
    #https://arxiv.org/abs/1311.6244
    #NOT TESTED /!\
    
    # Interaction 
    m, p = list(set(li).symmetric_difference(set(lj)))
    res = H_mono(m,p)

    res += sum ( H_bi(m,i,p,i)  -  H_bi(m,i,i,p) for i in li)
    res += sum ( H_bi(m,i,p,i)  -  H_bi(m,i,i,p) for i in lk)

    # Phase
    phase = 1
    for l, idx in ( (li,m), (lj,p) ):
        for v in l:
            phase = -phase
            if v == idx:
                break

    # Result    
    return phase*res

def H_i_j_doubleAA(li: Determinant_Spin, lj: Determinant_Spin) -> float:

    #Hole
    i, j = sorted(set(li) - set(lj))
    #Particle
    k, l = sorted(set(lj) - set(li))

    res = ( H_bi(i,j,k,l)  -  H_bi(i,j,l,k) )

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


def H_i_j_doubleAB(det_i: Determinant, det_j: Determinant_Spin) -> float:
    i, = set(det_i.alpha) - set(det_j.alpha)
    j, = set(det_i.beta) - set(det_j.beta)
    
    k, = set(det_j.alpha) - set(det_i.alpha)
    l, = set(det_j.beta) - set(det_i.beta)

    res =  H_bi(i,j,k,l)
  
    phase = 1
    for l_,mp in ( (det_i.alpha,i), (det_i.beta,j), (det_j.alpha,k), (det_j.beta,l) ):
        for v in l_:
            phase = -phase
            if v == mp:
                 break

    return phase * res 

def H_i_j(det_i: Determinant, det_j: Determinant, det_int, d_integral_bi) -> float:

    ed_up, ed_dn = get_ed(det_i, det_j)
    # Apply slater-Condon rules
    # (https://en.wikipedia.org/wiki/Slater%E2%80%93Condon_rules)

    # No excitation
    if ed_up + ed_dn == 0:
        return H_i_i(det_i)
    # Single excitation
    elif ed_up == 1 and ed_dn == 0:
        return H_i_j_single(det_i.alpha, det_j.alpha, det_i.beta)
    elif ed_up == 0 and ed_dn == 1:
        return H_i_j_single(det_i.beta, det_j.beta, det_i.alpha)
    # Double excitation
    elif ed_up == 2 and ed_dn == 0:
        return H_i_j_doubleAA(det_i.alpha,det_j.alpha)
    elif ed_up == 0 and ed_dn == 2:
        return H_i_j_doubleAA(det_i.beta,det_j.beta)
    elif ed_up == 1 and ed_dn == 1:
        return H_i_j_doubleAB(det_i, det_j)
    # More than doubly excited, no contribution
    else:
        return 0.


if __name__ == "__main__":
    # Fcidump contain the integral
    #fcidump_path='f2_631g.FCIDUMP'
    fcidump_path='kev.DSDKSL'
    wf_path='f2_631g.28det.wf'

    # Initilization
    E0, d_integral_mono, d_integral_bi = load_integral_monoegral(fcidump_path)
    psi_coef, det = load_wf(wf_path)

    # Computation of the Energy
    from itertools import product
    variational_energy = sum(psi_coef[i] * psi_coef[j] * H_i_j(det_i,det_j, d_integral_mono, d_integral_bi)  for (i,det_i),(j,det_j) in product(enumerate(det),enumerate(det)) )
    print (E0+variational_energy)
    expected_value = -198.71760085
    print ('expected value:', expected_value)
    print (E0+variational_energy  - expected_value)
