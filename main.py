#!/usr/bin/env python
# coding: utf-8

# # CIPSI mini app

# ## Hartree-Fock determinant
# 
# We consider a molecular system with $N_{\text{nuc}}$ nuclei, $N_{\alpha}$ $\alpha$-spin electrons and $N_{\beta}$ $\beta$-spin electrons.
# 
# The Hamiltonian operator is
# $$
# \hat{H} = \hat{T} + \hat{V}_{\text{nn}} + \hat{V}_{\text{en}}+ \hat{V}_{\text{ee}}
# $$
# where $\hat{T}$ is the kinetic energy operator, $\hat{V}_{\text{nn}}$ is the nuclear repulsion energy, $\hat{V}_{\text{en}}$ is the electron-nucleus attraction and $\hat{V}_{\text{ee}}$ is the electron repulsion.
# 
# We define a set of $N_{\text{AO}}$ one-electron functions called *atomic orbitals* (AOs) $\chi_k(\mathbf{r}),\; k\in [1,N_{\text{AO}}]$. These are atom-centered, and not orthonormal.
# 
# From the set of AOs, one can define a set of $N_{\text{MO}} \le N_{\text{AO}}$ one-electron functions called *molecular orbitals* (MOs), expressed as 
# $$
# \phi_i(\mathbf{r}) = \sum_{k=1}^{N_\text{AO}} C_{ki} \chi_k(\mathbf{r}).
# $$
# The MOs are orthonormal :
# $$
# \langle i | j \rangle  = \int \phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \text{d}\mathbf{r} = \delta_{ji}.
# $$ 
# 
# 
# The simplest $N$-electron wave function one can build is a *Slater determinant*. It can be written as:
# $$
# \begin{array}{c}
#  \Psi_{\text{HF}}({\bf r}_1,\dots,{\bf r}_{N_\alpha},{\bf r}_{N_\alpha+1},\dots,{\bf r}_N;
#       \alpha_1,\dots,\alpha_{N_\alpha},\beta_{N_\alpha+1},\dots,\beta_N) = \\
# \left|
#  \begin{array}{ccc}
#  \phi_1({\bf r}_1) & \dots & \phi_1({\bf r}_{N_\alpha}) \\
#  \vdots               & \dots &   \vdots             \\
#  \phi_{N_\alpha}({\bf r}_1) & \dots & \phi_{N_\alpha}({\bf r}_{N_\alpha}) \\
#  \end{array}
# \right|
# \left|
#  \begin{array}{ccc}
#  \phi_1({\bf r}_{N_\alpha+1}) & \dots & \phi_1({\bf r}_{N}) \\
#  \vdots               & \dots &   \vdots             \\
#  \phi_{N_\beta}({\bf r}_{N_\alpha+1}) & \dots & \phi_{N_\beta}({\bf r}_{N}) \\
#  \end{array}
# \right|
# \end{array}
# $$
# The Hartree-Fock (HF) determinant is the Slater determinant composed of the MOs which minimize the energy.
# 
# Usually, the set of MOs we use are Hartree-Fock molecular orbitals.
# 
# 
# 

# ## Configuration Interaction
# 
# The HF determinants is a crude approximation of the $N$-electron wave function. To improve it, one needs to allow variations in a space of $N$-electron basis functions. Given a set of molecular orbitals, the best wave function one can obtain is the wave function expressed in the basis of all possible Slater determinants.
# 
# Let us take for example a 4-electron wave function (2 $\alpha$-spin and two $\beta$-spin), and 3 molecular orbitals. The HF determinant is
# $$
# |1\, 2\, \bar{1}\, \bar{2} \rangle =
# \left| \begin{array}{cc}
#  \phi_1({\bf r}_1) & \phi_2({\bf r}_2) \\
#  \phi_2({\bf r}_1) & \phi_1({\bf r}_2) 
#  \end{array}
# \right|
# \left| \begin{array}{cc}
#  \phi_1({\bf r}_1) & \phi_2({\bf r}_2) \\
#  \phi_2({\bf r}_1) & \phi_1({\bf r}_2) 
#  \end{array}
# \right|
# $$
# where the bars on top of the numbers denote the $\beta$ spin.
# One can in addition build the following Slater determinants:
#  $|1\, 2\, \bar{1}\, \bar{3} \rangle,
#  |1\, 2\, \bar{2}\, \bar{3} \rangle,
#  |1\, 3\, \bar{1}\, \bar{2} \rangle,
#  |1\, 3\, \bar{1}\, \bar{3} \rangle,
#  |1\, 3\, \bar{2}\, \bar{3} \rangle,
#  |2\, 3\, \bar{1}\, \bar{2} \rangle,
#  |2\, 3\, \bar{1}\, \bar{3} \rangle,
#  |2\, 3\, \bar{2}\, \bar{3} \rangle$.
# 
# and express the wave function as linear combinations of these functions:
# $$
# \Psi = \sum_I c_I |I\rangle.
# $$
# When a wave function is expressed as a linear combination of Slater determinants, this is *Configuration Interaction*.
# Diagonalizing the Hamiltonian in this complete $N$-electron basis function is called Full Configuration Interaction (FCI).
# 
# One can remark that the number of determinant composing the FCI space is :
# $$ 
#  \aleph = 
# \binom {N_{\rm MO}}{N_\alpha}  
# \binom {N_{\rm MO}}{N_\beta}
# $$
# which grows exponentially. So FCI is a method that can be applied only when the number of electrons is small, or when the number of MOs is small.
# 
# Remarks:
# 1. As the MOs are orthonormal, the determinants constitute an orthonormal basis,
# 2. as a consequence, matrix elements of the Hamiltonian can be easily evaluated using Slater-Condon's rules (see [Wikipedia](https://en.wikipedia.org/wiki/Slater%E2%80%93Condon_rules) or https://arxiv.org/abs/1311.6244), and
# 3. when determinants $|I\rangle$ and $|J\rangle$ differ by more than 2 MOs, the matrix element $\langle I | \hat{H} | J\rangle = 0$. 
# 
# When $|J\rangle$ differs from $|I\rangle$ by one (resp. two) MO(s), $|J\rangle$ is a single (resp. double) excitation with respect to $|I\rangle$.
# 
# 

# ## CIPSI
# 
# The CIPSI algorithm is a way to obtain the best possible approximation of the FCI energy by building incrementally the FCI space and estimating with perturbation theory what is missing.
# 
# 1. Start with an internal space $\mathcal{I}$, and a trial wave function $\Psi_0$ defined in this space. Usually we start with the HF determinant.
# 2. Build all possible single and double excitations with respect to all determinants of the internal space, and discard the determinants which are already in $\mathcal{I}$. This set constitues the external space $\mathcal{E}$.
# 3. For each determinant $|\alpha \rangle$ of the external space, estimate by how much the energy will decrease by including the determinant in $\mathcal{I}$ and diagonalizing. An estimate can be obtained by perturbation theory:
# $$
# \epsilon_\alpha = \frac{\langle \Psi | \hat{H} | \alpha \rangle^2}{E - \langle \alpha | \hat{H} | \alpha \rangle}
# $$
# 4. Compute the second-order perturbative correction to the energy
# $$
# E_{\text{PT2}} = \sum_a \epsilon_\alpha
# $$
# The estimate of the FCI energy is $E + E_{\text{PT2}}$
# 5. If $|E_{\text{PT2}}| < \tau$, then the calculation has converged so we can exit. If not, go to next step 
# 6. Select the $|\alpha\rangle$'s for which $|\epsilon_\alpha|$ is the largest, and expand the internal space with those determinants
# 7. Diagonalize $\hat{H}$ in this new subspace, and go to step 2.
# 

# -------

# In[1]:


from collections import defaultdict
from itertools   import product
from math import sqrt
from enum import Enum

from typing import Tuple, Dict, NewType, NamedTuple, List

def grouper(iterable, n):
   '''Collect data into fixed-length chunks or blocks'''
   args = [iter(iterable)] * n
   return zip(*args)                                


# # Type definitions

# Orbital indices are integers in the [1:$N_{MO}$] range.
# 
# One- and two- electron integrals are stored as (key,value) pairs in dictionaries.
# 
# For the one-electron integral
# $$
# \langle i| \hat{h} |k \rangle = \int \phi_i(\mathbf{r}) \left( -\frac{1}{2} \hat{\Delta} +\hat{V}_{\text{en}} \right) \phi_k(\mathbf{r}) \text{d}\mathbf{r}
# $$
# the key is the tuple `(i,k)` and for the two-electron integral
# $$
# \langle ij|kl \rangle = \int \int \phi_i(\mathbf{r}_1) \phi_j(\mathbf{r}_2) \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|} \phi_k(\mathbf{r}_1) \phi_l(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2
# $$
# the key is the tuple `(i,j,k,l)`.
# 
# Since many integrals are zeros, a `defaultdict` is used and only non-zero integrals are stored.

# In[2]:


OrbitalIdx = NewType('OrbitalIdx', int)

One_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx], float]

Two_electron_integral = Dict[ Tuple[OrbitalIdx,OrbitalIdx,OrbitalIdx,OrbitalIdx], float]

Determinant_Spin = Tuple[OrbitalIdx, ...]
class Determinant(NamedTuple):
    '''Slater determinant: Product of 2 determinants.
       One for $\alpha$ electrons and one for \beta electrons.'''
    alpha: Determinant_Spin
    beta: Determinant_Spin
        
class Spin(Enum):
    ALPHA = 0
    BETA  = 1


# # Integrals

# In[3]:


N_mo = 0  # Number of MOs

def load_integrals(fcidump_path) -> Tuple[int, Two_electron_integral, One_electron_integral]:
    '''Read all the Hamiltonian integrals from the data file.
       Returns: (E0, d_one_e_integral, d_two_e_integral).
       E0 : a float containing the nuclear repulsion energy (V_nn),
       d_one_e_integral : a dictionary of one-electron integrals,
       d_two_e_integral : a dictionary of two-electron integrals.
       '''

    global N_mo
    with open(fcidump_path) as f:
        data_int = f.readlines()

    N_mo = 0
    
    # Only non-zero integrals are stored in the fci_dump.
    # Hence we use a defaultdict to handle the sparsity
    d_one_e_integral = defaultdict(float)
    d_two_e_integral = defaultdict(float)
    for line in data_int[4:]:
        v, *l = line.split()
        v = float(v)
        # Transform from Mulliken (ik|jl) to Dirac's <ij|kl> notation
        # (swap indices)
        i,k,j,l = list(map(int, l)) 
        N_mo = max(i,N_mo)
    
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


# In[4]:


def H_one_e(i: OrbitalIdx, j: OrbitalIdx) -> float :
    '''One-electron part of the Hamiltonian: Kinetic energy (T) and
       Nucleus-electron potential (V_{en}). This matrix is symmetric.'''
    return d_one_e_integral[ (i,j) ]
            

def H_two_e(i: OrbitalIdx, j: OrbitalIdx, k: OrbitalIdx, l: OrbitalIdx) -> float:
    '''Assume that *all* the integrals are in the global_variable
       `d_two_e_integral` In this function, for simplicity we don't use any
       symmetry sparse representation.  For real calculations, symmetries and
       storing only non-zeros needs to be implemented to avoid an explosion of
       the memory requirements.''' 
    return d_two_e_integral[ (i,j,k,l) ]


# # Hamiltonian matrix elements

# In[5]:


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


# ## Slater-Condon rules

# ### Excitation degree
# 
# Number of different MOs between two determinants

# In[6]:


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


# ### Diagonal elements

# In[7]:


def H_i_i(det_i: Determinant) -> float:
    '''Diagonal element of the Hamiltonian : <I|H|I>.'''
    res = E0
    res += sum(H_one_e(i,i) for i in det_i.alpha)
    res += sum(H_one_e(i,i) for i in det_i.beta)

    res += sum( (H_two_e(i,j,i,j) - H_two_e(i,j,j,i) ) for (i,j) in product(det_i.alpha, det_i.alpha)) / 2.
    res += sum( (H_two_e(i,j,i,j) - H_two_e(i,j,j,i) ) for (i,j) in product(det_i.beta , det_i.beta )) / 2.
       
    res += sum( H_two_e(i,j,i,j) for (i,j) in product(det_i.alpha, det_i.beta))

    return res


# ### Single excitations

# In[8]:


def get_phase_idx_single_exc(det_i: Determinant_Spin, det_j: Determinant_Spin) -> Tuple[int,int,int]:
    '''phase, hole, particle of <I|H|J> when I and J differ by exactly one orbital
       h is occupied only in I
       p is occupied only in J'''

    h, = set(det_i) - set(det_j)
    p, = set(det_j) - set(det_i)

    phase=1
    for det, idx in ((det_i,h),(det_j,p)):
        for occ in det:
            phase = -phase
            if occ == idx:
                break
    return (phase,h,p)



def H_i_j_single(li: Determinant_Spin, lj: Determinant_Spin, lk: Determinant_Spin) -> float:
    '''<I|H|J>, when I and J differ by exactly one orbital.'''
    
    # Interaction 
    phase, m, p = get_phase_idx_single_exc(li,lj)
    res = H_one_e(m,p)

    res += sum ( H_two_e(m,i,p,i)  -  H_two_e(m,i,i,p) for i in li)
    res += sum ( H_two_e(m,i,p,i) for i in lk)
    
    return phase*res


# ### Double excitations

# In[9]:


def get_phase_idx_double_exc(det_i: Determinant_Spin, det_j: Determinant_Spin) -> Tuple[int,int,int,int,int]:
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
    for l_,mp in ( (det_i,h1), (det_j,p1),  (det_j,p2), (det_i,h2) ):
        for v in l_:
            phase = -phase
            if v == mp:
                break
                
    # https://github.com/QuantumPackage/qp2/blob/master/src/determinants/slater_rules.irp.f:299
#    a = min(h1, p1)
    b = max(h1, p1)
    c = min(h2, p2)
 #   d = max(h2, p2)
    #if ((a<c) and (c<b) and (b<d)):
    if (c<b):
        phase = -phase

    return (phase, h1, h2, p1, p2)


def H_i_j_doubleAA(li: Determinant_Spin, lj: Determinant_Spin) -> float:
    '''<I|H|J>, when I and J differ by exactly two orbitals within
       the same spin.'''
    
    phase, h1, h2, p1, p2 = get_phase_idx_double_exc(li,lj)
    
    res = H_two_e(h1, h2, p1, p2) -  H_two_e(h1, h2, p2, p1)
    
    return phase * res
    

def H_i_j_doubleAB(det_i: Determinant, det_j: Determinant_Spin) -> float:
    '''<I|H|J>, when I and J differ by exactly one alpha spin-orbital and
       one beta spin-orbital.'''
    
    phaseA, hA, pA = get_phase_idx_single_exc(det_i.alpha, det_j.alpha)
    phaseB, hB, pB = get_phase_idx_single_exc(det_i.beta , det_j.beta)
    
    phase = phaseA * phaseB
    res = H_two_e(hA, hB, pA, pB)
  
    return phase * res


# ### General dispatch function

# In[10]:


def H_i_j(det_i: Determinant, det_j: Determinant) -> float:
    '''General function to dispatch the evaluation of H_ij'''
            
    ed_up, ed_dn = get_exc_degree(det_i, det_j)
    
    # Same determinant -> Diagonal element
    if (ed_up, ed_dn) == (0, 0):
        return H_i_i(det_i)

    # Single excitation
    elif (ed_up, ed_dn) == (1, 0):
        return H_i_j_single(det_i.alpha, det_j.alpha, det_i.beta)
    elif (ed_up, ed_dn) == (0, 1):
        return H_i_j_single(det_i.beta, det_j.beta, det_i.alpha)
    
    # Double excitation of same spin
    elif (ed_up, ed_dn) == (2, 0):
        return H_i_j_doubleAA(det_i.alpha,det_j.alpha)
    elif (ed_up, ed_dn) == (0, 2):
        return H_i_j_doubleAA(det_i.beta,det_j.beta)
    
    # Double excitation of opposite spins
    elif (ed_up, ed_dn) == (1, 1):
        return H_i_j_doubleAB(det_i, det_j)
    
    # More than doubly excited, zero
    else:
        return 0.


# ## Variational energy

# In[11]:


def E_var(det, psi_coef):
    norm2  = sum(c*c for c in psi_coef)
    result = sum(psi_coef[i] * psi_coef[j] * H_i_j(det_i,det_j)                  for (i,det_i),(j,det_j) in
                 product(enumerate(det),enumerate(det)))
    return result/norm2


# In[12]:


def E_var_with_matrices(det, psi_coef):
    import numpy as np
    
    h_mat = np.array( [ [H_i_j(det_i, det_j) for det_j in det]                 for det_i in det], dtype=float)
    
    psi = np.array(psi_coef, dtype=float)
    h_psi = np.matmul(h_mat, psi)
    
    return np.matmul(np.transpose(psi),h_psi) / np.dot(psi,psi)


# ## Generators of single and double excitations

# In[13]:


def single_exc(det_i: Determinant, spin: Spin, i: OrbitalIdx, a: OrbitalIdx) -> Determinant:
    '''Generates the single excitation from spin-orbital `i` of spin `spin` to
       spin-orbital `a` in determinant `det_i`.'''
    
    if det_i is None:
        return None
    
    if spin == Spin.ALPHA:
        s = list(det_i.alpha)
    else:
        s = list(det_i.beta)
        
    try:
        hole = s.index(i)
    except ValueError:
        # If i does not belong to det_i, the single excitation can't be made
        return None
    try:
        particle = s.index(a)
        # If a belongs to det_i, the single excitation can't be made
        return None
    except ValueError:
        pass
    
    s[hole] = a
    s.sort()
    
    if spin == Spin.ALPHA:
        return Determinant(tuple(s), det_i.beta)
    else:
        return Determinant(det_i.alpha, tuple(s))
    
    
def double_exc(det_i: Determinant,
               spin_1: Spin,
               spin_2: Spin,
               i: OrbitalIdx,
               a: OrbitalIdx,
               j: OrbitalIdx,
               b: OrbitalIdx):
    if a == j or i == b:  # It is a single
        return None
    else:
        return single_exc( 
               single_exc(det_i, spin_1, i, a),
                                 spin_2, j, b)

def generate_all_singles(det_i):
    s = [ single_exc(det_i, spin, hole, particle)           for (hole,particle) in product(range(1,N_mo),range(1,N_mo))
          for spin in [Spin.ALPHA, Spin.BETA] ]
    return set( x for x in s if x is not None )
    
    
def generate_all_doubles(det_i):
    s = [ double_exc(det_i, spin_1, spin_2,                      hole_1, particle_1,                      hole_2, particle_2)           for (hole_1, particle_1) in product(range(1,N_mo),range(1,N_mo))           for (hole_2, particle_2) in product(range(1,N_mo),range(1,N_mo))           for (spin_1, spin_2) in [ (Spin.ALPHA, Spin.ALPHA),
                                    (Spin.BETA , Spin.BETA ),
                                    (Spin.ALPHA, Spin.BETA ) ] ]
    return set( x for x in s if x is not None )
    
def generate_all_singles_and_doubles(det_i):
    return generate_all_singles(det_i).union (            generate_all_doubles(det_i) )


# # Selection and $E_\text{PT2}$

# ### Simple formulation

# In[14]:


def generate_external_space(det):
    internal_space = set(det)
    external_space = list(set.union( * [generate_all_singles_and_doubles(det_i)                         for det_i in det]).difference(internal_space))
    return external_space
    
def pt2_contrib(det_alpha, det, coef, E):
    '''The wave function is assumed normalized.'''
    psi_H_alpha = sum( ci * H_i_j(det_i, det_alpha) for (ci,det_i) in zip(coef, det) )
    delta_E = E - H_i_i(det_alpha)
    result = psi_H_alpha**2 / delta_E
    return result

def e_pt2(det, coef, E):
    external_space = generate_external_space(det)
    
    # Normalize wave function
    norm = 1./sqrt(sum(c*c for c in coef))
    coef = [c * norm for c in coef]

    return sum(pt2_contrib(det_alpha, det, coef, E) for det_alpha in external_space)

def select_determinants(det, coef, E, N):
    external_space = generate_external_space(det)
    res = [(pt2_contrib(det_alpha, det, coef, E), det_alpha) for det_alpha in external_space]
    res.sort()
    return res[:N]
    


# ### Matrix formulation

# $$
# E_\text{PT2} = \Psi^\dagger \mathbf{H}^\dagger \mathbf{\Delta}^{-1} \mathbf{H} \Psi
# $$
# where $\Psi$ is the wave function, $\mathbf{H}$ is the off-diagonal block of the Hamiltonian connecting the internal and the external spaces, and
# $$
# \Delta_{ij} = \delta_{ij}(E - H_{ii}).
# $$
# 

# In[15]:


def e_pt2_with_matrices(det, psi, E):
    import numpy as np
    
    external_space = generate_external_space(det)
    
    psi = np.array(psi)
    psi = psi / sqrt(np.dot(psi,psi))
    
    h_mat = np.array( [ [H_i_j(det_alpha, det_i) for det_i in det]                 for det_alpha in external_space ])
    
    h_psi = np.matmul(h_mat, psi)
    
    x = h_psi / np.array([E - H_i_i(det_alpha) for det_alpha in external_space])
    

    return np.dot(h_psi,x)


# In[16]:


def variance(det, psi):
    import numpy as np
    external_space = generate_external_space(det)
    
    psi = np.array(psi)
    psi = psi / sqrt(np.dot(psi,psi))
    
    h_mat = np.array( [ [H_i_j(det_alpha, det_i) for det_i in det]                for det_alpha in external_space ])

    h_psi = np.matmul(h_mat, psi)
    
    return np.dot(h_psi,h_psi)


# # Tests

# ## Hartree-Fock determinant

# In[17]:


fcidump_path='f2_631g.FCIDUMP'
wf_path='f2_631g.2det.wf'

_, det = load_wf(wf_path)

d_two_e_integral = defaultdict(float)
E0, d_one_e_integral, _ = load_integrals(fcidump_path)
E1  = H_i_i(det[0]) - E0

_, _, d_two_e_integral = load_integrals(fcidump_path)

E_hf = H_i_i(det[0]) 
E2  = H_i_i(det[0]) - E0 - E1

print(det[0])

print("Number of MOs       = {0:15d}  | ref :          18".format(N_mo))
print("Nuclear repulsion   = {0:15.6f}  | ref :   30.358633".format(E0))
print("One-electron energy = {0:15.6f}  | ref : -338.331811".format(E1))
print("Two-electron energy = {0:15.6f}  | ref :  109.327081".format(E2))
print("Hartree-Fock energy = {0:15.6f}  | ref : -198.646097".format(E_hf))


# ## Hamiltonian matrix elements

# In[18]:


fcidump_path='f2_631g.FCIDUMP'
wf_path='f2_631g.10det.wf'

psi_coef, det = load_wf(wf_path)

E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)


data = """           1           1  -228.839358707349     
           2           1  2.920537041552980E-002
           3           1 -1.255985888976286E-002
           4           1 -0.168761956584245     
           5           1  0.145829482733819     
           6           1  1.085314345825063E-002
           7           1  0.115306693893934     
           8           1 -4.136200013128970E-003
           9           1  0.101017162812122     
           1           2  2.920537041552980E-002
           2           2  -228.839358707349     
           3           2 -0.168761956584245     
           4           2 -1.255985888976286E-002
           5           2  1.085314345825063E-002
           6           2  0.145829482733819     
           7           2 -4.136200013128970E-003
           8           2  0.115306693893934     
          10           2  0.101017162812122     
           1           3 -1.255985888976286E-002
           2           3 -0.168761956584245     
           3           3  -228.249267026920     
           4           3  2.660919205800134E-002
           6           3  1.369740698241225E-015
           9           3  9.671138757713967E-003
           1           4 -0.168761956584245     
           2           4 -1.255985888976286E-002
           3           4  2.660919205800134E-002
           4           4  -228.249267026920     
           5           4  1.369740698241225E-015
          10           4  9.671138757713967E-003
           1           5  0.145829482733819     
           2           5  1.085314345825063E-002
           4           5  1.369740698241225E-015
           5           5  -228.249267026920     
           6           5  2.660919205800124E-002
          10           5 -8.356961432717070E-003
           1           6  1.085314345825063E-002
           2           6  0.145829482733819     
           3           6  1.369740698241225E-015
           5           6  2.660919205800124E-002
           6           6  -228.249267026920     
           9           6 -8.356961432717070E-003
           1           7  0.115306693893934     
           2           7 -4.136200013128970E-003
           7           7  -227.396544277563     
           8           7  1.461383175586562E-002
           9           7 -6.511227326576720E-002
           1           8 -4.136200013128970E-003
           2           8  0.115306693893934     
           7           8  1.461383175586562E-002
           8           8  -227.396544277563     
          10           8 -6.511227326576720E-002
           1           9  0.101017162812122     
           3           9  9.671138757713967E-003
           6           9 -8.356961432717070E-003
           7           9 -6.511227326576720E-002
           9           9  -226.710295336874     
           2          10  0.101017162812122     
           4          10  9.671138757713967E-003
           5          10 -8.356961432717070E-003
           8          10 -6.511227326576720E-002
          10          10  -226.710295336874      """.splitlines()

ref = defaultdict(float)

for line in data:
    i, j, v = line.split()
    ref[(int(i)-1,int(j)-1)] = float(v)

for (i,det_i), (j, det_j) in product( enumerate(det), enumerate(det) ):
    hij = H_i_j(det_i,det_j)
    if i == j:
        hij = hij - E0
    degree = get_exc_degree(det_i,det_j)
    if abs(hij - ref[(i,j)])/(max(abs(hij),1.e-15)) < 1.e-12: 
        marker = ""
    else:
        marker = "<-"
    print("{0:2d} {1:2d}  {2} {3:15.8e} | ref : {4:15.8e} {5}".format( i, j, degree, hij, ref[(i,j)], marker ) )
    


# ## Single-determinant PT2

# In[19]:


fcidump_path='f2_631g.FCIDUMP'
wf_path='f2_631g.2det.wf'

psi_coef, det = load_wf(wf_path)
psi_coef = [1.]
det= det[:1]

E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)


E = E_var(det, psi_coef)
EPT2 = e_pt2(det, psi_coef, E)
Variance = variance(det, psi_coef)

print("Energy   = {0:15.6f}  | ref : -198.646097".format(E))
print("Variance = {0:15.6f}  | ref :    1.114264".format(Variance))
print("E_PT2    = {0:15.6f}  | ref :   -0.367588".format(EPT2))
print("E+PT2    = {0:15.6f}  | ref : -199.013685".format(E+EPT2))

print(select_determinants(det, psi_coef, E, 3))


# ## Multi-determinant wave function

# In[20]:


fcidump_path='f2_631g.FCIDUMP'
wf_path='f2_631g.10det.wf'

psi_coef, det = load_wf(wf_path)

E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)


E = E_var(det, psi_coef)
EPT2 = e_pt2(det, psi_coef, E)
Variance = variance(det, psi_coef)

print("Energy   = {0:15.6f}  | ref : -198.548963".format(E))
print("Variance = {0:15.6f}  | ref :  0.92649843".format(Variance))
print("E_PT2    = {0:15.6f}  | ref : -0.24321128".format(EPT2))
print("E+PT2    = {0:15.6f}  | ref : -198.792174".format(E+EPT2))

print(select_determinants(det, psi_coef, E, 3))


# In[ ]:




