from qe.fundamental_types import (
    Tuple,
    One_electron_integral,
    Two_electron_integral,
    Determinant,
    Energy,
    List,
)
from collections import defaultdict
from qe.integral_indexing_utils import compound_idx4
import math
import time

from mpi4py import MPI  # Note this initializes and finalizes MPI session automatically
#   _____      _ _   _       _ _          _   _
#  |_   _|    (_) | (_)     | (_)        | | (_)
#    | | _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __
#    | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
#   _| || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#   \___/_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|

# ~
# Integrals of the Hamiltonian over molecular orbitals
# ~
def load_integrals(fcidump_path,
) -> Tuple[int, float, One_electron_integral, Two_electron_integral]:
    """Read all the Hamiltonian integrals from the data file.
    Returns: (E0, d_one_e_integral, d_two_e_integral).
    n_elec:  an integer containing the number of electrons in the system
    n_orb : an integer containing the number of orbitals in the system
    E0 : a float containing the nuclear repulsion energy (V_nn),
    d_one_e_integral : a dictionary of one-electron integrals,
    d_two_e_integral : a dictionary of two-electron integrals.
    """
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()

    E0=0
    n_elec =0
    n_orb=0
    d_one_e_integral = defaultdict(int)
    d_two_e_integral = defaultdict(int)

    import glob

    if(rank==0): 
       T_load_integrals_start = time.time()
       if len(glob.glob(fcidump_path)) == 1:
           fcidump_path = glob.glob(fcidump_path)[0]
       elif len(glob.glob(fcidump_path)) == 0:
           print("no matching fcidump file")
       else:
           print(f"multiple matches for {fcidump_path}")
           for i in glob.glob(fcidump_path):
               print(i)
     
       # Use an iterator to avoid storing everything in memory twice.
       if fcidump_path.split(".")[-1] == "gz":
           import gzip
     
           f = gzip.open(fcidump_path)
       elif fcidump_path.split(".")[-1] == "bz2":
           import bz2
     
           f = bz2.open(fcidump_path)
       else:
           f = open(fcidump_path)
     
       # Only non-zero integrals are stored in the fci_dump.
       # Hence we use a defaultdict to handle the sparsity
       split=next(f).split()
       n_orb=int(split[2])
       n_elec=int(split[5])

       for _ in range(3):
           next(f)
     
       d_one_e_integral = defaultdict(int)
       d_two_e_integral = defaultdict(int)
     
       for line in f:
           v, *l = line.split()
           v = float(v)
           # Transform from Mulliken (ik|jl) to Dirac's <ij|kl> notation
           # (swap indices)
           i, k, j, l = list(map(int, l))
     
           if i == 0:
               E0 = v
           elif j == 0:
               # One-electron integrals are symmetric (when real, not complex)
               d_one_e_integral[
                   (i - 1, k - 1)
               ] = v  # index minus one to be consistent with determinant orbital indexing starting at zero
               d_one_e_integral[(k - 1, i - 1)] = v
           else:
               # Two-electron integrals have many permutation symmetries:
               # Exchange r1 and r2 (indices i,k and j,l)
               # Exchange i,k
               # Exchange j,l
               key = compound_idx4(i - 1, j - 1, k - 1, l - 1)
               d_two_e_integral[key] = v
     
       f.close()
       T_load_integrals_stop = time.time()
       print("Time load integrals =", T_load_integrals_stop-T_load_integrals_start)

    n_elec=comm.bcast(n_elec, root=0) 
    n_orb=comm.bcast(n_orb, root=0) 
    E0=comm.bcast(E0, root=0) 
    d_one_e_integral=comm.bcast(d_one_e_integral, root=0) 
    d_two_e_integral=comm.bcast(d_two_e_integral, root=0) 

    return n_elec, n_orb, E0, d_one_e_integral, d_two_e_integral



def load_wf(n_elec,n_orb,path_wf) -> Tuple[List[float], List[Determinant]]:
    """Read the input file :
    Representation of the Slater determinants (basis) and
    vector of coefficients in this basis (wave function)."""

    import glob
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()

    det = []
    psi_coef = []

    if(rank==0): 
       if (path_wf!="Guess"):
          if len(glob.glob(path_wf)) == 1:
              path_wf = glob.glob(path_wf)[0]
          elif len(glob.glob(path_wf)) == 0:
              print(f"no matching wf file: {path_wf}")
          else:
              print(f"multiple matches for {path_wf}")
              for i in glob.glob(path_wf):
                  print(i)
      
          if path_wf.split(".")[-1] == "gz":
              import gzip
      
              with gzip.open(path_wf) as f:
                  data = f.read().decode().split()
          elif path_wf.split(".")[-1] == "bz2":
              import bz2
      
              with bz2.open(path_wf) as f:
                  data = f.read().decode().split()
          else:
              with open(path_wf) as f:
                  data = f.read().split()
      
          def decode_det(str_):
              for i, v in enumerate(str_):
                  if v == "+":
                      yield i
      
          def grouper(iterable, n):
              "Collect data into fixed-length chunks or blocks"
              args = [iter(iterable)] * n
              return zip(*args)
          for (coef, det_i, det_j) in grouper(data, 3):
              psi_coef.append(float(coef))
              det.append(Determinant(tuple(decode_det(det_i)), tuple(decode_det(det_j))))
       else:
          print("Bazinga")
          coef=1.0
          alpha=math.ceil(n_elec/2)
          beta=n_elec-alpha
          psi_coef.append(float(coef))
          #$This is Wrong!!!! Needs fixing!!!!
          det_i="+++++------" 
          det_j="+++++------" 
          det.append(Determinant(tuple(decode_det(det_i)), tuple(decode_det(det_j))))
 
 
       # Normalize psi_coef
 
       norm = math.sqrt(sum(c * c for c in psi_coef))
       psi_coef = [c / norm for c in psi_coef]

    psi_coef=comm.bcast(psi_coef, root=0) 
    det=comm.bcast(det, root=0) 


    return psi_coef, det


def load_eref(path_ref) -> Energy:
    """Read the input file :
    Representation of the Slater determinants (basis) and
    vector of coefficients in this basis (wave function)."""

    import glob

    if len(glob.glob(path_ref)) == 1:
        path_ref = glob.glob(path_ref)[0]
    elif len(glob.glob(path_ref)) == 0:
        print(f"no matching ref file: {path_ref}")
    else:
        print(f"multiple matches for {path_ref}")
        for i in glob.glob(path_ref):
            print(i)

    if path_ref.split(".")[-1] == "gz":
        import gzip

        with gzip.open(path_ref) as f:
            data = f.read().decode()
    elif path_ref.split(".")[-1] == "bz2":
        import bz2

        with bz2.open(path_ref) as f:
            data = f.read().decode()
    else:
        with open(path_ref) as f:
            data = f.read()

    import re

    return float(re.search(r"E +=.+", data).group(0).strip().split()[-1])
