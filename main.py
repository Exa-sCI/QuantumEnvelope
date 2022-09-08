#!/usr/bin/env python3
from qe.drivers import *
from qe.io import *
import sys
import time

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    fcidump_path = sys.argv[1]
    wf_path = sys.argv[2]
    N_det_target = int(sys.argv[3])
    try:
        driven_by = sys.argv[4]
    except IndexError:
        driven_by = "integral"

    # Load integrals
    T_load_integrals_start = time.time()
    if(comm.Get_rank()==0):
        n_ord, E0, d_one_e_integral, d_two_e_integral = load_integrals(fcidump_path)
    else:
        n_ord=None
        E0=None
        d_one_e_integral=None
        d_two_e_integral=None
    comm.bcast(n_ord, root=0)    
    comm.bcast(E0, root=0)    
    comm.bcast(d_one_e_integral, root=0)    
    comm.bcast(d_two_e_integral, root=0)    

    T_load_integrals_stop = time.time()
    T_load_integrals = T_load_integrals_stop - T_load_integrals_start
    if (comm.Get_rank()==0):
        print("T_load_integrals =",T_load_integrals)
    # Load wave function
    
    T_load_wf_start= time.time()
    psi_coef, psi_det = load_wf(wf_path)
    T_load_wf_stop= time.time()
    T_load_wf=T_load_wf_stop - T_load_wf_start  
    # Hamiltonian engine
    T_Hamiltonian_start= time.time()

    lewis = Hamiltonian_generator(
        comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
    )

    T_Hamiltonian_stop= time.time()
    T_Hamiltonian=T_Hamiltonian_stop - T_Hamiltonian_start
    
    T_Selection_start=time.time()
    while len(psi_det) < N_det_target:
        E, psi_coef, psi_det, E_PT2= selection_step(comm, lewis, n_ord, psi_coef, psi_det, len(psi_det))
        # Update Hamiltonian engine
        lewis = Hamiltonian_generator(
            comm, E0, d_one_e_integral, d_two_e_integral, psi_det, driven_by=driven_by
        )
        if (comm.Get_rank()==0):
          sys.stdout.write("N_det:%d,  E=%d,  E_PT2=%d \n" % (len(psi_det),E,E_PT2))
          #print(f"N_det: {len(psi_det)}, E {E}")
    
    T_Selection_stop=time.time()
    T_Selection=T_Selection_stop - T_Selection_start
    if (comm.Get_rank()==0):
        print("T_load_integrals =",T_load_integrals)
        print("T_load_wf =",T_load_wf)
        print("T_Hamiltonian =",T_Hamiltonian)
        print("T_Selection=",T_Selection)
