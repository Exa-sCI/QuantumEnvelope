# Hight level algorithm.

So in high level summary we have 2 phases:

1. Diagonalize the Hamiltonian ( psi_internal x psi_internal)
2. Do the selection and compute the PT2. 

## Diagonalize the Hamiltonian

We do on the fly computation of H, so we don't need to store it

## Selection and PT2.

1. Each process will handle psi_external by chunk
2. Inside each process:
      - For the maximun size of psi_external storable
      - Compute sum E_PT2 and max det
3. Do one reduciton to get full PT2

### Size psi_external: 
    
For one det:
- alpha*(orb-alpha) Single ALPHA
- beta*(orb-beta)  Sine Beta 
- alpha^2*(orb-alpha)^2 AB doubles 
- comb(alpha,2)*comb(orb-alpha,2) AA doubles
- comb(beta,2)*comb(orb-beta,2) BB doubles


         

   
