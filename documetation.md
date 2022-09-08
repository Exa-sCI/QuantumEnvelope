# Hight level algorithm.

So in high level summary we have 2 phases:

1. Diagonalize the Hamiltonian ( psi_internal x psi_internal)
2. Do the selection and compute the PT2. 

## Diagonalize the Hamiltonian

We do on the fly computation of H, so we don't need to store it

## Selection and PT2.

1. Split the psi_internal per process.
2. psi_internal_local compute the sum of PT2 and store the best external determinant.
         

   
