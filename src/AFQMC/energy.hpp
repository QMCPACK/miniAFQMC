


  /*
   * Vakbl(ak,bl) = sum_i sum_j conj(TrialWfn(i,a)) * conj(TrialWfn(j,b)) * (<ij|kl> - <ij|lk>)
   *    --> (alpha, beta) terms, no exchange term is included (e.g. <ij|lk>)
   * The 2-body contribution to the energy is obtained from:
   * Let G(i,j) {Gmod(i,j)} be the {modified} "Green's Function" of the walker 
   *            (defined by the Slater Matrix "W"), then:
   *   G    = conj(TrialWfn) * [transpose(W)*conj(TrialWfn)]^(-1) * transpose(W)    
   *   Gmod = [transpose(W)*conj(TrialWfn)]^(-1) * transpose(W)    
   *   E_2body = sum_i,j,k,l G(i,k) * (<ij|kl> - <ij|lk>) * G(j,l) + (alpha/beta) + (beta/beta)  
   *           = sum a,k,b,l Gmod(a,k) * Vabkl(ak,jl) * Gmod(jl) = Gmod * Vakbl * Gmod
   *   The expression can be evaluated with a sparse matrix-dense vector product, 
   *   followed by a dot product between vectors, if we interpret the matrix Gmod(a,k) 
   *   as a vector with "linearized" index ik=i*NMO+k.        
   */
