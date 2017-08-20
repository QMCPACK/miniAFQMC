////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source
//// License.  See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by:
////
//// File created by:
//////////////////////////////////////////////////////////////////////////////////

#ifndef  AFQMC_ENERGY_HPP 
#define  AFQMC_ENERGY_HPP 

#include <type_traits>
#include "Numerics/ma_operations.hpp"

// temporary
#include "Numerics/SparseMatrixOperations.hpp"

namespace qmcplusplus
{

namespace base
{

/*
 * Calculates the local energy from (already evaluated) mixed density matrices.
 *
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
 *   as a vector with "linearized" index ak=a*NMO+k.        
 */
// Serial implementation
template< class ValueMat,
          class SpMat,
          typename = typename std::enable_if<(ValueMat::dimensionality == 2)>::type,
          typename = typename std::enable_if<(SpMat::dimensionality == 2)>::type
        >
inline void calculate_energy(ValueMat& W_data, const ValueMat& Gc, ValueMat& Gcloc, const ValueMat& haj, const SpMat& Vakbl)
{
  // W[nwalk][2][NMO][NAEA]
  //
  assert(W_data.shape()[1] >= 4);
  assert(Gc.shape()[1] == W_data.shape()[0]);
  assert(Gc.shape()[1] == Gcloc.shape()[1]);
  assert(Gc.shape()[0] == Gcloc.shape()[0]);
  assert(Gc.shape()[0] == haj.num_elements());
  assert(Gc.shape()[0] == Vakbl.rows());
  assert(Gc.shape()[0] == Vakbl.cols());

  typedef typename std::decay<ValueMat>::type::element ValueType;
  index_gen indices;
  ValueType zero = ValueType(0.);
  ValueType one = ValueType(1.); 
  ValueType half = ValueType(0.5); 

  int nwalk = W_data.shape()[0];
  boost::multi_array_ref<const ValueType,1> haj_ref(haj.origin(), extents[haj.num_elements()]);

  // zero 
  for(int n=0; n<nwalk; n++) W_data[n][0] = zero;

  // Vakbl * Gc(bl,nw) = Gcloc(ak,nw)
  SparseMatrixOperators::product_SpMatM( Vakbl.rows(), Gc.shape()[1], Vakbl.cols(), 
          one, Vakbl.values(), Vakbl.column_data(), Vakbl.row_index(), 
          Gc.data(), Gc.strides()[0], 
          zero, Gcloc.data(), Gcloc.strides()[0] );

  // E2(nw) = 0.5*Gc(:,nw)*Gcloc(:,nw) 
    // how do I do this through BLAS?
  for(int i=0, iend=Gc.shape()[0]; i<iend; i++) 
    for(int n=0; n<nwalk; n++) 
      W_data[n][0] += Gc[i][n]*Gcloc[i][n];   
  for(int n=0; n<nwalk; n++) W_data[n][0] *= half;
    
  // one-body contribution
  ma::product(one,ma::T(Gc),haj_ref,one,W_data[indices[range_t(0,nwalk)][0]]);

}

}

}

#endif
