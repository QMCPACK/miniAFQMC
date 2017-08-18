////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
////////////////////////////////////////////////////////////////////////////////

#ifndef  AFQMC_DENSITY_MATRIX_HPP 
#define  AFQMC_DENSITY_MATRIX_HPP 

#include "Numerics/ma_operations.hpp"

namespace qmcplusplus
{

namespace base 
{

/*
 * Calculates the 1-body mixed density matrix:
 *   < A | c+i cj | B > / <A|B> = conj(A) * ( T(B) * conj(A) )^-1 * T(B) 
 *   If compact == True, returns [NEL x M] matrix:
 *   < A | c+i cj | B > / <A|B> = ( T(B) * conj(A) )^-1 * T(B) 
 * Parameters:
 *  - conjA = conj(A)
 *  - B
 *  - DM = < A | c+i cj | B > / <A|B>
 *  - IWORK1: [ 2*M ] integer work matrix   
 *  - TWORK1: [ NEL x NEL ] work matrix   
 *  - TWORK2: (only used if compact = False) [ NEL x M ] work matrix   
 *  - compact (default = True)
 *  returns:
 *  - <A|B> = det[ T(B) * conj(A) ]  
 */
// Serial Implementation
template< class Tp,
          class ValueMatA,
          class ValueMatB,
          class ValueMatC,
          class ValueMat,
          class IntVec,
	  typename = typename std::enable_if<(ValueMat::dimensionality == 2)>::type		
        >
inline Tp MixedDensityMatrix(const ValueMatA& conjA, const ValueMatB& B, ValueMatC& C, IntVec& I1, ValueMat& T1, ValueMat& T2, bool compact=true)
{
  // check dimensions are consistent
  assert( conjA.shape()[0] == B.shape()[0] );
  assert( conjA.shape()[1] == T1.shape()[1] );
  assert( B.shape()[1] == T1.shape()[0] );
  assert( T1.shape()[1] == B.shape()[1] );
  assert( T2.shape()[0] == T1.shape()[0] );
  if(compact) {
    assert( C.shape()[0] == T1.shape()[0] );
    assert( C.shape()[1] == B.shape()[0] );
  } else {
    assert( T2.shape()[1] == B.shape()[0] );
    assert( C.shape()[0] == conjA.shape()[0] );
    assert( C.shape()[1] == T2.shape()[1] );
  }

  using ma::T;

  // T(B)*conj(A) 
  ma::product(T(B),conjA,T1);  

  // NOTE: Using C as temporary 
  // T1 = T1^(-1)
  Tp ovlp = static_cast<Tp>(ma::invert(T1,I1,T2));

  if(compact) {

    // C = T1 * T(B)
    ma::product(T1,T(B),C); 

  } else {

    // T2 = T1 * T(B)
    ma::product(T1,T(B),T2); 

    // C = conj(A) * T2
    ma::product(conjA,T2,C);

  }

  return ovlp;
}

} // namespace base

} // namespace qmcplusplus 

#endif
