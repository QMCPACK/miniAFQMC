////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

/** @file mixed_density_matrix.hpp
 *  @brief Mixed Density Matrix
 */

#ifndef  AFQMC_DENSITY_MATRIX_HPP 
#define  AFQMC_DENSITY_MATRIX_HPP 

#include "Numerics/ma_operations.hpp"

namespace qmcplusplus
{

namespace base 
{

/**
 * Calculates the 1-body mixed density matrix:
 *
 *   \f$ \left< A | c+i cj | B \right> / \left<A|B\right> = A^\dagger ( B^T * A^\dagger )^{-1}  B^T \f$
 *
 *   If compact == True, returns [NEL x M] matrix:
 *
 *   \f$ \left< A | c+i cj | B \right> / \left<A|B\right> = ( B^T * A^\dagger )^{-1} B^T \f$
 *
 * Parameters:
 *  - conjA = conj(A)
 *  - B
 *  - C = < A | c+i cj | B > / <A|B>
 *  - T1: [ NEL x NEL ] work matrix   
 *  - T2: (only used if compact = False) [ NEL x M ] work matrix   
 *  - IWORK: [ N ] integer biffer for invert. Dimensions must be at least NEL. 
 *  - WORK: [ >=NEL ] Work space for invert. Dimensions must be at least NEL.   
 *  - compact (default = True)
 *  returns:
 *  - <A|B> = det[ T(B) * conj(A) ]  
 */
// Serial Implementation
template< class Tp,
          class MatA,
          class MatB,
          class MatC,
          class Mat,
          class IBuffer,
          class TBuffer 
        >
inline Tp MixedDensityMatrix(const MatA& conjA, const MatB& B, MatC& C, Mat& T1, Mat& T2, IBuffer& IWORK, TBuffer& WORK, bool compact=true)
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
  Tp ovlp = static_cast<Tp>(ma::invert(T1,IWORK,WORK));

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


/*
 * Returns the overlap of 2 Slater determinants:  <A|B> = det[ T(B) * conj(A) ]  
 * Parameters:
 *  - conjA = conj(A)
 *  - B
 *  - IWORK: [ M ] integer work matrix   
 *  - T1: [ NEL x NEL ] work matrix   
 *  returns:
 *  - <A|B> = det[ T(B) * conj(A) ]  
 */
// Serial Implementation
template< class Tp,
          class MatA,
          class MatB,
          class Mat,
          class IBuffer
        >
inline Tp Overlap(const MatA& conjA, const MatB& B, Mat& T1, IBuffer& IWORK)
{
  // check dimensions are consistent
  assert( conjA.shape()[0] == B.shape()[0] );
  assert( conjA.shape()[1] == T1.shape()[1] );
  assert( B.shape()[1] == T1.shape()[0] );
  assert( T1.shape()[1] == B.shape()[1] );

  using ma::T;

  // T(B)*conj(A) 
  ma::product(T(B),conjA,T1);  

  return static_cast<Tp>(ma::determinant(T1,IWORK));
}

} // namespace base

} // namespace qmcplusplus 

#endif
