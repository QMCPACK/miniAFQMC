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


#ifndef  AFQMC_DENSITY_MATRIX_HPP 
#define  AFQMC_DENSITY_MATRIX_HPP 

#include <Kokkos_Core.hpp>

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
  assert( conjA.dimension(0) == B.dimension(0) );
  assert( conjA.dimension(1) == T1.dimension(1) );
  assert( B.dimension(1) == T1.dimension(0) );
  assert( T1.dimension(1) == B.dimension(1) );
  assert( T2.dimension(0) == T1.dimension(0) );
  if(compact) {
    assert( C.dimension(0) == T1.dimension(0) );
    assert( C.dimension(1) == B.dimension(0) );
  } else {
    assert( T2.dimension(1) == B.dimension(0) );
    assert( C.dimension(0) == conjA.dimension(0) );
    assert( C.dimension(1) == T2.dimension(1) );
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
  assert( conjA.dimension(0) == B.dimension(0) );
  assert( conjA.dimension(1) == T1.dimension(1) );
  assert( B.dimension(1) == T1.dimension(0) );
  assert( T1.dimension(1) == B.dimension(1) );

  using ma::T;

  // T(B)*conj(A) 
  ma::product(T(B),conjA,T1);  

  return static_cast<Tp>(ma::determinant(T1,IWORK));
}

} // namespace base

} // namespace qmcplusplus 

#endif
