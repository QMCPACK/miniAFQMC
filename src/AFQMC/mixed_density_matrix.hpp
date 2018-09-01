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

#include "Numerics/detail/blas.hpp"
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
inline void MixedDensityMatrix(const MatA& conjA, const MatB& B, MatC&& C, Mat&& T1, Mat&& T2, IBuffer& IWORK, TBuffer& WORK, Tp* ovlp, bool compact=true)
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
  ma::product(T(B),conjA,std::forward<Mat>(T1));  

  // T1 = T1^(-1)
  ma::invert(std::forward<Mat>(T1),IWORK,WORK,ovlp);

  if(compact) {

    // C = T1 * T(B)
    ma::product(T1,T(B),std::forward<MatC>(C)); 

  } else {

    // T2 = T1 * T(B)
    ma::product(std::forward<Mat>(T1),T(B),std::forward<Mat>(T2)); 

    // C = conj(A) * T2
    ma::product(conjA,std::forward<Mat>(T2),std::forward<MatC>(C));

  }
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
          class IBuffer,
          class TBuffer
        >
inline void Overlap(const MatA& conjA, const MatB& B, Mat&& T1, IBuffer& IWORK, TBuffer& WORK, Tp* ovlp)
{
  // check dimensions are consistent
  assert( conjA.shape()[0] == B.shape()[0] );
  assert( conjA.shape()[1] == T1.shape()[1] );
  assert( B.shape()[1] == T1.shape()[0] );
  assert( T1.shape()[1] == B.shape()[1] );

  using ma::T;

  // T(B)*conj(A) 
  ma::product(T(B),conjA,std::forward<Mat>(T1));  

  ma::determinant(std::forward<Mat>(T1),IWORK,WORK,ovlp);
}

} // namespace base

namespace batched
{

// Batched Implementation
template< class MatA,
          class MatB,
          class MatD,
          class Mat,
          class IBuffer,
          class TBuffer
        >
// right now limited to NAEA == NAEB
inline void Overlap(MatA& conjA_alpha, MatA& conjA_beta, MatB& B, Mat&& TNN3D, IBuffer& IWORK, TBuffer& WORK, MatD && W_data)
{
  // check dimensions are consistent
  static_assert( MatB::dimensionality == 4, " MatB::dimensionality == 4" );
  static_assert( std::decay<Mat>::type::dimensionality == 3, "std::decay<Mat>::type::dimensionality == 3" );
  assert( B.shape()[1] == 2 );
  assert( conjA_alpha.shape()[0] == conjA_beta.shape()[0] );
  assert( conjA_alpha.shape()[1] == conjA_beta.shape()[1] );
  assert( conjA_alpha.shape()[0] == B.shape()[2] );
  assert( conjA_alpha.shape()[1] == TNN3D.shape()[2] );
  assert( B.shape()[3] == TNN3D.shape()[1] );
  assert( TNN3D.shape()[2] == B.shape()[3] );

  using ma::T;
  using pointer = typename MatB::allocator_type::pointer;
  using Type = typename pointer::value_type; 

  int nwalk = B.shape()[0];
  int nbatch = TNN3D.shape()[0];
  int M = conjA_alpha.shape()[1];
  int N = B.shape()[3]; 
  int K = conjA_alpha.shape()[0]; 
  int lda = conjA_alpha.strides()[0]; 
  int ldw = B.strides()[2]; 
  int ldN = TNN3D[0].strides()[0]; 

  std::vector<pointer> Aarray;
  std::vector<pointer> Warray;
  std::vector<pointer> NNarray(nbatch);
  Warray.reserve(nbatch);
  Aarray.reserve(nbatch);
  for(int i=0; i<nbatch; i++)
    NNarray[i] = TNN3D[i].origin();
  int nloop = (2*nwalk + nbatch - 1)/nbatch;
  for(int nb=0, nw=0; nb<nloop; nb++) {

    int nw_=nw;
    // create list of pointers
    Warray.clear();
    Aarray.clear();
    for(int i=0; i<nbatch; i++) {
      Warray.push_back(B[nw/2][nw%2].origin());
      if(nw%2==0)
        Aarray.push_back(conjA_alpha.origin());
      else
        Aarray.push_back(conjA_beta.origin());
      nw++;
      if(nw == 2*nwalk) break;
    }

    // T(B)*conj(A)
    using BLAS_CPU::gemmBatched;
    using BLAS_GPU::gemmBatched;
    // careful with fortan ordering
    gemmBatched('N','T',M,N,K,ComplexType(1.0),Aarray.data(),lda,Warray.data(),ldw,ComplexType(0.0),NNarray.data(),ldN,Warray.size());        

    using LAPACK_CPU::getrfBatched;
    using LAPACK_GPU::getrfBatched;
    getrfBatched(M,NNarray.data(),ldN, IWORK.origin(), IWORK.origin()+nbatch*M, Warray.size());

    // determinant
    for(int i=0; i<nbatch; i++) {

      ma::determinant_from_getrf<Type>(M, NNarray[i], ldN, IWORK.origin()+i*M, 
                                to_address(W_data[nw_/2].origin()+(nw_%2)));

      nw_++;
      if(nw_ == 2*nwalk) break;
    }
  }

}

// Serial Implementation
template< class MatA,
          class MatB,
          class MatC,
          class MatD,
          class Mat,
          class IBuffer,
          class TBuffer
        >
inline void MixedDensityMatrix(MatA& conjA_alpha, MatA& conjA_beta, MatB& B, MatC&& C, Mat&& TNN3D, Mat&& TNM3D, IBuffer& IWORK, TBuffer& WORK, MatD && W_data)
{
  // check dimensions are consistent

  static_assert( MatB::dimensionality == 4, " MatB::dimensionality == 4" );
  static_assert( std::decay<MatC>::type::dimensionality == 4, " MatC::dimensionality == 4" );
  static_assert( std::decay<Mat>::type::dimensionality == 3, "std::decay<Mat>::type::dimensionality == 3" );
  assert( B.shape()[1] == 2 );
  assert( C.shape()[1] == 2 );
  assert( conjA_alpha.shape()[0] == conjA_beta.shape()[0] );
  assert( conjA_alpha.shape()[1] == conjA_beta.shape()[1] );
  assert( conjA_alpha.shape()[0] == B.shape()[2] );
  assert( conjA_alpha.shape()[1] == TNN3D.shape()[2] );
  assert( B.shape()[3] == TNN3D.shape()[1] );
  assert( TNN3D.shape()[2] == B.shape()[3] );
  assert( TNM3D.shape()[1] == TNN3D.shape()[1] );
  assert( C.shape()[2] == TNN3D.shape()[1] );
  assert( C.shape()[3] == B.shape()[2] );

  using ma::T;
  using pointer = typename MatB::allocator_type::pointer;
  using Type = typename pointer::value_type;

  int nwalk = B.shape()[0];
  int nbatch = TNN3D.shape()[0];
  int M = conjA_alpha.shape()[1];
  int N = B.shape()[3];
  int K = conjA_alpha.shape()[0];
  int lda = conjA_alpha.strides()[0];
  int ldw = B.strides()[2];
  int ldN = TNN3D[0].strides()[0];
  int ldC = C.strides()[2];

  assert( WORK.size() >= nbatch*M*M );

  std::vector<pointer> Aarray;
  std::vector<pointer> Carray;
  std::vector<pointer> Warray;
  std::vector<pointer> workArray(nbatch);
  std::vector<pointer> NNarray(nbatch);
  Warray.reserve(nbatch);
  Aarray.reserve(nbatch);
  Carray.reserve(nbatch);
  for(int i=0; i<nbatch; i++) {
    NNarray[i] = TNN3D[i].origin();
    workArray[i] = WORK.origin()+i*M*M;
  }
  int nloop = (2*nwalk + nbatch - 1)/nbatch;
  for(int nb=0, nw=0; nb<nloop; nb++) {

    int nw_=nw;
    // create list of pointers
    Warray.clear();
    Aarray.clear();
    Carray.clear();
    for(int i=0; i<nbatch; i++) {
      Warray.push_back(B[nw/2][nw%2].origin());
      Carray.push_back(C[nw/2][nw%2].origin());
      if(nw%2==0)
        Aarray.push_back(conjA_alpha.origin());
      else
        Aarray.push_back(conjA_beta.origin());
      nw++;
      if(nw == 2*nwalk) break;
    }

    // T(B)*conj(A)
    using BLAS_CPU::gemmBatched;
    using BLAS_GPU::gemmBatched;
    // careful with fortan ordering
    gemmBatched('N','T',M,N,K,ComplexType(1.0),Aarray.data(),lda,Warray.data(),ldw,ComplexType(0.0),NNarray.data(),ldN,Warray.size());      

    // Ivnert
    using LAPACK_CPU::getrfBatched;
    using LAPACK_GPU::getrfBatched;
    getrfBatched(M,NNarray.data(),ldN, IWORK.origin(), IWORK.origin()+nbatch*M, Warray.size());

    for(int i=0; i<nbatch; i++) {

      ma::determinant_from_getrf<Type>(M, NNarray[i], ldN, IWORK.origin()+i*M,
                                to_address(W_data[nw_/2].origin()+(nw_%2)));

      nw_++;
      if(nw_ == 2*nwalk) break;
    }

    using LAPACK_CPU::getriBatched;
    using LAPACK_GPU::getriBatched;
    getriBatched(M,NNarray.data(),ldN, IWORK.origin(), workArray.data(), M*M, IWORK.origin()+nbatch*M, Warray.size());    

    // C = T1 * T(B)
    gemmBatched('T','N',K,M,M,ComplexType(1.0),Warray.data(),ldw,NNarray.data(),ldN,ComplexType(0.0),Carray.data(),ldC,Warray.size());      
  }

}

} // namespace batched

} // namespace qmcplusplus 

#endif
