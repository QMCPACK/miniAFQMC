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

/** @file vHS.hpp
 *  @brief HS potential
 */

#ifndef  AFQMC_VHS_HPP 
#define  AFQMC_VHS_HPP 

#include "Numerics/ma_operations.hpp"
#include<iostream>

namespace qmcplusplus
{

namespace base
{

/**
 * Calculate \f$S = \exp(V)*S \f$ using a Taylor expansion of exp(V)
 */ 
template< class MatA,
          class MatB,
          class MatC
        >
inline void apply_expM( const MatA& V, MatB&& S, MatC & T1, MatC & T2, int order=6)
{ 
  assert( V.shape()[0] == V.shape()[1] );
  assert( V.shape()[1] == S.shape()[0] );
  assert( S.shape()[0] == T1.shape()[0] );
  assert( S.shape()[1] == T1.shape()[1] );
  assert( S.shape()[0] == T2.shape()[0] );
  assert( S.shape()[1] == T2.shape()[1] );

  using ComplexType = typename std::decay<MatB>::type::element; 
  ComplexType zero(0.);
  ComplexType one(1.);
  MatC* rT1 = &T1;
  MatC* rT2 = &T2;

  cuda::copy_n(S.origin(),S.num_elements(),T1.origin());
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,*rT1,zero,*rT2);
    ma::axpy(one,*rT2,S);
    std::swap(rT1,rT2);
  }

}

}  // namespace base

namespace batched
{

/**
 * Calculates: P1 * W[w][s] = TMN[w][s]  using batchedGemm 
 */
template< class MatA,
          class MatB,
          class MatC
        >
inline void applyP1( MatA& P1, MatB && W, MatC && TMN)
{
  static_assert(MatA::dimensionality == 2, "P1::dimensionality == 2");
  static_assert(std::decay<MatB>::type::dimensionality == 3, "W::dimensionality == 3");
  static_assert(std::decay<MatC>::type::dimensionality == 3, "TMN::dimensionality == 3");
  int nbatch = W.shape()[0]; 
  int NMO =  W.shape()[1];
  int NEL =  W.shape()[2];
  assert( TMN.shape()[0] >= nbatch );
  assert( TMN.shape()[1] == NMO );
  assert( TMN.shape()[2] == NEL );
  assert( P1.shape()[0] == NMO );
  assert( P1.shape()[1] == NMO );
  
  using CType = typename std::decay<MatB>::type::element;
  using pointer = typename std::decay<MatB>::type::element_ptr;

  int ldw = W.strides()[1]; 
  int ldmn = TMN.strides()[1];
  int ldp = P1.strides()[0]; 

  std::vector<pointer> Warray(nbatch);
  std::vector<pointer> Parray(nbatch,P1.origin());
  std::vector<pointer> MNarray(nbatch);
  for(int i=0; i<nbatch; i++) {
    Warray[i] = W[i].origin();
    MNarray[i] = TMN[i].origin();
  }  

  // TMN = P1 * W
  using BLAS_CPU::gemmBatched;
  using BLAS_GPU::gemmBatched;
  // careful with fortan ordering
  gemmBatched('N','N',NEL,NMO,NMO,CType(1.0),Warray.data(),ldw,Parray.data(),ldp,CType(0.0),MNarray.data(),ldmn,nbatch);

}

/**
 * Calculate \f$S = \exp(V)*S \f$ using a Taylor expansion of exp(V)
 */
template< class MatA,
          class MatB,
          class MatC
        >
inline void apply_expM(MatA& V, MatB& S, MatC& T1, MatC& T2, int order=6)
{
// WARNING: Only works for collinear right now
  static_assert(MatA::dimensionality == 3, "V::dimensionality == 3");
  static_assert(MatB::dimensionality == 3, "S::dimensionality == 3");
  static_assert(MatC::dimensionality == 3, "T1::dimensionality == 3");
  int nbatch = S.shape()[0];
  int NMO =  S.shape()[1];
  int NEL =  S.shape()[2];
  assert( V.shape()[0] >= nbatch/2 );
  assert( V.shape()[1] == NMO );
  assert( V.shape()[2] == NMO );
  assert( T1.shape()[0] >= nbatch );
  assert( T1.shape()[1] == NMO );
  assert( T1.shape()[2] == NEL );
  assert( T2.shape()[0] >= nbatch );
  assert( T2.shape()[1] == NMO );
  assert( T2.shape()[2] == NEL );

  using CType = typename std::decay<MatB>::type::element;
  using pointer = typename MatB::element_ptr;

  int ldV = V.strides()[1];
  int ldT = T1.strides()[1];

  std::vector<pointer> Varray(nbatch);
  std::vector<pointer> T1array(nbatch);
  std::vector<pointer> T2array(nbatch);
  for(int i=0; i<nbatch; i++) {
    Varray[i] = V[i/2].origin();
    T1array[i] = T1[i].origin();
    T2array[i] = T2[i].origin();
  }
  std::vector<pointer>* rT1 = &T1array;
  std::vector<pointer>* rT2 = &T2array;

  using CType = typename std::decay<MatB>::type::element;
  CType zero(0.);
  CType one(1.);
  CType im(0.0,1.);

  cuda::copy_n(S.origin(),nbatch*NMO*NEL,T1.origin());
  for(int n=1; n<=order; n++) {
    CType fact = im*static_cast<CType>(1.0/static_cast<double>(n));

    using BLAS_CPU::gemmBatched;
    using BLAS_GPU::gemmBatched;
    // careful with fortan ordering
    gemmBatched('N','N',NEL,NMO,NMO,fact,rT1->data(),ldT,Varray.data(),ldV,zero,
                                         rT2->data(),ldT,nbatch);

    using BLAS_CPU::axpy;
    using BLAS_GPU::axpy;
    // in QMCPACK, loop through matrices and use S[i].origin(), rT2->origin()+i*stride 
    axpy(S.num_elements(), one, (*rT2)[0], 1, S.origin(), 1);

    std::swap(rT1,rT2);
  }
}

}

}

#endif
