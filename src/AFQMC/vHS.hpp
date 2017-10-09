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


#ifndef  AFQMC_VHS_HPP 
#define  AFQMC_VHS_HPP 

#include "Numerics/ma_operations.hpp"
#include "mpi.h"

namespace qmcplusplus
{

namespace base
{

/*
 * Calculates the H-S potential: 
 *  vHS = Spvn * X 
 *     vHS(ik,w) = sum_n Spvn(ik,n) * X(n,w) 
 */
// Serial Implementation
template< class SpMat,
	  class Mat	
        >
inline void get_vHS(const SpMat& Spvn, const Mat& X, Mat& v)
{
  // check dimensions are consistent
  assert( Spvn.cols() == X.shape()[0] );
  assert( Spvn.rows() == v.shape()[0] );
  assert( X.shape()[1] == v.shape()[1] );

  // Spvn*X 
  ma::product(Spvn,X,v);  
}

/*
 * Calculate S = exp(V)*S using a Taylor expansion of exp(V)
 */ 
template< class MatA,
          class MatB,
          class MatC
        >
inline void apply_expM( const MatA& V, MatB& S, MatC& T1, MatC& T2, int order=6)
{ 
  assert( V.shape()[0] == V.shape()[1] );
  assert( V.shape()[1] == S.shape()[0] );
  assert( S.shape()[0] == T1.shape()[0] );
  assert( S.shape()[1] == T1.shape()[1] );
  assert( S.shape()[0] == T2.shape()[0] );
  assert( S.shape()[1] == T2.shape()[1] );

  using ComplexType = typename std::decay<MatB>::type::element; 
  ComplexType zero(0.);
  MatC* pT1 = &T1;
  MatC* pT2 = &T2;

  T1 = S;
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,*pT1,zero,*pT2);
    // overload += ???
    for(int i=0, ie=S.shape()[0]; i<ie; i++)
     for(int j=0, je=S.shape()[1]; j<je; j++)
      S[i][j] += (*pT2)[i][j];
    std::swap(pT1,pT2);
  }

}

}

namespace shm 
{

/*
 * Calculates the H-S potential: 
 *  vHS = Spvn * X 
 *     vHS(ik,w) = sum_n Spvn(ik,n) * X(n,w) 
 */
template< class SpMat,
	  class MatA,
          class MatB	
        >
inline void get_vHS(const SpMat& Spvn, const MatA& X, MatB& v)
{
  // check dimensions are consistent
  assert( Spvn.cols() == X.shape()[0] );
  assert( Spvn.global_row() == v.shape()[0] );
  assert( X.shape()[1] == v.shape()[1] );
  assert( v.shape()[1] == v.strides()[0] );

  using ComplexType = typename std::decay<MatA>::type::element; 

  // Spvn*X 
  boost::multi_array_ref<ComplexType,2> v_(v.data()+Spvn.global_r0()*v.strides()[0], extents[Spvn.rows()][v.shape()[1]]);
  ma::product(Spvn,X,v_);  
}

/*
 * Calculate S = exp(V)*S using a Taylor expansion of exp(V)
 * In mpi3_shm version, S, T1, T2 are expected to be in shared memory.  
 */
template< class MatA,
          class MatB,
          class MatC
        >
inline void apply_expM( const MatA& V, MatB& S, MatC& T1, MatC& T2, int M0, int Mn, MPI_Comm& comm_, int order=6)
{
  assert( V.shape()[1] == S.shape()[0] );
  assert( S.shape()[0] == T1.shape()[0] );
  assert( S.shape()[1] == T1.shape()[1] );
  assert( S.shape()[0] == T2.shape()[0] );
  assert( S.shape()[1] == T2.shape()[1] );
  assert( M0 <= Mn );  
  assert( M0 >= 0);
  assert( (Mn-M0) == V.shape()[0]);

  using boost::indices;
  using range_t = boost::multi_array_types::index_range;
  using ComplexType = typename std::decay<MatB>::type::element;

  const ComplexType zero(0.);
  const ComplexType im(0.0,1.0);
  MatC* pT1 = &T1;
  MatC* pT2 = &T2;

  T1[indices[range_t(M0,Mn)][range_t()]] = S[indices[range_t(M0,Mn)][range_t()]];
  MPI_Barrier(comm_);
  for(int n=1; n<=order; n++) {
    const ComplexType fact = im*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,*pT1,zero,(*pT2)[indices[range_t(M0,Mn)][range_t()]]);
    MPI_Barrier(comm_);
    // overload += ???
    for(int i=M0; i<Mn; i++)
     for(int j=0, je=S.shape()[1]; j<je; j++)
      S[i][j] += (*pT2)[i][j];
    MPI_Barrier(comm_);
    std::swap(pT1,pT2);
  }

}

}

}

#endif
