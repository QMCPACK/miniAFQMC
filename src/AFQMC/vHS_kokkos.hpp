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

#include <Kokkos_Core.hpp>

#include "Numerics/ma_operations.hpp"

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
  assert( Spvn.cols() == X.dimesnion(0) );
  assert( Spvn.rows() == v.dimensione(0) );
  assert( X.dimension(1) == v.dimension(1) );

  // typedef typename std::decay<Mat>::type::element ComplexType;

  // Spvn*X 
  // ma::product(Spvn,X,v);
  KokkosSparse::spmv('n',1.0,Spvn,X,0.0,v);
  // Use Kokkos Sparse spmv
}

/*
 * Calculate S = exp(V)*S using a Taylor expansion of exp(V)
 */ 
template< class MatA,
          class MatB,
          class MatC,
          class MatD
        >
inline void apply_expM( const MatA& V, MatB& S, MatC& T1, MatrD& T2, int order=6)
{ 
  assert( V.dimension(0) == V.dimension(1) );
  assert( V.dimension(1) == S.dimension(0) );
  assert( S.dimension(0) == T1.dimension(0) );
  assert( S.dimension(1) == T1.dimension(1) );
  assert( S.dimension(0) == T2.dimension(0) );
  assert( S.dimension(1) == T2.dimension(1) );

  // typedef typename std::decay<MatB>::type::element ComplexType;
  typedef typename MatB::value_type ComplexType;
  ComplexType zero(0.);

  T1 = S;
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    // ma::product(fact,V,T1,zero,T2);
    KokkosBlas::gemm('n','n',fact,V,T1,0.0,T2);
    T1  = T2;
    // overload += ???
    for(int i=0, ie=S.dimension(0); i<ie; i++)
     for(int j=0, je=S.dimension(1); j<je; j++)
      S(i, j) += T1(i, j);
  }

}

}

}

#endif
