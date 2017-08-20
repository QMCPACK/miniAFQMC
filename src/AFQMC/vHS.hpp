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

#ifndef  AFQMC_VHS_HPP 
#define  AFQMC_VHS_HPP 

#include "Numerics/ma_operations.hpp"
//#include "Numerics/ma_sparse_operations.hpp"

//temporary
#include "Numerics/SparseMatrixOperations.hpp"

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
	  class ValueMat	
        >
inline void get_vHS(const SpMat& Spvn, const ValueMat& X, ValueMat& v)
{
  // check dimensions are consistent
  assert( Spvn.cols() == X.shape()[0] );
  assert( Spvn.rows() == v.shape()[0] );
  assert( X.shape()[1] == v.shape()[1] );

  typedef typename std::decay<ValueMat>::type::element ComplexType;

  // Spvn*X 
  //ma::product(Spvn,X,v);  
  SparseMatrixOperators::product_SpMatM( Spvn.rows(), X.shape()[1], Spvn.cols(),
          ComplexType(1.,0.), Spvn.values(), Spvn.column_data(), Spvn.row_index(),
          X.data(), X.strides()[0],
          ComplexType(0.,0.), v.data(), v.strides()[0] );
}

/*
 * Calculate S = exp(V)*S using a Taylor expansion of exp(V)
 */ 
template< class ValueMatA,
          class ValueMatB,
          class ValueMatC
        >
inline void apply_expM( const ValueMatA& V, ValueMatB& S, ValueMatC& T1, ValueMatC& T2, int order=6)
{ 
  assert( V.shape()[0] == V.shape()[1] );
  assert( V.shape()[1] == S.shape()[0] );
  assert( S.shape()[0] == T1.shape()[0] );
  assert( S.shape()[1] == T1.shape()[1] );
  assert( S.shape()[0] == T2.shape()[0] );
  assert( S.shape()[1] == T2.shape()[1] );

  typedef typename std::decay<ValueMatB>::type::element ComplexType;
  ComplexType zero(0.);

  T1 = S;
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,T1,zero,T2);
    T1  = T2;
    // overload += ???
    for(int i=0, ie=S.shape()[0]; i<ie; i++)
     for(int j=0, je=S.shape()[1]; j<je; j++)
      S[i][j] += T1[i][j];

  }

}

}

}

#endif
