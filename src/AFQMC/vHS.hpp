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
#include "Numerics/ma_sparse_operations.hpp"

/*
 * Calculates the H-S potential: 
 *  vHS = Spvn * X 
 *     vHS(ik,w) = sum_n Spvn(ik,n) * X(n,w) 
 */
// Serial Implementation
template< class ValueSpMat,
	  class ValueMat,	
	  typename = typename std::enable_if<(ValueSpMat::dimensionality == 2)>::type>,		
	  typename = typename std::enable_if<(ValueMat::dimensionality == 2)>::type>		
        >
inline void get_vHS(const ValueSpMat& Spvn, const ValueMat& X, ValueMat& v)
{
  // check dimensions are consistent
  assert( Spvn.shape()[1] == X.shape()[0] );
  assert( Spvn.shape()[0] == v.shape()[0] );
  assert( X.shape()[1] == v.shape()[1] );

  // Spvn*X 
  ma::product(Spvn,X,v);  
}


#endif
