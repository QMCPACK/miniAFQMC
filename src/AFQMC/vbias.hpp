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

#ifndef  AFQMC_VBIAS_HPP 
#define  AFQMC_VBIAS_HPP 

#include "Numerics/ma_operations.hpp"
#include "Numerics/ma_sparse_operations.hpp"

/*
 * Calculates the bias potential: 
 *  vbias = T(Spvn) * G 
 *     vbias(n,w) = sum_ik Spvn(ik,n) * G(ik,w) 
 */
// Serial Implementation
template< bool transpose, 
	  class Tp,
          class ValueSpMat,
          class ValueMat,
          typename = typename std::enable_if<(ValueSpMat::dimensionality == 2)>::type>,
          typename = typename std::enable_if<(ValueMat::dimensionality == 2)>::type>
        >
inline void get_vbias(const Tp alpha, const ValueSpMat& Spvn, const ValueMat& G, const Tp beta, ValueMat& v);


template< false,
	  class Tp,
          class ValueSpMat,
	  class ValueMat,	
	  typename = typename std::enable_if<(ValueSpMat::dimensionality == 2)>::type>,		
	  typename = typename std::enable_if<(ValueMat::dimensionality == 2)>::type>		
        >
inline void get_vbias(const Tp alpha, const ValueSpMat& Spvn, const ValueMat& G, const Tp beta, ValueMat& v)
{
  // check dimensions are consistent
  assert( Spvn.shape()[0] == G.shape()[0] );
  assert( Spvn.shape()[1] == v.shape()[0] );
  assert( G.shape()[1] == v.shape()[1] );

  using ma::T;

  // T(Spvn)*G 
  ma::product(static_cast<G::value_type>(alpha),T(Spvn),G,static_cast<v::value_type>(beta),v);  
}


#endif
