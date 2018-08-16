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
#include "Numerics/OhmmsBlas.h"
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
inline void apply_expM( const MatA& V, MatB&& S, MatC&& T1, MatC&& T2, int order=6)
{ 
  assert( V.dimension_0() == V.dimension_1() );
  assert( V.dimension_1() == S.dimension_0() );
  assert( S.dimension_0() == T1.dimension_0() );
  assert( S.dimension_1() == T1.dimension_1() );
  assert( S.dimension_0() == T2.dimension_0() );
  assert( S.dimension_1() == T2.dimension_1() );

  using ComplexType = typename std::decay<MatB>::value_type; 
  ComplexType zero(0.);
  MatC& rT1 = T1;
  MatC& rT2 = T2;

  T1 = S;
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,rT1,zero,rT2);
    // overload += ???
    // S += (*pT2); 
    for(int i=0, ie=S.dimension_0(); i<ie; i++)
     for(int j=0, je=S.dimension_1(); j<je; j++)
       S(i, j) += rT2(i, j);
    std::swap(rT1,rT2);
  }

}

}

}

#endif
