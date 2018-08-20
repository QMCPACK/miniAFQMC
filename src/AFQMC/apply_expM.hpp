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
inline void apply_expM( const MatA& V, MatB&& S, MatC&& T1, MatC&& T2, int order=6)
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
  MatC& rT1 = T1;
  MatC& rT2 = T2;

  T1 = S;
  for(int n=1; n<=order; n++) {
    ComplexType fact = ComplexType(0.0,1.0)*static_cast<ComplexType>(1.0/static_cast<double>(n));
    ma::product(fact,V,rT1,zero,rT2);
    // S += (*pT2); 
    ma::axpy(one,rT2,S);
/*
    for(int i=0, ie=S.shape()[0]; i<ie; i++)
     for(int j=0, je=S.shape()[1]; j<je; j++)
      S[i][j] += rT2[i][j];
*/
    std::swap(rT1,rT2);
  }

}

}

}

#endif
