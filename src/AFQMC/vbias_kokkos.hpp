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


#ifndef  AFQMC_VBIAS_HPP 
#define  AFQMC_VBIAS_HPP 

#include <Kokkos_Core.hpp>

#include "Numerics/ma_operations.hpp"

namespace qmcplusplus
{

namespace base 
{

/*
 * Calculates the bias potential: 
 *  vbias = T(Spvn) * G 
 *     vbias(n,w) = sum_ik Spvn(ik,n) * G(ik,w) 
 */
// Serial Implementation

template< class SpMat,
	  class MatA,
          class MatB  
        >
inline void get_vbias(const SpMat& Spvn, const MatA& G, MatB& v, bool transposed)
{
  assert( G.dimension(0) == G.dimension(1) );
  assert( G.dimension(1) == 1 );

  typedef typename std::decay<MatA>::type::element TypeA;
  if(transposed) {

    assert( Spvn.cols() == G.dimension(0) );
    assert( Spvn.rows() == v.dimension(0) );
    assert( G.dimension(1) == v.dimension(1) );

    // Spvn*G  
    ma::product(Spvn,G,v);

  } else {

    assert( Spvn.rows()*2 == G.dimension(0) );
    assert( Spvn.cols() == v.dimension(0) );
    assert( G.dimension(1) == v.dimension(1) );

    using ma::T;

    // only works if stride()[0] == shape()[1]

    // T(Spvn)*G 
    // boost::multi_array_ref<const TypeA,2> Gup(G.data(), extents[G.shape()[0]/2][G.shape()[1]]);
    auto Gup = Kokkos::subview(G, std::pair<size_t,size_t>(0, G.dimension(0)/2 - 1), Kokkos::ALL());
    // boost::multi_array_ref<const TypeA,2> Gdn(G.data()+G.shape()[0]*G.shape()[1]/2, 
    //                            extents[G.shape()[0]/2][G.shape()[1]]);
    auto Gdn = Kokkos::subview(G, std::pair<size_t,size_t>(G.dimension(0)/2, G.dimension(0)-1), Kokkos::ALL()); 
    // alpha
    ma::product(T(Spvn),Gup,v);  
    // beta
    ma::product(TypeA(1.),T(Spvn),Gdn,TypeA(1.),v);
  }
}

}

}

#endif
