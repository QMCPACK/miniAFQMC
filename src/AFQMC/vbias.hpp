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
  assert( G.strides()[0] == G.shape()[1] );
  assert( G.strides()[1] == 1 );

//  typedef typename std::decay<MatA>::type::element TypeA;
	using TypeA = MatA::element;

  if(transposed) {

    assert( Spvn.cols() == G.shape()[0] );
    assert( Spvn.rows() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    // Spvn*G  
    ma::product(Spvn,G,v);

  } else {

    assert( Spvn.rows()*2 == G.shape()[0] );
    assert( Spvn.cols() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    using ma::T;

    // only works if stride()[0] == shape()[1]

    // T(Spvn)*G 
//    boost::multi_array_ref<const TypeA,2> Gup(G.data(), extents[G.shape()[0]/2][G.shape()[1]]);
//    boost::multi_array_ref<const TypeA,2> Gdn(G.data()+G.shape()[0]*G.shape()[1]/2, 
//                                                        extents[G.shape()[0]/2][G.shape()[1]]);

//    boost::const_multi_array_ref<TypeA,2> Gup(G.data(), extents[G.shape()[0]/2][G.shape()[1]]);
//    boost::const_multi_array_ref<TypeA,2> Gdn(G.data()+G.shape()[0]*G.shape()[1]/2, 
//                                                        extents[G.shape()[0]/2][G.shape()[1]]);

	using boost::multi_array_types::index_range;
//	auto Gup = G[index_range() <  G.shape()[0]/2][index_range()];
//	auto Gdn = G[index_range() >= G.shape()[0]/2][index_range()]; 

    // alpha
    ma::product(
    	T(Spvn), /*Gup*/ G[index_range(0, G.shape()[0]/2)][index_range()], v
    );  
    // beta
    ma::product(
    	TypeA(1.), T(Spvn), /*Gdn*/ G[index_range(G.shape()[0]/2, G.shape()[0]) ][index_range()], TypeA(1.), v
    );
  }
}

}

}

#endif
