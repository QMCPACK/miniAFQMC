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

  using TypeA = typename MatA::element;
  if(transposed) {

    assert( Spvn.cols() == G.shape()[0] );
    assert( Spvn.rows() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    // Spvn*G  
    ma::product(Spvn,G,v);

  } else {

    assert( G.strides()[0] == G.shape()[1] );
    assert( MatA::dimensionality == 2);
    assert( G.shape()[0]%2 == 0 );
    assert( Spvn.rows()*2 == G.shape()[0] );
    assert( Spvn.cols() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    using ma::T;

    // alpha
    ma::product(
       T(Spvn), G[indices[range_t() < range_t::index(G.shape()[0]/2)][range_t()]], v
    );  
     // beta
    ma::product(
       TypeA(1.), T(Spvn), G[indices[range_t::index(G.shape()[0]/2) <= range_t()][range_t()]], TypeA(1.), v
    );

  }
}

}

namespace shm 
{

/*
 * Calculates the bias potential: 
 *  vbias = T(Spvn) * G 
 *     vbias(n,w) = sum_ik Spvn(ik,n) * G(ik,w) 
 */

template< class SpMat,
          class MatA,
	  class MatB
        >
inline void get_vbias(const SpMat& Spvn, const MatA& G, MatB& v, bool transposed)
{
  assert( G.strides()[0] == G.shape()[1] );   // temporary restriction
  assert( G.strides()[1] == 1 );

  using TypeA = typename MatA::element;
  if(transposed) {

    assert( Spvn.cols() == G.shape()[0] );
    assert( Spvn.global_row() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );
    assert( v.shape()[1] == v.strides()[0]); // temporary restriction

    // Spvn*G  
    boost::multi_array_ref<TypeA,2> v_(v.data()+Spvn.global_r0()*v.strides()[0], extents[Spvn.rows()][v.shape()[1]]);
    ma::product(Spvn,G,v_);

  } else {

    assert( Spvn.rows()*2 == G.shape()[0] );
    assert( Spvn.cols() == v.shape()[0] );    // Spvn.cols() in this case is cN
    assert( G.shape()[1] == v.shape()[1] );

    using ma::T;

    // alpha
    ma::product(
       T(Spvn), G[indices[range_t() < range_t::index(G.shape()[0]/2)][range_t()]], v
    ); 
     // beta
    ma::product(
       TypeA(1.), T(Spvn), G[indices[range_t::index(G.shape()[0]/2) <= range_t()][range_t()]], TypeA(1.), v
    );

  }

}

}

}

#endif
