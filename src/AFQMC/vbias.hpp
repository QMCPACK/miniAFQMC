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
//#include "Numerics/ma_sparse_operations.hpp"

//temporary
#include "Numerics/SparseMatrixOperations.hpp"

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
	  class Mat
        >
inline void get_vbias(const SpMat& Spvn, const Mat& G, Mat& v, bool transposed)
{

  typedef typename std::decay<Mat>::type::element Type;
  if(transposed) {

    assert( Spvn.cols() == G.shape()[0] );
    assert( Spvn.rows() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    // Spvn*G  
    //ma::product(Spvn,G,v);
    SparseMatrixOperators::product_SpMatM( Spvn.rows(), G.shape()[1], Spvn.cols(),
        Type(1.), Spvn.values(), Spvn.column_data(), Spvn.index_begin(), Spvn.index_end(), 
        G.data(), G.strides()[0],
        Type(0.), v.data(), v.strides()[0] );

  } else {

    assert( Spvn.rows()*2 == G.shape()[0] );
    assert( Spvn.cols() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    using ma::T;

    // only works if stride()[0] == shape()[1]

    // T(Spvn)*G 
    //ma::product(T(Spvn),G[indices[range_t(0,G.shape()[0]/2)][range_t(0,G.shape()[1])]],v);  
    SparseMatrixOperators::product_SpMatTM( Spvn.rows(), G.shape()[1], Spvn.cols(),
          Type(1.), Spvn.values(), Spvn.column_data(), Spvn.row_index(),
          G.data(), G.strides()[0],
          Type(0.), v.data(), v.strides()[0] );

    //ma::product(Type(1.),T(Spvn),
    //      G[indices[range_t(0,G.shape()[0]/2)][range_t(0,G.shape()[1])]],Type(1.),v);  
    SparseMatrixOperators::product_SpMatTM( Spvn.rows(), G.shape()[1], Spvn.cols(),
          Type(1.), Spvn.values(), Spvn.column_data(), Spvn.row_index(),
          G.data()+G.shape()[0]*G.shape()[1]/2, G.strides()[0],
          Type(1.), v.data(), v.strides()[0] );
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
// Serial Implementation

template< class SpMat,
          class MatA,
	  class MatB
        >
inline void get_vbias(const SpMat& Spvn, const MatA& G, MatB& v, bool transposed)
{

  typedef typename std::decay<MatA>::type::element Type;
  if(transposed) {

    assert( Spvn.cols() == G.shape()[0] );
    assert( Spvn.global_row() == v.shape()[0] );
    assert( G.shape()[1] == v.shape()[1] );

    // Spvn*G  
    //ma::product(Spvn,G,v);
    SparseMatrixOperators::product_SpMatM( Spvn.rows(), G.shape()[1], Spvn.cols(),
        Type(1.), Spvn.values(), Spvn.column_data(), Spvn.index_begin(), Spvn.index_end(), 
        G.data(), G.strides()[0],
        Type(0.), v.data() + Spvn.global_r0()*v.strides()[0], v.strides()[0] );


  } else {

    assert( Spvn.rows()*2 == G.shape()[0] );
    assert( Spvn.cols() == v.shape()[0] );    // Spvn.cols() in this case is cN
    assert( G.shape()[1] == v.shape()[1] );

    using ma::T;


    // Careful here!!!
    // v can not be in shared memory since all the rows below Spvn.cols() are touched!!!!
    // T(Spvn)*G 
    //ma::product(T(Spvn),G[indices[range_t(0,G.shape()[0]/2)][range_t(0,G.shape()[1])]],v);  
    SparseMatrixOperators::product_SpMatTM( Spvn.rows(), G.shape()[1], Spvn.cols(),
          Type(1.), Spvn.values(), Spvn.column_data(), Spvn.index_begin(), Spvn.index_end(), 
          G.data(), G.strides()[0],
          Type(0.), v.data(), v.strides()[0] );

    //ma::product(Type(1.),T(Spvn),
    //      G[indices[range_t(0,G.shape()[0]/2)][range_t(0,G.shape()[1])]],Type(1.),v);  
    SparseMatrixOperators::product_SpMatTM( Spvn.rows(), G.shape()[1], Spvn.cols(),
          Type(1.), Spvn.values(), Spvn.column_data(), Spvn.index_begin(), Spvn.index_end(),
          G.data()+G.shape()[0]*G.shape()[1]/2, G.strides()[0],
          Type(1.), v.data(), v.strides()[0] );

  }

}

}

}

#endif
