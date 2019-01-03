////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////


#ifndef  AFQMC_ROTATE_PADDED_HPP 
#define  AFQMC_ROTATE_PADDED_HPP 

#include <numeric>
#include "Configuration.h"
#include "Utilities/UtilityFunctions.h"
#include "Numerics/ma_operations.hpp"
#include "multi/array_ref.hpp"

#include "Numerics/detail/cuda_pointers.hpp"

namespace qmcplusplus
{

namespace afqmc 
{

namespace ma_rotate_padded
{

// Likn and Lank are padded to nmo_max and nchol_max
template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank(MultiArray2DA&& Aia, MultiArray3DB&& Likn, 
                                MultiArray3DC&& Lank, MultiArray2D && buff)  
{
  int na = Aia.shape()[1];
  int ni = Aia.shape()[0];

  int nmo =  Likn.shape()[0];
  int nchol =  Likn.shape()[2];
  assert(Likn.shape()[1]==nmo);

  assert(Lank.shape()[1]==nchol);
  assert(Lank.shape()[2]==nmo);

  assert(buff.shape()[0] >= nmo);
  assert(buff.shape()[1] >= nchol);

  // later on check that all element types are consistent!!!
  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrC = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmC = typename std::decay<MultiArray3DC>::type::element;

  using ptr2 = typename std::decay<MultiArray2D>::type::element_ptr;
  using elm2 = typename std::decay<MultiArray2D>::type::element;


  boost::multi::array_ref<elmB,2,ptrB> Li_kn(Likn.origin(),{ni,nmo*nchol});      
  boost::multi::array_ref<elm2,1,ptr2> buff1D(buff.origin(),{nmo*nchol});
  boost::multi::array_ref<elm2,2,ptr2> buff2D(buff.origin(),{nmo,nchol});

  for(int a=0; a<na; a++) {
    ma::product(ma::T(Li_kn),Aia({0,ni},a),buff1D);
    boost::multi::array_ref<elmC,2,ptrC> Lnk(Lank[a].origin(),{nchol,nmo});
    ma::transpose(buff2D({0,nmo},{0,nchol}),Lank[a]);
  }
}

// Lkin is padded to nmo_max and nchol_max
// Lank is not padded
template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank_from_Lkin(MultiArray2DA&& Aia, MultiArray3DB&& Lkin,
                                MultiArray3DC&& Lank, MultiArray2D && buff)
{
  int na = Aia.shape()[1];
  int ni = Aia.shape()[0];

  int nmo =  Lkin.shape()[0];
  int nchol =  Lkin.shape()[2];
  assert(Lkin.shape()[1]==nmo);

  assert(Lank.shape()[1]==nchol);
  assert(Lank.shape()[2]==nmo);

  assert(buff.num_elements() >= na*nchol);

  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrB = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DC>::type::element;

  using ptr2 = typename std::decay<MultiArray2D>::type::element_ptr;
  using elm2 = typename std::decay<MultiArray2D>::type::element;

  boost::multi::array_ref<elm2,2,ptr2> bna(buff.origin(),{nchol,na});      
  // Lank[a][n][k] = sum_i Aia[i][a] conj(Lkin[k][i][n])
  for(int k=0; k<nmo; k++) {
    ma::product(ma::H(Lkin[k].sliced(0,ni)),Aia,bna);
    for(int n=0; n<nchol; n++) {
      using BLAS_CPU::copy;
      using BLAS_GPU::copy;
      copy(na, bna[n].origin(), 1, Lank[0][n].origin()+k, Lank.stride(0));
    }
  }
}

// new routines

// Likn and Lakn are padded to nmo_max and nchol_max
// IMPORTANT: Likn must be zero for padded indexes!!! 
template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLakn(MultiArray2DA&& Aia, MultiArray3DB&& Likn,
                                MultiArray3DC&& Lakn, MultiArray2D && buff)
{
  int ni = Aia.shape()[0];
  int na = Aia.shape()[1];

  int nmo =  Likn.shape()[0];
  int nchol =  Likn.shape()[2];
  assert(Likn.shape()[1]==nmo);

  assert(Lakn.shape()[1]==nmo);
  assert(Lakn.shape()[2]==nchol);

  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrC = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmC = typename std::decay<MultiArray3DC>::type::element;

  boost::multi::array_ref<elmB,2,ptrB> Li_kn(Likn.origin(),{ni,nmo*nchol});      
  boost::multi::array_ref<elmC,2,ptrC> La_kn(Lakn.origin(),{na,nmo*nchol});      

  ma::product(ma::T(Aia),Li_kn,La_kn);
}

// Lkin is padded to nmo_max and nchol_max
// Lakn is not padded
template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLakn_from_Lkin(MultiArray2DA&& Aia, MultiArray3DB&& Lkin,
                                MultiArray3DC&& Lakn, MultiArray2D && buff)
{
  int ni = Aia.shape()[0];
  int na = Aia.shape()[1];

  int nmo =  Lkin.shape()[0];
  int nchol =  Lkin.shape()[2];
  assert(Lkin.shape()[1]==nmo);

  assert(Lakn.shape()[1]==nmo);
  assert(Lakn.shape()[2]==nchol);

  assert(buff.num_elements() >= na*nchol);

  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrB = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DC>::type::element;

  using ptr2 = typename std::decay<MultiArray2D>::type::element_ptr;
  using elm2 = typename std::decay<MultiArray2D>::type::element;

  boost::multi::array_ref<elm2,2,ptr2> bna(buff.origin(),{nchol,na});
  // Lakn[a][n][k] = sum_i Aia[i][a] conj(Lkin[k][i][n])
  for(int k=0; k<nmo; k++) {
    ma::product(ma::H(Lkin[k].sliced(0,ni)),Aia,bna);
    for(int n=0; n<nchol; n++) {
      using BLAS_CPU::copy;
      using BLAS_GPU::copy;
      copy(na, bna[n].origin(), 1, Lakn[0][k].origin()+n, Lakn.stride(0));
    }
  }
}


}

}

}

#endif
