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


#ifndef  AFQMC_ROTATE_HPP 
#define  AFQMC_ROTATE_HPP 

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

namespace ma_rotate
{

template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank(MultiArray2DA&& Aia, MultiArray3DB&& Likn, 
                                MultiArray3DC&& Lank, MultiArray2D && buff)  
{
  int na = Aia.shape()[1];
  int ni = Aia.shape()[0];
  int nk = Likn.shape()[1];
  int nchol = Likn.shape()[2];
  assert(Likn.shape()[0]==ni);
  assert(Lank.shape()[0]==na);
  assert(Lank.shape()[1]==nchol);
  assert(Lank.shape()[2]==nk);
  assert(buff.shape()[0] >= nk);
  assert(buff.shape()[1] >= nchol);

  // later on check that all element types are consistent!!!
  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrC = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmC = typename std::decay<MultiArray3DC>::type::element;

  boost::multi::array_ref<elmB,2,ptrB> Li_kn(Likn.origin(),{ni,nk*nchol});      
  boost::multi::array_ref<elmC,2,ptrC> La_kn(Lank.origin(),{na,nk*nchol});      

  ma::product(ma::T(Aia),Li_kn,La_kn);
  for(int a=0; a<na; a++) {
    boost::multi::array_ref<elmC,2,ptrC> Lkn(Lank[a].origin(),{nk,nchol});
    boost::multi::array_ref<elmC,2,ptrC> Lnk(Lank[a].origin(),{nchol,nk});
    for(int k=0; k<nk; k++)
      cuda::copy_n(Lkn[k].origin(),nchol,buff[k].origin());
    ma::transpose(buff({0,nk},{0,nchol}),Lnk);
  }
}

template< class MultiArray2DA, class MultiArray3DB, class MultiArray3DC, class MultiArray2D>
void getLank_from_Lkin(MultiArray2DA&& Aia, MultiArray3DB&& Lkin,
                                MultiArray3DC&& Lank, MultiArray2D && buff)
{
  int na = Aia.shape()[1];
  int ni = Aia.shape()[0];
  int nk = Lkin.shape()[0];
  int nchol = Lkin.shape()[2];
  assert(Lkin.shape()[0]==ni);
  assert(Lank.shape()[0]==na);
  assert(Lank.shape()[1]==nchol);
  assert(Lank.shape()[2]==nk);
  assert(buff.num_elements() >= na*nchol);

  using ptrB = typename std::decay<MultiArray3DB>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DB>::type::element;

  using ptrB = typename std::decay<MultiArray3DC>::type::element_ptr;
  using elmB = typename std::decay<MultiArray3DC>::type::element;

  using ptr2 = typename std::decay<MultiArray2D>::type::element_ptr;
  using elm2 = typename std::decay<MultiArray2D>::type::element;

  boost::multi::array_ref<elm2,2,ptr2> bna(buff.origin(),{nchol,na});      
  // Lank[a][n][k] = sum_i Aia[i][a] conj(Lkin[k][i][n])
  for(int k=0; k<nk; k++) {
    ma::product(ma::H(Lkin[k]),Aia,bna);
    for(int n=0; n<nchol; n++) {
//auto&& b_(bna[n]);
//auto&& L_(Lank({0,na},n,k));
//std::cout<<"   -  n: " <<n <<" " <<b_.shape()[0] <<" " <<L_.shape()[0] <<" " <<L_.stride(0) <<"\n";
      // bug in operator()  
      //ma::copy(bna[n],Lank({0,na},n,k));     
      using BLAS_CPU::copy;
      using BLAS_GPU::copy;
      copy(na, bna[n].origin(), 1, Lank[0][n].origin()+k, Lank.stride(0));
    }
    //for(int a=0; a<na; a++)    
      //for(int n=0; n<nchol; n++)
        //Lank[a][n][k] = bna[n][a];     
  }
}

}

}

}

#endif
