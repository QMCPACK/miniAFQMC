//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_BLAS_GPU_HPP
#define AFQMC_BLAS_GPU_HPP

#include<cassert>
#include "Numerics/detail/cublas_wrapper.hpp"
#include "Numerics/detail/cuda_pointers.hpp"

namespace BLAS_GPU
{

  template<typename T, 
           class ptr,
           typename = typename std::enable_if_t<ptr::blas_type == CUBLAS_BLAS_TYPE> 
          >
  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          T alpha,
                          ptr const& A, int lda,
                          ptr const& B, int ldb,
                          T beta,
                          ptr & C, int ldc)
  {
    // check that all handles are the same or consistent
std::cout<<" running cublasDgemm " <<" " <<M <<" " <<N <<" " <<K <<" " <<alpha <<" " <<beta <<" " <<lda <<" " <<ldb <<std::endl;
    if(CUBLAS_STATUS_SUCCESS != cublas_gemm(*A.cublas_handle,Atrans,Btrans,
                                            M,N,K,alpha,A.get(),lda,B.get(),ldb,beta,C.get(),ldc))
      throw std::runtime_error("Error: cublas_gemm returned error code.");
std::cout<<" done running cublasDgemm " <<std::endl;
  }

}

#endif
