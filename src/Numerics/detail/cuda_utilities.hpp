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

#ifndef AFQMC_CUDA_UTILITIES_HPP 
#define AFQMC_CUDA_UTILITIES_HPP

#include<cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"

  inline cublasOperation_t cublasOperation(char A) {
    if(A=='N' or A=='n')
      return CUBLAS_OP_N;
    else if(A=='T' or A=='t')
      return CUBLAS_OP_T;
    else if(A=='C' or A=='c')
      return CUBLAS_OP_C;
    else
      throw std::runtime_error("unknown cublasOperation option"); 
    return CUBLAS_OP_N;
  }

#endif
