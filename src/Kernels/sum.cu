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

#include<cassert>
#include <complex>
#include<cuda.h>
#include <thrust/complex.h>
#include<cuda_runtime.h>

namespace kernels 
{

template<typename T>
__global__ void kernel_sum(int M, int N, T const* A, int lda, T* res) 
{
   int i = threadIdx.x + blockDim.x*blockIdx.x;
   if (i<N) {
     y[i*incy] += alpha*A[i*lda+i];
   }
}

void sum(int M, int N, double const* A, int lda) 
{
  kernel_sum<<1,256>>>(M,N,A,lda);
}


}

