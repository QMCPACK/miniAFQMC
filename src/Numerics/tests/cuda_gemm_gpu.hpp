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


#ifndef MA_CUDA_GEMM_TESTING_HPP
#define MA_CUDA_GEMM_TESTING_HPP

#include <iostream>
#include <random>

#include <Utilities/Timer.h>

#include <boost/math/special_functions/relative_difference.hpp> 

#include "multi/array_ref.hpp"
#include "multi/array.hpp"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublasXt.h"

#include "Numerics/detail/cuda_pointers.hpp"
#include "Numerics/ma_operations.hpp"

using qmcplusplus::Timer;
using boost::multi::array;
using boost::multi::array_ref;
using ma::T;
using ma::product;

template<class T, class Allocator>
inline void time_gemm_batched(Timer timer, Allocator alloc, int M, int K, int N, int nloop, int nbatch=10) 
{
  array<T,3,Allocator> A( {nbatch,M,K}, alloc); 
  array<T,3,Allocator> B( {nbatch,K,N}, alloc);
  array<T,3,Allocator> C( {nbatch,M,N}, alloc);

  array<T,1,Allocator> alpha( {nbatch}, alloc);
  array<T,1,Allocator> beta( {nbatch}, alloc);

  array<T,2,Allocator> A_( {M,K}, alloc); 
  array<T,2,Allocator> B_( {K,N}, alloc);
  array<T,2,Allocator> C_( {M,N}, alloc);

  product(T(),A_,B_,T(),C_);
  if(nloop>0) {
    timer.restart();
    for(int i=0; i<nloop; i++)
      product(T(),A_,B_,T(),C_);
    double t = timer.elapsed();
    std::cout<<" Time for " <<nbatch <<" independent GEMMs: " <<nbatch*t/double(nloop) <<std::endl;
  }

  T alp{}; 
  T bet{}; 

  cublasZgemm3m(*alloc.handles_.cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,
                             &alp,to_address(A_.origin()),M,
                             to_address(B_.origin()),K,
                             &bet,to_address(C_.origin()),M);
  cudaDeviceSynchronize ();
  if(nloop>0) {
    timer.restart();
    for(int i=0; i<nloop; i++) {
      cublasZgemm3m(*alloc.handles_.cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,
                             &alp,to_address(A_.origin()),M,
                             to_address(B_.origin()),K,
                             &bet,to_address(C_.origin()),M);
      cudaDeviceSynchronize ();
    }
    double t = timer.elapsed();
    std::cout<<" Time for " <<nbatch <<" independent GEMM3ms: " <<nbatch*t/double(nloop) <<std::endl;
  }

  long long int  offsetA = M*K;
  long long int  offsetB = K*N;
  long long int  offsetC = M*N;
  // not really correct, but not checking the result anyway 
  cublasZgemmStridedBatched(*alloc.handles_.cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,
                             &alp,to_address(A.origin()),M,offsetA,
                             to_address(B.origin()),K,offsetB,
                             &bet,to_address(C.origin()),M,offsetC, nbatch);
  cudaDeviceSynchronize ();
  if(nloop>0) {
    timer.restart();
    for(int i=0; i<nloop; i++) {
      cublasZgemmStridedBatched(*alloc.handles_.cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,
                             &alp,to_address(A.origin()),M,offsetA,
                             to_address(B.origin()),K,offsetB,
                             &bet,to_address(C.origin()),M,offsetC, nbatch);
      cudaDeviceSynchronize ();
    }
    double t = timer.elapsed();
    std::cout<<" Time for StridedBatched GEMM: " <<t/double(nloop) <<std::endl;
  }
  std::cout<<std::endl;

}

template<typename Type>
inline void time_cuda_gemm_batched(int M, int K, int N, int nloop, int ndev, int nbatch)
{
  using cuda::cublas_check;
  using cuda::cusolver_check;

  Timer timer;

  cublasHandle_t cublas_handle;
  cublasXtHandle_t cublasXt_handle;
  cusolverDnHandle_t cusolverDn_handle;
  cublas_check(cublasCreate (& cublas_handle ), "cublasCreate");

  cuda::gpu_handles handles{&cublas_handle,&cublasXt_handle,&cusolverDn_handle};

  {
    // Unified Memory
    using Allocator = cuda::cuda_gpu_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" GPU Memory " <<std::endl;
    time_gemm_batched<Type,Allocator>(timer,alloc,M,K,N,nloop,nbatch);
  }

  cublasDestroy(cublas_handle);

}


#endif
