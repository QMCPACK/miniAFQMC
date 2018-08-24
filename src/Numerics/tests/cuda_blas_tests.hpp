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


#ifndef MA_CUDA_BLAS_TESTING_HPP
#define MA_CUDA_BLAS_TESTING_HPP

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
#include "cublasXt.h"
#include "cusolverDn.h"
#ifdef HAVE_MAGMA
#include "magma_v2.h"
#include "magma_lapack.h" // if you need BLAS & LAPACK
#endif

using qmcplusplus::Timer;
using boost::multi::array;
using boost::multi::array_ref;
using ma::T;
using ma::product;

template<class T, class Allocator>
inline void test_gemm(Timer timer, Allocator alloc, int M, int K, int N, int nloop,
                      array<T,2> const& A_cpu,
                      array<T,2> const& B_cpu,
                      array<T,2> const& C_cpu)
{

  array<T,2,Allocator> A( {M,K}, alloc);
  array<T,2,Allocator> B( {K,N}, alloc);
  array<T,2,Allocator> C( {M,N}, alloc);
  array<T,2> C_( {M,N} );

  using std::copy;
  //using detail::copy_n;
  copy_n(A_cpu.origin(),A_cpu.num_elements(),A.origin());
  copy_n(B_cpu.origin(),B_cpu.num_elements(),B.origin());

  bool error=false;
  product(A,B,C);
  copy_n(C.origin(),C.num_elements(),C_.origin());
  for(int i = 0; i<M; i++)
    for(int j = 0; j<N; j++)
      // what is an appropriate multiplier???
      if( boost::math::relative_difference(C_cpu[i][j],C_[i][j]) > 100*std::numeric_limits<T>::epsilon()) {
        error=true;
      }
  if(error)
    std::cerr<<" --> ERROR in DGEMM. " <<std::endl;
}

template<class T, class Allocator>
inline void time_gemm(Timer timer, Allocator alloc, int M, int K, int N, int nloop, 
                      array<T,2> const& A_cpu, 
                      array<T,2> const& B_cpu) 
{

  array<T,2,Allocator> A( {M,K}, alloc); 
  array<T,2,Allocator> B( {K,N}, alloc);
  array<T,2,Allocator> C( {M,N}, alloc);

  using std::copy;
  copy_n(A_cpu.origin(),A_cpu.num_elements(),A.origin());
  copy_n(B_cpu.origin(),B_cpu.num_elements(),B.origin());

  product(A,B,C);
  if(nloop>0) {
    timer.restart();
    for(int i=0; i<nloop; i++)
      product(A,B,C);
    double t = timer.elapsed();
    std::cout<<" Time for DGEMM: " <<t/double(nloop) <<std::endl;
  }
  std::cout<<std::endl;
}

template<typename Type>
inline void test_cuda_blas_3(int M, int K, int N, int nloop, int ndev)
{
  using cuda::cublas_check;
  using cuda::cusolver_check;

  Timer timer;

  std::default_random_engine generator;
  std::uniform_real_distribution<Type> distribution(0.0,1.0);

  cublasHandle_t cublas_handle;
  cublasXtHandle_t cublasXt_handle;
  cusolverDnHandle_t cusolverDn_handle;
  cublas_check(cublasCreate (& cublas_handle ), "cublasCreate");
  cublas_check(cublasXtCreate (& cublasXt_handle ), "cublasXtCreate");
  int devID[8] {0,1,2,3,4,5,6,7};
  cublas_check(cublasXtDeviceSelect(cublasXt_handle, ndev, devID), "cublasXtDeviceSelect");
  cublas_check(cublasXtSetPinningMemMode(cublasXt_handle, CUBLASXT_PINNING_ENABLED), "cublasXtSetPinningMemMode");
  cusolver_check(cusolverDnCreate (& cusolverDn_handle ), "cusolverDnCreate");

#ifdef HAVE_MAGMA
#else
  cuda::gpu_handles handles{&cublas_handle,&cublasXt_handle,&cusolverDn_handle};
#endif

  // reference CPU 
  array<Type,2> A_cpu( {M,K} );
  array<Type,2> B_cpu( {K,N} );
  array<Type,2> C_cpu( {M,N} );

  for(int i = 0; i<M; i++)
    for(int j = 0; j<K; j++) 
      A_cpu[i][j] = distribution(generator); 

  for(int i = 0; i<K; i++)
    for(int j = 0; j<N; j++) 
      B_cpu[i][j] = distribution(generator); 


  // GEMM
  product(A_cpu,B_cpu,C_cpu);
/* // too slow
  if(nloop>0) {
    timer.restart();
    for(int i=0; i<nloop; i++)
      product(A_cpu,B_cpu,C_cpu);
    double t = timer.elapsed();
    std::cout<<" Time for DGEMM with std allocators: " <<t/double(nloop) <<std::endl;
    std::cout<<std::endl;
  }
*/
  
  {
    // Unified Memory
    using Allocator = cuda::cuda_um_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" Unified Memory " <<std::endl;
    test_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu,C_cpu);
  }

  {
    // Out of card 
    using Allocator = cuda::cuda_ooc_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" Out of Card Memory " <<std::endl;
    test_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu,C_cpu);
  }

  {
    // Unified Memory
    using Allocator = cuda::cuda_gpu_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" GPU Memory " <<std::endl;
    test_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu,C_cpu);
  }

  cublasDestroy(cublas_handle);
  cublasXtDestroy(cublasXt_handle);

}

template<typename Type>
inline void time_cuda_blas_3(int M, int K, int N, int nloop, int ndev)
{
  using cuda::cublas_check;
  using cuda::cusolver_check;

  Timer timer;

  std::default_random_engine generator;
  std::uniform_real_distribution<Type> distribution(0.0,1.0);

  cublasHandle_t cublas_handle;
  cublasXtHandle_t cublasXt_handle;
  cusolverDnHandle_t cusolverDn_handle;
  cublas_check(cublasCreate (& cublas_handle ), "cublasCreate");
  cublas_check(cublasXtCreate (& cublasXt_handle ), "cublasXtCreate");
  int devID[8] {0,1,2,3,4,5,6,7};
  cublas_check(cublasXtDeviceSelect(cublasXt_handle, ndev, devID), "cublasXtDeviceSelect");
  cublas_check(cublasXtSetPinningMemMode(cublasXt_handle, CUBLASXT_PINNING_ENABLED), "cublasXtSetPinningMemMode");
  cusolver_check(cusolverDnCreate (& cusolverDn_handle ), "cusolverDnCreate");

#ifdef HAVE_MAGMA
#else
  cuda::gpu_handles handles{&cublas_handle,&cublasXt_handle,&cusolverDn_handle};
#endif

  // reference CPU 
  array<Type,2> A_cpu( {M,K} );
  array<Type,2> B_cpu( {K,N} );

  for(int i = 0; i<M; i++)
    for(int j = 0; j<K; j++) 
      A_cpu[i][j] = distribution(generator); 

  for(int i = 0; i<K; i++)
    for(int j = 0; j<N; j++) 
      B_cpu[i][j] = distribution(generator); 

  {
    // Unified Memory
    using Allocator = cuda::cuda_um_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" Unified Memory " <<std::endl;
    time_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu);
  }

  {
    // Out of card 
    using Allocator = cuda::cuda_ooc_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" Out of Card Memory " <<std::endl;
    time_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu);
  }

  {
    // Unified Memory
    using Allocator = cuda::cuda_gpu_allocator<Type>;
    Allocator alloc(handles);

    std::cout<<" GPU Memory " <<std::endl;
    time_gemm<Type,Allocator>(timer,alloc,M,K,N,nloop,A_cpu,B_cpu);
  }

  cublasDestroy(cublas_handle);
  cublasXtDestroy(cublasXt_handle);

}


#endif
