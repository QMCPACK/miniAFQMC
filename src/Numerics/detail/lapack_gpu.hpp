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

#ifndef AFQMC_LAPACK_GPU_HPP
#define AFQMC_LAPACK_GPU_HPP

#include<cassert>
#include "Utilities/type_conversion.hpp"
#include "Numerics/detail/raw_pointers.hpp"
#ifdef HAVE_MAGMA
#include "Numerics/detail/magma_wrapper.hpp"
#else
#include "Numerics/detail/lapack_cpu.hpp"
#endif
#include "Numerics/detail/cuda_pointers.hpp"
#include "Numerics/detail/cublas_wrapper.hpp"
//#include "Numerics/detail/cusolve_wrapper.hpp"

namespace LAPACK_GPU
{
  using qmcplusplus::afqmc::remove_complex;

  // hevr
  template<typename T,
           class ptr,
           class ptrR,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type != GPU_MEMORY_POINTER_TYPE) or 
                                                 (ptrR::memory_type != GPU_MEMORY_POINTER_TYPE) or 
                                                 (ptrI::memory_type != GPU_MEMORY_POINTER_TYPE) 
                                               >
          >
  inline static void hevr (char JOBZ, char RANGE, char UPLO, int N, 
                         ptr A, int LDA, T VL, T VU,int IL, int IU, T ABSTOL, int &M, 
                         ptrR W, ptr Z, int LDZ, ptrI ISUPPZ, 
                         ptr WORK, int &LWORK, 
                         ptrR RWORK, int &LRWORK, 
                         ptrI IWORK, int &LIWORK, int& INFO)
  {
#ifdef HAVE_MAGMA
    using MAGMA::hevr;
    hevr (JOBZ,RANGE,UPLO,N,to_address(A),LDA,VL,VU,IL,IU,ABSTOL,M,to_address(W),to_address(Z),LDZ,to_address(ISUPPZ),
           to_address(WORK),LWORK,to_address(RWORK),LRWORK,to_address(IWORK),LIWORK,INFO);
#else
    using LAPACK_CPU::hevr;
    hevr (JOBZ,RANGE,UPLO,N,to_address(A),LDA,VL,VU,IL,IU,ABSTOL,M,to_address(W),to_address(Z),LDZ,to_address(ISUPPZ),
           to_address(WORK),LWORK,to_address(RWORK),LRWORK,to_address(IWORK),LIWORK,INFO);
#endif
  }

  template<typename T,
           class ptr,
           class ptrR,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type == GPU_MEMORY_POINTER_TYPE) and
                                                 (ptrR::memory_type == GPU_MEMORY_POINTER_TYPE) and 
                                                 (ptrI::memory_type == GPU_MEMORY_POINTER_TYPE)
                                               >,
           typename = void 
          >
  inline static void hevr (char JOBZ, char RANGE, char UPLO, int N,    
                         ptr A, int LDA, T VL, T VU,int IL, int IU, T ABSTOL, int &M,                                                      
                         ptrR W, ptr Z, int LDZ, ptrI ISUPPZ,                  
                         ptr WORK, int &LWORK,      
                         ptrR RWORK, int &LRWORK,               
                         ptrI IWORK, int &LIWORK, int& INFO)
  {
#ifdef HAVE_MAGMA
    using MAGMA::hevr_gpu;
    hevr_gpu (JOBZ,RANGE,UPLO,N,to_address(A),LDA,VL,VU,IL,IU,ABSTOL,M,to_address(W),to_address(Z),LDZ,to_address(ISUPPZ),
           to_address(WORK),LWORK,to_address(RWORK),LRWORK,to_address(IWORK),LIWORK,INFO);
#else
    throw std::runtime_error("Error: hevr not implemented in gpu."); 
#endif
  }

  // getrf
  template<class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type != GPU_MEMORY_POINTER_TYPE) or
                                                 (ptrI::memory_type != GPU_MEMORY_POINTER_TYPE)
                                               >
          >
  inline static void getrf (const int n, const int m, ptr a, const int n0, ptrI piv, int &st) 
  {
#ifdef HAVE_MAGMA
    using MAGMA::getrf;
    getrf(n, m, to_address(a), n0, to_address(piv), st);
#else
    using LAPACK_CPU::getrf;
    getrf(n, m, to_address(a), n0, to_address(piv), st);
#endif
  }

  template<class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type == GPU_MEMORY_POINTER_TYPE) and
                                                 (ptrI::memory_type == GPU_MEMORY_POINTER_TYPE)
                                               >,
           typename = void 
          >
  inline static void getrf (const int n, const int m, ptr a, const int n0, ptrI piv, int &st) 
  {
#ifdef HAVE_MAGMA
    using MAGMA::getrf_gpu;
    getrf_gpu(n, m, to_address(a), n0, to_address(piv), st);
#else
    if(n!=m)
      throw std::runtime_error("Error: LAPACK_GPU::getrf only implemented for squared matrices.\n"); 
    std::vector<typename ptr::value_type> buff(n*n);
    std::vector<int> Ibuff(n+1);
    cudaMemcpy(buff.data(),to_address(a),sizeof(typename ptr::value_type)*n*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(Ibuff.data(),to_address(piv),sizeof(int)*(n+1),cudaMemcpyDeviceToHost);
    using LAPACK_CPU::getrf;
    getrf(n, n, buff.data(), n0, Ibuff.data(), st);
    cudaMemcpy(to_address(a),buff.data(),sizeof(typename ptr::value_type)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(to_address(piv),Ibuff.data(),sizeof(int)*(n+1),cudaMemcpyHostToDevice);
/*
    if(n!=m)
      throw std::runtime_error("Error: LAPACK_GPU::getrf only implemented for squared matrices.\n"); 
    auto a_(to_address(a));
std::cout<<" getrf n: " <<n <<std::endl;
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_getrfBatched(*a.handles.cublas_handle, n,
                                       &a_, n0, to_address(piv), to_address(piv)+n, 1))
      throw std::runtime_error("Error: cublas_getrf returned error code."); 
    cudaMemcpy(&st,to_address(piv)+n,sizeof(int),cudaMemcpyDeviceToHost);
std::cout<<" getrf status: " <<st <<std::endl;
*/
#endif
  }

  // getri: will fail if not called correctly, but removing checks on ptrI and ptrW for now
  template<class ptr,
           class ptrI,
           class ptrW,
           typename = typename std::enable_if_t< (ptr::memory_type != GPU_MEMORY_POINTER_TYPE) > 
          >
  inline static void getri(int n, ptr a, int n0, ptrI piv, ptrW work, int& n1, int& status)
  {
#ifdef HAVE_MAGMA
    using MAGMA::getri;
    using detail::to_address;
    getri(n, to_address(a), n0, to_address(piv), to_address(work), n1, status);
#else
    using LAPACK_CPU::getri;
    using detail::to_address;
    getri(n, to_address(a), n0, to_address(piv), to_address(work), n1, status);
#endif
  }

  // write separate query function to avoid hack!!!
  template<class ptr,
           class ptrI,
           class ptrW,
           typename = typename std::enable_if_t< (ptr::memory_type == GPU_MEMORY_POINTER_TYPE)>,
           typename = void
          >
  inline static void getri(int n, ptr a, int n0, ptrI piv, ptrW work, int& n1, int& status)
  {
#ifdef HAVE_MAGMA
    using MAGMA::getri_gpu;
    getri_gpu(n, to_address(a), n0, to_address(piv), to_address(work), n1, status);
#else
    if(n1==-1) {
      n1 = n*n;  
      status=0;  
      return;
    }
    if(n1 < n*n)
      throw std::runtime_error("Error: cublas_getri required buffer space of n*n."); 
    using detail::to_address;
    std::vector<typename ptr::value_type> buff(n*n);
    std::vector<typename ptr::value_type> w_(n*n);
    std::vector<int> Ibuff(n+1);
    cudaMemcpy(buff.data(),to_address(a),sizeof(typename ptr::value_type)*n*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(Ibuff.data(),to_address(piv),sizeof(int)*(n+1),cudaMemcpyDeviceToHost);
    using LAPACK_CPU::getri;
    getri(n, buff.data(), n0, Ibuff.data(), w_.data(), n1, status);
    cudaMemcpy(to_address(a),buff.data(),sizeof(typename ptr::value_type)*n*n,cudaMemcpyHostToDevice);
/**
    if(n0!=n) 
      throw std::runtime_error("Error: cublas_getri currently requires lda==n"); 
    auto a_(to_address(a));
    using detail::to_address;
    auto w_(to_address(work));
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_getriBatched(*a.handles.cublas_handle, n,
                   &a_, n0, to_address(piv), &w_, n, to_address(piv)+n, 1))
      throw std::runtime_error("Error: cublas_getri returned error code."); 
    cudaMemcpy(to_address(a),to_address(work),n*n,cudaMemcpyDeviceToDevice);
    cudaMemcpy(&status,to_address(piv)+n,sizeof(int),cudaMemcpyDeviceToHost);
*/
#endif
  }

  // geqrf
  template<class ptr,
           typename = typename std::enable_if_t< (ptr::memory_type != GPU_MEMORY_POINTER_TYPE) > 
          >
  inline static void geqrf(int M, int N, ptr A, const int LDA, ptr TAU, ptr WORK, int &LWORK, int& INFO) 
  {
#ifdef HAVE_MAGMA
    using MAGMA::geqrf;
    geqrf(M, N, to_address(A), LDA, to_address(TAU), to_address(WORK), LWORK, INFO);
#else
    using LAPACK_CPU::geqrf;
    geqrf(M, N, to_address(A), LDA, to_address(TAU), to_address(WORK), LWORK, INFO);
#endif
  }

  template<class ptr,
           typename = typename std::enable_if_t< (ptr::memory_type == GPU_MEMORY_POINTER_TYPE) >,
           typename = void
          >
  inline static void geqrf(int M, int N, ptr A, const int LDA, ptr TAU, ptr WORK, int &LWORK, int& INFO) 
  {
#ifdef HAVE_MAGMA
    using MAGMA::geqrf_gpu;
    geqrf_gpu(M, N, to_address(A), LDA, to_address(TAU), to_address(WORK), LWORK, INFO);
#else
    throw std::runtime_error("Error: geqrf not implemented in gpu.");
#endif
  }

}

#endif
