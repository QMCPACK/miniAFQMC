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
#include<vector>
#include "Numerics/detail/cuda_pointers.hpp"
#include "Numerics/detail/cublas_wrapper.hpp"
#include "Numerics/detail/cublasXt_wrapper.hpp"
// hand coded kernels for blas extensions
#include "Kernels/adotpby.cuh"
#include "Kernels/axty.cuh"
#include "Kernels/adiagApy.cuh"

// Currently available:
// Lvl-1: dot, axpy, scal
// Lvl-2: gemv
// Lvl-3: gemm

namespace BLAS_GPU
{
  // scal Specializations
  template<class T,
           class ptr,
           typename = typename std::enable_if_t< (ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE) > 
          >
  inline static void scal(int n, T alpha, ptr x, int incx)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_scal(*x.handles.cublas_handle,n,alpha,to_address(x),incx))
      throw std::runtime_error("Error: cublas_scal returned error code.");
  }

  template<class T,
           class ptr,
           typename = typename std::enable_if_t< (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE) >,
           typename = void 
          >
  inline static void scal(int n, T alpha, ptr x, int incx)
  {
    using BLAS_CPU::scal;
    return scal(n,alpha,to_address(x),incx);
  }

  // dot Specializations
  template<class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and 
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static auto dot(int const n, ptrA const& x, int const incx, ptrB const& y, int const incy)
  {
    return cublas::cublas_dot(*x.handles.cublas_handle,n,to_address(x),incx,to_address(y),incy);
  }

  template<class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static auto dot(int const n, ptrA const& x, int const incx, ptrB const& y, int const incy)
  {
    using BLAS_CPU::dot;
    return dot(n,to_address(x),incx,to_address(y),incy);  
  }

  // axpy Specializations
  template<typename T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void axpy(int n, T const a,
                          ptrA const& x, int incx, 
                          ptrB && y, int incy)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_axpy(*x.handles.cublas_handle,n,a,
                                                    to_address(x),incx,to_address(y),incy))
      throw std::runtime_error("Error: cublas_axpy returned error code.");
  }

  template<typename T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE)
                                               >,
           typename = void
          >
  inline static void axpy(int n, T const a,
                          ptrA const& x, int incx,
                          ptrB && y, int incy)
  {
    using BLAS_CPU::axpy;
    axpy(n,a,to_address(x),incx,to_address(y),incy);
  }

  // GEMV Specializations
  template<typename T, 
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and 
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrC::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void gemv(char Atrans, int M, int N,
                          T alpha,
                          ptrA const& A, int lda,
                          ptrB const& x, int incx,
                          T beta,
                          ptrC && y, int incy)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_gemv(*A.handles.cublas_handle,Atrans,
                                            M,N,alpha,to_address(A),lda,to_address(x),incx,
                                            beta,to_address(y),incy)) 
      throw std::runtime_error("Error: cublas_gemv returned error code.");
  }

  template<typename T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or 
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrC::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static void gemv(char Atrans, int M, int N, 
                          T alpha,
                          ptrA const& A, int lda,
                          ptrB const& x, int incx,
                          T beta,
                          ptrC && y, int incy)
  {
#ifdef HAVE_MAGMA
#else
    using BLAS_CPU::gemv;
    gemv(Atrans,M,N,alpha,to_address(A),lda,to_address(x),incx,beta,to_address(y),incy);
#endif    
/*
    const char Btrans('N');
    const int one(1);
    if(CUBLAS_STATUS_SUCCESS != cublas::cublasXt_gemm(*A.handles.cublasXt_handle,Atrans,Btrans,
                                            M,one,K,alpha,to_address(A),lda,to_address(x),incx,
                                            beta,to_address(y),incy))
      throw std::runtime_error("Error: cublasXt_gemv (gemm) returned error code.");
*/
  }

  // GEMM Specializations
  template<typename T, 
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and 
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrC::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          T alpha,
                          ptrA const& A, int lda,
                          ptrB const& B, int ldb,
                          T beta,
                          ptrC && C, int ldc)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_gemm(*A.handles.cublas_handle,Atrans,Btrans,
                                            M,N,K,alpha,to_address(A),lda,to_address(B),ldb,beta,to_address(C),ldc)) 
      throw std::runtime_error("Error: cublas_gemm returned error code.");
  }

  template<typename T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or 
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrC::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          T alpha,
                          ptrA const& A, int lda,
                          ptrB const& B, int ldb,
                          T beta,
                          ptrC && C, int ldc)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublasXt_gemm(*A.handles.cublasXt_handle,Atrans,Btrans,
                                            M,N,K,alpha,to_address(A),lda,to_address(B),ldb,beta,to_address(C),ldc))
      throw std::runtime_error("Error: cublasXt_gemm returned error code.");
  }

  // Blas Extensions
  // geam  
  template<class T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and 
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrC::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void geam(char Atrans, char Btrans, int M, int N,
                         T const alpha,
                         ptrA const& A, int lda,
                         T const beta,
                         ptrB const& B, int ldb,
                         ptrC C, int ldc)
  {
    if(CUBLAS_STATUS_SUCCESS != cublas::cublas_geam(*A.handles.cublas_handle,Atrans,Btrans,M,N,alpha,to_address(A),lda,
                                                    beta,to_address(B),ldb,to_address(C),ldc))
      throw std::runtime_error("Error: cublas_geam returned error code.");
  }

  template<class T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or 
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrC::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static void geam(char Atrans, char Btrans, int M, int N,
                         T const alpha,
                         ptrA const& A, int lda,
                         T const beta,
                         ptrB const& B, int ldb,
                         ptrC C, int ldc)
  {
    using BLAS_CPU::geam;
    return geam(Atrans,Btrans,M,N,alpha,to_address(A),lda,beta,to_address(B),ldb,to_address(C),ldc); 
  }

  //template<class T,
  template<
           class ptr,
           typename = typename std::enable_if_t<not (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE) >,
           typename = void
          >          
  //inline static void set1D(int n, T const alpha, ptr x, int incx)
  inline static void set1D(int n, typename ptr::value_type const alpha, ptr x, int incx)
  {
    // No set funcion in cuda!!! Avoiding kernels for now
    std::vector<typename ptr::value_type> buff(n,alpha); 
    if(CUBLAS_STATUS_SUCCESS != cublasSetVector(n,sizeof(typename ptr::value_type),buff.data(),1,to_address(x),incx)) 
      throw std::runtime_error("Error: cublasSetVector returned error code.");
  }

  template<class T,
           class ptr,
           typename = typename std::enable_if_t< (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE) >
          >
  inline static void set1D(int n, T const alpha, ptr x, int incx)
  {
    auto y = to_address(x);
    for(int i=0; i<n; i++, y+=incx)
      *y = alpha; 
  }

  // dot extension 
  template<class T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and 
                                                 (ptrC::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void adotpby(int const n, T const alpha, ptrA const& x, int const incx, ptrB const& y, int const incy, T const beta, ptrC result)
  {
    kernels::adotpby(n,alpha,to_address(x),incx,to_address(y),incy,beta,to_address(result));
  }

  template<class T,
           class ptrA,
           class ptrB,
           class ptrC,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or 
                                                 (ptrC::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static void adotpby(int const n, T const alpha, ptrA const& x, int const incx, ptrB const& y, int const incy, T const beta, ptrC result)
  {
    using BLAS_CPU::adotpby;
    adotpby(n,alpha,to_address(x),incx,to_address(y),incy,beta,to_address(result));
  }


  // axty
  template<class T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE) 
                                               >
          >
  inline static void axty(int n,
                         T const alpha,
                         ptrA const x, int incx,
                         ptrB y, int incy)
  {
    if(incx != 1 || incy != 1)
      throw std::runtime_error("Error: axty with inc != 1 not implemented.");
    kernels::axty(n,alpha,to_address(x),to_address(y));
  }

  template<class T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE) 
                                               >,
           typename = void
          >
  inline static void axty(int n,
                         T const alpha,
                         ptrA const x, int incx,
                         ptrB y, int incy)
  {
    using BLAS_CPU::axty;
    axty(n,alpha,to_address(x),incx,to_address(y),incy);
  }

  // adiagApy
  template<class T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type != CPU_OUTOFCARS_POINTER_TYPE) and
                                                 (ptrB::memory_type != CPU_OUTOFCARS_POINTER_TYPE)
                                               >
          >
  inline static void adiagApy(int n,
                         T const alpha,
                         ptrA const A, int lda,
                         ptrB y, int incy)
  {
    kernels::adiagApy(n,alpha,to_address(A),lda,to_address(y),incy);
  }

  template<class T,
           class ptrA,
           class ptrB,
           typename = typename std::enable_if_t< (ptrA::memory_type == CPU_OUTOFCARS_POINTER_TYPE) or
                                                 (ptrB::memory_type == CPU_OUTOFCARS_POINTER_TYPE)
                                               >,
           typename = void
          >
  inline static void adiagApy(int n,
                         T const alpha,
                         ptrA const A, int lda,
                         ptrB y, int incy)
  {
    using BLAS_CPU::adiagApy;
    adiagApy(n,alpha,to_address(A),lda,to_address(y),incy);
  }

  template<class ptr,
           typename = typename std::enable_if_t<(ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE)> 
          >
  inline static auto sum(int n, ptr const x, int incx) 
  {
    kernels::sum(n,to_address(x),incx);
  }

  template<class ptr,
           typename = typename std::enable_if_t<(ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE)>
          >
  inline static auto sum(int m, int n, ptr const A, int lda)
  {
    kernels::sum(m,n,to_address(A),lda);
  }

  template<class ptr,
           typename = typename std::enable_if_t<(ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE)>
          >
  inline static auto sum(int n, ptr const x, int incx)
  {
    using BLAS_CPU::sum;
    sum(n,to_address(x),incx);
  }

  template<class ptr,
           typename = typename std::enable_if_t<(ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE)>
          >
  inline static auto sum(int m, int n, ptr const A, int lda)
  {
    using BLAS_CPU::sum;
    sum(m,n,to_address(A),lda);
  }

}

#endif
