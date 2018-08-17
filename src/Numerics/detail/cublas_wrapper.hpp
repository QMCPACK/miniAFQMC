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

#ifndef CUBLAS_FUNCTIONDEFS_H
#define CUBLAS_FUNCTIONDEFS_H

#include<cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "Numerics/detail/cuda_utilities.hpp"

  // Level-2
  inline cublasStatus_t cublas_gemv(cublasHandle_t handle, 
                          char Atrans, int M, int N, 
                          const float alpha,
                          const float * A, int lda,
                          const float * x, int incx,
                          const float beta,
                          float * y, int incy)
  {
    cublasStatus_t sucess =
                cublasSgemv(handle,cublasOperation(Atrans),
                           M,N,&alpha,A,lda,x,incx,&beta,y,incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                          char Atrans, int M, int N, 
                          const double alpha,
                          const double * A, int lda,
                          const double * x, int incx,
                          const double beta,
                          double * y, int incy)
  {
    cublasStatus_t sucess =
                cublasDgemv(handle,cublasOperation(Atrans),
                           M,N,&alpha,A,lda,x,incx,&beta,y,incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                          char Atrans, int M, int N, 
                          const std::complex<float> alpha,
                          const std::complex<float> * A, int lda,
                          const std::complex<float> * x, int incx,
                          const std::complex<float> beta,
                          std::complex<float> *y, int incy)
  {
    cublasStatus_t sucess =
                cublasCgemv(handle,cublasOperation(Atrans),M,N,
                           reinterpret_cast<cuComplex const*>(&alpha),reinterpret_cast<cuComplex const*>(A),lda,
                           reinterpret_cast<cuComplex const*>(x),incx,reinterpret_cast<cuComplex const*>(&beta),
                           reinterpret_cast<cuComplex *>(y),incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                          char Atrans, int M, int N, 
                          const std::complex<double> alpha,
                          const std::complex<double> * A, int lda,
                          const std::complex<double> * x, int incx,
                          const std::complex<double> beta,
                          std::complex<double> * y, int incy)
  {
    cublasStatus_t sucess =
                cublasZgemv(handle,cublasOperation(Atrans),M,N,
                           reinterpret_cast<cuDoubleComplex const*>(&alpha),
                           reinterpret_cast<cuDoubleComplex const*>(A),lda,
                           reinterpret_cast<cuDoubleComplex const*>(x),incx,
                           reinterpret_cast<cuDoubleComplex const*>(&beta),
                           reinterpret_cast<cuDoubleComplex *>(y),incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  
  // Level-3
  inline cublasStatus_t cublas_gemm(cublasHandle_t handle,
                          char Atrans, char Btrans, int M, int N, int K,
                          const float alpha,
                          const float * A, int lda,
                          const float * B, int ldb,
                          const float beta,
                          float * C, int ldc)
  {
    cublasStatus_t sucess =
                cublasSgemm(handle,
                           cublasOperation(Atrans),cublasOperation(Btrans),
                           M,N,K,&alpha,A,lda,B,ldb,&beta,C,ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemm(cublasHandle_t handle,
                          char Atrans, char Btrans, int M, int N, int K,
                          const double alpha,
                          const double * A, int lda,
                          const double * B, int ldb,
                          const double beta,
                          double * C, int ldc)
  {
    cublasStatus_t sucess =
                cublasDgemm(handle,
                           cublasOperation(Atrans),cublasOperation(Btrans),
                           M,N,K,&alpha,A,lda,B,ldb,&beta,C,ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemm(cublasHandle_t handle,
                          char Atrans, char Btrans, int M, int N, int K,
                          const std::complex<float> alpha,
                          const std::complex<float> * A, int lda,
                          const std::complex<float> * B, int ldb,
                          const std::complex<float> beta,
                          std::complex<float> * C, int ldc)
  {
    cublasStatus_t sucess =
                cublasCgemm(handle,
                           cublasOperation(Atrans),cublasOperation(Btrans),M,N,K,
                           reinterpret_cast<cuComplex const*>(&alpha),
                           reinterpret_cast<cuComplex const*>(A),lda,
                           reinterpret_cast<cuComplex const*>(B),ldb,
                           reinterpret_cast<cuComplex const*>(&beta),
                           reinterpret_cast<cuComplex *>(C),ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_gemm(cublasHandle_t handle,
                          char Atrans, char Btrans, int M, int N, int K,
                          const std::complex<double> alpha,
                          const std::complex<double> * A, int lda,
                          const std::complex<double> * B, int ldb,
                          const std::complex<double> beta,
                          std::complex<double> * C, int ldc)
  {
    cublasStatus_t sucess =
                cublasZgemm(handle,
                           cublasOperation(Atrans),cublasOperation(Btrans),M,N,K,
                           reinterpret_cast<cuDoubleComplex const*>(&alpha),
                           reinterpret_cast<cuDoubleComplex const*>(A),lda,
                           reinterpret_cast<cuDoubleComplex const*>(B),ldb,
                           reinterpret_cast<cuDoubleComplex const*>(&beta),
                           reinterpret_cast<cuDoubleComplex *>(C),ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

#endif
