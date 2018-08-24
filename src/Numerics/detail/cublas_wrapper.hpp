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

namespace cublas {

  using cuda::cublasOperation;

  // Level-1
  inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const float alpha, float *x, int incx)
  {
    cublasStatus_t sucess =
                cublasSscal(handle,n,&alpha,x,incx);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const double alpha, double *x, int incx)
  {
    cublasStatus_t sucess =
                cublasDscal(handle,n,&alpha,x,incx);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const std::complex<float> alpha, std::complex<float> *x, int incx)
  {
    cublasStatus_t sucess =
                cublasCscal(handle,n,
                                reinterpret_cast<cuComplex const*>(&alpha),
                                reinterpret_cast<cuComplex *>(x),incx);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                           const std::complex<double> alpha, std::complex<double> *x, int incx)
  {
    cublasStatus_t sucess =
                cublasZscal(handle,n,
                                reinterpret_cast<cuDoubleComplex const*>(&alpha),
                                reinterpret_cast<cuDoubleComplex *>(x),incx);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline float cublas_dot(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           const float *y, int incy)
  {
    float result;
    cublasStatus_t sucess = cublasSdot(handle,n,x,incx,y,incy,&result);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess) 
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return result;
  }

  inline double cublas_dot(cublasHandle_t handle, int n,
                           const double *x, int incx,
                           const double *y, int incy)
  {
    double result;
    cublasStatus_t sucess = cublasDdot(handle,n,x,incx,y,incy,&result);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return result;
  }

  inline std::complex<float> cublas_dot(cublasHandle_t handle, int n,
                           const std::complex<float> *x, int incx,
                           const std::complex<float> *y, int incy)
  {
    std::complex<float> result;
    cublasStatus_t sucess = cublasCdotu(handle,n,
                                reinterpret_cast<cuComplex const*>(x),incx,
                                reinterpret_cast<cuComplex const*>(y),incy,
                                reinterpret_cast<cuComplex *>(&result));
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return result;
  }

  inline std::complex<double> cublas_dot(cublasHandle_t handle, int n,
                           const std::complex<double> *x, int incx,
                           const std::complex<double> *y, int incy)
  {
    std::complex<double> result;
    cublasStatus_t sucess = cublasZdotu(handle,n,
                                reinterpret_cast<cuDoubleComplex const*>(x),incx,
                                reinterpret_cast<cuDoubleComplex const*>(y),incy,
                                reinterpret_cast<cuDoubleComplex *>(&result));
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return result;
  }

  inline std::complex<double> cublas_dot(cublasHandle_t handle, int n,
                           const double *x, int incx,
                           const std::complex<double> *y, int incy)
  { 
    int incy_ = 2*incy;
    const double* y_ = reinterpret_cast<const double*>(y); 
    const double* y1_ = y_+1; 
    double resR, resI;
    cublasStatus_t sucess = cublasDdot(handle,n,x,incx,y_,incy_,&resR);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    sucess = cublasDdot(handle,n,x,incx,y1_,incy_,&resI);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return std::complex<double>{resR,resI}; 
  }

  inline std::complex<double> cublas_dot(cublasHandle_t handle, int n,
                           const std::complex<double> *x, int incx,
                           const double *y, int incy)
  { 
    int incx_ = 2*incx;
    const double* x_ = reinterpret_cast<const double*>(x);
    const double* x1_ = x_+1;
    double resR, resI;
    cublasStatus_t sucess = cublasDdot(handle,n,x_,incx_,y,incy,&resR);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    sucess = cublasDdot(handle,n,x1_,incx_,y,incy,&resI);
    cudaDeviceSynchronize ();
    if(CUBLAS_STATUS_SUCCESS != sucess)
      throw std::runtime_error("Error: cublas_dot returned error code.");
    return std::complex<double>{resR,resI};
  }

  inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, 
                           const float alpha, const float *x, int incx,
                           float *y, int incy)
  {
    cublasStatus_t sucess =
                cublasSaxpy(handle,n,&alpha,x,incx,y,incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const double alpha, const double *x, int incx,
                           double *y, int incy)
  {
    cublasStatus_t sucess =
                cublasDaxpy(handle,n,&alpha,x,incx,y,incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const std::complex<float> alpha, const std::complex<float> *x, int incx,
                           std::complex<float> *y, int incy)
  {
    cublasStatus_t sucess =
                cublasCaxpy(handle,n,
                                reinterpret_cast<cuComplex const*>(&alpha),
                                reinterpret_cast<cuComplex const*>(x),incx,
                                reinterpret_cast<cuComplex *>(y),incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const std::complex<double> alpha, const std::complex<double> *x, int incx,
                           std::complex<double> *y, int incy)
  {
    cublasStatus_t sucess =
                cublasZaxpy(handle,n,
                                reinterpret_cast<cuDoubleComplex const*>(&alpha),
                                reinterpret_cast<cuDoubleComplex const*>(x),incx,
                                reinterpret_cast<cuDoubleComplex *>(y),incy);
    cudaDeviceSynchronize ();
    return sucess;
  }

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

  // Extensions
  inline cublasStatus_t cublas_geam(cublasHandle_t handle, char Atrans, char Btrans, int M, int N,
                         float const alpha,
                         float const* A, int lda,
                         float const beta,
                         float const* B, int ldb,
                         float *C, int ldc)
  {
    cublasStatus_t sucess =
                cublasSgeam(handle,cublasOperation(Atrans),cublasOperation(Btrans),
                           M,N,&alpha,A,lda,&beta,B,ldb,C,ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_geam(cublasHandle_t handle, char Atrans, char Btrans, int M, int N,
                         double const alpha,
                         double const* A, int lda,
                         double const beta,
                         double const* B, int ldb,
                         double *C, int ldc)
  {
    cublasStatus_t sucess =
                cublasDgeam(handle,cublasOperation(Atrans),cublasOperation(Btrans),
                           M,N,&alpha,A,lda,&beta,B,ldb,C,ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_geam(cublasHandle_t handle, char Atrans, char Btrans, int M, int N,
                         std::complex<float> const alpha,
                         std::complex<float> const* A, int lda,
                         std::complex<float> const beta,
                         std::complex<float> const* B, int ldb,
                         std::complex<float> *C, int ldc)
  {
    cublasStatus_t sucess =
                cublasCgeam(handle,cublasOperation(Atrans),cublasOperation(Btrans),M,N,
                           reinterpret_cast<cuComplex const*>(&alpha),
                           reinterpret_cast<cuComplex const*>(A),lda,
                           reinterpret_cast<cuComplex const*>(&beta),
                           reinterpret_cast<cuComplex const*>(B),ldb,
                           reinterpret_cast<cuComplex *>(C),ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

  inline cublasStatus_t cublas_geam(cublasHandle_t handle, char Atrans, char Btrans, int M, int N,
                         std::complex<double> const alpha,
                         std::complex<double> const* A, int lda,
                         std::complex<double> const beta,
                         std::complex<double> const* B, int ldb,
                         std::complex<double> *C, int ldc)
  {
    cublasStatus_t sucess =
                cublasZgeam(handle,cublasOperation(Atrans),cublasOperation(Btrans),M,N,
                           reinterpret_cast<cuDoubleComplex const*>(&alpha),
                           reinterpret_cast<cuDoubleComplex const*>(A),lda,
                           reinterpret_cast<cuDoubleComplex const*>(&beta),
                           reinterpret_cast<cuDoubleComplex const*>(B),ldb,
                           reinterpret_cast<cuDoubleComplex *>(C),ldc);
    cudaDeviceSynchronize ();
    return sucess;
  }

}

#endif

