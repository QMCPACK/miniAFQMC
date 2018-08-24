////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_MAGMA_DEFINITIONS_HPP
#define AFQMC_MAGMA_DEFINITIONS_HPP

#include "magma_v2.h"

namespace MAGMA 
{

  #include "magma_lapack.h" // if you need BLAS & LAPACK

  inline void static getrf(
	const int &n, const int &m, double *a, const int &n0, int *piv, int &st)
  {
	magma_dgetrf(n, m, a, n0, piv, st);
  }

  inline void static getrf(
	const int &n, const int &m, float *a, const int &n0, int *piv, int &st)
  {
	magma_sgetrf(n, m, a, n0, piv, st);
  }

  inline void static getrf(
	const int &n, const int &m, std::complex<double> *a, const int &n0, int *piv, int &st)
  {
	magma_zgetrf(n, m, a, n0, piv, st);
  }

  inline void static getrf(
	const int &n, const int &m, std::complex<float> *a, const int &n0, int *piv, int &st)
  {
	magma_cgetrf(n, m, a, n0, piv, st);
  }

  inline void static getrf_gpu(
        const int &n, const int &m, double *a, const int &n0, int *piv, int &st)
  {
        magma_dgetrf_gpu(n, m, a, n0, piv, st);
  }

  inline void static getrf_gpu(
        const int &n, const int &m, float *a, const int &n0, int *piv, int &st)
  {
        magma_sgetrf_gpu(n, m, a, n0, piv, st);
  }

  inline void static getrf_gpu(
        const int &n, const int &m, std::complex<double> *a, const int &n0, int *piv, int &st)
  {
        magma_zgetrf_gpu(n, m, a, n0, piv, st);
  }

  inline void static getrf_gpu(
        const int &n, const int &m, std::complex<float> *a, const int &n0, int *piv, int &st)
  {
        magma_cgetrf_gpu(n, m, a, n0, piv, st);
  }

  inline void static getri_gpu(int n, float* restrict a, int n0, int const* restrict piv, float* restrict work, int& n1, int& status)
  {
        if(n1==-1) {
          n1 = magma_get_sgetri_nb(n)*n;
          return;
        }
        sgetri_gpu(n, a, n0, piv, work, n1, status);
  }

  inline void static getri_gpu(int n, double* restrict a, int n0, int const* restrict piv, double* restrict work, int& n1, int& status)
  {
        if(n1==-1) {
          n1 = magma_get_dgetri_nb(n)*n;
          return;
        }
        dgetri_gpu(n, a, n0, piv, work, n1, status);
  }

  inline void static getri_gpu(int n, std::complex<float>* restrict a, int n0, int const* restrict piv, std::complex<float>* restrict work, int& n1, int& status)
  {
        if(n1==-1) {
          n1 = magma_get_cgetri_nb(n)*n;
          return;
        }
        cgetri_gpu(n, a, n0, piv, work, n1, status);
  }

  inline void static getri_gpu(int n, std::complex<double>* restrict a, int n0, int const* restrict piv, std::complex<double>* restrict work, int& n1, int& status)
  {
        if(n1==-1) {
          n1 = magma_get_zgetri_nb(n)*n;
          return;
        }
        zgetri_gpu(n, a, n0, piv, work, n1, status);
  }

/*
  inline void static geqrf(int M, int N, std::complex<double> *A, const int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int& INFO)
  {
	zgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
  }

  inline void static geqrf(int M, int N, double *A, const int LDA, double *TAU, double *WORK, int LWORK, int& INFO)
  {
	dgeqrf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static geqrf(int M, int N, std::complex<float> *A, const int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int& INFO)
  {
	cgeqrf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static geqrf(int M, int N, float *A, const int LDA, float *TAU, float *WORK, int LWORK, int& INFO)
  {
	sgeqrf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gelqf(int M, int N, std::complex<double> *A, const int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int& INFO)
  {
	zgelqf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gelqf(int M, int N, double *A, const int LDA, double *TAU, double *WORK, int LWORK, int& INFO)
  {
	dgelqf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gelqf(int M, int N, std::complex<float> *A, const int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int& INFO)
  {
	cgelqf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gelqf(int M, int N, float *A, const int LDA,  float *TAU, float *WORK, int LWORK, int& INFO)
  {
	sgelqf(M,N,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gqr(int M, int N, int K, std::complex<double> *A, const int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int& INFO)
  {
	zungqr(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gqr(int M, int N, int K, double *A, const int LDA, double *TAU, double *WORK, int LWORK, int& INFO)
  {
	dorgqr(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gqr(int M, int N, int K, std::complex<float> *A, const int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int& INFO)
  {
	cungqr(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static gqr(int M, int N, int K, float *A, const int LDA, float *TAU, float *WORK, int LWORK, int& INFO)
  {
	sorgqr(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static glq(int M, int N, int K, std::complex<double> *A, const int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int& INFO)
  {
	zunglq(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static glq(int M, int N, int K, double *A, const int LDA, double *TAU, double *WORK, int LWORK, int& INFO)
  {
	dorglq(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }

  inline void static glq(int M, int N, int K, std::complex<float> *A, const int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int& INFO)
  {
	cunglq(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }
    
  inline void static glq(int M, int N, int K, float *A, const int LDA, float *TAU, float *WORK, int const LWORK, int& INFO) 
  {
	sorglq(M,N,K,A,LDA,TAU,WORK,LWORK,INFO);
  }
*/
}

#endif // OHMMS_BLAS_H
