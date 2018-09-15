//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
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
// -*- C++ -*-
// clang-format off
/** @file miniafqmc.cpp
    @brief Miniapp 
 
 @mainpage MiniAFQMC: miniapp for QMCPACK AFQMC kernels

 */

 /*!
 \page 
 */
// clang-format on
#include <random>

#include <Utilities/NewTimer.h>
#include <getopt.h>

#include "Numerics/tests/cuda_blas_tests.hpp"
#include "Numerics/tests/cuda_gemm_gpu.hpp"

using namespace qmcplusplus;

void print_help()
{
  printf("BLAS test - Standalone routine to time/test the cuda back-end for BLAS/LAPACK.\n");
  printf("\n");
  printf("Options:\n");
  printf("-m                Number of rows in left-hand matrix. (default: 100)\n");
  printf("-k                Number of columns in left-hand matrix. (default: 100)\n");
  printf("-n                Number of columns in right-hand matrix. (default: 100)\n");
  printf("-t                Number of iterations in timing loop (default: 4) \n"); 
  printf("-d                Number of GPUs in cudaXt calls (default: 1) \n"); 
  printf("-b                Number of batched in BatchedGemm test (default: 0) \n"); 
  printf("-v                Verbose output\n");
}

int main(int argc, char **argv)
{
  int M=100;
  int K=100;
  int N=100;
  int nloop=4;
  int ndev=1;
  int nbatch=0;

  bool verbose = false;

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "m:n:k:t:d:b:vh")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'm': // number of MC steps
      M = atoi(optarg);
      break;
    case 'k': // number of MC steps
      K = atoi(optarg);
      break;
    case 'n': // number of MC steps
      N = atoi(optarg);
      break;
    case 't': // number of MC steps
      nloop = atoi(optarg);
      break;
    case 'd': // number of MC steps
      ndev = atoi(optarg);
      break;
    case 'b': 
      nbatch = atoi(optarg);
      break;
    case 'v': verbose  = true; 
      break;
    }
  }

  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                   Begin Testing                    \n";
  std::cout<<"***********************************************************\n\n";

  if(nbatch>0)
    time_cuda_gemm_batched<cuDoubleComplex>(M,K,N,nloop,ndev,nbatch);
    //time_cuda_gemm_batched<std::complex<double>>(M,K,N,nloop,ndev,nbatch);
  else {
    time_cuda_blas_3<std::complex<double> >(M,K,N,nloop,ndev); 
//    test_cuda_blas_3<double>(M,K,N,nloop,ndev); 
  }

  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                   Finished Testing                    \n";
  std::cout<<"***********************************************************\n\n";

  return 0;
}
