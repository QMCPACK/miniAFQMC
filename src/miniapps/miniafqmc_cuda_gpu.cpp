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

#include <Configuration.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "AFQMC/afqmc_sys.hpp"
#include "Matrix/initialize_serial.hpp"
#include "Matrix/peek.hpp"
#include "AFQMC/KP3IndexFactorization.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cublasXt.h"
#include "cusolverDn.h"
#include "curand.h"
#include "cuda_profiler_api.h"

#include "Numerics/detail/cuda_pointers.hpp"
#include "Kernels/zero_complex_part.cuh"

using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
  Timer_eloc,
  Timer_DM,
  Timer_Propg,
  Timer_ovlp,
  Timer_vHS,
  Timer_vbias,
  Timer_X,
  Timer_ortho
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
    {Timer_eloc, "Energy"},
    {Timer_DM, "G_for_vbias"},
    {Timer_Propg, "Propagate"},
    {Timer_ovlp, "PseudoEnergy"},
    {Timer_vHS, "vHS"},
    {Timer_vbias, "vbias"},
    {Timer_X, "X"},
    {Timer_ortho, "Ortho"}
};

void print_help()
{
  printf("miniafqmc - QMCPACK AFQMC miniapp\n");
  printf("\n");
  printf("Options:\n");
  printf("-i                Number of MC steps (default: 10)\n");
  printf("-s                Number of substeps (default: 10)\n");
  printf("-w                Number of walkers (default: 16)\n");
  printf("-o                Number of substeps between orthogonalization (default: 10)\n");
  printf("-f                Input file name (default: ./afqmc.h5)\n"); 
  printf("-t                If set to no, do not use half-rotated transposed Cholesky matrix to calculate bias potential (default yes).\n"); 
  printf("-v                Verbose output\n");
}

int main(int argc, char **argv)
{

#ifndef QMC_COMPLEX
  std::cerr<<" Error: Please compile complex executable, QMC_COMPLEX=1. " <<std::endl;
  exit(1);
#endif

  OhmmsInfo("miniafqmc_cuda_gpu",0);

  int nsteps=10;
  int nsubsteps=10; 
  int nwalk=16;
  int northo = 10;
  int ndev=1;
  const double dt = 0.005;  
  const double sqrtdt = std::sqrt(dt);  

  bool verbose = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  ComplexType one(1.),zero(0.),half(0.5);
  ComplexType cone(1.),czero(0.);
  ComplexType im(0.0,1.);
  ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "thvi:s:w:o:f:d:")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 's': // the number of sub steps for drift/diffusion
      nsubsteps = atoi(optarg);
      break;
    case 'w': // the number of sub steps for drift/diffusion
      nwalk = atoi(optarg);
      break;
    case 'o': // the number of sub steps for drift/diffusion
      northo = atoi(optarg);
      break;
    case 'd': // the number of sub steps for drift/diffusion
      ndev = atoi(optarg);
      break;
    case 'f':
      init_file = std::string(optarg);
      break;    
    case 'v': verbose  = true; 
      break;
    }
  }

  // using Unified Memory allocator
  using Alloc = cuda::cuda_gpu_allocator<ComplexType>;
  using HamOps = afqmc::KP3IndexFactorization<Alloc,Alloc>;
  using cuda::cublas_check;
  using cuda::curand_check;
  using cuda::cusolver_check;

  cuda::cuda_check(cudaSetDevice(0),"cudaSetDevice()");

  cublasHandle_t cublas_handle;
  cublasXtHandle_t cublasXt_handle;
  cusolverDnHandle_t cusolverDn_handle;
  cublas_check(cublasCreate (& cublas_handle ), "cublasCreate");
  cublas_check(cublasXtCreate (& cublasXt_handle ), "cublasXtCreate");
  int devID[8] {0,1,2,3,4,5,6,7};
  cublas_check(cublasXtDeviceSelect(cublasXt_handle, ndev, devID), "cublasXtDeviceSelect");
  cublas_check(cublasXtSetPinningMemMode(cublasXt_handle, CUBLASXT_PINNING_ENABLED), "cublasXtSetPinningMemMode");
  cusolver_check(cusolverDnCreate (& cusolverDn_handle ), "cusolverDnCreate");
  curandGenerator_t gen;
  curand_check(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),"curandCreateGenerator");
  curand_check(curandSetPseudoRandomGeneratorSeed(gen,1234ULL),"curandSetPseudoRandomGeneratorSeed");
  
#ifdef HAVE_MAGMA
#else
  cuda::gpu_handles handles{&cublas_handle,&cublasXt_handle,&cusolverDn_handle};
#endif

  Alloc gpu_alloc(handles);

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  hdf_archive dump;
  if(!dump.open(init_file,H5F_ACC_RDONLY)) 
    APP_ABORT("Error: problems opening hdf5 file. \n");

  std::cout<<"***********************************************************\n";
  std::cout<<"                 Initializing from HDF5                    \n"; 
  std::cout<<"***********************************************************\n";

  int NMO;
  int NAEA;
  int NAEB;
  WALKER_TYPES walker_type;

  std::tie(NMO,NAEA,NAEB,walker_type) = afqmc::peek(dump);

  // Main AFQMC object. Control access to several algorithmic functions.
  base::afqmc_sys<Alloc> AFQMCSys(NMO,NAEA,walker_type,gpu_alloc,gpu_alloc);
  ComplexMatrix<Alloc> Propg1({NMO,NMO}, gpu_alloc);

  auto HOps(afqmc::Initialize<HamOps,base::afqmc_sys<Alloc>>(dump,dt,AFQMCSys,Propg1));

  RealType Eshift = 0;
  int nchol = HOps.local_number_of_cholesky_vectors();            // number of cholesky vectors  
  int nspin = (walker_type==COLLINEAR)?2:1;
  int NAK = nspin*NAEA*NMO;               // dimensions of linearized "compacted" green function


  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                         Summary                           \n";   
  std::cout<<"***********************************************************\n";
   
  std::cout<<"\n";
  AFQMCSys.print(std::cout);
  std::cout<<"\n";
  std::cout<<"  Execution details: \n"
           <<"    nsteps: " <<nsteps <<"\n"
           <<"    nsubsteps: " <<nsubsteps <<"\n" 
           <<"    nwalk: " <<nwalk <<"\n"
           <<"    northo: " <<northo <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<nchol <<std::endl;

  ComplexMatrix<Alloc> vbias( {nchol,nwalk}, gpu_alloc );     // bias potential
  ComplexMatrix<Alloc> vHS( {nwalk,NMO*NMO}, gpu_alloc );        // Hubbard-Stratonovich potential
  ComplexMatrix<Alloc> Gc( {NAK,nwalk}, gpu_alloc );           // compact density matrix for energy evaluation
  ComplexMatrix<Alloc> X( {nchol,nwalk}, gpu_alloc );         // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 
  ComplexVector<Alloc> eloc(typename ComplexVector<Alloc>::extensions_type{nwalk}, gpu_alloc );         // stores local energies

  WalkerContainer<Alloc> W( {nwalk,nspin,NMO,NAEA}, gpu_alloc );
  // 
  //  0: E1, 
  //  1: EXX,
  //  2: EJ, 
  //  3: ovlp_up, 
  //  4: ovlp_down, 
  //
  ComplexMatrix<Alloc> W_data( {nwalk,3+nspin}, gpu_alloc );
  // initialize walkers to trial wave function
  {
    // conj(A) = H( T(A) ) (avoids cuda kernels)
    // trick to avoid kernels
    using ma::H;
    using ma::T;
    ComplexMatrix<Alloc> PsiT( {NAEA,NMO}, gpu_alloc );
    ma::transform(T(AFQMCSys.trialwfn_alpha),PsiT);
    for(int n=0; n<nwalk; n++)
      ma::transform(H(PsiT),W[n][0]);
    if(nspin>1) {
      ma::transform(T(AFQMCSys.trialwfn_beta),PsiT);
      for(int n=0; n<nwalk; n++)
        ma::transform(H(PsiT),W[n][1]);
    }
  }

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
  RealType Eav = HOps.energy(W_data,Gc);

  {
    size_t free_,tot_;
    cudaMemGetInfo(&free_,&tot_);
    qmcplusplus::app_log()<<"\n GPU Memory Available,  Total in MB: "
                          <<free_/1024.0/1024.0 <<" " <<tot_/1024.0/1024.0 <<std::endl;
  }

  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                     Beginning Steps                       \n";
  std::cout<<"***********************************************************\n\n";
  std::cout<<"# Initial Energy: " <<Eav <<std::endl <<std::endl;
  std::cout<<"# Step   Energy   \n";

  cudaProfilerStart();

/*
  Timers[Timer_Total]->start();
  for(int step = 0, step_tot=0; step < nsteps; step++) {

    for(int substep = 0; substep < nsubsteps; substep++, step_tot++) {

      // propagate walker forward 

      // 1. calculate density matrix and bias potential 

      Timers[Timer_DM]->start();
      AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
      Timers[Timer_DM]->stop();

      Timers[Timer_vbias]->start();
      HOps.vbias(Gc,vbias,sqrtdt);
      Timers[Timer_vbias]->stop();

      // 2. calculate X and weight
      //  X(chol,nw) = rand + i*vbias(chol,nw)
      Timers[Timer_X]->start();
      // hack to get "complex" valued real random numbers 
      curand_check(curandGenerateNormalDouble(gen,
                        reinterpret_cast<double*>(to_address(X.origin())),2*X.num_elements(),0.0,1.0),
                                          "curandGenerateNormalDouble");
      // hack hack hack!!!
      kernels::zero_complex_part(X.num_elements(),to_address(X.origin()));
      ma::axpy(im,vbias,X);
      Timers[Timer_X]->stop();

      // 3. calculate vHS
      Timers[Timer_vHS]->start();
      HOps.vHS(X,vHS,sqrtdt);
      Timers[Timer_vHS]->stop();

      // 4. propagate walker
      // W(new) = Propg1 * exp(vHS) * Propg1 * W(old)
      Timers[Timer_Propg]->start();
      AFQMCSys.propagate(W,Propg1,vHS);
      Timers[Timer_Propg]->stop();

      // 5. Orthogonalize if needed 
      if(step_tot > 0 && step_tot%northo == 0) {
        Timers[Timer_ortho]->start();
        AFQMCSys.orthogonalize(W);
        Timers[Timer_ortho]->stop();
      }

      // 6. Update overlaps 
      Timers[Timer_ovlp]->start();
      AFQMCSys.calculate_overlaps(W,W_data);
      Timers[Timer_ovlp]->stop();

    }

    // Calculate energy
    Timers[Timer_eloc]->start();
    AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
    Eav = HOps.energy(W_data,Gc);
    std::cout<<step <<"   " <<Eav <<"\n";
    Timers[Timer_eloc]->stop();

  }
  Timers[Timer_Total]->stop();
*/

  cudaProfilerStop();

  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                   Finished Calculation                    \n";
  std::cout<<"***********************************************************\n\n";

  TimerManager.print();

  return 0;
}
