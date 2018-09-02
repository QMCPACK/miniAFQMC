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

#include "mpi.h"

#include "AFQMC/afqmc_sys_batched.hpp"
#include "Matrix/initialize_parallel.hpp"
#include "Matrix/peek.hpp"
#include "AFQMC/THCOps.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublasXt.h"
#include "cusolverDn.h"
#include <curand.h>
#ifdef HAVE_MAGMA
#include "magma_v2.h"
#include "magma_lapack.h" // if you need BLAS & LAPACK
#endif

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
  Timer_ortho,
  comm_overhead
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
    {Timer_ortho, "Ortho"},
    {comm_overhead, "Comm_Overhead"}
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

extern size_t cuda::TotalGPUAlloc;

int main(int argc, char **argv)
{
  int rank, nproc;
  MPI_Comm comm(MPI_COMM_WORLD); 
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&nproc);
  bool root=(rank==0);

  int node_rank, node_size;
  MPI_Comm node;
  MPI_Comm_split_type(comm,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node);
  MPI_Comm_rank(node,&node_rank);
  MPI_Comm_size(node,&node_size);

  int num_devices=0;
  cudaGetDeviceCount(&num_devices);
  if(num_devices < node_size) {
    std::cerr<<"Error: # GPU < # tasks in node. " <<std::endl;
    MPI_Abort(comm,1);  
  }

  cuda::cuda_check(cudaSetDevice(rank),"cudaSetDevice()");

  OhmmsInfo("miniafqmc_cuda_gpu_mpi",rank);
  
#ifndef QMC_COMPLEX
  std::cerr<<" Error: Please compile complex executable, QMC_COMPLEX=1. " <<std::endl;
  exit(1);
#endif

  int nsteps=10;
  int nsubsteps=10; 
  int nwalk=16;
  int northo = 10;
  int ndev=1;
  int nbatch=32;
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
  while ((opt = getopt(argc, argv, "thvi:s:w:o:f:d:b:")) != -1)
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
    case 'b': // the number of batches 
      nbatch = atoi(optarg);
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

  nbatch = std::min(nbatch,2*nwalk);

  // using Unified Memory allocator
  using Alloc = cuda::cuda_gpu_allocator<ComplexType>;
  using THCOps = afqmc::THCOps<Alloc>;
  using CMatrix = ComplexMatrix<Alloc>;
  using CMatrix_ref = ComplexMatrix_ref<Alloc::pointer>;
  using CVector = ComplexVector<Alloc>;
  using cuda::cublas_check;
  using cuda::curand_check;
  using cuda::cusolver_check;

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

  Alloc um_alloc(handles);

  Random.init(0, 1, iseed);
  int ip = 0;
  PrimeNumberSet<uint32_t> myPrimes;
  // create generator within the thread
  RandomGenerator<RealType> random_th(myPrimes[ip]);

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  hdf_archive dump;
  if(!dump.open(init_file,H5F_ACC_RDONLY)) 
    APP_ABORT("Error: problems opening hdf5 file. \n");

  app_log()<<"***********************************************************\n";
  app_log()<<"                 Initializing from HDF5                    \n"; 
  app_log()<<"***********************************************************\n";

  int NMO;
  int NAEA;
  int NAEB;

  std::tie(NMO,NAEA,NAEB) = afqmc::peek(dump);

  // Main AFQMC object. Control access to several algorithmic functions.
  base::afqmc_sys<Alloc> AFQMCSys(NMO,NAEA,um_alloc,nbatch);
  CMatrix Propg1({NMO,NMO}, um_alloc);

  THCOps THC(afqmc::Initialize<THCOps,base::afqmc_sys<Alloc>>(dump,dt,AFQMCSys,Propg1,comm));

  RealType Eshift = 0;
  int nchol = THC.number_of_cholesky_vectors();            // number of cholesky vectors  
  int NAK = 2*NAEA*NMO;               // dimensions of linearized "compacted" green function

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                         Summary                           \n";   
  app_log()<<"***********************************************************\n";
   
  app_log()<<"\n";
  AFQMCSys.print(app_log());
  app_log()<<"\n";
  app_log()<<"  Execution details: \n"
           <<"    nsteps: " <<nsteps <<"\n"
           <<"    nsubsteps: " <<nsubsteps <<"\n" 
           <<"    nwalk: " <<nwalk <<"\n"
           <<"    northo: " <<northo <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<nchol <<std::endl;

  CMatrix vbias( {nchol,nwalk}, um_alloc );  // bias potential
  CMatrix vHS( {nwalk,NMO*NMO}, um_alloc );  // Hubbard-Stratonovich potential
  CMatrix Gc( {nwalk,NAK}, um_alloc ); // compact density matrix for energy evaluation 
  CMatrix randNums( {nchol,nwalk}, um_alloc ); // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 
  CVector eloc( {nwalk}, um_alloc );         // stores local energies

  // work space
  CVector GcX( {nwalk*(NAK+nchol)}, um_alloc ); // combined storage for Gc and X  
  CMatrix_ref Gwork( GcX.origin(), {nwalk,NAK} ); // compact density matrix for energy evaluation 
  CMatrix_ref X( GcX.origin() + Gc.num_elements(), {nchol,nwalk} ); // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 
  CMatrix vHSwork( {nwalk,NMO*NMO}, um_alloc );  // Hubbard-Stratonovich potential

  // temporary buffer space since Reduce and Allreduce don't seem to work with gpu pointers yet
  std::vector<ComplexType> buffer( std::max(nwalk*NMO*NMO,nwalk*(NAK+nchol)) );

  WalkerContainer<Alloc> W( {nwalk,2,NMO,NAEA}, um_alloc );
  /* 
    0: E1, 
    1: EXX,
    2: EJ, 
    3: ovlp_up, 
    4: ovlp_down, 
  */
  CMatrix W_data( {nwalk,5}, um_alloc );
  // initialize walkers to trial wave function
  {
    // conj(A) = H( T(A) ) (avoids cuda kernels)
    // trick to avoid kernels
    using ma::H;
    using ma::T;
    CMatrix PsiT( {NAEA,NMO}, um_alloc );
    ma::transform(T(AFQMCSys.trialwfn_alpha),PsiT);
    for(int n=0; n<nwalk; n++)
      ma::transform(H(PsiT),W[n][0]);
    ma::transform(T(AFQMCSys.trialwfn_beta),PsiT);
    for(int n=0; n<nwalk; n++)
      ma::transform(H(PsiT),W[n][1]);
  }

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
  RealType Eav_, Eav;
  Eav_ = Eav = THC.energy(W_data,Gc,root);
  MPI_Reduce(&Eav_,&Eav,1,MPI_DOUBLE,MPI_SUM,0,comm);

  app_log()<<"\n Total GPU allocation: " <<cuda::TotalGPUAlloc <<" MB " <<std::endl;

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                     Beginning Steps                       \n";
  app_log()<<"***********************************************************\n\n";
  app_log()<<"# Initial Energy: " <<Eav <<std::endl <<std::endl;
  app_log()<<"# Step   Energy   \n";

  Timers[Timer_Total]->start();
  for(int step = 0, step_tot=0; step < nsteps; step++) {

    for(int substep = 0; substep < nsubsteps; substep++, step_tot++) {

      // propagate walker forward 

      // 1. calculate density matrix and bias potential 

      Timers[Timer_DM]->start();
      AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
      Timers[Timer_DM]->stop();

      // hack to get "complex" valued real random numbers 
      curand_check(curandGenerateNormalDouble(gen,
                        reinterpret_cast<double*>(to_address(randNums.origin())),
                                                  2*randNums.num_elements(),0.0,1.0),
                                                  "curandGenerateNormalDouble");
      // hack hack hack!!!
      kernels::zero_complex_part(randNums.num_elements(),to_address(randNums.origin()));

      // loop over all nodes  
      for(int np=0; np<nproc; np++) {

        // 2. bcast G (and X) to TG 
        Timers[comm_overhead]->start();
        if(np == rank) {
          cuda::copy_n(Gc.origin(),Gc.num_elements(),Gwork.origin());
          cuda::copy_n(randNums.origin(),randNums.num_elements(),X.origin());
        }
        MPI_Bcast(to_address(GcX.origin()),GcX.num_elements(),MPI_DOUBLE_COMPLEX,np,comm); 
        Timers[comm_overhead]->stop();

        Timers[Timer_vbias]->start();
        THC.vbias(Gc,vbias,sqrtdt);
        Timers[Timer_vbias]->stop();
        Timers[comm_overhead]->start();
//        MPI_Allreduce(MPI_IN_PLACE, to_address(vbias.origin()), vbias.num_elements(), MPI_DOUBLE_COMPLEX,
//                      MPI_SUM,comm);  
        cuda::copy_n(vbias.origin(),vbias.num_elements(),buffer.data()); 
        MPI_Allreduce(MPI_IN_PLACE, buffer.data(), vbias.num_elements(), MPI_DOUBLE_COMPLEX,
                      MPI_SUM,comm);  
        cuda::copy_n(buffer.data(),vbias.num_elements(),vbias.origin()); 
        Timers[comm_overhead]->stop();

        // 2. calculate X and weight
        //  X(chol,nw) = rand + i*vbias(chol,nw)
        Timers[Timer_X]->start();
        ma::axpy(im,vbias,X);
        Timers[Timer_X]->stop();

        // 3. calculate vHS
        Timers[Timer_vHS]->start();
        THC.vHS(X,vHSwork,sqrtdt);
        Timers[Timer_vHS]->stop();

        Timers[comm_overhead]->start();
        // This could be non-blocking in principle
//        MPI_Reduce(to_address(vHSwork.origin()),to_address(vHS.origin()),vHS.num_elements(),
//                   MPI_DOUBLE_COMPLEX,MPI_SUM,np,comm);
        cuda::copy_n(vHSwork.origin(),vHSwork.num_elements(),buffer.data()); 
        if(rank==np) {
          MPI_Reduce(MPI_IN_PLACE, buffer.data(), vHSwork.num_elements(), MPI_DOUBLE_COMPLEX,
                      MPI_SUM,np,comm);  
          cuda::copy_n(buffer.data(),vHS.num_elements(),vHS.origin()); 
        } else
          MPI_Reduce(buffer.data(), NULL, vHSwork.num_elements(), MPI_DOUBLE_COMPLEX,
                      MPI_SUM,np,comm);  
        Timers[comm_overhead]->stop();

      }  

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
    Eav_ = Eav = THC.energy(W_data,Gc,root);
    MPI_Reduce(&Eav_,&Eav,1,MPI_DOUBLE,MPI_SUM,0,comm);
    app_log()<<step <<"   " <<Eav <<"\n";
    Timers[Timer_eloc]->stop();

  }
  Timers[Timer_Total]->stop();

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                   Finished Calculation                    \n";
  app_log()<<"***********************************************************\n\n";

  if(rank==0)
    TimerManager.print();

  MPI_Finalize();

  return 0;
}
