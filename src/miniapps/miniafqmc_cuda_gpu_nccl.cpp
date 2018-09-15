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
#include <string>
#include <unistd.h>
#include <stdint.h>

#include <Configuration.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "mpi.h"

#include "AFQMC/afqmc_sys.hpp"
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
#include "nccl.h"

#include "Numerics/detail/cuda_pointers.hpp"
#include "Kernels/zero_complex_part.cuh"
#include "Kernels/sum.cuh"
#include "Kernels/print.cuh"

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
  comm_vhs,
  comm_vbias,
  comm_reduce,
  comm_allreduce,
  comm_bcast,
  Timer_copyn,
  Timer0
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
    {comm_vhs, "Comm_vhs"},
    {comm_vbias, "Comm_vbias"},
    {comm_reduce, "Comm_reduce"},
    {comm_allreduce, "Comm_allreduce"},
    {comm_bcast, "Comm_bcast"},
    {Timer_copyn, "GcX_copy_n"},
    {Timer0, "axpy"},
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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

template<class MatA, class MatB, class MatC, class THCOps>
double distributed_energy_v1(ncclComm_t& nccl_comm, cudaStream_t& s, int rank, int nnodes, 
                        MPI_Request &req_Gsend, MPI_Request& req_Grecv,
                        MatA& Gwork, MatB& Grecv, THCOps& THC, MatC& E);

template<class MatA, class MatB, class MatC, class THCOps>
double distributed_energy_v2(ncclComm_t& nccl_comm, cudaStream_t& s, int rank, int nnodes,
                        MatA& Gwork, MatB& Grecv, THCOps& THC, MatC& E);

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

  cuda::cuda_check(cudaSetDevice(node_rank),"cudaSetDevice()");

  char hostname[1024];
  gethostname(hostname, 1024);
  std::cout<<" rank ,hostname: " <<rank <<" " <<node_rank <<" " <<hostname <<std::endl;

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
  using THCOps = afqmc::THCOps<Alloc>;
  using CMatrix = ComplexMatrix<Alloc>;
  using CMatrix_ref = ComplexMatrix_ref<Alloc::pointer>;
  using CVector = ComplexVector<Alloc>;
  using cuda::cublas_check;
  using cuda::cuda_check;
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
  // replicated RNG string for now to keep things deterministic and independent of # of cores
  curand_check(curandSetPseudoRandomGeneratorSeed(gen,1234ULL),"curandSetPseudoRandomGeneratorSeed");

  // setup nccl
  ncclComm_t nccl_comm;
  // NOTE: using multiple streams allows you to overlap communication and computation
  // try it later
  cudaStream_t s;
  cuda_check(cudaStreamCreate(&s),"cudaStreamCreate(&s)");
  {
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, comm);
    NCCLCHECK(ncclCommInitRank(&nccl_comm, nproc, id, rank));  
  }

  cuda::gpu_handles handles{&cublas_handle,&cublasXt_handle,&cusolverDn_handle};

  Alloc um_alloc(handles);

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
  WALKER_TYPES walker_type;

  std::tie(NMO,NAEA,NAEB,walker_type) = afqmc::peek(dump);

  // Main AFQMC object. Control access to several algorithmic functions.
  base::afqmc_sys<Alloc> AFQMCSys(NMO,NAEA,walker_type,um_alloc,um_alloc);
  CMatrix Propg1({NMO,NMO}, um_alloc);

  THCOps THC(afqmc::Initialize<THCOps,base::afqmc_sys<Alloc>>(dump,dt,AFQMCSys,Propg1,comm));

  double E0 = THC.getE0();

  dump.close();

  RealType Eshift = 0;
  int nchol = THC.number_of_cholesky_vectors();            // number of cholesky vectors  
  int nspin = (walker_type==COLLINEAR)?2:1;
  int NAK = nspin*NAEA*NMO;               // dimensions of linearized "compacted" green function

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
           <<"    nwalk_per_node: " <<nwalk <<"\n"
           <<"    nnodes: " <<nproc <<"\n"
           <<"    northo: " <<northo <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<nchol <<std::endl;

  CMatrix vbias( {nchol,nwalk}, um_alloc );  // bias potential
  CMatrix vHS( {nwalk,NMO*NMO}, um_alloc );  // Hubbard-Stratonovich potential
  CMatrix Gc( {nwalk,NAK}, um_alloc ); // compact density matrix for energy evaluation 
  CMatrix randNums( {nchol,nwalk}, um_alloc ); // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 

  // work space
  CVector GcX( {nwalk*(NAK+nchol)}, um_alloc ); // combined storage for Gc and X  
  CMatrix_ref Gwork( GcX.origin(), {nwalk,NAK} ); // compact density matrix for energy evaluation 
  CMatrix_ref X( GcX.origin() + Gc.num_elements(), {nchol,nwalk} ); // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 
  CMatrix vHSwork( {nwalk,NMO*NMO}, um_alloc );  // Hubbard-Stratonovich potential

  // distributed Energy
  // doesn't seem to work with CUDA memory
  MPI_Request req_Gsend, req_Grecv;
  MPI_Send_init(to_address(Gc.origin()),Gc.num_elements()*sizeof(ComplexType),MPI_CHAR,
                  (rank==0)?(nproc-1):(rank-1),1234,comm,&req_Gsend);
  MPI_Recv_init(to_address(Gwork.origin()),Gwork.num_elements()*sizeof(ComplexType),MPI_CHAR,
                  (rank+1)%nproc,1234,comm,&req_Grecv);

  CMatrix Etot( {nwalk*nproc,3}, um_alloc );
  WalkerContainer<Alloc> W( {nwalk,nspin,NMO,NAEA}, um_alloc );
  /* 
    0: E1, 
    1: EXX,
    2: EJ, 
    3: ovlp_up, 
    4: ovlp_down, 
  */
  CMatrix W_data( {nwalk,3+nspin}, um_alloc );
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
    if(nspin>1) {
      ma::transform(T(AFQMCSys.trialwfn_beta),PsiT);
      for(int n=0; n<nwalk; n++)
        ma::transform(H(PsiT),W[n][1]);
    }
  }

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
  //RealType Eav = E0 + distributed_energy_v1(nccl_comm,s,rank,nproc,req_Gsend,req_Grecv,Gc,Gwork,THC,Etot);
  RealType Eav = E0 + distributed_energy_v2(nccl_comm,s,rank,nproc,Gc,Gwork,THC,Etot);

  if(rank==0) {
    size_t free_,tot_; 
    cudaMemGetInfo(&free_,&tot_);
    qmcplusplus::app_log()<<"\n GPU Memory Available,  Total in MB: " 
                          <<free_/1024.0/1024.0 <<" " <<tot_/1024.0/1024.0 <<std::endl;
  } 

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
        Timers[comm_bcast]->start();
        Timers[Timer_copyn]->start();
        if(np == rank) {
          cuda::copy_n(Gc.origin(),Gc.num_elements(),Gwork.origin());
          cuda::copy_n(randNums.origin(),randNums.num_elements(),X.origin());
        }
        Timers[Timer_copyn]->stop();
        NCCLCHECK(ncclBcast(to_address(GcX.origin()),2*GcX.num_elements(),ncclDouble,np,nccl_comm,s));
        cuda_check(cudaStreamSynchronize(s),"cudaStreamSynchronize(s)");
        Timers[comm_bcast]->stop();

        Timers[Timer_vbias]->start();
        THC.vbias(Gc,vbias,sqrtdt);
        Timers[Timer_vbias]->stop();
        Timers[comm_vbias]->start();
        NCCLCHECK(ncclAllReduce((const void*)to_address(vbias.origin()), 
                                (void*)to_address(vbias.origin()), 2*vbias.num_elements(), 
                                ncclDouble, ncclSum, nccl_comm, s));
        cuda_check(cudaStreamSynchronize(s),"cudaStreamSynchronize(s)");
        Timers[comm_vbias]->stop();

        // 2. calculate X and weight
        //  X(chol,nw) = rand + i*vbias(chol,nw)
        Timers[Timer_X]->start();
        ma::axpy(im,vbias,X);
        Timers[Timer_X]->stop();

        // 3. calculate vHS
        Timers[Timer_vHS]->start();
        THC.vHS(X,vHSwork,sqrtdt);
        Timers[Timer_vHS]->stop();

        // try a separate stream for this later and move sync to end of the loop
        Timers[comm_vhs]->start();
        if(rank==np)
          NCCLCHECK(ncclReduce(to_address(vHSwork.origin()), to_address(vHS.origin()), 
                               2*vHSwork.num_elements(), ncclDouble, ncclSum, np, nccl_comm, s));
        else
          NCCLCHECK(ncclReduce(to_address(vHSwork.origin()), NULL, 
                               2*vHSwork.num_elements(), ncclDouble, ncclSum, np, nccl_comm, s));
        cuda_check(cudaStreamSynchronize(s),"cudaStreamSynchronize(s)");
        Timers[comm_vhs]->stop();

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
    //Eav = E0 + distributed_energy_v1(nccl_comm,s,rank,nproc,req_Gsend,req_Grecv,Gc,Gwork,THC,Etot);
    Eav = E0 + distributed_energy_v2(nccl_comm,s,rank,nproc,Gc,Gwork,THC,Etot);
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

template<class MatA, class MatB, class MatC, class THCOps>
double distributed_energy_v1(ncclComm_t& nccl_comm, cudaStream_t& s, int rank, int nnodes, 
                        MPI_Request &req_Gsend, MPI_Request& req_Grecv,
                        MatA& Gwork, MatB& Grecv, THCOps& THC, MatC& E)
{
  MPI_Status st;
  int nwalk = Gwork.shape()[0];

  for(int k=0; k<nnodes; k++) {

    // wait for G from node behind you, copy to Gwork  
    if(k>0) {
      MPI_Wait(&req_Grecv,&st);
      MPI_Wait(&req_Gsend,&st);     // need to wait for Gsend in order to overwrite Gwork  
      cuda::copy_n(Grecv.origin(),Gwork.num_elements(),Gwork.origin());
    }

    // post send/recv messages with nodes ahead and behind you
    if(k < nnodes-1) {
      MPI_Start(&req_Gsend);
      MPI_Start(&req_Grecv);
    }

    // calculate your contribution of the local enery to the set of walkers in Gwork
    int q = (k+rank)%nnodes;
    THC.energy( E({q*nwalk,(q+1)*nwalk},{0,3}), Gwork, k==0);
  }
  if(nnodes>1)
    NCCLCHECK(ncclAllReduce(to_address(E.origin()),
                          to_address(E.origin()), E.num_elements(),
                          ncclDouble, ncclSum, nccl_comm, s));
  return real(kernels::sum(E.num_elements(),to_address(E.origin()),1))/double(nnodes*nwalk);
}

template<class MatA, class MatB, class MatC, class THCOps>
double distributed_energy_v2(ncclComm_t& nccl_comm, cudaStream_t& s, int rank, int nnodes,
                        MatA& Gc, MatB& Gwork, THCOps& THC, MatC& E)
{
  int nwalk = Gwork.shape()[0];
  for(int k=0; k<nnodes; k++) {

    if(rank==k)
      cuda::copy_n(Gc.origin(),Gc.num_elements(),Gwork.origin());
    NCCLCHECK(ncclBcast(to_address(Gwork.origin()),2*Gwork.num_elements(),ncclDouble,k,nccl_comm,s));
    cuda::cuda_check(cudaStreamSynchronize(s),"cudaStreamSynchronize(s)");

    THC.energy( E({k*nwalk,(k+1)*nwalk},{0,3}), Gwork, k==rank);
  }
  if(nnodes>1) {
    NCCLCHECK(ncclAllReduce(to_address(E.origin()),
                          to_address(E.origin()), 2*E.num_elements(),
                          ncclDouble, ncclSum, nccl_comm, s));
    cuda::cuda_check(cudaStreamSynchronize(s),"cudaStreamSynchronize(s)");
  }
  return real(kernels::sum(E.num_elements(),to_address(E.origin()),1))/double(nnodes*nwalk);
}



