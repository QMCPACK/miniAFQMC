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
#include <iomanip>

#include <Configuration.h>
#include <Utilities/NewTimer.h>
#include <Utilities/taskgroup.hpp>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "Message/ma_communications.hpp"
#include "AFQMC/afqmc_sys_shm.hpp"
#include "AFQMC/rotate.hpp"
#include "Matrix/initialize.hpp"
#include "Matrix/partition_SpMat.hpp"
#include "Numerics/OhmmsBlas.h"

using namespace std;
using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
};

void print_help()
{
  printf("benchmark AFQMC computation kernels \n");
  printf("\n");
  printf("Options:\n");
  printf("-i                Number of repetitions (default: 5)\n");
  printf("-w                Smallest number of walkers (default: 1)\n");
  printf("-p                Number of powers of 2 to attempt (default: 4)\n");
  printf("-r                Number of reader cores in a node (default all)"); 
  printf("-c                Number of cores in a task group (default: all cores)"); 
  printf("-n                Number of nodes in a task group (default: all nodes)\n");
  printf("-f                Input file name (default: ./afqmc.h5)\n");
  printf("-t                If set to no, do not use half-rotated transposed Cholesky matrix to calculate bias potential (default yes).\n");
}

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)/1000000.0;
}

int main(int argc, char **argv)
{

#ifndef QMC_COMPLEX
  std::cerr<<" Error: Please compile complex executable, QMC_COMPLEX=1. " <<std::endl;
  exit(1);
#endif

  // need new mpi wrapper
  MPI_Init(&argc,&argv);

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  OhmmsInfo("benchmark",rank);

  int nsteps=10;
  int nwalk=1;
  int npower=4;
  int nread = 0;  
  int ncores_per_TG = 0;
  int nnodes_per_TG = 0;
  const double dt = 0.01;  // 1-body propagators are assumed to be generated with a timestep = 0.01

  bool verbose = false;
  std::string init_file = "afqmc.h5";

  bool transposed_Spvn = true;

  bool runSpvnT=true;
  bool runSpvn=true;
  bool runDGEMM=true;

  const ComplexType one(1.),zero(0.),half(0.5);
  const ComplexType im(0.0,1.);
  const ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hvi:w:p:c:r:n:f:")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 'w': 
      nwalk = atoi(optarg);
      break;
    case 'p': 
      npower = atoi(optarg);
      break;
    case 'r': 
      nread = atoi(optarg);
      break;
    case 'n': 
      nnodes_per_TG = atoi(optarg);
      break;
    case 'c': 
      ncores_per_TG = atoi(optarg);
      break;
    case 'f':
      init_file = std::string(optarg);
      break;
    case 'v': verbose  = true;
      break;
    }
  }

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  // small replicated structures
  ComplexMatrix haj;    // 1-Body Hamiltonian Matrix
  ComplexMatrix Propg1;   // propagator for 1-body hamiltonian 
  std::vector<int> cholVec_partition; 

  // shared memory structures
  SMSparseMatrix<ComplexType> SMSpvn;    // (Symmetric) Factorized Hamiltonian, e.g. <ij|kl> = sum_n Spvn(ik,n) * Spvn(jl,n)
  SMSparseMatrix<ComplexType> SMSpvnT;   // (Symmetric) Half-rotated Factorized Hamiltonian, e.g. SpvnT(n,ak) = sum_i A*(i,a) * Spvn(ik,n) 
  SMSparseMatrix<ComplexType> SMVakbl;   // 2-Body Hamiltonian Matrix: (Half-Rotated) 2-electron integrals 

  // setup task group
  TaskGroup TG("TGunique");
  TG.setup(ncores_per_TG,nnodes_per_TG,verbose);
  std::string str0(std::to_string(TG.getTGNumber()));

  // setup comm buffer. Right now a hack to access mutex over local TG cores 
  SMDenseVector<ComplexType> TGcommBuff(std::string("SM_TGcommBuff")+str0,TG.getTGCommLocal(),10);
  TG.setBuffer(&TGcommBuff);

  shm::afqmc_sys AFQMCSys(TG,nwalk); // Main AFQMC object. Controls access to several algorithmic functions. 

  if(nnodes_per_TG > 1) {
    std::cerr<<" Error: benchmark currently limited to nnodes_per_TG=1 \n";
    MPI_Abort(MPI_COMM_WORLD,9);
  }

  nnodes_per_TG = TG.getNNodesPerTG();
  ncores_per_TG = TG.getNCoresPerTG();
  int nnodes = TG.getTotalNodes();      // Total number of nodes
  int nodeid = TG.getNodeID();          // global id of local node 
  int node_number = TG.getLocalNodeNumber();  // node number in TG 
  int ncores = TG.getTotalCores();      // Number of cores in local node
  int coreid = TG.getCoreID();          // global core id (core # within full node)
  int core_rank = TG.getCoreRank();     // core id within local TG (core # within cores in TG local to the node)

  hdf_archive dump;
  if(nread == 0 || TG.getCoreID() < nread) {
    if(!dump.open(init_file,H5F_ACC_RDONLY))
      APP_ABORT("Error: problems opening hdf5 file. \n");
  }

  app_log()<<"***********************************************************\n";
  app_log()<<"                 Initializing from HDF5                    \n"; 
  app_log()<<"***********************************************************" <<std::endl;

  if(!Initialize(dump,dt,TG,AFQMCSys,Propg1,SMSpvn,haj,SMVakbl,cholVec_partition,nread,true)) {
    std::cerr<<" Error initalizing data structures from hdf5 file: " <<init_file <<std::endl;
    MPI_Abort(MPI_COMM_WORLD,10);
  }

  if(transposed_Spvn) shm::halfrotate_cholesky(TG,
                                               AFQMCSys.trialwfn_alpha,
                                               AFQMCSys.trialwfn_beta,
                                               SMSpvn,
                                               SMSpvnT
                                              );

  // useful info, reduce over 1 task group only
  long global_Spvn_size=0, global_Vakbl_size=0;
  int global_nchol=0;
  if(TG.getTGNumber()==0) {
    MPI_Barrier(TG.getTGComm());
    long sz = SMSpvn.size();
    if(coreid != 0) sz = 0;
    MPI_Reduce(&sz,&global_Spvn_size,1,MPI_LONG,MPI_SUM,0,TG.getTGComm());
    int isz = SMSpvn.cols();
    if(coreid != 0) isz = 0;
    MPI_Reduce(&isz,&global_nchol,1,MPI_INT,MPI_SUM,0,TG.getTGComm());
  }
  MPI_Bcast(&global_nchol,1,MPI_INT,0,MPI_COMM_WORLD);

  RealType Eshift = 0;
  int NMO = AFQMCSys.NMO;              // number of molecular orbitals
  int NAEA = AFQMCSys.NAEA;            // number of up electrons
  int NIK = 2*NMO*NMO;                 // dimensions of linearized green function
  int NIK_0, NIK_N;                    // fair partitioning of NIK over ncores
  int NAK = 2*NAEA*NMO;                // dimensions of linearized "compacted" green function
  int NAK_0, NAK_N;                    // fair partitioning of NAK over ncores
  int MM_0,  MM_N;                     // fair partitioning of NMO*NMO over ncores
  int cvec0 = cholVec_partition[TG.getLocalNodeNumber()]; 
  int cvecN = cholVec_partition[TG.getLocalNodeNumber()+1]; 
  int nchol = SMSpvn.cols();           // local number of cholesky vectors
  int NX_0, NX_N;                      // fair partitioning of nchol over ncores 
  assert(nchol == cvecN-cvec0);   
  if(transposed_Spvn)
    NIK = NAK;  

  std::tie(NIK_0, NIK_N) = FairDivideBoundary(core_rank,NIK,ncores_per_TG); 
  std::tie(NAK_0, NAK_N) = FairDivideBoundary(core_rank,NAK,ncores_per_TG); 
  std::tie(NX_0, NX_N) = FairDivideBoundary(core_rank,nchol,ncores_per_TG); 
  std::tie(MM_0, MM_N) = FairDivideBoundary(core_rank,NMO*NMO,ncores_per_TG); 

  // partition local Spvn and Vakbl among cores in TG, generate lightweight SparseMatrix_ref
  SparseMatrix_ref<ComplexType> Spvn;    
  SparseMatrix_ref<ComplexType> Spvn_for_vbias;  

  if(!shm::balance_partition_SpMat(TG,byRows,SMSpvn,Spvn)) {
    std::cerr<<" Error partitioning Spvn matrix. " <<std::endl; 
    MPI_Abort(MPI_COMM_WORLD,30);
  }
  if(transposed_Spvn) {
    if(!shm::balance_partition_SpMat(TG,byRows,SMSpvnT,Spvn_for_vbias)) {
      std::cerr<<" Error partitioning SpvnT matrix into Spvn_for_vbias. " <<std::endl; 
      MPI_Abort(MPI_COMM_WORLD,30);
    }
  } else {
    if(!shm::balance_partition_SpMat(TG,byCols,SMSpvn,Spvn_for_vbias)) {
      std::cerr<<" Error partitioning Spvn matrix into Spvn_for_vbias. " <<std::endl;
      MPI_Abort(MPI_COMM_WORLD,30);
    }
  }

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                         Summary                           \n";   
  app_log()<<"***********************************************************\n";
   
  app_log()<<"\n";
  AFQMCSys.print(app_log());
  app_log()<<"\n";
  app_log()<<"  Execution details: \n"
           <<"    nsteps: " <<nsteps <<"\n"
           <<"    nwalk_0: " <<nwalk <<"\n"
           <<"    npowers: " <<npower <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<global_nchol <<"\n"
           <<"    transposed Spvn: " <<transposed_Spvn <<"\n"
           <<"    Chol. Matrix Sparsity: " <<double(global_Spvn_size)/double(global_nchol*NMO*NMO) <<"\n"
           <<"    # nodes per TG: " <<TG.getNNodesPerTG() <<"\n" 
           <<"    # cores per TG: " <<TG.getNCoresPerTG() <<"\n" 
           <<"    # reading cores: " <<((nread==0)?ncores:nread)
           <<std::endl;

  int nwalk_tot = nwalk*std::pow(2,npower-1);

  // setup SM memory containers
  // bias potential
  SMDenseVector<ComplexType> SM_vbias(std::string("SM_vbias")+str0,TG.getTGCommLocal(),nchol*nwalk_tot);
  // Hubbard-Stratonovich potential
  SMDenseVector<ComplexType> SM_vHS(std::string("SM_vHS")+str0,TG.getTGCommLocal(),NMO*NMO*nwalk_tot);
  // density matrix
  SMDenseVector<ComplexType> SM_G_for_vbias(std::string("SM_G_for_vbias")+str0,TG.getTGCommLocal(),NIK*nwalk_tot);
  // X(n,nw) = rand(n,nw) ( + vbias(n,nw))
  SMDenseVector<ComplexType> SM_X(std::string("SM_X")+str0,TG.getTGCommLocal(),nchol*nwalk_tot);     
  // initialize overlaps and energy
  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                        Beginning                           \n";   
  app_log()<<"***********************************************************\n\n";

  if(runDGEMM) {

    app_log()<<"  Testing DGEMM: \n ntasks     time    \n";

    SMDenseVector<ComplexType> SM_M1(std::string("SM_M1")+str0,TG.getTGCommLocal(),NMO*NMO);
    SMDenseVector<ComplexType> SM_M2(std::string("SM_M2")+str0,TG.getTGCommLocal(),NMO*NMO);
    SMDenseVector<ComplexType> SM_M3(std::string("SM_M3")+str0,TG.getTGCommLocal(),NMO*NMO);

    boost::multi_array_ref<ComplexType,2> M1(SM_M1.data(), extents[NMO][NMO]);
    boost::multi_array_ref<ComplexType,2> M2(SM_M2.data(), extents[NMO][NMO]);
    boost::multi_array_ref<ComplexType,2> M3(SM_M3.data(), extents[NMO][NMO]);
    
    for(int i=1; i<=ncores_per_TG; i++) {

      if(core_rank < i) {     
        std::tie(MM_0, MM_N) = FairDivideBoundary(core_rank,NMO,i);

        ma::product(M1[indices[range_t(MM_0,MM_N)][range_t()]],
                    M2,
                    M3[indices[range_t(MM_0,MM_N)][range_t()]]);
      }
      double t0=getTime();  
      for(int k=0; k<nsteps; k++) {
        if(core_rank < i) {
          ma::product(M1[indices[range_t(MM_0,MM_N)][range_t()]],
                    M2,
                    M3[indices[range_t(MM_0,MM_N)][range_t()]]);
        }
        TG.local_barrier();
      }
      double t1=getTime();
      app_log()<<i <<" " <<(t1-t0)/nsteps <<std::endl;

    }

  }

  app_log()<<std::endl <<"  Testing Sparse GEMM: \n nw   SpvnT   Spvn " <<std::endl;

  for(int np = 0; np < npower; np++) {

    int nw = nwalk*std::pow(2,np);  

    // setup light references to SM
    boost::multi_array_ref<ComplexType,2> vbias(SM_vbias.data(),extents[nchol][nw]);     
    boost::multi_array_ref<ComplexType,2> vHS(SM_vHS.data(), extents[NMO*NMO][nw]);     
    boost::multi_array_ref<ComplexType,2> G_for_vbias(SM_G_for_vbias.data(), extents[NIK][nw]);
    boost::multi_array_ref<ComplexType,2> X(SM_X.data(), extents[nchol][nw]);       
  
    double t0=0,t1=0,t2=0,t3=0;

    // 1. SpvnT * G_for_vbias = vbias
    if(runSpvnT)
    {
      ma::product(Spvn_for_vbias,G_for_vbias,vbias);
      TG.local_barrier(); 
      t0=getTime();  
      for(int i=0; i<nsteps; i++) { 
        ma::product(Spvn_for_vbias,G_for_vbias,vbias);
        TG.local_barrier(); 
      }  
      t1=getTime();  
    }

    if(runSpvnT)
    {
      ma::product(Spvn,X,vHS);
      TG.local_barrier();
      t2=getTime();
      for(int i=0; i<nsteps; i++) {
        ma::product(Spvn,X,vHS);
        TG.local_barrier();
      }
      t3=getTime();
    }

    app_log()<<nw  <<" "  <<(t1-t0)/nsteps <<" " <<(t3-t2)/nsteps <<std::endl;

/*

        shm::get_vHS(Spvn,X,vHS);      
        TG.local_barrier();  

        AFQMCSys.propagate(W,Propg1,vHS);
*/

  }    

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                   Finished Benchmark                    \n";   
  app_log()<<"***********************************************************\n\n";
  
  MPI_Finalize();

  return 0;
}
