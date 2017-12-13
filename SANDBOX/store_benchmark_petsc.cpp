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

#include <petscmat.h>

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

  static char help[] = "Test MatMatMult() for AIJ and Dense matrices.\n\n";
  PetscInitialize(&argc,&argv,(char*)0,help);

  // need new mpi wrapper
  //MPI_Init(&argc,&argv);

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

  Mat            A,B,C,D;

  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,SMSpvn.rows(),SMSpvn.cols());
  MatSetType(A,MATAIJ);
  MatSetFromOptions(A);
  MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);
  MatSeqAIJSetPreallocation(A,5,NULL);

  MatGetOwnershipRange(A,&Istart,&Iend);
  am   = Iend - Istart;
  bool root = Istart==0;
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
    if (i<m-1) {J = Ii + n; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
    if (j>0)   {J = Ii - 1; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
    if (j<n-1) {J = Ii + 1; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);
  }
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  MatInfo info;
  double  mal, nz_a, nz_u;

  MatGetInfo(A,MAT_LOCAL,&info);
  mal  = info.mallocs;
  nz_a = info.nz_allocated;
  int m_,n_;
  MatGetSize(A,&m_,&n_);
  std::cout<<" rank, info: " <<Istart <<" " <<nz_a <<" " <<m_ <<" " <<n_ <<std::endl;

  /* Create a dense matrix B */
  MatGetLocalSize(A,&am,&an);
  MatCreate(PETSC_COMM_WORLD,&B);
  MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,M);
  MatSetType(B,MATDENSE);
  MatSeqDenseSetPreallocation(B,NULL);
  MatMPIDenseSetPreallocation(B,NULL);
  MatSetFromOptions(B);
  MatSetRandom(B,r);
  PetscRandomDestroy(&r);
  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

  /* Test C = A*B (aij*dense) */
  PetscBarrier((PetscObject)A);
  double t1= getTime();
  MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);
  double t2= getTime();
  PetscBarrier((PetscObject)A);
  double t4= getTime();
  MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);
  double t3= getTime(); 
  if(root) std::cout<<" Time: " <<t2-t1 <<" " <<t3-t4 <<std::endl;



  MatDestroy(&A);
  MatDestroy(&C);
  MatDestroy(&B);
  
  //MPI_Finalize();
  PetscFinalize();

  return 0;
}
