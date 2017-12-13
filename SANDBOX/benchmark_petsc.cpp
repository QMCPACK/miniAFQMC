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
#include <Utilities/taskgroup_petsc.hpp>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "Message/ma_communications.hpp"
#include "AFQMC/afqmc_sys_shm.hpp"
#include "AFQMC/rotate.hpp"
#include "Matrix/initialize_petsc.hpp"
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
  int ncores_per_TG = 1;
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
  TaskGroup TG(std::string("TGunique")+std::to_string(rank));
  TG.setup(ncores_per_TG,nnodes_per_TG,verbose);
  std::string str0(std::to_string(rank));

  // setup comm buffer. Right now a hack to access mutex over local TG cores 
  SMDenseVector<ComplexType> TGcommBuff(std::string("SM_TGcommBuff")+str0,TG.getTGCommLocal(),10);
  TG.setBuffer(&TGcommBuff);

  shm::afqmc_sys AFQMCSys(TG,nwalk); // Main AFQMC object. Controls access to several algorithmic functions. 

  if(ncores_per_TG > 1) {
    std::cerr<<" Error: benchmark currently limited to ncores_per_TG=1 \n";
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

  if(!Initialize(dump,dt,TG,AFQMCSys,Propg1,SMSpvn,haj,SMVakbl,cholVec_partition,nread)) {
    std::cerr<<" Error initalizing data structures from hdf5 file: " <<init_file <<std::endl;
    MPI_Abort(MPI_COMM_WORLD,10);
  }

  int NMO = AFQMCSys.NMO;              // number of molecular orbitals
  int NAEA = AFQMCSys.NAEA;            // number of up electrons
  int cvec0 = cholVec_partition[TG.getLocalNodeNumber()];
  int cvecN = cholVec_partition[TG.getLocalNodeNumber()+1];

  long global_Spvn_size=0, global_Vakbl_size=0;
  int global_nchol=0;
  if(TG.getTGNumber()==0) {
    MPI_Barrier(TG.getTGComm());
    long sz = SMSpvn.size();
    if(coreid != 0) sz = 0;
    MPI_Reduce(&sz,&global_Spvn_size,1,MPI_LONG,MPI_SUM,0,TG.getTGComm());
    sz = SMVakbl.size();
    if(coreid != 0) sz = 0;
    MPI_Reduce(&sz,&global_Vakbl_size,1,MPI_LONG,MPI_SUM,0,TG.getTGComm());
    int isz = SMSpvn.cols();
    if(coreid != 0) isz = 0;
    MPI_Reduce(&isz,&global_nchol,1,MPI_INT,MPI_SUM,0,TG.getTGComm());
  }
  MPI_Bcast(&global_nchol,1,MPI_INT,0,MPI_COMM_WORLD);

  Mat            A,B,C,D;
  PetscInt       Istart,Iend,am,an;
  PetscInt       Jstart,Jend;

  
  int J0,Jn;
  std::tie(J0, Jn) = FairDivideBoundary(rank,2*NAEA*NMO,nproc);
  //cout<<" rows,J0,Jn: " <<SMVakbl.rows() <<" " <<2*NMO*NAEA  <<" " <<J0 <<" " <<Jn <<std::endl;

  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetSizes(A,SMVakbl.rows(),Jn-J0,PETSC_DECIDE,2*NMO*NAEA);
  MatSetType(A,MATAIJ);
  MatSetFromOptions(A);

  // Vakbl is not sorted 
  PetscInt nd=0, no=0;
  for(int i=0; i<SMVakbl.rows(); i++) {
    PetscInt nd_=0,no_=0;
    for(int k=*SMVakbl.row_index(i); k<*SMVakbl.row_index(i+1); k++) {
      if(*SMVakbl.column_data(k) >= J0 && *SMVakbl.column_data(k) < Jn)
        nd_++;
      else
        no_++;
    }    
    nd = std::max(nd,nd_);
    no = std::max(no,no_);
  }

  //cout<<" nd,no: " <<nd <<" " <<no <<std::endl;
  MatMPIAIJSetPreallocation(A,nd,NULL,no,NULL);
  MatSeqAIJSetPreallocation(A,nd,NULL);

  PetscInt r0 = std::get<0>(SMVakbl.getOffset());
  PetscInt Ii,J;
  PetscScalar v;
  MatGetOwnershipRange(A,&Istart,&Iend);
  MatGetOwnershipRangeColumn(A,&Jstart,&Jend);
  //cout<<" Ranges: " <<Istart <<" " <<Iend <<" " <<Jstart <<" " <<Jend <<std::endl; 
  for(int i=0; i<SMVakbl.size(); i++) {
    PetscScalar v=SMVakbl.values(i)->real();
    Ii = *SMVakbl.row_data(i) + r0;   
    J = *SMVakbl.column_data(i);   
    if( Ii < Istart && Ii >= Iend ) 
      std::cout<<" Range error I: " <<Ii <<" " <<Istart <<" " <<Iend <<std::endl;
    if( J < Jstart && J >= Jend ) 
      std::cout<<" Range error J: " <<J <<" " <<Jstart <<" " <<Jend <<std::endl;
    MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);
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
  //std::cout<<" rank, info: " <<Istart <<" " <<nz_a <<" " <<m_ <<" " <<n_ <<std::endl;

  for(int np = 0; np < npower; np++) {

    int nw = nwalk*std::pow(2,np);

    /* Create a dense matrix B */
    MatGetLocalSize(A,&am,&an);
    MatCreate(PETSC_COMM_WORLD,&B);
    MatSetSizes(B,an,PETSC_DECIDE,PETSC_DECIDE,nw);
    MatSetType(B,MATDENSE);
    MatSeqDenseSetPreallocation(B,NULL);
    MatMPIDenseSetPreallocation(B,NULL);
    MatSetFromOptions(B);
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

    PetscReal      fill = 1.0;

    /* Test C = A*B (aij*dense) */
    PetscBarrier((PetscObject)A);
    MatMatMult(A,B,MAT_INITIAL_MATRIX,fill,&C);
    PetscBarrier((PetscObject)A);
    MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);
    PetscBarrier((PetscObject)A);

    double t0=getTime();
    for(int k=0; k<nsteps; k++) {
      MatMatMult(A,B,MAT_REUSE_MATRIX,fill,&C);
      PetscBarrier((PetscObject)A);
    }
    double t1=getTime();
    app_log()<<nw <<" " <<nproc <<" " <<(t1-t0)/nsteps <<std::endl;

    MatDestroy(&C);
    MatDestroy(&B);
  }
  
  MatDestroy(&A);
  //MPI_Finalize();
  PetscFinalize();

  return 0;
}
