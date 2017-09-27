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
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/taskgroup.hpp>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "Matrix/ma_communications.hpp"
#include "AFQMC/afqmc_sys_shm.hpp"
#include "AFQMC/rotate.hpp"
#include "Matrix/initialize.hpp"
#include "Matrix/partition_SpMat.hpp"
#include "AFQMC/mixed_density_matrix.hpp"
#include "AFQMC/energy.hpp"
#include "AFQMC/vHS.hpp"
#include "AFQMC/vbias.hpp"

using namespace std;
using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
  Timer_DM,
  Timer_vbias,
  Timer_vHS,
  Timer_X,
  Timer_Propg,
  Timer_extra,
  Timer_ovlp,
  Timer_ortho,
  Timer_eloc,
  Timer_gather,
  Timer_reduce,
  Timer_comm
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
    {Timer_DM, "Mixed-Density-Matrix"},
    {Timer_vbias, "Bias-Potential"},
    {Timer_vHS, "H-S_Potential"},
    {Timer_X, "Sigma"},
    {Timer_Propg, "Propagation"},
    {Timer_extra, "Other"},
    {Timer_ovlp, "Overlap"},
    {Timer_ortho, "Orthgonalization"},
    {Timer_eloc, "Local-Energy"},
    {Timer_gather, "Allgather"},
    {Timer_reduce, "Allreduce"},
    {Timer_comm, "Comm"}
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
  printf("-r                Number of reader cores in a node (default all)"); 
  printf("-c                Number of cores in a task group (default: all cores)"); 
  printf("-n                Number of nodes in a task group (default: all nodes)\n");
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

  // need new mpi wrapper
  MPI_Init(&argc,&argv);

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  OhmmsInfo("miniafqmc_shm",rank);

  int nsteps=10;
  int nsubsteps=10; 
  int nwalk=16;
  int northo = 10;
  int nread = 0;  
  int ncores_per_TG = 0;
  int nnodes_per_TG = 0;
  const double dt = 0.01;  // 1-body propagators are assumed to be generated with a timestep = 0.01

  bool verbose = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  bool transposed_Spvn = true;

  ComplexType one(1.),zero(0.),half(0.5);
  ComplexType im(0.0,1.);
  ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hvi:s:w:o:v:c:r:n:t:f:")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 's': // the number of sub steps 
      nsubsteps = atoi(optarg);
      break;
    case 'w': 
      nwalk = atoi(optarg);
      break;
    case 'o': 
      northo = atoi(optarg);
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
    case 't':
      transposed_Spvn = (std::string(optarg) != "no");
      break;
    case 'f':
      init_file = std::string(optarg);
      break;
    case 'v': verbose  = true;
      break;
    }
  }

  // replicated RNG string for now to keep things deterministic and independent of # of cores 
  Random.init(0, 1, iseed);
  int ip = 0;
  PrimeNumberSet<uint32_t> myPrimes;
  // create generator within the thread
  RandomGenerator<RealType> random_th(myPrimes[ip]);

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

  nnodes_per_TG = TG.getNNodesPerTG();
  ncores_per_TG = TG.getNCoresPerTG();
  int nnodes = TG.getTotalNodes();      // Total number of nodes
  int nodeid = TG.getNodeID();          // glocal id of local node 
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

  if(transposed_Spvn) shm::halfrotate_cholesky(TG,
                                               AFQMCSys.trialwfn_alpha,
                                               AFQMCSys.trialwfn_beta,
                                               SMSpvn,
                                               SMSpvnT
                                              );

  // useful info, reduce over 1 task group only
  long global_Spvn_size=0, global_Vakbl_size=0;
  if(TG.getTGNumber()==0) {
    long sz = SMSpvn.size();
    if(coreid != 0) sz = 0;
    MPI_Reduce(&sz,&global_Spvn_size,1,MPI_LONG,MPI_SUM,0,TG.getTGComm());
    // not yet distributed
    global_Vakbl_size = SMVakbl.size();
  }

  RealType Eshift = 0;
  int NMO = AFQMCSys.NMO;              // number of molecular orbitals
  int NAEA = AFQMCSys.NAEA;            // number of up electrons
  int NIK = 2*NMO*NMO;                 // dimensions of linearized green function
  int NAK = 2*NAEA*NMO;                // dimensions of linearized "compacted" green function
  int cvec0 = cholVec_partition[TG.getLocalNodeNumber()]; 
  int cvecN = cholVec_partition[TG.getLocalNodeNumber()+1]; 
  int nchol = SMSpvn.cols();
  assert(nchol == cvecN-cvec0);   
  if(transposed_Spvn)
    NIK = NAK;  

  // partition local Spvn and Vakbl among cores in TG, generate lightweight SparseMatrix_ref
  SparseMatrix_ref<ComplexType> Spvn;    
  SparseMatrix_ref<ComplexType> Spvn_for_vbias;  
  SparseMatrix_ref<ComplexType> Vakbl; 

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
  if(!shm::balance_partition_SpMat(TG,byRows,SMVakbl,Vakbl)) {
    std::cerr<<" Error partitioning Vakbl matrix." <<std::endl; 
    MPI_Abort(MPI_COMM_WORLD,30);
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
           <<"    nsubsteps: " <<nsubsteps <<"\n" 
           <<"    nwalk: " <<nwalk <<"\n"
           <<"    northo: " <<northo <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<nchol <<"\n"
           <<"    transposed Spvn: " <<transposed_Spvn <<"\n"
           <<"    Chol. Matrix Sparsity: " <<global_Spvn_size/double(nchol*NMO*NMO) <<"\n"
           <<"    Hamiltonian Sparsity: " <<global_Vakbl_size/double(NAEA*NAEA*NMO*NMO*4.0) <<"\n"
           <<"    # nodes per TG: " <<TG.getNNodesPerTG() <<"\n" 
           <<"    # cores per TG: " <<TG.getNCoresPerTG() <<"\n" 
           <<"    # reading cores: " <<((nread==0)?ncores:nread)
           <<std::endl;

  int nwalk_tot = nwalk*nnodes_per_TG;  

  // setup SM memory containers
  // bias potential
  SMDenseVector<ComplexType> SM_vbias(std::string("SM_vbias")+str0,TG.getTGCommLocal(),nchol*nwalk_tot);
  // Hubbard-Stratonovich potential
  SMDenseVector<ComplexType> SM_vHS(std::string("SM_vHS")+str0,TG.getTGCommLocal(),NMO*NMO*nwalk_tot);
  // density matrix
  SMDenseVector<ComplexType> SM_G_for_vbias(std::string("SM_G_for_vbias")+str0,TG.getTGCommLocal(),NIK*nwalk);
  // compact density matrix for energy evaluation
  SMDenseVector<ComplexType> SM_Gc(std::string("SM_Gc")+str0,TG.getTGCommLocal(),NAK*nwalk);    
  // X(n,nw) = rand(n,nw) ( + vbias(n,nw))
  SMDenseVector<ComplexType> SM_X(std::string("SM_X")+str0,TG.getTGCommLocal(),nchol*nwalk_tot);     
  // stores weight factors
  SMDenseVector<ComplexType> SM_hybridW(std::string("SM_hybridW")+str0,TG.getTGCommLocal(),nwalk_tot);
  // stores local energies
  SMDenseVector<ComplexType> SM_eloc(std::string("SM_eloc")+str0,TG.getTGCommLocal(),nwalk_tot);  

  // setup light references to SM
  boost::multi_array_ref<ComplexType,2> vbias(SM_vbias.data(),extents[nchol][nwalk_tot]);     
  boost::multi_array_ref<ComplexType,2> vHS(SM_vHS.data(), extents[NMO*NMO][nwalk_tot]);     
  boost::multi_array_ref<ComplexType,2> G_for_vbias(SM_G_for_vbias.data(), extents[NIK][nwalk]);           
  boost::multi_array_ref<ComplexType,2> Gc(SM_Gc.data(), extents[NAK][nwalk]);        
  boost::multi_array_ref<ComplexType,2> X(SM_X.data(), extents[nchol][nwalk_tot]);       
  boost::multi_array_ref<ComplexType,1> eloc(SM_eloc.data(), extents[nwalk_tot]);       
  boost::multi_array_ref<ComplexType,1> hybridW(SM_hybridW.data(), extents[nwalk_tot]);

  // temporary local container needed if transposed_Spvn=false 
  boost::multi_array<ComplexType,2> loc_vbias;
  if(!transposed_Spvn) 
    loc_vbias.resize(extents[Spvn_for_vbias.cols()][nwalk_tot]);

  // some additional global structures
  SMDenseVector<ComplexType> SM_G_glob(std::string("SM_G_glob")+str0,TG.getTGCommLocal(),NIK*nwalk_tot);
  boost::multi_array_ref<ComplexType,2> G_glob(SM_G_glob.data(), extents[NIK][nwalk_tot]);

  // Walker container: Shared among local cores in a TG
  SMDenseVector<ComplexType> SM_W(std::string("SM_W")+str0,TG.getTGCommLocal(),nwalk*2*NMO*NAEA);
  boost::multi_array_ref<ComplexType,4> W(SM_W.data(),extents[nwalk][2][NMO][NAEA]);

  // 0: eloc, 1: weight, 2: ovlp_up, 3: ovlp_down, 4: w_eloc, 
  // 5: old_w_eloc, 6: old_ovlp_alpha, 7: old_ovlp_beta
  SMDenseVector<ComplexType> SM_W_data(std::string("SM_W_data")+str0,TG.getTGCommLocal(),nwalk*8);
  boost::multi_array_ref<ComplexType,2> W_data(SM_W_data.data(), extents[nwalk][8]);  
  if(TG.getCoreRank()==0) {
    // initialize walkers to trial wave function
    for(int n=0; n<nwalk; n++) 
      for(int nm=0; nm<NMO; nm++) 
        for(int na=0; na<NAEA; na++) {
          using std::conj;
          W[n][0][nm][na] = conj(AFQMCSys.trialwfn_alpha[nm][na]);
          W[n][1][nm][na] = conj(AFQMCSys.trialwfn_beta[nm][na]);
        }

    // set weights to 1
    for(int n=0; n<nwalk; n++) 
      W_data[n][1] = ComplexType(1.);
  }
  SM_W.barrier();  

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc,true);
  RealType Eav = AFQMCSys.calculate_energy(W_data,Gc,haj,Vakbl);

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                     Beginning Steps                       \n";   
  app_log()<<"***********************************************************\n\n";
  app_log()<<"# Step   Energy   " <<std::endl;

  Timers[Timer_Total]->start();
  for(int step = 0, step_tot=0; step < nsteps; step++) {
  
    for(int substep = 0; substep < nsubsteps; substep++, step_tot++) {

      // propagate walker forward 
      
      // 1. calculate density matrix and bias potential 

      Timers[Timer_DM]->start();
      AFQMCSys.calculate_mixed_density_matrix(W,W_data,G_for_vbias,transposed_Spvn);
      Timers[Timer_DM]->stop();

      if(nnodes_per_TG == 1) {

        Timers[Timer_vbias]->start();
        if(transposed_Spvn) {

          shm::get_vbias(Spvn_for_vbias,G_for_vbias,vbias,transposed_Spvn);  

        } else {

          // careful here, needs temporary matrix local to the core (NOT in SM!!!)
          shm::get_vbias(Spvn_for_vbias,G_for_vbias,loc_vbias,transposed_Spvn);
          // no need for lock, partitionings are non-overlapping 
          vbias[ indices[range_t(Spvn_for_vbias.global_c0(),Spvn_for_vbias.global_cN())][range_t(0,nwalk)] ] = 
              loc_vbias[ indices[range_t(Spvn_for_vbias.global_c0(),Spvn_for_vbias.global_cN())][range_t(0,nwalk)] ];

        } 
        TG.local_barrier();  
        Timers[Timer_vbias]->stop();

        // 2. calculate X and weight
        //  X(chol,nw) = rand + i*vbias(chol,nw)
        Timers[Timer_X]->start();
        // MAM: head node does this right now to keep results independent of # cores in single node runs
        if(core_rank == 0) {      
          random_th.generate_normal(X.data(),X.num_elements()); 
          std::fill(hybridW.begin(),hybridW.end(),ComplexType(0.)); 
          for(int n=0; n<nchol; n++)
            for(int nw=0; nw<nwalk; nw++) { 
              hybridW[nw] -= im*vbias[n][nw]*(X[n][nw]+halfim*vbias[n][nw]);
              X[n][nw] += im*vbias[n][nw];
            }
        }
        TG.local_barrier();
        Timers[Timer_X]->stop();

        // 3. calculate vHS
        // vHS(i,k,nw) = sum_n Spvn(i,k,n) * X(n,nw) 
        Timers[Timer_vHS]->start();
        shm::get_vHS(Spvn,X,vHS);      
        TG.local_barrier();  
        Timers[Timer_vHS]->stop();

        // 4. propagate walker
        // W(new) = Propg1 * exp(vHS) * Propg1 * W(old)
        Timers[Timer_Propg]->start();
        AFQMCSys.propagate(W,Propg1,vHS);
        Timers[Timer_Propg]->stop();

      } else {
        // Distributed algorithm
  
        // simple algorithm
        // 1. Allgather G_for_vbias
        // 2. calculate local contribution to vHS for all walkers in TG
        // 3. Allreduce vbias
        
        Timers[Timer_gather]->start();      
        if(TG.getCoreRank() == 0)
          ma::gather_matrix(TG.getTGCommHeads(),G_for_vbias,G_glob,byCols);
        Timers[Timer_gather]->stop();      
 
        Timers[Timer_vbias]->start();
        if(transposed_Spvn) {

          shm::get_vbias(Spvn_for_vbias,G_glob,vbias,transposed_Spvn);  

        } else {

          // careful here, needs temporary matrix local to the core (NOT in SM!!!)
          shm::get_vbias(Spvn_for_vbias,G_glob,loc_vbias,transposed_Spvn);
          // no need for lock, partitionings are non-overlapping 
          vbias[ indices[range_t(Spvn_for_vbias.global_c0(),Spvn_for_vbias.global_cN())][range_t(0,nwalk_tot)] ] = 
              loc_vbias[ indices[range_t(Spvn_for_vbias.global_c0(),Spvn_for_vbias.global_cN())][range_t(0,nwalk_tot)] ];

        } 
        TG.local_barrier();  
        Timers[Timer_vbias]->stop();

        // 2. calculate X and weight
        //  X(chol,nw) = rand + i*vbias(chol,nw)
        Timers[Timer_X]->start();
        // MAM: head node does this right now to keep results independent of # cores in single node runs
        if(core_rank == 0) {      
          random_th.generate_normal(X.data(),X.num_elements()); 
          std::fill(hybridW.begin(),hybridW.end(),ComplexType(0.)); 
          for(int n=0; n<nchol; n++)
            for(int nw=0; nw<nwalk_tot; nw++) { 
              hybridW[nw] -= im*vbias[n][nw]*(X[n][nw]+halfim*vbias[n][nw]);
              X[n][nw] += im*vbias[n][nw];
            }
        }
        TG.local_barrier();
        Timers[Timer_X]->stop();

        // 3. calculate vHS
        // vHS(i,k,nw) = sum_n Spvn(i,k,n) * X(n,nw) 
        Timers[Timer_vHS]->start();
        shm::get_vHS(Spvn,X,vHS);      
        TG.local_barrier();  
        Timers[Timer_vHS]->stop();

        Timers[Timer_reduce]->start();
        if(TG.getCoreRank() == 0)
          MPI_Allreduce(MPI_IN_PLACE,vHS.data(),vHS.num_elements()*2,MPI_DOUBLE,MPI_SUM,TG.getTGCommHeads());  
        Timers[Timer_reduce]->stop();
        TG.local_barrier();  

        // 4. propagate walker
        // W(new) = Propg1 * exp(vHS) * Propg1 * W(old)
        Timers[Timer_Propg]->start();
        AFQMCSys.propagate_from_global(W,Propg1,vHS);
        Timers[Timer_Propg]->stop();

      }


      // 5. update overlaps
      Timers[Timer_extra]->start();
      if(core_rank == 0) {      
        for(int nw=0; nw<nwalk; nw++) {
          W_data[nw][5] = W_data[nw][4];
          W_data[nw][6] = W_data[nw][2];
          W_data[nw][7] = W_data[nw][3];
        }
      }
      TG.local_barrier();  
      Timers[Timer_extra]->stop();
      Timers[Timer_ovlp]->start();
      AFQMCSys.calculate_overlaps(W,W_data);
      Timers[Timer_ovlp]->stop();

      // 6. adjust weights and walker data      
      Timers[Timer_extra]->start();
      if(core_rank == 0) {      
        RealType et = 0.;
        for(int nw=0; nw<nwalk; nw++) {
          ComplexType ratioOverlaps = W_data[nw][2]*W_data[nw][3]/(W_data[nw][6]*W_data[nw][7] );   
          RealType scale = std::max(0.0,std::cos( std::arg( ratioOverlaps )) );
          W_data[nw][4] = -( hybridW[nw] + std::log(ratioOverlaps) )/dt; 
          W_data[nw][1] *= ComplexType(scale*std::exp( -dt*(0.5*( W_data[nw][4].real() 
                                            + W_data[nw][5].real() ) - Eshift) ),0.0);
          et += W_data[nw][4].real();
        }
        Eshift = et/nwalk;
        // decide what to do with Eshift later
      }
      TG.local_barrier();
      Timers[Timer_extra]->stop();

      if(step_tot > 0 && step_tot%northo == 0) {
        Timers[Timer_ortho]->start();
        AFQMCSys.orthogonalize(W);
        Timers[Timer_ortho]->stop();
        Timers[Timer_ovlp]->start();
        AFQMCSys.calculate_overlaps(W,W_data);
        Timers[Timer_ovlp]->stop();
      }
       
    }

    Timers[Timer_eloc]->start();
    AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc,true);
    Eav = AFQMCSys.calculate_energy(W_data,Gc,haj,Vakbl);
    Timers[Timer_eloc]->stop();
    app_log()<<step <<"   " <<setprecision(12) <<Eav <<std::endl;

    // Branching in real code would happen here!!!
  
  }    
  Timers[Timer_Total]->stop();

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                   Finished Calculation                    \n";   
  app_log()<<"***********************************************************\n\n";
  
  // only root outputs time for now
  if(rank==0)
    TimerManager.print();

  MPI_Finalize();

  return 0;
}
