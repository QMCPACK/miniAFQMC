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
#include "Message/Communicate.h"
#include <iomanip>

#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include "io/hdf_archive.h"

#ifdef __bgq__
#include "Utilities/simple_stack_allocator.hpp"
std::unique_ptr<simple_stack_allocator> bgq_dummy_allocator(nullptr);
std::size_t BG_SHM_ALLOCATOR = 1024; 
#endif

// define global variables
namespace qmcplusplus
{
  TimerList_t AFQMCTimers;
}

#include "alf/boost/mpi3/environment.hpp"

#include "af_config.h"
#include "Matrix/initialize.hpp"
#include "Wavefunctions/Wavefunction.hpp"
#include "Wavefunctions/WavefunctionFactory.h"
#include "Propagators/Propagator.hpp"
#include "Propagators/PropagatorFactory.h"
#include "Walkers/WalkerSet.hpp"
#include "Utilities/taskgroup.hpp"

using namespace std;
using namespace qmcplusplus;
using namespace qmcplusplus::afqmc;
namespace mpi3 = boost::mpi3;

TimerNameList_t<AFQMCTimerIDs> AFQMCTimerNames =
{
  {block_timer, "Total"},
  {pseudo_energy_timer, "PseudoEnergy"},
  {energy_timer, "Energy"},
  {vHS_timer, "vHS"},
  {vbias_timer, "vbias"},
  {G_for_vbias_timer,"G_for_vbias"},
  {propagate_timer, "Propagate"},
  {E_comm_overhead_timer, "Energy_comm_overhead"},
  {vHS_comm_overhead_timer, "vHS_comm_overhead"},
  {StepPopControl, "Population_Control"}
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
  printf("-r                Number of reader cores in a node (default all)\n"); 
  printf("-c                Number of cores in a task group (default: all cores)\n"); 
  printf("-n                Number of nodes in a task group (default: all nodes)\n");
  printf("-f                Input file name (default: ./afqmc.h5)\n");
  printf("-b                Number of substeps with fixed vbias (default: 1)\n");
  printf("-t                Timestep (default: 0.005) \n"); 
#ifdef __bgq__
  printf("-m                Size (in MB) of pool shared memory. (default 1024 MB).\n");
#endif
  printf("-d                Debugging mode\n");
  printf("-v                Verbose output\n");
}

// Not the "real" energy, since weight are ignored for simplicity!!!
template<class WalkerSet>
double average_energy(WalkerSet& wset) {
  double et=0.0;
  for(auto it=wset.begin(); it!=wset.end(); ++it) et += real(it->energy());
  return et/wset.size();
}

int main(int argc, char **argv)
{

#ifndef QMC_COMPLEX
  std::cerr<<" Error: Please compile complex executable, QMC_COMPLEX=1. " <<std::endl;
  exit(1);
#endif

  mpi3::environment env(argc, argv);
  auto& world = env.world();

  TimerManager.set_timer_threshold(timer_level_coarse);
  setup_timers(AFQMCTimers, AFQMCTimerNames,timer_level_coarse);

  OhmmsInfo("miniafqmc_shm",world.rank());

  WALKER_TYPES walker_type(UNDEFINED_WALKER_TYPE);
  int nsteps=10;
  int nsubsteps=10; 
  int nwalk=16;
  int northo = 10;
  int nread = 0;  
  int ncores_per_TG = 0;
  int nnodes_per_TG = 0;
  double dt = 0.005;  // 1-body propagators are assumed to be generated with a timestep = 0.01
  int fix_bias = 1;

  bool verbose = false;
  bool debug = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  bool transposed_Spvn = true;

  const ComplexType one(1.),zero(0.),half(0.5);
  const ComplexType im(0.0,1.);
  const ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hvdi:s:w:o:v:c:r:n:t:f:m:")) != -1)
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
    case 'b':
      fix_bias= atoi(optarg);
      break;
    case 'n': 
      nnodes_per_TG = atoi(optarg);
      break;
    case 'c': 
      ncores_per_TG = atoi(optarg);
      break;
    case 't':
      dt = atof(optarg); 
      break;
    case 'f':
      init_file = std::string(optarg);
      break;
    case 'v': 
      verbose  = true;
      break;
    case 'd': 
      debug  = true;
      break;
#ifdef __bgq__
    case 'm': 
      BG_SHM_ALLOCATOR = atoi(optarg);
      break;
#endif
    }
  }

  // setup the Global Task Group object 
  GlobalTaskGroup gTG(world);
  
  // set defaults to single TG
  if(nnodes_per_TG < 1) nnodes_per_TG = gTG.getTotalNodes();
  if(ncores_per_TG < 1) ncores_per_TG = gTG.getTotalCores(); 

  // same nnodes/ncores for now
  TaskGroup_ TGprop(gTG,"TGprop",nnodes_per_TG,ncores_per_TG);
  nnodes_per_TG = TGprop.getNNodesPerTG();    // in case value is reset by objec
  ncores_per_TG = TGprop.getNCoresPerTG();    // in case value is reset by object 
  TaskGroup_ TGwfn(gTG,"TGwfn",nnodes_per_TG,ncores_per_TG);

  // replicated RNG string for now to keep things deterministic and independent of # of cores 
  Random.init(0, 1, iseed);
  int ip = 0;
  PrimeNumberSet<uint32_t> myPrimes;
  // create generator within the thread
  RandomGenerator<RealType> random_th(myPrimes[ip]);

  hdf_archive dump(world);
  if(!dump.open(init_file,H5F_ACC_RDONLY))
    APP_ABORT("Error: problems opening hdf5 file. \n");

  app_log()<<"***********************************************************\n";
  app_log()<<"                 Initializing from HDF5                    \n";
  app_log()<<"***********************************************************\n";

  int NMO;
  int NAEA;
  // NMO, NAEA read with Wavefunction
  Wavefunction Wfn = afqmc::getWavefunction(dump,TGprop,TGwfn,walker_type,NMO,NAEA,nwalk,1e-6,1e-6);

  Propagator Prop = afqmc::getPropagator(TGprop,NMO,NAEA,Wfn,std::addressof(random_th));

  // close hdf file
  dump.close();

  RealType Eshift = 0;
  int nchol = Wfn.global_number_of_cholesky_vectors(); // number of cholesky vectors  
  int NIK = 2*NMO*NMO;                // dimensions of linearized green function
  int NAK = 2*NAEA*NMO;               // dimensions of linearized "compacted" green function

  std::vector<ComplexType> curData;
  AFQMCInfo AFinfo("",NMO,NAEA,NAEA);
  TaskGroup_ walkTG(gTG,"Walker",1,ncores_per_TG);
  WalkerSet WSet(walkTG,AFinfo,&random_th);
  // initialize walkers to trial wave function
  {
    Matrix<ComplexType> A({NMO,NAEA});
    Matrix<ComplexType> B({NMO,NAEA});
    csr::CSR2MA('H',(*Wfn.getOrbMat())[0],A);
    if(walker_type==COLLINEAR) {
      csr::CSR2MA('H',(*Wfn.getOrbMat())[1],B);
      WSet.resize(nwalk,A,B);
    } else
      WSet.resize(nwalk,A,A);
  }

  // initialize overlaps and energy
  Wfn.Energy(WSet);

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                         Summary                           \n";
  app_log()<<"***********************************************************\n";


  app_log()<<"\n";
  app_log()<<"  Execution details: \n"
           <<"    nsteps: " <<nsteps <<"\n"
           <<"    nsubsteps: " <<nsubsteps <<"\n"
           <<"    nwalk per TG: " <<nwalk <<"\n"
           <<"    global nwalk: " <<WSet.GlobalPopulation() <<"\n"
           <<"    verbose: " <<std::boolalpha <<verbose <<"\n"
           <<"    # Chol Vectors: " <<nchol <<"\n"
           <<std::endl;

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                     Beginning Steps                       \n";
  app_log()<<"***********************************************************\n\n";
  app_log()<<"# Step   Energy   \n";

  AFQMCTimers[block_timer]->start();
  for(int step = 0, step_tot=0; step < nsteps; step++) {

    // propagate nsubsteps 
    Prop.Propagate(nsubsteps,WSet,Eshift,dt,fix_bias);

    Wfn.Orthogonalize(WSet,true);

    //AFQMCTimers[StepPopControl]->start();
    //WSet.popControl(curData);
    //AFQMCTimers[StepPopControl]->stop();

    // calculate energy
    AFQMCTimers[energy_timer]->start();
    Wfn.Energy(WSet);
    AFQMCTimers[energy_timer]->stop();

    // update Shift and print energy
    double et = average_energy(WSet);
    if(walkTG.TG_local().root()) {
      et /= walkTG.TG_heads().size(); 
      walkTG.TG_heads().all_reduce_in_place_n(&et,1,std::plus<>());
    }
    walkTG.TG_local().broadcast_value(et);
    Eshift = et;
    app_log()<<step <<"   " <<et <<"\n";

    // Branching in real code would happen here!!!

  }
  AFQMCTimers[block_timer]->stop();

  app_log()<<"\n";
  app_log()<<"***********************************************************\n";
  app_log()<<"                   Finished Calculation                    \n";
  app_log()<<"***********************************************************\n\n";

  if(world.root())
    TimerManager.print();

  return 0;
}
