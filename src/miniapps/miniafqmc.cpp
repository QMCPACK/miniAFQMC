////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
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

#include <Configuration.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include <mpi.h>

#include "AFQMC/AFQMCInfo.hpp"
#include "Matrix/initialize.hpp"
#include "Utilities/taskgroup.hpp"
#include "io/hdf_archive.h"

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
  printf("miniafqmc - QMCPACK AFQMC miniapp\n");
  printf("\n");
  printf("Options:\n");
  printf("-i                Number of MC steps (default: 100)\n");
  printf("-s                Number of substeps (default: 1)\n");
  printf("-v                Verbose output\n");
  printf("-r                Number of cores that read (default: all)\n");
}

int main(int argc, char **argv)
{

   MPI_Init(&argc,&argv);

  int nsteps=100;
  int nsubsteps=1; 
  int nread=0;

  bool verbose = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hdvs:g:i:b:c:a:r:")) != -1)
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
    case 'r': 
      nread = atoi(optarg);
      break;
    case 'v': verbose  = true; break;
    }
  }

  Random.init(0, 1, iseed);
  int ip = 0;
  PrimeNumberSet<uint32_t> myPrimes;
  // create generator within the thread
  RandomGenerator<RealType> random_th(myPrimes[ip]);

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  // Important Data Structures
  AFQMCInfo SysInfo; // Basic information about the simulation, e.g. #orbitals, #electrons, etc.
  ValueSpMat Spvn; // (Symmetric) Factorized Hamiltonian, e.g. <ij|kl> = sum_n Spvn(ik,n) * Spvn(jl,n)
  ValueMatrix hij;    // 1-Body Hamiltonian Matrix
  ValueSpMat Vakbl;   // 2-Body Hamiltonian Matrix: (Half-Rotated) 2-electron integrals 
  ValueMatrix trialwfn; // Slater Matrix representing the trial wave-function 
  ValueMatrix Propg1; // propagator for 1-body hamiltonian 

  afqmc::TaskGroup TG("TGunique");
  TG.setup(1,1);  

  hdf_archive dump;
  if(nread == 0 || TG.getCoreID() < nread) {
    if(!dump.open(init_file,H5F_ACC_RDONLY)) 
      APP_ABORT("Error: problems opening hdf5 file. \n");
  } 

  cout<<" Hello World. \n";

  if(!Initialize(dump,SysInfo,TG,Propg1,Spvn,hij,Vakbl,trialwfn,nread)) {
    app_error()<<" Error initalizing data structures from hdf5 file: " <<init_file <<std::endl;
    APP_ABORT(" Abort. \n\n\n");
  }
   
  SysInfo.print(cout);
   
  // finalize
  MPI_Finalize();

  return 0;
}
