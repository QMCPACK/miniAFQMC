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
#include "AFQMC/THCOps.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

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
  Timer_eloc
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
    {Timer_DM, "Mixed Density Matrix"},
    {Timer_vbias, "Bias Potential"},
    {Timer_vHS, "H-S Potential"},
    {Timer_X, "Sigma"},
    {Timer_Propg, "Propagation"},
    {Timer_extra, "Other"},
    {Timer_ovlp, "Overlap"},
    {Timer_ortho, "Orthgonalization"},
    {Timer_eloc, "Local Energy"}
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

  int nsteps=10;
  int nsubsteps=10; 
  int nwalk=16;
  int northo = 10;
  const double dt = 0.005;  // 1-body propagators are assumed to be generated with a timestep = 0.01

  bool verbose = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  ComplexType one(1.),zero(0.),half(0.5);
  ComplexType cone(1.),czero(0.);
  ComplexType im(0.0,1.);
  ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "thvi:s:w:o:f:")) != -1)
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
    case 'f':
      init_file = std::string(optarg);
      break;    
    case 'v': verbose  = true; 
      break;
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
  base::afqmc_sys AFQMCSys;   // Main AFQMC object. Control access to several apgorithmic functions. 
  ComplexMatrix Propg1;   // propagator for 1-body hamiltonian 

  hdf_archive dump;
  if(!dump.open(init_file,H5F_ACC_RDONLY)) 
    APP_ABORT("Error: problems opening hdf5 file. \n");

  std::cout<<"***********************************************************\n";
  std::cout<<"                 Initializing from HDF5                    \n"; 
  std::cout<<"***********************************************************\n";

  afqmc::THCOps THC(afqmc::Initialize(dump,dt,AFQMCSys,Propg1)); 

  RealType Eshift = 0;
  int NMO = AFQMCSys.NMO;              // number of molecular orbitals
  int NAEA = AFQMCSys.NAEA;            // number of up electrons
  int nchol = THC.number_of_cholesky_vectors();            // number of cholesky vectors  
  int NAK = 2*NAEA*NMO;               // dimensions of linearized "compacted" green function

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

  ComplexMatrix vbias(extents[nchol][nwalk]);     // bias potential
  ComplexMatrix vHS(extents[nwalk][NMO*NMO]);        // Hubbard-Stratonovich potential
  ComplexMatrix Gc(extents[nwalk][NAK]);           // compact density matrix for energy evaluation
  ComplexMatrix X(extents[nchol][nwalk]);         // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 
  ComplexVector hybridW(extents[nwalk]);         // stores weight factors
  ComplexVector eloc(extents[nwalk]);         // stores local energies

  WalkerContainer W(extents[nwalk][2][NMO][NAEA]);
  // 0: eloc, 1: weight, 2: ovlp_up, 3: ovlp_down, 4: w_eloc, 5: old_w_eloc, 6: old_ovlp_alpha, 7: old_ovlp_beta
  ComplexMatrix W_data(extents[nwalk][8]);  
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

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
  RealType Eav = THC.energy(W_data,Gc);

  
  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                     Beginning Steps                       \n";   
  std::cout<<"***********************************************************\n\n";
  std::cout<<"# Initial Energy: " <<Eav <<std::endl <<std::endl; 
  std::cout<<"# Step   Energy   \n";

  Timers[Timer_Total]->start();
  for(int step = 0, step_tot=0; step < nsteps; step++) {
  
    for(int substep = 0; substep < nsubsteps; substep++, step_tot++) {

      // propagate walker forward 
      
      // 1. calculate density matrix and bias potential 

      Timers[Timer_DM]->start();
      AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
      Timers[Timer_DM]->stop();

      Timers[Timer_vbias]->start();
      THC.vbias(Gc,vbias);  
      Timers[Timer_vbias]->stop();

      // 2. calculate X and weight
      //  X(chol,nw) = rand + i*vbias(chol,nw)
      Timers[Timer_X]->start();
      random_th.generate_normal(X.data(),X.num_elements()); 
      std::fill(hybridW.begin(),hybridW.end(),ComplexType(0.)); 
      for(int n=0; n<nchol; n++)
        for(int nw=0; nw<nwalk; nw++) { 
          hybridW[nw] -= im*vbias[n][nw]*(X[n][nw]+halfim*vbias[n][nw]);
          X[n][nw] += im*vbias[n][nw];
        }
      Timers[Timer_X]->stop();

      // 3. calculate vHS
      Timers[Timer_vHS]->start();
      THC.vHS(X,vHS);      
      Timers[Timer_vHS]->stop();

      // 4. propagate walker
      // W(new) = Propg1 * exp(vHS) * Propg1 * W(old)
      Timers[Timer_Propg]->start();
//      AFQMCSys.propagate(W,Propg1,vHS);
      Timers[Timer_Propg]->stop();

      // 5. update overlaps
      Timers[Timer_extra]->start();
      for(int nw=0; nw<nwalk; nw++) {
        W_data[nw][5] = W_data[nw][4];
        W_data[nw][6] = W_data[nw][2];
        W_data[nw][7] = W_data[nw][3];
      }
      Timers[Timer_extra]->stop();
      Timers[Timer_ovlp]->start();
      AFQMCSys.calculate_overlaps(W,W_data);
      Timers[Timer_ovlp]->stop();

      // 6. adjust weights and walker data      
      Timers[Timer_extra]->start();
      RealType et = 0.;
      for(int nw=0; nw<nwalk; nw++) {
        ComplexType ratioOverlaps = W_data[nw][2]*W_data[nw][3]/(W_data[nw][6]*W_data[nw][7] );   
        RealType scale = std::max(0.0,std::cos( std::arg( ratioOverlaps )) );
        W_data[nw][4] = -( hybridW[nw] + std::log(ratioOverlaps) )/dt; 
        W_data[nw][1] *= ComplexType(scale*std::exp( -dt*(0.5*( W_data[nw][4].real() + W_data[nw][5].real() ) - Eshift) ),0.0);
        et += W_data[nw][4].real();
      }

      // decide what to do with Eshift later
      Eshift = et/nwalk;
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
    AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc);
    Eav = THC.energy(W_data,Gc);
    std::cout<<step <<"   " <<Eav <<"\n";
    Timers[Timer_eloc]->stop();

    // Branching in real code would happen here!!!
  
  }    
  Timers[Timer_Total]->stop();

  std::cout<<"\n";
  std::cout<<"***********************************************************\n";
  std::cout<<"                   Finished Calculation                    \n";   
  std::cout<<"***********************************************************\n\n";
  
  TimerManager.print();

  return 0;
}
